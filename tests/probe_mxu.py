#!/usr/bin/env python3
"""Measure the lane layout of MetalFX's MXU GEMM (llvm.agx3.igemm) from MetalFX's own Conv3x3Stride2U8 kernel.

The kernel is built with a probe GEMM (air/intrinsics.py, MXU_MODE probe) that records the A and B operands of one
SIMD group and gives every accumulator element its own id (lane*8 + slot). Running it with label-encoded inputs
shows which (pixel, channel) each A element is, which (channel, output channel) each B element is, and where each
accumulator element is stored. A GEMM D[r][c] = C[r][c] + sum_k A[r][k] B[k][c] is the only product consistent with
those labels; the tables written to air/mxu_layout.py place each lane/slot in that logical [r][k] / [c][k] / [r][c]
order. Rows of A and C are matched by pixel order (input pixel ids rise with output pixel within a tile row).

usage: probe_mxu.py <MetalFX Resources dir>
"""
import os, subprocess, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'air'))
import intrinsics, metallib, m4r_air  # noqa: E402

HO, WO, K, C = 2, 16, 32, 32
HI, WI = 2 * HO, 2 * WO


def build(res, out):
    os.makedirs(out + '/k', exist_ok=True)
    mtl = open(os.path.join(res, 'default.metallib'), 'rb').read()
    lib = metallib.parse(mtl)
    f = next(f for f in lib['functions'] if f.name == 'Conv3x3Stride2U8')
    intrinsics.MXU_MODE['mode'] = 'probe'
    consts = {'H': HO, 'W': WO, 'K': K, 'C': C, 'QsI': 1.0, 'QsRinv': 1.0, 'inputBlockK': 0, 'useLeakyReLU': 0, 'outZp': 0.0}
    ir, man = m4r_air.translate(m4r_air.to_text(metallib.bitcode(mtl, lib, f)), 'Conv3x3Stride2U8', consts, [128, 1, 1])
    m4r_air.compile_ir(ir, out + '/k/probe.hsaco')
    m4r_air.write_manifest(man, out + '/k/probe')
    np.full(K, 1.0, np.float16).tofile(out + '/s.bin')
    np.zeros(K, np.float16).tofile(out + '/b.bin')
    np.zeros(K, np.uint8).tofile(out + '/z.bin')
    plan = [f'buffer act {HI * WI * C}', f'buffer w {9 * C * K}', f'buffer out {HO * WO * K}',
            f'buffer s {2 * K} file s.bin', f'buffer b {2 * K} file b.bin', f'buffer z {K} file z.bin',
            'kernel probe k/probe', 'pass probe threads 128 1 1', '  buf 0 act', '  buf 1 w', '  buf 2 out',
            '  buf 3 s', '  buf 4 b', '  buf 5 z']
    open(out + '/plan.txt', 'w').write('\n'.join(plan) + '\n')


def run(out, act, w):
    act.astype(np.uint8).tofile(out + '/act.in')
    w.astype(np.uint8).tofile(out + '/w.in')
    subprocess.run([os.path.join(ROOT, 'build/test_plan'), out, f'load=act:{out}/act.in,w:{out}/w.in', 'dump=out', 'runs=1', 'probe'],
                   check=True, capture_output=True)
    p = np.fromfile(out + '/probe-Conv3x3Stride2U8.bin', np.uint8)
    return p[:256].reshape(32, 8), p[256:512].reshape(32, 8), np.fromfile(out + '/dump-out.bin', np.uint8).reshape(HO * WO, K)


def main():
    res = sys.argv[1]
    out = os.path.join(ROOT, 'build', 'probe-mxu')
    build(res, out)
    iy, ix, c = np.meshgrid(np.arange(HI), np.arange(WI), np.arange(C), indexing='ij')
    a1, b1, ids = run(out, c + 1, np.broadcast_to(np.arange(C)[None, :, None] + 1, (9, C, K)))
    pix = iy * WI + ix + 1
    a2, b2, ids2 = run(out, pix, np.broadcast_to(np.arange(K)[None, None, :] + 1, (9, C, K)))
    assert (ids == ids2).all(), 'accumulator ids differ between runs'
    if (a1 == 0).any() or (a2 == 0).any():
        sys.exit('probe recorded masked-off (padding) A elements; enlarge the fixture')
    A, BT, Ct = [0] * 256, [0] * 256, [None] * 256
    # A: rows ranked by input pixel, k = channel label within the 16-channel block
    rows = sorted(set(a2.flatten().tolist()))
    assert len(rows) == 16, f'A holds {len(rows)} distinct pixels, expected 16'
    for l in range(32):
        for s in range(8):
            A[l * 8 + s] = rows.index(int(a2[l, s])) * 16 + (int(a1[l, s]) - 1) % 16
            BT[l * 8 + s] = ((int(b2[l, s]) - 1) % 16) * 16 + (int(b1[l, s]) - 1) % 16
    # C: where each accumulator id was stored (pixel o, output channel n); rank o within its 16-pixel group
    for o in range(HO * WO):
        for n in range(K):
            v = int(ids[o, n])
            rc = (o % 16) * 16 + n % 16
            if Ct[v] is not None and Ct[v] != rc:
                sys.exit(f'accumulator {v} stored at inconsistent positions')
            Ct[v] = rc
    for name, t in (('A', A), ('BT', BT), ('C', Ct)):
        if sorted(t) != list(range(256)):
            sys.exit(f'{name} table is not a permutation of the 16x16 tile')
    with open(os.path.join(ROOT, 'air', 'mxu_layout.py'), 'w') as f:
        f.write('"""MXU (llvm.agx3.igemm) lane layout measured by tests/probe_mxu.py.\n\n'
                'Index lane*8 + slot. A: row*16 + k, BT: col*16 + k (B stored transposed), C: row*16 + col."""\n')
        for name, t in (('A', A), ('BT', BT), ('C', Ct)):
            f.write(f'{name} = {t}\n')
    print('A ', A[:16], '...\nBT', BT[:16], '...\nC ', Ct[:16], '...')
    print('wrote air/mxu_layout.py')


if __name__ == '__main__':
    main()
