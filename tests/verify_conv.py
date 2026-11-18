#!/usr/bin/env python3
"""Check translated MetalFX conv kernels (MXU GEMM via air/mxu_layout.py) against a numpy reference convolution.

Random u8 activations and weights, zero padding of 1, identity weight scales and zero bias; the kernels' float
requantisation is reproduced as round(acc * QsI * QsRinv) clamped to [0, 255], so results may differ by one step.
usage: verify_conv.py <MetalFX Resources dir>
"""
import os, subprocess, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'air'))
sys.path.insert(0, os.path.join(ROOT, 'tools'))
import intrinsics, metallib, m4r_air  # noqa: E402
from prepare_bbr import block_major  # noqa: E402


def conv_ref(x, w, stride):
    """x [H][W][C] u8, w [3][3][C][K] u8 -> int accumulators [Ho][Wo][K]"""
    H, W, C = x.shape
    K = w.shape[-1]
    Ho, Wo = H // stride, W // stride
    xp = np.zeros((H + 2, W + 2, C), np.int64)
    xp[1:-1, 1:-1] = x
    acc = np.zeros((Ho, Wo, K), np.int64)
    for ky in range(3):
        for kx in range(3):
            patch = xp[ky:ky + stride * Ho:stride, kx:kx + stride * Wo:stride]
            acc += np.einsum('hwc,ck->hwk', patch, w[ky, kx].astype(np.int64))
    return acc


def run_case(res, name, fn, consts, tg, threads, x, wbuf, K, outsize, extra=()):
    out = os.path.join(ROOT, 'build', 'verify-' + name)
    os.makedirs(out + '/k', exist_ok=True)
    mtl = open(os.path.join(res, 'default.metallib'), 'rb').read()
    lib = metallib.parse(mtl)
    f = next(f for f in lib['functions'] if f.name == fn)
    intrinsics.MXU_MODE['mode'] = 'lds'
    import importlib, mxu_layout
    importlib.reload(mxu_layout)
    ir, man = m4r_air.translate(m4r_air.to_text(metallib.bitcode(mtl, lib, f)), fn, consts, tg)
    m4r_air.compile_ir(ir, out + '/k/conv.hsaco')
    m4r_air.write_manifest(man, out + '/k/conv')
    x.astype(np.uint8).tofile(out + '/x.bin')
    wbuf.astype(np.uint8).tofile(out + '/w.bin')
    np.full(K, 1.0, np.float16).tofile(out + '/s.bin')
    np.zeros(K, np.float16).tofile(out + '/b.bin')
    np.zeros(K, np.uint8).tofile(out + '/z.bin')
    plan = [f'buffer x {x.size} file x.bin', f'buffer w {wbuf.size} file w.bin', f'buffer out {outsize}',
            f'buffer s {2 * K} file s.bin', f'buffer b {2 * K} file b.bin', f'buffer z {K} file z.bin',
            'kernel conv k/conv', f'pass conv threads {threads} 1 1', '  buf 0 x', '  buf 1 w', '  buf 2 out',
            '  buf 3 s', '  buf 4 b', '  buf 5 z'] + list(extra)
    open(out + '/plan.txt', 'w').write('\n'.join(plan) + '\n')
    r = subprocess.run([os.path.join(ROOT, 'build/test_plan'), out, 'dump=out', 'runs=2'], capture_output=True, text=True)
    if r.returncode:
        sys.exit(f'{name}: test_plan failed\n{r.stdout}\n{r.stderr}')
    return np.fromfile(out + '/dump-out.bin', np.uint8)


def main():
    res = sys.argv[1]
    rng = np.random.default_rng(7)
    ok = True
    # stride 2: K = C = 32, output 16x8
    Ho, Wo, C, K = 8, 16, 32, 32
    x = rng.integers(0, 16, (2 * Ho, 2 * Wo, C))
    w = rng.integers(0, 16, (3, 3, C, K))
    acc = conv_ref(x, w, 2)
    q = 255.0 / acc.max()
    ref = np.clip(np.round(acc * q), 0, 255)
    consts = {'H': Ho, 'W': Wo, 'K': K, 'C': C, 'QsI': 1.0, 'QsRinv': q, 'inputBlockK': 0, 'useLeakyReLU': 0, 'outZp': 0.0}
    threads = ((((K + 31) & ~31) * ((Ho * Wo + 31) // 32)) + 127) & ~127
    got = run_case(res, 's2', 'Conv3x3Stride2U8', consts, [128, 1, 1], threads, x, w.reshape(-1), K, Ho * Wo * K).reshape(Ho, Wo, K)
    d = np.abs(got.astype(int) - ref.astype(int))
    print(f'Conv3x3Stride2U8: max |diff| {d.max()}, {np.mean(d > 0) * 100:.2f}% off by one or more')
    ok &= d.max() <= 1
    # stride 1: K = C = 32, 16x16, block-major weights
    H, W = 16, 16
    x = rng.integers(0, 16, (H, W, C))
    w = rng.integers(0, 16, (3, 3, C, K))
    acc = conv_ref(x, w, 1)
    q = 255.0 / acc.max()
    ref = np.clip(np.round(acc * q), 0, 255)
    consts = {'H': H, 'W': W, 'K': K, 'C': C, 'QsI': 1.0, 'QsRinv': q, 'NmatsHW': 2, 'NmatsK': 2, 'inputIsQuantized': 1,
              'blockMajorWeights': 1, 'useLeakyReLU': 0, 'outZp': 0.0}
    tiles = ((H * W + 31) // 32) * ((K + 31) // 32)
    threads = (tiles * 32 + 63) // 64 * 64
    got = run_case(res, 's1', 'Conv3x3Stride1U8', consts, [64, 1, 1], threads, x, block_major(w.reshape(-1), C, K, 3, 3), K, H * W * K).reshape(H, W, K)
    d = np.abs(got.astype(int) - ref.astype(int))
    print(f'Conv3x3Stride1U8: max |diff| {d.max()}, {np.mean(d > 0) * 100:.2f}% off by one or more')
    ok &= d.max() <= 1
    print('PASS' if ok else 'FAIL')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
