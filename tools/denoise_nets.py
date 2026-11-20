#!/usr/bin/env python3
"""MetalFX denoiser networks (DBFNet, UBFNet): layer tables, numpy reference and plan generator.

The networks are Apple's (weights from emit_dbfnet_v3_2_constants.dat / emit_ubfnet_v3_2_constants.dat in the user's MetalFX
Resources folder, layer tables in docs/metalfx-denoiser.md); the convolutions run on m4r's own plain kernels
(shaders/nn_*.comp: fp32 accumulation, fp16 activations) instead of Apple's fused MXU kernels.

usage: denoise_nets.py dbf <Resources dir> <out plan dir> <H> <W>     writes a plan (input dbf_in, output dbf_out, NHWC fp16)
       denoise_nets.py dbf-ref <Resources dir> <H> <W> <input.bin> <output.bin>   numpy reference of the same
SPDX-License-Identifier: GPL-3.0-only
"""
import os, shutil, subprocess, sys
import numpy as np

# (weights offset, bias offset, kernel, cin, cout) in the .dat; consecutive pairs are the 3x3 + 1x1 of one cell
DBF = [(0x40, 0x0, 3, 13, 32), (0x1dc0, 0x1d80, 1, 32, 32), (0x2640, 0x25c0, 3, 32, 64), (0xb6c0, 0xb640, 1, 64, 64),
       (0xd740, 0xd6c0, 3, 64, 64), (0x1f7c0, 0x1f740, 1, 64, 64), (0x21840, 0x217c0, 3, 64, 64), (0x338c0, 0x33840, 1, 64, 64),
       (0x35950, 0x358d0, 3, 64, 64), (0x479d0, 0x47950, 1, 64, 64), (0x49a60, 0x499e0, 3, 64, 64), (0x5baa0, 0x5ba60, 1, 64, 32),
       (0x5caf0, 0x5cab0, 3, 32, 32), (0x61310, 0x612f0, 1, 32, 16), (0x6171e, 0x61710, 1, 16, 7)]


def layer(dat, L):
    wo, bo, k, ci, co = L
    w = np.frombuffer(dat, '<f2', k * k * ci * co, wo).reshape(k, k, ci, co)
    b = np.frombuffer(dat, '<f2', co, bo)
    return w, b


def conv_ref(x, w, b, stride=1, relu=True):
    k = w.shape[0]; pad = (k - 1) // 2 if stride == 1 else 2
    H, W, _ = x.shape
    xp = np.pad(x.astype(np.float32), ((pad, pad), (pad, pad), (0, 0)))
    Ho, Wo = (H + 2 * pad - k) // stride + 1, (W + 2 * pad - k) // stride + 1
    out = np.zeros((Ho, Wo, w.shape[3]), np.float32) + b.astype(np.float32)
    for ky in range(k):
        for kx in range(k):
            patch = xp[ky:ky + stride * Ho:stride, kx:kx + stride * Wo:stride]
            out += patch @ w[ky, kx].astype(np.float32)
    if relu:
        out = np.maximum(out, 0)
    return out.astype(np.float16)


def pool_ref(x):
    H, W, C = x.shape
    return x.reshape(H // 2, 2, W // 2, 2, C).max(axis=(1, 3))


def up_ref(x, skip):
    H, W, C = skip.shape
    h, w, _ = x.shape
    ys = np.arange(H) * 0.5 - 0.25; xs = np.arange(W) * 0.5 - 0.25
    y0 = np.floor(ys).astype(int); x0 = np.floor(xs).astype(int)
    ty = (ys - y0)[:, None, None].astype(np.float32); tx = (xs - x0)[None, :, None].astype(np.float32)
    ya, yb = np.clip(y0, 0, h - 1), np.clip(y0 + 1, 0, h - 1)
    xa, xb = np.clip(x0, 0, w - 1), np.clip(x0 + 1, 0, w - 1)
    xf = x.astype(np.float32)
    top = xf[ya][:, xa] * (1 - tx) + xf[ya][:, xb] * tx
    bot = xf[yb][:, xa] * (1 - tx) + xf[yb][:, xb] * tx
    return (top * (1 - ty) + bot * ty + skip.astype(np.float32)).astype(np.float16)


def dbf_ref(dat, x):
    """x: [H][W][13] fp16 -> [H][W][7] fp16 (H, W multiples of 8)"""
    P = [layer(dat, L) for L in DBF]
    a = conv_ref(conv_ref(x, *P[0]), *P[1]); s0 = a
    a = conv_ref(conv_ref(pool_ref(a), *P[2]), *P[3]); s1 = a
    a = conv_ref(conv_ref(pool_ref(a), *P[4]), *P[5]); s2 = a
    a = conv_ref(conv_ref(pool_ref(a), *P[6]), *P[7])
    a = conv_ref(conv_ref(up_ref(a, s2), *P[8]), *P[9])
    a = conv_ref(conv_ref(up_ref(a, s1), *P[10]), *P[11])
    a = conv_ref(conv_ref(up_ref(a, s0), *P[12]), *P[13])
    return conv_ref(a, *P[14], relu=False)


class Plan:
    def __init__(self, out):
        self.out = out; self.lines = []; os.makedirs(out + '/blobs', exist_ok=True)
        self.kernels = set(); self.n = 0; self.wmma = False

    def buffer(self, name, size, blob=None):
        if blob is not None:
            open(f'{self.out}/blobs/{name}.bin', 'wb').write(blob)
            self.lines.append(f'buffer {name} {size} file blobs/{name}.bin')
        else:
            self.lines.append(f'buffer {name} {size}')

    def glsl(self, name, src, lx=8, ly=8):
        if name not in self.kernels:
            subprocess.run(['glslangValidator', '-V', '--target-env', 'vulkan1.3', src, '-o', f'{self.out}/{name}.spv'], check=True,
                           capture_output=True)
            shutil.copy(src, self.out + '/')
            self.lines.append(f'glsl {name} {name}.spv {lx} {ly} 1')
            self.kernels.add(name)

    def run(self, kernel, w, h, bufs, consts, z=1):
        self.lines.append(f'pass {kernel} threads {w} {h} {z}')
        for i, b in enumerate(bufs):
            self.lines.append(f'  buf {i} {b}')
        for i, c in enumerate(consts):
            self.lines.append(f'  bytes {8 + i} {int(c).to_bytes(4, "little").hex()}')

    def save(self):
        decl = [l for l in self.lines if l.split(' ', 1)[0] in ('buffer', 'glsl')]
        rest = [l for l in self.lines if l.split(' ', 1)[0] not in ('buffer', 'glsl')]
        open(self.out + '/plan.txt', 'w').write('\n'.join(decl + rest) + '\n')


def pack_wmma_weights(w, k, ci, co):
    """HWIO fp16 [k][k][ci][co] -> [tap][cin block][cout block][16 cout][16 cin], zero padded to blocks of 16 (nn_conv_wmma.cl)"""
    cib, cob = (ci + 15) // 16, (co + 15) // 16
    out = np.zeros((k * k, cib * 16, cob * 16), np.float16)
    out[:, :ci, :co] = w.reshape(k * k, ci, co)
    out = out.reshape(k * k, cib, 16, cob, 16).transpose(0, 1, 3, 4, 2)    # tap, cib, cob, cout, cin
    return np.ascontiguousarray(out).tobytes()


def wmma_kernel(p, defs):
    """builds nn_conv_wmma.cl for one layer (LLVM, clang -x cl) into <p.out>/k and declares it in the plan; returns the kernel name"""
    import hashlib, tempfile
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
    sys.path.insert(0, root)
    from air import m4r_air
    src = os.path.join(root, 'shaders', 'nn_conv_wmma.cl')
    key = hashlib.sha256((open(src).read() + repr(sorted(defs.items())) + m4r_air.GPU).encode()).hexdigest()[:12]
    name = 'nn_conv_wmma-' + key
    if ('k', name) in p.kernels:
        return name
    os.makedirs(p.out + '/k', exist_ok=True)
    base = f'{p.out}/k/{name}'
    if not os.path.exists(base + '.hsaco'):
        with tempfile.TemporaryDirectory() as d:
            ll = d + '/k.ll'
            subprocess.run(['clang', '-x', 'cl', '-cl-std=CL2.0', '--target=amdgcn-amd-amdhsa', f'-mcpu={m4r_air.GPU}', '-nogpulib', '-O3',
                            '-S', '-emit-llvm', src, '-o', ll] + [f'-D{k}={v}' for k, v in defs.items()], check=True)
            text = open(ll).read()
        m4r_air.compile_ir(text, base + '.hsaco')
        man = {'kernel': 'nn_conv', 'threadgroup': [32, 1, 1], 'kernarg_size': 64, 'grid_offset': 32, 'samplers': [],
               'args': [dict(kind='buffer', location=i, offset=8 * i, name=n, access='read' if i != 1 else 'write')
                        for i, n in enumerate(('src', 'dst', 'weights', 'bias'))]}
        m4r_air.write_manifest(man, base)
    p.kernels.add(('k', name)) if isinstance(p.kernels, set) else p.kernels.__setitem__(('k', name), name)
    p.lines.append(f'kernel {name} k/{name}')
    return name


def wmma_conv_pass(p, dat, L, idx, src, dst, H, W, relu, nt):
    wo, bo, k, ci, co = L
    w, b = layer(dat, L)
    blob = pack_wmma_weights(w, k, ci, co)
    p.buffer(f'w{idx}', len(blob), blob)
    p.buffer(f'b{idx}', co * 2, dat[bo:bo + co * 2])
    name = wmma_kernel(p, dict(W=W, H=H, CIN=ci, COUT=co, K=k, NT=nt, RELU=1 if relu else 0))
    cob = (co + 15) // 16
    p.lines.append(f'pass {name} threads {((W + 15) // 16) * 32} {H} {(cob + nt - 1) // nt}')
    for i, bn in enumerate((src, dst, f'w{idx}', f'b{idx}')):
        p.lines.append(f'  buf {i} {bn}')


def conv_pass(p, dat, L, idx, src, dst, Hin, Win, stride=1, relu=True):
    wo, bo, k, ci, co = L
    if getattr(p, 'wmma', False) and stride == 1:
        nt = 1 if Hin * Win < 40000 else min(4, (co + 15) // 16)   # low resolution: spread the channel tiles over waves
        wmma_conv_pass(p, dat, L, idx, src, dst, Hin, Win, relu, nt)
        return Hin, Win
    pad = (k - 1) // 2 if stride == 1 else 2
    Ho, Wo = (Hin + 2 * pad - k) // stride + 1, (Win + 2 * pad - k) // stride + 1
    p.buffer(f'w{idx}', k * k * ci * co * 2, dat[wo:wo + k * k * ci * co * 2])
    p.buffer(f'b{idx}', co * 2, dat[bo:bo + co * 2])
    p.glsl('nn_conv', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'shaders', 'nn_conv.comp'))
    p.run('nn_conv', Wo, Ho, [src, dst, f'w{idx}', f'b{idx}'],
          [Wo, Ho, ci, co, Win, Hin, k | stride << 8 | pad << 16, 1 if relu else 0], z=(co + 7) // 8)
    return Ho, Wo


def dbf_net(p, dat, H, W, in_name='dbf_in', out_name='dbf_out'):
    """appends the DBFNet passes to plan generator p (needs buffer(name, size[, blob]), glsl(name, src), run(kernel, w, h, bufs, consts));
    in_name is [H][W][13] f16 (NHWC), out_name [H][W][7]; H and W multiples of 8"""
    assert H % 8 == 0 and W % 8 == 0
    sh = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'shaders')
    P = DBF
    def buf(name, h, w, c):
        p.buffer(name, h * w * c * 2)
    def conv(i, src, dst, h, w, relu=True):
        buf(dst, h, w, P[i][4]); conv_pass(p, dat, P[i], i, src, dst, h, w, relu=relu)
    def pool(src, dst, h, w, c):
        buf(dst, h // 2, w // 2, c)
        p.glsl('nn_pool', sh + '/nn_pool.comp'); p.run('nn_pool', w // 2, h // 2, [src, dst], [w // 2, h // 2, c])
    def up(src, skip, dst, h, w, c):
        buf(dst, h, w, c)
        p.glsl('nn_upadd', sh + '/nn_upadd.comp'); p.run('nn_upadd', w, h, [src, skip, dst], [w, h, c])
    conv(0, in_name, 't0', H, W); conv(1, 't0', 's0', H, W)
    pool('s0', 'p0', H, W, 32)
    h1, w1 = H // 2, W // 2
    conv(2, 'p0', 't1', h1, w1); conv(3, 't1', 's1', h1, w1)
    pool('s1', 'p1', h1, w1, 64)
    h2, w2 = h1 // 2, w1 // 2
    conv(4, 'p1', 't2', h2, w2); conv(5, 't2', 's2', h2, w2)
    pool('s2', 'p2', h2, w2, 64)
    h3, w3 = h2 // 2, w2 // 2
    conv(6, 'p2', 't3', h3, w3); conv(7, 't3', 'bott', h3, w3)
    up('bott', 's2', 'u2', h2, w2, 64); conv(8, 'u2', 't4', h2, w2); conv(9, 't4', 'd2', h2, w2)
    up('d2', 's1', 'u1', h1, w1, 64); conv(10, 'u1', 't5', h1, w1); conv(11, 't5', 'd1', h1, w1)
    up('d1', 's0', 'u0', H, W, 32); conv(12, 'u0', 't6', H, W); conv(13, 't6', 't7', H, W)
    buf(out_name, H, W, 7); conv_pass(p, dat, P[14], 14, 't7', out_name, H, W, relu=False)


def dbf_plan(dat, out, H, W):
    p = Plan(out)
    p.buffer('dbf_in', H * W * 13 * 2)
    dbf_net(p, dat, H, W)
    p.save()


if __name__ == '__main__':
    mode, res = sys.argv[1], sys.argv[2]
    dat = open(os.path.join(res, 'emit_dbfnet_v3_2_constants.dat'), 'rb').read()
    if mode == 'dbf':
        dbf_plan(dat, sys.argv[3], int(sys.argv[4]), int(sys.argv[5]))
    elif mode == 'dbf-ref':
        H, W = int(sys.argv[3]), int(sys.argv[4])
        x = np.fromfile(sys.argv[5], '<f2').reshape(H, W, 13)
        dbf_ref(dat, x).astype('<f2').tofile(sys.argv[6])
