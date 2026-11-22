#!/usr/bin/env python3
"""MetalFX denoiser networks (DBFNet, UBFNet): layer tables, numpy reference and plan generator.

The networks are Apple's (weights from emit_dbfnet_v3_2_constants.dat / emit_ubfnet_v3_2_constants.dat in the user's MetalFX
Resources folder, layer tables in docs/metalfx-denoiser.md); the convolutions run on m4r's own plain kernels
(shaders/nn_*.comp: fp32 accumulation, fp16 activations) instead of Apple's fused MXU kernels.

usage: denoise_nets.py dbf <Resources dir> <out plan dir> <H> <W>     writes a plan (input dbf_in, output dbf_out, NHWC fp16)
       denoise_nets.py dbf-ref <Resources dir> <H> <W> <input.bin> <output.bin>   numpy reference of the same
SPDX-License-Identifier: GPL-3.0-only
"""
import os, re, shutil, subprocess, sys
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


def conv_ref(x, w, b, stride=1, relu=True, pad=None, sigmoid=False):
    k = w.shape[0]; pad = ((k - 1) // 2 if stride == 1 else 2) if pad is None else pad
    H, W, _ = x.shape
    xp = np.pad(x.astype(np.float32), ((pad, pad), (pad, pad), (0, 0)))
    Ho, Wo = (H + 2 * pad - k) // stride + 1, (W + 2 * pad - k) // stride + 1
    out = np.zeros((Ho, Wo, w.shape[3]), np.float32) + b.astype(np.float32)
    for ky in range(k):
        for kx in range(k):
            patch = xp[ky:ky + stride * Ho:stride, kx:kx + stride * Wo:stride]
            out += patch @ w[ky, kx].astype(np.float32)
    if sigmoid:
        out = 1.0 / (1.0 + np.exp(-out))
    elif relu:
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


def nn_impl():
    """how the convolutions run: 'wmma' (gfx11's WMMA, RDNA3 / 3.5), 'soft' (the WMMA kernels with each product as DPP dot products,
    shaders/nn_dot.h wmma_soft) or 'dot' (lane-per-pixel dot products with scalar weights, shaders/nn_conv_dot.cl / nn_cell.cl DOT=1).
    GPUs without gfx11 WMMA get 'dot' (faster than 'soft' on gfx1101 at RDNA2's issue rate: DBF cells 0.80 vs 1.45 ms, UBF convs
    0.40 vs 0.67 at 720p -> 1440p); M4R_NN_IMPL=wmma|soft|dot overrides (all three give the same bits)"""
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
    sys.path.insert(0, root)
    from air import m4r_air
    if os.environ.get('M4R_NN_IMPL'):
        return os.environ['M4R_NN_IMPL']
    return 'wmma' if m4r_air.family() == 'gfx11' else 'dot'


def use_dot():
    return nn_impl() == 'dot'


def cl_build(src, defs, out_ll):
    """OpenCL C (shaders/*.cl) -> LLVM IR text for m4r_air.GPU"""
    from air import m4r_air
    sh = os.path.dirname(src)
    subprocess.run(['clang', '-x', 'cl', '-cl-std=CL2.0', '--target=amdgcn-amd-amdhsa', f'-mcpu={m4r_air.GPU}', '-nogpulib', '-O3', f'-I{sh}',
                    '-S', '-emit-llvm', src, '-o', out_ll] + [f'-D{k}={v}' for k, v in defs.items()], check=True)
    return open(out_ll).read()


def src_key(src, defs, gpu):
    """cache key of a kernel built from src (and the headers it includes from shaders/) with defs"""
    import hashlib
    text = open(src).read()
    for inc in sorted(set(re.findall(r'#include "([^"]+)"', text))):
        text += open(os.path.join(os.path.dirname(src), inc)).read()
    return hashlib.sha256((text + repr(sorted(defs.items())) + gpu).encode()).hexdigest()[:12]


def dot_kernel(p, defs):
    """builds nn_conv_dot.cl for one layer into <p.out>/k and declares it in the plan; returns the kernel name"""
    import tempfile
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
    sys.path.insert(0, root)
    from air import m4r_air
    src = os.path.join(root, 'shaders', 'nn_conv_dot.cl')
    name = 'nn_conv_dot-' + src_key(src, defs, m4r_air.GPU)
    if ('k', name) in p.kernels:
        return name
    os.makedirs(p.out + '/k', exist_ok=True)
    base = f'{p.out}/k/{name}'
    if not os.path.exists(base + '.hsaco'):
        with tempfile.TemporaryDirectory() as d:
            text = cl_build(src, defs, d + '/k.ll')
        m4r_air.compile_ir(text, base + '.hsaco')
        if hsaco_scratch(base + '.hsaco'):
            raise RuntimeError(f'{name}: uses scratch ({defs})')
        man = {'kernel': 'nn_conv', 'threadgroup': [32, 1, 1], 'kernarg_size': 64, 'grid_offset': 32, 'samplers': [],
               'args': [dict(kind='buffer', location=i, offset=8 * i, name=n, access='read' if i != 1 else 'write')
                        for i, n in enumerate(('src', 'dst', 'weights', 'bias'))]}
        m4r_air.write_manifest(man, base)
    p.kernels.add(('k', name)) if isinstance(p.kernels, set) else p.kernels.__setitem__(('k', name), name)
    p.lines.append(f'kernel {name} k/{name}')
    return name


def dot_conv_pass(p, dat, L, idx, src, dst, Hin, Win, act, stride=1, pad=None):
    wo, bo, k, ci, co = L
    pad = (k - 1) // 2 if pad is None else pad
    H, W = (Hin + 2 * pad - k) // stride + 1, (Win + 2 * pad - k) // stride + 1
    w, b = layer(dat, L)
    blob = pack_wmma_weights(w, k, ci, co)
    p.buffer(f'w{idx}', len(blob), blob)
    p.buffer(f'b{idx}', co * 2, dat[bo:bo + co * 2])
    # output channels per wave: a multiple of 8 dividing cout rounded up to 8 (the packed weights are zero up to blocks of 16);
    # MPX rows per lane (weight reuse) and as many channels as keep enough waves for the GPU (M4R_DOT_WAVES)
    cop = (co + 7) // 8 * 8
    mpx = int(os.environ.get('M4R_DOT_CMPX', '2'))
    acc = int(os.environ.get('M4R_DOT_CACC', '64'))          # sums per lane: NC * MPX
    cands = [c for c in range(8, cop + 1, 8) if cop % c == 0 and c * mpx <= acc] or [8]
    want = int(os.environ.get('M4R_DOT_WAVES', '1024'))   # swept on gfx1101 (M4R_NN_IMPL=dot, VOPD off): 2 rows, 1024 waves
    while True:
        rows = ((W + 31) // 32) * ((H + mpx - 1) // mpx)
        nc = next((c for c in reversed(cands) if rows * (cop // c) >= want), cands[0])
        if rows * (cop // nc) >= want or mpx == 1:
            break
        mpx //= 2                                             # small layers: fewer rows per lane, more waves
        cands = [c for c in range(8, cop + 1, 8) if cop % c == 0 and c * mpx <= acc] or [8]
    if co % 8:   # all channels in one wave (its output goes out as one contiguous span, nn_conv_dot.cl)
        nc = cop
        while mpx > 1 and nc * mpx > acc:
            mpx //= 2
    name = dot_kernel(p, dict(W=W, H=H, WI=Win, HI=Hin, CIN=ci, COUT=co, K=k, ST=stride, PADV=pad, NC=nc, MPX=mpx, ACT=act))
    p.lines.append(f'pass {name} threads {((W + 31) // 32) * 32} {(H + mpx - 1) // mpx} {cop // nc}')
    for i, bn in enumerate((src, dst, f'w{idx}', f'b{idx}')):
        p.lines.append(f'  buf {i} {bn}')


def wmma_kernel(p, defs):
    """builds nn_conv_wmma.cl for one layer (LLVM, clang -x cl) into <p.out>/k and declares it in the plan; returns the kernel name"""
    import tempfile
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
    sys.path.insert(0, root)
    from air import m4r_air
    src = os.path.join(root, 'shaders', 'nn_conv_wmma.cl')
    if nn_impl() == 'soft':
        defs = dict(defs, SOFTW=1)
    name = 'nn_conv_wmma-' + src_key(src, defs, m4r_air.GPU)
    if ('k', name) in p.kernels:
        return name
    os.makedirs(p.out + '/k', exist_ok=True)
    base = f'{p.out}/k/{name}'
    if not os.path.exists(base + '.hsaco'):
        with tempfile.TemporaryDirectory() as d:
            text = cl_build(src, defs, d + '/k.ll')
        m4r_air.compile_ir(text, base + '.hsaco')
        if hsaco_scratch(base + '.hsaco'):
            raise RuntimeError(f'{name}: uses scratch ({defs})')
        man = {'kernel': 'nn_conv', 'threadgroup': [32, 1, 1], 'kernarg_size': 64, 'grid_offset': 32, 'samplers': [],
               'args': [dict(kind='buffer', location=i, offset=8 * i, name=n, access='read' if i != 1 else 'write')
                        for i, n in enumerate(('src', 'dst', 'weights', 'bias'))]}
        m4r_air.write_manifest(man, base)
    p.kernels.add(('k', name)) if isinstance(p.kernels, set) else p.kernels.__setitem__(('k', name), name)
    p.lines.append(f'kernel {name} k/{name}')
    return name


def wmma_conv_pass(p, dat, L, idx, src, dst, Hin, Win, act, nt, stride=1, pad=None):
    wo, bo, k, ci, co = L
    pad = (k - 1) // 2 if pad is None else pad
    H, W = (Hin + 2 * pad - k) // stride + 1, (Win + 2 * pad - k) // stride + 1
    w, b = layer(dat, L)
    blob = pack_wmma_weights(w, k, ci, co)
    p.buffer(f'w{idx}', len(blob), blob)
    p.buffer(f'b{idx}', co * 2, dat[bo:bo + co * 2])
    name = wmma_kernel(p, dict(W=W, H=H, WI=Win, HI=Hin, CIN=ci, COUT=co, K=k, ST=stride, PADV=pad, NT=nt, ACT=act))
    cob = (co + 15) // 16
    p.lines.append(f'pass {name} threads {((W + 15) // 16) * 32} {H} {(cob + nt - 1) // nt}')
    for i, bn in enumerate((src, dst, f'w{idx}', f'b{idx}')):
        p.lines.append(f'  buf {i} {bn}')


def conv_pass(p, dat, L, idx, src, dst, Hin, Win, stride=1, relu=True, pad=None, act=None):
    """one convolution; act None = relu if relu else none, or 0 none / 1 relu / 2 sigmoid; returns the output size"""
    wo, bo, k, ci, co = L
    act = (1 if relu else 0) if act is None else act
    pad = ((k - 1) // 2 if stride == 1 else 2) if pad is None else pad
    if getattr(p, 'wmma', False):
        Ho, Wo = (Hin + 2 * pad - k) // stride + 1, (Win + 2 * pad - k) // stride + 1
        # channel tiles per wave: 1 below 10000 output pixels (spread over waves), 2 above (swept per mode on gfx1101: 10000 beat 40000 by 2-5% at
        # Quality to Ultra Performance; 4 was 2x slower on the
        # 320x180 64->64 layers; results are identical for any value, only the split over waves changes)
        nt = 1 if Ho * Wo < int(os.environ.get('M4R_NT_PX', '10000')) else min(2, (co + 15) // 16)   # M4R_NT_PX: threshold experiment
        if use_dot():
            dot_conv_pass(p, dat, L, idx, src, dst, Hin, Win, act, stride, pad)
        else:
            wmma_conv_pass(p, dat, L, idx, src, dst, Hin, Win, act, nt, stride, pad)
        return Ho, Wo
    Ho, Wo = (Hin + 2 * pad - k) // stride + 1, (Win + 2 * pad - k) // stride + 1
    p.buffer(f'w{idx}', k * k * ci * co * 2, dat[wo:wo + k * k * ci * co * 2])
    p.buffer(f'b{idx}', co * 2, dat[bo:bo + co * 2])
    p.glsl('nn_conv', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'shaders', 'nn_conv.comp'))
    p.run('nn_conv', Wo, Ho, [src, dst, f'w{idx}', f'b{idx}'],
          [Wo, Ho, ci, co, Win, Hin, k | stride << 8 | pad << 16, act], z=(co + 7) // 8)
    return Ho, Wo


def hsaco_scratch(path):
    """private segment (scratch) bytes per lane of a code object's kernel"""
    notes = subprocess.run(['llvm-readelf', '--notes', path], capture_output=True, text=True, check=True).stdout
    m = re.search(r'\.private_segment_fixed_size:\s*(\d+)', notes)
    return int(m.group(1)) if m else -1


def cell_kernel(p, defs, spill_ok=True):
    """builds shaders/nn_cell.cl for one DBFNet cell into <p.out>/k and declares it in the plan; returns (kernel name, threads per group),
    or None (nothing declared) when the kernel still spills and spill_ok is false"""
    import hashlib, tempfile
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
    sys.path.insert(0, root)
    from air import m4r_air
    src = os.path.join(root, 'shaders', 'nn_cell.cl')
    if defs.get('DOT'):
        nth = defs['TY'] // defs.get('MPX', 1) * 32 * defs.get('WC', 1)
    else:
        nth = (defs['TX'] // 16) * defs['TY'] // defs['MP'] * 32 * defs.get('WC', 1)
    name = 'nn_cell-' + src_key(src, defs, m4r_air.GPU)
    if ('k', name) in p.kernels:
        return name, nth
    os.makedirs(p.out + '/k', exist_ok=True)
    base = f'{p.out}/k/{name}'
    if not os.path.exists(base + '.hsaco'):
        # a variant that spills (scratch) is rebuilt with less pipelining: PIPEU 1, then without the pipelined loop
        tries = [dict(defs)]
        if defs.get('DOT'):   # DOT: fewer fill loads in flight
            tries += [dict(defs, FILLU=f) for f in (2, 1) if f < defs.get('FILLU', 8)]
        else:
            if defs.get('PIPEU', 1) > 1: tries.append(dict(defs, PIPEU=1))
            tries.append(dict(defs, PIPE=0))
        for dd in tries:
            with tempfile.TemporaryDirectory() as d:
                text = cl_build(src, dd, d + '/k.ll')
            m4r_air.compile_ir(text, base + '.hsaco')
            scratch = hsaco_scratch(base + '.hsaco')
            if scratch == 0:
                break
            print(f'{name}: scratch {scratch} with PIPE={dd.get("PIPE", 1)} PIPEU={dd.get("PIPEU", 1)}', file=sys.stderr)
        names = ('src', 'skip', 'dst', 'w1', 'b1', 'w2', 'b2', 'w3', 'b3')
        tex = ('feat', 'noise', 'tw')   # OUTTEX: the network's output textures (MetalFX's dbfnetv1_output_channels done in the last cell)
        n = len(names) + len(tex)
        man = {'kernel': 'nn_cell', 'threadgroup': [nth, 1, 1], 'kernarg_size': 8 * n + 32, 'grid_offset': 8 * n, 'samplers': [],
               'args': [dict(kind='buffer', location=i, offset=8 * i, name=nm, access='write' if nm == 'dst' else 'read') for i, nm in enumerate(names)]
                       + ([dict(kind='texture', location=j, offset=8 * (len(names) + j), name=nm, access='write') for j, nm in enumerate(tex)]
                          if defs.get('OUTTEX') else [])}
        m4r_air.write_manifest(man, base)
    if not spill_ok and hsaco_scratch(base + '.hsaco'):
        for ext in os.listdir(p.out + '/k'):   # not part of the plan
            if ext.startswith(name + '.'):
                os.remove(f'{p.out}/k/{ext}')
        return None
    p.kernels.add(('k', name)) if isinstance(p.kernels, set) else p.kernels.__setitem__(('k', name), name)
    p.lines.append(f'kernel {name} k/{name}')
    return name, nth


# tile per cell size (TX, TY, MP): small levels get small tiles so that the GPU has enough workgroups (M4R_CELL_TILE=TX,TY,MP overrides all)
def cell_tile(H, W):
    if os.environ.get('M4R_CELL_TILE'):
        return tuple(map(int, os.environ['M4R_CELL_TILE'].split(',')))
    px = H * W
    return (32, 4, 2) if px >= 40000 else (16, 4, 1) if px >= 10000 else (16, 2, 1)


def dbf_net_fused(p, dat, H, W, in_name='dbf_in', out_name='dbf_out', out_tex=None, in_stride=None):
    """DBFNet as 7 fused cells (shaders/nn_cell.cl), bit-identical to the layer-by-layer passes of dbf_net. out_tex = (feat, noise, tw, width,
    height): the last cell writes MetalFX's three output textures itself (what dbfnetv1_output_channels does) instead of out_name"""
    P = DBF
    for i, L in enumerate(P):
        w, b = layer(dat, L)
        blob = pack_wmma_weights(w, L[2], L[3], L[4])
        p.buffer(f'w{i}', len(blob), blob)
        p.buffer(f'b{i}', L[4] * 2, dat[L[1]:L[1] + L[4] * 2])
    upmix = int(os.environ.get('M4R_UPMIX', '1'))
    fuse_mask = int(os.environ.get('M4R_FUSE_CELLS', '127'))   # experiment: bit k = cell k fused, else the old layer passes
    sh = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'shaders')
    ncell = [0]
    def cell(mode, src, skip, dst, h, w, i, last=False, sout=None):
        cin, c1, c2 = P[i][3], P[i][4], P[i + 1][4]
        c3 = P[i + 2][4] if last else 0
        cout = c3 or c2
        k = ncell[0]; ncell[0] += 1
        otex = out_tex if last and out_tex else None
        if not otex:
            p.buffer(dst, h * w * (sout or cout) * 2)
        if not fuse_mask >> k & 1:
            x = src
            if mode == 1:
                x = f'xp{k}'; p.buffer(x, h * w * cin * 2)
                p.glsl('nn_pool', sh + '/nn_pool.comp'); p.run('nn_pool', w * cin // 4, h, [src, x], [w, h, cin])
            elif mode == 2:
                x = f'xu{k}'; p.buffer(x, h * w * cin * 2)
                p.glsl('nn_upadd', sh + '/nn_upadd.comp'); p.run('nn_upadd', w * cin // 4, h, [src, skip, x], [w, h, cin])
            p.buffer(f'xa{k}', h * w * c1 * 2); conv_pass(p, dat, P[i], f'{i}u', x, f'xa{k}', h, w)
            if last:
                p.buffer(f'xb{k}', h * w * c2 * 2); conv_pass(p, dat, P[i + 1], f'{i + 1}u', f'xa{k}', f'xb{k}', h, w)
                conv_pass(p, dat, P[i + 2], f'{i + 2}u', f'xb{k}', dst, h, w, relu=False)
            else:
                conv_pass(p, dat, P[i + 1], f'{i + 1}u', f'xa{k}', dst, h, w)
            return
        if use_dot():
            # DOT: wave = 32 pixels of a row, TY rows per workgroup, WC waves per row splitting the output channels (small cells: more
            # waves in flight); swept on gfx1101 with M4R_NN_DOT=1
            # (swept on gfx1101 with M4R_NN_IMPL=dot and VOPD off, i.e. RDNA2's issue rate): MPX pixels per lane = reuse of each scalar weight
            # load, the cells' limit (4 on the 64-channel cells, 2 on the 32-channel ones and the smallest), at most 64 sums per lane
            # (~91 VGPRs) by splitting the output channels over WC waves, 8 of them below 20000 pixels (more waves in flight)
            ch = 4
            mpx = 4 if c1 >= 64 and h * w >= 5000 else 2
            mpx = int(os.environ.get(f'M4R_DOT_MPX{k}', os.environ.get('M4R_DOT_MPX', mpx)))
            ty = 4 if h * w >= 40000 else 2
            ty = int(os.environ.get(f'M4R_DOT_TY{k}', os.environ.get('M4R_DOT_TY', ty)))
            ty = (max(ty, mpx) + mpx - 1) // mpx * mpx
            wcs = max(1, c1 * mpx // int(os.environ.get('M4R_DOT_ACC', '64')))
            if h * w < 20000:
                wcs = max(wcs, 8)
            while wcs > 1 and (c1 % (wcs * ch) or c2 % (wcs * ch) or c3):
                wcs //= 2
            wcs = int(os.environ.get(f'M4R_CELL_WC{k}', os.environ.get('M4R_CELL_WC', wcs)))
            defs = dict(DOT=1, CH=ch, WC=wcs, W=w, H=h, MODE=mode, CIN=cin, SIN=(in_stride if (mode == 0 and in_stride) else cin), C1=c1, C2=c2, C3=c3,
                        SOUT=sout or cout, TX=32, TY=ty, MP=1, MPX=mpx,
                        UPMIX=upmix, FILLU=int(os.environ.get('M4R_DOT_FILLU', '2')))
            for env in ('M4R_CELL_DEFS', f'M4R_CELL_DEFS{k}'):
                defs.update({a: int(v) if v.lstrip('-').isdigit() else v for a, v in (kv.split('=', 1) for kv in os.environ.get(env, '').split(';') if kv)})
            if otex:
                defs.update(OUTTEX=1, TXW=otex[3], TXH=otex[4])
            name, nth = cell_kernel(p, defs)
            if hsaco_scratch(f'{p.out}/k/{name}.hsaco'):   # RDNA2 cannot run kernels with scratch (vkhip)
                raise RuntimeError(f'{name}: uses scratch ({defs})')
            p.lines.append(f'pass {name} threads {((w + 31) // 32) * nth} {(h + defs["TY"] - 1) // defs["TY"]} 1')
            args = (src, skip or src, src if otex else dst, f'w{i}', f'b{i}', f'w{i + 1}', f'b{i + 1}', f'w{i + 2}' if last else f'w{i + 1}', f'b{i + 2}' if last else f'b{i + 1}')
            for kk, bn in enumerate(args):
                p.lines.append(f'  buf {kk} {bn}')
            if otex:
                for j in range(3):
                    p.lines.append(f'  tex {j} {otex[j]}')
            return
        tile = cell_tile(h, w)
        if cin >= 64 and tile[2] > 1 and not os.environ.get('M4R_CELL_TILE'):
            tile = (16, 4, 1)   # 64 -> 64 3x3 cells: 1 pixel group per wave is faster (measured 720p and 2560x1440, with the pipelined loop)
        if os.environ.get(f'M4R_CELL_TILE{k}'):   # per-cell override (experiments)
            tile = tuple(map(int, os.environ[f'M4R_CELL_TILE{k}'].split(',')))
        # channel split across waves for the small cells (measured on gfx1101: 4 below 5000 pixels, 2 below 10000; exact either way)
        wcs = 4 if h * w < 5000 else 2 if h * w < 10000 else 1
        cb1 = (c1 + 15) // 16
        while wcs > 1 and (cb1 % wcs or c3):
            wcs //= 2
        wcs = int(os.environ.get(f'M4R_CELL_WC{k}', os.environ.get('M4R_CELL_WC', wcs)))   # experiments
        def cell_defs(tx, ty, mp):
            # unroll of the pipelined 3x3 step loop (measured 720p and 2560x1440): 6 on the 2-pixel-group cells (cells 1 and 6 faster, 0 the
            # same), 4 on the others (6 and 8 slower); a variant that spills falls back to less (cell_kernel)
            pipeu = 6 if mp > 1 else 4
            defs = dict(WC=wcs, W=w, H=h, MODE=mode, CIN=cin, SIN=(in_stride if (mode == 0 and in_stride) else cin), C1=c1, C2=c2, C3=c3, SOUT=sout or cout, TX=tx, TY=ty, MP=mp, UPMIX=upmix,
                        PIPEU=pipeu, TAPU=int(os.environ.get('M4R_TAPU', '1')), CBU=int(os.environ.get('M4R_CBU', '1')))
            for env in ('M4R_CELL_DEFS', f'M4R_CELL_DEFS{k}'):   # experiments (all cells / cell k)
                defs.update({a: int(v) if v.lstrip('-').isdigit() else v for a, v in (kv.split('=', 1) for kv in os.environ.get(env, '').split(';') if kv)})
            if otex:
                defs.update(OUTTEX=1, TXW=otex[3], TXH=otex[4])
            if nn_impl() == 'soft':
                defs['SOFTW'] = 1
            return defs
        # a 2-pixel-group tile that spills even without pipelining (cell 4 at 2560x1440) is slower than the 1-group tile without scratch
        r = cell_kernel(p, cell_defs(*tile), spill_ok=tile[2] == 1)
        if r is None:
            tile = (16, 4, 1)
            r = cell_kernel(p, cell_defs(*tile))
        name, nth = r
        tx, ty, mp = tile
        p.lines.append(f'pass {name} threads {((w + tx - 1) // tx) * nth} {(h + ty - 1) // ty} 1')
        args = (src, skip or src, src if otex else dst, f'w{i}', f'b{i}', f'w{i + 1}', f'b{i + 1}', f'w{i + 2}' if last else f'w{i + 1}', f'b{i + 2}' if last else f'b{i + 1}')
        for k, bn in enumerate(args):
            p.lines.append(f'  buf {k} {bn}')
        if otex:
            for j in range(3):
                p.lines.append(f'  tex {j} {otex[j]}')   # texture locations count separately from buffers
    h1, w1, h2, w2, h3, w3 = H // 2, W // 2, H // 4, W // 4, H // 8, W // 8
    cell(0, in_name, None, 's0', H, W, 0)
    cell(1, 's0', None, 's1', h1, w1, 2)
    cell(1, 's1', None, 's2', h2, w2, 4)
    cell(1, 's2', None, 'bott', h3, w3, 6)
    cell(2, 'bott', 's2', 'd2', h2, w2, 8)
    cell(2, 'd2', 's1', 'd1', h1, w1, 10)
    cell(2, 'd1', 's0', out_name, H, W, 12, last=True, sout=7)


def dbf_net(p, dat, H, W, in_name='dbf_in', out_name='dbf_out', out_tex=None, in_stride=None):
    """in_stride: halves per input pixel when the producer pads the 13 channels (m4r's LOWRES writes 16)"""
    """appends the DBFNet passes to plan generator p (needs buffer(name, size[, blob]), glsl(name, src), run(kernel, w, h, bufs, consts));
    in_name is [H][W][13] f16 (NHWC), out_name [H][W][7]; H and W multiples of 8"""
    assert H % 8 == 0 and W % 8 == 0
    if getattr(p, 'wmma', False) and not os.environ.get('M4R_DBF_UNFUSED'):
        dbf_net_fused(p, dat, H, W, in_name, out_name, out_tex, in_stride)
        return bool(out_tex)
    assert not out_tex, 'out_tex needs the fused cells'
    sh = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'shaders')
    P = DBF
    def buf(name, h, w, c):
        p.buffer(name, h * w * c * 2)
    def conv(i, src, dst, h, w, relu=True):
        buf(dst, h, w, P[i][4]); conv_pass(p, dat, P[i], i, src, dst, h, w, relu=relu)
    def pool(src, dst, h, w, c):
        buf(dst, h // 2, w // 2, c)
        p.glsl('nn_pool', sh + '/nn_pool.comp'); p.run('nn_pool', (w // 2) * c // 4, h // 2, [src, dst], [w // 2, h // 2, c])
    def up(src, skip, dst, h, w, c):
        buf(dst, h, w, c)
        p.glsl('nn_upadd', sh + '/nn_upadd.comp'); p.run('nn_upadd', w * c // 4, h, [src, skip, dst], [w, h, c])
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


# ---- UBFNet (the denoiser's upscaling network): 5 channels in at HxW, 11 channels out at H/2 x W/2 ---------------------------------
# (weights offset, bias offset, kernel, cin, cout) in emit_ubfnet_v3_2_constants.dat (docs/metalfx-denoiser.md); the head (3x3 16 -> 44)
# is not stored as plain fp16 there, so it comes from ubf.mtlpackage's resources.bin: bias at 2, weights at 39143 (blob headers of 2 and 3 bytes)
UBF = dict(E0=(0x10, 0x0, 6, 5, 8), E1=(0xb90, 0xb50, 6, 8, 32), E2=(0x53d0, 0x5390, 3, 32, 32), E3=(0x9c50, 0x9bd0, 3, 32, 64),
           E4=(0x12d50, 0x12c50, 3, 64, 128), B1=(0x36e50, 0x36d50, 1, 128, 128), B2=(0x3eed0, 0x3ee50, 1, 128, 64),
           U1a=(0x42f60, 0x42ee0, 1, 64, 64), U1b=(0x44fa0, 0x44f60, 1, 64, 32), U2a=(0x45ff0, 0x45fb0, 1, 32, 32),
           U2b=(0x46810, 0x467f0, 1, 32, 16), U3a=(0x46c60, 0x46c20, 1, 16, 32), U3b=(0x47080, 0x47060, 1, 32, 16),
           U3c=(0x474a0, 0x47480, 1, 16, 16))
UBF_HEAD = (39143, 2, 3, 16, 44)


def ubf_head_blob(res_dir):
    return open(os.path.join(res_dir, 'ubf.mtlpackage', 'library.mpsgraphpackage', 'resources.bin'), 'rb').read()


def ubf_ref(dat, head, x):
    """numpy reference: x [H][W][5] fp16 (H, W multiples of 32) -> [H/2][W/2][11] fp16"""
    P = {k: layer(dat, v) for k, v in UBF.items()}
    hw, hb = layer(head, UBF_HEAD)
    c = lambda n, a, st=1, pad=None: conv_ref(a, *P[n], stride=st, pad=pad)
    e0 = c('E0', x, 2, 2); e1 = c('E1', e0, 2, 2); e2 = c('E2', e1, 2, 1); e3 = c('E3', e2, 2, 1)
    a = c('E4', pool_ref(e3))
    a = c('B2', c('B1', a))
    a = c('U1b', c('U1a', up_ref(a, e3)))
    a = c('U2b', c('U2a', up_ref(a, e2)))
    a = c('U3c', c('U3b', c('U3a', up_ref(a, e1[..., :16]))))
    h = conv_ref(a, hw, hb, relu=False, sigmoid=False)         # [H/4][W/4][44]
    H4, W4, _ = h.shape
    t = h.astype(np.float32).reshape(H4, W4, 11, 2, 2).transpose(0, 3, 1, 4, 2).reshape(H4 * 2, W4 * 2, 11)   # channel = c * 4 + i * 2 + j
    return (1.0 / (1.0 + np.exp(-t))).astype(np.float16)


def ubf_net(p, dat, head, H, W, in_name='ubf_in', out_name='ubf_out', pre='', sigmoid=True):
    """appends the UBFNet passes to plan generator p; in_name [H][W][5] f16 (NHWC), out_name [H/2][W/2][11]; H, W multiples of 32.
    Buffers it creates are named pre + <layer>."""
    assert H % 32 == 0 and W % 32 == 0
    sh = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'shaders')
    def buf(name, h, w, c):
        p.buffer(pre + name, h * w * c * 2)
    n = [0]
    def conv(key, src, dst, h, w, stride=1, pad=None, L=None, d=None, act=None):
        L = L or UBF[key]
        ho = (h + 2 * (((L[2] - 1) // 2) if pad is None else pad) - L[2]) // stride + 1
        wo = (w + 2 * (((L[2] - 1) // 2) if pad is None else pad) - L[2]) // stride + 1
        buf(dst, ho, wo, L[4])
        n[0] += 1
        conv_pass(p, d if d is not None else dat, L, pre + 'u' + key, pre + src if src != in_name else src, pre + dst, h, w, stride=stride, pad=pad, act=act)
        return ho, wo
    def pool(src, dst, h, w, c):
        buf(dst, h // 2, w // 2, c)
        p.glsl('nn_pool', sh + '/nn_pool.comp'); p.run('nn_pool', (w // 2) * c // 4, h // 2, [pre + src, pre + dst], [w // 2, h // 2, c])
    def up(src, skip, dst, h, w, c, skip_c):
        buf(dst, h, w, c)
        p.glsl('nn_upadd', sh + '/nn_upadd.comp'); p.run('nn_upadd', w * c // 4, h, [pre + src, pre + skip, pre + dst], [w, h, c, skip_c])
    h0, w0 = conv('E0', in_name, 'e0', H, W, 2, 2, act=1)
    h1, w1 = conv('E1', 'e0', 'e1', h0, w0, 2, 2, act=1)
    h2, w2 = conv('E2', 'e1', 'e2', h1, w1, 2, 1, act=1)
    h3, w3 = conv('E3', 'e2', 'e3', h2, w2, 2, 1, act=1)
    pool('e3', 'p3', h3, w3, 64)
    h4, w4 = h3 // 2, w3 // 2
    conv('E4', 'p3', 'e4', h4, w4, act=1)
    conv('B1', 'e4', 'b1', h4, w4, act=1); conv('B2', 'b1', 'b2', h4, w4, act=1)
    up('b2', 'e3', 'u1', h3, w3, 64, 64)
    conv('U1a', 'u1', 'u1a', h3, w3, act=1); conv('U1b', 'u1a', 'u1b', h3, w3, act=1)
    up('u1b', 'e2', 'u2', h2, w2, 32, 32)
    conv('U2a', 'u2', 'u2a', h2, w2, act=1); conv('U2b', 'u2a', 'u2b', h2, w2, act=1)
    up('u2b', 'e1', 'u3', h1, w1, 16, 32)                     # skip: the first 16 channels of E1's 32
    conv('U3a', 'u3', 'u3a', h1, w1, act=1); conv('U3b', 'u3a', 'u3b', h1, w1, act=1); conv('U3c', 'u3b', 'u3c', h1, w1, act=1)
    conv('head', 'u3c', 'head', h1, w1, L=UBF_HEAD, d=head, act=0)
    if out_name == 'ubf_out':   # standalone plan: declare the output (callers that pass their own buffer declare it)
        p.buffer(out_name, h1 * 2 * w1 * 2 * 11 * 2)
    p.glsl('nn_d2s', sh + '/nn_d2s.comp'); p.run('nn_d2s', 2 * w1, 2 * h1, [pre + 'head', out_name], [w1, h1, 1 if sigmoid else 0])
    return h1 * 2, w1 * 2


def dbf_plan(dat, out, H, W):
    p = Plan(out)
    p.buffer('dbf_in', H * W * 13 * 2)
    dbf_net(p, dat, H, W)
    p.save()


def ubf_plan(res, out, H, W, wmma=True):
    p = Plan(out)
    p.wmma = wmma
    dat = open(os.path.join(res, 'emit_ubfnet_v3_2_constants.dat'), 'rb').read()
    p.buffer('ubf_in', H * W * 5 * 2)
    ho, wo = ubf_net(p, dat, ubf_head_blob(res), H, W, 'ubf_in', 'ubf_out')
    p.save()
    return ho, wo


if __name__ == '__main__':
    mode, res = sys.argv[1], sys.argv[2]
    if mode == 'ubf':
        ubf_plan(res, sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), os.environ.get('M4R_NO_WMMA') is None)
        sys.exit(0)
    if mode == 'ubf-ref':
        H, W = int(sys.argv[3]), int(sys.argv[4])
        x = np.fromfile(sys.argv[5], '<f2').reshape(H, W, 5)
        ubf_ref(open(os.path.join(res, 'emit_ubfnet_v3_2_constants.dat'), 'rb').read(), ubf_head_blob(res), x).astype('<f2').tofile(sys.argv[6])
        sys.exit(0)
    dat = open(os.path.join(res, 'emit_dbfnet_v3_2_constants.dat'), 'rb').read()
    if mode == 'dbf':
        dbf_plan(dat, sys.argv[3], int(sys.argv[4]), int(sys.argv[5]))
    elif mode == 'dbf-ref':
        H, W = int(sys.argv[3]), int(sys.argv[4])
        x = np.fromfile(sys.argv[5], '<f2').reshape(H, W, 13)
        dbf_ref(dat, x).astype('<f2').tofile(sys.argv[6])
