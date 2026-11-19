#!/usr/bin/env python3
"""Prepare MetalFX's ML temporal scaler (BBR) for one output size: weights, kernels and the execution plan.

Everything Apple-specific is read from the user's own MetalFX copy:
  --resources  MetalFX.framework/Resources (default.metallib, emit_tbbrnet_nhwc_q_constants.dat, ...)
  --binary     the MetalFX Mach-O extracted from the dyld shared cache (per-layer parameters)
The weight preparation reproduces MetalFX's host code (see docs/metalfx-bbr.md):
  tBBRNet_GenerateConv3x3*_Step*, mfx::weights::{extractPerOutputChannel(Slice), adjustBiasForInputActZp,
  dequantizeUint8WeightsToFP16, toBlockMajor}, -[Conv3x3Stride1/2 createBuffers], sBBRNet_MLP_GenerateDescriptor.

Output directory: blobs/*.bin (ready-to-upload buffers), k/*.hsaco + *.args (translated kernels) and plan.txt
(executed by runtime/plan.cpp).
"""
import argparse, hashlib, json, os, struct, subprocess, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'air'))
import bbr_params  # noqa: E402
import metallib  # noqa: E402
import m4r_air  # noqa: E402
import fuse as fuse_mod  # noqa: E402
from macho import MachO  # noqa: E402

TBBR_SIZE = 288576


def f32(x):
    return np.float32(x)


def fold_bias(bias16, W, zp, scales16, K, C, kh, kw, act_scale, act_zp):
    """mfx::weights::adjustBiasForInputActZp, in f32 with the same fused multiply-adds."""
    n = kh * kw * C
    out = np.empty(K, np.float16)
    Wm = W.reshape(n, K).astype(np.uint32)
    sums = Wm.sum(axis=0)
    for k in range(K):
        s2 = float(np.float32(sums[k]))                      # ucvtf (exact)
        s2 = float(np.float32(s2 - float(zp[k]) * float(n)))  # fmsub: one rounding (the product is exact)
        s3 = float(np.float32(bias16[k]))
        s4 = np.float32(np.float32(act_scale) * np.float32(scales16[k]))
        s4 = np.float32(s4 * np.float32(-act_zp))
        r = np.float32(float(s4) * s2 + s3)                  # fmadd: exact in double, rounded once
        out[k] = np.float16(r)
    return out


def dequant_fp16(W, zp, scales16, K):
    """mfx::weights::dequantizeUint8WeightsToFP16: half(f32(W - zp[k]) * scale[k]) with k = index % K."""
    Wf = W.reshape(-1, K).astype(np.float32)
    d = (Wf - zp.astype(np.float32)[None, :]) * scales16.astype(np.float32)[None, :]
    return d.astype(np.float32).astype(np.float16).reshape(-1)


def block_major(W, a, b, kh, kw):
    """mfx::weights::toBlockMajor<T>(W, a, b, kh, kw): per tap, the [a][b] matrix as 16x16 tiles (a/16, b/16)."""
    taps = kh * kw
    W = W.reshape(taps, a // 16, 16, b // 16, 16)
    return np.ascontiguousarray(W.transpose(0, 1, 3, 2, 4)).reshape(-1)


class Model:
    def __init__(self, resources, binary):
        self.res = resources
        self.bin = binary
        self.layers = bbr_params.layer_table(binary)
        self.tbbr = np.fromfile(os.path.join(resources, 'emit_tbbrnet_nhwc_q_constants.dat'), np.uint8)
        self.sbbr = np.fromfile(os.path.join(resources, 'emit_sbbrnet_nhwc_q_constants.dat'), np.uint8)
        if len(self.tbbr) < TBBR_SIZE:
            raise SystemExit(f'tBBRNet weight file too small: {len(self.tbbr)} (expected >= {TBBR_SIZE})')
        self.mtl = open(os.path.join(resources, 'default.metallib'), 'rb').read()
        self.lib = metallib.parse(self.mtl)
        self.macho = MachO(binary)
        self.syms = {}
        for l in subprocess.run(['llvm-nm', binary], capture_output=True, text=True).stdout.split('\n'):
            p = l.split()
            if len(p) == 3:
                self.syms[p[2]] = int(p[0], 16)

    def table(self, name, fmt, n):
        a = self.syms['__ZL' + name]
        return struct.unpack('<' + fmt * n, self.macho.read(a, struct.calcsize(fmt) * n))

    def sacneh(self):
        """the two 19-tap filters of emit_sacneh_constants.dat (graph constants tensor7 and tensor10, channel 0)"""
        d = np.fromfile(os.path.join(self.res, 'emit_sacneh_constants.dat'), np.uint8)
        x = d[0x30:0x30 + 38].view(np.float16)
        y = d[0x30 + 0x4c:0x30 + 0x4c + 38].view(np.float16)
        return x.copy(), y.copy()

    def u8(self, off, n):
        return self.tbbr[off:off + n].copy()

    def f16(self, off, n):
        return self.tbbr[off:off + 2 * n].view(np.float16).copy()

    def tbbr_layer(self, l):
        """Upload-ready buffers for one tBBRNet layer."""
        d, w = l['desc'], l['weights']
        K, C = d['K'], d['C']
        if 'slice' in w:
            s = w['slice']
            full = self.u8(s['w_off'], 9 * C * s['K_total']).reshape(9 * C, s['K_total'])
            W = np.ascontiguousarray(full[:, s['k_first']:s['k_first'] + s['k_count']]).reshape(-1)
            scales = self.f16(s['scale_off'], s['K_total'])[s['k_first']:s['k_first'] + s['k_count']]
        else:
            W = self.u8(w['weights'][0], w['weights'][1] - w['weights'][0])
            scales = self.f16(w['scales'][0], K)
        sub = w['subdata']                     # [zero points, bias] (bias first for layer 1, see below)
        if l['step'] == 1:
            bias, zp = self.f16(sub[0][0], K), self.u8(sub[1][0], K)
        else:
            zp, bias = self.u8(sub[0][0], K), self.f16(sub[1][0], K)
        if 'adjust_bias' in w:
            a = w['adjust_bias']
            bias = fold_bias(bias, W, zp, scales, K, C, 3, 3, a['act_scale'], a['act_zp'])
        return W, scales, zp, bias


def build(args):
    m = Model(args.resources, args.binary)
    ow, oh = args.output
    out = args.out
    os.makedirs(os.path.join(out, 'blobs'), exist_ok=True)
    os.makedirs(os.path.join(out, 'k'), exist_ok=True)
    plan = []
    kernels = {}
    src_cache = {}

    def blob(name, arr):
        p = os.path.join(out, 'blobs', name + '.bin')
        np.ascontiguousarray(arr).tofile(p)
        plan.append(f'buffer {name} {arr.nbytes} file blobs/{name}.bin')
        return name

    def buffer(name, size):
        plan.append(f'buffer {name} {max(16, int(size))}')
        return name

    def source(fn):
        if fn not in src_cache:
            f = next(f for f in m.lib['functions'] if f.name == fn)
            src_cache[fn] = m4r_air.to_text(metallib.bitcode(m.mtl, m.lib, f))
        return src_cache[fn]

    def kernel(fn, consts, tg, fuse=None):
        """fuse: producers merged into this kernel (air/fuse.py), [{kernel, out_loc, read_loc, grid, base, bind_map}]"""
        src = hashlib.sha256(m.mtl).hexdigest()[:16]
        if fuse:
            src = hashlib.sha256((src + json.dumps(fuse, sort_keys=True)).encode()).hexdigest()[:16]
        key = m4r_air.specialization_key(fn, {k: str(v) for k, v in consts.items()}, tg, src)
        name = f'{fn}-{key}'
        if name not in kernels:
            base = os.path.join(out, 'k', name)
            if not os.path.exists(base + '.hsaco') or args.force:
                text = source(fn)
                if fuse:
                    specs = [dict(p, text=source(p['kernel']), bind_map={(a, b): c for a, b, c in p['bind_map']}) for p in fuse]
                    ir, man = m4r_air.translate(fuse_mod.fuse(text, fn, specs), fn, consts, tg)
                    try:
                        m4r_air.compile_ir(ir, base + '.hsaco')
                        called = b's_swappc' in subprocess.run(['llvm-objdump', '-d', f'--mcpu={m4r_air.GPU}', base + '.hsaco'],
                                                               capture_output=True, check=True).stdout
                    except RuntimeError:
                        called = True
                    if called:   # the fused body was not inlined: force it (see air/fuse.py)
                        ir, man = m4r_air.translate(fuse_mod.fuse(text, fn, specs, force_inline=True), fn, consts, tg)
                        m4r_air.compile_ir(ir, base + '.hsaco')
                else:
                    ir, man = m4r_air.translate(text, fn, consts, tg)
                    m4r_air.compile_ir(ir, base + '.hsaco')
                m4r_air.write_manifest(man, base)
            kernels[name] = fn
            plan.append(f'kernel {name} k/{name}')
        return name
    kernel.source = source

    def dispatch(k, mode, grid, binds):
        plan.append(f'pass {k} {mode} {grid[0]} {grid[1]} {grid[2]}')
        for b in binds:
            plan.append('  ' + b)

    # ---- tensor sizes (BBRNet_Filter::init)
    sw, sh = ((ow + 7) >> 1) & ~3, ((oh + 7) >> 1) & ~3
    tw, th = ((ow + 15) >> 2) & ~3, ((oh + 15) >> 2) & ~3
    fused = 1
    buffer('spatial_in', sw * sh * 4)
    buffer('spatial_out', sw * sh * 8)
    buffer('temporal_in', tw * th * 64)
    buffer('temporal_in_u8', tw * th * 16)
    buffer('temporal_in_f16', tw * th * 32)
    buffer('temporal_out', tw * th * 4 << (1 if fused else 3))
    plan.append(f'texture temporal_tex R8_UNORM {ow} {oh}')
    plan.append(f'texture rnn_tex R8G8_UNORM {ow // 2} {oh // 2}')
    plan.append(f'# network {tw}x{th} (temporal), spatial {sw}x{sh}, output {ow}x{oh}')

    plan.append('#PRE')
    # ---- tBBRNet
    H, W = th, tw
    acts = {}
    for l in m.layers:
        st, d = l['step'], l['desc']
        h = H >> d['H']['shift'] if isinstance(d['H'], dict) else d['H']
        w = W >> d['W']['shift'] if isinstance(d['W'], dict) else d['W']
        K, C = d['K'], d['C']
        Wq, scales, zp, bias = m.tbbr_layer(l)
        pre = f't{st}'
        acts[st] = buffer(f'{pre}_out', h * w * K) if st < 11 else 'temporal_out'
        if l['kind'] == 'S2':
            consts = {'H': h, 'W': w, 'K': K, 'C': C, 'QsI': d['QsI'], 'QsRinv': d['QsRinv'], 'inputBlockK': 0,
                      'useLeakyReLU': d['useLeakyReLU'], 'outZp': d['outZp']}
            k = kernel('Conv3x3Stride2U8', consts, [128, 1, 1])
            blob(f'{pre}_w', Wq); blob(f'{pre}_s', scales); blob(f'{pre}_b', bias); blob(f'{pre}_z', zp)
            tiles = (h * w + 31) // 32
            threads = (((K + 31) & ~31) * tiles + 127) & ~127
            dispatch(k, 'threads', (threads, 1, 1), [f'buf 0 {acts[st - 1]}', f'buf 1 {pre}_w', f'buf 2 {acts[st]}',
                                                     f'buf 3 {pre}_s', f'buf 4 {pre}_b', f'buf 5 {pre}_z'])
            continue
        tgx = 32 * d['simdgroupsPerTG']
        consts = {n: d[n] for n in ('K', 'C', 'QsI', 'QsRinv', 'DqsBilinearImage', 'DqsSkip', 'CSkip', 'NmatsHW', 'NmatsK',
                                    'outputBlockK', 'inputIsQuantized', 'outputToTexture', 'useSigmoid', 'useD2S',
                                    'useLeakyReLU', 'outZp', 'combinedInZp', 'bilinearImageZp', 'skipZpAct',
                                    'blockMajorWeights', 'inputIsSigned')}
        consts.update(H=h, W=w, headTemporalTex=1 if st == 11 and fused else 0, headRnnTex=1 if st == 11 and fused else 0)
        tiles = ((h * w + 16 * d['NmatsHW'] - 1) // (16 * d['NmatsHW'])) * ((K + 16 * d['NmatsK'] - 1) // (16 * d['NmatsK']))
        threads = (tiles * 32 + tgx - 1) // tgx * tgx
        if st == 1:
            # Enc0FQSplit: channels 0-15 fp16 (dequantised weights), 16-31 u8
            deq = dequant_fp16(Wq, zp, scales, K).reshape(9, C, K)
            first16 = block_major(np.ascontiguousarray(deq[:, :16, :]).reshape(-1), 16, K, 3, 3)
            last16 = block_major(np.ascontiguousarray(Wq.reshape(9, C, K)[:, 16:, :]).reshape(-1), 16, K, 3, 3)
            sc = {k: v for k, v in consts.items() if k in ('H', 'W', 'K', 'QsRinv', 'NmatsHW', 'NmatsK', 'outputBlockK',
                                                          'outputToTexture', 'useSigmoid', 'useD2S', 'useLeakyReLU',
                                                          'outZp', 'blockMajorWeights', 'headTemporalTex', 'headRnnTex')}
            k = kernel('Conv3x3Stride1Enc0FQSplit', sc, [tgx, 1, 1])
            blob(f'{pre}_w16', first16); blob(f'{pre}_w8', last16)
            blob(f'{pre}_s', scales); blob(f'{pre}_b', bias); blob(f'{pre}_z', zp)
            dispatch(k, 'threads', (threads, 1, 1), ['buf 0 temporal_in_u8', f'buf 1 {pre}_w8', f'buf 2 {acts[st]}',
                                                     f'buf 3 {pre}_s', f'buf 4 {pre}_b', f'buf 5 {pre}_z',
                                                     'buf 6 temporal_in_f16', f'buf 7 {pre}_w16'])
            continue
        Wb = block_major(Wq, C, K, 3, 3) if d['blockMajorWeights'] else Wq
        blob(f'{pre}_w', Wb); blob(f'{pre}_s', scales); blob(f'{pre}_b', bias); blob(f'{pre}_z', zp)
        feed = acts[st - 1]
        if d['bilinearSkip']:
            # BilinearUpsample2xSkipU8(feed at h/2 x w/2, skip) -> scratch, then the conv without bilinear/skip
            skip = {6: 3, 8: 2, 10: 1}[st]
            bc = {'H': h, 'W': w, 'C': C, 'QsI': d['QsI'], 'DqsBilinearImage': d['DqsBilinearImage'], 'useSkip': 1,
                  'DqsSkip': d['DqsSkip'], 'CSkip': d['CSkip'], 'combinedInZp': d['combinedInZp'],
                  'bilinearImageZp': d['bilinearImageZp'], 'skipZpAct': d['skipZpAct']}
            kb = kernel('BilinearUpsample2xSkipU8', bc, [256, 1, 1])
            scratch = buffer(f'{pre}_combined', h * w * C)
            dispatch(kb, 'threads', ((h * w + 255) & ~255, 1, 1), [f'buf 0 {feed}', f'buf 1 {acts[skip]}', f'buf 2 {scratch}'])
            feed = scratch
            consts.update(useBilinear=0, useSkip=0)
        else:
            consts.update(useBilinear=0, useSkip=0)
        k = kernel('Conv3x3Stride1U8', consts, [tgx, 1, 1])
        binds = [f'buf 0 {feed}', f'buf 1 {pre}_w', f'buf 2 {acts[st]}', f'buf 3 {pre}_s', f'buf 4 {pre}_b', f'buf 5 {pre}_z']
        if st == 11 and fused:
            binds += ['tex 3 temporal_tex', 'tex 4 rnn_tex']
        dispatch(k, 'threads', (threads, 1, 1), binds)

    frame_passes(m, args, plan, blob, buffer, kernel, dispatch, sizes=(ow, oh, sw, sh, tw, th, fused))
    json.dump({'output': [ow, oh], 'input': list(args.input), 'temporal': [tw, th], 'spatial': [sw, sh],
               'history_pitch': 32 * ((2 * ow + 63) // 64), 'reversed_depth': args.reversed_depth},
              open(os.path.join(out, 'sizes.json'), 'w'))
    shader = os.path.join(os.path.dirname(HERE), 'shaders', 'sacneh.comp')
    subprocess.run(['glslangValidator', '-V', '--target-env', 'vulkan1.3', shader, '-o', os.path.join(out, 'sacneh.spv')],
                   check=True, capture_output=True)
    transpose_weight_tiles(out, plan)
    decl = [l for l in plan if l.split(' ', 1)[0] in ('buffer', 'texture', 'kernel', 'glsl')]
    rest = [l for l in plan if l.split(' ', 1)[0] not in ('buffer', 'texture', 'kernel', 'glsl')]
    open(os.path.join(out, 'plan.txt'), 'w').write('\n'.join([f'gpu {m4r_air.GPU}'] + decl + rest) + '\n')
    print(f'{len(kernels)} kernels, plan with {sum(1 for p in plan if p.startswith("pass"))} passes -> {out}')


def _bind_key(b):
    w = b.split()
    return ('t' if w[0] == 'tex' else 'b', int(w[1]))


def _state(lines):
    """Bindings in force after `lines` (Metal keeps a binding until it is replaced; runtime/plan.cpp does the same)."""
    st = {}
    for ln in lines:
        s = ln.strip()
        if s.split(' ', 1)[0] in ('buf', 'tex', 'bytes'):
            st[_bind_key(s)] = s
    return st


def fused_pass(kernel, lines, fn, binds, grid, producers, consts, tg):
    """One pass of `fn` with the producers merged in (air/fuse.py). Bindings are listed in full: the effective bindings of
    each pass at its place in the unfused plan (its own, then those inherited from earlier passes)."""
    st = _state(lines)
    effs = []
    for _, pb, *_ in producers:
        st = dict(st); st.update({_bind_key(b): b for b in pb}); effs.append(st)
    sa = dict(st); sa.update({_bind_key(b): b for b in binds})
    cargs = m4r_air.kernel_info(kernel.source(fn), fn)

    def res(b):
        w = b.split(' ', 2)
        return (w[0], w[2])
    out = []
    for a in cargs:
        k = {'air.texture': 't', 'air.buffer': 'b'}.get(a['kind'])
        if k and a.get('location') is not None and (k, a['location']) in sa:
            out.append(sa[(k, a['location'])])
    specs = []
    for pi, ((pfn, _, out_loc, read_loc, pgrid), pst) in enumerate(zip(producers, effs)):
        base = 100 * (pi + 1)
        bm = []
        for a in m4r_air.kernel_info(kernel.source(pfn), pfn):
            k = {'air.texture': 't', 'air.buffer': 'b'}.get(a['kind'])
            if not k or (k == 't' and a.get('location') == out_loc):
                continue
            b = pst.get((k, a['location']))
            if b is None:
                raise ValueError(f'{pfn}: nothing bound at {k} {a["location"]}')
            same = [ca for ca in cargs if ca['kind'] == a['kind'] and ca.get('location') is not None
                    and (k, ca['location']) in sa and res(sa[(k, ca['location'])]) == res(b)
                    and ca.get('address_space') == a.get('address_space') and ca.get('access', 'read') in ('read', 'sample')]
            if same:
                bm.append([a['kind'][4:], a['location'], same[0]['location']])
            else:
                w = b.split(' ', 2)
                out.append(f'{w[0]} {base + a["location"]} {w[2]}')
        specs.append({'kernel': pfn, 'out_loc': out_loc, 'read_loc': read_loc, 'grid': list(pgrid), 'base': base, 'bind_map': bm})
    k = kernel(fn, consts, list(tg), fuse=specs)
    return f'pass {k} threads {grid[0]} {grid[1]} {grid[2]}\n' + '\n'.join('  ' + b for b in out)


def frame_passes(m, args, plan, blob, buffer, kernel, dispatch, sizes):
    """Everything around the networks: BBRNet_Filter::encodePre / sBBRNet / encodeMid / encodePost and the
    sacneh history prefilter (see docs/metalfx-bbr.md for where each binding comes from)."""
    ow, oh, sw, sh, tw, th, fused = sizes
    iw, ih = args.input
    rev = 1 if args.reversed_depth else 0
    lines = bool(getattr(args, 'line_locking', False))   # MetalFX's line locking (off unless an app compat flag enables it)
    fc = {'#13': 1, '#30': 1, '#31': 1, '#32': 1, '#33': 1, '#34': fused, '#14': 0, '#17': 1 if lines else 0, '#16': 0, '#19': 0,
          '#25': 0, '#26': 0, '#22': 0, '#9': 0.0328}
    pre = []

    def tex(name, fmt, w, h):
        plan.insert(0, f'texture {name} {fmt} {w} {h}')

    def buf_init(name, data):
        blob(name, data)
        plan.insert(0, plan.pop())        # declare before use

    tex('crr', 'R16G16B16A16_SFLOAT', iw, ih); tex('flowd', 'R16G16_SFLOAT', iw, ih)
    tex('fah', 'R16G16B16A16_SFLOAT', ow, oh); tex('clampt', 'R16G16B16A16_SFLOAT', ow, oh)
    tex('hist0', 'R16G16B16A16_SFLOAT', ow, oh); tex('hist1', 'R16G16B16A16_SFLOAT', ow, oh)
    tex('ema0', 'R16_SFLOAT', ow // 2, oh // 2); tex('ema1', 'R16_SFLOAT', ow // 2, oh // 2)
    tex('hw0', 'R8G8_UNORM', ow, oh); tex('hw1', 'R8G8_UNORM', ow, oh)
    tex('splat', 'R16G16_SFLOAT', iw, ih); tex('diff', 'R8_UNORM', iw, ih)
    one = np.array([1.0], np.float16)
    buf_init('exp0', one); buf_init('exp1', one)
    for n, sz in (('splatbuf', iw * ih * 4), ('hinv', 16), ('lumabuf', 8192), ('params', 144), ('preexp', 16),
                  ('reset', 16), ('cpf', 16), ('npix', 16)):
        plan.insert(0, f'buffer {n} {sz}')
    hp = 32 * ((2 * ow + 63) // 64)
    for n in ('hist_t0', 'hist_t1', 'hist_tmp'):
        plan.insert(0, f'buffer {n} {oh * hp * 4}')

    def d(fn, mode, grid, binds, extra=None, tg=(8, 8, 1)):
        c = dict(fc); c.update(extra or {})
        k = kernel(fn, c, list(tg))
        pre.append(f'pass {k} {mode} {grid[0]} {grid[1]} {grid[2]}')
        pre.extend('  ' + b for b in binds)

    # encodePre: exposure. Without an exposure texture (DLSS AutoExposure) MetalFX's auto exposure: luma_log_sum (32 x 64 groups
    # of 8 x 4, one log-luma partial sum per group into lumabuf) and luma_log_sum_to_exposure (one group of 1024: average,
    # key value, EMA with EMA_factor, snap and history reset on a 10x change); with one, exposure_texture_to_buffer. The runtime
    # picks the path per frame (flag autoexp); the auto passes come first so that what later passes inherit is unchanged.
    d('luma_log_sum', 'threads', (256, 256, 1),
      ['only autoexp', 'tex 0 color', 'buf 0 lumabuf', 'buf 1 params', 'buf 2 preexp'], tg=(8, 4, 1))
    d('luma_log_sum_to_exposure', 'threads', (1024, 1, 1),
      ['only autoexp', 'buf 0 lumabuf', 'buf 1 npix', 'buf 2 exp0|exp1', 'buf 3 exp1|exp0', 'buf 4 hinv', 'buf 5 reset'],
      tg=(1024, 1, 1))
    d('exposure_texture_to_buffer', 'threads', (1, 1, 1),
      ['only !autoexp', 'tex 0 exposure', 'buf 0 exp0|exp1', 'buf 1 exp1|exp0', 'buf 2 hinv', 'buf 3 reset'], tg=(32, 1, 1))
    d('bbrnet_build_tensors_low_resolution', 'threads', (iw + 2, ih + 2, 1),
      ['tex 0 color', 'tex 1 depth', 'tex 2 motion', 'tex 3 crr', 'tex 4 flowd',
       'bytes 0 ' + struct.pack('<f', 1.0 if rev else -1.0).hex(), 'buf 1 params', 'buf 2 preexp',
       'buf 3 exp0|exp1', 'buf 4 exp1|exp0'])
    d('buildSpatialTensor', 'threads', (ow, oh, 1), ['buf 0 spatial_in', 'tex 0 crr', 'buf 1 params'])
    d('buildTemporalTensor', 'threads', (((ow + 15) >> 1) & ~7, ((oh + 15) >> 1) & ~7, 1),
      ['tex 0 crr', 'tex 1 flowd', 'tex 2 hist0|hist1', 'tex 4 ema0|ema1', 'tex 6 ema1|ema0', 'tex 8 temporal_tex',
       'tex 9 rnn_tex', 'buf 0 temporal_in', 'buf 1 params', 'buf 2 temporal_out', 'buf 3 exp0|exp1',
       'buf 4 exp1|exp0', 'buf 5 hinv', 'buf 6 temporal_in_u8', 'buf 7 temporal_in_f16'], tg=(16, 16, 1))
    i = plan.index('#PRE')
    plan[i:i + 1] = pre
    pre.clear()

    # sBBRNet (5-layer per-pixel MLP, fp16 intermediates)
    LK = m.table('7kLayerK', 'H', 5)
    WO, BO, SO, ZO = (m.table(n, 'Q', 5) for n in ('14kWeightOffsets', '12kBiasOffsets', '13kScaleOffsets', '10kZPOffsets'))
    AZ, AS = m.table('6kActZP', 'B', 5), m.table('9kActScale', 'f', 5)
    in_scale = sbbr_input_scale(m)
    s8 = m.sbbr
    w8, sc, bi, zp, w16, b16 = [], [], [], [], [], []
    for li in range(5):
        K = LK[li]
        W = s8[WO[li]:WO[li] + 64 * K].copy()
        scales = s8[SO[li]:SO[li] + 2 * K].view(np.float16).copy()
        z = s8[ZO[li]:ZO[li] + K].copy()
        b = s8[BO[li]:BO[li] + 2 * K].view(np.float16).copy()
        a_s, a_z = (in_scale, 0.0) if li == 0 else (AS[li - 1], float(AZ[li - 1]))
        w8.append(W); sc.append(scales); zp.append(z); bi.append(fold_bias(b, W, z, scales, K, 64, 1, 1, a_s, a_z))
        if li > 0:
            w16.append(dequant_fp16(W, z, scales, K)); b16.append(b)
    blob('s_w', np.concatenate(w8)); blob('s_s', np.concatenate(sc)); blob('s_b', np.concatenate(bi))
    blob('s_z', np.concatenate(zp)); blob('s_w16', np.concatenate(w16)); blob('s_b16', np.concatenate(b16))
    Hs, Ws = sh >> 2, sw >> 2
    k = kernel('SBBRNet', {'Hs': Hs, 'Ws': Ws, 'QsI': in_scale, 'NmatsHW': 1, 'inputIsFP16': 0, 'inputIsNCHW': 0,
                           'outputIsFP16': 0, 'outputIsNCHW': 0, 'fp16Intermediate': 1}, [128, 1, 1])
    tiles = (Hs * Ws + 15) // 16
    dispatch(k, 'threads', (((tiles + 3) // 4) * 128, 1, 1), ['buf 0 spatial_in', 'buf 1 s_w', 'buf 2 spatial_out',
                                                              'buf 3 s_s', 'buf 4 s_b', 'buf 5 s_z', 'buf 6 s_w16', 'buf 7 s_b16'])

    # encodeMid
    if not rev:
        d('brnetv3_flow_splat_init_forward_z', 'threads', (iw * ih, 1, 1), ['buf 0 splatbuf'], tg=(64, 1, 1))
    d('BBRNet_flow_splat', 'threads', (iw, ih, 1), ['tex 0 flowd', 'tex 1 depth', 'buf 0 splatbuf', 'buf 1 params'], {'#5': rev})
    d('brnetv3_flow_splat_texture', 'threads', (iw, ih, 1), ['tex 0 flowd', 'tex 1 splat', 'buf 0 splatbuf'])
    d('BBRNet_flow_diff_texture', 'threads', (iw, ih, 1), ['tex 0 flowd', 'tex 1 splat', 'tex 2 diff'])
    warp = ['tex 0 crr', 'tex 2 flowd', 'tex 3 fah', 'tex 6 hist0|hist1', 'tex 7 color', 'buf 2 exp0|exp1', 'buf 3 exp1|exp0',
            'buf 4 hinv', 'buf 5 preexp', 'buf 6 hist_t0']
    clamp = ['tex 2 flowd', 'tex 6 hist0|hist1', 'tex 7 color', 'tex 8 clampt']
    # encodePost
    post = ['buf 0 params', 'tex 0 crr', 'tex 1 fah', 'tex 6 hw0|hw1', 'tex 3 output', 'tex 4 hist1|hist0', 'tex 8 hw1|hw0',
            'tex 5 diff', 'tex 7 flowd', 'tex 10 clampt', 'tex 11 temporal_tex', 'buf 1 cpf', 'buf 2 hinv',
            'buf 3 preexp', 'buf 4 exp0|exp1', 'buf 6 spatial_out', 'buf 10 hist_t1']
    if lines:
        tex('lock0', 'R8_UNORM', ow, oh); tex('lock1', 'R8_UNORM', ow, oh)
        post += ['tex 13 lock1|lock0', 'tex 14 lock0|lock1']
    if not fused:
        post += ['tex 9 rnn_tex', 'buf 5 temporal_out']
    ag_grid = ((ow + 15) & ~15, (oh + 15) & ~15, 1)
    if os.environ.get('M4R_FUSE', '1') == '0':
        d('BBRNetWarpHistory', 'threads', (ow, (oh + 3) // 4, 1), warp, tg=(16, 8, 1))
        d('BBRNetComputeOvershootClamp', 'threads', (ow, oh, 1), clamp, tg=(16, 16, 1))
        d('BBRNetAnisoGaussian', 'threads', ag_grid, post, tg=(16, 16, 1))
    else:
        # WarpHistory and OvershootClamp computed inside AnisoGaussian at the pixel it reads (air/fuse.py): flowAdjHistory and
        # overshootClamp are never stored; the history they read is the previous frame's (hist0|hist1), so the fused kernel's
        # history write cannot be seen by them
        producers = [('BBRNetWarpHistory', warp, 3, 1, (ow, (oh + 3) // 4)), ('BBRNetComputeOvershootClamp', clamp, 8, 10, (ow, oh))]
        pre.append(fused_pass(kernel, plan + pre, 'BBRNetAnisoGaussian', post, ag_grid, producers, dict(fc), (16, 16, 1)))
    pre.append('glsl sacneh sacneh.spv 32 8 1')
    taps = m.sacneh()
    blob('sac_x', taps[0]); blob('sac_y', taps[1])
    # x and y filters in one dispatch (shaders/sacneh.comp), hist_t1 -> hist_t0
    pre.append(f'pass sacneh threads {ow} {(oh + 3) // 4} 2')   # 32 x 32 outputs per 32 x 8 group
    pre += ['  buf 0 hist_t1', '  buf 1 hist_t0', '  buf 2 sac_x', '  buf 3 sac_y'] + [
        f'  bytes {8 + j} ' + struct.pack('<I', v).hex() for j, v in enumerate((ow, oh, hp, oh * hp, 2))]
    plan.extend(pre)


def transpose_weight_tiles(out, plan):
    """Weight blobs that a kernel reads straight into WMMA layout (manifest transpose_b8, air/gemm_chain.py find_direct_b) are
    stored with each 128-byte half-tile transposed ([8 k][16 col] -> [16 col][8 k]). A blob bound anywhere else is refused."""
    passes, cur = [], None
    for line in plan:
        if line.startswith('pass '):
            cur = [line.split()[1], {}]
            passes.append(cur)
        elif cur and line.startswith('  buf '):
            _, loc, name = line.split()[:3]
            cur[1][int(loc)] = name
    want, other = set(), set()
    for kname, binds in passes:
        mf = os.path.join(out, 'k', kname + '.json')
        locs = set(json.load(open(mf)).get('transpose_b8', [])) if os.path.exists(mf) else set()
        for loc, name in binds.items():
            (want if loc in locs else other).add(name)
    for name in sorted(want):
        if name in other or '|' in name:
            raise SystemExit(f'weight blob {name} is read both as WMMA tiles and in another way')
        path = os.path.join(out, 'blobs', name + '.bin')
        a = np.fromfile(path, np.uint8)
        if a.size % 128:
            raise SystemExit(f'weight blob {name}: {a.size} bytes is not a whole number of 8 x 16 half-tiles')
        a.reshape(-1, 8, 16).transpose(0, 2, 1).copy().tofile(path)


def sbbr_input_scale(m):
    """QsI of SBBRNet: the float immediate -[SBBRNet_MLP setupMetalWithLibrary:] stores (mov/movk pair)."""
    funcs = bbr_params.objdump(m.bin)
    ins = next(v for k, v in funcs.items() if 'SBBRNet_MLP setupMetalWithLibrary' in k)
    val = None
    for addr, op, a in ins:
        if op == 'mov' and a.startswith('w8, #'):
            val = int(a.split('#')[1].split()[0], 0)
        elif op == 'movk' and a.startswith('w8,') and val is not None:
            hi = int(a.split('#')[1].split(',')[0], 0)
            val = (val & 0xffff) | (hi << 16)
            return struct.unpack('<f', struct.pack('<I', val))[0]
    raise SystemExit('SBBRNet input scale not found')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--resources', required=True)
    ap.add_argument('--binary', required=True)
    ap.add_argument('--output', required=True, type=lambda s: [int(x) for x in s.split('x')])
    ap.add_argument('--out', required=True)
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--input', required=True, type=lambda s: [int(x) for x in s.split('x')], help='max input size')
    ap.add_argument('--gpu', default=m4r_air.GPU, help='target GPU (' + ', '.join(m4r_air.GPUS) + ')')
    ap.add_argument('--reversed-depth', type=int, default=1)
    ap.add_argument('--line-locking', action='store_true',
                    help="MetalFX's line locking (thin-line stability; MetalFX leaves it off unless an app compatibility flag "
                         'sets it): two lock-counter textures, about 0.06 ms at 2560x1440')
    args = ap.parse_args()
    m4r_air.set_gpu(args.gpu)
    build(args)
