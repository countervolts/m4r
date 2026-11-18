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

    def kernel(fn, consts, tg):
        key = m4r_air.specialization_key(fn, {k: str(v) for k, v in consts.items()}, tg, hashlib.sha256(m.mtl).hexdigest()[:16])
        name = f'{fn}-{key}'
        if name not in kernels:
            base = os.path.join(out, 'k', name)
            if not os.path.exists(base + '.hsaco') or args.force:
                if fn not in src_cache:
                    f = next(f for f in m.lib['functions'] if f.name == fn)
                    src_cache[fn] = m4r_air.to_text(metallib.bitcode(m.mtl, m.lib, f))
                ir, man = m4r_air.translate(src_cache[fn], fn, consts, tg)
                m4r_air.compile_ir(ir, base + '.hsaco')
                m4r_air.write_manifest(man, base)
            kernels[name] = fn
            plan.append(f'kernel {name} k/{name}')
        return name

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

    json.dump({'output': [ow, oh], 'temporal': [tw, th], 'spatial': [sw, sh]}, open(os.path.join(out, 'sizes.json'), 'w'))
    open(os.path.join(out, 'plan.txt'), 'w').write('\n'.join(plan) + '\n')
    print(f'{len(kernels)} kernels, plan with {sum(1 for p in plan if p.startswith("pass"))} passes -> {out}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--resources', required=True)
    ap.add_argument('--binary', required=True)
    ap.add_argument('--output', required=True, type=lambda s: [int(x) for x in s.split('x')])
    ap.add_argument('--out', required=True)
    ap.add_argument('--force', action='store_true')
    build(ap.parse_args())
