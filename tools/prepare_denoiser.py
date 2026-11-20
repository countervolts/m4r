#!/usr/bin/env python3
"""Plan generator for MetalFX's temporal denoiser (DBFNet + SVGF-style filter), built from the user's own MetalFX files.

usage: prepare_denoiser.py --resources <MetalFX Resources dir> --capture <M4R_DUMP_RR dir> --first <frame id> --frames N --out <plan dir>
                           [--stage B|C|D]

The pass list follows docs/metalfx-denoiser-passes.txt (MetalFX's host code). Kernels are Apple's, translated by air/m4r_air.py;
the DBFNet convolutions are m4r's plain kernels (tools/denoise_nets.py). Work in progress: see docs/metalfx-denoiser.md.
SPDX-License-Identifier: GPL-3.0-only
"""
import argparse, hashlib, json, os, struct, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'tools'))
from air import m4r_air, metallib
import denoise_nets


def h16(x):
    return np.float16(x).tobytes()


class Gen:
    def __init__(self, res, out):
        self.out = out
        os.makedirs(out + '/k', exist_ok=True); os.makedirs(out + '/blobs', exist_ok=True)
        self.mtl = open(res + '/default.metallib', 'rb').read()
        self.lib = metallib.parse(self.mtl)
        self.lines = []; self.kernels = {}; self.src = {}; self.wmma = not os.environ.get('M4R_NO_WMMA')   # WMMA convs (shaders/nn_conv_wmma.cl); M4R_NO_WMMA=1: plain GLSL convs
        self.dat = {n: open(f'{res}/emit_{n}net_v3_2_constants.dat', 'rb').read() for n in ('dbf', 'ubf')}

    def source(self, fn):
        if fn not in self.src:
            f = next(f for f in self.lib['functions'] if f.name == fn)
            self.src[fn] = m4r_air.to_text(metallib.bitcode(self.mtl, self.lib, f))
        return self.src[fn]

    def kernel(self, fn, consts, tg):
        h = hashlib.sha256(self.mtl).hexdigest()[:16]
        key = m4r_air.specialization_key(fn, {k: str(v) for k, v in consts.items()}, list(tg), h)
        name = f'{fn}-{key}'
        if name not in self.kernels:
            base = f'{self.out}/k/{name}'
            if not os.path.exists(base + '.hsaco'):
                ir, man = m4r_air.translate(self.source(fn), fn, consts, list(tg))
                m4r_air.compile_ir(ir, base + '.hsaco')
                m4r_air.write_manifest(man, base)
            self.kernels[name] = fn
            self.lines.append(f'kernel {name} k/{name}')
        return name

    def buffer(self, name, size, init=None):
        if init is not None:
            open(f'{self.out}/blobs/{name}.bin', 'wb').write(init)
            self.lines.append(f'buffer {name} {len(init)} file blobs/{name}.bin')
        else:
            self.lines.append(f'buffer {name} {max(16, int(size))}')

    def texture(self, name, fmt, w, h):
        self.lines.append(f'texture {name} {fmt} {w} {h}')

    def glsl(self, name, src, lx=8, ly=8):
        if ('glsl', name) not in self.kernels:
            import shutil, subprocess
            subprocess.run(['glslangValidator', '-V', '--target-env', 'vulkan1.3', src, '-o', f'{self.out}/{name}.spv'], check=True, capture_output=True)
            self.lines.append(f'glsl {name} {name}.spv {lx} {ly} 1')
            self.kernels[('glsl', name)] = name

    def run(self, kernel, w, h, bufs, consts, z=1):   # GLSL pass (shaders/nn_*.comp): buffers in slots 0.., constants from binding 8
        self.lines.append(f'pass {kernel} threads {w} {h} {z}')
        for i, b in enumerate(bufs):
            self.lines.append(f'  buf {i} {b}')
        for i, c in enumerate(consts):
            self.lines.append(f'  bytes {8 + i} {int(c).to_bytes(4, "little").hex()}')

    def run_k(self, fn, consts, tg, grid, binds, only=None, groups=False):
        k = self.kernel(fn, consts, tg)
        self.lines.append(f'pass {k} {"groups" if groups else "threads"} {grid[0]} {grid[1]} {grid[2]}')
        if only:
            self.lines.append('  only ' + only)
        for b in binds:
            self.lines.append('  ' + b)

    def save(self):
        decl = [l for l in self.lines if l.split(' ', 1)[0] in ('buffer', 'texture', 'kernel', 'glsl')]
        rest = [l for l in self.lines if l.split(' ', 1)[0] not in ('buffer', 'texture', 'kernel', 'glsl')]
        open(self.out + '/plan.txt', 'w').write('\n'.join([f'gpu {m4r_air.GPU}'] + decl + rest) + '\n')


def build(g, iw, ih, stage, runtime=False):
    W, H = (iw + 7) & ~7, (ih + 7) & ~7          # BFNet size: render size rounded up to 8
    tw, th = ((iw + 15) >> 1) & ~7, ((ih + 15) >> 1) & ~7   # DBF tensor (half resolution)
    ROW = 26                                       # NHWC, 13 half channels per pixel
    # persistent state and intermediates (BFNet / BRNet_v3 constructors)
    g.buffer('exp0', 2, h16(1.0)); g.buffer('exp1', 2, h16(1.0)); g.buffer('hinv', 16); g.buffer('reset', 16); g.buffer('preexp', 16)
    g.buffer('dbf_params', 40); g.buffer('refl_params', 224)
    g.buffer('dbf_in', tw * th * ROW)
    g.texture('int_motion', 'R16G16_SFLOAT', iw, ih); g.texture('int_depth', 'R32_SFLOAT', iw, ih)
    # BFNet +0xa8: the previous frame's colour (texPrevInputFrame) and the first a-trous iteration's output (history_tex): one texture
    g.texture('hist', 'R16G16B16A16_SFLOAT', W, H)
    g.texture('mm0', 'R8_UNORM', W, H); g.texture('mm1', 'R8_UNORM', W, H)
    if runtime:
        # the game's normals carry roughness in alpha (pass 0): m4r copies the normals to nrm_raw, this pass extracts alpha into
        # rough_raw, m4r copies that into the roughness texture; specular hit distance is not read (spec_hit_dist_enabled false)
        g.buffer('nrm_raw', iw * ih * 8); g.buffer('rough_raw', iw * ih * 2)
        g.texture('roughness', 'R16_SFLOAT', iw, ih); g.texture('spechit', 'R32_SFLOAT', 1, 1)
        g.glsl('nn_rough', os.path.join(ROOT, 'shaders', 'nn_rough.comp'))
        g.run('nn_rough', iw, ih, ['nrm_raw', 'rough_raw'], [iw, ih])
    # DPreProcessing: BRNet_v3_Filter::encodeExposureCalcForDenoiser. With an exposure texture: exposure_texture_to_buffer; without one
    # (DLSS AutoExposure) MetalFX's auto exposure over the content area: luma_log_sum_denoiser (one group of 1024) and
    # luma_log_sum_to_exposure_denoiser (one group of 1024). The runtime picks the path per frame (flag autoexp).
    g.buffer('dlum', 8192); g.buffer('dexcl', 8192); g.buffer('bparams', 144); g.buffer('npix', 16)
    g.run_k('luma_log_sum_denoiser', {}, (1024, 1, 1), (1, 1, 1),
            ['tex 0 color', 'buf 0 dlum', 'buf 1 bparams', 'buf 2 preexp', 'buf 3 dexcl'], only='autoexp', groups=True)
    g.run_k('luma_log_sum_to_exposure_denoiser', {}, (1024, 1, 1), (1, 1, 1),
            ['buf 0 dlum', 'buf 1 npix', 'buf 2 exp0|exp1', 'buf 3 exp1|exp0', 'buf 4 hinv', 'buf 5 reset', 'buf 6 dexcl'],
            only='autoexp', groups=True)
    g.run_k('exposure_texture_to_buffer', {}, (32, 1, 1), (1, 1, 1),
            ['tex 0 exposure', 'buf 0 exp0|exp1', 'buf 1 exp1|exp0', 'buf 2 hinv', 'buf 3 reset'], only='!autoexp')
    # BFNet_v1_Filter::encodePre: dbfnetv1_lowres_signals
    g.run_k('dbfnetv1_lowres_signals', {'is_tensor_layout_nchw': False, 'spec_hit_dist_enabled': False, 'new_overshoot_clamp': False},
          (8, 8, 1), (tw, th, 1),
          ['tex 0 normals', 'tex 1 motion', 'tex 2 depth', 'tex 3 int_motion', 'tex 4 int_depth', 'tex 5 color', 'tex 6 hist',
           'tex 7 specular', 'tex 8 diffuse', 'tex 9 roughness', 'tex 10 spechit', 'tex 11 mm0|mm1', 'tex 12 mm1|mm0',
           'buf 0 dbf_params', 'buf 5 refl_params', 'buf 1 dbf_in', 'buf 3 exp0|exp1', 'buf 4 preexp'])
    if stage < 'C':
        return dict(W=W, H=H, tw=tw, th=th)
    # DNet: DBFNet (cells E1-E4, D1-D3), NHWC, rowBytes 26 in / 14 out
    denoise_nets.dbf_net(g, g.dat['dbf'], th, tw, 'dbf_in', 'dbf_out')
    if stage < 'D':
        return dict(W=W, H=H, tw=tw, th=th)
    hw, hh = W // 2, H // 2
    # intermediates of BFNet_v1_Filter (member offsets in docs/metalfx-denoiser-passes.txt)
    for n in ('warp_color', 'warp_prev', 'nr0', 'nr1'):
        g.texture(n, 'R16G16B16A16_SFLOAT', W, H)
    g.texture('shin', 'R8_UNORM', W, H); g.texture('mmr', 'R8_UNORM', W, H)
    for n in ('fc0', 'fc1', 'dy0', 'dy1'):
        g.texture(n, 'R16_SFLOAT', W, H)
    for n in ('rm0', 'rm1'):
        g.texture(n, 'R16G16B16A16_SFLOAT', W, H)
    for n in ('bw0', 'bw1'):
        g.texture(n, 'R8_UNORM', W, H)
    g.texture('spec_mvec', 'R16G16_SFLOAT', W, H); g.texture('flowdiff', 'R16_SFLOAT', W, H)
    g.texture('boost', 'R16_SFLOAT', W // 8, H // 8)
    g.texture('dbf_feat', 'R16G16B16A16_SFLOAT', hw, hh); g.texture('dbf_noise', 'R16G16_SFLOAT', hw, hh)
    g.texture('dbf_tw', 'R16_SFLOAT', hw, hh)
    g.texture('denoised', 'R16G16B16A16_SFLOAT', iw, ih)
    for n in ('dbf_params_out', 'rw_ms', 'rw_pj', 'rw_cj'):
        g.buffer(n, 40)
    # DPostProcessing: BFNet_v1_Filter::encodePost (dbfnetv1_output_channels)
    g.run_k('dbfnetv1_output_channels', {'is_tensor_layout_nchw': False}, (8, 8, 1), (hw, hh, 1),
            ['buf 0 dbf_params_out', 'buf 1 dbf_out', 'tex 0 dbf_feat', 'tex 1 dbf_noise', 'tex 2 dbf_tw'])
    # DMid: BFNet_v1_Filter::encodeMid (reproject_warp)
    g.run_k('reproject_warp', {'bfnet1_enabled': True}, (8, 8, 1), (W, H, 1),
            ['tex 0 color', 'tex 1 warp_color', 'tex 2 warp_prev', 'tex 3 normals', 'tex 4 int_depth', 'tex 5 specular',
             'tex 6 diffuse', 'tex 7 roughness', 'tex 8 nr0|nr1', 'tex 9 nr1|nr0', 'tex 10 int_motion', 'tex 11 hist',
             'tex 14 shin', 'tex 15 fc1|fc0', 'tex 16 fc0|fc1', 'tex 17 flowdiff', 'tex 18 spec_mvec', 'tex 21 rm0|rm1',
             'tex 22 boost', 'tex 23 rm1|rm0', 'tex 24 dy0|dy1', 'tex 25 dy1|dy0',
             'buf 0 rw_ms', 'buf 1 rw_pj', 'buf 2 rw_cj', 'buf 3 exp0|exp1', 'buf 4 exp1|exp0', 'buf 5 preexp', 'buf 6 reset'])
    # BFNet_v1_Filter::encodeAtrous: temporal blend, four a-trous iterations, remodulation
    g.run_k('reproject_blend', {'bfnet1_enabled': True}, (8, 8, 1), (W, H, 1),
            ['tex 0 warp_color', 'tex 1 warp_prev', 'tex 2 warp_color', 'tex 3 bw1|bw0', 'tex 22 boost', 'tex 5 dbf_tw',
             'tex 6 fc0|fc1', 'tex 7 dy0|dy1'])
    srcs = ['warp_color', 'hist', 'warp_color', 'warp_prev']; dsts = ['hist', 'warp_color', 'warp_prev', 'warp_color']
    for i in range(4):
        m = 4 << i
        g.run_k('bilateral', {'bfnet1_enabled': True, 'spec_hit_dist_enabled': False}, (8, 8, 1),
                ((W + m - 1) // m * m, (H + m - 1) // m * m, 1),
                [f'tex 0 {srcs[i]}', 'tex 1 nr1|nr0', f'tex 2 {dsts[i]}', 'tex 3 shin', 'tex 4 dbf_noise', 'tex 5 dbf_feat',
                 'tex 6 int_depth', 'tex 7 mmr',
                 'bytes 0 ' + struct.pack('<4I', 1 << i, i, (~(-1 << i)) & 0xffffffff, (-4 << i) & 0xffffffff).hex(),
                 'bytes 1 ' + struct.pack('<I', i).hex(), 'buf 2 reset'])
    g.run_k('denoiser_between_processing', {'bfnet1_enabled': True, 'str_mask_enabled': False, 'compose_buffer_enabled': False},
            (8, 8, 1), (W, H, 1),
            ['tex 0 warp_color', 'tex 1 color', 'tex 4 shin', 'tex 5 denoised', 'tex 6 specular', 'tex 7 diffuse', 'tex 8 roughness',
             'tex 9 dbf_tw', 'buf 10 exp0|exp1', 'buf 11 preexp'])
    if runtime:
        # last pass: interim firefly clamp on the denoised colour (m4r copies 'denoised' to den_raw first and den_cl back afterwards)
        g.buffer('den_raw', iw * ih * 8); g.buffer('den_cl', iw * ih * 8)
        g.glsl('nn_despeckle', os.path.join(ROOT, 'shaders', 'nn_despeckle.comp'))
        g.run('nn_despeckle', iw, ih, ['den_raw', 'den_cl'], [iw, ih])
        g.lines.append('  only despeckle')
    return dict(W=W, H=H, tw=tw, th=th)


def mat(v):
    return np.array(v, np.float64).reshape(4, 4)


def frame_buffers(cap, fid, t, sz, iw, ih, prev):
    meta = {}
    for line in open(f'{cap}/rr-{fid}.txt'):
        k, *v = line.split()
        meta[k] = v
    f = lambda k, i=0: float(meta[k][i])
    w2v, v2c = mat(meta['world_to_view']), mat(meta['view_to_clip'])      # NGX: row-major, row vectors
    Wc, Pc = w2v.T, v2c.T                                              # simd (column vectors)
    curr_vp = Pc @ Wc
    inv_vp = np.linalg.inv(curr_vp)
    cam = np.linalg.inv(Wc)[:, 3]
    cols = lambda M: M.astype('<f4').flatten(order='F').tobytes()      # column-major bytes
    prev_vp = prev['vp'] if prev else curr_vp
    ms = (f('mvscale', 0), f('mvscale', 1))
    jitter = (f('jitter', 0), f('jitter', 1))
    if os.environ.get('M4R_FLIP_JITTER'):
        jitter = (-jitter[0], -jitter[1])
    # the effect negates nothing here: MetalFX takes the NGX jitter as given by the integration (OptiScaler passes it through)
    pj = prev['jitter'] if prev else (0.0, 0.0)
    dbf = struct.pack('<6fiHHB', ms[0], ms[1], pj[0], pj[1], jitter[0], jitter[1], 26, sz['th'], sz['tw'], 1 if t == 0 else 0).ljust(40, b'\0')
    refl = (cols(inv_vp) + cols(prev_vp) + cols(curr_vp) + struct.pack('<4f', *cam) +
            struct.pack('<4f', 1.0 / sz['W'], 1.0 / sz['H'], sz['W'] / ms[0], sz['H'] / ms[1]))
    out_params = struct.pack('<6fiHHB', 0, 0, 0, 0, 0, 0, 14, sz['th'], sz['tw'], 1 if t == 0 else 0).ljust(40, b'\0')
    bufs = {'bparams': struct.pack('<6H', iw, ih, iw, ih, iw, ih).ljust(144, b'\0'), 'npix': struct.pack('<I', iw * ih),
            'dbf_params_out': out_params, 'rw_ms': struct.pack('<2f', *ms), 'rw_pj': struct.pack('<2f', *pj),
            'rw_cj': struct.pack('<2f', *jitter), 'dbf_params': dbf, 'refl_params': refl, 'preexp': struct.pack('<f', f('preexposure')),
            'reset': struct.pack('<B', 1 if t == 0 else 0).ljust(16, b'\0')}
    return bufs, {'vp': curr_vp, 'jitter': jitter}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--resources', required=True)
    ap.add_argument('--capture', help='M4R_DUMP_RR capture dir: gives the size and the per-frame buffers (offline harness)')
    ap.add_argument('--first', type=int); ap.add_argument('--frames', type=int, default=1)
    ap.add_argument('--input', help='render size WxH (instead of --capture)')
    ap.add_argument('--out', required=True); ap.add_argument('--stage', default='C')
    ap.add_argument('--runtime', action='store_true', help='plan for m4r_denoise (roughness taken from the normals, own spec-hit stub)')
    ap.add_argument('--gpu', help='target GPU (as prepare_bbr.py)')
    ap.add_argument('--force', action='store_true')
    a = ap.parse_args()
    if a.input:
        iw, ih = map(int, a.input.lower().split('x'))
    else:
        meta = dict(l.split(None, 1) for l in open(f'{a.capture}/rr-{a.first}.txt') if l.startswith(('content', 'output')))
        iw, ih = map(int, meta['content'].split())
    if a.gpu:
        m4r_air.set_gpu(a.gpu)
    g = Gen(a.resources, a.out)
    sz = build(g, iw, ih, a.stage, a.runtime)
    g.save()
    if a.capture:
        prev = None
        for t in range(a.frames):
            d = f'{a.out}/frames/{t:02d}'; os.makedirs(d, exist_ok=True)
            bufs, prev = frame_buffers(a.capture, a.first + t, t, sz, iw, ih, prev)
            for n, b in bufs.items():
                open(f'{d}/{n}.bin', 'wb').write(b)
            open(f'{d}/flags.txt', 'w').write('autoexp\n' if os.environ.get('M4R_AUTOEXPOSURE') else '')
    json.dump(sz, open(a.out + '/sizes.json', 'w'))
    print('plan', a.out, sz)


if __name__ == '__main__':
    main()
