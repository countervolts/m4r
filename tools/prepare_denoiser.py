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
        self.res = res
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
        for kv in filter(None, os.environ.get('M4R_FC', '').split(',')):   # experiments: M4R_FC=name=0|1,... overrides function constants
            n, v = kv.split('=')
            if n in consts:
                consts = dict(consts, **{n: bool(int(v))})
        k = self.kernel(fn, consts, tg)
        self.lines.append(f'pass {k} {"groups" if groups else "threads"} {grid[0]} {grid[1]} {grid[2]}')
        if only:
            self.lines.append('  only ' + only)
        for b in binds:
            self.lines.append('  ' + b)

    def save(self):
        names = [l.split()[1] for l in self.lines if l.split(' ', 1)[0] in ('buffer', 'texture')]
        dup = sorted({n for n in names if names.count(n) > 1})
        assert not dup, f'buffers/textures declared twice (they would share memory under different sizes): {dup}'
        decl = [l for l in self.lines if l.split(' ', 1)[0] in ('buffer', 'texture', 'kernel', 'glsl')]
        rest = [l for l in self.lines if l.split(' ', 1)[0] not in ('buffer', 'texture', 'kernel', 'glsl')]
        open(self.out + '/plan.txt', 'w').write('\n'.join([f'gpu {m4r_air.GPU}'] + decl + rest) + '\n')


def build(g, iw, ih, stage, runtime=False, ow=None, oh=None):
    ow, oh = ow or iw * 3, oh or ih * 3
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
    # BRNet_v3_Filter::encodeMidForDenoise (DMidProcessing, before BFNet's reproject): dilate the flow, splat it, and compare: flowDiff
    # (BFNet's flow_diff_tex) and the motion-reduced mirror mask (BFNet +0xe8, the bilateral's mirrorMask). Reversed depth skips the
    # forward-z init; flow_splat_reverse_depth follows the depth convention. bytes(2) of flow_splat in the host code has no kernel argument.
    rev = 1  # TODO runtime: both conventions (plans exist per convention like bbr's)
    g.texture('flowd', 'R16G16_SFLOAT', iw, ih); g.texture('splat', 'R16G16_SFLOAT', iw, ih)
    g.texture('flowdiff', 'R16_SFLOAT', W, H); g.texture('mmr', 'R8_UNORM', W, H)
    g.buffer('splatbuf', iw * ih * 4)
    g.run_k('dilateFlow', {'full_resolution_flow': False}, (8, 8, 1), (iw + 2, ih + 2, 1),
            ['tex 0 int_depth', 'tex 1 int_motion', 'tex 2 flowd', 'bytes 0 ' + struct.pack('<f', 1.0 if rev else -1.0).hex(), 'buf 1 bparams'])
    g.run_k('brnetv3_flow_splat', {'flow_splat_reverse_depth': bool(rev)}, (8, 8, 1), (iw, ih, 1),
            ['tex 0 flowd', 'tex 1 int_depth', 'buf 0 splatbuf', 'buf 1 bparams'])
    g.run_k('brnetv3_flow_splat_texture', {}, (8, 8, 1), (iw, ih, 1), ['tex 0 flowd', 'tex 1 splat', 'buf 0 splatbuf', 'buf 1 bparams'])
    g.run_k('brnetv3_flow_diff_texture', {'bfnet1_enabled': True, 'spec_hit_dist_enabled': False}, (8, 8, 1), (iw, ih, 1),
            ['tex 0 flowd', 'tex 1 splat', 'tex 2 flowdiff', 'tex 3 mm0|mm1', 'tex 4 mm1|mm0', 'tex 5 mmr', 'buf 1 bparams'])
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
    g.texture('shin', 'R8_UNORM', W, H)
    for n in ('fc0', 'fc1', 'dy0', 'dy1'):
        g.texture(n, 'R16_SFLOAT', W, H)
    for n in ('rm0', 'rm1'):
        g.texture(n, 'R16G16B16A16_SFLOAT', W, H)
    for n in ('bw0', 'bw1'):
        g.texture(n, 'R8_UNORM', W, H)
    g.texture('spec_mvec', 'R16G16_SFLOAT', W, H)
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
        # m4r's own passes on the denoised colour (not MetalFX): m4r copies 'denoised' out to buffers, runs these, copies den_cl back.
        # The interim firefly clamp (only despeckle, den_raw -> den_cl) on the BBR path only (V3 removes fireflies itself); the
        # glossy-in-motion smoothing (only glossy, shaders/nn_glossy.comp: three a-trous steps den_cl -> den_g -> den_raw -> den_cl) on
        # both, before V3 reads the denoised colour. m4r also copies int_depth / int_motion into gl_depth / gl_motion first.
        g.buffer('den_raw', iw * ih * 8); g.buffer('den_cl', iw * ih * 8)
        if stage < 'F':
            g.glsl('nn_despeckle', os.path.join(ROOT, 'shaders', 'nn_despeckle.comp'))
            g.run('nn_despeckle', iw, ih, ['den_raw', 'den_cl'], [iw, ih])
            g.lines.append('  only despeckle')
        g.buffer('gl_depth', iw * ih * 4); g.buffer('gl_motion', iw * ih * 4); g.buffer('den_g', iw * ih * 8)
        g.glsl('nn_glossy', os.path.join(ROOT, 'shaders', 'nn_glossy.comp'))
        for step, a, b in ((1, 'den_cl', 'den_g'), (2, 'den_g', 'den_raw'), (4, 'den_raw', 'den_cl')):
            g.run('nn_glossy', iw, ih, [a, b, 'nrm_raw', 'gl_depth', 'gl_motion', 'rw_ms'], [iw, ih, step])
            g.lines.append('  only glossy')
    if stage >= 'E':
        v3_ubf(g, iw, ih, W, H, ow, oh, full=stage >= 'F')
    return dict(W=W, H=H, tw=tw, th=th)


def v3_sizes(ow, oh):
    """UBFNet input size (half the output size, rounded up to 32) and its output size"""
    Wn, Hn = ((ow // 2 + 31) // 32) * 32, ((oh // 2 + 31) // 32) * 32
    return Wn, Hn, Wn // 2, Hn // 2


def v3_ubf(g, iw, ih, W, H, ow, oh, full=False):
    """BRNet_v3_Filter::encodePreForDenoise (build_tensors_low_resolution, warp_history_high_resolution, lowres_signals), the UBFNet
    pass and, with full, encodePost (the two output conversions and the anisotropic gaussian filter that writes 'v3out' and the history).
    Bindings from the host code (docs/metalfx-denoiser-passes.txt, fn_BRN_*.txt). The network input is half the output size padded
    to 32 (Wn x Hn); the textures around it are unpadded: output/2 (temporal, ema/var, aabb, sample statistics) and output/4 (spatial)."""
    Wn, Hn, Wt, Ht = v3_sizes(ow, oh)
    # textures at network resolution are output/2 and the spatial one output/4, NOT the 32-aligned tensor size: the filter and
    # lowres_signals address them with normalised output coordinates (BRNet_v3_Filter constructor: W/2 x H/2); only the tensor is padded.
    # (With the tensor size the recurrent ema/var/temporal state drifted by 736/720 per frame: dark bands and ghosts at the bottom.)
    Wh, Hh, Wq, Hq = ow // 2, oh // 2, ow // 4, oh // 4
    F = 'R16G16B16A16_SFLOAT'
    g.texture('crr', F, iw, ih)
    g.texture('v3hist', F, ow, oh)                                      # the previous output (written by the filter below)
    for n in ('fah', 'v3clamp', 'v3blend', 'v3fblend'):                 # output size
        g.texture(n, F, ow, oh)
    for n in ('v3ema0', 'v3ema1', 'v3var0', 'v3var1', 'v3lwv'):         # network-input size
        g.texture(n, 'R16_SFLOAT', Wh, Hh)
    g.texture('v3temporal2', F, Wh, Hh); g.texture('aabbmin', F, Wh, Hh); g.texture('aabbmax', F, Wh, Hh)
    g.texture('v3spatial', F, Wq, Hq)
    for n in ('v3se0', 'v3se1', 'v3sq0', 'v3sq1'):                      # sample statistics (BRNet +0xe0/+0xf0, ping-pong)
        g.texture(n, 'R16_SFLOAT', Wh, Hh)
    for n in ('v3cnt0', 'v3cnt1'):
        g.texture(n, 'R16_UINT', Wh, Hh)
    g.texture('v3zero', F, 16, 16); g.texture('v3const', F, 16, 16)
    g.buffer('v3reset', 16); g.buffer('v3resetA', 16); g.buffer('v3desc', 24); g.buffer('v3bd', 16, struct.pack('<iHH', 22, Ht, Wt).ljust(16, b'\0')); g.buffer('ubf_in', Wn * Hn * 10)
    g.run_k('brnetv3_build_tensors_low_resolution', {'bfnet1_enabled': True, 'full_resolution_flow': False, 'jittered_motion_vectors': False},
            (8, 8, 1), (iw + 2, ih + 2, 1),
            ['tex 0 denoised', 'tex 1 int_depth', 'tex 2 int_motion', 'tex 3 crr', 'tex 4 flowd', 'bytes 0 ' + struct.pack('<f', 1.0).hex(),
             'buf 1 bparams', 'buf 2 preexp', 'buf 3 exp0|exp1'])
    g.run_k('brnetv3_warp_history_high_resolution', {'bfnet1_enabled': True, 'stlanczos_enabled': False, 'new_overshoot_clamp': False},
            (8, 8, 1), (ow, oh, 1),
            ['tex 0 crr', 'tex 1 v3hist', 'tex 2 ' + ('v3zero' if os.environ.get('M4R_ZERO_FLOW_WARP') else 'flowd'), 'tex 3 fah', 'tex 4 v3blend', 'tex 5 v3fblend', 'tex 6 v3clamp', 'tex 7 denoised',
             'buf 0 v3desc', 'buf 1 bparams', 'buf 2 exp0|exp1', 'buf 3 exp1|exp0', 'buf 4 ' + ('v3resetA' if os.environ.get('M4R_RESET_WARP') else 'hinv'), 'buf 5 preexp'])
    g.run_k('brnetv3_lowres_signals', {'mid_bicubic_warp': False, 'half_res_input': False, 'quantized': False, 'is_tensor_layout_nchw': False,
                                      'bfnet1_enabled': True, 'big_epsilon_and_default_scales_enabled': False, 'new_overshoot_clamp': False},
            (8, 8, 1), (Wh, Hh, 1),
            ['tex 0 crr', 'tex 1 fah', 'tex 2 flowd', 'tex 3 v3hist', 'tex 4 v3ema0|v3ema1', 'tex 5 v3var0|v3var1',
             'tex 6 v3ema1|v3ema0', 'tex 7 v3var1|v3var0', 'tex 8 ' + ('v3zero' if os.environ.get('M4R_NO_TEMPORAL_FB') else 'v3temporal2'), 'tex 9 v3lwv', 'tex 15 aabbmin', 'tex 16 aabbmax',
             'tex 17 normals', 'tex 18 diffuse', 'tex 19 roughness', 'tex 20 specular',
             'buf 0 v3desc', 'buf 1 bparams', 'buf 2 ubf_in', 'buf 3 exp0|exp1', 'buf 4 exp1|exp0', 'buf 5 ' + ('v3resetA' if os.environ.get('M4R_RESET_LOWRES') else 'hinv'),
             'bytes 6 ' + struct.pack('<2H', ow // 2, oh // 2).hex()])
    # BRNet_v3_Filter::encodeMid: update_sample_statistics (per-pixel running mean / mean of squares of the colour, used by the filter's
    # history clamp); in/out ping-pong, the filter reads the updated ones
    g.run_k('brnetv3_update_sample_statistics', {}, (8, 8, 1), (Wh, Hh, 1),
            ['tex 0 flowd', 'tex 1 flowdiff', 'tex 2 v3se0|v3se1', 'tex 3 v3sq0|v3sq1', 'tex 4 v3se1|v3se0', 'tex 5 v3sq1|v3sq0',
             'tex 6 v3cnt0|v3cnt1', 'tex 8 v3cnt1|v3cnt0', 'tex 10 crr', 'buf 1 bparams', 'buf 3 hinv'])
    import denoise_nets as dn
    # sigmoid in the network: with bfnet1_enabled the conversion kernels copy the values (the fp16 + sigmoid branch is for bfnet1 off)
    dn.ubf_net(g, g.dat['ubf'], dn.ubf_head_blob(g.res), Hn, Wn, 'ubf_in', 'ubf_out', pre='ubf_', sigmoid=True)
    if not full:
        return
    fc = {'quantized': False, 'is_tensor_layout_nchw': False, 'bfnet1_enabled': True}
    g.run_k('brnetv3_output_spatial_lr', fc, (8, 8, 1), (Wq, Hq, 1), ['tex 0 v3spatial', 'buf 0 ubf_out', 'buf 1 v3bd'])
    g.run_k('brnetv3_output_temporal', fc, (8, 8, 1), (Wh, Hh, 1), ['tex 0 v3temporal2', 'buf 0 ubf_out', 'buf 1 v3bd'])
    g.texture('v3out', F, ow, oh)
    g.run_k('brnetv3_anisotropic_gaussian_filter',
            {'quantized': False, 'reactive_mask_enabled': False, 'bfnet1_enabled': True, 'line_locking_enabled': False,
             'exposure_viz_enabled': False, 'big_epsilon_and_default_scales_enabled': False, 'stlanczos_enabled': False,
             'new_overshoot_clamp': False}, (8, 8, 1), (ow, oh, 1),
            ['tex 0 crr', 'tex 1 fah', 'tex 3 v3out', 'tex 4 v3hist', 'tex 5 ' + ('v3zero' if os.environ.get('M4R_ZERO_FLOWDIFF') else 'flowdiff'), 'tex 6 ' + ('v3zero' if os.environ.get('M4R_ZERO_LWV') else 'v3lwv'), 'tex 7 flowd', 'tex 10 v3clamp',
             'tex 11 ' + ('v3zero' if os.environ.get('M4R_ZERO_SPATIAL') else 'v3const' if os.environ.get('M4R_CONST_SPATIAL') else 'v3spatial'), 'tex 12 ' + ('v3zero' if os.environ.get('M4R_ZERO_TEMPORAL') else 'v3const' if os.environ.get('M4R_CONST_TEMPORAL') else 'v3temporal2'),
             'tex 13 aabbmin', 'tex 14 aabbmax', 'tex 15 v3se1|v3se0', 'tex 16 v3sq1|v3sq0',
             'tex 19 v3fblend', 'tex 20 v3blend', 'buf 0 bparams', 'bytes 1 ' + (struct.pack('<2f', ow / iw, oh / ih) if os.environ.get('M4R_CPF') == 'inv' else struct.pack('<2f', 1.0, 1.0) if os.environ.get('M4R_CPF') == 'one' else struct.pack('<2f', iw / ow, ih / oh)).hex(),
             'buf 2 ' + ('v3resetA' if os.environ.get('M4R_RESET_ANISO') else 'hinv'), 'buf 3 preexp', 'buf 4 exp0|exp1'])
    # m4r's hybrid output (not MetalFX; shaders/nn_hybrid.comp): after the run, m4r upscales 'denoised' with BBR into the game's output,
    # copies v3out and that output into these buffers, runs this pass (only hybrid) and copies hy_bbr back
    g.buffer('hy_v3', ow * oh * 8); g.buffer('hy_bbr', ow * oh * 8)
    g.glsl('nn_hybrid', os.path.join(ROOT, 'shaders', 'nn_hybrid.comp'))
    g.run('nn_hybrid', ow, oh, ['hy_v3', 'hy_bbr'], [ow, oh])
    g.lines.append('  only hybrid')


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
    ow, oh = (int(meta['output'][0]), int(meta['output'][1])) if 'output' in meta else (iw * 3, ih * 3)
    Wn, Hn, Wt, Ht = v3_sizes(ow, oh)
    # BRNetShaderParams (144 B): contentSize, maxInputSize, outputSize, five offsets (zero), jitter, motionScale, outputResMotionMultiplier,
    # normalizeMotionScale, tensor strides (zero)
    jb = (-jitter[0], -jitter[1]) if os.environ.get('M4R_FLIP_JITTER_BP') else jitter
    jd = (-jitter[0], -jitter[1]) if os.environ.get('M4R_FLIP_JITTER_DESC') else jitter
    bsp = (struct.pack('<6H', iw, ih, iw, ih, ow, oh) + bytes(20) +
           struct.pack('<8f', jb[0], jb[1], ms[0], ms[1], iw / ow, ih / oh, ms[0] / iw, ms[1] / ih)).ljust(144, b'\0')
    # brnet_v3_build_tensors_t (24 B): motionScale * outputSize, jitter, tensor row bytes (5 halves), height, width
    v3d = struct.pack('<ffffiHH', ms[0] * ow, ms[1] * oh, jd[0], jd[1], 10, Hn, Wn)
    bufs = {'bparams': bsp, 'v3desc': v3d, 'v3bd': struct.pack('<iHH', 22, Ht, Wt), 'v3resetA': b'\x01'.ljust(16, b'\0'), 'v3reset': (b'\x01' if (t == 0 or os.environ.get('M4R_V3_RESET')) else b'\x00').ljust(16, b'\0'), 'npix': struct.pack('<I', iw * ih),
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
    ap.add_argument('--output', help='output size WxH (stage F; with --capture it is read from the capture)')
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
    ow = oh = None
    if a.output:
        ow, oh = map(int, a.output.lower().split('x'))
    if a.capture:
        o = dict(l.split(None, 1) for l in open(f'{a.capture}/rr-{a.first}.txt') if l.startswith('output'))
        ow, oh = map(int, o['output'].split())
    sz = build(g, iw, ih, a.stage, a.runtime, ow, oh)
    g.save()
    if a.capture:
        prev = None
        for t in range(a.frames):
            d = f'{a.out}/frames/{t:02d}'; os.makedirs(d, exist_ok=True)
            bufs, prev = frame_buffers(a.capture, a.first + (t % int(os.environ['M4R_LOOP']) if os.environ.get('M4R_LOOP') else t), t, sz, iw, ih, prev)
            for n, b in bufs.items():
                open(f'{d}/{n}.bin', 'wb').write(b)
            open(f'{d}/flags.txt', 'w').write('autoexp\n' if os.environ.get('M4R_AUTOEXPOSURE') else '')
    json.dump(sz, open(a.out + '/sizes.json', 'w'))
    print('plan', a.out, sz)


if __name__ == '__main__':
    main()
