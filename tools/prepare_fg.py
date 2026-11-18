#!/usr/bin/env python3
"""Prepare MetalFX frame interpolation (FrameGenImpl<MFXDevice3>) for one output/input size: kernels and plan.

The plan reproduces FrameGenImpl::encodeTo (docs/metalfx-framegen.md) for the common configuration: no distortion
field, the game supplies the previous colour, UI not composited, disocclusion-blend off. Frame parity p alternates;
resources written "a|b" pick a for even and b for odd frames.

External (bound by the host): color (current frame, output size), prevcolor, depth / motion (input size),
output (interpolated frame, output size). Per-frame buffer: fgparams (176-byte FrameGenShaderParams).
"""
import argparse, hashlib, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'air'))
import metallib  # noqa: E402
import m4r_air  # noqa: E402


def build(a):
    ow, oh = a.output
    iw, ih = a.input
    out = a.out
    os.makedirs(out + '/k', exist_ok=True)
    mtl = open(os.path.join(a.resources, 'default.metallib'), 'rb').read()
    lib = metallib.parse(mtl)
    src = {}
    decl, passes, kernels = [], [], set()

    def kernel(fn, consts, tg):
        key = m4r_air.specialization_key(fn, {k: str(v) for k, v in consts.items()}, tg, hashlib.sha256(mtl).hexdigest()[:16])
        name = f'{fn}-{key}'
        if name not in kernels:
            base = os.path.join(out, 'k', name)
            if not os.path.exists(base + '.hsaco') or a.force:
                if fn not in src:
                    f = next(f for f in lib['functions'] if f.name == fn)
                    src[fn] = m4r_air.to_text(metallib.bitcode(mtl, lib, f))
                ir, man = m4r_air.translate(src[fn], fn, consts, tg)
                m4r_air.compile_ir(ir, base + '.hsaco')
                m4r_air.write_manifest(man, base)
            kernels.add(name)
            extra = f' lds={tg[0] * tg[1] * 8}' if fn == 'warpImage' else ''
            decl.append(f'kernel {name} k/{name}{extra}')
        return name

    def tex(n, fmt, w, h):
        decl.append(f'texture {n} {fmt} {max(1, w)} {max(1, h)}')

    def d(fn, consts, tg, mode, grid, binds):
        k = kernel(fn, consts, tg)
        passes.append(f'pass {k} {mode} {grid[0]} {grid[1]} {grid[2]}')
        passes.extend('  ' + b for b in binds)

    r16 = lambda v: (v + 15) & ~15
    levels = min(max(min(ow, oh), 1).bit_length() - 1 - 2, 7)
    # resources (FrameGenImpl constructor)
    tex('osplat0', 'R32_UINT', ow // 8, oh // 8); tex('osplat1', 'R32_UINT', ow // 8, oh // 8)
    tex('gsplat0', 'R32_UINT', iw, ih); tex('gsplat1', 'R32_UINT', iw, ih)
    tex('ddepth0', 'R32_SFLOAT', iw, ih); tex('ddepth1', 'R32_SFLOAT', iw, ih)
    for l in range(levels):
        for k in (0, 1):
            tex(f'pyr{l}_{k}', 'R8_UNORM', ow >> l, oh >> l)
        tex(f'flowA{l}', 'R16G16_SFLOAT', ow >> (l + 3), oh >> (l + 3))
        tex(f'flowB{l}', 'R16G16_SFLOAT', ow >> (l + 3), oh >> (l + 3))
    tex('flowlast', 'R16G16_SFLOAT', ow >> (levels + 2), oh >> (levels + 2))
    tex('gmv', 'R16G16_SFLOAT', ow, oh); tex('mask', 'R8_UINT', ow, oh)
    tex('mover', 'R16_SFLOAT', iw, ih); tex('omv', 'R16G16_SFLOAT', ow, oh)
    decl.append('buffer hist 18436')
    decl.append('view hist_off hist 9216')
    decl.append('view hist_change hist 18432')
    decl.append('buffer fgparams 176')
    fc1 = 1  # constant 1 is false only for a few packed/sRGB colour formats

    # Downsample
    d('computeLuminance', {'#1': fc1, '#3': 0, '#4': 0, '#5': 0}, [32, 32, 1], 'threads', (ow, oh, 1),
      ['buf 0 fgparams', 'tex 0 color', 'tex 1 pyr0_0|pyr0_1', 'tex 2 nil', 'buf 1 hist|hist_off'])
    for l in range(1, levels):
        d('downSample', {}, [16, 16, 1], 'threads', (ow >> l, oh >> l, 1),
          [f'tex 0 pyr{l - 1}_0|pyr{l - 1}_1', f'tex 1 pyr{l}_0|pyr{l}_1'])
    # Analyze histogram
    d('analyzeHistogram', {}, [256, 1, 1], 'groups', (1, 9, 1),
      ['buf 1 hist|hist_off', 'buf 2 hist_off|hist', 'buf 3 hist_change'])
    # Optical motion vectors, coarse to fine
    for i in range(levels):
        l = levels - 1 - i
        fw, fh = ow >> (l + 3), oh >> (l + 3)
        coarse = f'flowB{l + 1}' if i > 0 else 'nil'
        d('findOpticalMotionVectorsV3', {}, [8, 4, 1], 'groups', (max(1, fw), max(1, fh), 1),
          ['tex 4 motion', f'tex 0 flowA{l}', f'tex 1 pyr{l}_0|pyr{l}_1', f'tex 2 pyr{l}_1|pyr{l}_0', f'tex 3 {coarse}'])
        d('medianFilter', {}, [16, 16, 1], 'threads', (r16(max(1, fw)), r16(max(1, fh)), 1), [f'tex 0 flowB{l}', f'tex 1 flowA{l}'])
    d('initMotionVectors', {}, [16, 16, 1], 'threads', (ow // 8, oh // 8, 1), ['tex 0 osplat0', 'tex 1 osplat1'])
    d('splatOpticalMotionVectors', {}, [8, 8, 1], 'threads', (ow // 8, oh // 8, 1),
      ['tex 0 osplat0', 'tex 1 osplat1', 'tex 2 ddepth0|ddepth1', 'tex 3 ddepth1|ddepth0', 'tex 4 flowB0', 'tex 5 flowlast'])
    d('inpaintMotionVectors', {}, [16, 16, 1], 'threads', (r16(ow), r16(oh), 1), ['tex 0 omv', 'tex 1 osplat0', 'tex 2 osplat1'])
    d('clearAppMotionVectors', {}, [16, 16, 1], 'threads', (r16(iw), r16(ih), 1), ['tex 0 gsplat0', 'tex 1 gsplat1'])
    d('dilateDepth', {'#2': 1, '#5': 0}, [16, 16, 1], 'threads', (r16(iw), r16(ih), 1),
      ['tex 2 motion', 'tex 3 gmv', 'tex 0 depth', 'tex 1 ddepth0|ddepth1', 'tex 4 mover'])
    d('splatAppMotionVectors', {}, [16, 16, 1], 'threads', (r16(iw), r16(ih), 1),
      ['tex 0 gsplat0', 'tex 1 gsplat1', 'tex 2 color', 'tex 3 prevcolor', 'tex 4 motion', 'tex 5 ddepth0|ddepth1',
       'tex 6 ddepth1|ddepth0'])
    up = 'upsampleAppMotionVectors' if (iw, ih) != (ow, oh) else 'fillCracksAppMotionVectors'
    d(up, {'#2': 1, '#5': 0}, [16, 16, 1], 'threads', (r16(ow), r16(oh), 1),
      ['tex 0 gmv', 'tex 1 gsplat0', 'tex 2 gsplat1', 'tex 3 mask', 'tex 4 ddepth0|ddepth1', 'tex 5 ddepth1|ddepth0'])
    # Warp (tile shader, one 32x32 threadgroup per tile; render target = output)
    d('warpImage', {'#3': 0, '#4': 0, '#5': 0}, [32, 32, 1], 'groups', ((ow + 31) // 32, (oh + 31) // 32, 1),
      ['buf 0 fgparams', 'buf 1 hist_change', 'tex 1 mask', 'tex 2 ddepth1|ddepth0', 'tex 3 color', 'tex 4 prevcolor',
       'tex 5 gmv', 'tex 6 omv', 'tex 7 nil', 'tex 8 nil', 'tex 9 nil', 'tex 10 color', 'tex 11 ddepth0|ddepth1',
       'tex 14 mover', 'tex 255 output', 'tgmem 0 8192'])
    open(os.path.join(out, 'plan.txt'), 'w').write('\n'.join(decl + passes) + '\n')
    json.dump({'output': [ow, oh], 'input': [iw, ih], 'levels': levels}, open(os.path.join(out, 'sizes.json'), 'w'))
    print(f'{len(kernels)} kernels, {sum(1 for p in passes if p.startswith("pass"))} passes, {levels} levels -> {out}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--resources', required=True)
    ap.add_argument('--output', required=True, type=lambda s: [int(x) for x in s.split('x')])
    ap.add_argument('--input', required=True, type=lambda s: [int(x) for x in s.split('x')])
    ap.add_argument('--out', required=True)
    ap.add_argument('--force', action='store_true')
    build(ap.parse_args())
