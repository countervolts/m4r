#!/usr/bin/env python3
"""Synthetic Ray Reconstruction capture (the M4R_DUMP_RR layout run_denoise_api reads) from the moving-camera scene of
tests/scene_motion.py, for checking the denoiser where no game capture is at hand: per frame rr-N.txt and rr-N-<image>.raw with
noisy colour (the clean shading times heavy-tailed per-pixel noise, as a few-samples-per-pixel path tracer gives, plus fireflies),
reversed infinite hardware depth and the matching view_to_clip / world_to_view (row-vector convention, as NGX gives them), motion
in render pixels (previous - current), world-space normals with roughness in w, diffuse and specular albedo. reference.rgba16f is
the clean, 2x2-supersampled last frame at output size (scene_motion's).
env: M4R_OW/M4R_OH output size (default 1280x720; render size is half), M4R_N frames (8), M4R_RR_DIR (build/fixtures/rr),
M4R_RR_NOISE noise strength (gamma shape; lower = noisier, default 1.5), M4R_RR_SEED.
SPDX-License-Identifier: GPL-3.0-only"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('M4R_MOTION_DIR', os.environ.get('M4R_RR_DIR', 'rr'))
import numpy as np  # noqa: E402
import scene_motion as s  # noqa: E402

NEAR = 0.1
SX, SY, CY = 1.0 / (s.ASPECT * .65), 1.0 / .65, .13 / .65   # scene_motion's rays as a projection (off-centre vertically)
VIEW_TO_CLIP = [SX, 0, 0, 0, 0, SY, 0, 0, 0, CY, 0, 1, 0, 0, NEAR, 0]   # reversed infinite: d = near / z_view


def gbuffer(t, jitter):
    w, h = s.INW, s.INH
    o = s.origin_at(t)
    depth = np.empty((h, w), np.float32)
    nrm = np.zeros((h, w, 4), np.float16)
    dif = np.zeros((h, w, 4), np.uint8)
    spc = np.zeros((h, w, 4), np.uint8)
    for y0 in range(0, h, 32):
        y1 = min(h, y0 + 32)
        d = s.rays(w, h, jitter, y0, y1)
        tt, ob = s.trace(o, d)
        depth[y0:y1] = NEAR / (tt * d[..., 2])
        p = o + tt[..., None] * d
        n = np.zeros(p.shape, np.float32); n[...] = (0, 0, -1)
        n[ob == 1] = (0, 1, 0)
        rough = np.where(ob == 1, 0.3, 0.8).astype(np.float32)
        base = np.clip(s.shade(o, d, tt, ob), 0, 1)
        for k, (c, r, col) in enumerate(s.CENTERS):
            c = s.sphere0(t) if k == 0 else c
            m = ob == 2 + k
            n[m] = ((p - np.array(c, np.float32)) / r)[m]
            rough[m] = 0.45
            base[m] = col
        nrm[y0:y1, :, :3] = n; nrm[y0:y1, :, 3] = rough
        dif[y0:y1, :, :3] = np.clip(base * 255 + .5, 0, 255); dif[y0:y1, :, 3] = 255
        spc[y0:y1, :, :3] = 10; spc[y0:y1, :, 3] = 255
    return depth, nrm, dif, spc


def main():
    rng = np.random.default_rng(int(os.environ.get('M4R_RR_SEED', '7')))
    shape = float(os.environ.get('M4R_RR_NOISE', '1.5'))
    out = s.OUT
    out.mkdir(parents=True, exist_ok=True)
    w, h = s.INW, s.INH
    for t in range(s.N):
        s.CUR[0] = t
        j = (s.halton(t + 1, 2), s.halton(t + 1, 3))
        clean = s.render(s.origin_at(t), w, h, j)
        noise = rng.gamma(shape, 1.0 / shape, (h, w, 1)).astype(np.float32)
        fire = rng.random((h, w, 1)) < 0.002
        noisy = clean.copy()
        noisy[..., :3] = clean[..., :3] * noise + fire * rng.uniform(2, 8, (h, w, 1))
        _, mv = s.lowres_inputs(t, j)
        depth, nrm, dif, spc = gbuffer(t, j)
        mot = np.zeros((h, w, 4), np.float16); mot[..., :2] = mv
        noisy.astype(np.float16).tofile(out / f'rr-{t}-color.raw')
        clean.astype(np.float16).tofile(out / f'clean-{t}.rgba16f')
        depth.tofile(out / f'rr-{t}-depth.raw')
        mot.tofile(out / f'rr-{t}-motion.raw')
        nrm.tofile(out / f'rr-{t}-normals.raw')
        dif.tofile(out / f'rr-{t}-diffuse_albedo.raw')
        spc.tofile(out / f'rr-{t}-specular_albedo.raw')
        o = s.origin_at(t)
        w2v = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, -o[0], -o[1], -o[2], 1]
        lines = [f'output {s.W} {s.H}', f'content {w} {h}', f'jitter {j[0]} {j[1]}', 'mvscale 1 1', 'preexposure 1',
                 f'reset {1 if t == 0 else 0}', 'depth_reversed 1', 'hardware_depth 1',
                 'world_to_view ' + ' '.join(map(str, w2v)), 'view_to_clip ' + ' '.join(map(str, VIEW_TO_CLIP)),
                 f'image color format 97 size {w} {h} bpp 8', f'image depth format 100 size {w} {h} bpp 4',
                 f'image motion format 97 size {w} {h} bpp 8', f'image normals format 97 size {w} {h} bpp 8',
                 f'image diffuse_albedo format 37 size {w} {h} bpp 4', f'image specular_albedo format 37 size {w} {h} bpp 4']
        (out / f'rr-{t}.txt').write_text('\n'.join(lines) + '\n')
        print(f'frame {t}')
    ref = s.render(s.origin_at(s.N - 1), s.W * 2, s.H * 2, (0, 0))
    ref = ref.reshape(s.H, 2, s.W, 2, 4).mean((1, 3))
    ref.astype(np.float16).tofile(out / 'reference.rgba16f')
    print(f'-> {out}')


if __name__ == '__main__':
    main()
