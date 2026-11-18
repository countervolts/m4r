#!/usr/bin/env python3
"""Frame-interpolation fixture from the moving-camera scene (tests/scene_motion.py): unjittered colour frames at
integer times (output size), depth and motion at input size, and the true in-between frames at t - 0.5.
Writes build/fixtures/fg/: color-NN.rgba16f, depth-NN.r32f, mv-NN.rg16f (input pixels, previous - current), mid-NN.rgba16f.
env: M4R_OW/M4R_OH output size, M4R_IW/M4R_IH input size, M4R_N frames, M4R_OBJ=mv for a moving sphere"""
# SPDX-License-Identifier: GPL-3.0-only
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('M4R_MOTION_DIR', 'fg')
import scene_motion as s  # noqa: E402

if __name__ == '__main__':
    out = s.ROOT / 'build' / 'fixtures' / 'fg'
    out.mkdir(parents=True, exist_ok=True)
    for t in range(s.N):
        s.CUR[0] = t
        s.render(s.origin_at(t), s.W, s.H, (0, 0)).astype('<f2').tofile(out / f'color-{t:02}.rgba16f')
        dep, mv = s.lowres_inputs(t, (0, 0))
        dep.astype('<f4').tofile(out / f'depth-{t:02}.r32f')
        mv.astype('<f2').tofile(out / f'mv-{t:02}.rg16f')
        if t:
            s.CUR[0] = t - 0.5
            s.render(s.origin_at(t - 0.5), s.W, s.H, (0, 0)).astype('<f2').tofile(out / f'mid-{t:02}.rgba16f')
        print('frame', t, flush=True)
