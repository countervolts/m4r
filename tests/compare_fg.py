#!/usr/bin/env python3
"""Interpolated frames vs the true in-between frames (sRGB PSNR), with naive baselines; writes PNG strips.
usage: compare_fg.py <plan dir> <fixture dir> <frames> <ow> <oh>"""
import sys
import numpy as np
from PIL import Image
sys.path.insert(0, __import__("os").path.dirname(__file__))
from compare import srgb, psnr  # noqa: E402

d, fx, n, ow, oh = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
ld = lambda p: np.fromfile(p, np.float16).reshape(oh, ow, 4)[..., :3].astype(np.float32)
for t in range(1, n):
    mid, prev, cur = ld(f'{fx}/mid-{t:02}.rgba16f'), ld(f'{fx}/color-{t - 1:02}.rgba16f'), ld(f'{fx}/color-{t:02}.rgba16f')
    out = ld(f'{d}/interp-{t:02}.rgba16f')
    print(f'frame {t}: interpolated {psnr(out, mid):.2f} dB | repeat previous {psnr(prev, mid):.2f} | blend {psnr((prev + cur) / 2, mid):.2f}'
          f' | nan {np.isnan(out).mean() * 100:.2f}%')
    strip = np.concatenate([srgb(out), srgb(mid)], 0)
    Image.fromarray((strip * 255 + .5).astype(np.uint8)).save(f'{d}/interp-{t:02}.png')
