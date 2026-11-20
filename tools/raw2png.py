#!/usr/bin/env python3
"""Convert a raw texture dump (run_denoise M4R_DUMP_TEX / M4R_DUMP_RR) to a tonemapped PNG.
usage: raw2png.py <in.raw> <w> <h> <fmt: rgba16f|r16f|rg16f|r32f|rgba8|r8> <out.png> [scale]
SPDX-License-Identifier: GPL-3.0-only"""
import sys
import numpy as np
from PIL import Image
f, w, h, fmt, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5]
scale = float(sys.argv[6]) if len(sys.argv) > 6 else 1.0
d = open(f, 'rb').read()
ch = {'rgba16f': 4, 'r16f': 1, 'rg16f': 2, 'r32f': 1, 'rgba8': 4, 'r8': 1}[fmt]
dt = {'rgba16f': '<f2', 'r16f': '<f2', 'rg16f': '<f2', 'r32f': '<f4', 'rgba8': 'u1', 'r8': 'u1'}[fmt]
a = np.frombuffer(d, dt).astype(np.float32).reshape(h, w, ch)
if dt == 'u1':
    a = a / 255.0
a = a * scale
if ch == 1:
    a = np.repeat(a, 3, 2)
elif ch == 2:
    a = np.concatenate([a, np.zeros_like(a[..., :1]) + 0.5], 2)
else:
    a = a[..., :3]
a = np.nan_to_num(a)
if fmt in ('rgba16f',) and ch == 4:
    a = np.clip(a, 0, None); a = a / (1 + a)          # simple Reinhard
    a = a ** (1 / 2.2)
else:
    a = np.clip(a, 0, 1)
Image.fromarray((a * 255 + .5).astype(np.uint8)).save(out)
