#!/usr/bin/env python3
"""Noise report for an RR capture (M4R_DUMP_RR): relative noise of the game's input, m4r's denoised colour and m4r's final output,
split by surface roughness (the captures also hold what m4r produced in the game: out_denoised, out_output).
usage: noise_report.py <capture dir> <first frame id> <frame index within the capture, default last> [--motion]
  relative noise = rms of (x - 5x5 mean) / mean over a pixel set (luma). --motion also prints mean |motion| and, per roughness
  band, noise of the pixels that move more than 3 px. SPDX-License-Identifier: GPL-3.0-only"""
import sys, numpy as np
from scipy.ndimage import uniform_filter

cap, first = sys.argv[1], int(sys.argv[2])
idx = int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3].isdigit() else 5
meta = {}
for l in open(f'{cap}/rr-{first + idx}.txt'):
    k, *v = l.split(); meta[k] = v
W, H = int(meta['content'][0]), int(meta['content'][1]); ow, oh = int(meta['output'][0]), int(meta['output'][1])
f16 = lambda n, w, h: np.fromfile(f'{cap}/rr-{first + idx}-{n}.raw', '<f2').astype(np.float32).reshape(h, w, 4)
inp, nrm = f16('color', W, H)[..., :3].mean(-1), f16('normals', W, H)
rough = nrm[..., 3]
den = f16('out_denoised', W, H)[..., :3].mean(-1) if 'out_denoised' in open(f'{cap}/rr-{first + idx}.txt').read() else None
out = f16('out_output', ow, oh)[..., :3].mean(-1)
mv = f16('motion', W, H)[..., :2] * np.array([W, H], np.float32); sp = np.hypot(mv[..., 0], mv[..., 1])


def rel(x, m):
    hp, mu = x - uniform_filter(x, 5), uniform_filter(x, 5)
    return np.sqrt((hp ** 2)[m].mean()) / max(mu[m].mean(), 1e-6)


up = lambda m: np.kron(m.astype(np.uint8), np.ones((oh // H + 1, ow // W + 1), np.uint8))[:oh, :ow].astype(bool)
print('capture %s frame %d: %dx%d -> %dx%d, mean |motion| %.1f px' % (cap, first + idx, W, H, ow, oh, sp.mean()))
bands = (('all', np.ones_like(rough, bool)), ('mirror r<0.08', rough < 0.08), ('glossy 0.08-0.2', (rough >= 0.08) & (rough < 0.2)),
         ('rest r>=0.2', rough >= 0.2))
for name, m in bands:
    mm = m & (sp > 3) if '--motion' in sys.argv else m
    if mm.sum() < 50: continue
    row = 'input %.2f' % rel(inp, mm)
    if den is not None: row += ' | denoised %.2f' % rel(den, mm)
    print('%-16s %5.1f%% | %s | final %.2f' % (name, 100 * mm.mean(), row, rel(out, up(mm)) if True else 0))
