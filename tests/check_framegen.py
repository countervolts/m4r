#!/usr/bin/env python3
"""Checks a harness run with MetalFX frame generation (tests/run_optiscaler.sh with M4R_FG_PLAN) and its reference replay
(build/ref_framegen). For every frame the run presented twice:
  - the real frame put back by the output (bbk1-NN) must equal the upscaled frame (out-NN) bit for bit,
  - the generated frame (bbk-NN) must match the reference replay (ref-gen-NN): bit for bit, or at least 60 dB PSNR (a few pixels
    of one frame differ between the two environments, see docs/optiscaler.md),
and it prints how far the generated frame is from its neighbours (it should lie between them, not equal either).
For frames presented once, the presented buffer (bbk-NN) must be the upscaled frame.
usage: check_framegen.py <app dir> <frames> [fg fixture dir with mid-NN.rgba16f: also prints PSNR against the true in-between frame]
"""
import re, sys
import numpy as np

app, frames = sys.argv[1].rstrip('/'), int(sys.argv[2])
mids = sys.argv[3] if len(sys.argv) > 3 else None
if mids:
    sys.path.insert(0, __file__.rsplit('/', 1)[0])
    from compare import psnr  # noqa: E402
log = open(app + '/harness.log').read()
presents = {int(m.group(1)): int(m.group(2)) for m in re.finditer(r'frame (\d+): Present .* presents (\d+)', log)}


def load(name):
    return np.fromfile(f'{app}/{name}', dtype=np.float16).reshape(-1, 4)


bad = 0
for t in range(frames):
    p = presents.get(t)
    if p is None:
        print(f'frame {t}: no present logged'); bad += 1; continue
    out = load(f'out/out-{t:02d}.rgba16f')
    if p == 1:
        same = np.array_equal(load(f'out/bbk-{t:02d}.rgba16f').view(np.uint16), out.view(np.uint16))
        print(f'frame {t}: presented once, buffer {"is" if same else "is NOT"} the upscaled frame')
        bad += not same
        continue
    real = load(f'out/bbk1-{t:02d}.rgba16f'); gen = load(f'out/bbk-{t:02d}.rgba16f')
    ref = load(f'ref/ref-gen-{t:02d}.rgba16f'); refreal = load(f'ref/ref-real-{t:02d}.rgba16f')
    prev = load(f'out/out-{t - 1:02d}.rgba16f')
    u = lambda a: a.view(np.uint16)
    real_ok = np.array_equal(u(real), u(out)); refreal_ok = np.array_equal(u(refreal), u(out))
    gen_exact = np.array_equal(u(gen), u(ref))
    mse = float(np.mean((gen[:, :3].astype(np.float32) - ref[:, :3].astype(np.float32)) ** 2))
    gen_ok = gen_exact or (mse > 0 and 10 * np.log10(1.0 / mse) >= 60)
    f = lambda a: a[:, :3].astype(np.float32)
    d_prev, d_cur, d_mid = (float(np.abs(f(gen) - f(x)).mean()) for x in (prev, out, (f(prev) + f(out)) / 2))
    nd = int((u(gen) != u(ref)).sum())
    print(f'frame {t}: presented twice | real frame restored exactly: {real_ok} | generated matches reference: {gen_ok}'
          f'{"" if gen_exact else f" ({nd} values differ, {10 * np.log10(1.0 / mse):.0f} dB)"} | reference restore exact: {refreal_ok} | '
          f'mean |gen - prev| {d_prev:.4f}, |gen - cur| {d_cur:.4f}, |gen - blend| {d_mid:.4f}')
    if mids:
        mid = np.fromfile(f'{mids}/mid-{t:02d}.rgba16f', dtype=np.float16).reshape(-1, 4)[:, :3].astype(np.float32)
        print(f'          vs the true in-between frame: generated {psnr(f(gen).reshape(720, 1280, 3), mid.reshape(720, 1280, 3)):.2f} dB, '
              f'blend {psnr(((f(prev) + f(out)) / 2).reshape(720, 1280, 3), mid.reshape(720, 1280, 3)):.2f} dB, '
              f'repeat previous {psnr(f(prev).reshape(720, 1280, 3), mid.reshape(720, 1280, 3)):.2f} dB')
    bad += not (real_ok and gen_ok and refreal_ok)
print('PASS' if bad == 0 else f'FAIL ({bad} frames)')
sys.exit(1 if bad else 0)
