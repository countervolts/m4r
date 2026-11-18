#!/usr/bin/env python3
"""Compare upscaled frames with the fixture's supersampled reference (PSNR, sRGB) and write PNGs.
usage: compare.py <plan dir> <fixture dir> <frames> <ow> <oh> <iw> <ih>"""
import sys
import numpy as np
from PIL import Image


def srgb(x):
    x = np.clip(np.nan_to_num(x), 0, 1)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055)


def psnr(a, b):
    mse = np.mean((srgb(a) - srgb(b)) ** 2)
    return 10 * np.log10(1 / max(mse, 1e-12))


def main():
    d, fx, n, ow, oh, iw, ih = sys.argv[1], sys.argv[2], int(sys.argv[3]), *map(int, sys.argv[4:8])
    ref = np.fromfile(f'{fx}/reference.rgba16f', np.float16).reshape(oh, ow, 4)[..., :3].astype(np.float32)
    for t in range(n):
        o = np.fromfile(f'{d}/out-{t:02}.rgba16f', np.float16).reshape(oh, ow, 4)[..., :3].astype(np.float32)
        Image.fromarray((srgb(o) * 255 + .5).astype(np.uint8)).save(f'{d}/out-{t:02}.png')
        nan = np.isnan(o).mean() * 100
        print(f'frame {t}: mean {np.nanmean(o):.4f} nan {nan:.2f}%' + (f'  PSNR vs reference {psnr(o, ref):.2f} dB' if t == n - 1 else ''))
    inp = np.fromfile(f'{fx}/input-{n - 1:02}.rgba16f', np.float16).reshape(ih, iw, 4)[..., :3].astype(np.float32)
    bil = np.asarray(Image.fromarray((srgb(inp) * 255).astype(np.uint8)).resize((ow, oh), Image.BILINEAR)).astype(np.float32) / 255
    mse = np.mean((bil - srgb(ref)) ** 2)
    print(f'bilinear baseline: {10 * np.log10(1 / mse):.2f} dB')
    Image.fromarray((srgb(ref) * 255 + .5).astype(np.uint8)).save(f'{d}/reference.png')


if __name__ == '__main__':
    main()
