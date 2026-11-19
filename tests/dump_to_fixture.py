#!/usr/bin/env python3
"""Converts frames captured with M4R_DUMP_UP (up-NN-{color,depth,motion}.raw + up-NN.txt) into a run_bbr fixture:
input-NN.rgba16f, depth-NN.r32f, mvlo-NN.rg16f (pixels, motion scale applied) and params-NN.txt (the frame's jitter).
usage: tests/dump_to_fixture.py <dump dir> <fixture dir>
SPDX-License-Identifier: GPL-3.0-only
"""
import glob, os, re, sys
import numpy as np

UNORM8 = {37, 43, 44, 50}          # R8G8B8A8_UNORM/SRGB, B8G8R8A8_UNORM/SRGB (sRGB formats are decoded by the hardware; not here)
FMT_F16X4, FMT_R32F, FMT_D32, FMT_D32S8, FMT_RG16F, FMT_B10G11R11 = 97, 100, 126, 130, 83, 122


def meta(path):
    m = {'image': {}}
    for line in open(path):
        k, *v = line.split()
        if k == 'image':
            m['image'][int(v[0])] = {'format': int(v[2]), 'w': int(v[4]), 'h': int(v[5])}
        else:
            m[k] = [float(x) for x in v]
    return m


def unpack_b10g11r11(u):
    def f(bits, mant):
        e = (bits >> mant) & 31; f_ = bits & ((1 << mant) - 1)
        return np.where(e == 0, f_ / (1 << mant) * 2.0 ** -14, (1 + f_ / (1 << mant)) * 2.0 ** (e.astype(np.float64) - 15))
    return np.stack([f(u & 0x7ff, 6), f((u >> 11) & 0x7ff, 6), f((u >> 22) & 0x3ff, 5)], -1)


def main():
    src, dst = sys.argv[1], sys.argv[2]
    os.makedirs(dst, exist_ok=True)
    for n, txt in enumerate(sorted(glob.glob(os.path.join(src, 'up-*.txt')))):
        base = txt[:-4]; m = meta(txt)
        cw, ch = int(m['content'][0]), int(m['content'][1])
        ci, di, mi = m['image'][0], m['image'][1], m['image'][2]
        raw = np.fromfile(base + '-color.raw', np.uint8)
        if ci['format'] in UNORM8:
            c = raw.reshape(ci['h'], ci['w'], 4).astype(np.float32) / 255.0
            if ci['format'] in (44, 50):
                c = c[..., [2, 1, 0, 3]]
        elif ci['format'] == FMT_F16X4:
            c = raw.view(np.float16).reshape(ci['h'], ci['w'], 4).astype(np.float32)
        elif ci['format'] == FMT_B10G11R11:
            c = np.concatenate([unpack_b10g11r11(raw.view(np.uint32).reshape(ci['h'], ci['w'])), np.ones((ci['h'], ci['w'], 1))], -1)
        else:
            sys.exit(f'color format {ci["format"]} not handled')
        z = np.fromfile(base + '-depth.raw', np.float32).reshape(di['h'], di['w'])
        if mi['format'] != FMT_RG16F:
            sys.exit(f'motion format {mi["format"]} not handled')
        mv = np.fromfile(base + '-motion.raw', np.float16).reshape(mi['h'], mi['w'], 2).astype(np.float32)
        mv *= np.array(m['mvscale'], np.float32)
        c[:ch, :cw].astype(np.float16).tofile(f'{dst}/input-{n:02}.rgba16f')
        z[:ch, :cw].astype(np.float32).tofile(f'{dst}/depth-{n:02}.r32f')
        mv[:ch, :cw].astype(np.float16).tofile(f'{dst}/mvlo-{n:02}.rg16f')
        with open(f'{dst}/params-{n:02}.txt', 'w') as f:
            f.write(f"jitter {m['jitter'][0]:.9g} {m['jitter'][1]:.9g}\n")
        print(f'frame {n}: color fmt {ci["format"]} {ci["w"]}x{ci["h"]}, content {cw}x{ch}, jitter {m["jitter"]}')


if __name__ == '__main__':
    main()
