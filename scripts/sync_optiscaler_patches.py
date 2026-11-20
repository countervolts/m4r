#!/usr/bin/env python3
"""Folds edits made in a patched OptiScaler checkout back into patches/optiscaler/*.patch.

The checkout is the pinned commit with the patches (and optiscaler/clang-cl.patch) applied. Each file that one patch owns has its
section in that patch replaced by the checkout's current diff against the pinned commit (new files: against nothing). Files that
both patches touch (Config.*, menu_common.cpp, the vcxproj, the ini) are not handled: edit those in the patch by hand.
runtime/m4r.h is vendored into patch 0001 and is copied into the checkout first.

usage: sync_optiscaler_patches.py <patched OptiScaler checkout>
SPDX-License-Identifier: GPL-3.0-only"""
import os, re, shutil, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OWNED = {
    '0001-metalfx-upscaler.patch': [
        'OptiScaler/OptiTypes.h', 'OptiScaler/OptiTypes.cpp', 'OptiScaler/upscalers/FeatureProvider_Dx12.cpp',
        'OptiScaler/upscalers/metalfx/MetalFXFeature_Dx12.cpp', 'OptiScaler/upscalers/metalfx/MetalFXFeature_Dx12.h',
        'OptiScaler/upscalers/metalfx/M4R_D3D12.h', 'OptiScaler/upscalers/metalfx/VkD3D_Interop.h',
        'OptiScaler/upscalers/metalfx/m4r.h'],
    '0002-metalfx-frame-generation.patch': [
        'OptiScaler/State.h', 'OptiScaler/hooks/FG_Hooks.cpp', 'OptiScaler/framegen/metalfx/MetalFXFG_Dx12.cpp',
        'OptiScaler/framegen/metalfx/MetalFXFG_Dx12.h'],
    '0003-metalfx-ray-reconstruction.patch': [
        'OptiScaler/upscalers/metalfx/MetalFXDenoiserFeature_Dx12.cpp', 'OptiScaler/upscalers/metalfx/MetalFXDenoiserFeature_Dx12.h'],
}


def current_diff(checkout, path):
    full = os.path.join(checkout, path)
    tracked = subprocess.run(['git', '-C', checkout, 'ls-files', '--error-unmatch', path], capture_output=True).returncode == 0
    if tracked:
        return subprocess.run(['git', '-C', checkout, 'diff', '--binary', 'HEAD', '--', path], capture_output=True, text=True).stdout
    d = subprocess.run(['git', 'diff', '--no-index', '--binary', '/dev/null', full], capture_output=True, text=True).stdout
    return d.replace('a/' + full.lstrip('/'), 'a/' + path).replace('b/' + full.lstrip('/'), 'b/' + path)


def main():
    checkout = os.path.abspath(sys.argv[1])
    shutil.copyfile(ROOT + '/runtime/m4r.h', checkout + '/OptiScaler/upscalers/metalfx/m4r.h')
    for patch, files in OWNED.items():
        pp = ROOT + '/patches/optiscaler/' + patch
        text = open(pp).read()
        changed = 0
        for path in files:
            new = current_diff(checkout, path)
            m = re.search(r'^diff --git a/' + re.escape(path) + r' .*?(?=^diff --git |\Z)', text, re.S | re.M)
            if not m:
                sys.exit(f'{patch} has no section for {path}')
            if new and new != m.group(0):
                text = text[:m.start()] + new + text[m.end():]
                changed += 1
                print('updated', path, 'in', patch)
        open(pp, 'w').write(text)
    print('done')


if __name__ == '__main__':
    main()
