#!/usr/bin/env python3
"""Patch 0001 vendors runtime/m4r.h as OptiScaler/upscalers/metalfx/m4r.h. After changing the header, run this to replace that
file's section in the patch (and update the copy in a patched checkout when one is given).
usage: sync_optiscaler_header.py [patched OptiScaler checkout]
SPDX-License-Identifier: GPL-3.0-only"""
import os, re, shutil, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATCH = ROOT + '/patches/optiscaler/0001-metalfx-upscaler.patch'
TARGET = 'OptiScaler/upscalers/metalfx/m4r.h'

new = subprocess.run(['git', 'diff', '--no-index', '--binary', '/dev/null', ROOT + '/runtime/m4r.h'], capture_output=True, text=True).stdout
new = new.replace('a/' + ROOT.lstrip('/') + '/runtime/m4r.h', 'a/' + TARGET).replace('b/' + ROOT.lstrip('/') + '/runtime/m4r.h', 'b/' + TARGET)
text = open(PATCH).read()
m = re.search(r'^diff --git a/' + re.escape(TARGET) + r' .*?(?=^diff --git |\Z)', text, re.S | re.M)
if not m:
    sys.exit('the patch has no section for ' + TARGET)
if new == m.group(0):
    print('patch already up to date')
else:
    open(PATCH, 'w').write(text[:m.start()] + new + text[m.end():])
    print('updated', PATCH)
if len(sys.argv) > 1:
    shutil.copyfile(ROOT + '/runtime/m4r.h', os.path.join(sys.argv[1], TARGET))
    print('copied into', sys.argv[1])
