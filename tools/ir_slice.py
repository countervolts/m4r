#!/usr/bin/env python3
"""Backward slice of the values a MetalFX AIR kernel stores (reading aid for the RE notes).
usage: ir_slice.py <kernel name> [store-regex]   (reads metalfx-dump/shaders-27/ir/<kernel>.ll; the .ll files are for reading only)
Prints, per matching store, the textures / buffers / intrinsics the stored value depends on. SPDX-License-Identifier: GPL-3.0-only"""
import re, sys, collections, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
name = sys.argv[1]
pat = re.compile(sys.argv[2] if len(sys.argv) > 2 else r'^store ')
s = open(f'{ROOT}/metalfx-dump/shaders-27/ir/{name}.ll').read()
i = s.index(f'define void @{name}')
body = s[i:]; body = body[:body.index('\n}\n')]
hdr = body.split('\n')[0]
# argument names from the kernel metadata, in order
meta = re.search(r'!air.kernel = !\{!(\d+)\}', s)
argnames = {}
k = re.search(r'^!\d+ = !\{void \(.*?\)\* @%s, !\d+, !(\d+)\}' % name, s, re.M)
arglist = re.findall(r'^!(\d+) = !\{(.*)\}$', s, re.M)
md = {a: b for a, b in arglist}
if k:
    kids = re.findall(r'!(\d+)', md[k.group(1)])
    for idx, kid in enumerate(kids):
        m = re.search(r'"air.arg_name", !"([^"]+)"', md.get(kid, ''))
        t = re.search(r'"air.(texture|buffer|thread_position_in_grid|thread_index_in_simdgroup|[a-z_]+)"', md.get(kid, ''))
        argnames[idx] = (m.group(1) if m else '?') + ('(' + t.group(1) + ')' if t else '')
params = re.findall(r'(?:^|, )(?:[^,%]*? )?%(\d+)(?=[,)])', hdr[hdr.index('('):])
lines = [l.strip() for l in body.split('\n')]
defs = {}
for l in lines:
    m = re.match(r'(%[\w.]+) = (.*)', l)
    if m: defs[m.group(1)] = m.group(2)
def deps(v, depth=14):
    seen = set(); tex = collections.Counter(); other = collections.Counter(); q = [(v, 0)]
    while q:
        x, d = q.pop()
        if x in seen or d > depth: continue
        seen.add(x)
        if re.fullmatch(r'%\d+', x) and x not in defs:
            idx = int(x[1:])
            if idx < len(argnames) or True:
                other['arg %s' % x] += 1
            continue
        if x not in defs: continue
        e = defs[x]
        m = re.search(r'@air\.(sample|read)_texture_2d\.[\w.]+\(.*?(%\d+)\b', e)
        if m: tex['%s %s' % (m.group(1), m.group(2))] += 1
        for mm in re.finditer(r'@(air\.[\w.]+|_Z\w+)\(', e):
            n = mm.group(1)
            if 'texture' not in n and not n.startswith('air.convert'): other[n[:40]] += 1
        for o in re.findall(r'%[\w.]+', e): q.append((o, d + 1))
    return tex, other
print('kernel args by position:', {i: argnames[i] for i in sorted(argnames)})
for n, l in enumerate(lines):
    if not pat.match(l): continue
    m = re.match(r'store \S+(?: \S+)? (%[\w.]+)', l) or re.search(r'(%[\w.]+)', l)
    val = re.search(r'(?:half|i8|float|<\d x half>) (%[\w.]+),', l)
    if not val: continue
    tex, other = deps(val.group(1))
    print('line %d: %s\n   textures %s\n   other %s' % (n, l[:90], dict(tex), dict(other.most_common(10))))
