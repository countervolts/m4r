#!/usr/bin/env python3
"""Pass lists from Ghidra-decompiled MFX encode functions and constructors.

usage: dispatch_extract.py enc <decompiled .c> [<decompiled .c> ...]    # a method: pipelines, bindings, dispatches
       dispatch_extract.py ctor <decompiled .c>                         # a constructor: pipelines, textures (format, size)
The decompiled files come from a Ghidra headless run of the MetalFX binary (see docs/metalfx-denoiser.md). The Metal calls are
recognised by their address in the dyld-cache stub area: 0x258004690 setComputePipelineState, 0x258015f70 setTexture,
0x258015ad0 setBuffer, 0x258004620 setBytes, 0x258001e20 dispatchThreads, 0x258014e60 dispatchThreadgroups, 0x258015b30
setConstantValue, 0x2580157f0 newTextureWithDescriptor, 0x258004e90/0x2580053e0/0x258004a30/0x2580052d0 pixelFormat/width/
height/usage.
SPDX-License-Identifier: GPL-3.0-only
"""
import os, re, struct, sys

_M = None


def cfstr(addr):
    """the string of the CFString at `addr` (shared-cache pointers: 0x180000000 + low 34 bits)"""
    global _M
    if _M is None:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from macho import MachO
        _M = MachO(os.environ.get('MFX_BIN', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..',
                                                           'metalfx-dump', 'bins-27', 'MetalFX')))
    o, sn = _M.off(int(addr, 16))
    if o is None:
        return '?'
    p = 0x180000000 + (struct.unpack_from('<Q', _M.d, o + 16)[0] & 0x3ffffffff)
    return _M.cstr(p) or '?'

CALLS = {'4690': 'pipe', '15f70': 'tex', '15ad0': 'buf', '4620': 'bytes', '1e20': 'dispatch', '14e60': 'dispatchTG',
         '15b30': 'const', '157f0': 'newtex', '4e90': 'fmt', '53e0': 'w', '4a30': 'h', '52d0': 'usage', '3960': 'newbuf'}


def stmts(path):
    body = open(path).read().split('\n')
    text = '\n'.join(l for l in body if not l.startswith('//') and not l.startswith('/*'))
    text = re.sub(r'\s+', ' ', text)
    return [s.strip() for s in re.split(r'(?<=[;{}])', text) if s.strip()]


def clean(e):
    e = re.sub(r'\*\(undefined8 \*\)', '@', e)
    e = re.sub(r'\(\(ulong\)([^()]*)\)', r'\1', e)
    e = re.sub(r'\(ulong\)', '', e)
    return e.strip()


def args(call):
    inner = call[call.index('(') + 1: call.rindex(')')]
    out, depth, cur = [], 0, ''
    for ch in inner:
        if ch in '([': depth += 1
        if ch in ')]': depth -= 1
        if ch == ',' and depth == 0:
            out.append(cur); cur = ''
        else:
            cur += ch
    out.append(cur)
    return [a.strip() for a in out]


def triple(name):
    """&local_b0 -> the three consecutive stack slots of a MTLSize: local_b0, uStack_a8, local_a0"""
    m = re.match(r'&?(local|uStack)_([0-9a-f]+)', name)
    if not m:
        return None
    b = int(m.group(2), 16)
    n = lambda off: ('local_%x' if (b - off) % 16 == 0 else 'uStack_%x')
    # Ghidra names 16-byte aligned slots local_X and the 8-byte-offset ones uStack_X, descending addresses
    return ['local_%x' % b if m.group(1) == 'local' else 'uStack_%x' % b,
            'uStack_%x' % (b - 8) if m.group(1) == 'local' else 'local_%x' % (b - 8),
            'local_%x' % (b - 16) if m.group(1) == 'local' else 'uStack_%x' % (b - 16)]


def enc(path):
    print('==', path.split('/')[-1])
    pipe = None
    indent = ''
    alias, assign = {}, {}
    ss = stmts(path)
    for s in ss:
        a = re.match(r'(\w+) = func_0x0002580592c0\((\w+)\);', s)
        if a:
            alias[a.group(1)] = a.group(2)
        a = re.match(r'(?:\(\w+\) )?(\w+) = (.*);$', s)
        if a and re.match(r'(local|uStack)_', a.group(1)):
            assign[a.group(1)] = a.group(2)
    res = lambda e: re.sub(r'\b(uVar\d+|auVar\d+\._0_8_)\b', lambda m: alias.get(m.group(1), m.group(1)), e)
    for s in ss:
        m = re.search(r'func_0x000258(\w+)\(', s)
        if re.match(r'(do|while|for|if|else)\b', s):
            print(indent + s[:110])
        if not m or m.group(1).lstrip('0') not in CALLS:
            continue
        kind = CALLS[m.group(1).lstrip('0')]
        a = args(s[m.start():])
        if kind == 'pipe':
            print('  pipeline', clean(a[2]))
        elif kind == 'tex':
            print('    tex[%s] <- %s' % (a[3].rstrip(');'), res(clean(a[2]))))
        elif kind == 'buf':
            print('    buf[%s] <- %s +%s' % (a[4].rstrip(');'), res(clean(a[2])), a[3]))
        elif kind == 'bytes':
            print('    bytes[%s] len %s <- %s' % (a[4].rstrip(');'), a[3], clean(a[2])))
        elif kind in ('dispatch', 'dispatchTG'):
            g, t = triple(a[2]), triple(a[3].rstrip(');'))
            show = lambda ns: ' x '.join(res(assign.get(n, '?'))[:60] for n in ns) if ns else '?'
            print('  %s grid=[%s] tg=[%s]' % (kind, show(g), show(t)))
        elif kind == 'const':
            print('    const[%s] type %s <- %s' % (a[4].rstrip(');'), a[3], clean(a[2])))


def ctor(path):
    fmt = w = h = usage = None
    for s in stmts(path):
        m = re.search(r'func_0x000258(\w+)\(', s)
        if re.search(r'createComputePipeline', s):
            n = re.search(r'cfstringStruct_(\w+)', s)
            print('pipeline %s' % (cfstr(n.group(1)) if n else '?'), end=' ')
        if m and m.group(1).lstrip('0') in CALLS:
            kind = CALLS[m.group(1).lstrip('0')]; a = args(s[m.start():])
            if kind == 'fmt': fmt = a[2].rstrip(');')
            elif kind == 'w': w = clean(a[2].rstrip(');'))
            elif kind == 'h': h = clean(a[2].rstrip(');'))
            elif kind == 'usage': usage = a[2].rstrip(');')
            elif kind == 'newtex': print('texture fmt=%s w=%s h=%s usage=%s ->' % (fmt, w, h, usage), end=' ')
            elif kind == 'newbuf': print('buffer len=%s ->' % a[2], end=' ')
            elif kind == 'const': print('const[%s]=%s' % (a[4].rstrip(');'), clean(a[2])))
        t = re.match(r'\*\(undefined8 \*\)\((param_1 \+ (?:0x[0-9a-f]+|\d+))\) = (\w+);', s)
        if t and not t.group(2) in ('0',):
            print('   store', t.group(1).replace('param_1 + ', 'this+'), '=', t.group(2))


if __name__ == '__main__':
    mode = sys.argv[1]
    for p in sys.argv[2:]:
        (enc if mode == 'enc' else ctor)(p)
