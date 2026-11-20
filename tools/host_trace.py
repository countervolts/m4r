#!/usr/bin/env python3
"""Per-dispatch trace of an MFX encode function from tools/annotate.py output.

usage: annotate.py <MetalFX> <regex> <a2s> | host_trace.py
Follows registers symbolically (ldr/mov/str on sp) and prints, for every dispatch, the pipeline expression, the textures
(slot <- source), the buffers, the setBytes (index, length) and the grid. Sources are `argN` (incoming register xN),
`[expr+off]` (a load, e.g. [arg0+0xc0] = an ivar of this) or `stk(off)`. Names of the pipelines and textures come from the
class constructor (see docs/metalfx-denoiser.md); kernel argument names from the AIR metadata (!air.arg_name).
SPDX-License-Identifier: GPL-3.0-only
"""
import re, sys

STUB = {'setComputePipelineState:': 'pipe', 'setTexture:atIndex:': 'tex', 'setBuffer:offset:atIndex:': 'buf',
        'setBytes:length:atIndex:': 'bytes', 'dispatchThreads:threadsPerThreadgroup:': 'dispatchT',
        'dispatchThreadgroups?:threadsPerThreadgroup:': 'dispatchG', 'pushDebugGroup:': 'group'}


def run(lines):
    reg = {'x%d' % i: 'arg%d' % i for i in range(8)}
    mem = {}
    imm = {}
    cur = {'pipe': None, 'tex': {}, 'buf': {}, 'bytes': {}}
    out = []

    def val(r):
        r = r.replace('w', 'x')
        return reg.get(r, r)
    for l in lines:
        m = re.match(r'(\w+)\s+(\S+)\s+(.*?)(?:\s*;\s*(.*))?$', l.rstrip())
        if not m:
            continue
        ad, op, args, note = m.groups()
        a = [x.strip() for x in re.split(r',\s*(?![^\[]*\])', args)] if args else []
        if op == 'mov' and len(a) == 2:
            d = a[0].replace('w', 'x')
            if a[1].startswith('#'):
                imm[d] = int(a[1][1:], 0); reg[d] = a[1][1:]
            elif a[1] in ('xzr', 'wzr'):
                imm[d] = 0; reg[d] = '0'
            else:
                reg[d] = val(a[1]); imm.pop(d, None)
        elif op in ('ldr', 'ldrb', 'ldrh', 'ldur') and len(a) == 2 and a[0][0] in 'xw':
            d = a[0].replace('w', 'x')
            mm = re.match(r'\[(\w+)(?:, #(-?0x[0-9a-f]+|-?\d+))?\]!?', a[1])
            if mm:
                b, off = mm.group(1), int(mm.group(2) or '0', 0)
                if b == 'sp':
                    reg[d] = mem.get(off, 'stk(%#x)' % off)
                else:
                    reg[d] = '[%s+%#x]' % (val(b), off)
            else:
                reg[d] = '?'
            imm.pop(d, None)
        elif op in ('str', 'stur') and len(a) == 2:
            mm = re.match(r'\[sp(?:, #(0x[0-9a-f]+))?\]', a[1])
            if mm:
                mem[int(mm.group(1) or '0', 0)] = val(a[0])
        elif op == 'cset' or op in ('add', 'sub', 'lsr', 'lsl', 'and', 'orr', 'csel', 'ubfx', 'ubfiz', 'sxtw') and a:
            d = a[0].replace('w', 'x')
            if op == 'add' and len(a) == 3 and a[2].startswith('#') and not a[1].startswith('sp'):
                reg[d] = '(%s+%s)' % (val(a[1]), a[2][1:])
            else:
                reg[d] = '?'
            imm.pop(d, None)
        elif op == 'bl' and note in STUB.values() or (op == 'bl' and note in STUB):
            kind = STUB.get(note, note)
            if kind == 'pipe':
                cur['pipe'] = val('x2')
            elif kind == 'tex':
                cur['tex'][val('x3')] = val('x2')
            elif kind == 'buf':
                cur['buf'][val('x4')] = val('x2')
            elif kind == 'bytes':
                cur['bytes'][val('x4')] = val('x3')
            elif kind in ('dispatchT', 'dispatchG'):
                out.append(dict(cur, kind=kind))
                cur = {'pipe': cur['pipe'], 'tex': {}, 'buf': {}, 'bytes': {}}   # pipeline stays until changed
        elif op == 'bl' and note and 'retain' in note:
            pass
        elif op == 'bl':
            reg['x0'] = '?'
    return out


def main():
    lines = sys.stdin.read().split('\n')
    chunks, cur = [], None
    for l in lines:
        if l.startswith('====='):
            cur = [l, []]; chunks.append(cur)
        elif cur is not None:
            cur[1].append(l)
    for head, body in chunks:
        print(head[:110])
        for i, d in enumerate(run(body)):
            print('  dispatch %d: pipe=%s' % (i, d['pipe']))
            print('    tex   ' + ' '.join('%s<-%s' % (k, v) for k, v in sorted(d['tex'].items(), key=lambda kv: int(kv[0]) if kv[0].isdigit() else 99)))
            print('    buf   ' + ' '.join('%s<-%s' % (k, v) for k, v in sorted(d['buf'].items())))
            print('    bytes ' + ' '.join('%s:len %s' % (k, v) for k, v in sorted(d['bytes'].items())))


if __name__ == '__main__':
    main()
