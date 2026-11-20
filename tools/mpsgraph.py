#!/usr/bin/env python3
"""Read the op graph of a MetalFX .mtlpackage/library.mpsgraphpackage/model_N.mpsgraph (MLIR bytecode v6, `mps` dialect).

usage: mpsgraph.py <MetalFX Resources dir> <package name: dbf|ubf|brnet|...> [model index] [--dat emit_xxx_constants.dat]

No mlir-opt is needed (the `mps` dialect is not available outside Apple's stack). What is decoded:
ops with result/operand value ids, tensor shapes (zigzag varints, -1 = dynamic), constants (inline dense data or
resource handle -> tmp name -> resources.bin offset), and for conv layers the weight/bias location. What is NOT decoded:
`mps` property attributes (strides, dilation, resize mode, pool window). Their attribute table is printed raw (`--raw`).
Apple's own fused kernels (BR_Net/FusedNetwork/*) take those from their descriptors, so the graph is a cross-check, not
the source of truth for them.

With --dat, each conv's weight/bias is searched in the fused path's constants file; the offsets found are the ones the
DBFNet/UBFNet cell constructors use (checked for all 15 DBF and 14 of 15 UBF layers).
"""
import os, pickle, plistlib, struct, sys


def varint(d, p):
    b = d[p]
    if b & 1:
        return b >> 1, p + 1
    if b == 0:
        return int.from_bytes(d[p + 1:p + 9], 'little'), p + 9
    n = (b & -b).bit_length() - 1
    return int.from_bytes(d[p:p + n + 1], 'little') >> (n + 1), p + n + 1


def zigzag(v):
    return (v >> 1) ^ -(v & 1)


def sections(d):
    assert d[:4] == b'ML\xefR'
    ver, p = varint(d, 4)
    p = d.index(0, p) + 1
    secs = {}
    while p < len(d):
        b = d[p]; p += 1
        ln, p = varint(d, p)
        if b & 0x80:
            al, p = varint(d, p)
            while p % al:
                p += 1
        secs[b & 0x7f] = d[p:p + ln]
        p += ln
    return ver, secs


class Graph:
    def __init__(self, path):
        d = open(path, 'rb').read()
        self.version, self.secs = sections(d)
        s = self.secs[0]
        self.strs = [x.decode('latin1') for x in s[s.index(b'mps\0'):].split(b'\0')]
        self._attrs_types()
        self._props()
        self.ops = self._ir()

    def _attrs_types(self):
        o = self.secs[3]
        na, p = varint(o, 0)
        nt, p = varint(o, p)

        def groups(cnt, p):
            ents = []
            while len(ents) < cnt:
                dia, p = varint(o, p)
                k, p = varint(o, p)
                for _ in range(k):
                    v, p = varint(o, p)
                    ents.append((dia, v & 1, v >> 1))
            return ents, p
        A, p = groups(na, p)
        T, p = groups(nt, p)
        a = self.secs[2]
        pos = 0
        out = []
        for e in A + T:
            out.append(a[pos:pos + e[2]])
            pos += e[2]
        self.A, self.T = out[:na], out[na:]

    def _props(self):
        pd = self.secs[8]
        n, p = varint(pd, 0)
        self.props = []
        for _ in range(n):
            try:
                k, p = varint(pd, p)
                l = []
                for _ in range(k):
                    v, p = varint(pd, p)
                    l.append(v)
                self.props.append(l)
            except IndexError:
                break

    def shape(self, i):
        b = self.T[i]
        if b[0] != 0x1b:       # RankedTensorType
            return None
        c, p = varint(b, 0)
        n, p = varint(b, p)
        dims = []
        for _ in range(n):
            v, p = varint(b, p)
            x = zigzag(v)
            dims.append(-1 if x < -(1 << 62) else x)
        return dims

    def _ir(self):
        d = self.secs[4]
        names = self.strs[2:self.strs.index('return') + 1]
        # func body starts at the first `constant` op after the module/func headers (always after a 0x00 byte here)
        pos = d.find(bytes([0, 1, 0x42])) + 1
        ops, nval = [], 1       # %0 is the function argument
        while pos < len(d):
            n, pos = varint(d, pos)
            mask = d[pos]; pos += 1
            _, pos = varint(d, pos)          # location
            o = {'op': names[n], 'res': [], 'opr': []}
            if mask & 1:
                o['attrs'], pos = varint(d, pos)
            if mask & 0x40:
                o['prop'], pos = varint(d, pos)
            if mask & 2:
                k, pos = varint(d, pos)
                for _ in range(k):
                    t, pos = varint(d, pos)
                    o['res'].append((nval, t)); nval += 1
            if mask & 4:
                k, pos = varint(d, pos)
                for _ in range(k):
                    v, pos = varint(d, pos)
                    o['opr'].append(v)
            ops.append(o)
        return ops

    def const_value(self, op, res=None, offs=None):
        """-> ('inline', bytes) or ('resource', tmp name, resources.bin offset)"""
        a = self.A[self.props[op['prop']][0]]
        if a[0] == 0x25:                      # DenseIntOrFPElementsAttr
            _, q = varint(a, 1)
            ln, q = varint(a, q)
            return 'inline', a[q:q + ln]
        h = varint(a, 2)[0]
        nm = self.strs[self.strs.index('tmp') + h]
        return 'resource', nm, (offs or {}).get(nm)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    res_dir, name = args[0], args[1]
    idx = int(args[2]) if len(args) > 2 else 0
    pk = os.path.join(res_dir, name + '.mtlpackage', 'library.mpsgraphpackage')
    g = Graph(os.path.join(pk, 'model_%d.mpsgraph' % idx))
    man = plistlib.load(open(os.path.join(pk, 'manifest.plist'), 'rb'))['Package Version']
    offs = list(man.values())[0]['Resource Offsets']
    rb = open(os.path.join(pk, 'resources.bin'), 'rb').read()
    dat = None
    if '--dat' in sys.argv:
        dat = open(sys.argv[sys.argv.index('--dat') + 1], 'rb').read()
    consts = {}
    for o in g.ops:
        if o['op'] == 'constant':
            consts[o['res'][0][0]] = (g.shape(o['res'][0][1]), g.const_value(o, offs=offs))
    adds = {a['opr'][0]: a for a in g.ops if a['op'] == 'add'}
    for o in g.ops:
        r = ','.join('%%%d:%s' % (v, g.shape(t)) for v, t in o['res'])
        extra = ''
        if o['op'] == 'constant':
            shp, cv = consts[o['res'][0][0]]
            extra = '%s %s' % (cv[0], cv[1] if cv[0] == 'resource' else cv[1].hex()[:32])
            if cv[0] == 'resource' and cv[2] is not None:
                extra += ' @resources.bin+%d' % cv[2]
        elif o['op'] == 'conv_2d' and dat is not None:
            w = consts[o['opr'][1]]
            hits = []
            if w[1][0] == 'resource' and w[1][2] is not None:
                n = 1
                for x in w[0]:
                    n *= x
                for hl in (3, 4):
                    blob = rb[w[1][2] + hl:w[1][2] + hl + 2 * n]
                    if dat.find(blob) >= 0:
                        hits.append(dat.find(blob)); break
            extra = 'weights in .dat @%s' % ([hex(h) for h in hits] or '?')
        print(o['op'], r, '<-', ['%%%d' % x for x in o['opr']], extra)


if __name__ == '__main__':
    main()
