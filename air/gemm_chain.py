# SPDX-License-Identifier: GPL-3.0-only
"""GEMM chains in WMMA layout.

Apple's MXU GEMM (llvm.agx3.fgemm16x16x16 / llvm.agx3.igemm) keeps its accumulator in Apple's lane layout; m4r computes it
with RDNA3 WMMA, whose accumulator layout differs, so every call converted C to WMMA's layout and D back through LDS. In
MetalFX's network kernels the accumulator of a GEMM is mostly the result of the previous GEMM (a loop over taps and channel
blocks: phi -> gemm -> phi), so this pass keeps those accumulators in WMMA's layout:

    %d = gemm(A, B, %c)   ->   %m4rw.d = wgemm(A, B, cw)        cw: WMMA-layout value of %c
                               %d = w2a(%m4rw.d)                 (removed when nothing else reads %d)

cw is zeroinitializer for a zero accumulator (the same in every layout), the WMMA value of a GEMM result, a phi rebuilt over
WMMA values when every incoming value is one of those, and otherwise a2w(%c). The arithmetic is unchanged: the same WMMA
instruction with C as its accumulator (air/intrinsics.py wgemm_f / wgemm_i), so results are bit-identical.

Runs on optimised IR (inlined, scalarised) before the AIR / AGX intrinsics get bodies (m4r_air.translate, step 7b).
"""
import os
import re

FGEMM = re.compile(r'^(\s*)(%[-\w.]+) = (?:tail )?call (?:fast |nnan |ninf |nsz |arcp |contract |afn |reassoc )*<8 x float> '
                   r'@llvm\.agx3\.fgemm16x16x16\.v8f16\.v8f16\((.*)\)(.*)$')
IGEMM = re.compile(r'^(\s*)(%[-\w.]+) = (?:tail )?call <8 x i32> @llvm\.agx3\.igemm\.v8i32\.i64\.i64\.v8i32\((.*)\)(.*)$')
PHI = re.compile(r'^(\s*)(%[-\w.]+) = phi (?:fast |nnan |ninf |nsz |arcp |contract |afn |reassoc )*(<8 x float>|<8 x i32>) (.*)$')
LABEL = re.compile(r'^([-\w.$"]+):')
TYPES = {'f': '<8 x float>', 'i': '<8 x i32>'}
SUFFIX = {'f': 'v8f32', 'i': 'v8i32'}


def split_top(s):
    out, depth, cur = [], 0, ''
    for ch in s:
        if ch in '<{[(':
            depth += 1
        elif ch in '>}])':
            depth -= 1
        if ch == ',' and depth == 0:
            out.append(cur.strip()); cur = ''
        else:
            cur += ch
    if cur.strip():
        out.append(cur.strip())
    return out


def arg_value(a):
    """'<8 x half> noundef %590' -> ('<8 x half>', '%590')"""
    v = a.split()[-1]
    t = a[:a.rindex(v)].strip()
    t = re.sub(r'\s+(noundef|signext|zeroext|nonnull|immarg)\b', '', ' ' + t).strip()
    return t, v


def wname(v):
    return '%m4rw.' + v[1:]


DIRECT_B = False   # set by m4r_air.translate: the kernel's weights are block-major tiles (function constant blockMajorWeights)
PERMUTED = []   # parameters of the last rewritten kernel whose weight tiles are read in WMMA order (set by rewrite)


def find_direct_b(lines, params):
    """Integer GEMMs whose B operand is a weight tile read as two 4-byte loads at P and P + 128 from a kernel buffer argument
    that nothing else reads: {gemm result: P}, and the argument indices. Apple stores those tiles [k][col] (16 x 16 bytes; a
    lane's offset is c0 + 16 kk); rows 0-7 at P, rows 8-15 a constant distance further (128 or more bytes). The plan generator
    transposes each 128-byte half ([8 k][16 col] -> [16 col][8 k]), so a lane loads its WMMA column as two 8-byte loads."""
    defs = {}
    for ln in lines:
        m = re.match(r'^\s*(%[-\w.]+) = (.*)$', ln)
        if m:
            defs[m.group(1)] = m.group(2)

    def root(v, seen=None):
        """the kernel argument a pointer is derived from through geps and pointer phis, or None"""
        seen = set() if seen is None else seen
        while v in defs:
            if v in seen:
                return 'cycle'
            seen.add(v)
            d = defs[v]
            m = re.match(r'getelementptr (?:inbounds )?(?:nuw )?(?:nusw )?i8, ptr addrspace\(1\) (%[-\w.]+), ', d)
            if m:
                v = m.group(1)
                continue
            if d.startswith('phi ptr addrspace(1) '):
                rs = {root(x, seen) for x in re.findall(r'\[\s*(%[-\w.]+)\s*,', d)} - {'cycle'}
                return rs.pop() if len(rs) == 1 and None not in rs else None
            return None
        return v if v in params else None
    LOAD4 = re.compile(r'(?:load <4 x i8>, ptr addrspace\(1\) (%[-\w.]+), align \d+|tail call <4 x i8> @llvm\.agx3\.load\.with\.emask\.global\.v4i8\('
                       r'ptr addrspace\(1\) (?:noundef )?(?:nonnull )?(%[-\w.]+), i16 (?:noundef )?(?:zeroext )?15, i16 (?:noundef )?(?:zeroext )?15,)')

    def load_ptr(v):
        m = LOAD4.match(defs.get(v, ''))
        return (m.group(1) or m.group(2)) if m else None

    cands = {}
    loads = set()
    for ln in lines:
        m = IGEMM.match(ln)
        if not m:
            continue
        args = split_top(m.group(3))
        _, bv = arg_value(args[6])
        mb = re.match(r'bitcast <8 x i8> (%[-\w.]+) to i64$', defs.get(bv, ''))
        ms = mb and re.match(r'shufflevector <4 x i8> (%[-\w.]+), <4 x i8> (%[-\w.]+), <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>$',
                             defs.get(mb.group(1), ''))
        if not ms:
            continue
        l1, l2 = ms.group(1), ms.group(2)
        p, q = load_ptr(l1), load_ptr(l2)
        if not (p and q):
            continue

        def rel(q, p):   # q = p + constant through constant i8 geps -> the constant, else None
            off = 0
            while q != p:
                mm = re.match(r'getelementptr (?:inbounds )?(?:nuw )?(?:nusw )?i8, ptr addrspace\(1\) (%[-\w.]+), i64 (-?\d+)$', defs.get(q, ''))
                if not mm:
                    return None
                q, off = mm.group(1), off + int(mm.group(2))
            return off

        def top(v):      # outermost gep: (base, variable index, constant)
            mm = re.match(r'getelementptr (?:inbounds )?(?:nuw )?(?:nusw )?i8, ptr addrspace\(1\) (%[-\w.]+), i64 (\S+)$', defs.get(v, ''))
            if not mm:
                return None
            idx = mm.group(2)
            if re.fullmatch(r'-?\d+', idx):
                return mm.group(1), None, int(idx)
            ma = re.match(r'(?:add|or)(?: disjoint)?(?: nuw)?(?: nsw)? i64 (%[-\w.]+), (-?\d+)$', defs.get(idx, ''))
            return (mm.group(1), ma.group(1), int(ma.group(2))) if ma else (mm.group(1), idx, 0)
        half = rel(q, p)
        if half is None:
            tp, tq = top(p), top(q)
            half = tq[2] - tp[2] if tp and tq and tp[0] == tq[0] and tp[1] == tq[1] else None
        if half is None or half < 128 or half % 128:
            continue
        r = root(p)
        if r is None or r == 'cycle':
            continue
        cands[m.group(2)] = (p, r, half)
        loads |= {l1, l2}
    # every use of a pointer derived from such an argument must be a gep or one of those loads
    ok_roots = set(r for _, r, _ in cands.values())
    derived = {r: {r} for r in ok_roots}
    changed = True
    while changed:
        changed = False
        for v, d in defs.items():
            m = re.match(r'getelementptr [^,]*, ptr addrspace\(1\) (%[-\w.]+), ', d)
            srcs = [m.group(1)] if m else (re.findall(r'\[\s*(%[-\w.]+)\s*,', d) if d.startswith('phi ptr addrspace(1) ') else [])
            for r in ok_roots:
                if any(x in derived[r] for x in srcs) and v not in derived[r]:
                    derived[r].add(v); changed = True
    for r in list(ok_roots):
        for ln in lines:
            m = re.match(r'^\s*(%[-\w.]+) = (.*)$', ln)
            body = m.group(2) if m else ln
            name = m.group(1) if m else None
            for v in derived[r]:
                if re.search(re.escape(v) + r'(?![-\w.])', body):
                    if name in derived[r] and (body.startswith('getelementptr') or body.startswith('phi ptr addrspace(1)')):
                        continue
                    if name in loads:
                        continue
                    ok_roots.discard(r)
                    break
            if r not in ok_roots:
                break
    return {g: (p, h) for g, (p, r, h) in cands.items() if r in ok_roots}, sorted(params.index(r) for r in ok_roots)


def rewrite_function(lines, used, params=()):
    gemms = {}     # result -> kind
    phis = {}      # result -> (kind, [(value, block)])
    for ln in lines:
        m = FGEMM.match(ln) or IGEMM.match(ln)
        if m:
            gemms[m.group(2)] = 'f' if FGEMM.match(ln) else 'i'
            continue
        m = PHI.match(ln)
        if m:
            kind = 'f' if m.group(3) == '<8 x float>' else 'i'
            inc = re.findall(r'\[\s*([^,\]]+?)\s*,\s*(%[-\w.$"]+)\s*\]', m.group(4))
            phis[m.group(2)] = (kind, inc)
    if not gemms:
        return lines
    direct, perm = find_direct_b(lines, list(params)) if DIRECT_B and os.environ.get('M4R_DIRECT_B', '1') != '0' else ({}, [])
    PERMUTED.extend(perm)
    # phis rebuilt in WMMA layout: incoming values that are GEMM results, zero or other rebuilt phis are used as they are; any
    # other incoming value is converted once at the end of its predecessor block (typically the value entering a loop).
    # A phi is rebuilt when at least one incoming value is a GEMM result or a rebuilt phi (greatest fixpoint).
    conv = set(phis)
    changed = True
    while changed:
        changed = False
        for p in list(conv):
            k, inc = phis[p]
            good = any(gemms.get(v) == k or (v in conv and phis[v][0] == k) for v, _ in inc)
            if not good:
                conv.discard(p); changed = True
    edge_conv = {}      # predecessor block -> [(value, kind)] needing a2w at its end
    for p in conv:
        k, inc = phis[p]
        for v, bb in inc:
            if not (v == 'zeroinitializer' or gemms.get(v) == k or (v in conv and phis[v][0] == k)):
                edge_conv.setdefault(bb, []).append((v, k))

    out, n = [], [0]
    pending = []

    def edge_name(v, bb):
        return '%m4re.' + bb.strip('%"').replace('"', '') + '.' + v.strip('%"')        # w2a of rebuilt phis, emitted after the block's last phi

    def wval(v, kind, ind, pre):
        if v == 'zeroinitializer':
            return v
        if gemms.get(v) == kind or (v in conv and phis[v][0] == kind):
            return wname(v)
        n[0] += 1
        t = TYPES[kind]
        name = f'%m4ra.{n[0]}'
        pre.append(f'{ind}{name} = call {t} @llvm.agx3.a2w.{SUFFIX[kind]}({t} {v})')
        used.add(('a2w', kind))
        return name

    for ln in lines:
        if pending and not PHI.match(ln) and not LABEL.match(ln.strip()) and ln.strip():
            out += pending
            pending = []
        mf, mi = FGEMM.match(ln), IGEMM.match(ln)
        if mf or mi:
            kind = 'f' if mf else 'i'
            m = mf or mi
            ind, d, args = m.group(1), m.group(2), split_top(m.group(3))
            t = TYPES[kind]
            pre = []
            if kind == 'f':
                (_, a), (_, b), (_, c) = arg_value(args[0]), arg_value(args[2]), arg_value(args[4])
                cw = wval(c, kind, ind, pre)
                call = f'call {t} @llvm.agx3.wgemm.f(<8 x half> {a}, <8 x half> {b}, {t} {cw})'
                used.add(('wgemm', kind))
            else:
                (_, a), (_, ta), (_, b), (_, tb), (_, c) = (arg_value(args[i]) for i in (4, 5, 6, 7, 8))
                cw = wval(c, kind, ind, pre)
                if d in direct:
                    call = f'call {t} @llvm.agx3.wgemm.ib(i64 {a}, i16 {ta}, ptr addrspace(1) {direct[d][0]}, i16 {tb}, {t} {cw}, i32 {direct[d][1]})'
                    used.add(('wgemmb', kind))
                else:
                    call = f'call {t} @llvm.agx3.wgemm.i(i64 {a}, i16 {ta}, i64 {b}, i16 {tb}, {t} {cw})'
                    used.add(('wgemm', kind))
            used.add(('w2a', kind))
            out += pre
            out.append(f'{ind}{wname(d)} = {call}')
            out.append(f'{ind}{d} = call {t} @llvm.agx3.w2a.{SUFFIX[kind]}({t} {wname(d)})')
            continue
        m = PHI.match(ln)
        if m and m.group(2) in conv:
            ind, p = m.group(1), m.group(2)
            kind, inc = phis[p]
            t = TYPES[kind]
            def inval(v, bb):
                if v == 'zeroinitializer' or gemms.get(v) == kind or (v in conv and phis[v][0] == kind):
                    return v if v == 'zeroinitializer' else wname(v)
                return edge_name(v, bb)
            parts = ', '.join(f'[ {inval(v, bb)}, {bb} ]' for v, bb in inc)
            out.append(f'{ind}{wname(p)} = phi {t} {parts}')
            pending.append(f'{ind}{p} = call {t} @llvm.agx3.w2a.{SUFFIX[kind]}({t} {wname(p)})')
            used.add(('w2a', kind))
            continue
        out.append(ln)
    out += pending
    # a2w of values entering rebuilt phis, at the end of the predecessor block (before its terminator)
    if edge_conv:
        blocks, cur = {}, None   # block name -> index of its terminator line in out
        first = True
        for idx, ln in enumerate(out):
            mm = LABEL.match(ln.strip()) if ln and not ln.startswith(' ') else None
            if mm:
                cur = '%' + mm.group(1); first = False
            elif ln.strip() and not ln.strip().startswith(';'):
                if re.match(r'^\s*(br|switch|ret|unreachable|indirectbr|callbr)\b', ln):
                    blocks[cur if cur else '__entry'] = idx
        inserts = {}
        for bb, vals in edge_conv.items():
            idx = blocks.get(bb)
            if idx is None and len(blocks):
                idx = blocks.get('__entry')
            seen = set()
            for v, k in vals:
                if (v, bb) in seen:
                    continue
                seen.add((v, bb))
                t = TYPES[k]
                inserts.setdefault(idx, []).append(f'  {edge_name(v, bb)} = call {t} @llvm.agx3.a2w.{SUFFIX[k]}({t} {v})')
                used.add(('a2w', k))
        new = []
        for idx, ln in enumerate(out):
            new += inserts.get(idx, [])
            new.append(ln)
        out = new
    # move each conversion to Apple's layout to the blocks that read it when none of its readers is in its own block or a phi
    # (a loop's accumulator read after the loop is then converted once, not in every iteration); the WMMA value it converts is
    # defined at the same point, so it dominates those readers too
    def block_of(lines_):
        res, cur = [], '__entry'
        for ln in lines_:
            mm = LABEL.match(ln.strip()) if ln and not ln.startswith(' ') else None
            if mm:
                cur = '%' + mm.group(1)
            res.append(cur)
        return res
    blk = block_of(out)
    defs = {}
    for idx, ln in enumerate(out):
        mm = re.match(r'^(\s*)(%[-\w.]+) = call (<8 x (?:float|i32)>) @llvm\.agx3\.w2a\.(\w+)\((.*)\)$', ln)
        if mm:
            defs[mm.group(2)] = (idx, mm.group(1), mm.group(3), mm.group(4), mm.group(5))
    moved = {}
    for d, (idx, ind, t, suf, arg) in defs.items():
        pat = re.compile(re.escape(d) + r'(?![-\w.])')
        users = [i for i, ln in enumerate(out) if i != idx and pat.search(ln)]
        if not users or any(blk[i] == blk[idx] or PHI.match(out[i]) or ' phi ' in out[i] for i in users):
            continue
        firsts = {}
        for i in users:
            firsts.setdefault(blk[i], i)
        moved[idx] = None
        for bb, i in firsts.items():
            nm = f'%m4rs.{d[1:]}.{len(moved)}.{len(firsts)}'
            moved.setdefault(i, [])
            moved[i] = (moved[i] or []) + [f'{ind}{nm} = call {t} @llvm.agx3.w2a.{suf}({arg})']
            for j in users:
                if blk[j] == bb:
                    out[j] = pat.sub(nm, out[j])
    if moved:
        new = []
        for idx, ln in enumerate(out):
            ins = moved.get(idx, 'keep')
            if ins is None:
                continue             # the original conversion, now in its readers' blocks
            if ins != 'keep':
                new += ins
            new.append(ln)
        out = new
    # drop conversions to Apple's layout that nothing reads
    while True:
        text = '\n'.join(out)
        dead = set()
        for ln in out:
            mm = re.match(r'^\s*(%[-\w.]+) = call <8 x (?:float|i32)> @llvm\.agx3\.w2a\.', ln)
            if mm and len(re.findall(re.escape(mm.group(1)) + r'(?![-\w.])', text)) == 1:
                dead.add(mm.group(1))
        if not dead:
            break
        out = [ln for ln in out if not (re.match(r'^\s*(%[-\w.]+) = ', ln) and re.match(r'^\s*(%[-\w.]+) = ', ln).group(1) in dead)]
    return out


def rewrite(text):
    lines = text.split('\n')
    out, used, i = [], set(), 0
    PERMUTED.clear()
    while i < len(lines):
        if lines[i].startswith('define '):
            j = i
            while lines[j] != '}':
                j += 1
            mp = re.match(r'define [^(]*\((.*)\)[^)]*\{$', lines[i])
            params = [split_top(mp.group(1))[k].split()[-1] for k in range(len(split_top(mp.group(1))))] if mp and mp.group(1) else []
            out += [lines[i]] + rewrite_function(lines[i + 1:j], used, params) + [lines[j]]
            i = j + 1
        else:
            out.append(lines[i]); i += 1
    decls = []
    for what, kind in sorted(used):
        t = TYPES[kind]
        if what == 'wgemmb':
            decls.append(f'declare {t} @llvm.agx3.wgemm.ib(i64, i16, ptr addrspace(1), i16, {t}, i32)')
        elif what == 'wgemm':
            decls.append(f'declare {t} @llvm.agx3.wgemm.f(<8 x half>, <8 x half>, {t})' if kind == 'f' else
                         f'declare {t} @llvm.agx3.wgemm.i(i64, i16, i64, i16, {t})')
        else:
            decls.append(f'declare {t} @llvm.agx3.{what}.{SUFFIX[kind]}({t})')
    return '\n'.join(out) + ('\n' + '\n'.join(decls) + '\n' if decls else '')
