"""Per-pixel kernel fusion: a producer kernel that writes one texel per thread is merged into the consumer that reads that texel
back, so the intermediate texture is neither written nor read.

MetalFX's upscaler ends with BBRNetWarpHistory -> flowAdjHistory and BBRNetComputeOvershootClamp -> overshootClamp, two
RGBA16F textures at output size that only BBRNetAnisoGaussian reads, each at its own pixel. With fusion the consumer runs the
producer's code for the pixel it reads, at the top of its entry block (the read coordinate must be a kernel parameter):

- the producer kernel becomes an internal function; its texture write stores the value (and a flag) to the caller instead;
- the producer runs for the thread that wrote the pixel: `pixel` producers write at their thread position, `rows4` producers
  (WarpHistory: thread (x, y) writes rows 4y .. 4y + 3 in a loop whose iterations only share the counter) run one iteration;
- a producer thread outside the producer's grid, or one whose own bounds test skips the write, leaves the flag clear and the
  consumer reads the texture as before (those texels are never written, in the separate passes too);
- the value goes through no texture: a half written to an RGBA16F texel and read back is the same half, so the results are
  bit-identical as long as nothing else writes what the producer reads (the plan's job: the history the consumer writes is
  double-buffered so that producers read the previous frame's);
- the code generated for the consumer's own instructions must not change either (Apple's fast-math flags let LLVM contract
  multiply-adds, and what it contracts depends on use counts and basic blocks). So the producer shares no value with the
  consumer (its inputs and result pass through empty asm statements), it runs before any consumer code instead of splitting a
  consumer block at the read, and at the read an asm copy defines the result as the image load did. Each of these was needed:
  without them one to 8,800 pixels per frame differed by about 1 ulp.

Arguments of the producer become new arguments of the fused kernel (Metal locations base + producer location) unless the plan
binds the same resource to an argument of the consumer (bind_map). Module merging is textual: both kernels come from Apple's
BR_Net_v3 source, so types, sampler states and function-constant globals the producer uses must exist identically in the
consumer (anything else is refused rather than guessed)."""
import re

import intrinsics
from m4r_air import md_nodes, md_items, kernel_info, named_body

META = re.compile(r',\s*!(?:[\w.]+)\s+!\d+')


def _functions(lines):
    """[(start, end)] line ranges of the function definitions."""
    out, i = [], 0
    while i < len(lines):
        if lines[i].startswith('define '):
            j = i
            while lines[j] != '}':
                j += 1
            out.append((i, j))
            i = j + 1
        else:
            i += 1
    return out


def _params(define_line):
    m = re.match(r'define [^(]*?@("?[-\w.$]+"?)\((.*)\)([^)]*)\{$', define_line)
    return m.group(1), intrinsics.split_params(m.group(2)), m.group(3)


def _pname(p):
    return p.split()[-1]


def _attr_renum(line, off):
    return re.sub(r'#(\d+)\b', lambda m: f'#{int(m.group(1)) + off}', line)


def fuse(ctext, ckernel, producers):
    """ctext: AIR text of the consumer; producers: list of dicts with
    text, kernel, out_loc (location of the texture the producer writes), read_loc (consumer's texture location that reads it),
    grid (producer dispatch grid [x, y]), base (first new Metal location), bind_map {('texture'|'buffer', producer location):
    consumer location} for arguments the plan binds to the same resource in the consumer. Returns the fused AIR text."""
    lines = ctext.split('\n')
    cargs = kernel_info(ctext, ckernel)
    fr = [r for r in _functions(lines) if _params(lines[r[0]])[0].strip('"') == ckernel]
    if len(fr) != 1:
        raise ValueError(f'fuse: consumer {ckernel} not found')
    cs, ce = fr[0]
    cname, cparams, ctail = _params(lines[cs])
    cbody = named_body(lines[cs:ce + 1])
    cglobals = {ln.split(' = ', 1)[0]: ln for ln in lines if ln.startswith('@')}
    ctypes = {ln.split(' = ', 1)[0]: ln for ln in lines if re.match(r'^%[-\w.]+ = type ', ln)}
    cdecl = set(re.findall(r'^declare [^@]*@([-\w.$]+)\(', ctext, re.M)) | set(re.findall(r'^define [^@]*@([-\w.$]+)\(', ctext, re.M))
    cstores = {g for r in _functions(lines) if 'air.static_init' in lines[r[0]] for ln in lines[r[0] + 1:r[1]]
               for g in re.findall(r'^\s*store [^@]*(@[-\w.]+)', ln)}
    nodes = md_nodes(ctext)
    next_md = max(int(k) for k in nodes) + 1
    km = re.search(r'^!air\.(?:kernel|fragment) = !\{(.*)\}$', ctext, re.M)
    knode = next(r for r in md_items(km.group(1)) if md_items(nodes[int(r[1:])])[0].endswith('@' + ckernel))
    arglist_id = int(md_items(nodes[int(knode[1:])])[2][1:])
    new_arg_nodes, new_md, extra_params, appended = [], [], [], []
    entry_allocas, entry_code, n_index = [], [], len(cparams)

    def param_by_loc(args, kind, loc):
        a = [x for x in args if x['kind'] == f'air.{kind}' and x.get('location') == loc]
        return a[0] if len(a) == 1 else None

    for pi, p in enumerate(producers):
        off = 1000 * (pi + 1)
        sfx = f'.m4rp{pi}'
        ptext = p['text']
        pl = ptext.split('\n')
        pargs = kernel_info(ptext, p['kernel'])
        pnodes = md_nodes(ptext)
        pkm = re.search(r'^!air\.(?:kernel|fragment) = !\{(.*)\}$', ptext, re.M)
        pk = next(r for r in md_items(pkm.group(1)) if md_items(pnodes[int(r[1:])])[0].endswith('@' + p['kernel']))
        pnode_of = {int(md_items(pnodes[int(r[1:])])[0].split()[1]): r
                    for r in md_items(pnodes[int(md_items(pnodes[int(pk[1:])])[2][1:])])}
        # module-level state the producer needs
        for ln in pl:
            if re.match(r'^%[-\w.]+ = type ', ln):
                nm = ln.split(' = ', 1)[0]
                if nm in ctypes and ctypes[nm] != ln:
                    raise ValueError(f'fuse: type {nm} differs')
                if nm not in ctypes:
                    appended.append(ln); ctypes[nm] = ln
            elif ln.startswith('@'):
                nm = ln.split(' = ', 1)[0]
                if nm in ('@llvm.global_ctors', '@llvm.compiler.used', '@llvm.used'):
                    continue
                if cglobals.get(nm) != ln:
                    raise ValueError(f'fuse: producer global {nm} has no identical consumer global')
        # the producer's static initialiser may only copy function constants the consumer's initialiser copies too
        pstores = {g for r in _functions(pl) if 'air.static_init' in pl[r[0]] for ln in pl[r[0] + 1:r[1]]
                   for g in re.findall(r'^\s*store [^@]*(@[-\w.]+)', ln)}
        bad = [ln.strip() for r in _functions(pl) if 'air.static_init' in pl[r[0]] for ln in pl[r[0] + 1:r[1]]
               if ln.strip() and not re.match(r'^(%[-\w.]+ = (load|trunc|zext|icmp|and|or|select) |store |ret void|[-\w.]+:)', ln.strip())]
        if bad or not pstores <= cstores:
            raise ValueError(f'fuse: producer static initialiser not covered by the consumer\'s: {bad or pstores - cstores}')
        for ln in pl:
            m = re.match(r'^declare [^@]*@([-\w.$]+)\(', ln)
            if m and m.group(1) not in cdecl:
                appended.append(_attr_renum(ln, off)); cdecl.add(m.group(1))
            m = re.match(r'^attributes #(\d+) = (.*)$', ln)
            if m:
                appended.append(f'attributes #{int(m.group(1)) + off} = {m.group(2)}')
        # functions: helpers renamed, the kernel rewritten
        helpers = [_params(pl[r[0]])[0] for r in _functions(pl) if 'air.static_init' not in pl[r[0]]
                   and _params(pl[r[0]])[0].strip('"') != p['kernel']]

        def ren(ln):
            for h in helpers:
                ln = re.sub(r'@' + re.escape(h) + r'(?![-\w.$])', '@' + h + sfx, ln)
            return _attr_renum(META.sub('', ln), off)
        for r in _functions(pl):
            nm = _params(pl[r[0]])[0]
            if nm in helpers:
                body = [ren(ln) for ln in pl[r[0]:r[1] + 1]]
                if any(re.search(r'![\d]', ln) for ln in body):
                    raise ValueError(f'fuse: metadata left in {nm}')
                appended += [''] + body
        kr = [r for r in _functions(pl) if _params(pl[r[0]])[0].strip('"') == p['kernel']][0]
        _, pparams, ptail = _params(pl[kr[0]])
        body = [ren(ln) for ln in named_body(pl[kr[0]:kr[1] + 1])]
        if any('alloca' in ln for ln in body) or any(re.search(r'![\d]', ln) for ln in body):
            raise ValueError(f'fuse: {p["kernel"]} has allocas or metadata operands')
        out_arg = param_by_loc(pargs, 'texture', p['out_loc'])
        gid_arg = [a for a in pargs if a['kind'] == 'air.thread_position_in_grid']
        if out_arg is None or len(gid_arg) != 1:
            raise ValueError(f'fuse: {p["kernel"]}: output texture or thread position not found')
        O, G = _pname(pparams[out_arg['index']]), _pname(pparams[gid_arg[0]['index']])
        builtins = [a for a in pargs if a['kind'] not in ('air.texture', 'air.buffer') and a is not gid_arg[0]]
        if builtins:
            raise ValueError(f'fuse: {p["kernel"]}: unsupported inputs {[a["kind"] for a in builtins]}')
        writes = [k for k, ln in enumerate(body) if re.search(r'@air\.write_texture_2d\.\w+\.v4f16\(ptr addrspace\(1\)[^%]*' + re.escape(O) + r',', ln)]
        if len(writes) != 1 or sum(len(re.findall(re.escape(O) + r'(?![-\w.])', ln)) for ln in body) != 1:
            raise ValueError(f'fuse: {p["kernel"]}: the output texture is used other than by one write')
        wm = re.match(r'^\s*tail call void @air\.write_texture_2d\.\w+\.v4f16\(ptr addrspace\(1\)[^%]*%[-\w.]+, <2 x i16> (%[-\w.]+), '
                      r'<4 x half> (%[-\w.]+), i16 0, i32 2\)', body[writes[0]])
        if not wm:
            raise ValueError(f'fuse: {p["kernel"]}: write not understood: {body[writes[0]].strip()}')
        coord, val = wm.groups()
        defs = {m.group(1): (k, m.group(2)) for k, ln in enumerate(body) if (m := re.match(r'^\s*(%[-\w.]+) = (.*)$', ln))}
        if coord == G:
            kind = 'pixel'
        else:
            # rows4: coord = (gid.x, 4 * gid.y + k), k a loop counter from 0; one iteration with k = y & 3
            m1 = re.fullmatch(r'insertelement <2 x i16> ' + re.escape(G) + r', i16 (%[-\w.]+), i64 1', defs.get(coord, (0, ''))[1])
            m2 = m1 and re.fullmatch(r'add nuw nsw i16 (%[-\w.]+), (%[-\w.]+)', defs.get(m1.group(1), (0, ''))[1])
            m3 = m2 and re.fullmatch(r'shl i16 (%[-\w.]+), 2', defs.get(m2.group(2), (0, ''))[1])
            m4 = m3 and re.fullmatch(r'extractelement <2 x i16> ' + re.escape(G) + r', i64 1', defs.get(m3.group(1), (0, ''))[1])
            K = m2.group(1) if m4 else None
            mk = K and re.fullmatch(r'phi i16 \[ 0, (%[-\w.]+) \], \[ (%[-\w.]+), (%[-\w.]+) \]', defs.get(K, (0, ''))[1])
            if not mk:
                raise ValueError(f'fuse: {p["kernel"]}: write coordinate is neither the thread position nor a 4-row loop')
            # the loop: header exit `icmp eq k, rows_left` (-> uge: k0 may be past the rows), latch `icmp eq k + 1, 4` (-> exit)
            inc = mk.group(2)
            if not re.fullmatch(r'add nuw nsw i16 ' + re.escape(K) + r', 1', defs.get(inc, (0, ''))[1]):
                raise ValueError(f'fuse: {p["kernel"]}: loop increment not understood')
            ex = [n for n, (k, d) in defs.items() if re.fullmatch(r'icmp eq i16 ' + re.escape(K) + r', %[-\w.]+', d)]
            lt = [n for n, (k, d) in defs.items() if d == f'icmp eq i16 {inc}, 4']
            uses_k = sum(len(re.findall(re.escape(K) + r'(?![-\w.])', ln)) for ln in body)
            if len(ex) != 1 or len(lt) != 1 or uses_k != 4:   # phi def, header compare, the add for the row, the increment
                raise ValueError(f'fuse: {p["kernel"]}: loop not understood')
            kl, _ = defs[K]
            body[kl] = body[kl].replace('[ 0, ', '[ %m4r.k0, ', 1)
            body[defs[ex[0]][0]] = body[defs[ex[0]][0]].replace('icmp eq i16', 'icmp uge i16', 1)
            body[defs[lt[0]][0]] = re.sub(r'icmp eq i16 .*$', f'icmp eq i16 {inc}, {inc}', body[defs[lt[0]][0]])
            kind = 'rows4'
        ind = re.match(r'^(\s*)', body[writes[0]]).group(1)
        # The producer shares no value with the consumer: its inputs and its result cross empty asm statements (one per value,
        # each with its own text so that none are merged). Apple's kernels carry fast-math flags, and once the code sits in
        # one function LLVM merges equal subexpressions of the two (the same parameter loads, coordinates, texture reads);
        # that changes use counts, and with them whether a multiply-add is contracted to an fma, which changed one consumer
        # pixel of the first test frame even though the produced values were right.
        body[writes[0]] = (f'{ind}%m4r.vb = bitcast <4 x half> {val} to <2 x i32>\n'
                           f'{ind}%m4r.vf = call <2 x i32> asm "; m4r p{pi} out", "=v,0"(<2 x i32> %m4r.vb)\n'
                           f'{ind}store <2 x i32> %m4r.vf, ptr %m4r.out, align 8\n{ind}store i1 true, ptr %m4r.flag, align 1')
        fences, fparams = [], []
        for j, pp in enumerate(pparams):
            ty = intrinsics.clean_type(pp.rsplit(' ', 1)[0])
            if _pname(pp) != f'%{j}':
                raise ValueError(f'fuse: {p["kernel"]}: parameter {j} is not %{j}')
            fparams.append(f'{ty} %m4r.i{j}')
            fences.append(f'  %{j} = call {ty} asm "; m4r p{pi} in{j}", "={"s" if ty.startswith("ptr") else "v"},0"({ty} %m4r.i{j})')
        fences.append(f'  %m4r.k0 = call i16 asm "; m4r p{pi} k0", "=v,0"(i16 %m4r.k0i)')
        # the producer's grid: threads outside it never ran
        first = next(ln for ln in body if re.match(r'^[-\w.$"]+:', ln)).split(':')[0]
        gx, gy = p['grid'][:2]
        guard = ['m4r.e:'] + fences + [
            f'  %m4r.gx = extractelement <2 x i16> {G}, i64 0', f'  %m4r.gy = extractelement <2 x i16> {G}, i64 1',
            f'  %m4r.ix = icmp ult i16 %m4r.gx, {gx}', f'  %m4r.iy = icmp ult i16 %m4r.gy, {gy}',
            f'  %m4r.in = and i1 %m4r.ix, %m4r.iy', f'  br i1 %m4r.in, label %{first}, label %m4r.skip', 'm4r.skip:', '  ret void']
        group = int(re.search(r'#(\d+)', ptail).group(1)) + off
        gtext = next(ln for ln in appended if ln.startswith(f'attributes #{group} = '))
        appended.append(f'attributes #{off + 999} = ' + gtext.split(' = ', 1)[1].replace('{', '{ alwaysinline', 1))
        fname = f'@m4r.fuse.p{pi}'
        appended += ['', f'define internal void {fname}(' + ', '.join(fparams) + f', i16 %m4r.k0i, ptr %m4r.out, ptr %m4r.flag) #{off + 999} {{']
        appended += guard + body + ['}']

        # consumer: the call at each read of the intermediate
        rd = param_by_loc(cargs, 'texture', p['read_loc'])
        if rd is None:
            raise ValueError(f'fuse: consumer has no texture at location {p["read_loc"]}')
        R = _pname(cparams[rd['index']])
        rre = re.compile(r'^(\s*)(%[-\w.]+) = tail call \{ <4 x half>, i8 \} @air\.read_texture_2d\.i16\.v4f16\(ptr addrspace\(1\)[^%]*'
                         + re.escape(R) + r', ptr addrspace\(2\) ([^,]+), <2 x i16> ([^,]+), <2 x i16> zeroinitializer, i16 0, i32 (\d+)\)(.*)$')
        reads = [k for k, ln in enumerate(cbody) if re.search(re.escape(R) + r'(?![-\w.])', ln)]
        if not reads or any(not rre.match(cbody[k]) for k in reads):
            raise ValueError(f'fuse: consumer uses {R} other than by plain reads')
        # producer arguments
        call_args = []
        for j, pp in enumerate(pparams):
            a = next(x for x in pargs if x['index'] == j)
            ty = intrinsics.clean_type(pp.rsplit(' ', 1)[0])
            if a is out_arg:
                call_args.append(f'{ty} null'); continue
            if a is gid_arg[0]:
                call_args.append(f'{ty} %m4r.g'); continue
            k = a['kind'][4:]
            cl = p['bind_map'].get((k, a.get('location')))
            ca = param_by_loc(cargs, k, cl) if cl is not None else None
            if ca is not None:
                call_args.append(f'{ty} {_pname(cparams[ca["index"]])}'); continue
            # a new argument of the fused kernel
            nm = f'%m4r.a{pi}.{j}'
            extra_params.append(f'{ty} {nm}')
            call_args.append(f'{ty} {nm}')
            fields = md_items(pnodes[int(pnode_of[j][1:])])
            fields[0] = f'i32 {n_index}'
            li = fields.index('!"air.location_index"')
            fields[li + 1] = f'i32 {p["base"] + a["location"]}'

            def copy_ref(ref):
                nonlocal next_md
                items = md_items(pnodes[int(ref[1:])])
                items = [copy_ref(x) if re.fullmatch(r'!\d+', x) else x for x in items]
                nid = next_md; next_md += 1
                new_md.append(f'!{nid} = !{{{", ".join(items)}}}')
                return f'!{nid}'
            fields = [copy_ref(x) if re.fullmatch(r'!\d+', x) else x for x in fields]
            nid = next_md; next_md += 1
            new_md.append(f'!{nid} = !{{{", ".join(fields)}}}')
            new_arg_nodes.append(f'!{nid}')
            n_index += 1
        entry_allocas += [f'  %m4r.fo{pi} = alloca <4 x half>, align 8', f'  %m4r.fl{pi} = alloca i1, align 1']
        # The producer and the fallback read run at the top of the consumer's entry block, before any of its code: inlined at
        # the read, the producer's branches would split the consumer's block there, and instruction selection forms fmas per
        # block, so a multiply-add of the consumer around the split would no longer be contracted (seen as one changed pixel).
        # So the read coordinate must be available there: a parameter (the thread position, as in BBRNetAnisoGaussian).
        names = {_pname(x) for x in cparams}
        for ri, k in enumerate(reads):
            _, res, smp, crd, acc, rest = rre.match(cbody[k]).groups()
            if crd not in names:
                raise ValueError(f'fuse: consumer reads {R} at {crd}, not at a parameter')
            t = f'%m4r.r{pi}.{ri}'
            seq = [f'  store i1 false, ptr %m4r.fl{pi}, align 1']
            if kind == 'rows4':
                # (from a fenced copy of the coordinate: the consumer's own uses of it keep their use counts)
                seq += [f'  {t}.c = call <2 x i16> asm "; m4r p{pi} c{ri}", "=v,0"(<2 x i16> {crd})',
                        f'  {t}.cy = extractelement <2 x i16> {t}.c, i64 1', f'  {t}.gy = lshr i16 {t}.cy, 2',
                        f'  {t}.g = insertelement <2 x i16> {t}.c, i16 {t}.gy, i64 1', f'  {t}.k = and i16 {t}.cy, 3']
                g, k0 = f'{t}.g', f'{t}.k'
            else:
                g, k0 = crd, '0'
            args = ', '.join(x.replace('%m4r.g', g) if x.endswith(' %m4r.g') else x for x in call_args)
            seq += [f'  call void {fname}({args}, i16 {k0}, ptr %m4r.fo{pi}, ptr %m4r.fl{pi})',
                    f'  {t}.h = load i1, ptr %m4r.fl{pi}, align 1', f'  {t}.v = load <4 x half>, ptr %m4r.fo{pi}, align 8',
                    f'  {t}.s = tail call ptr addrspace(2) @air.get_read_sampler()',
                    f'  {t}.r = call {{ <4 x half>, i8 }} @m4r.fuse.read{acc}(ptr addrspace(1) {R}, ptr addrspace(2) {t}.s, '
                    f'<2 x i16> {crd}, i1 {t}.h, <4 x half> {t}.v)']
            entry_code += seq
            # at the read itself an opaque copy defines the result, as the image load did, so the consumer's block keeps the
            # same shape for instruction selection (with the value only flowing in from the entry block, the selector chose
            # other f16 multiply-add patterns there: ~1 ulp in 8,800 pixels of the second test frame)
            ind = rre.match(cbody[k]).group(1)
            cbody[k] = '\n'.join([
                f'{ind}{t}.e = extractvalue {{ <4 x half>, i8 }} {t}.r, 0', f'{ind}{t}.b = bitcast <4 x half> {t}.e to <2 x i32>',
                f'{ind}{t}.u = call <2 x i32> asm "; m4r p{pi} use{ri}", "=v,0"(<2 x i32> {t}.b)',
                f'{ind}{t}.x = bitcast <2 x i32> {t}.u to <4 x half>',
                f'{ind}{t}.y = insertvalue {{ <4 x half>, i8 }} poison, <4 x half> {t}.x, 0',
                f'{ind}{res} = insertvalue {{ <4 x half>, i8 }} {t}.y, i8 1, 1'])
            helper = f'@m4r.fuse.read{acc}'
            if not any(f'define internal {{ <4 x half>, i8 }} {helper}(' in ln for ln in appended):
                appended += ['', f'define internal {{ <4 x half>, i8 }} {helper}(ptr addrspace(1) %t, ptr addrspace(2) %s, <2 x i16> %c, '
                             f'i1 %h, <4 x half> %v) #{off + 998} {{',
                             '  br i1 %h, label %y, label %n', 'y:', '  %r0 = insertvalue { <4 x half>, i8 } poison, <4 x half> %v, 0',
                             '  %r1 = insertvalue { <4 x half>, i8 } %r0, i8 1, 1', '  ret { <4 x half>, i8 } %r1', 'n:',
                             f'  %r = tail call {{ <4 x half>, i8 }} @air.read_texture_2d.i16.v4f16(ptr addrspace(1) %t, ptr addrspace(2) %s, '
                             f'<2 x i16> %c, <2 x i16> zeroinitializer, i16 0, i32 {acc})', '  ret { <4 x half>, i8 } %r', '}']
                appended.append(f'attributes #{off + 998} = {{ alwaysinline nounwind }}')

    # the consumer with its new arguments and allocas
    lab = next(k for k, ln in enumerate(cbody) if re.match(r'^[-\w.$"]+:', ln))
    cbody[lab + 1:lab + 1] = entry_allocas + entry_code
    head = re.sub(r'\((.*)\)([^)]*)\{$', lambda m: '(' + ', '.join(cparams + extra_params) + ')' + m.group(2) + '{', lines[cs])
    lines[cs:ce + 1] = [head] + cbody + ['}']
    text = '\n'.join(lines)
    al = re.search(r'^!' + str(arglist_id) + r' = !\{(.*)\}$', text, re.M)
    text = text[:al.start()] + f'!{arglist_id} = !{{' + ', '.join(md_items(al.group(1)) + new_arg_nodes) + '}' + text[al.end():]
    return text + '\n' + '\n'.join(appended + new_md) + '\n'
