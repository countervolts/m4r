#!/usr/bin/env python3
"""m4r-air: translate a Metal (AIR) compute kernel into an AMDGPU code object.

Input is the AIR bitcode of one function from a .metallib (see metallib.py) or its LLVM IR text. Output is an
ELF code object for gfx11 (wave32) plus a JSON manifest describing the kernel-argument block, so that the
host can bind buffers and textures by their Metal binding index.

The kernel's own code is kept as is: the translator only
  * specialises Metal function constants (like MTLFunctionConstantValues),
  * maps AIR address spaces onto AMDGPU ones (constant 2 -> 4, private allocas -> 5),
  * supplies AMDGPU bodies for the AIR built-ins (intrinsics.py),
  * wraps the kernel in an amdgpu_kernel entry that derives Metal's thread-position built-ins,
then lets LLVM's AMDGPU backend compile it.

Kernel argument block (what the host writes; the code object reads it through the kernarg pointer):
  one 8-byte slot per Metal buffer / texture / sampler argument in AIR parameter order
    buffer  -> device address of the buffer (+offset)
    texture -> device address of a 64-byte record: sampled-image descriptor, then storage-image descriptor
    sampler -> device address of a 16-byte sampler descriptor
  then u32 grid[3] (threads per grid; threads outside it return immediately, as with dispatchThreads)
  then u32 lds_offset[k] for each threadgroup-memory argument (byte offset into dynamic LDS)
Constexpr samplers declared inside the kernel become writable globals `__m4r_sampler_<n>` (16 bytes each);
the manifest lists their Metal sampler-state words so the host can store matching descriptors there.
"""
import argparse, hashlib, json, os, re, struct, subprocess, sys, tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import intrinsics  # noqa: E402

AMDGPU_DL = ('e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-'
             'p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-'
             'v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9')
TRIPLE = 'amdgcn-amd-amdhsa'

# The GPU the kernels are built for (set_gpu, or M4R_GPU): RDNA2 gfx1030-gfx1036, RDNA3 gfx1100-gfx1103, RDNA 3.5
# gfx1150-gfx1153, RDNA4 gfx1200-gfx1201. Code objects are per target; the matrix lowering depends on the generation:
# gfx11 has WMMA (Apple's 16x16x16 MXU products on v_wmma, accumulators kept in WMMA layout across GEMM chains), gfx12 has
# WMMA with another register layout, gfx10.3 none (dot-product instructions; bit-identical to the WMMA build on gfx1101).
GPUS = ('gfx1030', 'gfx1031', 'gfx1032', 'gfx1033', 'gfx1034', 'gfx1035', 'gfx1036', 'gfx1100', 'gfx1101', 'gfx1102',
        'gfx1103', 'gfx1150', 'gfx1151', 'gfx1152', 'gfx1153', 'gfx1200', 'gfx1201')
GPU = os.environ.get('M4R_GPU', 'gfx1101')


def family(gpu=None):
    g = gpu or GPU
    if g not in GPUS:
        raise ValueError(f'unsupported GPU {g} (RDNA2 gfx1030-gfx1036, RDNA3 gfx1100-gfx1103 / gfx1150-gfx1153, RDNA4 gfx1200-gfx1201)')
    return g[:5]   # gfx10, gfx11 or gfx12


def set_gpu(g):
    """Target GPU for translate() / compile_ir(); picks the matrix lowering unless M4R_MXU_INT / M4R_MXU_FP16 say otherwise."""
    global GPU
    family(g)
    GPU = g
    intrinsics.CTX['family'] = family(g)
    wmma = 'wmma'   # every generation runs the WMMA lowering; wmma_impl() says how the instruction itself is executed
    intrinsics.MXU_MODE['int'] = os.environ.get('M4R_MXU_INT') or (wmma or 'dot')
    intrinsics.MXU_MODE['fp16'] = os.environ.get('M4R_MXU_FP16') or (wmma or 'closed')


def wmma_impl():
    """how the gfx11 WMMA instructions the GEMM lowering emits are executed (intrinsics.wmma_impl_defs); M4R_WMMA_IMPL overrides"""
    return os.environ.get('M4R_WMMA_IMPL') or {'gfx10': 'soft', 'gfx11': 'native', 'gfx12': 'gfx12'}[family()]


def target_features(wave):
    # -real-true16: LLVM's gfx11/gfx12 true-16-bit register mode miscompiles findOpticalMotionVectorsV3 (docs/air-translation.md)
    return f'+wavefrontsize{wave}' + (',-real-true16' if family() in ('gfx11', 'gfx12') else '')

BUILTINS = {
    'air.thread_position_in_grid', 'air.thread_position_in_threadgroup', 'air.threadgroup_position_in_grid',
    'air.thread_index_in_threadgroup', 'air.threads_per_threadgroup', 'air.threadgroups_per_grid',
    'air.threads_per_grid', 'air.simdgroup_index_in_threadgroup', 'air.thread_index_in_simdgroup',
    'air.simdgroups_per_threadgroup', 'air.threads_per_simdgroup', 'air.position',
}


def run(cmd, **kw):
    r = subprocess.run(cmd, capture_output=True, **kw)
    if r.returncode:
        err = r.stderr.decode(errors='replace')   # the message comes first, a crash's stack trace after it
        raise RuntimeError(f'{cmd[0]} failed:\n' + (err if len(err) <= 4000 else err[:2000] + '\n...\n' + err[-2000:]))
    return r.stdout


def to_text(path_or_bytes):
    """AIR bitcode (with Apple's wrapper header) or IR text -> modern (opaque pointer) IR text."""
    if isinstance(path_or_bytes, bytes):
        data = path_or_bytes
    else:
        data = open(path_or_bytes, 'rb').read()
    if data[:4] in (b'\xde\xc0\x17\x0b', b'BC\xc0\xde'):
        return run(['llvm-dis', '-o', '-', '-'], input=data).decode()
    return run(['llvm-dis', '-o', '-', '-'], input=run(['llvm-as', '-o', '-', '-'], input=data)).decode()


# ---------------------------------------------------------------------------------------------------------
# metadata parsing

def md_nodes(text):
    nodes = {}
    for m in re.finditer(r'^!(\d+) = (?:distinct )?!\{(.*)\}$', text, re.M):
        nodes[int(m.group(1))] = m.group(2)
    return nodes


def md_items(body):
    return intrinsics.split_params(body)


def kernel_info(text, kernel):
    nodes = md_nodes(text)
    km = re.search(r'^!air\.(?:kernel|fragment) = !\{(.*)\}$', text, re.M)
    if not km:
        raise ValueError('no !air.kernel or !air.fragment')
    for ref in md_items(km.group(1)):
        items = md_items(nodes[int(ref[1:])])
        if items[0].endswith('@' + kernel) or items[0].endswith('@"' + kernel + '"'):
            args = []
            for aref in md_items(nodes[int(items[2][1:])]):
                fields = md_items(nodes[int(aref[1:])])
                idx = int(fields[0].split()[1])
                strs = [f for f in fields[1:]]
                a = {'index': idx, 'kind': strs[0].strip('!"')}
                i = 1
                while i < len(strs):
                    f = strs[i]
                    if f == '!"air.location_index"':
                        a['location'] = int(strs[i + 1].split()[1]); i += 3; continue
                    if f == '!"air.address_space"':
                        a['address_space'] = int(strs[i + 1].split()[1]); i += 2; continue
                    if f == '!"air.arg_type_name"':
                        a['type_name'] = strs[i + 1].strip('!"'); i += 2; continue
                    if f == '!"air.arg_name"':
                        a['name'] = strs[i + 1].strip('!"'); i += 2; continue
                    if f == '!"air.arg_type_size"':
                        a['type_size'] = int(strs[i + 1].split()[1]); i += 2; continue
                    if f == '!"air.buffer_size"':
                        a['buffer_size'] = int(strs[i + 1].split()[1]); i += 2; continue
                    if f in ('!"air.read"', '!"air.write"', '!"air.read_write"', '!"air.sample"'):
                        a['access'] = f.strip('!"')[4:]; i += 1; continue
                    if f == '!"air.struct_type_info"':
                        a['struct'] = struct_info(nodes, strs[i + 1]); i += 2; continue
                    if f == '!"air.arg_unused"':
                        a['unused'] = True
                    i += 1
                if a['kind'] == 'air.function_constant':
                    # !{i32 7, !"air.function_constant", !pred, !"air.texture", ...}
                    pred = md_items(nodes[int(strs[1][1:])])
                    a['predicate'] = pred[2].strip('!"')
                    a['kind'] = strs[2].strip('!"')
                    rest = strs[3:]
                    j = 0
                    while j < len(rest):
                        if rest[j] == '!"air.location_index"':
                            a['location'] = int(rest[j + 1].split()[1]); j += 3; continue
                        if rest[j] == '!"air.arg_name"':
                            a['name'] = rest[j + 1].strip('!"')
                        if rest[j] == '!"air.address_space"':
                            a['address_space'] = int(rest[j + 1].split()[1])
                        j += 1
                args.append(a)
            return args
    raise ValueError('kernel metadata not found for ' + kernel)


def struct_info(nodes, ref):
    items = md_items(nodes[int(ref[1:])])
    fields = []
    i = 0
    while i + 4 < len(items):
        # offset, size, <kind>, type name, member name; then optional tags: !"air.xxx" followed by an i32 value (imageblock
        # members carry !"air.render_target", i32 N)
        off = int(items[i].split()[1]); size = int(items[i + 1].split()[1])
        field = {'offset': off, 'size': size, 'type': items[i + 3].strip('!"'), 'name': items[i + 4].strip('!"')}
        i += 5
        while i < len(items) and items[i].startswith('!"air.'):
            tag = items[i].strip('!"')[4:]
            i += 1
            if i < len(items) and items[i].startswith('i32 '):
                field[tag] = int(items[i].split()[1])
                i += 1
        fields.append(field)
    return fields


def function_constants(text):
    nodes = md_nodes(text)
    m = re.search(r'^!air\.function_constants = !\{(.*)\}$', text, re.M)
    out = []
    if not m:
        return out
    for ref in md_items(m.group(1)):
        it = md_items(nodes[int(ref[1:])])
        g = re.search(r'@([\w.$"]+)', it[0]).group(1)
        out.append({'global': g, 'type': it[1].strip('!"'), 'name': it[2].strip('!"'), 'index': int(it[3].split()[1])})
    return out


# ---------------------------------------------------------------------------------------------------------

VEC_FC = {'ushort': ('i16', 'ushort'), 'short': ('i16', 'short'), 'uint': ('i32', 'uint'), 'int': ('i32', 'int'),
          'float': ('float', 'float'), 'half': ('half', 'half'), 'uchar': ('i8', 'uint'), 'char': ('i8', 'int'),
          'bool': ('i8', 'bool')}


def fc_literal(fc_type, value):
    m = re.fullmatch(r'([a-z]+?)([234])', fc_type)
    if m:
        base, n = m.group(1), int(m.group(2))
        if isinstance(value, str):
            value = [x for x in re.split(r'[,\s]+', value.strip('[]() ')) if x]
        if not isinstance(value, (list, tuple)):
            value = [value] * n
        et = VEC_FC[base][0]
        lits = [fc_literal(VEC_FC[base][1] if base not in ('uchar', 'char') else base, v)[1] for v in value]
        return f'<{n} x {et}>', '<' + ', '.join(f'{et} {l}' for l in lits) + '>'
    if fc_type in ('uchar', 'char'):
        return 'i8', str(int(value) & 0xff)
    if fc_type == 'bool':
        return 'i8', str(1 if value in (True, 1, '1', 'true', 'True') else 0)
    if fc_type in ('ushort', 'short'):
        return 'i16', str(int(value) & 0xffff if fc_type == 'ushort' else int(value))
    if fc_type in ('uint', 'int'):
        return 'i32', str(int(value))
    if fc_type == 'float':
        f = struct.unpack('<f', struct.pack('<f', float(value)))[0]
        return 'float', '0x%016X' % struct.unpack('<Q', struct.pack('<d', f))[0]
    if fc_type == 'half':
        h = struct.unpack('<H', struct.pack('<e', float(value)))[0]
        return 'half', '0xH%04X' % h
    raise ValueError('function constant type ' + fc_type)


def half_division_in_f32(text):
    """Half-precision divisions are computed in f32 and the quotient rounded to half.

    LLVM lowers a fast-math half division to v_rcp_f16(b) * a, and v_rcp_f16 treats a subnormal b as zero: 0 / 1.4e-5 gives
    0 * inf = NaN where Apple's GPU, which keeps fp16 subnormals, gives 0 (BBRNetAnisoGaussian guards a weight sum with
    `> 1e-5`, which lets half subnormals through; the NaNs then spread through the history as dark blocks). In f32 every half
    subnormal is a normal number, so the quotient is right; this only widens, it approximates nothing."""
    n = [0]

    def rep(m):
        ind, name, flags, t, rest = m.groups()
        a, b = intrinsics.split_params(rest)[:2]   # (drops trailing metadata such as !fpmath)
        ft = t.replace('half', 'float')
        # no arcp / afn: LLVM narrows fptrunc(fdiv(fpext a, fpext b)) back to a half division, and with those flags lowers
        # that to v_rcp_f16 again; without them the half division is the exact sequence (f32 reciprocal, v_div_fixup_f16)
        fl = ' '.join(f for f in flags.split() for f in (['nnan', 'ninf', 'nsz', 'contract', 'reassoc'] if f == 'fast' else [f])
                      if f not in ('arcp', 'afn'))
        flags = fl + ' ' if fl else ''
        n[0] += 1
        base = f'%m4r.div{n[0]}'
        return (f'{ind}{base}.a = fpext {t} {a} to {ft}\n{ind}{base}.b = fpext {t} {b} to {ft}\n'
                f'{ind}{base}.q = fdiv {flags}{ft} {base}.a, {base}.b\n{ind}{name} = fptrunc {ft} {base}.q to {t}')
    return re.sub(r'^(\s+)(%[-\w.]+) = fdiv ((?:(?:fast|nnan|ninf|nsz|arcp|contract|afn|reassoc) )*)(half|<\d+ x half>) (.*)$',
                  rep, text, flags=re.M)


def named_body(fn):
    """Body lines of a function (fn = its lines from `define` to `}`) with the numbered values and blocks renamed (%m4rv<n>),
    so definitions can move or disappear; parameters keep their numbers."""
    m = re.match(r'define [^(]*\((.*)\)[^)]*\{$', fn[0])
    nparams = len(intrinsics.split_params(m.group(1))) if m else 0

    def ren(m):
        return f'%m4rv{m.group(1)}' if int(m.group(1)) >= nparams else m.group(0)
    body = [re.sub(r'%(\d+)(?![\w.])', ren, ln) for ln in fn[1:-1]]
    body = [re.sub(r'^(\d+):', lambda m: f'm4rv{m.group(1)}:' if int(m.group(1)) >= nparams else m.group(0), ln) for ln in body]
    first = next((x for x in body if x.strip() and not x.strip().startswith(';')), '')
    if not re.match(r'^[-\w.$"]+:', first):   # the entry block's implicit number is now a name
        body.insert(0, f'm4rv{nparams}:')
    return body


def scatter_simd_sums(text, lane):
    """32 half simd_sums that each end up in one lane become one reduce-scatter (air.simd_sum_scatter32.f16).

    findOpticalMotionVectorsV3 scores its candidates 32 at a time: candidate K's cost is a simd_sum, and a chain
    `v = select(lane != K, v, sum_K)` over K = 0..31 (in some order) leaves candidate K's sum in lane K. With all 32 sums
    (and the chain values) used nowhere else, the chain is the same as a reduce-scatter of the 32 per-lane inputs ordered by K.
    The lowering (intrinsics.sum_scatter32) halves the set of values at each xor step 1, 2, 4, 8, 16, the order of the
    simd_sum butterfly, so every candidate's sum is the same addition tree as before: bit-identical, with 31 lane moves per 32
    sums instead of 160 and no selects. `lane` is the kernel's thread_index_in_simdgroup parameter (e.g. '%2')."""
    call = re.compile(r'^\s*(%[-\w.]+) = tail call fast half @air\.simd_sum\.f16\(half ([^)]+)\)( #\d+)?.*$')
    cmp_re = re.compile(r'^\s*(%[-\w.]+) = icmp ne i16 ' + re.escape(lane) + r', (\d+)\s*$')
    sel_re = re.compile(r'^(\s*)(%[-\w.]+) = select (?:fast )?i1 (%[-\w.]+), half ([^,]+), half (%[-\w.]+)\s*$')
    fabs_re = re.compile(r'^\s*(%[-\w.]+) = tail call fast <2 x half> @air\.fabs\.v2f16\(<2 x half> (%[-\w.]+)\)( #\d+)?\s*$')
    dot_re = re.compile(r'^\s*(%[-\w.]+) = tail call fast half @air\.dot\.v2f16\(<2 x half> (%[-\w.]+), '
                        r'<2 x half> (?:<half 0xH3C00, half 0xH3C00>|splat \(half (?:0xH3C00|1\.000000e\+00)\))\)( #\d+)?\s*$')
    absd = False
    lines, res, i, changed = text.split('\n'), [], 0, False
    while i < len(lines):
        if not lines[i].startswith('define '):
            res.append(lines[i]); i += 1
            continue
        j = i
        while lines[j] != '}':
            j += 1
        fn = lines[i:j + 1]
        i = j + 1
        if sum(1 for ln in fn if call.match(ln)) < 32:
            res += fn
            continue
        body = named_body(fn)
        uses = {}
        for ln in body:
            for t in re.findall(r'%[-\w.]+', ln.split(' = ', 1)[1] if re.match(r'^\s*%[-\w.]+ = ', ln) else ln):
                uses[t] = uses.get(t, 0) + 1
        lanes = {}                  # icmp result -> K
        sums = {}                   # simd_sum result -> (operand, line, attr)
        links = {}                  # select result -> (prev, K, sum, line)
        block = {}                  # line -> block number
        fabs = {}                   # fabs result -> operand
        absdot = {}                 # dot(fabs(d), (1, 1)) result -> (d, fabs result)
        nb = 0
        for k, ln in enumerate(body):
            if re.match(r'^[-\w.$"]+:', ln):
                nb += 1
            block[k] = nb
            if (m := fabs_re.match(ln)):
                fabs[m.group(1)] = m.group(2)
            elif (m := dot_re.match(ln)) and m.group(2) in fabs:
                absdot[m.group(1)] = (fabs[m.group(2)], m.group(2))
            elif (m := cmp_re.match(ln)):
                lanes[m.group(1)] = int(m.group(2))
            elif (m := call.match(ln)):
                sums[m.group(1)] = (m.group(2).strip(), k, m.group(3) or '')
            elif (m := sel_re.match(ln)) and m.group(3) in lanes and m.group(5) in sums and uses.get(m.group(5)) == 1:
                links[m.group(2)] = (m.group(4).strip(), lanes[m.group(3)], m.group(5), k)
        nxt = {}
        for r, (prev, _, _, _) in links.items():
            if prev in links and uses.get(prev) == 1:
                nxt[prev] = r
        starts = [r for r, (prev, _, _, _) in links.items() if not (prev in links and nxt.get(prev) == r)]
        ngroups = 0
        for r in starts:
            chain = [r]
            while chain[-1] in nxt:
                chain.append(nxt[chain[-1]])
            for g in range(0, len(chain) - 31, 32):
                grp = chain[g:g + 32]
                ks = [links[x][1] for x in grp]
                last = links[grp[-1]][3]
                if sorted(ks) != list(range(32)) or any(block[links[x][3]] != block[last] or
                                                        block[sums[links[x][2]][1]] != block[last] for x in grp):
                    continue
                if any(uses.get(x) != 1 for x in grp[:-1]) or (g > 0 and uses.get(chain[g - 1]) != 1):
                    continue
                ops = [None] * 32
                attr = sums[links[grp[0]][2]][2]
                for x in grp:
                    _, kk, s, k = links[x]
                    ops[kk] = sums[s][0]
                    body[sums[s][1]] = None
                    if x != grp[-1]:
                        body[k] = None
                ind = sel_re.match(body[last]).group(1)
                # Apple's costs are dot(|d|, 1) = |d.x| + |d.y|: with the d's the lowering adds two candidates' halves with one
                # packed add, already in the word layout of the reduce-scatter (the same f16 additions)
                ds = [absdot.get(o) for o in ops]
                if all(d and uses.get(o) == 1 and uses.get(d[1]) == 1 for o, d in zip(ops, ds)):
                    body[last] = (f'{ind}{grp[-1]} = tail call fast half @air.simd_sum_scatter32.absdot.f16('
                                  + ', '.join(f'<2 x half> {d[0]}' for d in ds) + f'){attr}')
                    absd = True
                else:
                    body[last] = (f'{ind}{grp[-1]} = tail call fast half @air.simd_sum_scatter32.f16('
                                  + ', '.join(f'half {o}' for o in ops) + f'){attr}')
                ngroups += 1
        if ngroups:
            changed = True
            fn = [fn[0]] + [ln for ln in body if ln is not None] + [fn[-1]]
        res += fn
    text = '\n'.join(res)
    if changed:
        attr = re.search(r'^declare half @air\.simd_sum\.f16\(half\)( local_unnamed_addr)?( #\d+)?', text, re.M)
        tail = (attr.group(1) or '') + (attr.group(2) or '') + '\n'
        text += '\ndeclare half @air.simd_sum_scatter32.f16(' + ', '.join(['half'] * 32) + ')' + tail
        if absd:
            text += 'declare half @air.simd_sum_scatter32.absdot.f16(' + ', '.join(['<2 x half>'] * 32) + ')' + tail
    return text


def word_selects(text):
    """`select <2 x i1> splat(c), <2 x half> a, b` becomes one 32-bit select of the bitcast words. LLVM (gfx11, fake16) splits
    the vector select into two 16-bit v_cndmask and a v_perm to repack; findOpticalMotionVectorsV3 has ~300 of them. The same
    bits are chosen either way."""
    splat_ins = dict(re.findall(r'^\s*(%[-\w.]+) = insertelement <2 x i1> (?:poison|undef), i1 (%[-\w.]+|true|false), i64 0\s*$',
                                text, re.M))
    splat = {}
    for r, src in re.findall(r'^\s*(%[-\w.]+) = shufflevector <2 x i1> (%[-\w.]+), <2 x i1> (?:poison|undef), '
                             r'<2 x i32> zeroinitializer\s*$', text, re.M):
        if src in splat_ins:
            splat[r] = splat_ins[src]
    n = [0]

    def rep(m):
        ind, r, c, a, b = m.groups()
        if c not in splat:
            return m.group(0)
        n[0] += 1
        t = f'%m4r.ws{n[0]}'
        return (f'{ind}{t}.a = bitcast <2 x half> {a} to i32\n{ind}{t}.b = bitcast <2 x half> {b} to i32\n'
                f'{ind}{t}.s = select i1 {splat[c]}, i32 {t}.a, i32 {t}.b\n{ind}{r} = bitcast i32 {t}.s to <2 x half>')
    return re.sub(r'^(\s*)(%[-\w.]+) = select (?:fast )?<2 x i1> (%[-\w.]+), <2 x half> ([^,]+), <2 x half> ([^,\s]+)\s*$',
                  rep, text, flags=re.M)


def pair_simd_sums(text):
    """Two independent half simd_sums of a basic block become one simd_sum of a <2 x half> pair (air.simd_sum.v2f16): each
    half of the pair goes through the same butterfly in the same order, so the results are the same, with one lane shuffle
    per step instead of two (findOpticalMotionVectorsV3 sums ~290 independent costs per thread). The first sum of a pair moves
    to the second one's place, so it is paired only when nothing reads it in between."""
    call = re.compile(r'^(\s*)(%[-\w.]+) = tail call fast half @air\.simd_sum\.f16\(half ([^)]+)\)(.*)$')
    out_fns, i, lines, changed = [], 0, text.split('\n'), False
    res = []
    while i < len(lines):
        if not lines[i].startswith('define '):
            res.append(lines[i]); i += 1
            continue
        j = i
        while lines[j] != '}':
            j += 1
        fn = lines[i:j + 1]
        if sum(1 for ln in fn if call.match(ln)) >= 2:
            body = named_body(fn)
            k, npair = 0, 0
            while k < len(body):
                m1 = call.match(body[k])
                if not m1:
                    k += 1; continue
                r1, a1 = m1.group(2), m1.group(3).strip()
                # readers of r1 before the next sum (and what depends on them) move after the pair when they are plain
                # arithmetic / selects / compares and do not feed the next sum
                dep, moved_lines, q, ok = {r1}, [], k + 1, True
                while q < len(body) and not call.match(body[q]) and not re.match(r'^[-\w.$"]+:', body[q]) \
                        and not re.match(r'^\s*(br|switch|ret|unreachable)\b', body[q]):
                    ln = body[q]
                    if any(re.search(re.escape(d) + r'(?![-\w.])', ln) for d in dep):
                        dm = re.match(r'^\s*(%[-\w.]+) = (\w+)', ln)
                        if not dm or dm.group(2) in ('call', 'tail', 'load', 'store', 'atomicrmw', 'cmpxchg', 'fence', 'phi'):
                            ok = False
                            break
                        dep.add(dm.group(1))
                        moved_lines.append(q)
                    q += 1
                m2 = call.match(body[q]) if ok and q < len(body) else None
                if not m2 or any(re.search(re.escape(d) + r'(?![-\w.])', m2.group(3)) for d in dep):
                    k += 1; continue
                ind, r2, a2 = m2.group(1), m2.group(2), m2.group(3).strip()
                npair += 1
                pk = f'%m4rpk{npair}'
                body[q] = (f'{ind}{pk}.a = insertelement <2 x half> poison, half {a1}, i32 0\n'
                           f'{ind}{pk}.b = insertelement <2 x half> {pk}.a, half {a2}, i32 1\n'
                           f'{ind}{pk}.s = tail call fast <2 x half> @air.simd_sum.v2f16(<2 x half> {pk}.b){m2.group(4)}\n'
                           f'{ind}{r1} = extractelement <2 x half> {pk}.s, i32 0\n'
                           f'{ind}{r2} = extractelement <2 x half> {pk}.s, i32 1')
                body[q] += ''.join('\n' + body[x] for x in moved_lines)
                for x in moved_lines:
                    body[x] = None
                body[k] = None
                k = q + 1
            if npair:
                changed = True
                fn = [fn[0]] + [ln for ln in body if ln is not None] + [fn[-1]]
        res += fn
        i = j + 1
    text = '\n'.join(res)
    if changed and 'declare <2 x half> @air.simd_sum.v2f16' not in text:
        text += '\ndeclare <2 x half> @air.simd_sum.v2f16(<2 x half>) #0\n'.replace(' #0', '')
    return text


LANE_OPS = re.compile(r'@air\.(simd_|quad_)|@llvm\.agx|implicit_imageblock|imageblock|"air\.(thread_index_in_simdgroup|'
                      r'simdgroup_index_in_threadgroup|threads_per_simdgroup|dispatch_simdgroups_per_threadgroup|'
                      r'simdgroups_per_threadgroup|thread_index_in_quadgroup|quadgroup_index_in_threadgroup)"')


def lane_free(text, tg):
    """No subgroup operation or subgroup-index input and at least 64 threads per group: the kernel computes the same in wave64
    (d4r measured 13-42% on such kernels; on MetalFX's kernels it was slower overall: WarpHistory +0.05 ms, one
    BilinearUpsample +0.09 ms, AnisoGaussian -0.02 ms), so `wave='auto'` uses it only with M4R_WAVE64=1."""
    return tg[0] * tg[1] * tg[2] >= 64 and LANE_OPS.search(text) is None


def translate(text, kernel, constants, tg, wave='auto', widen=False):
    """Return (amdgpu IR text, manifest dict)."""
    args = kernel_info(text, kernel)
    fcs = function_constants(text)
    is_fragment = re.search(r'^!air\.fragment = ', text, re.M) is not None
    uses_imageblock = '@air.store.implicit_imageblock' in text
    intrinsics.CTX['tg'] = tg
    rt_count = 0
    if uses_imageblock:
        # tile kernels: the imageblock is the render pass's colour attachment; each implicit-imageblock store names a
        # render target by a literal index and goes to that attachment's texture (kernel arguments __rendertarget,
        # __rendertarget1, ...; plan locations 255, 254, ...)
        def rt_call(m):
            nonlocal rt_count
            args = intrinsics.split_params(m.group(2))
            if not re.fullmatch(r'i32 \d+', args[1]):
                raise ValueError(f'{kernel}: render target index of an imageblock store is not a literal: {args[1]}')
            k = int(args[1].split()[1])
            rt_count = max(rt_count, k + 1)
            return f'{m.group(1)}(ptr addrspace(1) %__m4r_rt{k}, {m.group(2)}){m.group(3)}'
        text = re.sub(r'(call void @air\.store\.implicit_imageblock\.\w+)\((.*)\)( #\d+.*)$', rt_call, text, flags=re.M)
        if len(re.findall(r'call void @air\.store\.implicit_imageblock\.\w+\(<', text)):
            raise ValueError(f'{kernel}: an imageblock store was not rewritten')
        text = re.sub(r'(declare void @air\.store\.implicit_imageblock\.\w+\()', r'\1ptr addrspace(1), ', text)
        if re.search(r'@air\.(write_imageblock_slice_to_texture_2d|imageblock_data)', text) and constants.get('using_compute_imageblock'):
            raise ValueError(f'{kernel}: the compute-imageblock path is not supported')
    if wave == 'auto':
        wave = 64 if os.environ.get('M4R_WAVE64', '0') == '1' and lane_free(text, tg) else 32   # opt-in: measured slower
    manifest = {'kernel': kernel, 'threadgroup': tg, 'wave': wave, 'constants': {}, 'args': [], 'samplers': []}

    # 1. function constants
    # constants by name, or by Metal index as '#<n>' (MTLFunctionConstantValues setConstantValue:atIndex:);
    # index entries that a kernel does not declare are ignored, as Metal does
    by_name = {fc['name']: fc for fc in fcs}
    for name in constants:
        if not name.startswith('#') and name not in by_name:
            raise ValueError(f'{kernel}: unknown function constant {name}; have {sorted(by_name)}')
    for fc in fcs:
        val = constants.get(fc['name'], constants.get('#%d' % fc['index'], 0))
        manifest['constants'][fc['name']] = val
        lt, lit = fc_literal(fc['type'], val)
        pat = re.compile(r'^(@' + re.escape(fc['global']) + r' = )[^\n]*$', re.M)
        text, n = pat.subn(lambda m: f'{m.group(1)}internal unnamed_addr addrspace(2) constant {lt} {lit}, align 4', text)
        if n != 1:
            raise ValueError('function constant global not found: ' + fc['global'])

    # 2. constexpr samplers -> writable globals filled by the host
    sampler_globals = re.findall(r'^@(__air_sampler_state[\w.]*) = internal addrspace\(2\) constant \[2 x i64\] \[i64 (-?\d+), i64 (-?\d+)\]', text, re.M)
    for i, (g, w0, w1) in enumerate(sampler_globals):
        new = f'__m4r_sampler_{i}'
        manifest['samplers'].append({'global': new, 'state': [int(w0) & (2**64 - 1), int(w1) & (2**64 - 1)]})
        text = re.sub(r'^@' + re.escape(g) + r' = [^\n]*$',
                      f'@{new} = protected addrspace(1) externally_initialized global <4 x i32> zeroinitializer, align 16',
                      text, flags=re.M)
        # every use (the definition line was rewritten above) becomes a cast of the new global
        lines = text.split('\n')
        for li, ln in enumerate(lines):
            if ln.startswith('@' + new + ' ') or ln.startswith('!'):
                continue
            lines[li] = re.sub(r'@' + re.escape(g) + r'(?![\w.])',
                               f'addrspacecast (ptr addrspace(1) @{new} to ptr addrspace(2))', ln)
        text = '\n'.join(lines)
        if ((int(w0) >> 9) & 3) == 2 or ((int(w0) >> 11) & 3) == 2:
            # bicubic sampler (filter mode 2): its samples go to a Catmull-Rom lowering (intrinsics.sample_tex_bicubic); the host
            # sampler is linear. M4R_LINEAR_BICUBIC=1 keeps plain linear filtering (the old approximation).
            if not os.environ.get('M4R_LINEAR_BICUBIC'):
                cast = f'addrspacecast (ptr addrspace(1) @{new} to ptr addrspace(2))'
                names = set()
                def ren(m):
                    names.add(m.group(1)); return '@air.sample_texture_2d_bicubic' + m.group(1)
                text = '\n'.join(re.sub(r'@air\.sample_texture_2d(\.[\w.]+)\(', lambda m: ren(m) + '(', ln) if cast in ln and 'call ' in ln else ln
                                 for ln in text.split('\n'))
                for sfx in names:
                    dm = re.search(r'^declare [^\n]*@air\.sample_texture_2d' + re.escape(sfx) + r'\([^\n]*$', text, re.M)
                    if dm and not re.search(r'^declare [^\n]*@air\.sample_texture_2d_bicubic' + re.escape(sfx) + r'\(', text, re.M):
                        text = text[:dm.end()] + '\n' + dm.group(0).replace('@air.sample_texture_2d' + sfx, '@air.sample_texture_2d_bicubic' + sfx) + text[dm.end():]

    # 2a. cache-policy experiments (M4R_CPOL="kernel:N=bits,N=bits;kernel2:..."): the texture reads, samples and writes of the kernel's
    # IR argument %N get the gfx11 cache-policy bits (1 glc, 2 slc = L2 stream, 4 dlc = no Infinity Cache allocation); hints only, the
    # results are the same
    for ent in filter(None, os.environ.get('M4R_CPOL', '').split(';')):
        kn, _, lst = ent.partition(':')
        if kn != kernel:
            continue
        for item in filter(None, lst.split(',')):
            an, bits = item.split('=')
            sfxs = set()
            def ren(m):
                sfxs.add((m.group(1), m.group(2)))
                return f'@air.{m.group(1)}{m.group(2)}.cpol{bits}({m.group(3)}%{an},'
            text = re.sub(r'@air\.((?:read|sample|write)_texture_2d)((?:\.[\w]+)*)\(([^%]*)%' + re.escape(an) + ',', ren, text)
            for base, sfx in sfxs:
                dm = re.search(r'^declare [^\n]*@air\.' + base + re.escape(sfx) + r'\([^\n]*$', text, re.M)
                nn = f'@air.{base}{sfx}.cpol{bits}('
                if dm and not re.search(r'^declare [^\n]*' + re.escape(nn), text, re.M):
                    text = text[:dm.end()] + '\n' + dm.group(0).replace(f'@air.{base}{sfx}(', nn) + text[dm.end():]

    # 2b. fast-math tier (plan-time M4R_FAST_MATH, pow): fast_pow with a constant exponent 2 is a multiplication, with a constant base 2 an
    # exp2. The lowering of fast_pow (intrinsics.fpow) is written once per function and cannot see the constant, so each call costs a
    # log and an exp (four of each for a float4): the bilateral filter has one scalar and one float4 pow(x, 2), a third of its transcendentals.
    if intrinsics.fast_raw('pow'):
        text = re.sub(r'^(\s*%[\w.]+ = )(?:tail )?call (?:fast )?float @air\.fast_pow\.f32\(float (%[\w.]+), float 2\.000000e\+00\)[^\n]*$',
                      r'\1fmul fast float \2, \2', text, flags=re.M)
        text = re.sub(r'^(\s*%[\w.]+ = )(?:tail )?call (?:fast )?<4 x float> @air\.fast_pow\.v4f32\(<4 x float> (%[\w.]+), '
                      r'<4 x float> (?:<float 2\.000000e\+00, float 2\.000000e\+00, float 2\.000000e\+00, float 2\.000000e\+00>|'
                      r'splat \(float 2\.000000e\+00\))\)[^\n]*$',   # LLVM 23 prints the constant vector as splat (...)
                      r'\1fmul fast <4 x float> \2, \2', text, flags=re.M)
        text, n2 = re.subn(r'^(\s*%[\w.]+ = )(?:tail )?call (?:fast )?float @air\.fast_pow\.f32\(float 2\.000000e\+00, float (%[\w.]+)\)[^\n]*$',
                           r'\1call fast float @llvm.amdgcn.exp2.f32(float \2)', text, flags=re.M)
        if n2 and 'declare float @llvm.amdgcn.exp2.f32' not in text:
            text += '\ndeclare float @llvm.amdgcn.exp2.f32(float)\n'

    # 3. strip AIR named metadata (refers to old types); keep numbered nodes
    text = re.sub(r'^(!\d+) = !\{!"air\.sampler_state"[^\n]*$', r'\1 = !{}', text, flags=re.M)
    # lifetime markers are hints only, and LLVM requires them on an alloca (ours are behind a cast)
    text = re.sub(r'^\s+call void @llvm\.lifetime\.(start|end)[^\n]*\n', '', text, flags=re.M)
    text = re.sub(r'^!air\.[\w.]+ = [^\n]*\n', '', text, flags=re.M)
    text = re.sub(r'^!llvm\.module\.flags = [^\n]*\n', '', text, flags=re.M)

    # 4. target
    text = re.sub(r'^target datalayout = .*$', f'target datalayout = "{AMDGPU_DL}"', text, flags=re.M)
    text = re.sub(r'^target triple = .*$', f'target triple = "{TRIPLE}"', text, flags=re.M)
    text = text.replace('section "air.static_init"', '')
    text = text.replace('section "air.fc_initializer"', '')

    # threadgroup memory is uninitialised in Metal; AMDGPU rejects LDS initialisers
    text = re.sub(r'^(@[^\n]*? addrspace\(3\) global [^\n]*?) zeroinitializer', r'\1 poison', text, flags=re.M)

    # 5. address spaces: AIR constant (2) -> AMDGPU constant (4)
    text = text.replace('addrspace(2)', 'addrspace(4)')
    # the sampler casts above are now addrspace(1) -> addrspace(4), which AMDGPU allows (both 64-bit global)

    # 6. private allocas -> addrspace(5)
    def fix_alloca(m):
        ind, name, rest = m.group(1), m.group(2), m.group(3)
        tmp = '%m4r.p5.' + name[1:].strip('"')
        return f'{ind}{tmp} = alloca {rest}, addrspace(5)\n{ind}{name} = addrspacecast ptr addrspace(5) {tmp} to ptr'
    text = re.sub(r'^(\s+)(%[\w.]+) = alloca ([^\n]*?)$', fix_alloca, text, flags=re.M)

    # 7. attributes: drop target-specific ones from Apple's toolchain
    def fix_attrs(m):
        body = m.group(2)
        body = re.sub(r'"(frame-pointer|min-legal-vector-width|stack-protector-buffer-size|target-cpu|target-features|no-builtins)"(="[^"]*")?', '', body)
        body = re.sub(r'\bno-builtins\b', '', body)
        return f'attributes #{m.group(1)} = {{{body}}}'
    text = re.sub(r'^attributes #(\d+) = \{(.*)\}$', fix_attrs, text, flags=re.M)

    # 7a. half divisions in f32 (see half_division_in_f32); independent half simd_sums in pairs (see pair_simd_sums)
    text = half_division_in_f32(text)
    if os.environ.get('M4R_WORD_SELECTS', '1') != '0':
        text = word_selects(text)
    lane_arg = next((a['index'] for a in args if a['kind'] == 'air.thread_index_in_simdgroup'), None)
    if lane_arg is not None and os.environ.get('M4R_SCATTER_SUMS', '1') != '0':
        text = scatter_simd_sums(text, f'%{lane_arg}')
    if os.environ.get('M4R_PAIR_SUMS', '1') != '0':
        text = pair_simd_sums(text)

    # 7b. on inlined, scalarised IR, before the intrinsics get bodies: GEMM accumulators in WMMA layout (air/gemm_chain.py) and
    # the optional widening of half arithmetic (air/widen.py)
    # accumulators kept in WMMA layout (gfx11's on every generation, see wmma_impl; M4R_GEMM_CHAIN=0 turns it off)
    chain = os.environ.get('M4R_GEMM_CHAIN', '1') != '0' and re.search(r'@llvm\.agx3\.(fgemm|igemm)', text) is not None
    if widen or chain:
        text = run(['opt', '-O3', f'-mtriple={TRIPLE}', f'-mcpu={GPU}', f'-mattr={target_features(wave)}', '-S', '-',
                    '-o', '-'], input=text.encode()).decode()
    permuted = []
    if chain:
        import gemm_chain
        # weight tiles read straight in WMMA layout only where Apple's kernel says its weights are block-major 16 x 16 tiles
        gemm_chain.DIRECT_B = str(constants.get('blockMajorWeights', 0)) in ('1', 'True', 'true')
        text = gemm_chain.rewrite(text)
        permuted = list(gemm_chain.PERMUTED)
    if widen:
        import widen as widen_mod
        text = widen_mod.widen(text)
        manifest['widened'] = True

    # 8. AIR intrinsic bodies
    extra_decls = set()
    defs = []
    for line in re.findall(r'^declare [^\n]*@air\.[^\n]*$', text, re.M):
        body, decls = intrinsics.implement(line)
        defs.append(body)
        extra_decls |= decls
        text = text.replace(line + '\n', '')
    text = text.replace('@llvm.agx', '@m4r.agx')   # llvm.* names cannot be defined
    text = re.sub(r'\bimmarg\b ?', '', text)       # nor carry immarg
    agx = re.findall(r'^declare [^\n]*@m4r\.agx[^\n]*$', text, re.M)
    for line in agx:
        body, decls = intrinsics.implement_agx(line)
        defs.append(body)
        extra_decls |= decls
        text = text.replace(line + '\n', '')
    if agx:
        defs.append(intrinsics.mask_helpers())
        extra_decls = {d for d in extra_decls if '@__m4r_mask' not in d}
        mxu_waves = max(1, (tg[0] * tg[1] * tg[2] + 31) // 32)
        defs.append(f'@__m4r_mxu = internal addrspace(3) global [{intrinsics.MXU_WAVE_BYTES * mxu_waves} x i8] poison, align 16')
        if intrinsics.MXU_MODE['mode'] == 'probe':
            defs.append('@__m4r_probe = protected addrspace(1) externally_initialized global [1024 x i8] zeroinitializer, align 16')
        else:
            import mxu_layout
            for name, tab in (('__m4r_mxu_a', mxu_layout.A), ('__m4r_mxu_bt', mxu_layout.BT), ('__m4r_mxu_c', mxu_layout.C)):
                defs.append(f'@{name} = internal addrspace(4) constant [256 x i8] [' + ', '.join(f'i8 {v if v < 128 else v - 256}' for v in tab) + '], align 16')
    existing = set(re.findall(r'^declare [^\n]*@(llvm\.[\w.]+)\(', text, re.M))
    for d in sorted(extra_decls):
        nm = re.search(r'@(llvm\.[\w.]+)\(', d).group(1)
        if nm not in existing:
            text += '\n' + d
            existing.add(nm)
    text += '\n\n' + '\n\n'.join(defs) + '\n'
    text += 'attributes #900 = { alwaysinline nounwind }\n'

    # 8b. WMMA on other generations: the two gfx11 instructions become functions with the same signature (see wmma_impl)
    impl = wmma_impl()
    if impl != 'native' and re.search(r'@llvm\.amdgcn\.wmma\.', text):
        text = text.replace('@' + intrinsics.WMMA_F16 + '(', '@__m4r_wmma_f16(').replace('@' + intrinsics.WMMA_I8 + '(', '@__m4r_wmma_iu8(')
        text = re.sub(r'^declare [^\n]*@__m4r_wmma_(f16|iu8)\([^\n]*\n', '', text, flags=re.M)
        wdefs, wdecls = intrinsics.wmma_impl_defs(impl)
        existing_w = set(re.findall(r'^declare [^\n]*@(llvm\.[\w.]+)\(', text, re.M))
        for d in sorted(wdecls):
            nm = re.search(r'@(llvm\.[\w.]+)\(', d).group(1)
            if nm not in existing_w:
                text += '\n' + d
                existing_w.add(nm)
        text += '\n\n' + wdefs + '\n'

    # 9. kernel wrapper
    m = re.search(r'^define (?:[\w ]*?)(void|<\d+ x \w+>) @' + re.escape(kernel) + r'\((.*)\)(.*)\{$', text, re.M)
    if not m:
        raise ValueError('kernel definition not found')
    ret_type = m.group(1)
    params = [re.sub(r'\s*%[-\w.]+$', '', intrinsics.clean_type(p)).strip() for p in intrinsics.split_params(m.group(2))]
    inner = '__air_' + kernel
    head = m.group(0).replace('define ', 'define internal ', 1).replace('@' + kernel + '(', '@' + inner + '(', 1)
    if uses_imageblock:
        head = head.replace(m.group(2) + ')', m.group(2) + ''.join(f', ptr addrspace(1) %__m4r_rt{k}' for k in range(rt_count)) + ')', 1)
    text = text[:m.start()] + head + text[m.end():]

    by_index = {a['index']: a for a in args}
    kparams, call_args, pre = [], [], []
    slot = 0
    lds_args = []
    for i, ptype in enumerate(params):
        a = by_index.get(i, {'kind': 'unknown'})
        kind = a['kind']
        if kind in BUILTINS:
            call_args.append((ptype, f'%b{i}'))
            pre.append(('builtin', i, kind, ptype))
            continue
        if kind == 'air.imageblock':
            call_args.append((ptype, 'poison'))
            continue
        if kind == 'air.buffer' and a.get('address_space') == 3:
            lds_args.append(i)
            call_args.append((ptype, f'%lds{i}'))
            continue
        if kind in ('air.buffer', 'air.texture', 'air.sampler', 'air.indirect_buffer'):
            kp = ptype if ptype.startswith('ptr') else 'ptr addrspace(1)'
            noalias = ' noalias' if kind == 'air.buffer' else ''
            kparams.append(f'{kp}{noalias} %k{i}')
            call_args.append((ptype, f'%k{i}'))
            ent = {'slot': slot, 'offset': slot * 8, 'param': i, 'kind': kind[4:], 'location': a.get('location'),
                   'name': a.get('name'), 'access': a.get('access'), 'address_space': a.get('address_space'),
                   'type': a.get('type_name')}
            for k in ('predicate', 'struct', 'buffer_size', 'unused'):
                if k in a:
                    ent[k] = a[k]
            manifest['args'].append(ent)
            slot += 1
            continue
        raise ValueError(f'{kernel}: unsupported argument {i} kind {kind}')
    if is_fragment or uses_imageblock:
        for k in range(max(rt_count, 1)):
            kparams.append(f'ptr addrspace(1) %krt{k}' if k else 'ptr addrspace(1) %krt')
            manifest['args'].append({'slot': slot, 'offset': slot * 8, 'kind': 'texture', 'location': 255 - k,
                                     'name': '__rendertarget' + (str(k) if k else ''), 'access': 'write'})
            slot += 1
            if uses_imageblock:
                call_args.append(('ptr addrspace(1)', f'%krt{k}' if k else '%krt'))
    grid_off = slot * 8
    manifest['grid_offset'] = grid_off
    if permuted:   # weight buffers the plan must store as transposed 16 x 16 byte tiles (air/gemm_chain.py find_direct_b)
        manifest['transpose_b8'] = sorted(a['location'] for a in manifest['args'] if a.get('param') in permuted)
    kparams += ['i32 %gx', 'i32 %gy', 'i32 %gz']
    lds_off = grid_off + 12
    for j, i in enumerate(lds_args):
        kparams.append(f'i32 %ldsoff{i}')
        manifest['args'].append({'param': i, 'kind': 'threadgroup', 'offset': lds_off + 4 * j,
                                 'location': by_index[i].get('location'), 'name': by_index[i].get('name')})
    manifest['kernarg_size'] = lds_off + 4 * len(lds_args)

    tgx, tgy, tgz = tg
    tile = uses_imageblock   # tile kernel: Apple forms each SIMD group of a tile from an 8x4 pixel block
    phys = (tgx * tgy * tgz, 1, 1) if tile else (tgx, tgy, tgz)
    manifest['threadgroup'] = list(phys)
    w = [f'define amdgpu_kernel void @{kernel}({", ".join(kparams)}) #901 !reqd_work_group_size !{{i32 {phys[0]}, i32 {phys[1]}, i32 {phys[2]}}} {{',
         'entry:']
    if tile:
        assert tgx % 8 == 0 and tgy % 4 == 0 and tgz == 1
        w.append('  %tflat = call i32 @llvm.amdgcn.workitem.id.x()')
        w.append('  %twave = lshr i32 %tflat, 5')
        w.append('  %tlane = and i32 %tflat, 31')
        w.append(f'  %tbx = urem i32 %twave, {tgx // 8}')
        w.append(f'  %tby = udiv i32 %twave, {tgx // 8}')
        w.append('  %tbx8 = shl i32 %tbx, 3')
        w.append('  %tl4 = lshr i32 %tlane, 2')
        w.append('  %lx = add i32 %tbx8, %tl4')
        w.append('  %tby4 = shl i32 %tby, 2')
        w.append('  %tl3 = and i32 %tlane, 3')
        w.append('  %ly = add i32 %tby4, %tl3')
        w.append('  %lz = add i32 0, 0')
    else:
        w.append('  %lx = call i32 @llvm.amdgcn.workitem.id.x()')
        w.append('  %ly = call i32 @llvm.amdgcn.workitem.id.y()' if tgy > 1 else '  %ly = add i32 0, 0')
        w.append('  %lz = call i32 @llvm.amdgcn.workitem.id.z()' if tgz > 1 else '  %lz = add i32 0, 0')
    w.append('  %wx = call i32 @llvm.amdgcn.workgroup.id.x()')
    w.append('  %wy = call i32 @llvm.amdgcn.workgroup.id.y()')
    w.append('  %wz = call i32 @llvm.amdgcn.workgroup.id.z()')
    for c, t in (('x', tgx), ('y', tgy), ('z', tgz)):
        w.append(f'  %o{c} = mul i32 %w{c}, {t}')
        w.append(f'  %p{c} = add i32 %o{c}, %l{c}')
        w.append(f'  %r{c} = sub i32 %g{c}, %o{c}')
        w.append(f'  %n{c} = call i32 @llvm.umin.i32(i32 %r{c}, i32 {t})')
        w.append(f'  %ng{c}a = add i32 %g{c}, {t - 1}')
        w.append(f'  %ng{c} = udiv i32 %ng{c}a, {t}')
        w.append(f'  %out{c} = icmp uge i32 %p{c}, %g{c}')
    w.append('  %oxy = or i1 %outx, %outy')
    w.append('  %oxyz0 = or i1 %oxy, %outz')
    w.append('  %oxyz = and i1 %oxyz0, false' if tile else '  %oxyz = and i1 %oxyz0, true')
    w.append('  br i1 %oxyz, label %done, label %body')
    w.append('body:')
    if tile:
        w.append('  %flat = add i32 %tflat, 0')   # SIMD groups are the 8x4 blocks
    else:
        w.append(f'  %ly8 = mul i32 %ly, {tgx}')
        w.append(f'  %lz8 = mul i32 %lz, {tgx * tgy}')
        w.append('  %flat0 = add i32 %lx, %ly8')
        w.append('  %flat = add i32 %flat0, %lz8')
    w.append(f'  %sgi = lshr i32 %flat, {5 if wave == 32 else 6}')
    w.append(f'  %tis = and i32 %flat, {wave - 1}')
    for j, i in enumerate(lds_args):
        w.append(f'  %lds{i} = getelementptr i8, ptr addrspace(3) @__m4r_dynamic_lds, i32 %ldsoff{i}')
    src = {
        'air.thread_position_in_grid': ('p', 3), 'air.thread_position_in_threadgroup': ('l', 3),
        'air.threadgroup_position_in_grid': ('w', 3), 'air.threads_per_threadgroup': ('n', 3),
        'air.threadgroups_per_grid': ('ng', 3), 'air.threads_per_grid': ('g', 3),
    }
    nsg = (tgx * tgy * tgz + wave - 1) // wave
    for (_, i, kind, ptype) in pre:
        n, e = intrinsics.vec(ptype)
        if kind in src:
            pfx = src[kind][0]
            comps = [f'%{pfx}x', f'%{pfx}y', f'%{pfx}z'][:n]
        elif kind == 'air.thread_index_in_threadgroup':
            comps = ['%flat']
        elif kind == 'air.simdgroup_index_in_threadgroup':
            comps = ['%sgi']
        elif kind == 'air.thread_index_in_simdgroup':
            comps = ['%tis']
        elif kind == 'air.simdgroups_per_threadgroup':
            comps = [str(nsg)]
        elif kind == 'air.threads_per_simdgroup':
            comps = [str(wave)]
        elif kind == 'air.position':
            w.append(f'  %pfx{i} = uitofp i32 %px to float')
            w.append(f'  %pfy{i} = uitofp i32 %py to float')
            w.append(f'  %pcx{i} = fadd float %pfx{i}, 0.5')
            w.append(f'  %pcy{i} = fadd float %pfy{i}, 0.5')
            w.append(f'  %b{i}v0 = insertelement <4 x float> <float 0.0, float 0.0, float 0.0, float 1.0>, float %pcx{i}, i32 0')
            w.append(f'  %b{i} = insertelement <4 x float> %b{i}v0, float %pcy{i}, i32 1')
            continue
        vals = []
        for k, cv in enumerate(comps):
            if e == 'i32':
                vals.append(cv)
            else:
                nm = f'%b{i}c{k}'
                w.append(f'  {nm} = trunc i32 {cv} to {e}')
                vals.append(nm)
        if n == 1:
            if vals[0].startswith('%'):
                w.append(f'  %b{i} = add {e} {vals[0]}, 0')
            else:
                w.append(f'  %b{i} = add {e} {vals[0]}, 0')
        else:
            cur = 'poison'
            for k, v in enumerate(vals):
                nm = f'%b{i}v{k}'
                w.append(f'  {nm} = insertelement {ptype} {cur}, {e} {v}, i32 {k}')
                cur = nm
            w.append(f'  %b{i} = shufflevector {ptype} {cur}, {ptype} poison, <{n} x i32> <' + ', '.join(f'i32 {k}' for k in range(n)) + '>')
    ca = ', '.join(f'{t} {v}' for t, v in call_args)
    if ret_type == 'void':
        w.append(f'  call void @{inner}({ca})')
    else:
        # fragment function: its colour goes to the render target at this pixel
        w.append(f'  %frag = call {ret_type} @{inner}({ca})')
        rt_store = intrinsics.rt_store_def(ret_type)
        text += '\n' + rt_store[0] + '\n'
        extra_rt = rt_store[1]
        w.append(f'  call void @__m4r_rt_write({ret_type} %frag, ptr addrspace(1) %krt, i32 %px, i32 %py)')
        for d in extra_rt:
            nm = re.search(r'@([\w.]+)\(', d).group(1)
            if nm not in existing:
                text += d + '\n'
                existing.add(nm)
    w.append('  br label %done')
    w.append('done:')
    w.append('  ret void')
    w.append('}')
    text += '\n' + '\n'.join(w) + '\n'
    if lds_args:
        text += '@__m4r_dynamic_lds = external addrspace(3) global [0 x i8], align 16\n'
    for d in ('declare i32 @llvm.amdgcn.workitem.id.x()', 'declare i32 @llvm.amdgcn.workitem.id.y()',
              'declare i32 @llvm.amdgcn.workitem.id.z()', 'declare i32 @llvm.amdgcn.workgroup.id.x()',
              'declare i32 @llvm.amdgcn.workgroup.id.y()', 'declare i32 @llvm.amdgcn.workgroup.id.z()',
              'declare i32 @llvm.umin.i32(i32, i32)'):
        nm = re.search(r'@([\w.]+)\(', d).group(1)
        if nm not in existing:
            text += d + '\n'
            existing.add(nm)
    text += ('attributes #901 = { nounwind "amdgpu-flat-work-group-size"="%d,%d" "uniform-work-group-size"="false" '
             '"amdgpu-no-dispatch-ptr" "amdgpu-no-queue-ptr" "amdgpu-no-implicitarg-ptr" "amdgpu-no-dispatch-id" '
             '"amdgpu-no-hostcall-ptr" "amdgpu-no-heap-ptr" "amdgpu-no-multigrid-sync-arg" "amdgpu-no-lds-kernel-id" '
             '"amdgpu-no-default-queue" "amdgpu-no-completion-action" "amdgpu-no-flat-scratch-init" '
             '"target-cpu"="%s" "target-features"="+wavefrontsize%d" }\n') % (phys[0] * phys[1] * phys[2], phys[0] * phys[1] * phys[2], GPU, wave)
    # the inner kernel and its helpers get the same target
    text = re.sub(r'^define internal (.*?)#(\d+) \{$', r'define internal \1#\2 {', text, flags=re.M)
    manifest['wrapper_params'] = kparams
    if wave == 64:
        text = '; m4r.wave 64\n' + text
    return text, manifest


def opt_flags():
    """Extra opt options from M4R_OPT_FLAGS (space separated), e.g. -unroll-threshold=1000: optimisation choices only, the kernel computes
    the same values"""
    return os.environ.get('M4R_OPT_FLAGS', '').split()


def llc_flags():
    """Extra llc options from M4R_LLC_FLAGS (space separated), e.g. --amdgpu-sched-strategy=iterative-ilp: code generation only, no change
    to the arithmetic. The plan generators set it per kernel."""
    return os.environ.get('M4R_LLC_FLAGS', '').split()


def expand_bool_vector_bitcasts(text):
    """`%r = bitcast <N x i1> %v to iN` (what InstCombine makes of Metal's any() / all() on vector comparisons) as extracts, shifts and ors.
    The AMDGPU backend lowers that bitcast through a stack temporary: an unused 8-byte private segment in the kernel descriptor, which
    RDNA2 kernels may not have here (vkhip, no architected flat scratch). Same value, no stack."""
    def rep(m):
        ind, r, n, v = m.group(1), m.group(2), int(m.group(3)), m.group(4)
        t = f'i{n}'
        tag = 'm4r.bv' + re.sub(r'[^A-Za-z0-9_]', '_', r[1:])
        out, acc = [], None
        for k in range(n):
            out.append(f'{ind}%{tag}.e{k} = extractelement <{n} x i1> {v}, i64 {k}')
            out.append(f'{ind}%{tag}.z{k} = zext i1 %{tag}.e{k} to {t}')
            x = f'%{tag}.z{k}'
            if k:
                out.append(f'{ind}%{tag}.s{k} = shl {t} {x}, {k}')
                x = f'%{tag}.s{k}'
            if acc is None:
                acc = x
            else:
                nm = r if k == n - 1 else f'%{tag}.o{k}'
                out.append(f'{ind}{nm} = or {t} {acc}, {x}')
                acc = nm
        if n == 1:
            out.append(f'{ind}{r} = or {t} {acc}, 0')
        return '\n'.join(out)
    return re.sub(r'^(\s*)(%[-\w.$]+) = bitcast <(\d+) x i1> (%[-\w.$]+) to i\3$', rep, text, flags=re.M)


def compile_ir(text, out_path, mcpu=None, wave=32, keep=None, opt_level='3'):
    mcpu = mcpu or GPU
    if re.match(r'; m4r\.wave 64\n', text):   # translate() chose wave64 for this kernel
        wave = 64
    with tempfile.TemporaryDirectory() as d:
        ll = os.path.join(d, 'k.ll'); bc = os.path.join(d, 'k.bc'); obj = os.path.join(d, 'k.o')
        open(ll, 'w').write(text)
        if keep:
            open(keep + '.amdgpu.ll', 'w').write(text)
        run(['llvm-as', ll, '-o', bc])
        run(['opt', f'-O{opt_level}', f'-mtriple={TRIPLE}', f'-mcpu={mcpu}', f'-mattr={target_features(wave)}'] + opt_flags() + [bc, '-o', bc])
        dis = run(['llvm-dis', bc, '-o', '-']).decode()
        if 'llvm.global_ctors' in dis and re.search(r'@llvm\.global_ctors = appending global \[[1-9]', dis):
            raise RuntimeError('static initialiser of function constants was not folded')
        fixed = expand_bool_vector_bitcasts(dis)
        if fixed != dis:
            dis = fixed
            open(ll, 'w').write(dis)
            run(['llvm-as', ll, '-o', bc])
        if keep:
            open(keep + '.opt.ll', 'w').write(dis)
        run(['llc', f'-O{opt_level}', f'-mtriple={TRIPLE}', f'-mcpu={mcpu}', f'-mattr={target_features(wave)}'] + llc_flags() +
            ['-filetype=obj', bc, '-o', obj])
        if keep:
            open(keep + '.s', 'wb').write(run(['llc', f'-O{opt_level}', f'-mtriple={TRIPLE}', f'-mcpu={mcpu}',
                                                f'-mattr={target_features(wave)}', bc, '-o', '-']))
        run(['ld.lld', '-shared', obj, '-o', out_path])


def write_manifest(man, base):
    """<base>.json (full) and <base>.args (the line format the C++ runtime reads)."""
    json.dump(man, open(base + '.json', 'w'), indent=1)
    lines = [f"kernel {man['kernel']}", 'tg ' + ' '.join(map(str, man['threadgroup'])),
             f"kernarg {man['kernarg_size']}", f"grid {man['grid_offset']}"]
    for a in man['args']:
        loc = a.get('location') if a.get('location') is not None else -1
        acc = 'r' if a.get('access') in ('read', 'sample') else 'w'   # the runtime skips barriers between passes that only read
        lines.append(f"arg {a['kind']} {loc} {a['offset']} {a.get('name') or '-'} {acc}")
    for smp in man['samplers']:
        lines.append(f"sampler {smp['global']} {smp['state'][0]} {smp['state'][1]}")
    open(base + '.args', 'w').write('\n'.join(lines) + '\n')


def translator_hash():
    """Hash of the translator's own sources: kernels translated by another version of it are never reused."""
    h = hashlib.sha256()
    d = os.path.dirname(os.path.abspath(__file__))
    for name in sorted(os.listdir(d)):
        if name.endswith('.py'):
            h.update(name.encode()); h.update(open(os.path.join(d, name), 'rb').read())
    return h.hexdigest()[:16]


def specialization_key(kernel, constants, tg, source_hash):
    s = json.dumps([kernel, sorted(constants.items()), tg, source_hash, translator_hash(), GPU, wmma_impl()], sort_keys=True)
    return hashlib.sha256(s.encode()).hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('source', help='.metallib, AIR bitcode or IR text')
    ap.add_argument('kernel')
    ap.add_argument('-o', '--out', required=True, help='output code object (.hsaco); manifest is written next to it')
    ap.add_argument('--tg', default='32,1,1', help='threadgroup size x,y,z')
    ap.add_argument('--gpu', default=GPU, help='target GPU (' + ', '.join(GPUS) + ')')
    ap.add_argument('-D', action='append', default=[], help='function constant name=value')
    ap.add_argument('--widen', action='store_true', help='compute half arithmetic in f32 (air/widen.py)')
    ap.add_argument('--keep', action='store_true', help='also write the intermediate IR and assembly')
    ap.add_argument('--mxu-probe', action='store_true', help='build the MXU layout probe (tests/probe_mxu.py)')
    a = ap.parse_args()
    set_gpu(a.gpu)
    if a.mxu_probe:
        intrinsics.MXU_MODE['mode'] = 'probe'
    data = open(a.source, 'rb').read()
    if data[:4] == b'MTLB':
        import metallib
        lib = metallib.parse(data)
        f = next((f for f in lib['functions'] if f.name == a.kernel), None)
        if f is None:
            sys.exit(f'{a.kernel} not in {a.source}')
        data = metallib.bitcode(data, lib, f)
    text = to_text(data)
    consts = {}
    for d in a.D:
        k, v = d.split('=', 1)
        consts[k] = v
    tg = [int(x) for x in a.tg.split(',')]
    ir, man = translate(text, a.kernel, consts, tg, widen=a.widen)
    compile_ir(ir, a.out, keep=os.path.splitext(a.out)[0] if a.keep else None)
    write_manifest(man, os.path.splitext(a.out)[0])
    print(a.out)


set_gpu(GPU)   # M4R_GPU's matrix lowering

if __name__ == '__main__':
    main()
