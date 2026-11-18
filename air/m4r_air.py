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

BUILTINS = {
    'air.thread_position_in_grid', 'air.thread_position_in_threadgroup', 'air.threadgroup_position_in_grid',
    'air.thread_index_in_threadgroup', 'air.threads_per_threadgroup', 'air.threadgroups_per_grid',
    'air.threads_per_grid', 'air.simdgroup_index_in_threadgroup', 'air.thread_index_in_simdgroup',
    'air.simdgroups_per_threadgroup', 'air.threads_per_simdgroup', 'air.position',
}


def run(cmd, **kw):
    r = subprocess.run(cmd, capture_output=True, **kw)
    if r.returncode:
        raise RuntimeError(f'{cmd[0]} failed:\n' + r.stderr.decode(errors='replace')[-4000:])
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
    for i in range(0, len(items), 5):
        if i + 4 < len(items):
            off = int(items[i].split()[1]); size = int(items[i + 1].split()[1])
            fields.append({'offset': off, 'size': size, 'type': items[i + 3].strip('!"'), 'name': items[i + 4].strip('!"')})
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


def translate(text, kernel, constants, tg, wave=32):
    """Return (amdgpu IR text, manifest dict)."""
    args = kernel_info(text, kernel)
    fcs = function_constants(text)
    is_fragment = re.search(r'^!air\.fragment = ', text, re.M) is not None
    uses_imageblock = '@air.store.implicit_imageblock' in text
    intrinsics.CTX['tg'] = tg
    if uses_imageblock:
        # tile kernels: the imageblock is the render pass's colour attachment; stores go to a render-target texture
        text = re.sub(r'(call void @air\.store\.implicit_imageblock\.\w+\()', r'\1ptr addrspace(1) %__m4r_rt, ', text)
        text = re.sub(r'(declare void @air\.store\.implicit_imageblock\.\w+\()', r'\1ptr addrspace(1), ', text)
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

    # 9. kernel wrapper
    m = re.search(r'^define (?:[\w ]*?)(void|<\d+ x \w+>) @' + re.escape(kernel) + r'\((.*)\)(.*)\{$', text, re.M)
    if not m:
        raise ValueError('kernel definition not found')
    ret_type = m.group(1)
    params = [re.sub(r'\s*%\d+$', '', intrinsics.clean_type(p)).strip() for p in intrinsics.split_params(m.group(2))]
    inner = '__air_' + kernel
    head = m.group(0).replace('define ', 'define internal ', 1).replace('@' + kernel + '(', '@' + inner + '(', 1)
    if uses_imageblock:
        head = head.replace(m.group(2) + ')', m.group(2) + ', ptr addrspace(1) %__m4r_rt)', 1)
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
        kparams.append('ptr addrspace(1) %krt')
        manifest['args'].append({'slot': slot, 'offset': slot * 8, 'kind': 'texture', 'location': 255,
                                 'name': '__rendertarget', 'access': 'write'})
        slot += 1
        if uses_imageblock:
            call_args.append(('ptr addrspace(1)', '%krt'))
    grid_off = slot * 8
    manifest['grid_offset'] = grid_off
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
             '"target-cpu"="gfx1101" "target-features"="+wavefrontsize%d" }\n') % (phys[0] * phys[1] * phys[2], phys[0] * phys[1] * phys[2], wave)
    # the inner kernel and its helpers get the same target
    text = re.sub(r'^define internal (.*?)#(\d+) \{$', r'define internal \1#\2 {', text, flags=re.M)
    manifest['wrapper_params'] = kparams
    return text, manifest


def compile_ir(text, out_path, mcpu='gfx1101', wave=32, keep=None, opt_level='3'):
    with tempfile.TemporaryDirectory() as d:
        ll = os.path.join(d, 'k.ll'); bc = os.path.join(d, 'k.bc'); obj = os.path.join(d, 'k.o')
        open(ll, 'w').write(text)
        if keep:
            open(keep + '.amdgpu.ll', 'w').write(text)
        run(['llvm-as', ll, '-o', bc])
        run(['opt', f'-O{opt_level}', f'-mtriple={TRIPLE}', f'-mcpu={mcpu}', f'-mattr=+wavefrontsize{wave},-real-true16', bc, '-o', bc])
        dis = run(['llvm-dis', bc, '-o', '-']).decode()
        if 'llvm.global_ctors' in dis and re.search(r'@llvm\.global_ctors = appending global \[[1-9]', dis):
            raise RuntimeError('static initialiser of function constants was not folded')
        if keep:
            open(keep + '.opt.ll', 'w').write(dis)
        run(['llc', f'-O{opt_level}', f'-mtriple={TRIPLE}', f'-mcpu={mcpu}', f'-mattr=+wavefrontsize{wave},-real-true16',
             '-filetype=obj', bc, '-o', obj])
        if keep:
            open(keep + '.s', 'wb').write(run(['llc', f'-O{opt_level}', f'-mtriple={TRIPLE}', f'-mcpu={mcpu}',
                                                f'-mattr=+wavefrontsize{wave},-real-true16', bc, '-o', '-']))
        run(['ld.lld', '-shared', obj, '-o', out_path])


def write_manifest(man, base):
    """<base>.json (full) and <base>.args (the line format the C++ runtime reads)."""
    json.dump(man, open(base + '.json', 'w'), indent=1)
    lines = [f"kernel {man['kernel']}", 'tg ' + ' '.join(map(str, man['threadgroup'])),
             f"kernarg {man['kernarg_size']}", f"grid {man['grid_offset']}"]
    for a in man['args']:
        loc = a.get('location') if a.get('location') is not None else -1
        lines.append(f"arg {a['kind']} {loc} {a['offset']} {a.get('name') or '-'}")
    for smp in man['samplers']:
        lines.append(f"sampler {smp['global']} {smp['state'][0]} {smp['state'][1]}")
    open(base + '.args', 'w').write('\n'.join(lines) + '\n')


def specialization_key(kernel, constants, tg, source_hash):
    s = json.dumps([kernel, sorted(constants.items()), tg, source_hash], sort_keys=True)
    return hashlib.sha256(s.encode()).hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('source', help='.metallib, AIR bitcode or IR text')
    ap.add_argument('kernel')
    ap.add_argument('-o', '--out', required=True, help='output code object (.hsaco); manifest is written next to it')
    ap.add_argument('--tg', default='32,1,1', help='threadgroup size x,y,z')
    ap.add_argument('-D', action='append', default=[], help='function constant name=value')
    ap.add_argument('--keep', action='store_true', help='also write the intermediate IR and assembly')
    ap.add_argument('--mxu-probe', action='store_true', help='build the MXU layout probe (tests/probe_mxu.py)')
    a = ap.parse_args()
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
    ir, man = translate(text, a.kernel, consts, tg)
    compile_ir(ir, a.out, keep=os.path.splitext(a.out)[0] if a.keep else None)
    write_manifest(man, os.path.splitext(a.out)[0])
    print(a.out)


if __name__ == '__main__':
    main()
