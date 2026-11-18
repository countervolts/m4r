"""AMDGPU (gfx11, wave32) implementations of the AIR intrinsics used by Metal kernels.

Each `declare ... @air.NAME(...)` in a translated module is replaced by a `define internal` body generated
here. The bodies are written in LLVM IR against the AMDGPU target intrinsics, so the result of a Metal
built-in is computed by the matching AMD instruction (texture filtering by the texture unit, conversions by
the ALU) rather than by an emulation of Apple's hardware.

Conventions shared with m4r_air.py:
  * a texture argument is a `ptr addrspace(1)` to a 64-byte record: bytes [0,32) hold the sampled-image
    descriptor (T#) used for read/sample/gather/size queries, bytes [32,64) the storage-image descriptor
    used for writes and atomics. A null texture is a record of zeros.
  * a sampler argument is a `ptr addrspace(4)` to a 16-byte sampler descriptor (S#).
  * SIMD groups are wave32 waves; lane = mbcnt over the full mask.
"""
import re

FLOATS = {'half': 'f16', 'float': 'f32'}


class Sig:
    """Parsed `declare` line."""

    def __init__(self, line):
        m = re.match(r'declare\s+(.*?)\s*@(air\.[\w.]+)\((.*)\)\s*(local_unnamed_addr)?\s*(#\d+)?\s*$', line.strip())
        if not m:
            raise ValueError('cannot parse declaration: ' + line)
        self.ret = clean_type(m.group(1))
        self.name = m.group(2)
        self.params = [clean_type(p) for p in split_params(m.group(3))]

    def header(self):
        args = ', '.join(f'{t} %a{i}' for i, t in enumerate(self.params))
        return f'define internal {self.ret} @{self.name}({args}) #900 {{'


ATTR_WORDS = r'\b(noundef|nocapture|readonly|writeonly|readnone|nonnull|noalias|returned|signext|zeroext|inreg|' \
             r'captures\([^)]*\)|align \d+|dereferenceable\(\d+\)|dereferenceable_or_null\(\d+\))'


def clean_type(t):
    t = re.sub(ATTR_WORDS, '', t)
    t = re.sub(r'(?<![%@])"[^"]*"(="[^"]*")?', '', t)   # attribute strings, not quoted type names
    return ' '.join(t.split())


def split_params(s):
    out, depth, cur = [], 0, ''
    for ch in s:
        if ch in '<{[(':
            depth += 1
        elif ch in '>}])':
            depth -= 1
        if ch == ',' and depth == 0:
            out.append(cur)
            cur = ''
        else:
            cur += ch
    if cur.strip():
        out.append(cur)
    return [p.strip() for p in out]


def vec(t):
    """'<4 x half>' -> (4, 'half'); 'half' -> (1, 'half')."""
    m = re.match(r'<(\d+) x (\w+)>', t)
    return (int(m.group(1)), m.group(2)) if m else (1, t)


def ivsuffix(t):
    n, e = vec(t)
    s = {'half': 'f16', 'float': 'f32', 'i1': 'i1', 'i8': 'i8', 'i16': 'i16', 'i32': 'i32', 'i64': 'i64'}[e]
    return f'v{n}{s}' if n > 1 else s


def splat(t, scalar_value):
    n, e = vec(t)
    if n == 1:
        return scalar_value
    return '<' + ', '.join(f'{e} {scalar_value}' for _ in range(n)) + '>'


class Body:
    def __init__(self):
        self.lines = []
        self.n = 0
        self.decls = set()

    def v(self, expr):
        self.n += 1
        name = f'%t{self.n}'
        self.lines.append(f'  {name} = {expr}')
        return name

    def emit(self, line):
        self.lines.append('  ' + line)

    def call(self, ret, fn, args, decl_params=None):
        if decl_params is not None:
            self.decls.add(f'declare {ret} @{fn}({", ".join(decl_params)})')
        return self.v(f'call {ret} @{fn}({args})') if ret != 'void' else self.emit(f'call void @{fn}({args})')


def intr(b, ret, name, typed_args):
    """Call an LLVM intrinsic: typed_args = [(type, value)]."""
    b.decls.add(f'declare {ret} @{name}({", ".join(t for t, _ in typed_args)})')
    args = ', '.join(f'{t} {v}' for t, v in typed_args)
    if ret == 'void':
        b.emit(f'call void @{name}({args})')
        return None
    return b.v(f'call {ret} @{name}({args})')


FAST = 'fast'

# ---------------------------------------------------------------------------------------------------------
# math


def unary_llvm(llvm_name):
    def f(b, s):
        t = s.ret
        return intr(b, t, f'llvm.{llvm_name}.{ivsuffix(t)}', [(t, '%a0')])
    return f


def binary_llvm(llvm_name):
    def f(b, s):
        t = s.ret
        return intr(b, t, f'llvm.{llvm_name}.{ivsuffix(t)}', [(t, '%a0'), (t, '%a1')])
    return f


def fclamp(b, s):
    t = s.ret
    lo = intr(b, t, f'llvm.maxnum.{ivsuffix(t)}', [(t, '%a0'), (t, '%a1')])
    return intr(b, t, f'llvm.minnum.{ivsuffix(t)}', [(t, lo), (t, '%a2')])


def fsaturate(b, s):
    t = s.ret
    lo = intr(b, t, f'llvm.maxnum.{ivsuffix(t)}', [(t, '%a0'), (t, splat(t, '0.0'))])
    return intr(b, t, f'llvm.minnum.{ivsuffix(t)}', [(t, lo), (t, splat(t, '1.0'))])


def iclamp(signed):
    def f(b, s):
        t = s.ret
        mx, mn = ('smax', 'smin') if signed else ('umax', 'umin')
        lo = intr(b, t, f'llvm.{mx}.{ivsuffix(t)}', [(t, '%a0'), (t, '%a1')])
        return intr(b, t, f'llvm.{mn}.{ivsuffix(t)}', [(t, lo), (t, '%a2')])
    return f


def fmix(b, s):
    t = s.ret
    d = b.v(f'fsub {FAST} {t} %a1, %a0')
    m = b.v(f'fmul {FAST} {t} {d}, %a2')
    return b.v(f'fadd {FAST} {t} %a0, {m}')


def fdot(b, s):
    t = s.params[0]
    n, e = vec(t)
    p = b.v(f'fmul {FAST} {t} %a0, %a1')
    acc = b.v(f'extractelement {t} {p}, i32 0')
    for i in range(1, n):
        x = b.v(f'extractelement {t} {p}, i32 {i}')
        acc = b.v(f'fadd {FAST} {e} {acc}, {x}')
    return acc


def ffract(b, s):
    t = s.ret
    fl = intr(b, t, f'llvm.floor.{ivsuffix(t)}', [(t, '%a0')])
    d = b.v(f'fsub {FAST} {t} %a0, {fl}')
    one_minus = '0xH3BFF' if vec(t)[1] == 'half' else '0x3FEFFFFFE0000000'
    return intr(b, t, f'llvm.minnum.{ivsuffix(t)}', [(t, d), (t, splat(t, one_minus))])


def fpow(b, s):
    # Metal pow/powr with fast math: exp2(y * log2(x))
    t = s.ret
    l = intr(b, t, f'llvm.log2.{ivsuffix(t)}', [(t, '%a0')])
    m = b.v(f'fmul {FAST} {t} {l}, %a1')
    return intr(b, t, f'llvm.exp2.{ivsuffix(t)}', [(t, m)])


def frsqrt(b, s):
    t = s.ret
    q = intr(b, t, f'llvm.sqrt.{ivsuffix(t)}', [(t, '%a0')])
    return b.v(f'fdiv {FAST} {t} {splat(t, "1.0")}, {q}')


def via_f32(fn):
    """Evaluate a half function in float (sinpi/cospi/tanh)."""
    def f(b, s):
        x = b.v('fpext half %a0 to float')
        r = fn(b, x)
        return b.v(f'fptrunc float {r} to half')
    return f


def _sinpi(b, x):
    m = b.v(f'fmul {FAST} float {x}, 0x400921FB60000000')
    return intr(b, 'float', 'llvm.sin.f32', [('float', m)])


def _cospi(b, x):
    m = b.v(f'fmul {FAST} float {x}, 0x400921FB60000000')
    return intr(b, 'float', 'llvm.cos.f32', [('float', m)])


def _tanh(b, x):
    # tanh(x) = 1 - 2 / (exp(2x) + 1); saturates correctly for large |x|
    t = b.v(f'fmul {FAST} float {x}, 2.0')
    e = intr(b, 'float', 'llvm.exp.f32', [('float', t)])
    d = b.v(f'fadd {FAST} float {e}, 1.0')
    q = b.v(f'fdiv {FAST} float 2.0, {d}')
    return b.v(f'fsub {FAST} float 1.0, {q}')


def fmax3(name):
    def f(b, s):
        t = s.ret
        a = intr(b, t, f'llvm.{name}.{ivsuffix(t)}', [(t, '%a0'), (t, '%a1')])
        return intr(b, t, f'llvm.{name}.{ivsuffix(t)}', [(t, a), (t, '%a2')])
    return f


def reduce_bool(kind):
    def f(b, s):
        t = s.params[0]
        return intr(b, 'i1', f'llvm.vector.reduce.{kind}.{ivsuffix(t)}', [(t, '%a0')])
    return f


def convert(b, s):
    # air.convert.<dk>.<dt>.<sk>.<st>: dk/sk in {f,s,u}
    m = re.match(r'air\.convert\.([fsu])\.(\w+)\.([fsu])\.(\w+)$', s.name)
    dk, sk = m.group(1), m.group(3)
    src, dst = s.params[0], s.ret
    se, de = vec(src)[1], vec(dst)[1]
    if dk == 'f' and sk == 'f':
        if se == de:
            return '%a0'
        op = 'fpext' if de == 'float' else 'fptrunc'
        return b.v(f'{op} {src} %a0 to {dst}')
    if dk == 'f':
        op = 'sitofp' if sk == 's' else 'uitofp'
        return b.v(f'{op} {src} %a0 to {dst}')
    if sk == 'f':
        # float -> integer: round toward zero, saturating (NaN -> 0)
        sat = 'fptosi.sat' if dk == 's' else 'fptoui.sat'
        return intr(b, dst, f'llvm.{sat}.{ivsuffix(dst)}.{ivsuffix(src)}', [(src, '%a0')])
    # integer -> integer
    sb, db = int(se[1:]), int(de[1:])
    if sb == db:
        return '%a0'
    if db < sb:
        return b.v(f'trunc {src} %a0 to {dst}')
    return b.v(f'{"sext" if sk == "s" else "zext"} {src} %a0 to {dst}')


# ---------------------------------------------------------------------------------------------------------
# wave32 SIMD-group and quad-group operations

def lane_id(b):
    return intr(b, 'i32', 'llvm.amdgcn.mbcnt.lo', [('i32', '-1'), ('i32', '0')])


def to_i32_parts(b, t, v):
    """Split a value into a list of i32 words (for bpermute) and return (parts, rebuild)."""
    n, e = vec(t)
    parts = []
    for i in range(n):
        x = v if n == 1 else b.v(f'extractelement {t} {v}, i32 {i}')
        if e == 'float':
            x = b.v(f'bitcast float {x} to i32')
        elif e == 'half':
            x = b.v(f'bitcast half {x} to i16')
            x = b.v(f'zext i16 {x} to i32')
        elif e == 'i16':
            x = b.v(f'zext i16 {x} to i32')
        elif e == 'i8':
            x = b.v(f'zext i8 {x} to i32')
        parts.append(x)

    def rebuild(words):
        out = f'poison' if n > 1 else None
        res = []
        for w in words:
            if e == 'float':
                w = b.v(f'bitcast i32 {w} to float')
            elif e == 'half':
                w = b.v(f'trunc i32 {w} to i16')
                w = b.v(f'bitcast i16 {w} to half')
            elif e in ('i16', 'i8'):
                w = b.v(f'trunc i32 {w} to {e}')
            res.append(w)
        if n == 1:
            return res[0]
        cur = 'poison'
        for i, w in enumerate(res):
            cur = b.v(f'insertelement {t} {cur}, {e} {w}, i32 {i}')
        return cur
    return parts, rebuild


def bpermute(b, lane_src, word):
    addr = b.v(f'shl i32 {lane_src}, 2')
    return intr(b, 'i32', 'llvm.amdgcn.ds.bpermute', [('i32', addr), ('i32', word)])


def shuffle_xor(b, s):
    t = s.ret
    lane = lane_id(b)
    m = b.v('zext i16 %a1 to i32')
    src = b.v(f'xor i32 {lane}, {m}')
    parts, rebuild = to_i32_parts(b, t, '%a0')
    return rebuild([bpermute(b, src, w) for w in parts])


def reduce_op(kind, width):
    """simd_sum/min/max (width 32) and quad_sum (width 4) via an xor butterfly over the active lanes.
    Inactive lanes contribute the identity (set.inactive inside a whole-wave region)."""
    def f(b, s):
        t = s.ret
        n, e = vec(t)
        is_f = e in FLOATS
        signed = '.s.' in s.name
        if kind == 'sum':
            ident = '0.0' if is_f else '0'
        elif kind == 'min':
            ident = ('0xH7C00' if e == 'half' else '0x7FF0000000000000') if is_f else ('-1' if not signed else str((1 << (int(e[1:]) - 1)) - 1))
        else:
            ident = ('0xHFC00' if e == 'half' else '0xFFF0000000000000') if is_f else ('0' if not signed else str(-(1 << (int(e[1:]) - 1))))
        lane = lane_id(b)
        cur = intr(b, t, f'llvm.amdgcn.set.inactive.{ivsuffix(t)}', [(t, '%a0'), (t, splat(t, ident))]) if width == 32 else '%a0'
        step = 1
        while step < width:
            src = b.v(f'xor i32 {lane}, {step}')
            parts, rebuild = to_i32_parts(b, t, cur)
            other = rebuild([bpermute(b, src, w) for w in parts])
            if kind == 'sum':
                cur = b.v(f'{"fadd " + FAST if is_f else "add"} {t} {cur}, {other}')
            else:
                fn = ('minnum' if kind == 'min' else 'maxnum') if is_f else \
                     (('smin' if kind == 'min' else 'smax') if signed else ('umin' if kind == 'min' else 'umax'))
                cur = intr(b, t, f'llvm.{fn}.{ivsuffix(t)}', [(t, cur), (t, other)])
            step *= 2
        if width == 32:
            cur = intr(b, t, f'llvm.amdgcn.strict.wwm.{ivsuffix(t)}', [(t, cur)])
        return cur
    return f


def simd_any(b, s):
    m = intr(b, 'i32', 'llvm.amdgcn.ballot.i32', [('i1', '%a0')])
    return b.v(f'icmp ne i32 {m}, 0')


def simd_all(b, s):
    m = intr(b, 'i32', 'llvm.amdgcn.ballot.i32', [('i1', '%a0')])
    e = intr(b, 'i32', 'llvm.amdgcn.ballot.i32', [('i1', 'true')])
    return b.v(f'icmp eq i32 {m}, {e}')


def simd_is_first(b, s):
    e = intr(b, 'i32', 'llvm.amdgcn.ballot.i32', [('i1', 'true')])
    first = intr(b, 'i32', 'llvm.cttz.i32', [('i32', e), ('i1', 'true')])
    return b.v(f'icmp eq i32 {lane_id(b)}, {first}')


# ---------------------------------------------------------------------------------------------------------
# textures

def desc(b, tex, storage=False):
    p = b.v(f'addrspacecast ptr addrspace(1) {tex} to ptr addrspace(4)')
    if storage:
        p = b.v(f'getelementptr inbounds i8, ptr addrspace(4) {p}, i64 32')
    return b.v(f'load <8 x i32>, ptr addrspace(4) {p}, align 32, !invariant.load !{{}}')


def sampler_desc(b, smp):
    return b.v(f'load <4 x i32>, ptr addrspace(4) {smp}, align 16, !invariant.load !{{}}')


def fp_twin(t):
    """Image intrinsics are typed on fp vectors; integer texels travel as the same bits."""
    n, e = vec(t)
    f = {'i16': 'half', 'i32': 'float'}.get(e, e)
    return t if f == e else (f'<{n} x {f}>' if n > 1 else f)


def img_call(b, vt, name_fmt, args):
    """Call an image intrinsic whose data type is vt (possibly integer); name_fmt has {s} for the suffix."""
    ft = fp_twin(vt)
    v = intr(b, ft, name_fmt.format(s=ivsuffix(ft)), args)
    return v if ft == vt else b.v(f'bitcast {ft} {v} to {vt}')


def coords_i32(b, ct, c):
    n, e = vec(ct)
    xs = []
    for i in range(n):
        x = b.v(f'extractelement {ct} {c}, i32 {i}')
        if e != 'i32':
            x = b.v(f'zext {e} {x} to i32')
        xs.append(x)
    return xs


def pack_result(b, rt, v):
    """AIR texture reads return {vec, i8}; the i8 is a residency flag (always resident here)."""
    r = b.v(f'insertvalue {rt} poison, {rt[2:rt.index(",")].strip()} {v}, 0')
    return b.v(f'insertvalue {rt} {r}, i8 1, 1')


def struct_first(rt):
    return rt[1:rt.index(',')].strip()


def read_tex(b, s):
    vt = struct_first(s.ret)
    array = 'texture_2d_array' in s.name
    d = desc(b, '%a0')
    if array:
        xs = coords_i32(b, s.params[2], '%a2')
        sl = '%a3' if s.params[3] == 'i32' else b.v(f'zext {s.params[3]} %a3 to i32')
        args = [('i32', '15'), ('i32', xs[0]), ('i32', xs[1]), ('i32', sl)]
        name = 'llvm.amdgcn.image.load.2darray.{s}.i32'
    else:
        xs = coords_i32(b, s.params[2], '%a2')
        args = [('i32', '15'), ('i32', xs[0]), ('i32', xs[1])]
        name = 'llvm.amdgcn.image.load.2d.{s}.i32'
    v = img_call(b, vt, name, args + [('<8 x i32>', d), ('i32', '0'), ('i32', '0')])
    r = b.v(f'insertvalue {s.ret} poison, {vt} {v}, 0')
    return b.v(f'insertvalue {s.ret} {r}, i8 1, 1')


def pack_offset(b, ot, o):
    """<2 x i32> texel offset -> MIMG offset dword (6-bit signed fields at bits 0 and 8)."""
    x = b.v(f'extractelement {ot} {o}, i32 0')
    y = b.v(f'extractelement {ot} {o}, i32 1')
    x = b.v(f'and i32 {x}, 63')
    y = b.v(f'and i32 {y}, 63')
    y = b.v(f'shl i32 {y}, 8')
    return b.v(f'or i32 {x}, {y}')


def is_zero_vec(v):
    return v in ('zeroinitializer',) or re.fullmatch(r'<i32 0, i32 0>', v or '') is not None


def sample_tex(b, s):
    vt = struct_first(s.ret)
    array = 'texture_2d_array' in s.name
    d = desc(b, '%a0')
    smp = sampler_desc(b, '%a1')
    cx = b.v(f'extractelement <2 x float> %a2, i32 0')
    cy = b.v(f'extractelement <2 x float> %a2, i32 1')
    coords = [('float', cx), ('float', cy)]
    if array:
        sl = b.v('uitofp i32 %a3 to float')
        coords.append(('float', sl))
        off_t, off_v = s.params[5], '%a5'
        dim = '2darray'
    else:
        off_t, off_v = s.params[4], '%a4'
        dim = '2d'
    off = pack_offset(b, off_t, off_v)
    v = img_call(b, vt, 'llvm.amdgcn.image.sample.lz.o.{dim}.{s}.f32'.replace('{dim}', dim),
             [('i32', '15'), ('i32', off)] + coords +
             [('<8 x i32>', d), ('<4 x i32>', smp), ('i1', 'false'), ('i32', '0'), ('i32', '0')])
    r = b.v(f'insertvalue {s.ret} poison, {vt} {v}, 0')
    return b.v(f'insertvalue {s.ret} {r}, i8 1, 1')


def gather_tex(b, s):
    vt = struct_first(s.ret)
    d = desc(b, '%a0')
    smp = sampler_desc(b, '%a1')
    cx = b.v('extractelement <2 x float> %a2, i32 0')
    cy = b.v('extractelement <2 x float> %a2, i32 1')
    off = pack_offset(b, s.params[4], '%a4')
    # component selects the dmask bit; it is a constant in every MetalFX kernel, so lower it per value
    outs = []
    for comp in range(4):
        outs.append(img_call(b, vt, 'llvm.amdgcn.image.gather4.lz.o.2d.{s}.f32',
                         [('i32', str(1 << comp)), ('i32', off), ('float', cx), ('float', cy),
                          ('<8 x i32>', d), ('<4 x i32>', smp), ('i1', 'false'), ('i32', '0'), ('i32', '0')]))
    sel = outs[0]
    for comp in range(1, 4):
        c = b.v(f'icmp eq i32 %a5, {comp}')
        sel = b.v(f'select i1 {c}, {vt} {outs[comp]}, {vt} {sel}')
    r = b.v(f'insertvalue {s.ret} poison, {vt} {sel}, 0')
    return b.v(f'insertvalue {s.ret} {r}, i8 1, 1')


def write_tex(b, s):
    array = 'texture_2d_array' in s.name
    d = desc(b, '%a0', storage=True)
    xs = coords_i32(b, s.params[1], '%a1')
    if array:
        sl = '%a2' if s.params[2] == 'i32' else b.v(f'zext {s.params[2]} %a2 to i32')
        vt, vv = s.params[3], '%a3'
        cargs = [('i32', xs[0]), ('i32', xs[1]), ('i32', sl)]
        dim = '2darray'
    else:
        vt, vv = s.params[2], '%a2'
        cargs = [('i32', xs[0]), ('i32', xs[1])]
        dim = '2d'
    ft = fp_twin(vt)
    if ft != vt:
        vv = b.v(f'bitcast {vt} {vv} to {ft}')
        vt = ft
    intr(b, 'void', f'llvm.amdgcn.image.store.{dim}.{ivsuffix(vt)}.i32',
         [(vt, vv), ('i32', '15')] + cargs + [('<8 x i32>', d), ('i32', '0'), ('i32', '0')])
    return None


def tex_size(which):
    def f(b, s):
        d = desc(b, '%a0')
        v = intr(b, '<4 x float>', 'llvm.amdgcn.image.getresinfo.2d.v4f32.i32',
                 [('i32', '15'), ('i32', '%a1'), ('<8 x i32>', d), ('i32', '0'), ('i32', '0')])
        x = b.v(f'extractelement <4 x float> {v}, i32 {which}')
        return b.v(f'bitcast float {x} to i32')
    return f


def is_null_tex(b, s):
    p = b.v('addrspacecast ptr addrspace(1) %a0 to ptr addrspace(4)')
    d = b.v(f'load <8 x i32>, ptr addrspace(4) {p}, align 32, !invariant.load !{{}}')
    w = b.v(f'extractelement <8 x i32> {d}, i32 1')
    w3 = b.v(f'extractelement <8 x i32> {d}, i32 3')
    o = b.v(f'or i32 {w}, {w3}')
    return b.v(f'icmp eq i32 {o}, 0')


def atomic_max_tex(b, s):
    d = desc(b, '%a0', storage=True)
    xs = coords_i32(b, s.params[1], '%a1')
    val = b.v(f'extractelement <4 x i32> %a3, i32 0')
    old = intr(b, 'i32', 'llvm.amdgcn.image.atomic.umax.2d.i32.i32',
               [('i32', val), ('i32', xs[0]), ('i32', xs[1]), ('<8 x i32>', d), ('i32', '0'), ('i32', '0')])
    return b.v(f'insertelement <4 x i32> zeroinitializer, i32 {old}, i32 0')


def read_sampler(b, s):
    return 'null'


# ---------------------------------------------------------------------------------------------------------
# atomics, barriers

def atomic_rmw(op, space):
    def f(b, s):
        scope = 'syncscope("agent")' if space == 'global' else 'syncscope("workgroup")'
        return b.v(f'atomicrmw {op} ptr addrspace({1 if space == "global" else 3}) %a0, i32 %a1 {scope} monotonic, align 4')
    return f


def atomic_load_local(b, s):
    return b.v('load atomic i32, ptr addrspace(3) %a0 syncscope("workgroup") monotonic, align 4')


def atomic_store_local(b, s):
    b.emit('store atomic i32 %a1, ptr addrspace(3) %a0 syncscope("workgroup") monotonic, align 4')
    return None


def wg_barrier(b, s):
    b.emit('fence syncscope("workgroup") release')
    intr(b, 'void', 'llvm.amdgcn.s.barrier', [])
    b.emit('fence syncscope("workgroup") acquire')
    return None


def imageblock_store(b, s):
    """air.store.implicit_imageblock(rt, value, index, coord-in-tile, ...): store to the render target at
    workgroup * tile + coord (m4r dispatches one threadgroup per tile)."""
    vt = s.params[1]
    d = desc(b, '%a0', storage=True)
    tx, ty = CTX['tg'][0], CTX['tg'][1]
    gx = intr(b, 'i32', 'llvm.amdgcn.workgroup.id.x', [])
    gy = intr(b, 'i32', 'llvm.amdgcn.workgroup.id.y', [])
    lx, ly = coords_i32(b, s.params[3], '%a3')
    ox = b.v(f'mul i32 {gx}, {tx}'); oy = b.v(f'mul i32 {gy}, {ty}')
    x = b.v(f'add i32 {ox}, {lx}'); y = b.v(f'add i32 {oy}, {ly}')
    intr(b, 'void', f'llvm.amdgcn.image.store.2d.{ivsuffix(fp_twin(vt))}.i32',
         [(fp_twin(vt), '%a1'), ('i32', '15'), ('i32', x), ('i32', y), ('<8 x i32>', d), ('i32', '0'), ('i32', '0')])
    return None


def rt_store_def(vt):
    """@__m4r_rt_write(value, rt, x, y): a fragment function's return value to the render target."""
    b = Body()
    d = desc(b, '%rt', storage=True)
    intr(b, 'void', f'llvm.amdgcn.image.store.2d.{ivsuffix(fp_twin(vt))}.i32',
         [(fp_twin(vt), '%v'), ('i32', '15'), ('i32', '%x'), ('i32', '%y'), ('<8 x i32>', d), ('i32', '0'), ('i32', '0')])
    text = [f'define internal void @__m4r_rt_write({vt} %v, ptr addrspace(1) %rt, i32 %x, i32 %y) #900 {{'] + b.lines + ['  ret void', '}']
    return '\n'.join(text), b.decls


def identity(b, s):
    return '%a0'


# ---------------------------------------------------------------------------------------------------------

TABLE = [
    (r'air\.convert\.', convert),
    (r'air\.normalize_function_constant_predicate\.', identity),
    (r'air\.(fast_)?fabs\.', unary_llvm('fabs')),
    (r'air\.(fast_)?floor\.', unary_llvm('floor')),
    (r'air\.(fast_)?ceil\.', unary_llvm('ceil')),
    (r'air\.(fast_)?sqrt\.', unary_llvm('sqrt')),
    (r'air\.(fast_)?rint\.', unary_llvm('rint')),
    (r'air\.(fast_)?round\.', unary_llvm('round')),
    (r'air\.(fast_)?exp\.', unary_llvm('exp')),
    (r'air\.(fast_)?exp2\.', unary_llvm('exp2')),
    (r'air\.(fast_)?log\.', unary_llvm('log')),
    (r'air\.(fast_)?log2\.', unary_llvm('log2')),
    (r'air\.(fast_)?rsqrt\.', frsqrt),
    (r'air\.(fast_)?fract\.', ffract),
    (r'air\.(fast_)?fmax3\.', fmax3('maxnum')),
    (r'air\.(fast_)?fmin3\.', fmax3('minnum')),
    (r'air\.(fast_)?fmax\.', binary_llvm('maxnum')),
    (r'air\.(fast_)?fmin\.', binary_llvm('minnum')),
    (r'air\.(fast_)?clamp\.s\.', iclamp(True)),
    (r'air\.(fast_)?clamp\.u\.', iclamp(False)),
    (r'air\.(fast_)?clamp\.', fclamp),
    (r'air\.(fast_)?saturate\.', fsaturate),
    (r'air\.(fast_)?mix\.', fmix),
    (r'air\.(fast_)?dot\.', fdot),
    (r'air\.(fast_)?powr?\.', fpow),
    (r'air\.(fast_)?sinpi\.', via_f32(_sinpi)),
    (r'air\.(fast_)?cospi\.', via_f32(_cospi)),
    (r'air\.(fast_)?tanh\.', via_f32(_tanh)),
    (r'air\.abs\.s\.', unary_llvm('abs_s')),
    (r'air\.min\.s\.', binary_llvm('smin')),
    (r'air\.min\.u\.', binary_llvm('umin')),
    (r'air\.max\.s\.', binary_llvm('smax')),
    (r'air\.max\.u\.', binary_llvm('umax')),
    (r'air\.all\.', reduce_bool('and')),
    (r'air\.any\.', reduce_bool('or')),
    (r'air\.(simd|quad)_shuffle_xor\.', shuffle_xor),
    (r'air\.simd_sum\.', reduce_op('sum', 32)),
    (r'air\.simd_min\.', reduce_op('min', 32)),
    (r'air\.simd_max\.', reduce_op('max', 32)),
    (r'air\.quad_sum\.', reduce_op('sum', 4)),
    (r'air\.simd_any$', simd_any),
    (r'air\.simd_all$', simd_all),
    (r'air\.simd_is_first$', simd_is_first),
    (r'air\.read_texture_2d', read_tex),
    (r'air\.sample_texture_2d', sample_tex),
    (r'air\.gather_texture_2d', gather_tex),
    (r'air\.write_texture_2d', write_tex),
    (r'air\.get_width_texture_2d', tex_size(0)),
    (r'air\.get_height_texture_2d', tex_size(1)),
    (r'air\.is_null_texture_2d', is_null_tex),
    (r'air\.atomic_fetch_max_explicit_texture_2d', atomic_max_tex),
    (r'air\.get_read_sampler$', read_sampler),
    (r'air\.atomic\.global\.add\.', atomic_rmw('add', 'global')),
    (r'air\.atomic\.global\.and\.', atomic_rmw('and', 'global')),
    (r'air\.atomic\.global\.or\.', atomic_rmw('or', 'global')),
    (r'air\.atomic\.global\.max\.u\.', atomic_rmw('umax', 'global')),
    (r'air\.atomic\.global\.min\.u\.', atomic_rmw('umin', 'global')),
    (r'air\.atomic\.local\.add\.', atomic_rmw('add', 'local')),
    (r'air\.atomic\.local\.load\.', atomic_load_local),
    (r'air\.atomic\.local\.store\.', atomic_store_local),
    (r'air\.wg\.barrier$', wg_barrier),
    (r'air\.store\.implicit_imageblock\.', imageblock_store),
]


def implement(decl_line):
    """Return (definition text, set of extra declarations) for one AIR declaration line."""
    s = Sig(decl_line)
    for pat, fn in TABLE:
        if re.match(pat, s.name):
            b = Body()
            if s.name.startswith('air.abs.s.'):
                r = intr(b, s.ret, f'llvm.abs.{ivsuffix(s.ret)}', [(s.ret, '%a0'), ('i1', 'false')])
            else:
                r = fn(b, s)
            text = [s.header()] + b.lines
            text.append('  ret void' if s.ret == 'void' else f'  ret {s.ret} {r}')
            text.append('}')
            return '\n'.join(text), b.decls
    raise NotImplementedError(s.name)


# ---------------------------------------------------------------------------------------------------------
# Apple GPU (AGX) intrinsics used by MetalFX's MXU kernels.
#
# `llvm.agx3.igemm` / `llvm.agx3.fgemm16x16x16` are 16x16x16 matrix multiply-accumulates over a SIMD group: each
# of the 32 lanes holds 8 elements of A, of B and of the accumulator. Which (row, column) a lane's element is,
# is the hardware's convention; tests/probe_mxu.py measures it from MetalFX's own kernels (see
# docs/mxu.md) and air/mxu_layout.py holds the result. The lowering below exchanges the operands through LDS in
# that layout and computes each accumulator element with dot-product instructions.
#
# `load/store.with.emask` move 4 consecutive elements; element i takes part when bit i of (mask & mask2) is set
# (masked-off loads return 0). MetalFX passes either 15 or 0 as the first mask and 15 as the second.

MXU_MODE = {'mode': 'lds'}
CTX = {'tg': [1, 1, 1]}   # threadgroup (= tile) size of the kernel being translated   # 'lds' (table-driven) or 'probe'


def emask_load(b, s):
    t = s.ret
    n, e = vec(t)
    m = b.v('and i16 %a1, %a2')
    out = 'zeroinitializer'
    for i in range(n):
        bit = b.v(f'and i16 {m}, {1 << i}')
        on = b.v(f'icmp ne i16 {bit}, 0')
        p = b.v(f'getelementptr inbounds {e}, ptr addrspace(1) %a0, i64 {i}')
        out_i = b.v(f'call {e} @__m4r_maskload_{e}(i1 {on}, ptr addrspace(1) {p})')
        out = b.v(f'insertelement {t} {out}, {e} {out_i}, i32 {i}')
    b.decls.add(f'declare {e} @__m4r_maskload_{e}(i1, ptr addrspace(1))')
    return out


def emask_store(b, s):
    t = s.params[1]
    n, e = vec(t)
    m = b.v('and i16 %a2, %a3')
    for i in range(n):
        bit = b.v(f'and i16 {m}, {1 << i}')
        on = b.v(f'icmp ne i16 {bit}, 0')
        p = b.v(f'getelementptr inbounds {e}, ptr addrspace(1) %a0, i64 {i}')
        x = b.v(f'extractelement {t} %a1, i32 {i}')
        b.emit(f'call void @__m4r_maskstore_{e}(i1 {on}, ptr addrspace(1) {p}, {e} {x})')
    b.decls.add(f'declare void @__m4r_maskstore_{e}(i1, ptr addrspace(1), {e})')
    return None


def plain_load(b, s):
    return b.v(f'load {s.ret}, ptr addrspace(1) %a0, align 1')


def _lds_wave_base(b, bytes_per_wave):
    lid = intr(b, 'i32', 'llvm.amdgcn.workitem.id.x', [])
    w = b.v(f'lshr i32 {lid}, 5')
    off = b.v(f'mul i32 {w}, {bytes_per_wave}')
    return b.v(f'getelementptr inbounds i8, ptr addrspace(3) @__m4r_mxu, i32 {off}')


def _layout_index(b, table, lane, slot):
    """byte index into the 256-element tile for (lane, slot) from a 256-entry i8 table global."""
    idx = b.v(f'mul i32 {lane}, 8')
    idx = b.v(f'add i32 {idx}, {slot}')
    p = b.v(f'getelementptr inbounds [256 x i8], ptr addrspace(4) @{table}, i32 0, i32 {idx}')
    v = b.v(f'load i8, ptr addrspace(4) {p}, align 1')
    return b.v(f'zext i8 {v} to i32')


def mxu_gemm(kind):
    """kind 'i' (8 x i8 per lane packed in i64, i32 accumulators) or 'f' (8 x half, f32 accumulators)."""
    def f(b, s):
        if kind == 'i':
            A, B, C = '%a4', '%a6', '%a8'
            # operand type immediates: 11 = u8, 75 = s8 (the only pairs MetalFX uses: 11/11 and 75/75)
            signed = b.v('icmp eq i16 %a5, 75')
            et, esz, at = 'i8', 1, '<8 x i32>'
            av = b.v(f'bitcast i64 {A} to <8 x i8>')
            bv = b.v(f'bitcast i64 {B} to <8 x i8>')
        else:
            A, B, C = '%a0', '%a2', '%a4'
            et, esz, at = 'half', 2, '<8 x float>'
            av, bv = A, B
        lane = lane_id(b)
        if MXU_MODE['mode'] == 'probe':
            # record the operands of SIMD group 0 of threadgroup 0 (last call wins) and give every accumulator its id
            wg = intr(b, 'i32', 'llvm.amdgcn.workgroup.id.x', [])
            lid = intr(b, 'i32', 'llvm.amdgcn.workitem.id.x', [])
            first = b.v(f'or i32 {wg}, {lid}')
            sg0 = b.v(f'icmp ult i32 {first}, 32')
            for opi, (vv, base) in enumerate(((av, 0), (bv, 256 * esz))):
                for sl in range(8):
                    x = b.v(f'extractelement <8 x {et}> {vv}, i32 {sl}')
                    o = b.v(f'mul i32 {lane}, {8 * esz}')
                    o = b.v(f'add i32 {o}, {base + sl * esz}')
                    p = b.v(f'getelementptr inbounds i8, ptr addrspace(1) @__m4r_probe, i32 {o}')
                    b.emit(f'call void @__m4r_maskstore_{et}(i1 {sg0}, ptr addrspace(1) {p}, {et} {x})')
            b.decls.add(f'declare void @__m4r_maskstore_{et}(i1, ptr addrspace(1), {et})')
            zero = b.v(f'icmp eq {at} {C}, zeroinitializer')
            allz = intr(b, 'i1', 'llvm.vector.reduce.and.v8i1', [('<8 x i1>', zero)])
            ids = 'poison'
            for sl in range(8):
                v = b.v(f'mul i32 {lane}, 8')
                v = b.v(f'add i32 {v}, {sl}')
                if kind == 'f':
                    v = b.v(f'uitofp i32 {v} to float')
                ids = b.v(f'insertelement {at} {ids}, {"i32" if kind == "i" else "float"} {v}, i32 {sl}')
            return b.v(f'select i1 {allz}, {at} {ids}, {at} {C}')
        # table-driven: scatter A and B into LDS in logical [row][k] / [col][k] order, then dot products
        base = _lds_wave_base(b, 512 * esz)
        for (vv, table, off) in ((av, '__m4r_mxu_a', 0), (bv, '__m4r_mxu_bt', 256 * esz)):
            for sl in range(8):
                x = b.v(f'extractelement <8 x {et}> {vv}, i32 {sl}')
                idx = _layout_index(b, table, lane, sl)
                o = b.v(f'mul i32 {idx}, {esz}')
                o = b.v(f'add i32 {o}, {off}')
                p = b.v(f'getelementptr inbounds i8, ptr addrspace(3) {base}, i32 {o}')
                b.emit(f'store {et} {x}, ptr addrspace(3) {p}, align {esz}')
        b.emit('fence syncscope("wavefront") release')
        b.emit('fence syncscope("wavefront") acquire')
        out = C
        for sl in range(8):
            rc = _layout_index(b, '__m4r_mxu_c', lane, sl)   # row * 16 + col
            row = b.v(f'lshr i32 {rc}, 4')
            col = b.v(f'and i32 {rc}, 15')
            acc = b.v(f'extractelement {at} {C}, i32 {sl}')
            for q in range(16 // (4 if kind == 'i' else 2)):
                step = 4 if kind == 'i' else 2
                ao = b.v(f'mul i32 {row}, {16 * esz}')
                ao = b.v(f'add i32 {ao}, {q * step * esz}')
                bo = b.v(f'mul i32 {col}, {16 * esz}')
                bo = b.v(f'add i32 {bo}, {256 * esz + q * step * esz}')
                pa = b.v(f'getelementptr inbounds i8, ptr addrspace(3) {base}, i32 {ao}')
                pb = b.v(f'getelementptr inbounds i8, ptr addrspace(3) {base}, i32 {bo}')
                if kind == 'i':
                    xa = b.v(f'load i32, ptr addrspace(3) {pa}, align 4')
                    xb = b.v(f'load i32, ptr addrspace(3) {pb}, align 4')
                    su = intr(b, 'i32', 'llvm.amdgcn.udot4', [('i32', xa), ('i32', xb), ('i32', acc), ('i1', 'false')])
                    ss = intr(b, 'i32', 'llvm.amdgcn.sdot4', [('i32', xa), ('i32', xb), ('i32', acc), ('i1', 'false')])
                    acc = b.v(f'select i1 {signed}, i32 {ss}, i32 {su}')
                else:
                    xa = b.v(f'load <2 x half>, ptr addrspace(3) {pa}, align 4')
                    xb = b.v(f'load <2 x half>, ptr addrspace(3) {pb}, align 4')
                    acc = intr(b, 'float', 'llvm.amdgcn.fdot2', [('<2 x half>', xa), ('<2 x half>', xb), ('float', acc), ('i1', 'false')])
            out = b.v(f'insertelement {at} {out}, {"i32" if kind == "i" else "float"} {acc}, i32 {sl}')
        return out
    return f


AGX_TABLE = [
    (r'm4r\.agx3\.load\.with\.emask\.', emask_load),
    (r'm4r\.agx3\.store\.with\.emask\.', emask_store),
    (r'm4r\.agx2\.device\.load\.cachectl\.', plain_load),
    (r'm4r\.agx3\.igemm\.', mxu_gemm('i')),
    (r'm4r\.agx3\.fgemm16x16x16\.', mxu_gemm('f')),
]


def implement_agx(decl_line):
    m = re.match(r'declare\s+(.*?)\s*@(m4r\.agx[\w.]+)\((.*)\)', decl_line.strip())
    s = Sig(decl_line.replace('@m4r.agx', '@air.__agx'))
    s.name = m.group(2)
    for pat, fn in AGX_TABLE:
        if re.match(pat, s.name):
            b = Body()
            r = fn(b, s)
            args = ', '.join(f'{t} %a{i}' for i, t in enumerate(s.params))
            text = [f'define internal {s.ret} @{s.name}({args}) #900 {{'] + b.lines
            text.append('  ret void' if s.ret == 'void' else f'  ret {s.ret} {r}')
            text.append('}')
            return '\n'.join(text), b.decls
    raise NotImplementedError(s.name)


def mask_helpers():
    """Masked element load/store helpers (branch around the access)."""
    out = []
    for e, al in (('i8', 1), ('i16', 2), ('half', 2), ('float', 4)):
        out.append(f'''define internal {e} @__m4r_maskload_{e}(i1 %on, ptr addrspace(1) %p) #900 {{
  br i1 %on, label %l, label %z
l:
  %v = load {e}, ptr addrspace(1) %p, align {al}
  ret {e} %v
z:
  ret {e} {'0' if e[0] == 'i' else '0.0'}
}}
define internal void @__m4r_maskstore_{e}(i1 %on, ptr addrspace(1) %p, {e} %v) #900 {{
  br i1 %on, label %s, label %z
s:
  store {e} %v, ptr addrspace(1) %p, align {al}
  ret void
z:
  ret void
}}''')
    return '\n'.join(out)
