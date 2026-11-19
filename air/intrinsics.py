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
import os
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
    if n == 2 and e in ('half', 'i16'):   # one 32-bit word: a packed pair moves through a lane shuffle as one value
        return [b.v(f'bitcast {t} {v} to i32')], (lambda words: b.v(f'bitcast i32 {words[0]} to {t}'))
    parts = []
    for i in range(n):
        x = v if n == 1 else b.v(f'extractelement {t} {v}, i32 {i}')
        if e == 'float':
            x = b.v(f'bitcast float {x} to i32')
        elif e == 'half':   # upper 16 bits undefined: nothing reads them, and zeroing them costs a v_and per word
            x = b.v(f'insertelement <2 x half> poison, half {x}, i32 0')
            x = b.v(f'bitcast <2 x half> {x} to i32')
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
                w = b.v(f'bitcast i32 {w} to <2 x half>')
                w = b.v(f'extractelement <2 x half> {w}, i32 0')
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

        def butterfly(cur):
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
            return cur

        if width != 32:
            return butterfly('%a0')
        # inactive lanes contribute the identity (a fast path for full waves, tried 2026-11-18, doubled the code and spilled)
        cur = intr(b, t, f'llvm.amdgcn.set.inactive.{ivsuffix(t)}', [(t, '%a0'), (t, splat(t, ident))])
        cur = butterfly(cur)
        cur = intr(b, t, f'llvm.amdgcn.strict.wwm.{ivsuffix(t)}', [(t, cur)])
        return cur
    return f


def sum_scatter32(b, s):
    """air.simd_sum_scatter32.f16(x_0 .. x_31) (made by m4r_air.scatter_simd_sums): lane K gets simd_sum(x_K).

    Recursive halving: at xor step 2^i (i = 0..4, the order of the simd_sum butterfly) a lane keeps the values whose index has
    bit i equal to its own lane bit i and adds the partner's copies of them; after five steps lane K holds value K. Each
    candidate's sum is the butterfly's addition tree (fadd is commutative, so both lanes of a pair hold the same partial sum
    there too), and inactive lanes contribute +0 as in reduce_op: the results are bit-identical to 32 simd_sums.
    Values travel two halves per 32-bit word: word w holds values (2w, 2w + 1) of the current set, so one v_perm picks the
    kept halves of two words, one picks the sent ones, and one lane move and one packed add finish both."""
    lane = lane_id(b)
    words = []
    for i in range(16):
        if s.name.endswith('.absdot.f16'):
            # value K = dot(|d_K|, (1, 1)) = |d_K.x| + |d_K.y|: the x halves of d_2i and d_2i+1 in one word, the y halves in
            # another, one packed add (the same two f16 additions as Apple's dot, just side by side)
            lo = intr(b, 'i32', 'llvm.amdgcn.perm', [('i32', b.v(f'bitcast <2 x half> %a{2 * i + 1} to i32')),
                                                      ('i32', b.v(f'bitcast <2 x half> %a{2 * i} to i32')), ('i32', str(0x05040100))])
            hi = intr(b, 'i32', 'llvm.amdgcn.perm', [('i32', b.v(f'bitcast <2 x half> %a{2 * i + 1} to i32')),
                                                      ('i32', b.v(f'bitcast <2 x half> %a{2 * i} to i32')), ('i32', str(0x07060302))])
            lo = b.v(f'and i32 {lo}, {0x7fff7fff}')
            hi = b.v(f'and i32 {hi}, {0x7fff7fff}')
            v = b.v(f'fadd {FAST} <2 x half> {b.v(f"bitcast i32 {lo} to <2 x half>")}, {b.v(f"bitcast i32 {hi} to <2 x half>")}')
        else:
            v = b.v(f'insertelement <2 x half> poison, half %a{2 * i}, i32 0')
            v = b.v(f'insertelement <2 x half> {v}, half %a{2 * i + 1}, i32 1')
        w = b.v(f'bitcast <2 x half> {v} to i32')
        words.append(intr(b, 'i32', 'llvm.amdgcn.set.inactive.i32', [('i32', w), ('i32', '0')]))
    for bit in range(4):
        st = 1 << bit
        lb = b.v(f'and i32 {lane}, {st}')
        on = b.v(f'icmp ne i32 {lb}, 0')
        selk = b.v(f'select i1 {on}, i32 {0x07060302}, i32 {0x05040100}')   # odd values (bit set) or even ones
        sels = b.v(f'select i1 {on}, i32 {0x05040100}, i32 {0x07060302}')
        src = b.v(f'xor i32 {lane}, {st}')
        nw = []
        for j in range(0, len(words), 2):
            keep = intr(b, 'i32', 'llvm.amdgcn.perm', [('i32', words[j + 1]), ('i32', words[j]), ('i32', selk)])
            send = intr(b, 'i32', 'llvm.amdgcn.perm', [('i32', words[j + 1]), ('i32', words[j]), ('i32', sels)])
            recv = bpermute(b, src, send)
            kh = b.v(f'bitcast i32 {keep} to <2 x half>')
            rh = b.v(f'bitcast i32 {recv} to <2 x half>')
            sm = b.v(f'fadd {FAST} <2 x half> {kh}, {rh}')
            nw.append(b.v(f'bitcast <2 x half> {sm} to i32'))
        words = nw
    w = words[0]
    lb = b.v(f'and i32 {lane}, 16')
    on = b.v(f'icmp ne i32 {lb}, 0')
    hi = b.v(f'lshr i32 {w}, 16')
    keep = b.v(f'select i1 {on}, i32 {hi}, i32 {w}')
    send = b.v(f'select i1 {on}, i32 {w}, i32 {hi}')
    src = b.v(f'xor i32 {lane}, 16')
    recv = bpermute(b, src, send)
    kh = b.v(f'bitcast i16 {b.v(f"trunc i32 {keep} to i16")} to half')
    rh = b.v(f'bitcast i16 {b.v(f"trunc i32 {recv} to i16")} to half')
    r = b.v(f'fadd {FAST} half {kh}, {rh}')
    return intr(b, 'half', 'llvm.amdgcn.strict.wwm.f16', [('half', r)])


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


def compute_imageblock_unused(b, s):
    """Compute-kernel imageblocks (air.imageblock_data, air.write_imageblock_slice_to_texture_2d) are the other branch of the
    `using_compute_imageblock` function constant; with the tile pipeline's constant they are dead code. translate() refuses
    the kernel when the constant selects them."""
    return 'null' if s.ret != 'void' else None


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
    (r'air\.simd_sum_scatter32\.(absdot\.)?f16$', sum_scatter32),
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
    (r'air\.(imageblock_data|write_imageblock_slice_to_texture_2d)', compute_imageblock_unused),
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

MXU_MODE = {'mode': 'lds', 'int': os.environ.get('M4R_MXU_INT', 'wmma'),   # integer GEMM: 'wmma' or 'dot' (udot4/sdot4)
            'fp16': os.environ.get('M4R_MXU_FP16', 'wmma')}   # fp16 GEMM: 'wmma', 'closed' (dot2 chain) or 'table'
CTX = {'tg': [1, 1, 1]}   # threadgroup (= tile) size of the kernel being translated   # 'lds' (table-driven) or 'probe'


ESIZE = {'i8': 1, 'i16': 2, 'half': 2, 'float': 4, 'i32': 4}


def emask_load(b, s):
    """All elements on (MetalFX passes 15 or 0): one vector load; otherwise element by element. Same values either way: a
    full mask makes every address of the vector one Apple's kernel reads."""
    t = s.ret
    n, e = vec(t)
    m = b.v('and i16 %a1, %a2')
    allm = (1 << n) - 1
    mm = b.v(f'and i16 {m}, {allm}')
    full = b.v(f'icmp eq i16 {mm}, {allm}')
    b.emit(f'br i1 {full}, label %ld_full, label %ld_part')
    b.emit('ld_full:')
    vf = b.v(f'load {t}, ptr addrspace(1) %a0, align {ESIZE[e]}')
    b.emit('br label %ld_end')
    b.emit('ld_part:')
    out = 'zeroinitializer'
    for i in range(n):
        bit = b.v(f'and i16 {m}, {1 << i}')
        on = b.v(f'icmp ne i16 {bit}, 0')
        p = b.v(f'getelementptr inbounds {e}, ptr addrspace(1) %a0, i64 {i}')
        out_i = b.v(f'call {e} @__m4r_maskload_{e}(i1 {on}, ptr addrspace(1) {p})')
        out = b.v(f'insertelement {t} {out}, {e} {out_i}, i32 {i}')
    b.emit('br label %ld_end')
    b.emit('ld_end:')
    b.decls.add(f'declare {e} @__m4r_maskload_{e}(i1, ptr addrspace(1))')
    return b.v(f'phi {t} [ {vf}, %ld_full ], [ {out}, %ld_part ]')


def emask_store(b, s):
    t = s.params[1]
    n, e = vec(t)
    m = b.v('and i16 %a2, %a3')
    allm = (1 << n) - 1
    mm = b.v(f'and i16 {m}, {allm}')
    full = b.v(f'icmp eq i16 {mm}, {allm}')
    b.emit(f'br i1 {full}, label %st_full, label %st_part')
    b.emit('st_full:')
    b.emit(f'store {t} %a1, ptr addrspace(1) %a0, align {ESIZE[e]}')
    b.emit('br label %st_end')
    b.emit('st_part:')
    for i in range(n):
        bit = b.v(f'and i16 {m}, {1 << i}')
        on = b.v(f'icmp ne i16 {bit}, 0')
        p = b.v(f'getelementptr inbounds {e}, ptr addrspace(1) %a0, i64 {i}')
        x = b.v(f'extractelement {t} %a1, i32 {i}')
        b.emit(f'call void @__m4r_maskstore_{e}(i1 {on}, ptr addrspace(1) {p}, {e} {x})')
    b.emit('br label %st_end')
    b.emit('st_end:')
    b.decls.add(f'declare void @__m4r_maskstore_{e}(i1, ptr addrspace(1), {e})')
    return None


def plain_load(b, s):
    return b.v(f'load {s.ret}, ptr addrspace(1) %a0, align 1')


# One LDS slot per wave, the same size for every GEMM kind: an fp16 tile (A and B^T, 256 half each) needs 1024 bytes and a
# kernel may mix integer and fp16 GEMMs (SBBRNet), so slots sized per kind would overlap between waves in different layers.
MXU_WAVE_BYTES = 1024


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


def _mxu_closed_forms_ok():
    """The measured lane layout (air/mxu_layout.py) has a closed form in the lane bits; the lowering below uses it."""
    import mxu_layout
    for lane in range(32):
        a0 = ((lane & 1) << 2) | ((lane & 2) << 3) | ((lane & 4) << 3) | (lane & 8) | ((lane & 16) << 2)
        c0 = ((lane & 1) << 2) | (lane & 8)
        kk = ((lane & 2) >> 1) | ((lane & 4) >> 1) | ((lane & 16) >> 2)
        for sl in range(8):
            if mxu_layout.A[lane * 8 + sl] != a0 + (sl & 3) + (128 if sl >= 4 else 0): return False
            if mxu_layout.C[lane * 8 + sl] != a0 + (sl & 3) + (128 if sl >= 4 else 0): return False
            if mxu_layout.BT[lane * 8 + sl] != ((c0 + (sl & 3)) << 4) + kk + (8 if sl >= 4 else 0): return False
    return True


WMMA_I8 = 'llvm.amdgcn.wmma.i32.16x16x16.iu8.v8i32.v4i32'


def _igemm_wmma(b, lane):
    """D = C + A.B for 16x16x16 u8/s8 with RDNA3 WMMA. Exact (integer), so equal to the dot-product lowering. The operands
    still go through LDS: Apple's lane layout holds 4 consecutive k of two rows of A, but WMMA wants a whole row of k per lane."""
    assert _mxu_closed_forms_ok(), 'air/mxu_layout.py no longer matches the closed forms in _igemm_wmma'
    A, B, C = '%a4', '%a6', '%a8'
    base = _lds_wave_base(b, MXU_WAVE_BYTES)

    def lane_bits(mask, shift):
        t = b.v(f'and i32 {lane}, {mask}')
        if shift > 0:
            return b.v(f'shl i32 {t}, {shift}')
        return b.v(f'lshr i32 {t}, {-shift}') if shift < 0 else t

    def at(off_expr, align):   # pointer into the wave's LDS slot
        return b.v(f'getelementptr inbounds i8, ptr addrspace(3) {base}, i32 {off_expr}')

    # tile coordinates of this lane in Apple's layout
    a0 = b.v(f'or i32 {lane_bits(1, 2)}, {lane_bits(2, 3)}')
    a0 = b.v(f'or i32 {a0}, {lane_bits(4, 3)}')
    a0 = b.v(f'or i32 {a0}, {lane_bits(8, 0)}')
    a0 = b.v(f'or i32 {a0}, {lane_bits(16, 2)}')          # row * 16 + k of the first element of A (and of C)
    c0 = b.v(f'or i32 {lane_bits(1, 2)}, {lane_bits(8, 0)}')
    kk = b.v(f'or i32 {lane_bits(2, -1)}, {lane_bits(4, -1)}')
    kk = b.v(f'or i32 {kk}, {lane_bits(16, -2)}')
    b0 = b.v(f'shl i32 {c0}, 4')
    b0 = b.v(f'add i32 {b0}, {kk}')                         # col * 16 + k of the first element of B^T

    # A: two dwords (4 consecutive k) at rows r and r + 8
    a2 = b.v(f'bitcast i64 {A} to <2 x i32>')
    for half in range(2):
        x = b.v(f'extractelement <2 x i32> {a2}, i32 {half}')
        p = at(b.v(f'add i32 {a0}, {128 * half}'), 4)
        b.emit(f'store i32 {x}, ptr addrspace(3) {p}, align 4')
    # B^T: 8 bytes, 4 consecutive columns at k and k + 8
    bv = b.v(f'bitcast i64 {B} to <8 x i8>')
    for sl in range(8):
        x = b.v(f'extractelement <8 x i8> {bv}, i32 {sl}')
        p = at(b.v(f'add i32 {b0}, {256 + (sl & 3) * 16 + (8 if sl >= 4 else 0)}'), 1)
        b.emit(f'store i8 {x}, ptr addrspace(3) {p}, align 1')
    b.emit('fence syncscope("wavefront") release')
    b.emit('fence syncscope("wavefront") acquire')

    # WMMA wants lane l (and l + 16) to hold row / column (l & 15) of A / B, all 16 k
    l16 = b.v(f'and i32 {lane}, 15')
    row_off = b.v(f'shl i32 {l16}, 4')
    xa = b.v(f'load <4 x i32>, ptr addrspace(3) {at(row_off, 16)}, align 16')
    xb = b.v(f'load <4 x i32>, ptr addrspace(3) {at(b.v(f"add i32 {row_off}, 256"), 16)}, align 16')
    sa = b.v('icmp eq i16 %a5, 75')
    sb = b.v('icmp eq i16 %a7, 75')
    b.decls.add(f'declare <8 x i32> @{WMMA_I8}(i1 immarg, <4 x i32>, i1 immarg, <4 x i32>, <8 x i32>, i1 immarg)')
    b.emit(f'br i1 {sa}, label %wm_a1, label %wm_a0')
    results = []
    for ia in (0, 1):
        b.emit(f'wm_a{ia}:')
        b.emit(f'br i1 {sb}, label %wm_{ia}1, label %wm_{ia}0')
    for ia in (0, 1):
        for ib in (0, 1):
            b.emit(f'wm_{ia}{ib}:')
            r = b.v(f'call <8 x i32> @{WMMA_I8}(i1 {"true" if ia else "false"}, <4 x i32> {xa}, i1 {"true" if ib else "false"}, '
                    f'<4 x i32> {xb}, <8 x i32> zeroinitializer, i1 false)')
            b.emit('br label %wm_end')
            results.append((r, f'wm_{ia}{ib}'))
    b.emit('wm_end:')
    d = b.v('phi <8 x i32> ' + ', '.join(f'[ {r}, %{lbl} ]' for r, lbl in results))

    # D: lane holds column (lane & 15) and rows 2 * i + (lane >> 4); write it row-major over the operand tiles
    hi = b.v(f'lshr i32 {lane}, 4')
    for i in range(8):
        x = b.v(f'extractelement <8 x i32> {d}, i32 {i}')
        row = b.v(f'add i32 {hi}, {2 * i}')
        o = b.v(f'shl i32 {row}, 4')
        o = b.v(f'add i32 {o}, {l16}')
        o = b.v(f'shl i32 {o}, 2')
        b.emit(f'store i32 {x}, ptr addrspace(3) {at(o, 4)}, align 4')
    b.emit('fence syncscope("wavefront") release')
    b.emit('fence syncscope("wavefront") acquire')
    # back to Apple's layout: 4 consecutive columns of rows r and r + 8
    o = b.v(f'shl i32 {a0}, 2')
    g0 = b.v(f'load <4 x i32>, ptr addrspace(3) {at(o, 16)}, align 16')
    g1 = b.v(f'load <4 x i32>, ptr addrspace(3) {at(b.v(f"add i32 {o}, 512"), 16)}, align 16')
    g = b.v(f'shufflevector <4 x i32> {g0}, <4 x i32> {g1}, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>')
    out = b.v(f'add <8 x i32> {C}, {g}')
    b.emit('fence syncscope("wavefront") release')   # the slot is reused by the next GEMM
    b.emit('fence syncscope("wavefront") acquire')
    return out


def _fgemm_dot2(b, lane):
    """D = C + A.B for 16x16x16 fp16 with f32 accumulators, as a chain of v_dot2_f32_f16 over k in increasing order (the same
    arithmetic as before this lowering used the closed-form layout, so results are bit-identical to it): a lane holds rows
    r and r + 8 and columns c0 .. c0 + 3 of the result, so it needs only 2 rows of A and 4 columns of B (each 32 bytes)."""
    assert _mxu_closed_forms_ok(), 'air/mxu_layout.py no longer matches the closed forms'
    A, B, C = '%a0', '%a2', '%a4'
    base = _lds_wave_base(b, MXU_WAVE_BYTES)

    def lane_bits(mask, shift):
        t = b.v(f'and i32 {lane}, {mask}')
        if shift > 0:
            return b.v(f'shl i32 {t}, {shift}')
        return b.v(f'lshr i32 {t}, {-shift}') if shift < 0 else t

    def at(off_expr):
        return b.v(f'getelementptr inbounds i8, ptr addrspace(3) {base}, i32 {off_expr}')

    a0 = b.v(f'or i32 {lane_bits(1, 2)}, {lane_bits(2, 3)}')
    a0 = b.v(f'or i32 {a0}, {lane_bits(4, 3)}')
    a0 = b.v(f'or i32 {a0}, {lane_bits(8, 0)}')
    a0 = b.v(f'or i32 {a0}, {lane_bits(16, 2)}')          # row * 16 + k0 (elements) of A's and C's first element
    c0 = b.v(f'or i32 {lane_bits(1, 2)}, {lane_bits(8, 0)}')
    kk = b.v(f'or i32 {lane_bits(2, -1)}, {lane_bits(4, -1)}')
    kk = b.v(f'or i32 {kk}, {lane_bits(16, -2)}')
    b0 = b.v(f'shl i32 {c0}, 4')
    b0 = b.v(f'add i32 {b0}, {kk}')                         # col * 16 + k (elements) of B^T's first element

    # A tile [row][k] halves at byte 0, B^T tile [col][k] at byte 512: 4 consecutive k of rows r and r + 8
    for half in range(2):
        x = b.v(f'shufflevector <8 x half> {A}, <8 x half> poison, <4 x i32> <i32 {4 * half}, i32 {4 * half + 1}, i32 {4 * half + 2}, i32 {4 * half + 3}>')
        o = b.v(f'add i32 {a0}, {128 * half}')
        o = b.v(f'shl i32 {o}, 1')
        b.emit(f'store <4 x half> {x}, ptr addrspace(3) {at(o)}, align 8')
    for sl in range(8):
        x = b.v(f'extractelement <8 x half> {B}, i32 {sl}')
        o = b.v(f'add i32 {b0}, {(sl & 3) * 16 + (8 if sl >= 4 else 0)}')
        o = b.v(f'shl i32 {o}, 1')
        o = b.v(f'add i32 {o}, 512')
        b.emit(f'store half {x}, ptr addrspace(3) {at(o)}, align 2')
    b.emit('fence syncscope("wavefront") release')
    b.emit('fence syncscope("wavefront") acquire')

    row0 = b.v(f'lshr i32 {a0}, 4')                          # first row of this lane's results
    col0 = b.v(f'and i32 {a0}, 15')                          # first column
    arow = []   # the 2 rows of A this lane needs, each as 8 x <2 x half> pairs in k order
    for h in range(2):
        r = b.v(f'add i32 {row0}, {8 * h}')
        o = b.v(f'shl i32 {r}, 5')
        lo = b.v(f'load <8 x half>, ptr addrspace(3) {at(o)}, align 16')
        hi = b.v(f'load <8 x half>, ptr addrspace(3) {at(b.v(f"add i32 {o}, 16"))}, align 16')
        arow.append((lo, hi))
    bcol = []   # the 4 columns of B
    for j in range(4):
        c = b.v(f'add i32 {col0}, {j}')
        o = b.v(f'shl i32 {c}, 5')
        o = b.v(f'add i32 {o}, 512')
        lo = b.v(f'load <8 x half>, ptr addrspace(3) {at(o)}, align 16')
        hi = b.v(f'load <8 x half>, ptr addrspace(3) {at(b.v(f"add i32 {o}, 16"))}, align 16')
        bcol.append((lo, hi))

    def pair(vec, q):   # elements 2q, 2q + 1 of a 16-element row held as (lo, hi) halves of 8
        src = vec[0] if q < 4 else vec[1]
        e = (q % 4) * 2
        return b.v(f'shufflevector <8 x half> {src}, <8 x half> poison, <2 x i32> <i32 {e}, i32 {e + 1}>')

    out = C
    for sl in range(8):
        acc = b.v(f'extractelement <8 x float> {C}, i32 {sl}')
        ar, bc = arow[sl // 4], bcol[sl & 3]
        for q in range(8):
            acc = intr(b, 'float', 'llvm.amdgcn.fdot2', [('<2 x half>', pair(ar, q)), ('<2 x half>', pair(bc, q)), ('float', acc), ('i1', 'false')])
        out = b.v(f'insertelement <8 x float> {out}, float {acc}, i32 {sl}')
    b.emit('fence syncscope("wavefront") release')   # the slot is reused by the next GEMM
    b.emit('fence syncscope("wavefront") acquire')
    return out


WMMA_F16 = 'llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v16f16'


# --- the gfx11 WMMA instructions on other generations ------------------------------------------------------------------
# Every GEMM above is written against RDNA3's v_wmma_f32_16x16x16_f16 / v_wmma_i32_16x16x16_iu8 and their register layout
# (operand lane l: row / column l & 15, all 16 k; accumulator VGPR i: row 2i + (l >> 4), column l & 15). On other
# generations m4r_air replaces just those two calls by @__m4r_wmma_f16 / @__m4r_wmma_iu8 with the same operands and results:
#   soft     RDNA2 (no WMMA): each accumulator element is the v_dot2 (fp16) / v_dot4 (int8) chain over k in increasing order
#            starting from C, the A row broadcast with DPP row_share (half 1 first rotated by one lane so that row_share n
#            reads row n there and row n + 1 in half 1). On gfx1101 the fp16 chain is bit-identical to WMMA (docs/mxu.md).
#   gfx12    RDNA4's WMMA (operand: 8 of the 16 k, half h holds k 8h .. 8h + 7; accumulator VGPR i: row i + 8h) with the
#            operands cut to the lane's half and the accumulators exchanged between the halves around it.
#   gfx12emu validation on RDNA3: the gfx12 path above, with the gfx12 instruction itself emulated by gfx11 WMMA in gfx12's
#            layout (as d4r's layout shim). Bit-identical to the native build only if every layout conversion is right.
WMMA_IMPLS = ('native', 'soft', 'gfx12', 'gfx12emu')


def _other_half(b, v):
    """the value of the lane with the same index in the other half of the wave (v_permlanex16, identity selects)"""
    return intr(b, 'i32', 'llvm.amdgcn.permlanex16.i32', [('i32', 'poison'), ('i32', v), ('i32', str(0x76543210)),
                                                           ('i32', str(0xfedcba98 - (1 << 32))), ('i1', 'false'), ('i1', 'false')])


def _words(b, v, n, t):
    """<n x t> (t: half or i32 words) -> list of n * 16 / 32 i32 words"""
    if t == 'half':
        w = b.v(f'bitcast <{n} x half> {v} to <{n // 2} x i32>')
        k = n // 2
    else:
        w, k = v, n
    return [b.v(f'extractelement <{k} x i32> {w}, i32 {j}') for j in range(k)]


def _vec(b, words, t, n):
    cur = 'poison'
    for j, x in enumerate(words):
        cur = b.v(f'insertelement <{len(words)} x i32> {cur}, i32 {x}, i32 {j}')
    return b.v(f'bitcast <{len(words)} x i32> {cur} to <{n} x {t}>') if t == 'half' else cur


def _acc_11_to_12(b, half, acc, t):
    """accumulators in gfx11 layout (VGPR i = row 2i + h) -> gfx12 layout (VGPR i = row i + 8h); acc: list of 8 values"""
    w = [b.v(f'bitcast {t} {x} to i32') if t != 'i32' else x for x in acc]
    x = [_other_half(b, v) for v in w]
    out = []
    for i in range(8):
        lo = w[i // 2] if i % 2 == 0 else x[(i - 1) // 2]          # h = 0: row i is gfx11 half i & 1, VGPR i >> 1
        hi = w[4 + (i - 1) // 2] if i % 2 == 1 else x[4 + i // 2]  # h = 1: row 8 + i
        v = b.v(f'select i1 {half}, i32 {hi}, i32 {lo}')
        out.append(b.v(f'bitcast i32 {v} to {t}') if t != 'i32' else v)
    return out


def _acc_12_to_11(b, half, acc, t):
    """gfx12 layout (VGPR i = row i + 8h) -> gfx11 layout (VGPR i = row 2i + h)"""
    w = [b.v(f'bitcast {t} {x} to i32') if t != 'i32' else x for x in acc]
    x = [_other_half(b, v) for v in w]
    out = []
    for i in range(8):
        lo = w[2 * i] if i < 4 else x[2 * i - 8]        # h = 0: row 2i is gfx12 half (2i) >> 3, VGPR 2i & 7
        hi = x[2 * i + 1] if i < 4 else w[2 * i - 7]    # h = 1: row 2i + 1
        v = b.v(f'select i1 {half}, i32 {hi}, i32 {lo}')
        out.append(b.v(f'bitcast i32 {v} to {t}') if t != 'i32' else v)
    return out


def _elems(b, v, t):
    return [b.v(f'extractelement <8 x {t}> {v}, i32 {i}') for i in range(8)]


def _pack8(b, xs, t):
    cur = 'poison'
    for i, x in enumerate(xs):
        cur = b.v(f'insertelement <8 x {t}> {cur}, {t} {x}, i32 {i}')
    return cur


WMMA12_F16 = 'llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v8f16'
WMMA12_I8 = 'llvm.amdgcn.wmma.i32.16x16x16.iu8.v8i32.v2i32'


def _wmma12(b, kind, a, bb, c, sa='false', sb='false', emu=False):
    """the gfx12 instruction on gfx12-layout operands (a, bb: 4 / 2 words of the lane's k half; c: 8 values); emu: built from
    the gfx11 instruction (operands and accumulators converted to gfx11's layout and back)"""
    t = 'float' if kind == 'f' else 'i32'
    if not emu:
        if kind == 'f':
            b.decls.add(f'declare <8 x float> @{WMMA12_F16}(<8 x half>, <8 x half>, <8 x float>)')
            d = b.v(f'call <8 x float> @{WMMA12_F16}(<8 x half> {_vec(b, a, "half", 8)}, <8 x half> {_vec(b, bb, "half", 8)}, '
                    f'<8 x float> {_pack8(b, c, t)})')
        else:
            b.decls.add(f'declare <8 x i32> @{WMMA12_I8}(i1 immarg, <2 x i32>, i1 immarg, <2 x i32>, <8 x i32>, i1 immarg)')
            d = b.v(f'call <8 x i32> @{WMMA12_I8}(i1 {sa}, <2 x i32> {_vec(b, a, "i32", 2)}, i1 {sb}, <2 x i32> {_vec(b, bb, "i32", 2)}, '
                    f'<8 x i32> {_pack8(b, c, t)}, i1 false)')
        return _elems(b, d, t)
    lane = lane_id(b)
    half = b.v(f'icmp uge i32 {lane}, 16')
    full = []
    for v in (a, bb):   # gfx11 operand: k 0..15 in order = (half 0's words, half 1's words)
        o = [_other_half(b, x) for x in v]
        full.append([b.v(f'select i1 {half}, i32 {o[j]}, i32 {v[j]}') for j in range(len(v))] +
                    [b.v(f'select i1 {half}, i32 {v[j]}, i32 {o[j]}') for j in range(len(v))])
    c11 = _acc_12_to_11(b, half, c, t)
    d11 = _wmma11(b, kind, full[0], full[1], c11, sa, sb)
    return _acc_11_to_12(b, half, d11, t)


def _wmma11(b, kind, a, bb, c, sa='false', sb='false'):
    t = 'float' if kind == 'f' else 'i32'
    if kind == 'f':
        b.decls.add(f'declare <8 x float> @{WMMA_F16}(<16 x half>, <16 x half>, <8 x float>)')
        d = b.v(f'call <8 x float> @{WMMA_F16}(<16 x half> {_vec(b, a, "half", 16)}, <16 x half> {_vec(b, bb, "half", 16)}, '
                f'<8 x float> {_pack8(b, c, t)})')
    else:
        b.decls.add(f'declare <8 x i32> @{WMMA_I8}(i1 immarg, <4 x i32>, i1 immarg, <4 x i32>, <8 x i32>, i1 immarg)')
        d = b.v(f'call <8 x i32> @{WMMA_I8}(i1 {sa}, <4 x i32> {_vec(b, a, "i32", 4)}, i1 {sb}, <4 x i32> {_vec(b, bb, "i32", 4)}, '
                f'<8 x i32> {_pack8(b, c, t)}, i1 false)')
    return _elems(b, d, t)


def _wmma_soft(b, kind, a, bb, c, signed=None):
    """gfx11 semantics from dot products: acc(row 2i + h, col r) = C, then the dot over the k words in order.
    One inline-asm statement per output row: LLVM would otherwise merge the broadcasts of products that share their A
    operand and keep 64 of them alive across those products (spills; RDNA2 cannot run kernels with scratch in vkhip). fp16:
    v_dot2c_f32_f16 (gfx11: v_dot2acc_f32_f16) takes the DPP broadcast itself; int8: a DPP move into a private temporary,
    then the VOP3P dot. Each statement starts with s_nop 1: LLVM does not see the DPP read inside it, and a DPP source
    written by the previous VALU instruction needs two wait states."""
    fam = CTX.get('family', 'gfx11')
    if fam not in ('gfx10', 'gfx11'):
        raise ValueError(f'software WMMA is written for gfx10/gfx11, not {fam}')
    lane = lane_id(b)
    half = b.v(f'icmp uge i32 {lane}, 16')
    # half 1 rotated by one lane (row_ror:15): row_share n then reads lane n + 1 there, which holds row n + 1
    ar = []
    for w in a:
        rot = intr(b, 'i32', 'llvm.amdgcn.update.dpp.i32', [('i32', 'poison'), ('i32', w), ('i32', str(0x12f)), ('i32', '15'),
                                                             ('i32', '15'), ('i1', 'false')])
        ar.append(b.v(f'select i1 {half}, i32 {rot}, i32 {w}'))
    n = len(ar)
    dpp = 'row_mask:0xf bank_mask:0xf'
    out = []
    for i in range(8):
        sh = f'row_share:{2 * i}'
        if kind == 'f':
            op = 'v_dot2c_f32_f16_dpp' if fam == 'gfx10' else 'v_dot2acc_f32_f16_dpp'
            body = '\\0As_nop 1' + ''.join(f'\\0A{op} $0, ${1 + j}, ${1 + n + j} {sh} {dpp}' for j in range(n))
            cons = '=v,' + ','.join(['v'] * (2 * n)) + ',0'
            acc = b.v(f'call float asm "{body[3:]}", "{cons}"(' + ', '.join([f'i32 {x}' for x in ar] + [f'i32 {x}' for x in bb])
                      + f', float {c[i]})')
        else:
            dot = ('v_dot4_i32_i8 $0, $1, ${0}, $0' if fam == 'gfx10' else 'v_dot4_i32_iu8 $0, $1, ${0}, $0 neg_lo:[1,1,0]') \
                if signed else 'v_dot4_u32_u8 $0, $1, ${0}, $0'
            body = '\\0As_nop 1' + ''.join(f'\\0Av_mov_b32_dpp $1, ${2 + j} {sh} {dpp}\\0A' + dot.replace('${0}', f'${2 + n + j}')
                                            for j in range(n))
            cons = '=v,=&v,' + ','.join(['v'] * (2 * n)) + ',0'
            r = b.v(f'call {{ i32, i32 }} asm "{body[3:]}", "{cons}"(' + ', '.join([f'i32 {x}' for x in ar] + [f'i32 {x}' for x in bb])
                    + f', i32 {c[i]})')
            acc = b.v(f'extractvalue {{ i32, i32 }} {r}, 0')
        out.append(acc)
    return out


def wmma_impl_defs(impl):
    """IR definitions of @__m4r_wmma_f16 and @__m4r_wmma_iu8 (the gfx11 instructions' signatures) for `impl`"""
    defs, decls = [], set()
    for kind in ('f', 'i'):
        b = Body()
        t = 'float' if kind == 'f' else 'i32'
        if kind == 'f':
            hdr = 'define internal <8 x float> @__m4r_wmma_f16(<16 x half> %a, <16 x half> %b, <8 x float> %c) #900 {'
            a, bb = _words(b, '%a', 16, 'half'), _words(b, '%b', 16, 'half')
        else:
            hdr = 'define internal <8 x i32> @__m4r_wmma_iu8(i1 %sa, <4 x i32> %a, i1 %sb, <4 x i32> %b, <8 x i32> %c, i1 %clamp) #900 {'
            a, bb = _words(b, '%a', 4, 'i32'), _words(b, '%b', 4, 'i32')
        c = _elems(b, '%c', t)
        if impl == 'soft':
            if kind == 'f':
                d = _wmma_soft(b, kind, a, bb, c)
            else:   # MetalFX multiplies u8 by u8 or s8 by s8 (mixed pairs are never selected at run time)
                ds = _wmma_soft(b, kind, a, bb, c, signed=True)
                du = _wmma_soft(b, kind, a, bb, c, signed=False)
                d = [b.v(f'select i1 %sa, i32 {x}, i32 {y}') for x, y in zip(ds, du)]
        else:
            lane = lane_id(b)
            half = b.v(f'icmp uge i32 {lane}, 16')
            n = len(a) // 2
            a12 = [b.v(f'select i1 {half}, i32 {a[n + j]}, i32 {a[j]}') for j in range(n)]
            b12 = [b.v(f'select i1 {half}, i32 {bb[n + j]}, i32 {bb[j]}') for j in range(n)]
            c12 = _acc_11_to_12(b, half, c, t)
            emu = impl == 'gfx12emu'
            if kind == 'f':
                d12 = _wmma12(b, kind, a12, b12, c12, emu=emu)
            else:   # the sign flags are immediates of the instruction: one call per pair, picked by the flags
                outs = {}
                for sa in ('false', 'true'):
                    for sb in ('false', 'true'):
                        outs[(sa, sb)] = _wmma12(b, kind, a12, b12, c12, sa, sb, emu=emu)
                d12 = []
                for i in range(8):
                    x = b.v(f'select i1 %sb, i32 {outs[("true", "true")][i]}, i32 {outs[("true", "false")][i]}')
                    y = b.v(f'select i1 %sb, i32 {outs[("false", "true")][i]}, i32 {outs[("false", "false")][i]}')
                    d12.append(b.v(f'select i1 %sa, i32 {x}, i32 {y}'))
            d = _acc_12_to_11(b, half, d12, t)
        r = _pack8(b, d, t)
        defs.append('\n'.join([hdr] + b.lines + [f'  ret <8 x {t}> {r}', '}']))
        decls |= b.decls
    return '\n\n'.join(defs), decls


def _fgemm_wmma(b, lane):
    """D = C + A.B for 16x16x16 fp16 with f32 accumulators on RDNA3 WMMA (v_wmma_f32_16x16x16_f16).

    On gfx1101 this instruction gives bit-identical results to the chain of v_dot2_f32_f16 over k in increasing order that
    starts from C (_fgemm_dot2), also with a nonzero C (checked on 768k results, docs/mxu.md). C therefore goes into the WMMA
    as its accumulator; adding it afterwards would round differently. The wave's LDS slot is used three times: A and B^T in
    Apple's layout -> rows of k per lane; C in Apple's layout -> WMMA's (lane: column l & 15, rows 2i + (l >> 4)); D back."""
    assert _mxu_closed_forms_ok(), 'air/mxu_layout.py no longer matches the closed forms'
    A, B, C = '%a0', '%a2', '%a4'
    base = _lds_wave_base(b, MXU_WAVE_BYTES)

    def lane_bits(mask, shift):
        t = b.v(f'and i32 {lane}, {mask}')
        if shift > 0:
            return b.v(f'shl i32 {t}, {shift}')
        return b.v(f'lshr i32 {t}, {-shift}') if shift < 0 else t

    def at(off_expr):
        return b.v(f'getelementptr inbounds i8, ptr addrspace(3) {base}, i32 {off_expr}')

    def sync():
        b.emit('fence syncscope("wavefront") release')
        b.emit('fence syncscope("wavefront") acquire')

    a0 = b.v(f'or i32 {lane_bits(1, 2)}, {lane_bits(2, 3)}')
    a0 = b.v(f'or i32 {a0}, {lane_bits(4, 3)}')
    a0 = b.v(f'or i32 {a0}, {lane_bits(8, 0)}')
    a0 = b.v(f'or i32 {a0}, {lane_bits(16, 2)}')          # row * 16 + k0 (elements) of A's and C's first element
    c0 = b.v(f'or i32 {lane_bits(1, 2)}, {lane_bits(8, 0)}')
    kk = b.v(f'or i32 {lane_bits(2, -1)}, {lane_bits(4, -1)}')
    kk = b.v(f'or i32 {kk}, {lane_bits(16, -2)}')
    b0 = b.v(f'shl i32 {c0}, 4')
    b0 = b.v(f'add i32 {b0}, {kk}')                         # col * 16 + k (elements) of B^T's first element

    # 1. A tile [row][k] halves at byte 0, B^T tile [col][k] at byte 512 (Apple's layout: 4 consecutive k of rows r, r + 8)
    for half in range(2):
        x = b.v(f'shufflevector <8 x half> {A}, <8 x half> poison, <4 x i32> <i32 {4 * half}, i32 {4 * half + 1}, i32 {4 * half + 2}, i32 {4 * half + 3}>')
        o = b.v(f'add i32 {a0}, {128 * half}')
        o = b.v(f'shl i32 {o}, 1')
        b.emit(f'store <4 x half> {x}, ptr addrspace(3) {at(o)}, align 8')
    for sl in range(8):
        x = b.v(f'extractelement <8 x half> {B}, i32 {sl}')
        o = b.v(f'add i32 {b0}, {(sl & 3) * 16 + (8 if sl >= 4 else 0)}')
        o = b.v(f'shl i32 {o}, 1')
        o = b.v(f'add i32 {o}, 512')
        b.emit(f'store half {x}, ptr addrspace(3) {at(o)}, align 2')
    sync()
    # WMMA operands: lane l (and l + 16) holds row / column (l & 15) of A / B, all 16 k
    l16 = b.v(f'and i32 {lane}, 15')
    ro = b.v(f'shl i32 {l16}, 5')
    xa = b.v(f'load <16 x half>, ptr addrspace(3) {at(ro)}, align 32')
    xb = b.v(f'load <16 x half>, ptr addrspace(3) {at(b.v(f"add i32 {ro}, 512"))}, align 32')
    sync()

    # 2. C (f32 [row][col], 1024 bytes, over the operand tiles): this lane's elements are rows r, r + 8, columns c0 .. c0 + 3
    row0 = b.v(f'lshr i32 {a0}, 4')
    col0 = b.v(f'and i32 {a0}, 15')
    for h in range(2):
        x = b.v(f'shufflevector <8 x float> {C}, <8 x float> poison, <4 x i32> <i32 {4 * h}, i32 {4 * h + 1}, i32 {4 * h + 2}, i32 {4 * h + 3}>')
        o = b.v(f'add i32 {row0}, {8 * h}')
        o = b.v(f'shl i32 {o}, 4')
        o = b.v(f'add i32 {o}, {col0}')
        o = b.v(f'shl i32 {o}, 2')
        b.emit(f'store <4 x float> {x}, ptr addrspace(3) {at(o)}, align 16')
    sync()
    hi = b.v(f'lshr i32 {lane}, 4')
    offs = []
    cw = 'poison'
    for i in range(8):
        o = b.v(f'add i32 {hi}, {2 * i}')
        o = b.v(f'shl i32 {o}, 4')
        o = b.v(f'add i32 {o}, {l16}')
        o = b.v(f'shl i32 {o}, 2')
        offs.append(o)
        x = b.v(f'load float, ptr addrspace(3) {at(o)}, align 4')
        cw = b.v(f'insertelement <8 x float> {cw}, float {x}, i32 {i}')
    b.decls.add(f'declare <8 x float> @{WMMA_F16}(<16 x half>, <16 x half>, <8 x float>)')
    d = b.v(f'call <8 x float> @{WMMA_F16}(<16 x half> {xa}, <16 x half> {xb}, <8 x float> {cw})')
    sync()

    # 3. D back to Apple's layout
    for i in range(8):
        x = b.v(f'extractelement <8 x float> {d}, i32 {i}')
        b.emit(f'store float {x}, ptr addrspace(3) {at(offs[i])}, align 4')
    sync()
    out = 'poison'
    for h in range(2):
        o = b.v(f'add i32 {row0}, {8 * h}')
        o = b.v(f'shl i32 {o}, 4')
        o = b.v(f'add i32 {o}, {col0}')
        o = b.v(f'shl i32 {o}, 2')
        x = b.v(f'load <4 x float>, ptr addrspace(3) {at(o)}, align 16')
        for j in range(4):
            e = b.v(f'extractelement <4 x float> {x}, i32 {j}')
            out = b.v(f'insertelement <8 x float> {out}, float {e}, i32 {4 * h + j}')
    sync()   # the slot is reused by the next GEMM
    return out


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
        if kind == 'i' and MXU_MODE['int'] == 'wmma':
            return _igemm_wmma(b, lane)
        if kind == 'f' and MXU_MODE['fp16'] == 'wmma':
            return _fgemm_wmma(b, lane)
        if kind == 'f' and MXU_MODE['fp16'] == 'closed':
            return _fgemm_dot2(b, lane)
        # table-driven: scatter A and B into LDS in logical [row][k] / [col][k] order, then dot products
        base = _lds_wave_base(b, MXU_WAVE_BYTES)
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


# --- GEMM chains in WMMA layout (air/gemm_chain.py) -------------------------------------------------------------------
# A GEMM whose accumulator comes from another GEMM keeps it in WMMA's layout (lane: column l & 15, rows 2i + (l >> 4)); only
# accumulators read by other code are converted to Apple's layout. Same WMMA instructions with C as the accumulator, so the
# results equal _fgemm_wmma / _igemm_wmma.

def _apple_coords(b, lane):
    def lane_bits(mask, shift):
        t = b.v(f'and i32 {lane}, {mask}')
        if shift > 0:
            return b.v(f'shl i32 {t}, {shift}')
        return b.v(f'lshr i32 {t}, {-shift}') if shift < 0 else t
    a0 = b.v(f'or i32 {lane_bits(1, 2)}, {lane_bits(2, 3)}')
    a0 = b.v(f'or i32 {a0}, {lane_bits(4, 3)}')
    a0 = b.v(f'or i32 {a0}, {lane_bits(8, 0)}')
    a0 = b.v(f'or i32 {a0}, {lane_bits(16, 2)}')          # row * 16 + k0 of A's and C's first element
    c0 = b.v(f'or i32 {lane_bits(1, 2)}, {lane_bits(8, 0)}')
    kk = b.v(f'or i32 {lane_bits(2, -1)}, {lane_bits(4, -1)}')
    kk = b.v(f'or i32 {kk}, {lane_bits(16, -2)}')
    b0 = b.v(f'shl i32 {c0}, 4')
    b0 = b.v(f'add i32 {b0}, {kk}')                         # col * 16 + k of B^T's first element
    return a0, b0


def _wsync(b):
    b.emit('fence syncscope("wavefront") release')
    b.emit('fence syncscope("wavefront") acquire')


def _at(b, base, off):
    return b.v(f'getelementptr inbounds i8, ptr addrspace(3) {base}, i32 {off}')


def _acc_offsets(b, lane):
    """byte offsets of this lane's 8 accumulator elements in a row-major 16x16 x 4-byte tile: (Apple: 2 x 16-byte rows),
    (WMMA: 8 scalars)"""
    a0, _ = _apple_coords(b, lane)
    row0 = b.v(f'lshr i32 {a0}, 4')
    col0 = b.v(f'and i32 {a0}, 15')
    apple = []
    for h in range(2):
        o = b.v(f'add i32 {row0}, {8 * h}')
        o = b.v(f'shl i32 {o}, 4')
        o = b.v(f'add i32 {o}, {col0}')
        apple.append(b.v(f'shl i32 {o}, 2'))
    l16 = b.v(f'and i32 {lane}, 15')
    hi = b.v(f'lshr i32 {lane}, 4')
    wm = []
    for i in range(8):
        o = b.v(f'add i32 {hi}, {2 * i}')
        o = b.v(f'shl i32 {o}, 4')
        o = b.v(f'add i32 {o}, {l16}')
        wm.append(b.v(f'shl i32 {o}, 2'))
    return apple, wm


def acc_convert(to_wmma):
    def f(b, s):
        t = s.ret
        e = vec(t)[1]
        lane = lane_id(b)
        base = _lds_wave_base(b, MXU_WAVE_BYTES)
        apple, wm = _acc_offsets(b, lane)
        if to_wmma:
            for h in range(2):
                x = b.v(f'shufflevector {t} %a0, {t} poison, <4 x i32> <i32 {4 * h}, i32 {4 * h + 1}, i32 {4 * h + 2}, i32 {4 * h + 3}>')
                b.emit(f'store <4 x {e}> {x}, ptr addrspace(3) {_at(b, base, apple[h])}, align 16')
            _wsync(b)
            out = 'poison'
            for i in range(8):
                x = b.v(f'load {e}, ptr addrspace(3) {_at(b, base, wm[i])}, align 4')
                out = b.v(f'insertelement {t} {out}, {e} {x}, i32 {i}')
        else:
            for i in range(8):
                x = b.v(f'extractelement {t} %a0, i32 {i}')
                b.emit(f'store {e} {x}, ptr addrspace(3) {_at(b, base, wm[i])}, align 4')
            _wsync(b)
            out = 'poison'
            for h in range(2):
                x = b.v(f'load <4 x {e}>, ptr addrspace(3) {_at(b, base, apple[h])}, align 16')
                for j in range(4):
                    y = b.v(f'extractelement <4 x {e}> {x}, i32 {j}')
                    out = b.v(f'insertelement {t} {out}, {e} {y}, i32 {4 * h + j}')
        _wsync(b)   # the slot is reused next
        return out
    return f


def wgemm_f(b, s):
    """m4r.agx3.wgemm.f(<8 x half> A, <8 x half> B, <8 x float> C in WMMA layout) -> D in WMMA layout"""
    A, B, C = '%a0', '%a1', '%a2'
    lane = lane_id(b)
    base = _lds_wave_base(b, MXU_WAVE_BYTES)
    a0, b0 = _apple_coords(b, lane)
    for half in range(2):
        x = b.v(f'shufflevector <8 x half> {A}, <8 x half> poison, <4 x i32> <i32 {4 * half}, i32 {4 * half + 1}, i32 {4 * half + 2}, i32 {4 * half + 3}>')
        o = b.v(f'add i32 {a0}, {128 * half}')
        o = b.v(f'shl i32 {o}, 1')
        b.emit(f'store <4 x half> {x}, ptr addrspace(3) {_at(b, base, o)}, align 8')
    for sl in range(8):
        x = b.v(f'extractelement <8 x half> {B}, i32 {sl}')
        o = b.v(f'add i32 {b0}, {(sl & 3) * 16 + (8 if sl >= 4 else 0)}')
        o = b.v(f'shl i32 {o}, 1')
        o = b.v(f'add i32 {o}, 512')
        b.emit(f'store half {x}, ptr addrspace(3) {_at(b, base, o)}, align 2')
    _wsync(b)
    l16 = b.v(f'and i32 {lane}, 15')
    ro = b.v(f'shl i32 {l16}, 5')
    xa = b.v(f'load <16 x half>, ptr addrspace(3) {_at(b, base, ro)}, align 32')
    xb = b.v(f'load <16 x half>, ptr addrspace(3) {_at(b, base, b.v(f"add i32 {ro}, 512"))}, align 32')
    _wsync(b)
    b.decls.add(f'declare <8 x float> @{WMMA_F16}(<16 x half>, <16 x half>, <8 x float>)')
    return b.v(f'call <8 x float> @{WMMA_F16}(<16 x half> {xa}, <16 x half> {xb}, <8 x float> {C})')


def wgemm_i(b, s):
    """m4r.agx3.wgemm.i(i64 A, i16 typeA, i64 B, i16 typeB, <8 x i32> C in WMMA layout) -> D in WMMA layout (type 75 = s8)"""
    A, B, C = '%a0', '%a2', '%a4'
    lane = lane_id(b)
    base = _lds_wave_base(b, MXU_WAVE_BYTES)
    a0, b0 = _apple_coords(b, lane)
    a2 = b.v(f'bitcast i64 {A} to <2 x i32>')
    for half in range(2):
        x = b.v(f'extractelement <2 x i32> {a2}, i32 {half}')
        b.emit(f'store i32 {x}, ptr addrspace(3) {_at(b, base, b.v(f"add i32 {a0}, {128 * half}"))}, align 4')
    bv = b.v(f'bitcast i64 {B} to <8 x i8>')
    for sl in range(8):
        x = b.v(f'extractelement <8 x i8> {bv}, i32 {sl}')
        b.emit(f'store i8 {x}, ptr addrspace(3) {_at(b, base, b.v(f"add i32 {b0}, {256 + (sl & 3) * 16 + (8 if sl >= 4 else 0)}"))}, align 1')
    _wsync(b)
    l16 = b.v(f'and i32 {lane}, 15')
    row_off = b.v(f'shl i32 {l16}, 4')
    xa = b.v(f'load <4 x i32>, ptr addrspace(3) {_at(b, base, row_off)}, align 16')
    xb = b.v(f'load <4 x i32>, ptr addrspace(3) {_at(b, base, b.v(f"add i32 {row_off}, 256"))}, align 16')
    _wsync(b)
    sa = b.v('icmp eq i16 %a1, 75')
    sb = b.v('icmp eq i16 %a3, 75')
    b.decls.add(f'declare <8 x i32> @{WMMA_I8}(i1 immarg, <4 x i32>, i1 immarg, <4 x i32>, <8 x i32>, i1 immarg)')
    b.emit(f'br i1 {sa}, label %wm_a1, label %wm_a0')
    results = []
    for ia in (0, 1):
        b.emit(f'wm_a{ia}:')
        b.emit(f'br i1 {sb}, label %wm_{ia}1, label %wm_{ia}0')
    for ia in (0, 1):
        for ib in (0, 1):
            b.emit(f'wm_{ia}{ib}:')
            r = b.v(f'call <8 x i32> @{WMMA_I8}(i1 {"true" if ia else "false"}, <4 x i32> {xa}, i1 {"true" if ib else "false"}, '
                    f'<4 x i32> {xb}, <8 x i32> {C}, i1 false)')
            b.emit('br label %wm_end')
            results.append((r, f'wm_{ia}{ib}'))
    b.emit('wm_end:')
    return b.v('phi <8 x i32> ' + ', '.join(f'[ {r}, %{lbl} ]' for r, lbl in results))


def wgemm_ib(b, s):
    """m4r.agx3.wgemm.ib(i64 A, i16 typeA, ptr addrspace(1) P, i16 typeB, <8 x i32> C in WMMA layout, i32 H): B is the weight tile
    Apple's kernel reads as 4 bytes at P (rows k 0-7 of a [k][col] half-tile, lane offset c0 + 16 kk) and 4 bytes at P + H
    (rows 8-15). The plan stores each 128-byte half transposed ([col][8 k]), so the lane's WMMA column (l & 15) is 8 bytes at
    P - (c0 + 16 kk) + 8 (l & 15) and 8 bytes H further."""
    A, P, C, H = '%a0', '%a2', '%a4', '%a5'
    lane = lane_id(b)
    base = _lds_wave_base(b, MXU_WAVE_BYTES)
    a0, _ = _apple_coords(b, lane)
    a2 = b.v(f'bitcast i64 {A} to <2 x i32>')
    for half in range(2):
        x = b.v(f'extractelement <2 x i32> {a2}, i32 {half}')
        b.emit(f'store i32 {x}, ptr addrspace(3) {_at(b, base, b.v(f"add i32 {a0}, {128 * half}"))}, align 4')
    # Apple's lane offset in the half-tile: c0 = 4 lane[0] + 8 lane[3], kk = lane[1] | lane[2] << 1 | lane[4] << 2
    def bit(i, sh):
        t = b.v(f'lshr i32 {lane}, {i}')
        t = b.v(f'and i32 {t}, 1')
        return b.v(f'shl i32 {t}, {sh}')
    c0 = b.v(f'or i32 {bit(0, 2)}, {bit(3, 3)}')
    kk = b.v(f'or i32 {bit(1, 0)}, {bit(2, 1)}')
    kk = b.v(f'or i32 {kk}, {bit(4, 2)}')
    k16 = b.v(f'shl i32 {kk}, 4')
    off = b.v(f'add i32 {c0}, {k16}')
    l16 = b.v(f'and i32 {lane}, 15')
    col = b.v(f'shl i32 {l16}, 3')
    delta = b.v(f'sub i32 {col}, {off}')
    d64 = b.v(f'sext i32 {delta} to i64')
    p0 = b.v(f'getelementptr i8, ptr addrspace(1) {P}, i64 {d64}')
    h64 = b.v(f'sext i32 {H} to i64')
    p1 = b.v(f'getelementptr i8, ptr addrspace(1) {p0}, i64 {h64}')
    lo = b.v(f'load <2 x i32>, ptr addrspace(1) {p0}, align 8')
    hi = b.v(f'load <2 x i32>, ptr addrspace(1) {p1}, align 8')
    xb = b.v(f'shufflevector <2 x i32> {lo}, <2 x i32> {hi}, <4 x i32> <i32 0, i32 1, i32 2, i32 3>')
    _wsync(b)
    row_off = b.v(f'shl i32 {l16}, 4')
    xa = b.v(f'load <4 x i32>, ptr addrspace(3) {_at(b, base, row_off)}, align 16')
    _wsync(b)
    sa = b.v('icmp eq i16 %a1, 75')
    sb = b.v('icmp eq i16 %a3, 75')
    b.decls.add(f'declare <8 x i32> @{WMMA_I8}(i1 immarg, <4 x i32>, i1 immarg, <4 x i32>, <8 x i32>, i1 immarg)')
    b.emit(f'br i1 {sa}, label %wm_a1, label %wm_a0')
    results = []
    for ia in (0, 1):
        b.emit(f'wm_a{ia}:')
        b.emit(f'br i1 {sb}, label %wm_{ia}1, label %wm_{ia}0')
    for ia in (0, 1):
        for ib in (0, 1):
            b.emit(f'wm_{ia}{ib}:')
            r = b.v(f'call <8 x i32> @{WMMA_I8}(i1 {"true" if ia else "false"}, <4 x i32> {xa}, i1 {"true" if ib else "false"}, '
                    f'<4 x i32> {xb}, <8 x i32> {C}, i1 false)')
            b.emit('br label %wm_end')
            results.append((r, f'wm_{ia}{ib}'))
    b.emit('wm_end:')
    return b.v('phi <8 x i32> ' + ', '.join(f'[ {r}, %{lbl} ]' for r, lbl in results))


# --- A operands staged once for several GEMMs (air/gemm_chain.py shares them) ------------------------------------------

def wa_f(b, s):
    """m4r.agx3.wa.f(<8 x half> A in Apple's layout) -> <16 x half>: row (l & 15) of A, the WMMA A operand"""
    lane = lane_id(b)
    base = _lds_wave_base(b, MXU_WAVE_BYTES)
    a0, _ = _apple_coords(b, lane)
    for half in range(2):
        x = b.v(f'shufflevector <8 x half> %a0, <8 x half> poison, <4 x i32> <i32 {4 * half}, i32 {4 * half + 1}, i32 {4 * half + 2}, i32 {4 * half + 3}>')
        o = b.v(f'add i32 {a0}, {128 * half}')
        o = b.v(f'shl i32 {o}, 1')
        b.emit(f'store <4 x half> {x}, ptr addrspace(3) {_at(b, base, o)}, align 8')
    _wsync(b)
    l16 = b.v(f'and i32 {lane}, 15')
    xa = b.v(f'load <16 x half>, ptr addrspace(3) {_at(b, base, b.v(f"shl i32 {l16}, 5"))}, align 32')
    _wsync(b)
    return xa


def wa_i(b, s):
    """m4r.agx3.wa.i(i64 A in Apple's layout) -> <4 x i32>: row (l & 15) of A (16 bytes), the WMMA A operand"""
    lane = lane_id(b)
    base = _lds_wave_base(b, MXU_WAVE_BYTES)
    a0, _ = _apple_coords(b, lane)
    a2 = b.v('bitcast i64 %a0 to <2 x i32>')
    for half in range(2):
        x = b.v(f'extractelement <2 x i32> {a2}, i32 {half}')
        b.emit(f'store i32 {x}, ptr addrspace(3) {_at(b, base, b.v(f"add i32 {a0}, {128 * half}"))}, align 4')
    _wsync(b)
    l16 = b.v(f'and i32 {lane}, 15')
    xa = b.v(f'load <4 x i32>, ptr addrspace(3) {_at(b, base, b.v(f"shl i32 {l16}, 4"))}, align 16')
    _wsync(b)
    return xa


def wgemm_fa(b, s):
    """m4r.agx3.wgemm.fa(<16 x half> staged A, <8 x half> B, <8 x float> C in WMMA layout) -> D in WMMA layout"""
    B, C = '%a1', '%a2'
    lane = lane_id(b)
    base = _lds_wave_base(b, MXU_WAVE_BYTES)
    _, b0 = _apple_coords(b, lane)
    for sl in range(8):
        x = b.v(f'extractelement <8 x half> {B}, i32 {sl}')
        o = b.v(f'add i32 {b0}, {(sl & 3) * 16 + (8 if sl >= 4 else 0)}')
        o = b.v(f'shl i32 {o}, 1')
        o = b.v(f'add i32 {o}, 512')
        b.emit(f'store half {x}, ptr addrspace(3) {_at(b, base, o)}, align 2')
    _wsync(b)
    l16 = b.v(f'and i32 {lane}, 15')
    xb = b.v(f'load <16 x half>, ptr addrspace(3) {_at(b, base, b.v(f"add i32 {b.v(f"shl i32 {l16}, 5")}, 512"))}, align 32')
    _wsync(b)
    b.decls.add(f'declare <8 x float> @{WMMA_F16}(<16 x half>, <16 x half>, <8 x float>)')
    return b.v(f'call <8 x float> @{WMMA_F16}(<16 x half> %a0, <16 x half> {xb}, <8 x float> {C})')


def _wmma_i8(b, xa, ta, xb, tb, C):
    sa = b.v(f'icmp eq i16 {ta}, 75')
    sb = b.v(f'icmp eq i16 {tb}, 75')
    b.decls.add(f'declare <8 x i32> @{WMMA_I8}(i1 immarg, <4 x i32>, i1 immarg, <4 x i32>, <8 x i32>, i1 immarg)')
    b.emit(f'br i1 {sa}, label %wm_a1, label %wm_a0')
    results = []
    for ia in (0, 1):
        b.emit(f'wm_a{ia}:')
        b.emit(f'br i1 {sb}, label %wm_{ia}1, label %wm_{ia}0')
    for ia in (0, 1):
        for ib in (0, 1):
            b.emit(f'wm_{ia}{ib}:')
            r = b.v(f'call <8 x i32> @{WMMA_I8}(i1 {"true" if ia else "false"}, <4 x i32> {xa}, i1 {"true" if ib else "false"}, '
                    f'<4 x i32> {xb}, <8 x i32> {C}, i1 false)')
            b.emit('br label %wm_end')
            results.append((r, f'wm_{ia}{ib}'))
    b.emit('wm_end:')
    return b.v('phi <8 x i32> ' + ', '.join(f'[ {r}, %{lbl} ]' for r, lbl in results))


def wgemm_ia(b, s):
    """m4r.agx3.wgemm.ia(<4 x i32> staged A, i16 typeA, i64 B, i16 typeB, <8 x i32> C in WMMA layout) -> D in WMMA layout"""
    lane = lane_id(b)
    base = _lds_wave_base(b, MXU_WAVE_BYTES)
    _, b0 = _apple_coords(b, lane)
    bv = b.v('bitcast i64 %a2 to <8 x i8>')
    for sl in range(8):
        x = b.v(f'extractelement <8 x i8> {bv}, i32 {sl}')
        b.emit(f'store i8 {x}, ptr addrspace(3) {_at(b, base, b.v(f"add i32 {b0}, {256 + (sl & 3) * 16 + (8 if sl >= 4 else 0)}"))}, align 1')
    _wsync(b)
    l16 = b.v(f'and i32 {lane}, 15')
    xb = b.v(f'load <4 x i32>, ptr addrspace(3) {_at(b, base, b.v(f"add i32 {b.v(f"shl i32 {l16}, 4")}, 256"))}, align 16')
    _wsync(b)
    return _wmma_i8(b, '%a0', '%a1', xb, '%a3', '%a4')


def wgemm_iab(b, s):
    """m4r.agx3.wgemm.iab(<4 x i32> staged A, i16 typeA, ptr addrspace(1) P, i16 typeB, <8 x i32> C, i32 H): B read straight
    from the transposed weight tile (see wgemm_ib)"""
    lane = lane_id(b)

    def bit(i, sh):
        t = b.v(f'lshr i32 {lane}, {i}')
        t = b.v(f'and i32 {t}, 1')
        return b.v(f'shl i32 {t}, {sh}')
    c0 = b.v(f'or i32 {bit(0, 2)}, {bit(3, 3)}')
    kk = b.v(f'or i32 {bit(1, 0)}, {bit(2, 1)}')
    kk = b.v(f'or i32 {kk}, {bit(4, 2)}')
    off = b.v(f'add i32 {c0}, {b.v(f"shl i32 {kk}, 4")}')
    l16 = b.v(f'and i32 {lane}, 15')
    delta = b.v(f'sub i32 {b.v(f"shl i32 {l16}, 3")}, {off}')
    p0 = b.v(f'getelementptr i8, ptr addrspace(1) %a2, i64 {b.v(f"sext i32 {delta} to i64")}')
    p1 = b.v(f'getelementptr i8, ptr addrspace(1) {p0}, i64 {b.v("sext i32 %a5 to i64")}')
    lo = b.v(f'load <2 x i32>, ptr addrspace(1) {p0}, align 8')
    hi = b.v(f'load <2 x i32>, ptr addrspace(1) {p1}, align 8')
    xb = b.v(f'shufflevector <2 x i32> {lo}, <2 x i32> {hi}, <4 x i32> <i32 0, i32 1, i32 2, i32 3>')
    return _wmma_i8(b, '%a0', '%a1', xb, '%a3', '%a4')


AGX_TABLE = [
    (r'm4r\.agx3\.wa\.f', wa_f),
    (r'm4r\.agx3\.wa\.i', wa_i),
    (r'm4r\.agx3\.wgemm\.fa', wgemm_fa),
    (r'm4r\.agx3\.wgemm\.iab', wgemm_iab),
    (r'm4r\.agx3\.wgemm\.ia', wgemm_ia),
    (r'm4r\.agx3\.wgemm\.f', wgemm_f),
    (r'm4r\.agx3\.wgemm\.ib', wgemm_ib),
    (r'm4r\.agx3\.wgemm\.i', wgemm_i),
    (r'm4r\.agx3\.a2w\.', acc_convert(True)),
    (r'm4r\.agx3\.w2a\.', acc_convert(False)),
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
