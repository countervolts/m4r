# SPDX-License-Identifier: GPL-3.0-only
"""Widening: a kernel's half-precision arithmetic is computed in f32 instead.

Where Apple's kernel rounds a value to half and keeps computing, the widened kernel keeps f32 and rounds only where the
value has to be half: memory (device, threadgroup and private arrays that survive SROA), the fp16 GEMM operands, opaque
calls and bitcasts. It is the same formula in a wider format, so it is not an approximation, but it is also not
bit-identical to the half-precision kernel (see docs/air-translation.md, "Widening").

It runs on the module after the translator's first steps (target, address spaces, attributes) and before the AIR intrinsic
bodies are added, on IR that `opt -O3` has already inlined and scalarised, so private arrays between the layers of a kernel
are registers. Rules for each SSA value of type half / <N x half>:

- arithmetic, fcmp, select, phi, freeze, vector element ops, AIR / LLVM math intrinsics: the same op in f32;
- `fptrunc float -> half` and AIR conversions between half and float: identities (that is the widening);
- int -> half conversions convert to f32; half -> int conversions take the f32 value;
- texture reads, samples, gathers and writes use the f32 variant (the hardware converts the texel format directly);
- loads, bitcasts, extractvalue of other aggregates, results of other calls and function parameters stay half and get an
  `fpext` after them; stores, other calls, bitcasts, insertvalue and returns get an `fptrunc` before them.
"""
import re
import struct

FLAG = r'(?:(?:fast|nnan|ninf|nsz|arcp|contract|afn|reassoc)\s+)'
HT = r'(?:half|<\d+ x half>)'
VAL = r'(?:%[-\w.$]+|%"[^"]*")'
PATTR = r'(?:\s+(?:noundef|nonnull|signext|zeroext|inreg|immarg|nofpclass\([^)]*\)))*'

# pure math whose lowering (air/intrinsics.py) is generic over the float type
AIR_MATH = re.compile(r'air\.(fast_)?(fabs|floor|ceil|sqrt|rint|round|trunc|exp|exp2|log|log2|rsqrt|fract|fmax3|fmin3|fmax|fmin|'
                      r'clamp|saturate|mix|dot|powr?|sinpi|cospi|tanh|sign|copysign)\.')
AIR_LANE = re.compile(r'air\.(simd|quad)_(shuffle_xor|sum|min|max)\.')
AIR_CONVERT = re.compile(r'air\.convert\.')
AIR_TEX_READ = re.compile(r'air\.(read|sample|gather)_texture_2d')
AIR_TEX_WRITE = re.compile(r'air\.write_texture_2d')
LLVM_MATH = re.compile(r'llvm\.(fabs|minnum|maxnum|minimum|maximum|minimumnum|maximumnum|copysign|floor|ceil|trunc|rint|'
                       r'nearbyint|round|roundeven|sqrt|fma|fmuladd|exp|exp2|log|log2|pow|sin|cos|canonicalize|'
                       r'fpto[su]i\.sat|vector\.reduce\.f(add|mul|min|max|minimum|maximum))\.')


def f16_to_f32_name(name):
    return re.sub(r'(?<=[.v\d])f16\b', 'f32', re.sub(r'\.f16\b', '.f32', name))


def float_type(t):
    return t.replace('half', 'float')


def half_const(tok):
    """A half constant (decimal or 0xH) -> the same value as an LLVM float literal (double hex)."""
    if tok.startswith('0xH'):
        v = struct.unpack('<e', int(tok[3:], 16).to_bytes(2, 'little'))[0]
    else:
        v = struct.unpack('<e', struct.pack('<e', float(tok)))[0]
    return '0x%016X' % struct.unpack('<Q', struct.pack('<d', v))[0]


CONST = re.compile(r'\bhalf\s+(0xH[0-9A-Fa-f]{4}|[-+]?\d+\.\d+(?:e[-+]?\d+)?)')


def widen_text(s):
    """Retype an instruction's half operands and constants to float."""
    s = CONST.sub(lambda m: 'float ' + half_const(m.group(1)), s)
    return re.sub(r'\bhalf\b', 'float', s)


def split_top(s):
    """Split a type/argument list at top-level commas."""
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


class Fn:
    def __init__(self, struct_types):
        self.kind = {}          # value -> 'w' (now f32) / 'p' (still half, f32 copy is %m4rw.<name>) / 'agg' (texture result)
        self.htype = {}         # physical value -> its half type
        self.n = 0
        self.structs = struct_types

    def tmp(self):
        self.n += 1
        return f'%m4rt.{self.n}'

    def wide(self, v):
        return '%m4rw.' + v[1:] if not v.startswith('%"') else '%"m4rw.' + v[2:]

    def use_wide(self, s):
        """In an instruction that now computes in f32: physical half operands -> their f32 copies."""
        return re.sub(VAL + r'(?![-\w.$])', lambda m: self.wide(m.group(0)) if self.kind.get(m.group(0)) == 'p' else m.group(0), s)

    def use_half(self, s, pre):
        """In an instruction that needs half operands: f32 values -> an fptrunc emitted before it."""
        def rep(m):
            t, attrs, v = m.group(1), m.group(2), m.group(3)
            if self.kind.get(v) == 'w':
                n = self.tmp()
                pre.append(f'  {n} = fptrunc {float_type(t)} {v} to {t}')
                v = n
            return f'{t}{attrs} {v}'
        return re.sub(r'(' + HT + r')(' + PATTR + r')\s+(' + VAL + r')(?![-\w.$])', rep, s)

    def physical(self, name, t, post):
        self.kind[name] = 'p'
        self.htype[name] = t
        post.append(f'  {self.wide(name)} = fpext {t} {name} to {float_type(t)}')

    def agg_elem(self, aggt, idx):
        aggt = aggt.strip()
        if aggt.startswith('%'):
            aggt = self.structs.get(aggt, '')
        if aggt.startswith('<{'):
            aggt = aggt[1:-1]
        if not aggt.startswith('{'):
            m = re.match(r'\[\d+ x (.*)\]$', aggt)
            return m.group(1) if m else ''
        parts = split_top(aggt.strip()[1:-1])
        return parts[idx] if idx < len(parts) else ''


def rewrite_function(lines, struct_types, decls):
    # pass 1 only classifies the values (a phi can use a value defined further down); pass 2 rewrites with all kinds known
    kinds = {}
    for _ in range(2):
        fn = Fn(struct_types)
        fn.kind.update(kinds)
        head = lines[0]
        out = [head]
        entry_post = []
        m = re.match(r'define [^@]*@[-\w.$"]+\((.*)\)[^)]*\{$', head)
        if m:
            for p in split_top(m.group(1)):
                pm = re.match(r'(' + HT + r')\b.*?(' + VAL + r')$', p)
                if pm:
                    fn.physical(pm.group(2), pm.group(1), entry_post)
        # the copies of half parameters go after the entry label (or at the top)
        started = False
        for line in lines[1:]:
            if not started and line.strip() and not line.strip().startswith(';'):
                started = True
                if re.match(r'^[-\w.$"]+:', line):
                    out.append(line)
                    out += entry_post
                    continue
                out += entry_post
            out += rewrite_line(fn, line, decls)
        kinds = fn.kind
    return out


def rewrite_line(fn, line, decls):
    s = line.rstrip()
    st = s.strip()
    if not st or st.startswith(';') or re.match(r'^[-\w.$"]+:', st) or st == '}' or 'half' not in st:
        return [line]
    pre, post = [], []
    m = re.match(r'(\s*)(' + VAL + r') = (.*)$', s)
    if m:
        ind, name, rhs = m.groups()
    else:
        ind, name, rhs = re.match(r'(\s*)()(.*)$', s).groups()
    op = rhs.split()[0]
    if op in ('tail', 'musttail', 'notail'):
        op = rhs.split()[1]

    def emit(new_rhs):
        return pre + [f'{ind}{name} = {new_rhs}' if name else f'{ind}{new_rhs}'] + post

    if op in ('fadd', 'fsub', 'fmul', 'fdiv', 'frem', 'fneg', 'fcmp', 'select', 'phi', 'freeze',
              'extractelement', 'insertelement', 'shufflevector'):
        if name and op != 'fcmp':
            fn.kind[name] = 'w'
        return emit(fn.use_wide(widen_text(rhs)))
    if op == 'fptrunc':
        mm = re.match(r'fptrunc ' + FLAG + r'*(\S+(?: x \w+>)?) (' + VAL + r'|\S+) to (' + HT + r')$', rhs)
        if not mm:
            raise ValueError(f'widen: {s}')
        src, v = mm.group(1), mm.group(2)
        fn.kind[name] = 'w'
        if 'double' in src:
            return emit(f'fptrunc {src} {v} to {float_type(mm.group(3))}')
        return emit(f'bitcast {src} {v} to {src}')
    if op == 'fpext':
        mm = re.match(r'fpext ' + FLAG + r'*(' + HT + r') (\S+) to (\S+(?: x \w+>)?)$', rhs)
        if not mm:
            raise ValueError(f'widen: {s}')
        t, v, dst = mm.groups()
        if fn.kind.get(v) == 'w':
            ft = float_type(t)
            return emit(f'bitcast {ft} {v} to {ft}' if 'double' not in dst else f'fpext {ft} {v} to {dst}')
        return [line]
    if op in ('sitofp', 'uitofp'):
        if re.search(r'to ' + HT + r'$', rhs):
            fn.kind[name] = 'w'
            return emit(widen_text(rhs))
        return [line]
    if op in ('fptosi', 'fptoui'):
        mm = re.search(r'(' + VAL + r') to', rhs)
        if mm and fn.kind.get(mm.group(1)) == 'w':
            return emit(widen_text(rhs))
        return [line]
    if op == 'load':
        mm = re.match(r'load (?:atomic )?(?:volatile )?(' + HT + r'),', rhs)
        if mm:
            fn.physical(name, mm.group(1), post)
        return emit(rhs)
    if op == 'bitcast':
        mm = re.match(r'bitcast (.*) to (.*)$', rhs)
        if re.fullmatch(HT, mm.group(2).strip()):
            fn.physical(name, mm.group(2).strip(), post)
        return emit(fn.use_half(rhs, pre))
    if op == 'extractvalue':
        mm = re.match(r'extractvalue (.*?) (' + VAL + r'), (\d+)', rhs)
        if mm and fn.kind.get(mm.group(2)) == 'agg':
            if mm.group(3) == '0':
                fn.kind[name] = 'w'
            return emit(widen_text(rhs))
        if mm:
            el = fn.agg_elem(mm.group(1), int(mm.group(3)))
            if re.fullmatch(HT, el):
                fn.physical(name, el, post)
        return [line] + post
    if op in ('store', 'insertvalue', 'ret'):
        return emit(fn.use_half(rhs, pre))
    if op == 'call':
        cm = re.search(r'@([-\w.$]+)\(', rhs)
        callee = cm.group(1) if cm else ''
        rm = re.match(r'(?:(?:tail|musttail|notail) )?call ' + FLAG + r'*(?:\w+cc )?(?:(?:noundef|nonnull|zeroext|signext|inreg|nofpclass\([^)]*\))\s+)*((?:\{[^}]*\})|\S+(?: x \w+>)?) ', rhs)
        rt = rm.group(1) if rm else ''
        if AIR_MATH.match(callee) or AIR_LANE.match(callee) or LLVM_MATH.match(callee):
            new = f16_to_f32_name(callee)
            decls[callee] = new
            if name:
                fn.kind[name] = 'w' if re.fullmatch(HT, rt) else fn.kind.get(name)
            return emit(fn.use_wide(widen_text(rhs.replace('@' + callee + '(', '@' + new + '('))))
        if AIR_CONVERT.match(callee):
            new = f16_to_f32_name(callee)
            cv = re.fullmatch(r'air\.convert\.f\.(\w+)\.f\.(\w+)', new)
            if cv and cv.group(1) == cv.group(2):   # half <-> float: identity
                args = split_top(rhs[rhs.index('(') + 1:rhs.rindex(')')])
                am = re.match(r'(\S+(?: x \w+>)?)' + PATTR + r'\s+(\S+)$', args[0])
                ft = float_type(am.group(1))
                if name:
                    fn.kind[name] = 'w'
                v = am.group(2)
                if fn.kind.get(v) == 'p':
                    v = fn.wide(v)
                elif v.startswith('0xH') or re.match(r'[-+]?\d', v):
                    v = half_const(v)
                return emit(f'bitcast {ft} {v} to {ft}')
            decls[callee] = new
            if name and re.fullmatch(HT, rt):
                fn.kind[name] = 'w'
            return emit(fn.use_wide(widen_text(rhs.replace('@' + callee + '(', '@' + new + '('))))
        if AIR_TEX_READ.match(callee) and re.search(r'<\d+ x half>', rt):
            new = f16_to_f32_name(callee)
            decls[callee] = new
            if name:
                fn.kind[name] = 'agg'
            return emit(widen_text(rhs.replace('@' + callee + '(', '@' + new + '(')))
        if AIR_TEX_WRITE.match(callee):
            new = f16_to_f32_name(callee)
            decls[callee] = new
            return emit(fn.use_wide(widen_text(rhs.replace('@' + callee + '(', '@' + new + '('))))
        # any other call: half arguments and results stay half
        new_rhs = fn.use_half(rhs, pre)
        if name and re.fullmatch(HT, rt):
            fn.physical(name, rt, post)
        return emit(new_rhs)
    if op in ('getelementptr', 'alloca', 'addrspacecast', 'ptrtoint', 'inttoptr', 'icmp', 'br', 'switch', 'unreachable'):
        return [line]
    if re.search(r'\b' + HT + r'\b', rhs):
        raise ValueError(f'widen: unsupported instruction: {s}')
    return [line]


def widen(text):
    lines = text.split('\n')
    structs = {}
    for ln in lines:
        mm = re.match(r'^(%[-\w.$"]+) = type (.*)$', ln)
        if mm:
            structs[mm.group(1)] = mm.group(2)
    out, decls, i = [], {}, 0
    while i < len(lines):
        if lines[i].startswith('define '):
            j = i
            while lines[j] != '}':
                j += 1
            out += rewrite_function(lines[i:j + 1], structs, decls)
            i = j + 1
        else:
            out.append(lines[i])
            i += 1
    text = '\n'.join(out)
    # declarations of the f32 variants
    declared = set(re.findall(r'^declare [^\n]*?@([-\w.$]+)\(', text, re.M))
    for old, new in decls.items():
        if new in declared or old == new:
            continue
        dm = re.search(r'^declare [^\n]*?@' + re.escape(old) + r'\([^\n]*$', text, re.M)
        if not dm:
            raise ValueError(f'widen: no declaration of {old}')
        d = widen_text(dm.group(0).replace('@' + old + '(', '@' + new + '('))
        text += '\n' + d
        declared.add(new)
    return text
