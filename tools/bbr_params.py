#!/usr/bin/env python3
"""Extract tBBRNet's per-layer parameters from the user's MetalFX binary.

Each `tBBRNet_Generate*_StepN(device, NSData*, H, W[, bool])` fills a Conv3x3Stride1/Stride2 descriptor with
constants (channel counts, quantisation scales and zero points, flags) and slices the weight file. Instead of
copying those numbers into m4r's sources, this tool runs a small symbolic interpreter over the function's arm64
code: it tracks immediates, constant-pool loads and the stores into the descriptor (x8 at entry, the sret
pointer), and records the arguments of the weight-slicing helpers.

usage: bbr_params.py <MetalFX binary> [out.json]

Output (per layer): the descriptor bytes written (offset -> value/kind), H/W as a function of the network size
(scale and shift), and the weight-file ranges (weights, scales, zero points, bias) plus the bias-folding
arguments.
"""
import json, os, re, shutil, struct, subprocess, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from macho import MachO  # noqa: E402

HELPERS = {
    'extractPerOutputChannelSlice': 'slice',
    'extractPerOutputChannel': 'per_channel',
    'adjustBiasForInputActZp': 'adjust_bias',
    'dequantizeUint8WeightsToFP16': 'dequant',
}
SUBDATA_STUB = 0x258016160  # -[NSData subdataWithRange:] (stub island, identified by call shape)

# Conv3x3Stride1Descriptor field offsets -> function constant / meaning (read from -[Conv3x3Stride1
# setupMetalWithLibrary:], which loads ivars at descriptor offset + 0x20)
S1_FIELDS = {0x08: ('K', 'u16'), 0x0a: ('C', 'u16'), 0x0c: ('CSkip', 'u16'), 0x0e: ('H', 'u16'), 0x10: ('W', 'u16'),
             0x20: ('bilinearSkip', 'u8'), 0x24: ('DqsBilinearImage', 'f32'), 0x28: ('DqsSkip', 'f32'),
             0x2c: ('QsI', 'f32'), 0x30: ('QsRinv', 'f32'), 0x68: ('outputBlockK', 'u16'),
             0x6a: ('simdgroupsPerTG', 'u16'), 0x6c: ('NmatsHW', 'u16'), 0x6e: ('NmatsK', 'u16'),
             0x70: ('inputIsQuantized', 'u8'), 0x71: ('inputIsSigned', 'u8'), 0x88: ('outputToTexture', 'u8'),
             0x8a: ('separateBilinear', 'u8'), 0x8b: ('enc0FQSplit', 'u8'), 0xa0: ('useSigmoid', 'u8'),
             0xa1: ('useD2S', 'u8'), 0xa2: ('useLeakyReLU', 'u8'), 0xa4: ('outZp', 'f32'),
             0xa8: ('combinedInZp', 'f32'), 0xac: ('bilinearImageZp', 'f32'), 0xb0: ('skipZpAct', 'f32'),
             0xb4: ('blockMajorWeights', 'u8'), 0xb5: ('headTemporalTex', 'u8'), 0xb6: ('headRnnTex', 'u8'),
             0x18: ('buf_weights', 'ptr'), 0x38: ('buf_scales', 'ptr'), 0x40: ('buf_bias', 'ptr'),
             0x60: ('buf_zp', 'ptr')}


def objdump(binary):
    # M4R_HOST_OBJDUMP: an llvm-objdump with the AArch64 target, for LLVM builds without it (AMD's TheRock SDK on Windows)
    tool = os.environ.get('M4R_HOST_OBJDUMP', 'llvm-objdump')
    r = subprocess.run([shutil.which(tool) or tool, '-d', '--no-show-raw-insn', binary], capture_output=True, text=True)
    if r.returncode and 'cannot find target' in r.stderr:
        raise SystemExit(f'{tool} cannot disassemble the MetalFX binary (no AArch64 target): set M4R_HOST_OBJDUMP to an llvm-objdump that has it')
    out = r.stdout
    funcs, cur = {}, None
    for l in out.split('\n'):
        m = re.match(r'^([0-9a-f]+) <(.*)>:$', l)
        if m:
            cur = m.group(2); funcs[cur] = []; continue
        if cur:
            mm = re.match(r'\s*([0-9a-f]+):\s+(\S+)\s*(.*)', l)
            if mm:
                funcs[cur].append((int(mm.group(1), 16), mm.group(2), mm.group(3).split(';')[0].strip()))
    return funcs


def imm(s):
    s = s.strip().lstrip('#')
    return int(s, 0)


class Sym:
    """A symbolic value: ('arg', name, scale, shift) for H/W derived values, or a python number."""
    def __init__(self, base, shift=0):
        self.base, self.shift = base, shift

    def __repr__(self):
        return f'{self.base}>>{self.shift}' if self.shift else self.base


def run(m, insns, nsym):
    """Interpret one Generate function. nsym: names for w2/w3 (and w4)."""
    regs = {'x1': 'data'}
    for i, n in enumerate(nsym):
        regs[f'x{2 + i}'] = Sym(n)
    fregs = {}
    desc = {}
    calls = []
    pages = {}

    def get(r):
        r = r.replace('w', 'x') if r.startswith('w') else r
        if r in ('xzr', 'wzr'):
            return 0
        return regs.get(r)

    def setr(r, v):
        regs[r.replace('w', 'x') if r.startswith('w') else r] = v

    base_reg = 'x8'
    regs['x8'] = 'DESC'
    for addr, op, args in insns:
        a = [x.strip() for x in re.split(r',(?![^\[]*\])', args)] if args else []
        try:
            if op == 'mov' and len(a) == 2:
                if a[1].startswith('#'):
                    setr(a[0], imm(a[1].split(';')[0]))
                else:
                    setr(a[0], get(a[1]))
            elif op == 'movk':
                v = get(a[0]) or 0
                sh = int(a[2].split('#')[1]) if len(a) > 2 else 0
                v = (v & ~(0xffff << sh)) | (imm(a[1]) << sh)
                setr(a[0], v)
            elif op == 'lsr' and isinstance(get(a[1]), Sym):
                s = get(a[1]); setr(a[0], Sym(s.base, s.shift + imm(a[2])))
            elif op == 'adrp':
                pages[a[0]] = int(a[1].split()[0], 16)
            elif op == 'add' and len(a) == 3 and a[2].startswith('#') and a[1] in pages:
                pages[a[0]] = pages[a[1]] + imm(a[2])
            elif op in ('ldr', 'ldur') and a[1].startswith('['):
                mm = re.match(r'\[(x\d+)(?:, #(-?0x[0-9a-f]+|-?\d+))?\]', a[1])
                if mm and mm.group(1) in pages:
                    ad = pages[mm.group(1)] + (int(mm.group(2), 0) if mm.group(2) else 0)
                    w = {'s': 4, 'd': 8, 'q': 16}.get(a[0][0])
                    if w:
                        fregs[a[0][1:]] = ('mem', m.read(ad, w))
            elif op == 'fmov':
                if a[1].startswith('#'):
                    v = float(a[1][1:])
                    fregs[a[0][1:]] = ('mem', struct.pack('<f', v))
            elif op == 'fmov.4s':
                v = float(a[1][1:])
                fregs[a[0][1:]] = ('mem', struct.pack('<4f', v, v, v, v))
            elif op == 'movi.2d':
                fregs[a[0][1:]] = ('mem', bytes(16))
            elif op in ('str', 'stur', 'strh', 'strb', 'stp') and '[x20' in args or (op in ('str', 'stur', 'strh', 'strb', 'stp') and base_reg in args and 'sp' not in args):
                mm = re.search(r'\[(x\d+)(?:, #(-?0x[0-9a-f]+|-?\d+))?\]', args)
                if not mm or regs.get(mm.group(1)) != 'DESC':
                    continue
                off = int(mm.group(2), 0) if mm.group(2) else 0
                srcs = a[:2] if op == 'stp' else a[:1]
                for j, r in enumerate(srcs):
                    width = {'strh': 2, 'strb': 1}.get(op)
                    if r[0] in 'qsdh' and r[1:].isdigit():
                        kind, data = fregs.get(r[1:], ('mem', None))
                        n = {'q': 16, 'd': 8, 's': 4, 'h': 2}[r[0]]
                        if data is not None:
                            data = (data + bytes(16))[:n]
                            for b in range(n):
                                desc[off + j * n + b] = data[b]
                        continue
                    v = get(r)
                    n = width or (8 if r.startswith('x') else 4)
                    if isinstance(v, Sym):
                        desc[('sym', off + j * n)] = (v, n)
                        continue
                    if isinstance(v, int):
                        for b in range(n):
                            desc[off + j * n + b] = (v >> (8 * b)) & 0xff
                    else:
                        desc[('ptr', off + j * n)] = v
            elif op == 'bl':
                t = int(a[0].split()[0], 16)
                name = next((v for k, v in HELPERS.items() if k in args), None)
                fargs = [get(f'x{i}') for i in range(8)]
                fl = [fregs.get(f'{i}') for i in range(2)]
                if name:
                    calls.append({'helper': name, 'x': fargs, 'f': [struct.unpack('<f', f[1][:4])[0] if f else None for f in fl]})
                    setr('x0', f'ret_{name}_{len(calls)}')
                elif t == SUBDATA_STUB:
                    calls.append({'helper': 'subdata', 'x': fargs})
                    setr('x0', f'ret_subdata_{len(calls)}')
                else:
                    if isinstance(get('x0'), str) and get('x0').startswith('ret_'):
                        pass  # objc_retainAutoreleasedReturnValue / retain keep x0
                    else:
                        setr('x0', None)
                for r in ('x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x9', 'x10', 'x11', 'x16', 'x17'):
                    regs.pop(r, None)
            elif op == 'mov' or op == 'orr':
                pass
            if op in ('mov',) and len(a) == 2 and a[1] == 'x8' and regs.get('x8') == 'DESC':
                pass
        except (ValueError, IndexError, KeyError):
            continue
    return desc, calls


def decode(desc, fields):
    out = {}
    for off, (name, kind) in fields.items():
        if kind == 'ptr':
            continue
        n = {'u8': 1, 'u16': 2, 'f32': 4}[kind]
        if ('sym', off) in desc:
            s, _ = desc[('sym', off)]
            out[name] = {'size': s.base, 'shift': s.shift}
            continue
        bs = [desc.get(off + i) for i in range(n)]
        if any(b is None for b in bs):
            continue
        raw = bytes(bs)
        out[name] = struct.unpack('<f', raw)[0] if kind == 'f32' else int.from_bytes(raw, 'little')
    return out


def u(v):
    return v if isinstance(v, int) else None


def layer_table(binary):
    m = MachO(binary)
    funcs = objdump(binary)
    layers = []
    for name, insns in funcs.items():
        mm = re.match(r'__Z\d+tBBRNet_GenerateConv3x3(S[12])_Step(\d+)', name)
        if not mm:
            continue
        kind, step = mm.group(1), int(mm.group(2))
        desc, calls = run(m, insns, ['H', 'W', 'flag'])
        d = decode(desc, S1_FIELDS) if kind == 'S1' else decode(desc, {
            0x08: ('K', 'u16'), 0x0a: ('C', 'u16'), 0x0c: ('H', 'u16'), 0x0e: ('W', 'u16'), 0x18: ('QsI', 'f32'),
            0x1c: ('QsRinv', 'f32'), 0x30: ('flag30', 'u8'), 0x52: ('useLeakyReLU', 'u8'), 0x54: ('outZp', 'f32')})
        w = {}
        for c in calls:
            x = c['x']
            if c['helper'] == 'per_channel':
                w.update(weights=[u(x[1]), u(x[1]) + x[3] * x[4] * x[5] * x[6] if u(x[1]) is not None else None],
                         scales=[u(x[2]), u(x[2]) + 2 * x[3] if u(x[2]) is not None else None], K=x[3], C=x[4], kh=x[5], kw=x[6])
            elif c['helper'] == 'slice':
                w.update(slice={'w_off': u(x[1]), 'scale_off': u(x[2]), 'K_total': x[3], 'C': x[4], 'k_first': x[5],
                                'k_count': x[6], 'kh': x[7]})
            elif c['helper'] == 'subdata':
                w.setdefault('subdata', []).append([u(x[2]), u(x[3])])
            elif c['helper'] == 'adjust_bias':
                w['adjust_bias'] = {'act_scale': c['f'][0], 'act_zp': c['f'][1]}
        layers.append({'step': step, 'kind': kind, 'function': name, 'desc': d, 'weights': w})
    layers.sort(key=lambda l: l['step'])
    return layers


if __name__ == '__main__':
    t = layer_table(sys.argv[1])
    s = json.dumps(t, indent=1, default=str)
    if len(sys.argv) > 2:
        open(sys.argv[2], 'w').write(s)
    else:
        print(s)
