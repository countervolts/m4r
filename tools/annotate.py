#!/usr/bin/env python3
"""Annotated disassembly of functions in a MetalFX binary (extracted from the dyld shared cache).

usage: annotate.py <MetalFX binary> <function regex> [ipsw .a2s symbol map]
Resolves adrp+add/ldr pairs to C strings, CFStrings (shared-cache pointers: 0x180000000 + low 34 bits), symbols,
GOT entries and constants, and names the dyld-cache stub islands whose selector is known from call shape.
"""
import os,re,sys,pickle,struct,subprocess
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import os
from macho import MachO
BIN=sys.argv[1]
m=MachO(BIN)
syms={}
if len(sys.argv) > 3:
    import a2s
    syms.update(a2s.load(sys.argv[3]))
for l in subprocess.run(['llvm-nm','-C',BIN],capture_output=True,text=True).stdout.split('\n'):
    p=l.split(' ',2)
    if len(p)==3 and p[0]: syms.setdefault(int(p[0],16),p[2])
def desc(a,isldr,width):
    if a in syms: s=syms[a]
    else: s=None
    o,sn=m.off(a)
    if sn=='__cstring': return '"'+m.cstr(a)+'"'
    if sn=='__cfstring':
        p=0x180000000+(struct.unpack_from("<Q",m.d,o+16)[0]&0x3ffffffff)
        return '@"'+(m.cstr(p) or '?')+'"'
    if sn in('__got','__auth_got','__weak_auth_got') and s: return s
    if isldr and o is not None and sn and 'const' in sn:
        raw=m.d[o:o+width]
        if width==4: return f'{s or ""}[{struct.unpack("<f",raw)[0]:g} / 0x{raw[::-1].hex()}]'
        if width==8: f=struct.unpack('<2f',raw); return f'{s or ""}[{f[0]:g},{f[1]:g} / 0x{raw[::-1].hex()}]'
        if width==16: return f'{s or ""}i{struct.unpack("<4I",raw)} f{tuple(round(x,6) for x in struct.unpack("<4f",raw))}'
    if s: return s
    return sn and f'{sn}+{hex(a)}'
STUBS={0x258004690:'setComputePipelineState:',0x258015f70:'setTexture:atIndex:',0x258004620:'setBytes:length:atIndex:',0x258015ad0:'setBuffer:offset:atIndex:',0x258014e60:'dispatchThreadgroups?:threadsPerThreadgroup:',0x2580592c0:'retain',0x2580592b0:'release',0x2580591e0:'retainAutoreleased?',0x258001e20:'dispatchThreads:threadsPerThreadgroup:'}
def load_funcs():
    out = subprocess.run(['llvm-objdump', '-d', '--no-show-raw-insn', BIN], capture_output=True, text=True).stdout
    funcs = {}; cur = None
    for l in out.split('\n'):
        mm = re.match(r'^([0-9a-f]+) <(.*)>:$', l)
        if mm: cur = mm.group(2); funcs[cur] = []; continue
        if cur: funcs[cur].append(l)
    return funcs
import os
cache = os.path.join(os.environ.get('XDG_CACHE_HOME', os.path.expanduser('~/.cache')), 'm4r', 'objdump.pkl')
if os.path.exists(cache):
    funcs = pickle.load(open(cache, 'rb'))
else:
    funcs = load_funcs(); os.makedirs(os.path.dirname(cache), exist_ok=True); pickle.dump(funcs, open(cache, 'wb'))
pat=sys.argv[2]
for name,lines in funcs.items():
    if not re.search(pat,name): continue
    print('=====',name)
    pages={}
    for l in lines:
        mm=re.match(r'\s*([0-9a-f]+):\s+(\S+)\s*(.*)',l)
        if not mm: continue
        ad,op,args=mm.groups(); note=''
        if op=='adrp':
            r,v=re.match(r'(\w+), (0x[0-9a-f]+)',args).groups(); pages[r]=int(v,16)
        elif op in('add','ldr','ldrsw','ldrb','ldrh') :
            g=re.match(r'([wxsdq]\d+), \[?(x\d+), #(0x[0-9a-f]+|\d+)\]?',args)
            if g and g.group(2) in pages:
                a=pages[g.group(2)]+int(g.group(3),0)
                w={'s':4,'d':8,'q':16,'w':4,'x':8}[g.group(1)[0]]
                note=desc(a,op!='add',w)
                if op=='add' and g.group(1).startswith('x'): pages[g.group(1)]=a  # keep for chained add
        elif op in('bl','b'):
            t=re.match(r'(0x[0-9a-f]+)',args)
            if t:
                ta=int(t.group(1),16)
                if ta in syms: note=syms[ta]
                elif ta>=0x258000000: note=STUBS.get(ta,'stub_'+hex(ta))
        args=re.sub(r'<.*?>','',args).split(';')[0].strip()
        print(f'{ad[-5:]} {op:7} {args:40} {("; "+note) if note else ""}')
