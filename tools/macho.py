"""Minimal Mach-O (arm64e dyld-cache extract) section reader."""
import struct
class MachO:
    def __init__(s,path):
        s.d=open(path,'rb').read(); d=s.d
        magic,cpu,sub,ft,ncmds,sz,fl,_=struct.unpack_from('<IiiIIIII',d,0)
        p=32; s.secs=[]
        for _ in range(ncmds):
            cmd,cs=struct.unpack_from('<II',d,p)
            if cmd==0x19:
                nsect=struct.unpack_from('<I',d,p+64)[0]
                q=p+72
                for i in range(nsect):
                    sn=d[q:q+16].rstrip(b'\0').decode(); seg=d[q+16:q+32].rstrip(b'\0').decode()
                    addr,size,off=struct.unpack_from('<QQI',d,q+32)
                    s.secs.append((sn,seg,addr,size,off)); q+=80
            p+=cs
    def off(s,a):
        for sn,seg,addr,size,off in s.secs:
            if addr<=a<addr+size and off: return off+(a-addr),sn
        return None,None
    def read(s,a,n):
        o,_=s.off(a); return None if o is None else s.d[o:o+n]
    def cstr(s,a):
        o,sn=s.off(a)
        if o is None: return None
        e=s.d.index(b'\0',o); return s.d[o:e].decode('latin1')
