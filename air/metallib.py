#!/usr/bin/env python3
"""Read Apple .metallib containers (MTLB). Lists functions and extracts each one's AIR bitcode module.

Layout (MTLB v1.2, as found in macOS 15/27 MetalFX): header with offsets/sizes of the function list,
public/private metadata and the bitcode section; the function list is u32 count, then per function a u32
entry size and tagged fields (NAME, TYPE, HASH, OFFT = 3 x u64 offsets, MDSZ = bitcode size, ..., ENDT).
"""
import struct, sys

class Function:
    def __init__(self):
        self.name = None; self.type = None; self.offsets = None; self.size = None; self.tags = {}

def parse(data):
    if data[:4] != b'MTLB':
        raise ValueError('not a metallib')
    fl_off, fl_size, pub_off, pub_size, priv_off, priv_size, bc_off, bc_size = struct.unpack_from('<8Q', data, 0x18)
    count = struct.unpack_from('<I', data, fl_off)[0]
    p = fl_off + 4
    funcs = []
    for _ in range(count):
        size = struct.unpack_from('<I', data, p)[0]
        q, end = p + 4, p + size
        f = Function()
        while q < end:
            tag = data[q:q+4]; q += 4
            if tag == b'ENDT':
                break
            n = struct.unpack_from('<H', data, q)[0]; q += 2
            body = data[q:q+n]; q += n
            f.tags[tag.decode('latin1')] = body
            if tag == b'NAME': f.name = body.rstrip(b'\0').decode()
            elif tag == b'TYPE': f.type = body[0]
            elif tag == b'OFFT': f.offsets = struct.unpack('<3Q', body)
            elif tag == b'MDSZ': f.size = struct.unpack('<Q', body)[0]
        funcs.append(f)
        p = end
    return {'bitcode_offset': bc_off, 'bitcode_size': bc_size, 'functions': funcs}

def bitcode(data, lib, f):
    start = lib['bitcode_offset'] + f.offsets[2]
    return data[start:start + f.size]

if __name__ == '__main__':
    data = open(sys.argv[1], 'rb').read()
    lib = parse(data)
    for f in lib['functions']:
        print(f'{f.name:45} type={f.type} bc_off={f.offsets[2]:#x} size={f.size}')
        if len(sys.argv) > 3 and f.name == sys.argv[2]:
            open(sys.argv[3], 'wb').write(bitcode(data, lib, f))
