#!/usr/bin/env python3
"""SPIR-V binary -> C++ header with a uint32_t array."""
import struct, sys
spv, out, name = sys.argv[1:4]
d = open(spv, 'rb').read()
w = struct.unpack(f'<{len(d)//4}I', d)
body = ',\n'.join(', '.join(f'0x{x:08x}u' for x in w[i:i+8]) for i in range(0, len(w), 8))
open(out, 'w').write(f'// generated from {spv.rsplit("/",1)[-1]}\n#pragma once\n#include <cstdint>\n'
                     f'static constexpr uint32_t {name}[] = {{\n{body}\n}};\n')
