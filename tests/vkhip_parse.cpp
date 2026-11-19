// Offline check of vkhip's code-object parsing and patching (no GPU): vkhip_parse <dataAddressHex> <file.hsaco>... prints each
// module's generation, kernels and patched getpc sites and writes <file>.patched.txt (the patched .text as hex words, for
// llvm-mc --disassemble). SPDX-License-Identifier: GPL-3.0-only
#include "vkhip/vkhip.h"
#include <cstdio>
#include <fstream>
#include <iterator>
int main(int argc, char** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: vkhip_parse <dataAddressHex> <file.hsaco>...\n"); return 2; }
    const uint64_t data = std::stoull(argv[1], nullptr, 16);
    int rc = 0;
    for (int i = 2; i < argc; ++i) {
        std::ifstream f(argv[i], std::ios::binary);
        std::vector<uint8_t> b((std::istreambuf_iterator<char>(f)), {});
        vkhip::Module m;
        if (!m.parse(b)) { std::printf("%s: FAIL %s\n", argv[i], m.error.c_str()); rc = 1; continue; }
        std::printf("%s: gfx%d mach %#x kernels %zu data sites %zu code getpc %u\n", argv[i], m.family, m.mach, m.kernels.size(),
                    m.sites.size(), m.textGetpc);
        const auto w = m.patchedText(data);
        std::ofstream o(std::string(argv[i]) + ".patched.txt");
        for (size_t k = 0; k < w.size(); ++k) {
            char s[64];
            std::snprintf(s, sizeof s, "0x%02x 0x%02x 0x%02x 0x%02x\n", w[k] & 255, (w[k] >> 8) & 255, (w[k] >> 16) & 255, w[k] >> 24);
            o << s;
        }
    }
    return rc;
}
