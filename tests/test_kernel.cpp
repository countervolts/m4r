// GPU smoke test for translated kernels: runs MetalFX's copyTexture (texture -> half buffer) on a known image
// and checks every value. Usage: test_kernel <kernel dir> (containing copyTexture.hsaco/.args, tg 8x8)
// SPDX-License-Identifier: GPL-3.0-only
#include "kernel.h"
#include <cmath>
#include <cstdio>

using namespace m4r;

static uint16_t toHalf(float f) {
    uint32_t x; std::memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000, e = (x >> 23) & 0xff, m = x & 0x7fffff;
    int ee = int(e) - 127 + 15;
    if (ee <= 0) return uint16_t(sign);
    if (ee >= 31) return uint16_t(sign | 0x7c00);
    return uint16_t(sign | (ee << 10) | ((m + 0x1000) >> 13));
}

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: %s <kernel dir>\n", argv[0]); return 2; }
    Context vk;
    if (!vk.create()) { std::fprintf(stderr, "vulkan: %s\n", vk.error.c_str()); return 1; }
    Runtime rt(vk);
    if (!rt.init()) { std::fprintf(stderr, "runtime: %s\n", rt.error.c_str()); return 1; }
    Kernel* k = rt.load(std::string(argv[1]) + "/copyTexture");
    if (!k) { std::fprintf(stderr, "load: %s\n", rt.error.c_str()); return 1; }

    const uint32_t W = 61, H = 37;  // not multiples of the 8x8 threadgroup: exercises the grid bounds
    Texture tex;
    if (!vk.createTexture(tex, VK_FORMAT_R16G16B16A16_SFLOAT, W, H)) { std::fprintf(stderr, "texture\n"); return 1; }
    std::vector<uint16_t> pix(W * H * 4);
    for (uint32_t y = 0; y < H; ++y)
        for (uint32_t x = 0; x < W; ++x)
            for (int c = 0; c < 4; ++c) pix[(y * W + x) * 4 + c] = toHalf(float(x) + float(y) / 64.0f + c * 0.25f);
    if (!vk.upload(tex, pix.data(), pix.size() * 2)) { std::fprintf(stderr, "upload %s\n", vk.error.c_str()); return 1; }

    Buffer out;
    vk.createBuffer(out, W * H * 4 * 2 + 256, true);
    std::memset(out.map, 0xff, out.size);
    VkCommandBuffer cb = vk.begin();
    Encoder enc(rt, cb);
    enc.setKernel(k);
    enc.setTexture(&tex, 0);
    enc.setBuffer(out, 0);
    const uint8_t channels = 4;
    enc.setBytes(&channels, 1, 1);
    enc.dispatchThreads(W, H);
    if (!vk.submit(cb)) { std::fprintf(stderr, "submit: %s\n", vk.error.c_str()); return 1; }

    const uint16_t* got = static_cast<const uint16_t*>(out.map);
    size_t bad = 0;
    for (size_t i = 0; i < pix.size(); ++i)
        if (got[i] != pix[i] && bad++ < 5) std::fprintf(stderr, "mismatch at %zu: %04x vs %04x\n", i, got[i], pix[i]);
    const bool tailUntouched = got[pix.size()] == 0xffff;
    std::printf("copyTexture %ux%u: %zu mismatches, tail %s\n", W, H, bad, tailUntouched ? "untouched" : "WRITTEN");
    vk.destroyBuffer(out);
    vk.destroyTexture(tex);
    rt.destroy();
    vk.destroy();
    return bad || !tailUntouched;
}
