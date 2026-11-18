// Scratch (private memory) in injected kernels: tests/scratch_test.ll keeps 7200 bytes per lane in private memory; this
// dispatches it with many waves and checks every lane's result. Kernels that spill registers need this to work.
// build the kernel: python3 air/m4r_air.py tests/scratch_test.ll scratch_test -o build/k/scratch_test.hsaco --tg 128,1,1
// usage: test_scratch <kernel base path (without extension)> [threads]
// SPDX-License-Identifier: GPL-3.0-only
#include "kernel.h"
#include <cstdio>
#include <cstdlib>

using namespace m4r;

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: %s <kernel base path> [threads]\n", argv[0]); return 2; }
    const uint32_t threads = argc > 2 ? uint32_t(std::atoi(argv[2])) : 128 * 900;
    Context vk;
    if (!vk.create()) { std::fprintf(stderr, "vulkan: %s\n", vk.error.c_str()); return 1; }
    Runtime rt(vk);
    if (!rt.init()) { std::fprintf(stderr, "runtime: %s\n", rt.error.c_str()); return 1; }
    Kernel* k = rt.load(argv[1]);
    if (!k) { std::fprintf(stderr, "load: %s\n", rt.error.c_str()); return 1; }
    Buffer out;
    vk.createBuffer(out, size_t(threads) * 4 + 256, true);
    std::memset(out.map, 0xff, out.size);
    VkCommandBuffer cb = vk.begin();
    Encoder enc(rt, cb);
    enc.setKernel(k);
    enc.setBuffer(out, 0);
    enc.dispatchThreads(threads, 1, 1);
    if (!vk.submit(cb)) { std::fprintf(stderr, "submit: %s\n", vk.error.c_str()); return 1; }
    const uint32_t* got = static_cast<const uint32_t*>(out.map);
    size_t bad = 0;
    for (uint32_t t = 0; t < threads; ++t) {
        const uint32_t want = 1800u * 7u * t + 1799u * 1800u / 2u;
        if (got[t] != want && bad++ < 5) std::fprintf(stderr, "thread %u: %u, wanted %u\n", t, got[t], want);
    }
    std::printf("scratch_test, %u threads (%u waves): %zu wrong\n", threads, threads / 32, bad);
    vk.destroyBuffer(out);
    rt.destroy();
    vk.destroy();
    return bad != 0;
}
