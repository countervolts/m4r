// Runs a prepared plan on synthetic inputs: test_plan <plan dir> [fill=name,...] [dump=name,...] [runs=N]
// Fills the named buffers with deterministic pseudo-random bytes, runs the plan, checks that repeated runs give
// identical output, reports GPU time per run and writes the dumped buffers to <plan dir>/dump-<name>.bin.
// SPDX-License-Identifier: GPL-3.0-only
#include "plan.h"
#include <chrono>
#include <cstdio>
#include <fstream>
#include <sstream>

using namespace m4r;

static std::vector<std::string> split(const std::string& s) {
    std::vector<std::string> out;
    std::stringstream ss(s);
    std::string x;
    while (std::getline(ss, x, ',')) if (!x.empty()) out.push_back(x);
    return out;
}

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: %s <plan dir> [fill=a,b] [dump=c] [runs=N]\n", argv[0]); return 2; }
    std::string dir = argv[1];
    std::vector<std::string> fill, dump, load;
    bool probe = false;
    int runs = 3;
    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        if (a.rfind("fill=", 0) == 0) fill = split(a.substr(5));
        else if (a.rfind("dump=", 0) == 0) dump = split(a.substr(5));
        else if (a.rfind("runs=", 0) == 0) runs = std::stoi(a.substr(5));
        else if (a.rfind("load=", 0) == 0) load = split(a.substr(5));
        else if (a == "probe") probe = true;
    }
    Context vk;
    if (!vk.create()) { std::fprintf(stderr, "vulkan: %s\n", vk.error.c_str()); return 1; }
    Runtime rt(vk);
    if (!rt.init()) { std::fprintf(stderr, "runtime: %s\n", rt.error.c_str()); return 1; }
    int rc = 0;
    {
        Plan plan(rt);
        auto t0 = std::chrono::steady_clock::now();
        if (!plan.load(dir)) { std::fprintf(stderr, "plan: %s\n", plan.error.c_str()); return 1; }
        auto t1 = std::chrono::steady_clock::now();
        std::printf("loaded %zu passes in %.1f ms\n", plan.passes(), std::chrono::duration<double, std::milli>(t1 - t0).count());
        uint32_t seed = 12345;
        for (auto& n : fill) {
            Buffer* b = plan.buffer(n);
            if (!b) { std::fprintf(stderr, "no buffer %s\n", n.c_str()); return 1; }
            auto* p = static_cast<uint8_t*>(b->map);
            for (VkDeviceSize i = 0; i < b->size; ++i) { seed = seed * 1664525u + 1013904223u; p[i] = uint8_t(seed >> 24); }
        }
        for (auto& l : load) {   // name:path
            auto c = l.find(':');
            Buffer* b = plan.buffer(l.substr(0, c));
            std::ifstream f(l.substr(c + 1), std::ios::binary);
            if (!b || !f) { std::fprintf(stderr, "load %s failed\n", l.c_str()); return 1; }
            f.read(static_cast<char*>(b->map), std::streamsize(b->size));
        }
        VkQueryPool qp;
        VkQueryPoolCreateInfo qci{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
        qci.queryType = VK_QUERY_TYPE_TIMESTAMP; qci.queryCount = 2;
        vkCreateQueryPool(vk.device, &qci, nullptr, &qp);
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(vk.physical, &props);
        std::vector<std::vector<uint8_t>> first;
        for (int r = 0; r < runs; ++r) {
            rt.resetTransient();
            VkCommandBuffer cb = vk.begin();
            vkCmdResetQueryPool(cb, qp, 0, 2);
            vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, qp, 0);
            if (!plan.record(cb)) { std::fprintf(stderr, "record: %s\n", plan.error.c_str()); return 1; }
            vk.barrier(cb);
            vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, qp, 1);
            if (!vk.submit(cb)) { std::fprintf(stderr, "submit: %s\n", vk.error.c_str()); return 1; }
            uint64_t ts[2];
            vkGetQueryPoolResults(vk.device, qp, 0, 2, sizeof(ts), ts, 8, VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
            std::printf("run %d: %.3f ms GPU\n", r, double(ts[1] - ts[0]) * props.limits.timestampPeriod * 1e-6);
            for (size_t i = 0; i < dump.size(); ++i) {
                Buffer* b = plan.buffer(dump[i]);
                if (!b) continue;
                std::vector<uint8_t> v(static_cast<uint8_t*>(b->map), static_cast<uint8_t*>(b->map) + b->size);
                if (r == 0) first.push_back(v);
                else if (v != first[i]) { std::printf("  %s differs from run 0\n", dump[i].c_str()); rc = 1; }
            }
        }
        for (size_t i = 0; i < dump.size() && i < first.size(); ++i) {
            std::ofstream(dir + "/dump-" + dump[i] + ".bin", std::ios::binary).write(reinterpret_cast<const char*>(first[i].data()), first[i].size());
            size_t zero = 0, ff = 0;
            for (uint8_t x : first[i]) { zero += x == 0; ff += x == 255; }
            std::printf("%s: %zu bytes, %.1f%% zero, %.1f%% 255\n", dump[i].c_str(), first[i].size(),
                        100.0 * zero / first[i].size(), 100.0 * ff / first[i].size());
        }
        if (probe)
            for (auto& [name, k] : plan.kernels()) {
                auto g = k->module.globals.find("__m4r_probe");
                if (g == k->module.globals.end() || !k->data.map) continue;
                const char* p = static_cast<const char*>(k->data.map) + (g->second.va - k->module.dataLo);
                std::ofstream(dir + "/probe-" + k->name + ".bin", std::ios::binary).write(p, std::streamsize(g->second.size));
                std::printf("probe of %s written\n", k->name.c_str());
            }
        vkDestroyQueryPool(vk.device, qp, nullptr);
    }
    rt.destroy();
    vk.destroy();
    return rc;
}
