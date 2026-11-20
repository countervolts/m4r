// Runs a denoiser plan (tools/prepare_denoiser.py) on captured Ray Reconstruction frames (M4R_DUMP_RR).
// usage: run_denoise <plan dir> <capture dir> <first frame id> <frames>
// Textures the plan finds by name: color depth motion normals diffuse specular spechit roughness exposure (uploaded each frame from
// rr-<id>-*.raw), output. Per-frame buffers: <plan dir>/frames/NN/<buffer>.bin (written by prepare_denoiser.py --frames) and
// frames/NN/flags.txt (one flag name per line, set for the frame). M4R_DUMP_TEX=name,... writes tex-<name>-NN.raw after each frame.
// SPDX-License-Identifier: GPL-3.0-only
#include "plan.h"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <sstream>

using namespace m4r;

static std::vector<uint8_t> readAll(const std::string& p) {
    std::ifstream f(p, std::ios::binary);
    return std::vector<uint8_t>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}
static uint16_t f2h(float f) {
    uint32_t x; std::memcpy(&x, &f, 4);
    uint32_t s = (x >> 16) & 0x8000, e = (x >> 23) & 0xff, m = x & 0x7fffff;
    if (e == 255) return s | 0x7c00 | (m ? 0x200 : 0);
    int ne = int(e) - 127 + 15;
    if (ne >= 31) return s | 0x7c00;
    if (ne <= 0) { if (ne < -10) return s; m |= 0x800000; int sh = 14 - ne; uint32_t r = m >> sh; if ((m >> (sh - 1)) & 1) ++r; return s | r; }
    uint32_t r = s | (ne << 10) | (m >> 13);
    if (m & 0x1000) ++r;
    return r;
}
static float h2f(uint16_t h) {
    uint32_t s = (h & 0x8000) << 16, e = (h >> 10) & 31, m = h & 0x3ff, x;
    if (e == 0) { if (!m) x = s; else { e = 1; while (!(m & 0x400)) { m <<= 1; --e; } m &= 0x3ff; x = s | ((e + 127 - 15) << 23) | (m << 13); } }
    else if (e == 31) x = s | 0x7f800000 | (m << 13);
    else x = s | ((e + 127 - 15) << 23) | (m << 13);
    float f; std::memcpy(&f, &x, 4); return f;
}

int main(int argc, char** argv) {
    if (argc < 5) { std::fprintf(stderr, "usage: %s <plan dir> <capture dir> <first frame id> <frames>\n", argv[0]); return 2; }
    const std::string dir = argv[1], cap = argv[2];
    const int first = std::stoi(argv[3]), frames = std::stoi(argv[4]);
    Context vk;
    if (!vk.create()) { std::fprintf(stderr, "vulkan: %s\n", vk.error.c_str()); return 1; }
    Runtime rt(vk);
    if (!rt.init()) { std::fprintf(stderr, "runtime: %s\n", rt.error.c_str()); return 1; }
    int rc = 0;
    {
        Plan plan(rt);
        if (!plan.load(dir)) { std::fprintf(stderr, "plan: %s\n", plan.error.c_str()); return 1; }
        // sizes from the first frame's text
        unsigned rw = 0, rh = 0, ow = 0, oh = 0;
        {
            std::ifstream f(cap + "/rr-" + std::to_string(first) + ".txt");
            std::string k;
            while (f >> k) {
                if (k == "content") f >> rw >> rh;
                else if (k == "output") f >> ow >> oh;
                else { std::string rest; std::getline(f, rest); }
            }
        }
        struct In { const char* name; const char* file; VkFormat fmt; int bpp; };
        const In ins[] = {{"color", "color", VK_FORMAT_R16G16B16A16_SFLOAT, 8}, {"depth", "depth", VK_FORMAT_R32_SFLOAT, 4},
                          {"motion", "motion", VK_FORMAT_R16G16B16A16_SFLOAT, 8}, {"normals", "normals", VK_FORMAT_R16G16B16A16_SFLOAT, 8},
                          {"diffuse", "diffuse_albedo", VK_FORMAT_R8G8B8A8_UNORM, 4}, {"specular", "specular_albedo", VK_FORMAT_R8G8B8A8_UNORM, 4},
                          {"spechit", "specular_hit", VK_FORMAT_R32_SFLOAT, 4}, {"beforeparticles", "before_particles", VK_FORMAT_R16G16B16A16_SFLOAT, 8}};
        std::vector<Texture> tex(sizeof ins / sizeof *ins);
        for (size_t i = 0; i < tex.size(); ++i) { vk.createTexture(tex[i], ins[i].fmt, rw, rh); plan.bindTexture(ins[i].name, &tex[i]); }
        Texture rough, exposure;
        vk.createTexture(rough, VK_FORMAT_R16_SFLOAT, rw, rh); plan.bindTexture("roughness", &rough);
        vk.createTexture(exposure, VK_FORMAT_R32_SFLOAT, 1, 1); plan.bindTexture("exposure", &exposure);
        const float one = 1.0f; vk.upload(exposure, &one, 4);
        VkQueryPool qp;
        { VkQueryPoolCreateInfo qci{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO}; qci.queryType = VK_QUERY_TYPE_TIMESTAMP; qci.queryCount = 2; vkCreateQueryPool(vk.device, &qci, nullptr, &qp); }
        VkPhysicalDeviceProperties props; vkGetPhysicalDeviceProperties(vk.physical, &props);
        std::vector<double> ptotal(plan.passes(), 0.0);
        for (int t = 0; t < frames; ++t) {
            const std::string id = std::to_string(first + t);
            for (size_t i = 0; i < tex.size(); ++i) {
                auto d = readAll(cap + "/rr-" + id + "-" + ins[i].file + ".raw");
                if (d.size() != size_t(rw) * rh * ins[i].bpp) { std::fprintf(stderr, "frame %s: %s size %zu\n", id.c_str(), ins[i].file, d.size()); rc = 1; goto done; }
                vk.upload(tex[i], d.data(), d.size());
                if (std::string(ins[i].name) == "normals") {   // roughness is the normals' alpha
                    std::vector<uint16_t> r(size_t(rw) * rh);
                    for (size_t p = 0; p < r.size(); ++p) std::memcpy(&r[p], d.data() + p * 8 + 6, 2);
                    vk.upload(rough, r.data(), r.size() * 2);
                }
            }
            {   // per-frame buffers and flags
                char nm[64]; std::snprintf(nm, sizeof nm, "/frames/%02d", t);
                for (auto& [name, b] : plan.buffers()) {
                    auto d = readAll(dir + nm + "/" + name + ".bin");
                    if (!d.empty()) std::memcpy(plan.buffer(name)->map, d.data(), std::min<size_t>(d.size(), size_t(b.size)));
                }
                std::ifstream fl(dir + nm + "/flags.txt");
                std::string f;
                while (fl >> f) plan.setFlag(f, true);
                plan.setFlag("autoexp", std::getenv("M4R_AUTOEXPOSURE") != nullptr);
            }
            plan.setParity(t);
            const auto t0 = std::chrono::steady_clock::now();
            rt.resetTransient();
            if (std::getenv("M4R_PASS_TIME")) {   // GPU time of every pass (one submission each), averaged over frames >= 1
                for (size_t i = 0; i < plan.passes(); ++i) {
                    VkCommandBuffer pcb = vk.begin();
                    vkCmdResetQueryPool(pcb, qp, 0, 2);
                    vkCmdWriteTimestamp(pcb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, qp, 0);
                    if (!plan.record(pcb, i, 1)) { std::fprintf(stderr, "record: %s\n", plan.error.c_str()); rc = 1; goto done; }
                    vk.barrier(pcb);
                    vkCmdWriteTimestamp(pcb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, qp, 1);
                    if (!vk.submit(pcb)) { std::fprintf(stderr, "submit failed in pass %zu (%s): %s\n", i, plan.passLabel(i).c_str(), vk.error.c_str()); rc = 1; goto done; }
                    uint64_t ts[2];
                    vkGetQueryPoolResults(vk.device, qp, 0, 2, sizeof ts, ts, 8, VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
                    if (t > 0) ptotal[i] += double(ts[1] - ts[0]) * props.limits.timestampPeriod * 1e-6;
                }
                if (t == frames - 1) {
                    double sum = 0;
                    for (size_t i = 0; i < plan.passes(); ++i) { std::printf("pass %2zu %-56s %7.3f ms\n", i, plan.passLabel(i).c_str(), ptotal[i] / std::max(1, frames - 1)); sum += ptotal[i] / std::max(1, frames - 1); }
                    std::printf("sum %.3f ms\n", sum);
                }
                continue;
            }
            VkCommandBuffer cb = vk.begin();
            if (!plan.record(cb)) { std::fprintf(stderr, "record: %s\n", plan.error.c_str()); rc = 1; break; }
            vk.barrier(cb);
            if (!vk.submit(cb)) { std::fprintf(stderr, "submit: %s\n", vk.error.c_str()); rc = 1; break; }
            if (Texture* dn = plan.texture("denoised"); dn && plan.texture("v3hist") && !plan.texture("v3out")) {   // stand-in history for the V3 experiment
                std::vector<uint8_t> tmp(size_t(dn->width) * dn->height * 8);
                vk.download(*dn, tmp.data(), tmp.size());
                vk.upload(*plan.texture("v3hist"), tmp.data(), tmp.size());
            }
            if (t == 0) if (Texture* c = plan.texture("v3const"); c && std::getenv("M4R_V3CONST")) {   // experiment: constant RGBA texture, M4R_V3CONST="r,g,b,a"
                float v[4] = {0, 0, 0, 0};
                std::sscanf(std::getenv("M4R_V3CONST"), "%f,%f,%f,%f", &v[0], &v[1], &v[2], &v[3]);
                std::vector<uint16_t> px(size_t(c->width) * c->height * 4);
                for (size_t i = 0; i < px.size(); ++i) px[i] = f2h(v[i & 3]);
                vk.upload(*c, px.data(), px.size() * 2);
            }
            std::printf("frame %d done %.2f ms\n", t, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
            if (const char* names = std::getenv("M4R_DUMP_TEX")) {
                std::string all = names;
                for (size_t a = 0; a < all.size();) {
                    size_t b = all.find(',', a); if (b == std::string::npos) b = all.size();
                    std::string n = all.substr(a, b - a); a = b + 1;
                    std::vector<uint8_t> bytes;
                    if (Texture* x = plan.texture(n)) {
                        size_t bpp = 0;
                        switch (x->format) {
                        case VK_FORMAT_R8_UNORM: bpp = 1; break;
                        case VK_FORMAT_R8G8_UNORM: case VK_FORMAT_R16_SFLOAT: bpp = 2; break;
                        case VK_FORMAT_R16G16_SFLOAT: case VK_FORMAT_R32_SFLOAT: case VK_FORMAT_R8G8B8A8_UNORM: bpp = 4; break;
                        case VK_FORMAT_R16G16B16A16_SFLOAT: bpp = 8; break;
                        default: break;
                        }
                        if (!bpp) { std::fprintf(stderr, "dump %s: format %d\n", n.c_str(), int(x->format)); continue; }
                        bytes.resize(size_t(x->width) * x->height * bpp);
                        vk.download(*x, bytes.data(), bytes.size());
                    } else if (Buffer* y = plan.buffer(n)) {
                        bytes.assign(static_cast<uint8_t*>(y->map), static_cast<uint8_t*>(y->map) + y->size);
                    } else { std::fprintf(stderr, "dump: no %s\n", n.c_str()); continue; }
                    char nm[256]; std::snprintf(nm, sizeof nm, "/tex-%s-%02d.raw", n.c_str(), t);
                    std::ofstream(dir + nm, std::ios::binary).write(reinterpret_cast<char*>(bytes.data()), std::streamsize(bytes.size()));
                }
            }
        }
    done:
        vkDestroyQueryPool(vk.device, qp, nullptr);
        for (auto& x : tex) vk.destroyTexture(x);
        vk.destroyTexture(rough); vk.destroyTexture(exposure);
    }
    rt.destroy();
    vk.destroy();
    return rc;
}
