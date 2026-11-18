// Replays an OptiScaler frame generation run through m4r's C API on a native Vulkan device, as the reference for the same
// frames and parameters: the backbuffer images are the upscaled frames (out-NN.rgba16f of the harness), depth and motion
// vectors come from the fixture, and the per-frame parameters from the m4r.log the run wrote (M4R_LOG).
// Writes <result dir>/ref-gen-NN.rgba16f (the generated frame, when one was made) and ref-real-NN.rgba16f (what
// m4r_framegen_restore puts back).
// usage: ref_framegen <data dir (holds fg-<w>x<h>-<w>x<h>)> <run out dir> <fixture dir> <m4r.log> <frames> <result dir>
// SPDX-License-Identifier: GPL-3.0-only
#include "vk.h"
#include "m4r.h"
#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace m4r;

static std::vector<uint8_t> readAll(const std::string& p) {
    std::ifstream f(p, std::ios::binary);
    return std::vector<uint8_t>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

static double halton(int i, int b) {
    double v = 0, s = 1;
    while (i) { s /= b; v += (i % b) * s; i /= b; }
    return v - 0.5;
}

struct Line { unsigned bw, bh, dw, dh; float msx, msy, dt, nearP, farP; int rev, reset, generated; };

int main(int argc, char** argv) {
    if (argc < 7) { std::fprintf(stderr, "usage: %s <data dir> <run out dir> <fixture dir> <m4r.log> <frames> <result dir>\n", argv[0]); return 2; }
    const std::string dataDir = argv[1], runDir = argv[2], fx = argv[3], resDir = argv[6];
    const int frames = std::atoi(argv[5]);
    std::vector<Line> lines;
    {
        std::ifstream f(argv[4]);
        std::string s;
        while (std::getline(f, s)) {
            Line l{};
            if (std::sscanf(s.c_str(), "capture bb=%ux%u depth=%ux%u mvscale=%f,%f dt=%f near=%f far=%f rev=%d reset=%d generated=%d",
                            &l.bw, &l.bh, &l.dw, &l.dh, &l.msx, &l.msy, &l.dt, &l.nearP, &l.farP, &l.rev, &l.reset, &l.generated) == 12)
                lines.push_back(l);
        }
    }
    if (lines.empty()) { std::fprintf(stderr, "no capture lines in %s\n", argv[4]); return 1; }
    const int first = frames - int(lines.size());
    Context vk;
    if (!vk.create()) { std::fprintf(stderr, "vulkan: %s\n", vk.error.c_str()); return 1; }
    if (std::getenv("REF_JUNK")) {
        // fill free video memory with junk and release it, so later allocations that are read before written show it
        for (int i = 0; i < 12; ++i) {
            Texture junk;
            if (!vk.createTexture(junk, VK_FORMAT_R16G16B16A16_SFLOAT, 8192, 8192)) break;
            VkCommandBuffer jb = vk.begin();
            vk.toGeneral(jb, junk);
            VkClearColorValue v{};
            v.float32[0] = v.float32[1] = v.float32[2] = v.float32[3] = 12345.0f + float(i);
            VkImageSubresourceRange rg{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            vkCmdClearColorImage(jb, junk.image, VK_IMAGE_LAYOUT_GENERAL, &v, 1, &rg);
            vk.submit(jb);
            vk.destroyTexture(junk);
        }
    }
    auto gipa = reinterpret_cast<PFN_vkGetInstanceProcAddr>(dlsym(RTLD_DEFAULT, "vkGetInstanceProcAddr"));
    m4r_context* ctx = nullptr;
    if (!gipa || m4r_create(&ctx, gipa, vk.instance, vk.physical, vk.device, vk.queueFamily, dataDir.c_str()) != 0) {
        std::fprintf(stderr, "m4r_create: %s\n", ctx ? m4r_last_error(ctx) : "no vkGetInstanceProcAddr");
        return 1;
    }
    const uint32_t bw = lines[0].bw, bh = lines[0].bh, dw = lines[0].dw, dh = lines[0].dh;
    Texture bb, restored, depth, motion;
    vk.createTexture(bb, VK_FORMAT_R16G16B16A16_SFLOAT, bw, bh);
    vk.createTexture(restored, VK_FORMAT_R16G16B16A16_SFLOAT, bw, bh);
    vk.createTexture(depth, VK_FORMAT_R32_SFLOAT, dw, dh);
    vk.createTexture(motion, VK_FORMAT_R16G16_SFLOAT, dw, dh);
    const auto image = [](const Texture& t) { return m4r_image{t.image, t.format, t.width, t.height, VK_IMAGE_LAYOUT_GENERAL}; };
    std::vector<uint8_t> px(size_t(bw) * bh * 8);
    // REF_UPSCALE=1: run the upscaler (plan bbr-<w>x<h>-<w>x<h> in the data dir) on every frame first, as the harness does, so the
    // frame generation kernels run after the upscaler's kernels like in the OptiScaler run
    const bool upscale = std::getenv("REF_UPSCALE") != nullptr;
    const uint32_t iw = dw, ih = dh;
    Texture ucolor, udepth, umotion, uout;
    if (upscale) {
        vk.createTexture(ucolor, VK_FORMAT_R16G16B16A16_SFLOAT, iw, ih);
        vk.createTexture(udepth, VK_FORMAT_R32_SFLOAT, iw, ih);
        vk.createTexture(umotion, VK_FORMAT_R16G16_SFLOAT, iw, ih);
        vk.createTexture(uout, VK_FORMAT_R16G16B16A16_SFLOAT, bw, bh);
        if (m4r_upscale_init(ctx, bw, bh, iw, ih) != 0) { std::fprintf(stderr, "upscale init: %s\n", m4r_last_error(ctx)); return 1; }
    }
    int rc = 0;
    for (int t = upscale ? 0 : first; t < frames && rc == 0; ++t) {
        char nm[64];
        if (upscale) {
            std::snprintf(nm, sizeof nm, "/input-%02d.rgba16f", t); auto uc = readAll(fx + nm);
            std::snprintf(nm, sizeof nm, "/depth-%02d.r32f", t); auto ud = readAll(fx + nm);
            std::snprintf(nm, sizeof nm, "/mvlo-%02d.rg16f", t); auto um = readAll(fx + nm);
            vk.upload(ucolor, uc.data(), uc.size()); vk.upload(udepth, ud.data(), ud.size()); vk.upload(umotion, um.data(), um.size());
            m4r_upscale_desc ud2{};
            ud2.color = image(ucolor); ud2.depth = image(udepth); ud2.motion = image(umotion); ud2.output = image(uout);
            ud2.content_width = iw; ud2.content_height = ih;
            ud2.jitter_x = float(halton(t + 1, 2)); ud2.jitter_y = float(halton(t + 1, 3));
            ud2.motion_scale_x = ud2.motion_scale_y = 1.0f; ud2.pre_exposure = 1.0f; ud2.reset = t == 0; ud2.depth_reversed = 1;   // the fixtures use reversed depth
            VkCommandBuffer ucb = vk.begin();
            if (m4r_upscale(ctx, ucb, &ud2) != 0 || !vk.submit(ucb)) { std::fprintf(stderr, "frame %d: upscale: %s\n", t, m4r_last_error(ctx)); rc = 1; break; }
        }
        if (t < first) continue;
        const Line& l = lines[size_t(t - first)];
        std::snprintf(nm, sizeof nm, "/out-%02d.rgba16f", t); auto c = readAll(runDir + nm);
        std::snprintf(nm, sizeof nm, "/depth-%02d.r32f", t); auto d = readAll(fx + nm);
        std::snprintf(nm, sizeof nm, "/mvlo-%02d.rg16f", t); auto m = readAll(fx + nm);
        if (c.size() != px.size() || d.size() != size_t(dw) * dh * 4 || m.size() != size_t(dw) * dh * 4) {
            std::fprintf(stderr, "frame %d: input size mismatch\n", t); rc = 1; break;
        }
        vk.upload(bb, c.data(), c.size()); vk.upload(depth, d.data(), d.size()); vk.upload(motion, m.data(), m.size());
        m4r_framegen_desc fd{};
        fd.backbuffer = image(bb); fd.depth = image(depth); fd.motion = image(motion);
        fd.motion_scale_x = l.msx; fd.motion_scale_y = l.msy; fd.delta_time = l.dt;
        fd.near_plane = l.nearP; fd.far_plane = l.farP; fd.depth_reversed = l.rev; fd.reset = l.reset;
        VkCommandBuffer cb = vk.begin();
        const int r = m4r_framegen_capture(ctx, cb, &fd);
        if (r < 0 || !vk.submit(cb)) { std::fprintf(stderr, "frame %d: capture: %s\n", t, r < 0 ? m4r_last_error(ctx) : vk.error.c_str()); rc = 1; break; }
        if (r != l.generated) { std::fprintf(stderr, "frame %d: generated %d, the run had %d\n", t, r, l.generated); rc = 1; }
        if (r == 1) {
            vk.download(bb, px.data(), px.size());
            std::snprintf(nm, sizeof nm, "/ref-gen-%02d.rgba16f", t);
            std::ofstream(resDir + nm, std::ios::binary).write(reinterpret_cast<char*>(px.data()), std::streamsize(px.size()));
        }
        cb = vk.begin();
        const m4r_image rimg = image(restored);
        if (m4r_framegen_restore(ctx, cb, &rimg) != 0 || !vk.submit(cb)) { std::fprintf(stderr, "frame %d: restore: %s\n", t, m4r_last_error(ctx)); rc = 1; break; }
        vk.download(restored, px.data(), px.size());
        std::snprintf(nm, sizeof nm, "/ref-real-%02d.rgba16f", t);
        std::ofstream(resDir + nm, std::ios::binary).write(reinterpret_cast<char*>(px.data()), std::streamsize(px.size()));
        std::printf("frame %d: capture -> %d\n", t, r);
    }
    m4r_destroy(ctx);
    for (Texture* x : {&bb, &restored, &depth, &motion}) vk.destroyTexture(*x);
    if (upscale) for (Texture* x : {&ucolor, &udepth, &umotion, &uout}) vk.destroyTexture(*x);
    vk.destroy();
    return rc;
}
