// Drives m4r_denoise (the C API OptiScaler calls) with captured Ray Reconstruction frames (M4R_DUMP_RR).
// usage: run_denoise_api <data dir> <capture dir> <first frame id> <frames> <out dir>
// <data dir> holds dn-<w>x<h> (tools/prepare_denoiser.py --runtime) and bbr-<out>-<in> plans. Writes <out dir>/denoised-NN.raw
// (RGBA16F, render size) and <out dir>/output-NN.raw (RGBA16F, output size).
// SPDX-License-Identifier: GPL-3.0-only
#include "m4r.h"
#include "plan.h"
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <iterator>
#include <sstream>

using namespace m4r;

static std::vector<uint8_t> readAll(const std::string& p) {
    std::ifstream f(p, std::ios::binary);
    return std::vector<uint8_t>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

int main(int argc, char** argv) {
    if (argc < 6) { std::fprintf(stderr, "usage: %s <data dir> <capture dir> <first frame id> <frames> <out dir>\n", argv[0]); return 2; }
    const std::string data = argv[1], cap = argv[2], out = argv[5];
    const int first = std::stoi(argv[3]), frames = std::stoi(argv[4]);
    Context vk;
    if (!vk.create()) { std::fprintf(stderr, "vulkan: %s\n", vk.error.c_str()); return 1; }
    m4r_context* ctx = nullptr;
    if (m4r_create(&ctx, nullptr, vk.instance, vk.physical, vk.device, vk.queueFamily, data.c_str()) != 0) {
        std::fprintf(stderr, "m4r_create: %s\n", m4r_last_error(ctx)); return 1;
    }
    auto meta = [&](int id) {
        std::map<std::string, std::vector<double>> m;
        std::ifstream f(cap + "/rr-" + std::to_string(id) + ".txt");
        std::string line;
        while (std::getline(f, line)) {
            std::istringstream ls(line); std::string k; ls >> k;
            if (k == "image") continue;
            double v; while (ls >> v) m[k].push_back(v);
        }
        return m;
    };
    auto m0 = meta(first);
    const uint32_t ow = uint32_t(m0["output"][0]), oh = uint32_t(m0["output"][1]), iw = uint32_t(m0["content"][0]), ih = uint32_t(m0["content"][1]);
    if (m4r_denoise_init(ctx, ow, oh, iw, ih) != 0) { std::fprintf(stderr, "m4r_denoise_init: %s\n", m4r_last_error(ctx)); return 1; }
    struct In { const char* file; VkFormat fmt; int bpp; };
    const In ins[] = {{"color", VK_FORMAT_R16G16B16A16_SFLOAT, 8}, {"depth", VK_FORMAT_R32_SFLOAT, 4}, {"motion", VK_FORMAT_R16G16B16A16_SFLOAT, 8},
                      {"normals", VK_FORMAT_R16G16B16A16_SFLOAT, 8}, {"diffuse_albedo", VK_FORMAT_R8G8B8A8_UNORM, 4}, {"specular_albedo", VK_FORMAT_R8G8B8A8_UNORM, 4}};
    Texture tex[6], output;
    for (int i = 0; i < 6; ++i) vk.createTexture(tex[i], ins[i].fmt, iw, ih);
    vk.createTexture(output, VK_FORMAT_R16G16B16A16_SFLOAT, ow, oh);
    { VkCommandBuffer cb = vk.begin(); vk.toGeneral(cb, output); vk.submit(cb); }
    int rc = 0;
    for (int t = 0; t < frames && !rc; ++t) {
        const int id = first + t;
        for (int i = 0; i < 6; ++i) {
            auto d = readAll(cap + "/rr-" + std::to_string(id) + "-" + ins[i].file + ".raw");
            if (d.size() != size_t(iw) * ih * ins[i].bpp) { std::fprintf(stderr, "frame %d: %s size\n", id, ins[i].file); return 1; }
            vk.upload(tex[i], d.data(), d.size());
        }
        auto m = meta(id);
        m4r_denoise_desc d{};
        auto im = [&](Texture& x) { return m4r_image{x.image, x.format, x.width, x.height, VK_IMAGE_LAYOUT_GENERAL}; };
        d.color = im(tex[0]); d.depth = im(tex[1]); d.motion = im(tex[2]); d.normals = im(tex[3]); d.diffuse_albedo = im(tex[4]);
        d.specular_albedo = im(tex[5]); d.output = im(output);
        for (int i = 0; i < 16; ++i) { d.world_to_view[i] = float(m["world_to_view"][i]); d.view_to_clip[i] = float(m["view_to_clip"][i]); }
        d.content_width = iw; d.content_height = ih;
        d.jitter_x = float(m["jitter"][0]); d.jitter_y = float(m["jitter"][1]);
        d.motion_scale_x = float(m["mvscale"][0]); d.motion_scale_y = float(m["mvscale"][1]);
        d.pre_exposure = float(m["preexposure"][0]); d.reset = int(m["reset"][0]);
        d.depth_reversed = int(m["depth_reversed"][0]); d.hardware_depth = 1; d.roughness_in_normals_w = 1;
        d.flip_upscale_jitter = std::getenv("M4R_NO_FLIP") ? 0 : 1;
        VkCommandBuffer cb = vk.begin();
        if (m4r_denoise(ctx, cb, &d) != 0) { std::fprintf(stderr, "m4r_denoise: %s\n", m4r_last_error(ctx)); rc = 1; break; }
        if (!vk.submit(cb)) { std::fprintf(stderr, "submit: %s\n", vk.error.c_str()); rc = 1; break; }
        auto dump = [&](Texture& x, const char* name) {
            std::vector<uint8_t> b(size_t(x.width) * x.height * 8);
            vk.download(x, b.data(), b.size());
            char nm[256]; std::snprintf(nm, sizeof nm, "%s/%s-%02d.raw", out.c_str(), name, t);
            std::ofstream(nm, std::ios::binary).write(reinterpret_cast<char*>(b.data()), std::streamsize(b.size()));
        };
        Texture den; vk.wrap(den, m4r_denoise_texture(ctx, "denoised"), VK_FORMAT_R16G16B16A16_SFLOAT, iw, ih);
        dump(den, "denoised"); dump(output, "output");
        std::printf("frame %d done\n", t);
    }
    m4r_destroy(ctx);
    vk.destroy();
    return rc;
}
