// Drives m4r_denoise (the C API OptiScaler calls) with captured Ray Reconstruction frames (M4R_DUMP_RR).
// usage: run_denoise_api <data dir> <capture dir> <first frame id> <frames> <out dir>
// <data dir> holds dn-<w>x<h> (tools/prepare_denoiser.py --runtime) and bbr-<out>-<in> plans. Writes <out dir>/denoised-NN.raw
// (RGBA16F, render size) and <out dir>/output-NN.raw (RGBA16F, output size).
// SPDX-License-Identifier: GPL-3.0-only
#include "m4r.h"
#include "plan.h"
#include <algorithm>
#include <cmath>
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
    // experiments: M4R_UPSCALE_ONLY=1 runs only MetalFX's upscaler (BBR) on the capture's colour (e.g. a colour already denoised
    // elsewhere); M4R_ZERO_JITTER=1 tells it the colour has no jitter (an input that was already resolved onto the pixel grid)
    const bool upscaleOnly = std::getenv("M4R_UPSCALE_ONLY") != nullptr, zeroJitter = std::getenv("M4R_ZERO_JITTER") != nullptr;
    if (upscaleOnly && m4r_upscale_init(ctx, ow, oh, iw, ih) != 0) { std::fprintf(stderr, "m4r_upscale_init: %s\n", m4r_last_error(ctx)); return 1; }
    struct In { const char* file; VkFormat fmt; int bpp; };
    In ins[] = {{"color", VK_FORMAT_R16G16B16A16_SFLOAT, 8}, {"depth", VK_FORMAT_R32_SFLOAT, 4}, {"motion", VK_FORMAT_R16G16B16A16_SFLOAT, 8},
                      {"normals", VK_FORMAT_R16G16B16A16_SFLOAT, 8}, {"diffuse_albedo", VK_FORMAT_R8G8B8A8_UNORM, 4}, {"specular_albedo", VK_FORMAT_R8G8B8A8_UNORM, 4}};
    // formats of the capture's images (rr-N.txt "image <name> format <VkFormat> size W H bpp B"): other games give other formats
    std::map<std::string, std::pair<int, int>> fmts;
    { std::ifstream f(cap + "/rr-" + std::to_string(first) + ".txt"); std::string line;
      while (std::getline(f, line)) { std::istringstream ls(line); std::string k, n, a, b, c2, d2; int fm = 0, w = 0, h = 0, bp = 0;
          if (ls >> k >> n >> a >> fm >> b >> w >> h >> c2 >> bp && k == "image") fmts[n] = {fm, bp}; } }
    for (auto& in : ins) if (fmts.count(in.file)) { in.fmt = VkFormat(fmts[in.file].first); in.bpp = fmts[in.file].second; }
    // M4R_LINEAR_DEPTH=1: gives the depth the way Alan Wake 2 does (DLSS.Use.HW.Depth = 0, linear view depth in R16F, sky = 65504)
    const bool linDepth = std::getenv("M4R_LINEAR_DEPTH") != nullptr;
    Texture tex[6], output;
    for (int i = 0; i < 6; ++i) vk.createTexture(tex[i], i == 1 && linDepth ? VK_FORMAT_R16_SFLOAT : ins[i].fmt, iw, ih);
    // M4R_K_OUT=1: the caller upscales (denoised_out / depth_out given, no BBR): dumps kout-colour-NN.raw (RGBA16F) and kout-depth-NN.raw (R32F)
    const bool kOutTest = std::getenv("M4R_K_OUT") != nullptr;
    Texture kColour, kDepth;
    if (kOutTest) { vk.createTexture(kColour, VK_FORMAT_R16G16B16A16_SFLOAT, iw, ih); vk.createTexture(kDepth, VK_FORMAT_R32_SFLOAT, iw, ih); }
    Texture exposureTex, specHitTex;
    const bool haveHit = fmts.count("specular_hit") && !std::getenv("M4R_NO_SPECHIT_INPUT");
    if (haveHit) vk.createTexture(specHitTex, VkFormat(fmts["specular_hit"].first), iw, ih);
    const bool haveExposure = fmts.count("exposure") && !std::getenv("M4R_NO_EXPOSE_INPUT");
    if (haveExposure) vk.createTexture(exposureTex, VkFormat(fmts["exposure"].first), 1, 1);
    vk.createTexture(output, VK_FORMAT_R16G16B16A16_SFLOAT, ow, oh);
    { VkCommandBuffer cb = vk.begin(); vk.toGeneral(cb, output); vk.submit(cb); }
    int rc = 0;
    for (int t = 0; t < frames && !rc; ++t) {
        const int id = first + t;
        for (int i = 0; i < 6; ++i) {
            auto d = readAll(cap + "/rr-" + std::to_string(id) + "-" + ins[i].file + ".raw");
            if (d.size() != size_t(iw) * ih * ins[i].bpp) { std::fprintf(stderr, "frame %d: %s size\n", id, ins[i].file); return 1; }
            if (i == 1 && linDepth) {   // reversed hardware depth of the capture -> linear view depth (its projection: z = [3][2] / (d - [2][2]))
                auto mm = meta(id);
                const float m22 = float(mm["view_to_clip"][10]), m32 = float(mm["view_to_clip"][14]);
                std::vector<_Float16> h(size_t(iw) * ih);
                const float* f = reinterpret_cast<const float*>(d.data());
                for (size_t k = 0; k < h.size(); ++k) h[k] = _Float16(f[k] <= 0.0f ? 65504.0f : std::min(65504.0f, m32 / (f[k] - m22)));
                vk.upload(tex[i], h.data(), h.size() * 2);
                continue;
            }
            if (i == 3 && std::getenv("M4R_ROUGH_POW")) {   // experiment: roughness (normals.w) ^ p
                const float pw = std::strtof(std::getenv("M4R_ROUGH_POW"), nullptr);
                _Float16* h = reinterpret_cast<_Float16*>(d.data());
                for (size_t k = 0; k + 3 < d.size() / 2; k += 4) h[k + 3] = _Float16(std::pow(std::clamp(float(h[k + 3]), 0.0f, 1.0f), pw));
            }
            vk.upload(tex[i], d.data(), d.size());
        }
        auto m = meta(id);
        m4r_denoise_desc d{};
        if (haveHit) {
            auto hd = readAll(cap + "/rr-" + std::to_string(id) + "-specular_hit.raw");
            if (const char* sc = std::getenv("M4R_HIT_SCALE")) {   // experiment: rescale the hit distance (R16F data)
                const float k = std::strtof(sc, nullptr);
                _Float16* h = reinterpret_cast<_Float16*>(hd.data());
                for (size_t i = 0; i < hd.size() / 2; ++i) h[i] = _Float16(std::min(65504.0f, float(h[i]) * k));
            }
            vk.upload(specHitTex, hd.data(), hd.size());
        }
        if (haveExposure) {
            auto e = readAll(cap + "/rr-" + std::to_string(id) + "-exposure.raw");
            if (const char* mul = std::getenv("M4R_EXPOSURE_MUL")) {   // experiment: scale the exposure the plan sees (R32G32F)
                float* ef = reinterpret_cast<float*>(e.data());
                ef[0] *= std::strtof(mul, nullptr);
            }
            vk.upload(exposureTex, e.data(), e.size());
        }
        auto im = [&](Texture& x) { return m4r_image{x.image, x.format, x.width, x.height, VK_IMAGE_LAYOUT_GENERAL}; };
        d.color = im(tex[0]); d.depth = im(tex[1]); d.motion = im(tex[2]); d.normals = im(tex[3]); d.diffuse_albedo = im(tex[4]);
        d.specular_albedo = im(tex[5]); d.output = im(output);
        if (haveExposure) d.exposure = im(exposureTex);
        if (haveHit) d.specular_hit = im(specHitTex);
        if (kOutTest) { d.denoised_out = im(kColour); d.depth_out = im(kDepth); }
        for (int i = 0; i < 16; ++i) { d.world_to_view[i] = float(m["world_to_view"][i]); d.view_to_clip[i] = float(m["view_to_clip"][i]); }
        d.content_width = iw; d.content_height = ih;
        d.jitter_x = float(m["jitter"][0]); d.jitter_y = float(m["jitter"][1]);
        d.motion_scale_x = float(m["mvscale"][0]); d.motion_scale_y = float(m["mvscale"][1]);
        d.pre_exposure = float(m["preexposure"][0]); d.reset = int(m["reset"][0]);
        d.depth_reversed = int(m["depth_reversed"][0]); d.hardware_depth = (linDepth && std::getenv("M4R_LINEAR_DEPTH")[0] != '2') ? 0 : (m.count("hardware_depth") ? int(m["hardware_depth"][0]) : 1); d.roughness_in_normals_w = 1;   // M4R_LINEAR_DEPTH=2: linear data declared as hardware depth (the old behaviour, must look wrong)
        if (linDepth) d.depth_reversed = 0;   // the game's own flag means nothing for linear depth
        d.flip_upscale_jitter = std::getenv("M4R_NO_FLIP") ? 0 : 1;
        VkCommandBuffer cb = vk.begin();
        if (zeroJitter) { d.jitter_x = 0.0f; d.jitter_y = 0.0f; }
        if (upscaleOnly) {
            m4r_upscale_desc u{};
            u.color = d.color; u.depth = d.depth; u.motion = d.motion; u.output = d.output;
            u.content_width = iw; u.content_height = ih;
            const float fj = d.flip_upscale_jitter ? -1.0f : 1.0f;
            u.jitter_x = fj * d.jitter_x; u.jitter_y = fj * d.jitter_y;
            u.motion_scale_x = d.motion_scale_x; u.motion_scale_y = d.motion_scale_y;
            u.pre_exposure = d.pre_exposure; u.reset = d.reset; u.depth_reversed = d.depth_reversed;
            if (m4r_upscale(ctx, cb, &u) != 0) { std::fprintf(stderr, "m4r_upscale: %s\n", m4r_last_error(ctx)); rc = 1; break; }
        } else if (m4r_denoise(ctx, cb, &d) != 0) { std::fprintf(stderr, "m4r_denoise: %s\n", m4r_last_error(ctx)); rc = 1; break; }
        if (!vk.submit(cb)) { std::fprintf(stderr, "submit: %s\n", vk.error.c_str()); rc = 1; break; }
        auto dump = [&](Texture& x, const char* name) {
            std::vector<uint8_t> b(size_t(x.width) * x.height * 8);
            vk.download(x, b.data(), b.size());
            char nm[256]; std::snprintf(nm, sizeof nm, "%s/%s-%02d.raw", out.c_str(), name, t);
            std::ofstream(nm, std::ios::binary).write(reinterpret_cast<char*>(b.data()), std::streamsize(b.size()));
        };
        if (!upscaleOnly) {
            Texture den; vk.wrap(den, m4r_denoise_texture(ctx, "denoised"), VK_FORMAT_R16G16B16A16_SFLOAT, iw, ih);
            dump(den, "denoised");
        }
        if (const char* names = std::getenv("M4R_DUMP_TEX")) {   // plan textures by name: M4R_DUMP_TEX=shin,mm0,... (R8, R16F, RG16F, RGBA16F ones)
            std::string all = names; size_t pos = 0;
            while (pos <= all.size()) {
                size_t e = all.find(',', pos); if (e == std::string::npos) e = all.size();
                const std::string n = all.substr(pos, e - pos); pos = e + 1;
                if (n.empty()) continue;
                VkImage img = m4r_denoise_texture(ctx, n.c_str());
                if (!img) { std::fprintf(stderr, "no plan texture %s\n", n.c_str()); continue; }
                static const std::map<std::string, std::pair<VkFormat, int>> known = {
                    {"shin", {VK_FORMAT_R8_UNORM, 1}}, {"mm0", {VK_FORMAT_R8_UNORM, 1}}, {"mm1", {VK_FORMAT_R8_UNORM, 1}}, {"mmr", {VK_FORMAT_R8_UNORM, 1}},
                    {"bw0", {VK_FORMAT_R8_UNORM, 1}}, {"bw1", {VK_FORMAT_R8_UNORM, 1}}, {"flowdiff", {VK_FORMAT_R16_SFLOAT, 2}}, {"fc0", {VK_FORMAT_R16_SFLOAT, 2}},
                    {"fc1", {VK_FORMAT_R16_SFLOAT, 2}}, {"spec_mvec", {VK_FORMAT_R16G16_SFLOAT, 4}}, {"int_motion", {VK_FORMAT_R16G16_SFLOAT, 4}},
                    {"int_depth", {VK_FORMAT_R32_SFLOAT, 4}}, {"hist", {VK_FORMAT_R16G16B16A16_SFLOAT, 8}}, {"warp_color", {VK_FORMAT_R16G16B16A16_SFLOAT, 8}},
                    {"warp_prev", {VK_FORMAT_R16G16B16A16_SFLOAT, 8}}, {"roughness", {VK_FORMAT_R16_SFLOAT, 2}}};
                auto k = known.find(n);
                if (k == known.end()) { std::fprintf(stderr, "unknown format of %s\n", n.c_str()); continue; }
                Texture x; vk.wrap(x, img, k->second.first, iw, ih);
                std::vector<uint8_t> b(size_t(iw) * ih * k->second.second);
                vk.download(x, b.data(), b.size());
                char nm[256]; std::snprintf(nm, sizeof nm, "%s/tex-%s-%02d.raw", out.c_str(), n.c_str(), t);
                std::ofstream(nm, std::ios::binary).write(reinterpret_cast<char*>(b.data()), std::streamsize(b.size()));
            }
        }
        if (kOutTest) {
            auto dumpK = [&](Texture& x, const char* name, size_t bpp) {
                std::vector<uint8_t> b(size_t(x.width) * x.height * bpp);
                vk.download(x, b.data(), b.size());
                char nm[256]; std::snprintf(nm, sizeof nm, "%s/%s-%02d.raw", out.c_str(), name, t);
                std::ofstream(nm, std::ios::binary).write(reinterpret_cast<char*>(b.data()), std::streamsize(b.size()));
            };
            dumpK(kColour, "kout-colour", 8); dumpK(kDepth, "kout-depth", 4);
        }
        dump(output, "output");
        std::printf("frame %d done\n", t);
    }
    m4r_destroy(ctx);
    vk.destroy();
    return rc;
}
