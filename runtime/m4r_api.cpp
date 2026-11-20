// m4r: C API (see m4r.h).
// SPDX-License-Identifier: GPL-3.0-only
#include "plan.h"
#include "params.h"
#include "m4r.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <memory>

using namespace m4r;

struct m4r_context {
    Context vk;
    Runtime rt{vk};
    std::string dir, error;
    std::unique_ptr<Plan> bbr[2], fg, dn;   // dn: the denoiser plan (render size)
    uint32_t dnSize[2] = {};
    bool dnV3 = false;   // the plan is the V3 one (dn3-*): it writes the final image, no separate upscaler
    bool dnHybrid = false;   // V3 plan + BBR upscaler: BBR (clamped to V3) on rough surfaces, V3 on shiny ones (m4r, shaders/nn_hybrid.comp; default, M4R_NO_HYBRID=1 = V3 alone)
    uint32_t dnOut[2] = {};
    uint64_t dnFrame = 0;
    double dnPrevVp[16] = {};
    float dnPrevJitter[2] = {};
    double dnPrevCam[3] = {};
    bool exposureOneCleared = false;   // bbr[1]: plan for reversed depth, bbr[0]: forward depth (plans are specialised for it)
    uint32_t bbrSize[4] = {}, fgSize[4] = {};
    bool bbrInit = false;
    std::map<uint64_t, bool> bbrNoPlan;   // content sizes without a plan of their own (looked up once)
    uint64_t bbrFrame[2] = {}, fgFrame = 0;
    float prevDt = 1.0f / 60.0f;
    std::map<VkImage, Texture> wrapped;
    Texture exposureOne;   // 1x1 exposure 1.0 when the game provides none
    // present-time frame generation: the last two finished frames and the generated frame (RGBA16F, display size)
    Texture frames[2], generated;
    int current = -1;
    // M4R_DUMP_FG=<dir>: depth and motion vectors as the frame generation saw them, written when the GPU is done with them
    Buffer dumpDepth, dumpMotion;
    int dumpPending = -1, dumpCount = 0;
    uint32_t dumpW = 0, dumpH = 0;
    // M4R_DUMP_UP=<dir>: the upscaler's inputs as the game gave them (raw texels) and each frame's parameters, for replay
    struct UpDump { Buffer buf[3]; uint32_t w[3], h[3], bpp[3]; VkFormat fmt[3]; uint64_t call; std::string meta; bool written = false; };
    std::vector<UpDump> upDumps;
    uint64_t upCalls = 0;

    const Texture* wrap(const m4r_image& im) {
        if (!im.image) return nullptr;
        auto it = wrapped.find(im.image);
        if (it != wrapped.end() && it->second.format == im.format && it->second.width == im.width) return &it->second;
        if (it != wrapped.end()) vk.destroyTexture(it->second);
        Texture t;
        if (!vk.wrap(t, im.image, im.format, im.width, im.height)) return nullptr;
        return &(wrapped[im.image] = t);
    }
};

// M4R_LOG=<file>: one line per frame generation call, so tests can replay the same parameters
static void logLine(const char* fmt, ...) {
    const char* path = std::getenv("M4R_LOG");
    if (!path) return;
    FILE* f = std::fopen(path, "a");
    if (!f) return;
    va_list ap;
    va_start(ap, fmt);
    std::vfprintf(f, fmt, ap);
    va_end(ap);
    std::fclose(f);
}


static uint32_t texelBytes(VkFormat f) {
    switch (f) {
    case VK_FORMAT_R16G16B16A16_SFLOAT: case VK_FORMAT_R16G16B16A16_UNORM: case VK_FORMAT_R32G32_SFLOAT: return 8;
    case VK_FORMAT_R32G32B32A32_SFLOAT: return 16;
    case VK_FORMAT_R8G8B8A8_UNORM: case VK_FORMAT_R8G8B8A8_SRGB: case VK_FORMAT_B8G8R8A8_UNORM: case VK_FORMAT_B8G8R8A8_SRGB:
    case VK_FORMAT_A2B10G10R10_UNORM_PACK32: case VK_FORMAT_A2R10G10B10_UNORM_PACK32: case VK_FORMAT_B10G11R11_UFLOAT_PACK32:
    case VK_FORMAT_E5B9G9R9_UFLOAT_PACK32: case VK_FORMAT_R16G16_SFLOAT: case VK_FORMAT_R16G16_SNORM: case VK_FORMAT_R16G16_UNORM:
    case VK_FORMAT_R32_SFLOAT: case VK_FORMAT_D32_SFLOAT: case VK_FORMAT_D32_SFLOAT_S8_UINT: case VK_FORMAT_D24_UNORM_S8_UINT:
    case VK_FORMAT_X8_D24_UNORM_PACK32: case VK_FORMAT_R32_UINT: return 4;
    case VK_FORMAT_R16_SFLOAT: case VK_FORMAT_R16_UNORM: case VK_FORMAT_D16_UNORM: return 2;
    default: return 0;
    }
}

static void writeUpDumps(m4r_context* c, bool all) {
    const char* dir = std::getenv("M4R_DUMP_UP");
    if (!dir) return;
    static const char* names[3] = {"color", "depth", "motion"};
    for (size_t i = 0; i < c->upDumps.size(); ++i) {
        auto& u = c->upDumps[i];
        if (u.written || (!all && c->upCalls < u.call + 6)) continue;   // a few frames later the GPU is done with the copies
        char nm[512];
        for (int k = 0; k < 3; ++k) {
            if (!u.buf[k].buffer) continue;
            std::snprintf(nm, sizeof nm, "%s/up-%02zu-%s.raw", dir, i, names[k]);
            if (FILE* f = std::fopen(nm, "wb")) { std::fwrite(u.buf[k].map, 1, size_t(u.w[k]) * u.h[k] * u.bpp[k], f); std::fclose(f); }
            c->vk.destroyBuffer(u.buf[k]);
        }
        std::snprintf(nm, sizeof nm, "%s/up-%02zu.txt", dir, i);
        if (FILE* f = std::fopen(nm, "w")) { std::fputs(u.meta.c_str(), f); std::fclose(f); }
        u.written = true;
    }
}


static void flushDump(m4r_context* c) {
    if (c->dumpPending < 0) return;
    const char* dir = std::getenv("M4R_DUMP_FG");
    if (dir) {
        char nm[512];
        std::snprintf(nm, sizeof nm, "%s/in-depth-%02d.r32f", dir, c->dumpPending);
        if (FILE* f = std::fopen(nm, "wb")) { std::fwrite(c->dumpDepth.map, 1, size_t(c->dumpW) * c->dumpH * 4, f); std::fclose(f); }
        std::snprintf(nm, sizeof nm, "%s/in-motion-%02d.rg16f", dir, c->dumpPending);
        if (FILE* f = std::fopen(nm, "wb")) { std::fwrite(c->dumpMotion.map, 1, size_t(c->dumpW) * c->dumpH * 4, f); std::fclose(f); }
    }
    c->dumpPending = -1;
}

// moves caller images to GENERAL for m4r's passes and back afterwards
struct LayoutGuard {
    VkCommandBuffer cb;
    std::vector<VkImageMemoryBarrier> to, back;
    void add(const m4r_image& im) {
        if (!im.image || im.layout == VK_IMAGE_LAYOUT_UNDEFINED || im.layout == VK_IMAGE_LAYOUT_GENERAL) return;
        const bool depth = im.format == VK_FORMAT_D32_SFLOAT || im.format == VK_FORMAT_D32_SFLOAT_S8_UINT ||
                           im.format == VK_FORMAT_D24_UNORM_S8_UINT || im.format == VK_FORMAT_D16_UNORM ||
                           im.format == VK_FORMAT_X8_D24_UNORM_PACK32;
        const bool stencil = im.format == VK_FORMAT_D32_SFLOAT_S8_UINT || im.format == VK_FORMAT_D24_UNORM_S8_UINT;
        VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        b.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT; b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        b.oldLayout = im.layout; b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image = im.image;
        b.subresourceRange = {VkImageAspectFlags(depth ? VK_IMAGE_ASPECT_DEPTH_BIT | (stencil ? VK_IMAGE_ASPECT_STENCIL_BIT : 0)
                                                       : VK_IMAGE_ASPECT_COLOR_BIT), 0, 1, 0, 1};
        to.push_back(b);
        std::swap(b.oldLayout, b.newLayout);
        b.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT; b.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
        back.push_back(b);
    }
    void begin() {
        if (!to.empty())
            vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0,
                                 nullptr, uint32_t(to.size()), to.data());
    }
    void end() {
        if (!back.empty())
            vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0,
                                 nullptr, uint32_t(back.size()), back.data());
    }
};

static void dumpUpscale(m4r_context* c, VkCommandBuffer cb, const m4r_upscale_desc* d, uint32_t cw, uint32_t ch) {
    const char* dir = std::getenv("M4R_DUMP_UP");
    if (!dir) return;
    writeUpDumps(c, false);
    const char* st = std::getenv("M4R_DUMP_UP_START");
    const char* nn = std::getenv("M4R_DUMP_UP_COUNT");
    const uint64_t start = st ? std::strtoull(st, nullptr, 10) : 300, count = nn ? std::strtoull(nn, nullptr, 10) : 6;
    if (c->upCalls < start || c->upCalls >= start + count) return;
    m4r_context::UpDump u;
    u.call = c->upCalls;
    const m4r_image* ims[3] = {&d->color, &d->depth, &d->motion};
    LayoutGuard g{cb};
    for (auto* im : ims) g.add(*im);
    g.begin();
    char line[1024];
    std::string meta;
    for (int k = 0; k < 3; ++k) {
        const m4r_image& im = *ims[k];
        u.fmt[k] = im.format; u.w[k] = im.width; u.h[k] = im.height; u.bpp[k] = texelBytes(im.format);
        std::snprintf(line, sizeof line, "image %d format %d size %u %u bpp %u\n", k, int(im.format), im.width, im.height, u.bpp[k]);
        meta += line;
        if (!u.bpp[k]) continue;
        c->vk.createBuffer(u.buf[k], size_t(im.width) * im.height * u.bpp[k], true);
        const bool depthFmt = im.format == VK_FORMAT_D32_SFLOAT || im.format == VK_FORMAT_D32_SFLOAT_S8_UINT ||
                              im.format == VK_FORMAT_D24_UNORM_S8_UINT || im.format == VK_FORMAT_X8_D24_UNORM_PACK32 ||
                              im.format == VK_FORMAT_D16_UNORM;
        VkBufferImageCopy r{};
        r.imageSubresource = {VkImageAspectFlags(depthFmt ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT), 0, 0, 1};
        r.imageExtent = {im.width, im.height, 1};
        vkCmdCopyImageToBuffer(cb, im.image, VK_IMAGE_LAYOUT_GENERAL, u.buf[k].buffer, 1, &r);
    }
    c->vk.barrier(cb);
    g.end();
    std::snprintf(line, sizeof line, "output %u %u\ncontent %u %u\nplan %u %u %u %u\njitter %.9g %.9g\nmvscale %.9g %.9g\n"
                  "preexposure %.9g\nreset %d\ndepth_reversed %d\nexposure %d\n", d->output.width, d->output.height, cw, ch,
                  c->bbrSize[0], c->bbrSize[1], c->bbrSize[2], c->bbrSize[3], d->jitter_x, d->jitter_y, d->motion_scale_x,
                  d->motion_scale_y, d->pre_exposure, int(d->reset), int(d->depth_reversed), d->exposure.image ? 1 : 0);
    meta += line;
    u.meta = meta;
    c->upDumps.push_back(std::move(u));
}

// M4R_DUMP_RR: Ray Reconstruction inputs
struct RrDump {
    static const int N = 11;
    Buffer buf[N]; uint32_t w[N] = {}, h[N] = {}, bpp[N] = {}; uint64_t call = 0; std::string meta; bool written = false;
    // what m4r made of the frame (written by m4r_denoise): the denoised colour at render size and the final output
    Buffer outBuf[2]; uint32_t ow[2] = {}, oh[2] = {}, obpp[2] = {};
};
static RrDump* g_rrCurrent = nullptr;   // the capture of the call being recorded (set by m4r_denoise_capture), else null
static const char* const kRrOutNames[2] = {"out_denoised", "out_output"};
static std::vector<RrDump> g_rrDumps;
static uint64_t g_rrCalls = 0;
static const char* const kRrNames[RrDump::N] = {"color", "depth", "motion", "diffuse_albedo", "specular_albedo", "normals", "roughness",
                                                "specular_hit", "diffuse_hit", "before_particles", "exposure"};

static void writeRrDumps(m4r_context* c, bool all) {
    const char* dir = std::getenv("M4R_DUMP_RR");
    if (!dir) return;
    for (size_t i = 0; i < g_rrDumps.size(); ++i) {
        auto& u = g_rrDumps[i];
        if (u.written || (!all && g_rrCalls < u.call + 6)) continue;
        char nm[512];
        for (int k = 0; k < RrDump::N; ++k) {
            if (!u.buf[k].buffer) continue;
            std::snprintf(nm, sizeof nm, "%s/rr-%02llu-%s.raw", dir, (unsigned long long) u.call, kRrNames[k]);
            if (FILE* f = std::fopen(nm, "wb")) { std::fwrite(u.buf[k].map, 1, size_t(u.w[k]) * u.h[k] * u.bpp[k], f); std::fclose(f); }
            c->vk.destroyBuffer(u.buf[k]);
        }
        for (int k = 0; k < 2; ++k) {
            if (!u.outBuf[k].buffer) continue;
            std::snprintf(nm, sizeof nm, "%s/rr-%02llu-%s.raw", dir, (unsigned long long) u.call, kRrOutNames[k]);
            if (FILE* f = std::fopen(nm, "wb")) { std::fwrite(u.outBuf[k].map, 1, size_t(u.ow[k]) * u.oh[k] * u.obpp[k], f); std::fclose(f); }
            c->vk.destroyBuffer(u.outBuf[k]);
        }
        std::snprintf(nm, sizeof nm, "%s/rr-%02llu.txt", dir, (unsigned long long) u.call);
        if (FILE* f = std::fopen(nm, "w")) { std::fputs(u.meta.c_str(), f); std::fclose(f); }
        u.written = true;
    }
}

// full-image layout change with a full memory dependency (used around transfers on caller images)
static void transition(VkCommandBuffer cb, VkImage image, VkImageLayout from, VkImageLayout to) {
    VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    b.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
    b.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    b.oldLayout = from; b.newLayout = to;
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = image;
    b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
}

// copies src to dst with format conversion (both colour images, same size); layouts are restored afterwards
static void blit(VkCommandBuffer cb, VkImage src, VkImageLayout srcLayout, VkImage dst, VkImageLayout dstLayout, uint32_t w, uint32_t h) {
    const auto general = [](VkImageLayout l) { return l == VK_IMAGE_LAYOUT_UNDEFINED ? VK_IMAGE_LAYOUT_GENERAL : l; };
    srcLayout = general(srcLayout); dstLayout = general(dstLayout);
    if (srcLayout != VK_IMAGE_LAYOUT_GENERAL) transition(cb, src, srcLayout, VK_IMAGE_LAYOUT_GENERAL);
    if (dstLayout != VK_IMAGE_LAYOUT_GENERAL) transition(cb, dst, dstLayout, VK_IMAGE_LAYOUT_GENERAL);
    if (srcLayout == VK_IMAGE_LAYOUT_GENERAL && dstLayout == VK_IMAGE_LAYOUT_GENERAL) {
        VkMemoryBarrier m{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        m.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT; m.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &m, 0, nullptr, 0, nullptr);
    }
    VkImageBlit r{};
    r.srcSubresource = r.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    r.srcOffsets[1] = r.dstOffsets[1] = {int32_t(w), int32_t(h), 1};
    vkCmdBlitImage(cb, src, VK_IMAGE_LAYOUT_GENERAL, dst, VK_IMAGE_LAYOUT_GENERAL, 1, &r, VK_FILTER_NEAREST);
    if (srcLayout != VK_IMAGE_LAYOUT_GENERAL) transition(cb, src, VK_IMAGE_LAYOUT_GENERAL, srcLayout);
    if (dstLayout != VK_IMAGE_LAYOUT_GENERAL) transition(cb, dst, VK_IMAGE_LAYOUT_GENERAL, dstLayout);
    else {
        VkMemoryBarrier m{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        m.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT; m.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &m, 0, nullptr, 0, nullptr);
    }
}

static m4r_image ownedImage(const Texture& t) { return m4r_image{t.image, t.format, t.width, t.height, VK_IMAGE_LAYOUT_GENERAL}; }

static std::string sizeName(const char* kind, const uint32_t* s) {
    return std::string(kind) + "-" + std::to_string(s[0]) + "x" + std::to_string(s[1]) + "-" + std::to_string(s[2]) + "x" + std::to_string(s[3]);
}

extern "C" {

M4R_API int m4r_create(m4r_context** out, PFN_vkGetInstanceProcAddr gipa, VkInstance instance, VkPhysicalDevice physical,
                       VkDevice device, uint32_t family, const char* dataDir) {
    auto* c = new m4r_context();
    c->dir = dataDir ? dataDir : ".";
    *out = c;
    if (!loadVulkan(gipa, instance, device)) { c->error = "missing Vulkan entry points"; return 1; }
    if (!c->vk.adopt(instance, physical, device, family)) { c->error = c->vk.error; return 1; }
    if (!c->rt.init()) { c->error = c->rt.error; return 1; }
    return 0;
}

M4R_API void m4r_destroy(m4r_context* c) {
    if (!c) return;
    flushDump(c);
    writeUpDumps(c, true);
    writeRrDumps(c, true);
    if (c->dumpDepth.buffer) { c->vk.destroyBuffer(c->dumpDepth); c->vk.destroyBuffer(c->dumpMotion); }
    c->bbr[0].reset(); c->bbr[1].reset(); c->fg.reset(); c->dn.reset();
    for (auto& [img, t] : c->wrapped) c->vk.destroyTexture(t);
    if (c->exposureOne.image) c->vk.destroyTexture(c->exposureOne);
    for (Texture* t : {&c->frames[0], &c->frames[1], &c->generated})
        if (t->image) c->vk.destroyTexture(*t);
    c->rt.destroy();
    c->vk.destroyBuffer(c->vk.records);
    delete c;
}

M4R_API const char* m4r_last_error(m4r_context* c) { return c ? c->error.c_str() : "no context"; }

M4R_API int m4r_upscale_init(m4r_context* c, uint32_t ow, uint32_t oh, uint32_t iw, uint32_t ih) {
    const uint32_t s[4] = {ow, oh, iw, ih};
    if (c->bbrInit && !std::memcmp(s, c->bbrSize, sizeof s)) return 0;
    // The plan depends on the depth convention (the kernels differ), so each one has its own: bbr-WxH-wxh for reversed depth
    // (what most engines use), bbr-WxH-wxh-fwd for forward depth. Either is enough here; m4r_upscale picks by desc->depth_reversed.
    std::unique_ptr<Plan> plans[2];
    std::string why;
    for (int rev = 0; rev < 2; ++rev) {
        auto p = std::make_unique<Plan>(c->rt);
        if (p->load(c->dir + "/" + sizeName("bbr", s) + (rev ? "" : "-fwd"))) plans[rev] = std::move(p);
        else why += (why.empty() ? "" : "; ") + p->error;
    }
    if (!plans[0] && !plans[1]) { c->error = why; return 1; }
    if (!c->exposureOne.image) {
        if (!c->vk.createTexture(c->exposureOne, VK_FORMAT_R32_SFLOAT, 1, 1)) { c->error = "exposure texture"; return 1; }
    }
    c->bbr[0] = std::move(plans[0]); c->bbr[1] = std::move(plans[1]);
    std::memcpy(c->bbrSize, s, sizeof s);
    c->bbrInit = true;
    c->bbrFrame[0] = c->bbrFrame[1] = 0;
    return 0;
}

M4R_API int m4r_denoise_capture(m4r_context* c, VkCommandBuffer cb, const m4r_denoise_desc* d) {
    const char* dir = std::getenv("M4R_DUMP_RR");
    g_rrCurrent = nullptr;
    if (!dir) return 0;
    writeRrDumps(c, false);
    const char* st = std::getenv("M4R_DUMP_RR_START");
    const char* nn = std::getenv("M4R_DUMP_RR_COUNT");
    // Capture is started by creating the file <dir>/trigger (checked every 10 frames; the file is removed when it is seen), or
    // at frame M4R_DUMP_RR_START when that is set.
    static uint64_t start = ~0ull;
    const uint64_t count = nn ? std::strtoull(nn, nullptr, 10) : 6;
    const uint64_t call = g_rrCalls++;
    if (start == ~0ull) {
        if (st && call >= std::strtoull(st, nullptr, 10)) start = call;
        else if (call % 10 == 0) {
            char tr[512];
            std::snprintf(tr, sizeof tr, "%s/trigger", dir);
            if (FILE* f = std::fopen(tr, "rb")) { std::fclose(f); std::remove(tr); start = call + 1; }
        }
    }
    if (start != ~0ull && call >= start + count) start = ~0ull;   // window over: arm again for the next trigger
    if (call < start || call >= start + count) return 0;
    RrDump u;
    u.call = call;
    const m4r_image* ims[RrDump::N] = {&d->color, &d->depth, &d->motion, &d->diffuse_albedo, &d->specular_albedo, &d->normals,
                                       &d->roughness, &d->specular_hit, &d->diffuse_hit, &d->color_before_particles, &d->exposure};
    LayoutGuard g{cb};
    for (auto* im : ims) g.add(*im);
    g.begin();
    std::string meta;
    char line[2048];
    for (int k = 0; k < RrDump::N; ++k) {
        const m4r_image& im = *ims[k];
        if (!im.image) continue;
        u.w[k] = im.width; u.h[k] = im.height; u.bpp[k] = texelBytes(im.format);
        std::snprintf(line, sizeof line, "image %s format %d size %u %u bpp %u\n", kRrNames[k], int(im.format), im.width, im.height, u.bpp[k]);
        meta += line;
        if (!u.bpp[k]) continue;
        c->vk.createBuffer(u.buf[k], size_t(im.width) * im.height * u.bpp[k], true);
        const bool depthFmt = im.format == VK_FORMAT_D32_SFLOAT || im.format == VK_FORMAT_D32_SFLOAT_S8_UINT ||
                              im.format == VK_FORMAT_D24_UNORM_S8_UINT || im.format == VK_FORMAT_X8_D24_UNORM_PACK32 ||
                              im.format == VK_FORMAT_D16_UNORM;
        VkBufferImageCopy r{};
        r.imageSubresource = {VkImageAspectFlags(depthFmt ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT), 0, 0, 1};
        r.imageExtent = {im.width, im.height, 1};
        vkCmdCopyImageToBuffer(cb, im.image, VK_IMAGE_LAYOUT_GENERAL, u.buf[k].buffer, 1, &r);
    }
    c->vk.barrier(cb);
    g.end();
    std::snprintf(line, sizeof line, "output %u %u\ncontent %u %u\njitter %.9g %.9g\nmvscale %.9g %.9g\npreexposure %.9g\nreset %d\n"
                  "depth_reversed %d\nhardware_depth %d\nroughness_in_normals_w %d\nmotion_low_res %d\nspecular_hit_in_w %d\n",
                  d->output.width, d->output.height, d->content_width, d->content_height, d->jitter_x, d->jitter_y, d->motion_scale_x,
                  d->motion_scale_y, d->pre_exposure, d->reset, d->depth_reversed, d->hardware_depth, d->roughness_in_normals_w,
                  d->motion_low_res, d->specular_hit_in_w);
    meta += line;
    for (int m = 0; m < 2; ++m) {
        meta += m ? "view_to_clip" : "world_to_view";
        for (int i = 0; i < 16; ++i) { std::snprintf(line, sizeof line, " %.9g", m ? d->view_to_clip[i] : d->world_to_view[i]); meta += line; }
        meta += "\n";
    }
    u.meta = meta;
    g_rrDumps.push_back(std::move(u));
    g_rrCurrent = &g_rrDumps.back();
    return 0;
}

// ---- MetalFX denoiser ----------------------------------------------------------------------------------------------------
using M4 = std::array<double, 16>;   // row-major
static M4 mul(const M4& a, const M4& b) {
    M4 r{};
    for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) for (int k = 0; k < 4; ++k) r[i * 4 + j] += a[i * 4 + k] * b[k * 4 + j];
    return r;
}
static M4 transposed(const M4& a) { M4 r; for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) r[i * 4 + j] = a[j * 4 + i]; return r; }
static M4 inverse(const M4& m) {   // Gauss-Jordan with partial pivoting; singular input gives the identity
    double a[4][8];
    for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) { a[i][j] = m[i * 4 + j]; a[i][4 + j] = i == j; }
    for (int c = 0; c < 4; ++c) {
        int p = c;
        for (int r = c + 1; r < 4; ++r) if (std::fabs(a[r][c]) > std::fabs(a[p][c])) p = r;
        if (std::fabs(a[p][c]) < 1e-300) return M4{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
        if (p != c) for (int j = 0; j < 8; ++j) std::swap(a[p][j], a[c][j]);
        const double d = a[c][c];
        for (int j = 0; j < 8; ++j) a[c][j] /= d;
        for (int r = 0; r < 4; ++r) if (r != c) { const double f = a[r][c]; for (int j = 0; j < 8; ++j) a[r][j] -= f * a[c][j]; }
    }
    M4 r; for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) r[i * 4 + j] = a[i][4 + j];
    return r;
}
static void putColumnMajor(uint8_t* dst, const M4& m) {
    for (int c = 0; c < 4; ++c) for (int r = 0; r < 4; ++r) { const float f = float(m[r * 4 + c]); std::memcpy(dst + (c * 4 + r) * 4, &f, 4); }
}

// copies a GENERAL-layout colour image into a host buffer of the current capture (slot 0 denoised, 1 output)
static void captureOut(m4r_context* c, VkCommandBuffer cb, int slot, VkImage image, uint32_t w, uint32_t h, VkFormat fmt) {
    RrDump* u = g_rrCurrent;
    const uint32_t bpp = texelBytes(fmt);
    if (!u || !bpp || !image) return;
    if (!c->vk.createBuffer(u->outBuf[slot], size_t(w) * h * bpp, true)) return;
    u->ow[slot] = w; u->oh[slot] = h; u->obpp[slot] = bpp;
    char line[128];
    std::snprintf(line, sizeof line, "%s format %d size %u %u bpp %u\n", kRrOutNames[slot], int(fmt), w, h, bpp);
    u->meta += line;
    VkBufferImageCopy r{};
    r.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    r.imageExtent = {w, h, 1};
    c->vk.barrier(cb);
    vkCmdCopyImageToBuffer(cb, image, VK_IMAGE_LAYOUT_GENERAL, u->outBuf[slot].buffer, 1, &r);
    c->vk.barrier(cb);
}

M4R_API int m4r_denoise_init(m4r_context* c, uint32_t ow, uint32_t oh, uint32_t iw, uint32_t ih) {
    if (c->dn && c->dnSize[0] == iw && c->dnSize[1] == ih && c->dnOut[0] == ow && c->dnOut[1] == oh)
        return c->dnV3 && !c->dnHybrid ? 0 : m4r_upscale_init(c, ow, oh, iw, ih);
    char name[96];
    // preferred: the plan with MetalFX's own upscaling network (dn3-<out>-<in>); M4R_NO_V3=1 keeps the older denoiser + BBR upscaler
    if (!std::getenv("M4R_NO_V3")) {
        auto p3 = std::make_unique<Plan>(c->rt);
        std::snprintf(name, sizeof name, "/dn3-%ux%u-%ux%u", ow, oh, iw, ih);
        if (p3->load(c->dir + name)) {
            c->dn = std::move(p3);
            c->dnSize[0] = iw; c->dnSize[1] = ih; c->dnOut[0] = ow; c->dnOut[1] = oh;
            c->dnFrame = 0; c->dnV3 = true; c->dnHybrid = false;
            if (!c->exposureOne.image && !c->vk.createTexture(c->exposureOne, VK_FORMAT_R32_SFLOAT, 1, 1)) { c->error = "exposure texture"; return 1; }
            // hybrid (m4r, not MetalFX; default since 2026-11-20, M4R_NO_HYBRID=1 = V3 alone): also the BBR upscaler; without its plan
            // pass or the BBR plan, V3 alone
            if (!std::getenv("M4R_NO_HYBRID") && c->dn->firstOnly("hybrid") < c->dn->passes()) {
                const std::string keep = c->error;
                if (m4r_upscale_init(c, ow, oh, iw, ih) == 0) c->dnHybrid = true;
                else c->error = keep;
            }
            return 0;
        }
    }
    auto p = std::make_unique<Plan>(c->rt);
    std::snprintf(name, sizeof name, "/dn-%ux%u", iw, ih);
    if (!p->load(c->dir + name)) { c->error = p->error; return 1; }
    if (m4r_upscale_init(c, ow, oh, iw, ih) != 0) return 1;
    c->dn = std::move(p);
    c->dnSize[0] = iw; c->dnSize[1] = ih; c->dnOut[0] = ow; c->dnOut[1] = oh;
    c->dnFrame = 0; c->dnV3 = false;
    return 0;
}

M4R_API int m4r_denoise(m4r_context* c, VkCommandBuffer cb, const m4r_denoise_desc* d) {
    if (!c->dn) { c->error = "m4r_denoise_init not called"; return 1; }
    const uint32_t iw = c->dnSize[0], ih = c->dnSize[1];
    if (d->content_width != iw || d->content_height != ih) {
        char b[160];
        std::snprintf(b, sizeof b, "denoiser plan is for %ux%u, the game renders %ux%u", iw, ih, d->content_width, d->content_height);
        c->error = b;
        return 1;
    }
    Plan& p = *c->dn;
    const Texture *color = c->wrap(d->color), *depth = c->wrap(d->depth), *motion = c->wrap(d->motion), *normals = c->wrap(d->normals),
                  *diffuse = c->wrap(d->diffuse_albedo), *specular = c->wrap(d->specular_albedo);
    if (!color || !depth || !motion || !normals || !diffuse || !specular || !d->output.image) {
        c->error = "Ray Reconstruction: missing color, depth, motion vectors, normals, albedo or output";
        return 1;
    }
    const Texture* rough = d->roughness.image ? c->wrap(d->roughness) : nullptr;
    if (!d->roughness_in_normals_w && !rough) { c->error = "Ray Reconstruction: no roughness"; return 1; }
    // The denoiser's output is scaled by the exposure it was given (it feeds MetalFX's upscaling network in exposed units); the
    // upscaler here wants the game's own colour range, so the denoiser always runs with exposure 1 (its auto-exposure passes stay off).
    const Texture* exposure = &c->exposureOne;
    p.setFlag("autoexp", false);
    LayoutGuard g{cb};
    for (const m4r_image* im : {&d->color, &d->depth, &d->motion, &d->normals, &d->diffuse_albedo, &d->specular_albedo, &d->roughness})
        g.add(*im);
    if (c->dnV3 && !c->dnHybrid) g.add(d->output);
    g.begin();
    if (!c->exposureOneCleared) {
        c->vk.toGeneral(cb, c->exposureOne);
        VkClearColorValue one{{1.0f, 1.0f, 1.0f, 1.0f}};
        VkImageSubresourceRange r{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCmdClearColorImage(cb, c->exposureOne.image, VK_IMAGE_LAYOUT_GENERAL, &one, 1, &r);
        c->exposureOneCleared = true;
    }
    p.bindTexture("color", color); p.bindTexture("depth", depth); p.bindTexture("motion", motion); p.bindTexture("normals", normals);
    p.bindTexture("diffuse", diffuse); p.bindTexture("specular", specular); p.bindTexture("exposure", exposure);
    if (rough) p.bindTexture("roughness", rough);

    // Per-frame parameters (BRNetShaderParams-style structs of MetalFX's BFNet filter, docs/metalfx-denoiser.md)
    const uint32_t W = (iw + 7) & ~7u, H = (ih + 7) & ~7u, tw = ((iw + 15) >> 1) & ~7u, th = ((ih + 15) >> 1) & ~7u;
    const bool reset = d->reset || c->dnFrame == 0;
    M4 w2v, v2c;
    for (int i = 0; i < 16; ++i) { w2v[i] = d->world_to_view[i]; v2c[i] = d->view_to_clip[i]; }
    const M4 Wc = transposed(w2v), Pc = transposed(v2c);   // NGX: row-major, row vectors; MetalFX: column vectors
    const M4 curr = mul(Pc, Wc), invVp = inverse(curr), invW = inverse(Wc);
    M4 prev = curr;
    if (!reset) std::memcpy(prev.data(), c->dnPrevVp, sizeof c->dnPrevVp);
    const float ms[2] = {d->motion_scale_x, d->motion_scale_y}, jit[2] = {d->jitter_x, d->jitter_y};
    const float pj[2] = {reset ? 0.0f : c->dnPrevJitter[0], reset ? 0.0f : c->dnPrevJitter[1]};
    uint8_t dbf[40] = {}, dbfOut[40] = {}, refl[224] = {}, resetb[16] = {};
    auto dbfParams = [&](uint8_t* b, bool in) {   // dbfnet_v1_build_tensors_t
        if (in) { std::memcpy(b, ms, 8); std::memcpy(b + 8, pj, 8); std::memcpy(b + 16, jit, 8); }
        const int32_t rowBytes = in ? 26 : 14;
        const uint16_t hh = uint16_t(th), ww = uint16_t(tw);
        std::memcpy(b + 24, &rowBytes, 4); std::memcpy(b + 28, &hh, 2); std::memcpy(b + 30, &ww, 2); b[32] = reset;
    };
    dbfParams(dbf, true); dbfParams(dbfOut, false);
    putColumnMajor(refl, invVp); putColumnMajor(refl + 64, prev); putColumnMajor(refl + 128, curr);
    const float cam[4] = {float(invW[3]), float(invW[7]), float(invW[11]), float(invW[15])};
    const float rp[4] = {1.0f / W, 1.0f / H, W / ms[0], H / ms[1]};
    std::memcpy(refl + 192, cam, 16); std::memcpy(refl + 208, rp, 16);
    resetb[0] = reset;
    const float pre = d->pre_exposure > 0 ? d->pre_exposure : 1.0f;
    p.bindBuffer("dbf_params", c->rt.transient(dbf, sizeof dbf)); p.bindBuffer("dbf_params_out", c->rt.transient(dbfOut, sizeof dbfOut));
    p.bindBuffer("refl_params", c->rt.transient(refl, sizeof refl)); p.bindBuffer("preexp", c->rt.transient(&pre, 4));
    p.bindBuffer("reset", c->rt.transient(resetb, sizeof resetb));
    p.bindBuffer("rw_ms", c->rt.transient(ms, 8)); p.bindBuffer("rw_pj", c->rt.transient(pj, 8)); p.bindBuffer("rw_cj", c->rt.transient(jit, 8));
    if (c->dnV3) {   // BRNet_v3_Filter's parameter blocks (docs/metalfx-denoiser.md): BRNetShaderParams and brnet_v3_build_tensors_t
        const uint32_t ow = c->dnOut[0], oh = c->dnOut[1];
        const uint32_t Wn = ((ow / 2 + 31) / 32) * 32, Hn = ((oh / 2 + 31) / 32) * 32;
        uint8_t bsp[144] = {}, v3d[24] = {};
        const uint16_t sz[6] = {uint16_t(iw), uint16_t(ih), uint16_t(iw), uint16_t(ih), uint16_t(ow), uint16_t(oh)};
        std::memcpy(bsp, sz, sizeof sz);
        const float fl[8] = {jit[0], jit[1], ms[0], ms[1], float(iw) / ow, float(ih) / oh, ms[0] / iw, ms[1] / ih};
        std::memcpy(bsp + 32, fl, sizeof fl);
        const float v3f[4] = {ms[0] * ow, ms[1] * oh, jit[0], jit[1]};
        std::memcpy(v3d, v3f, sizeof v3f);
        const int32_t rowBytes = 10; const uint16_t hh = uint16_t(Hn), ww = uint16_t(Wn);
        std::memcpy(v3d + 16, &rowBytes, 4); std::memcpy(v3d + 20, &hh, 2); std::memcpy(v3d + 22, &ww, 2);
        p.bindBuffer("bparams", c->rt.transient(bsp, sizeof bsp)); p.bindBuffer("v3desc", c->rt.transient(v3d, sizeof v3d));
        // reflection-lag switch (m4r, not MetalFX; opt-in M4R_REFLECTION_LAG_FIX=1 or =<cap>): gate from the camera's translation per
        // frame (walking in Cyberpunk: 0.03-0.05 m/frame; turning or standing: <= 0.001), cap on the history weight
        static const char* lagFix = std::getenv("M4R_REFLECTION_LAG_FIX");
        float gate[4] = {0.0f, 0.3f, 0.0f, 0.0f};
        if (lagFix) {
            const double v = std::atof(lagFix);
            if (v > 0.0 && v < 1.0) gate[1] = float(v);
            if (!reset) {
                const double dx = cam[0] - c->dnPrevCam[0], dy = cam[1] - c->dnPrevCam[1], dz = cam[2] - c->dnPrevCam[2];
                const double m = std::sqrt(dx * dx + dy * dy + dz * dz), x = std::clamp((m - 0.003) / (0.02 - 0.003), 0.0, 1.0);
                gate[0] = float(x * x * (3.0 - 2.0 * x));
            }
        }
        p.bindBuffer("mc_gate", c->rt.transient(gate, sizeof gate));
        p.setFlag("mircap", lagFix != nullptr && gate[0] > 0.0f && p.buffer("rough_raw") != nullptr && d->roughness_in_normals_w && !rough);
        c->dnPrevCam[0] = cam[0]; c->dnPrevCam[1] = cam[1]; c->dnPrevCam[2] = cam[2];
        if (!c->dnHybrid) p.bindTexture("v3out", c->wrap(d->output));   // hybrid: V3 writes the plan's own v3out
    }
    p.setParity(int(c->dnFrame & 1));
    // MetalFX's auto exposure (no exposure texture) scales the denoised colour by the exposure; that is only right once the stage after
    // the denoiser (UBFNet / V3) is MetalFX's own, so it is opt-in for now
    p.setFlag("autoexp", !d->exposure.image && std::getenv("M4R_AUTOEXPOSURE") != nullptr);

    bool ok = p.record(cb, 0, 0);   // textures: first-use clear
    size_t first = 0;
    if (d->roughness_in_normals_w && !rough) {   // roughness = normals.w (pass 0 of the plan)
        Buffer *nrm = p.buffer("nrm_raw"), *rr = p.buffer("rough_raw");
        Texture* rt = p.texture("roughness");
        if (!nrm || !rr || !rt) { c->error = "denoiser plan lacks the roughness extraction (prepare_denoiser.py --runtime)"; return 1; }
        VkBufferImageCopy cp{};
        cp.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        cp.imageExtent = {iw, ih, 1};
        c->vk.barrier(cb);
        vkCmdCopyImageToBuffer(cb, normals->image, VK_IMAGE_LAYOUT_GENERAL, nrm->buffer, 1, &cp);
        c->vk.barrier(cb);
        ok = ok && p.record(cb, 0, 1);
        c->vk.barrier(cb);
        vkCmdCopyBufferToImage(cb, rr->buffer, rt->image, VK_IMAGE_LAYOUT_GENERAL, 1, &cp);
        c->vk.barrier(cb);
        first = 1;
    }
    // m4r's own passes on the denoised colour, a block of `only despeckle` / `only glossy` passes right after MetalFX's denoiser (before
    // the V3 stage, if the plan has one), run on buffer copies: the interim firefly clamp (den_raw -> den_cl) and the glossy-in-motion
    // smoothing (den_cl -> ... -> den_cl; needs the normals copy with roughness in w and buffer copies of int_depth / int_motion).
    // Neither is MetalFX behaviour; M4R_NO_DESPECKLE=1 turns the clamp off, M4R_GLOSSY_SMOOTH=1 turns the smoothing on.
    static const bool noDespeckle = std::getenv("M4R_NO_DESPECKLE") != nullptr;
    // the glossy smoothing is opt-in since 2026-11-20 (in game it flattened moving reflections into plastic-looking patches)
    static const bool noGlossy = std::getenv("M4R_GLOSSY_SMOOTH") == nullptr;
    Buffer *dRaw = p.buffer("den_raw"), *dCl = p.buffer("den_cl"), *glD = p.buffer("gl_depth"), *glM = p.buffer("gl_motion");
    Texture *denTex = p.texture("denoised"), *iDepth = p.texture("int_depth"), *iMotion = p.texture("int_motion");
    const size_t lo = std::min(p.firstOnly("despeckle"), p.firstOnly("glossy"));
    size_t hi = lo;
    while (hi < p.passes() && (p.onlyFlag(hi) == "despeckle" || p.onlyFlag(hi) == "glossy")) ++hi;
    const bool despeckle = !noDespeckle && dRaw && dCl && denTex && p.firstOnly("despeckle") < p.passes();
    const bool glossy = !noGlossy && dRaw && dCl && denTex && glD && glM && iDepth && iMotion && first == 1 &&
                        p.firstOnly("glossy") < p.passes();
    p.setFlag("despeckle", despeckle); p.setFlag("glossy", glossy);
    ok = ok && p.record(cb, first, lo > first ? lo - first : 0);
    c->vk.barrier(cb);
    if ((despeckle || glossy) && ok) {
        VkBufferImageCopy cp{};
        cp.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        cp.imageExtent = {iw, ih, 1};
        // without the clamp the smoothing starts from den_cl
        vkCmdCopyImageToBuffer(cb, denTex->image, VK_IMAGE_LAYOUT_GENERAL, despeckle ? dRaw->buffer : dCl->buffer, 1, &cp);
        if (glossy) {
            vkCmdCopyImageToBuffer(cb, iDepth->image, VK_IMAGE_LAYOUT_GENERAL, glD->buffer, 1, &cp);
            vkCmdCopyImageToBuffer(cb, iMotion->image, VK_IMAGE_LAYOUT_GENERAL, glM->buffer, 1, &cp);
        }
        c->vk.barrier(cb);
        ok = p.record(cb, lo, hi - lo);
        c->vk.barrier(cb);
        vkCmdCopyBufferToImage(cb, dCl->buffer, denTex->image, VK_IMAGE_LAYOUT_GENERAL, 1, &cp);
        c->vk.barrier(cb);
    }
    if (ok && hi < p.passes()) {   // the rest of the plan (the V3 stage)
        ok = p.record(cb, hi);
        c->vk.barrier(cb);
    }
    g.end();
    if (!ok) { c->error = p.error; return 1; }
    std::memcpy(c->dnPrevVp, curr.data(), sizeof c->dnPrevVp);
    c->dnPrevJitter[0] = jit[0]; c->dnPrevJitter[1] = jit[1];
    ++c->dnFrame;

    Texture* den = p.texture("denoised");
    if (!den) { c->error = "denoiser plan has no 'denoised' texture"; return 1; }
    captureOut(c, cb, 0, den->image, den->width, den->height, den->format);
    if (c->dnV3 && !c->dnHybrid) {   // the plan wrote the final image into the game's output texture
        if (g_rrCurrent) {
            LayoutGuard og{cb};
            og.add(d->output);
            og.begin();
            captureOut(c, cb, 1, d->output.image, d->output.width, d->output.height, d->output.format);
            og.end();
        }
        return 0;
    }
    // upscaling: MetalFX's own upscaler on the denoised colour
    m4r_upscale_desc u{};
    u.color = ownedImage(*den); u.depth = d->depth; u.motion = d->motion; u.exposure = d->exposure; u.output = d->output;
    u.content_width = iw; u.content_height = ih;
    const float fj = d->flip_upscale_jitter ? -1.0f : 1.0f;
    u.jitter_x = fj * d->jitter_x; u.jitter_y = fj * d->jitter_y;
    u.motion_scale_x = d->motion_scale_x; u.motion_scale_y = d->motion_scale_y;
    u.pre_exposure = d->pre_exposure; u.reset = d->reset; u.depth_reversed = d->depth_reversed;
    int r = m4r_upscale(c, cb, &u);
    if (r == 0 && c->dnHybrid) {
        // hybrid (m4r, not MetalFX): BBR's output clamped to the V3 output's neighbourhood range (last plan pass, only hybrid)
        Texture* v3 = p.texture("v3out");
        Buffer *hv = p.buffer("hy_v3"), *hb = p.buffer("hy_bbr");
        const Texture* out = c->wrap(d->output);
        if (v3 && hv && hb && out && out->format == VK_FORMAT_R16G16B16A16_SFLOAT && out->width == v3->width && out->height == v3->height) {
            LayoutGuard og{cb};
            og.add(d->output);
            og.begin();
            VkBufferImageCopy cp{};
            cp.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            cp.imageExtent = {v3->width, v3->height, 1};
            c->vk.barrier(cb);
            vkCmdCopyImageToBuffer(cb, v3->image, VK_IMAGE_LAYOUT_GENERAL, hv->buffer, 1, &cp);
            vkCmdCopyImageToBuffer(cb, out->image, VK_IMAGE_LAYOUT_GENERAL, hb->buffer, 1, &cp);
            c->vk.barrier(cb);
            p.setFlag("hybrid", true);
            const bool hok = p.record(cb, p.firstOnly("hybrid"), 1);
            p.setFlag("hybrid", false);
            c->vk.barrier(cb);
            if (hok) vkCmdCopyBufferToImage(cb, hb->buffer, out->image, VK_IMAGE_LAYOUT_GENERAL, 1, &cp);
            c->vk.barrier(cb);
            og.end();
            if (!hok) { c->error = p.error; r = 1; }
        }
    }
    if (r == 0 && g_rrCurrent) {   // the output (UAV state for the caller) as m4r left it
        LayoutGuard og{cb};
        og.add(d->output);
        og.begin();
        captureOut(c, cb, 1, d->output.image, d->output.width, d->output.height, d->output.format);
        og.end();
    }
    return r;
}

M4R_API VkImage m4r_denoise_texture(m4r_context* c, const char* name) {
    Texture* t = c->dn ? c->dn->texture(name) : nullptr;
    return t ? t->image : VK_NULL_HANDLE;
}

M4R_API int m4r_upscale(m4r_context* c, VkCommandBuffer cb, const m4r_upscale_desc* d) {
    if (!c->bbrInit) { c->error = "m4r_upscale_init not called"; return 1; }
    // The size a game renders can differ from the one the feature was created with (Ghost of Tsushima creates DLSS at the
    // optimal 1505x847 and renders 1512x848). Prefer the plan made for the content size; otherwise the current plan serves
    // content up to its input size (MetalFX's maximum input size), and larger content is refused: the plan's buffers are
    // sized for its input, and running it would write past them.
    const uint32_t cw = d->content_width ? d->content_width : c->bbrSize[2], ch = d->content_height ? d->content_height : c->bbrSize[3];
    if (cw != c->bbrSize[2] || ch != c->bbrSize[3]) {
        const uint64_t key = (uint64_t(cw) << 32) | ch;
        if (!c->bbrNoPlan.count(key)) {
            const uint32_t ow = c->bbrSize[0], oh = c->bbrSize[1];
            const std::string keep = c->error;
            if (m4r_upscale_init(c, ow, oh, cw, ch) != 0) { c->bbrNoPlan[key] = true; c->error = keep; }
        }
        if (cw > c->bbrSize[2] || ch > c->bbrSize[3]) {
            char b[256];
            std::snprintf(b, sizeof b, "the game renders %ux%u, larger than the plan's %ux%u input; prepare a plan for %ux%u -> %ux%u "
                          "(install.py --input %ux%u)", cw, ch, c->bbrSize[2], c->bbrSize[3], cw, ch, c->bbrSize[0], c->bbrSize[1], cw, ch);
            c->error = b;
            return 1;
        }
    }
    const int rev = d->depth_reversed ? 1 : 0;
    if (!c->bbr[rev]) { c->error = std::string("no plan for ") + (rev ? "reversed" : "forward") + " depth (prepare it with --reversed-depth " + (rev ? "1" : "0") + ")"; return 1; }
    Plan& p = *c->bbr[rev];
    uint64_t& frame = c->bbrFrame[rev];
    const Texture *color = c->wrap(d->color), *depth = c->wrap(d->depth), *motion = c->wrap(d->motion), *out = c->wrap(d->output);
    if (!color || !depth || !motion || !out) { c->error = "missing input or output image"; return 1; }
    const Texture* exposure = d->exposure.image ? c->wrap(d->exposure) : &c->exposureOne;
    if (frame == 0 && !d->exposure.image) {   // exposure 1.0: cleared to 1 in the first recorded frame
        c->vk.toGeneral(cb, c->exposureOne);
        VkClearColorValue one{{1.0f, 1.0f, 1.0f, 1.0f}};
        VkImageSubresourceRange r{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCmdClearColorImage(cb, c->exposureOne.image, VK_IMAGE_LAYOUT_GENERAL, &one, 1, &r);
    }
    dumpUpscale(c, cb, d, cw, ch);
    ++c->upCalls;
    p.bindTexture("color", color); p.bindTexture("depth", depth); p.bindTexture("motion", motion);
    p.bindTexture("exposure", exposure); p.bindTexture("output", out);
    const uint32_t* s = c->bbrSize;
    BRNetShaderParams bp = bbrParams(int(s[0]), int(s[1]), int(s[2]), int(s[3]), int(cw), int(ch),
                                     d->jitter_x, d->jitter_y, d->motion_scale_x, d->motion_scale_y);
    const float pre = d->pre_exposure > 0 ? d->pre_exposure : 1.0f;
    const uint8_t reset = (d->reset || frame == 0) ? 1 : 0;
    const float cpf[2] = {float(cw) / s[0], float(ch) / s[1]};
    p.bindBuffer("params", c->rt.transient(&bp, sizeof bp));
    p.bindBuffer("preexp", c->rt.transient(&pre, 4));
    p.bindBuffer("reset", c->rt.transient(&reset, 1));
    p.bindBuffer("cpf", c->rt.transient(cpf, 8));
    // no exposure texture (the game asked for DLSS AutoExposure): MetalFX's auto exposure over the content area
    const uint32_t npix = cw * ch;
    p.bindBuffer("npix", c->rt.transient(&npix, 4));
    p.setFlag("autoexp", d->exposure.image == VK_NULL_HANDLE && !std::getenv("M4R_NO_AUTOEXPOSURE"));
    p.setParity(int(frame & 1));
    LayoutGuard g{cb};
    for (const m4r_image* im : {&d->color, &d->depth, &d->motion, &d->exposure, &d->output}) g.add(*im);
    g.begin();
    const bool ok = p.record(cb);
    c->vk.barrier(cb);
    g.end();
    if (!ok) { c->error = p.error; return 1; }
    ++frame;
    return 0;
}

M4R_API int m4r_interpolate_init(m4r_context* c, uint32_t ow, uint32_t oh, uint32_t iw, uint32_t ih) {
    const uint32_t s[4] = {ow, oh, iw, ih};
    if (c->fg && !std::memcmp(s, c->fgSize, sizeof s)) return 0;
    auto p = std::make_unique<Plan>(c->rt);
    if (!p->load(c->dir + "/" + sizeName("fg", s))) { c->error = p->error; return 1; }
    c->fg = std::move(p);
    std::memcpy(c->fgSize, s, sizeof s);
    c->fgFrame = 0;
    return 0;
}

M4R_API int m4r_interpolate(m4r_context* c, VkCommandBuffer cb, const m4r_interpolate_desc* d) {
    if (!c->fg) { c->error = "m4r_interpolate_init not called"; return 1; }
    Plan& p = *c->fg;
    const Texture *color = c->wrap(d->color), *prev = c->wrap(d->prev_color), *depth = c->wrap(d->depth),
                  *motion = c->wrap(d->motion), *out = c->wrap(d->output);
    if (!color || !prev || !depth || !motion || !out) { c->error = "missing input or output image"; return 1; }
    p.bindTexture("color", color); p.bindTexture("prevcolor", prev); p.bindTexture("depth", depth);
    p.bindTexture("motion", motion); p.bindTexture("output", out);
    const uint32_t* s = c->fgSize;
    const float dt = d->delta_time > 0 ? d->delta_time : 1.0f / 60.0f;
    FrameGenShaderParams fp = fgParams(int(s[0]), int(s[1]), int(s[2]), int(s[3]), d->motion_scale_x, d->motion_scale_y,
                                       d->depth_reversed != 0, dt, c->prevDt, d->near_plane > 0 ? d->near_plane : 0.1f,
                                       d->far_plane > 0 ? d->far_plane : 1000.0f);
    c->prevDt = dt;
    if (std::getenv("M4R_LOG")) {
        std::string hex;
        char b[3];
        for (size_t i = 0; i < sizeof fp; ++i) { std::snprintf(b, sizeof b, "%02x", reinterpret_cast<const uint8_t*>(&fp)[i]); hex += b; }
        logLine("interpolate params=%s\n", hex.c_str());
    }
    p.bindBuffer("fgparams", c->rt.transient(&fp, sizeof fp));
    p.setParity(int(c->fgFrame & 1));
    LayoutGuard g{cb};
    for (const m4r_image* im : {&d->color, &d->prev_color, &d->depth, &d->motion, &d->output}) g.add(*im);
    g.begin();
    const bool ok = p.record(cb);
    c->vk.barrier(cb);
    g.end();
    if (!ok) { c->error = p.error; return 1; }
    ++c->fgFrame;
    return 0;
}

M4R_API int m4r_framegen_capture(m4r_context* c, VkCommandBuffer cb, const m4r_framegen_desc* d) {
    if (std::getenv("M4R_DUMP_FG")) flushDump(c);   // the previous capture's commands have finished
    if (c->fg && std::getenv("M4R_LOG")) {   // the histogram analysis of the previous interpolation: bit i = histogram group i changed
        uint32_t w = 0xffffffffu;               // beyond the rejection threshold (frame change word, hist + 0x4800)
        if (Buffer* b = c->fg->buffer("hist")) std::memcpy(&w, static_cast<uint8_t*>(b->map) + 0x4800, 4);
        logLine("framechange %u\n", w);
    }
    const m4r_image& bb = d->backbuffer;
    if (!bb.image || !d->depth.image || !d->motion.image) { c->error = "missing backbuffer, depth or motion vectors"; return -1; }
    if (m4r_interpolate_init(c, bb.width, bb.height, d->depth.width, d->depth.height) != 0) return -1;
    if (c->generated.width != bb.width || c->generated.height != bb.height) {
        for (Texture* t : {&c->frames[0], &c->frames[1], &c->generated}) {
            if (auto w = c->wrapped.find(t->image); t->image && w != c->wrapped.end()) {
                c->vk.destroyTexture(w->second);
                c->wrapped.erase(w);
            }
            if (t->image) c->vk.destroyTexture(*t);
            if (!c->vk.createTexture(*t, VK_FORMAT_R16G16B16A16_SFLOAT, bb.width, bb.height)) { c->error = "frame textures"; return -1; }
            c->vk.toGeneral(cb, *t);
        }
        c->current = -1;
    }
    const int next = c->current < 0 ? 0 : c->current ^ 1;
    blit(cb, bb.image, bb.layout, c->frames[next].image, VK_IMAGE_LAYOUT_GENERAL, bb.width, bb.height);
    const bool generate = c->current >= 0 && !d->reset;
    if (generate) {
        m4r_interpolate_desc in{};
        in.color = ownedImage(c->frames[next]);
        in.prev_color = ownedImage(c->frames[c->current]);
        in.depth = d->depth; in.motion = d->motion;
        in.output = ownedImage(c->generated);
        in.motion_scale_x = d->motion_scale_x; in.motion_scale_y = d->motion_scale_y;
        in.delta_time = d->delta_time; in.near_plane = d->near_plane; in.far_plane = d->far_plane;
        in.depth_reversed = d->depth_reversed; in.reset = 0;
        if (m4r_interpolate(c, cb, &in) != 0) return -1;
        blit(cb, c->generated.image, VK_IMAGE_LAYOUT_GENERAL, bb.image, bb.layout, bb.width, bb.height);
    }
    if (generate && std::getenv("M4R_DUMP_FG")) {
        const uint32_t w = d->depth.width, h = d->depth.height;
        if (c->dumpW != w || c->dumpH != h) {
            if (c->dumpDepth.buffer) { c->vk.destroyBuffer(c->dumpDepth); c->vk.destroyBuffer(c->dumpMotion); }
            c->vk.createBuffer(c->dumpDepth, size_t(w) * h * 4, true);
            c->vk.createBuffer(c->dumpMotion, size_t(w) * h * 4, true);
            c->dumpW = w; c->dumpH = h;
        }
        LayoutGuard g{cb};
        g.add(d->depth); g.add(d->motion);
        g.begin();
        const bool depthFmt = d->depth.format == VK_FORMAT_D32_SFLOAT || d->depth.format == VK_FORMAT_D32_SFLOAT_S8_UINT;
        VkBufferImageCopy r{};
        r.imageSubresource = {VkImageAspectFlags(depthFmt ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT), 0, 0, 1};
        r.imageExtent = {w, h, 1};
        vkCmdCopyImageToBuffer(cb, d->depth.image, VK_IMAGE_LAYOUT_GENERAL, c->dumpDepth.buffer, 1, &r);
        r.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        vkCmdCopyImageToBuffer(cb, d->motion.image, VK_IMAGE_LAYOUT_GENERAL, c->dumpMotion.buffer, 1, &r);
        c->vk.barrier(cb);
        g.end();
        c->dumpPending = c->dumpCount;
    }
    ++c->dumpCount;
    c->current = next;
    logLine("capture bb=%ux%u depth=%ux%u mvscale=%.9g,%.9g dt=%.9g near=%.9g far=%.9g rev=%d reset=%d generated=%d\n",
            bb.width, bb.height, d->depth.width, d->depth.height, d->motion_scale_x, d->motion_scale_y, d->delta_time,
            d->near_plane, d->far_plane, d->depth_reversed, d->reset, generate ? 1 : 0);
    return generate ? 1 : 0;
}

M4R_API int m4r_framegen_restore(m4r_context* c, VkCommandBuffer cb, const m4r_image* bb) {
    if (c->current < 0 || !bb || !bb->image) { c->error = "no captured frame"; return -1; }
    const Texture& f = c->frames[c->current];
    if (bb->width != f.width || bb->height != f.height) { c->error = "backbuffer size changed"; return -1; }
    blit(cb, f.image, VK_IMAGE_LAYOUT_GENERAL, bb->image, bb->layout, f.width, f.height);
    return 0;
}

} // extern "C"
