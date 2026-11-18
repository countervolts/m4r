// m4r: C API (see m4r.h).
// SPDX-License-Identifier: GPL-3.0-only
#include "plan.h"
#include "params.h"
#include "m4r.h"
#include <map>
#include <memory>

using namespace m4r;

struct m4r_context {
    Context vk;
    Runtime rt{vk};
    std::string dir, error;
    std::unique_ptr<Plan> bbr, fg;
    uint32_t bbrSize[4] = {}, fgSize[4] = {};
    uint64_t bbrFrame = 0, fgFrame = 0;
    float prevDt = 1.0f / 60.0f;
    std::map<VkImage, Texture> wrapped;
    Texture exposureOne;   // 1x1 exposure 1.0 when the game provides none
    // present-time frame generation: the last two finished frames and the generated frame (RGBA16F, display size)
    Texture frames[2], generated;
    int current = -1;

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
    c->bbr.reset(); c->fg.reset();
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
    if (c->bbr && !std::memcmp(s, c->bbrSize, sizeof s)) return 0;
    auto p = std::make_unique<Plan>(c->rt);
    if (!p->load(c->dir + "/" + sizeName("bbr", s))) { c->error = p->error; return 1; }
    if (!c->exposureOne.image) {
        if (!c->vk.createTexture(c->exposureOne, VK_FORMAT_R32_SFLOAT, 1, 1)) { c->error = "exposure texture"; return 1; }
    }
    c->bbr = std::move(p);
    std::memcpy(c->bbrSize, s, sizeof s);
    c->bbrFrame = 0;
    return 0;
}

M4R_API int m4r_upscale(m4r_context* c, VkCommandBuffer cb, const m4r_upscale_desc* d) {
    if (!c->bbr) { c->error = "m4r_upscale_init not called"; return 1; }
    Plan& p = *c->bbr;
    const Texture *color = c->wrap(d->color), *depth = c->wrap(d->depth), *motion = c->wrap(d->motion), *out = c->wrap(d->output);
    if (!color || !depth || !motion || !out) { c->error = "missing input or output image"; return 1; }
    const Texture* exposure = d->exposure.image ? c->wrap(d->exposure) : &c->exposureOne;
    if (c->bbrFrame == 0 && !d->exposure.image) {   // exposure 1.0: cleared to 1 in the first recorded frame
        c->vk.toGeneral(cb, c->exposureOne);
        VkClearColorValue one{{1.0f, 1.0f, 1.0f, 1.0f}};
        VkImageSubresourceRange r{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCmdClearColorImage(cb, c->exposureOne.image, VK_IMAGE_LAYOUT_GENERAL, &one, 1, &r);
    }
    p.bindTexture("color", color); p.bindTexture("depth", depth); p.bindTexture("motion", motion);
    p.bindTexture("exposure", exposure); p.bindTexture("output", out);
    const uint32_t* s = c->bbrSize;
    BRNetShaderParams bp = bbrParams(int(s[0]), int(s[1]), int(s[2]), int(s[3]), int(d->content_width), int(d->content_height),
                                     d->jitter_x, d->jitter_y, d->motion_scale_x, d->motion_scale_y);
    const float pre = d->pre_exposure > 0 ? d->pre_exposure : 1.0f;
    const uint8_t reset = (d->reset || c->bbrFrame == 0) ? 1 : 0;
    const float cpf[2] = {float(d->content_width) / s[0], float(d->content_height) / s[1]};
    p.bindBuffer("params", c->rt.transient(&bp, sizeof bp));
    p.bindBuffer("preexp", c->rt.transient(&pre, 4));
    p.bindBuffer("reset", c->rt.transient(&reset, 1));
    p.bindBuffer("cpf", c->rt.transient(cpf, 8));
    p.setParity(int(c->bbrFrame & 1));
    LayoutGuard g{cb};
    for (const m4r_image* im : {&d->color, &d->depth, &d->motion, &d->exposure, &d->output}) g.add(*im);
    g.begin();
    const bool ok = p.record(cb);
    c->vk.barrier(cb);
    g.end();
    if (!ok) { c->error = p.error; return 1; }
    ++c->bbrFrame;
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
    c->current = next;
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
