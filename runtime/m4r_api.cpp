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
    if (!p.record(cb)) { c->error = p.error; return 1; }
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
    if (!p.record(cb)) { c->error = p.error; return 1; }
    ++c->fgFrame;
    return 0;
}

} // extern "C"
