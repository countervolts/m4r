/* m4r: MetalFX ML temporal upscaling and frame interpolation for Vulkan callers (OptiScaler backend, harnesses).
 *
 * The caller owns the Vulkan device; m4r records its passes into the caller's command buffer. Images are passed as
 * VkImage + format + size + current layout; m4r transitions them to GENERAL around its passes. Prepared plans (tools/prepare_bbr.py, tools/prepare_fg.py) are read from
 * <data dir>/bbr-<out w>x<out h>-<in w>x<in h>[-fwd] and <data dir>/fg-<out w>x<out h>-<in w>x<in h>. The upscaler plan is specialised
 * for the depth convention: the plain name is for reversed depth, `-fwd` for forward depth (m4r_upscale_desc.depth_reversed).
 * SPDX-License-Identifier: GPL-3.0-only */
#pragma once
#include <stdint.h>
#ifndef VK_NO_PROTOTYPES
#define VK_NO_PROTOTYPES
#endif
#include <vulkan/vulkan.h>

#ifdef _WIN32
#define M4R_API __declspec(dllexport)
#else
#define M4R_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct m4r_context m4r_context;

typedef struct m4r_image {
    VkImage image;          /* VK_NULL_HANDLE = not provided */
    VkFormat format;
    uint32_t width, height;
    VkImageLayout layout;   /* layout at the time the commands run; m4r moves it to GENERAL and back.
                               VK_IMAGE_LAYOUT_UNDEFINED is treated as GENERAL */
} m4r_image;

typedef struct m4r_upscale_desc {
    m4r_image color, depth, motion, exposure, output;
    uint32_t content_width, content_height;  /* rendered area of color/depth/motion */
    float jitter_x, jitter_y;                /* sub-pixel offset the frame was rendered with (pixels) */
    float motion_scale_x, motion_scale_y;    /* converts motion texture values to pixels (previous - current) */
    float pre_exposure;
    int reset, depth_reversed;
} m4r_upscale_desc;

typedef struct m4r_interpolate_desc {
    m4r_image color, prev_color, depth, motion, output;  /* color/output at output size; depth/motion at input size */
    float motion_scale_x, motion_scale_y;                /* to input pixels, previous - current */
    float delta_time, near_plane, far_plane;
    int depth_reversed, reset;
} m4r_interpolate_desc;

/* Ray Reconstruction (NGX feature 13) inputs, as the game gives them. Matrices are row-major float[16], row-vector convention
 * (v' = v * M) as NGX uses them. Images that the game does not provide are VK_NULL_HANDLE. */
typedef struct m4r_denoise_desc {
    m4r_image color, depth, motion, diffuse_albedo, specular_albedo, normals, roughness, specular_hit, diffuse_hit,
              color_before_particles, exposure, output;
    float world_to_view[16], view_to_clip[16];
    uint32_t content_width, content_height;
    float jitter_x, jitter_y, motion_scale_x, motion_scale_y, pre_exposure;
    int reset, depth_reversed, hardware_depth, roughness_in_normals_w, motion_low_res, specular_hit_in_w;
} m4r_denoise_desc;

/* Frame generation at present time. The finished frame (backbuffer, display size) and this frame's render-size depth
 * and motion vectors go in; m4r keeps a copy of the frame and, once it has the previous one, writes the MetalFX
 * in-between frame into the backbuffer. */
typedef struct m4r_framegen_desc {
    m4r_image backbuffer, depth, motion;
    float motion_scale_x, motion_scale_y;   /* to render pixels, previous - current */
    float delta_time, near_plane, far_plane;
    int depth_reversed, reset;
} m4r_framegen_desc;

/* get_instance_proc_addr resolves every Vulkan entry point on the caller's instance/device */
M4R_API int m4r_create(m4r_context** out, PFN_vkGetInstanceProcAddr get_instance_proc_addr, VkInstance instance,
                       VkPhysicalDevice physical, VkDevice device, uint32_t queue_family, const char* data_dir);
M4R_API void m4r_destroy(m4r_context* ctx);
/* 0 on success; a non-zero return means the configuration is not prepared or not supported (see m4r_last_error) */
M4R_API int m4r_upscale_init(m4r_context* ctx, uint32_t out_w, uint32_t out_h, uint32_t max_in_w, uint32_t max_in_h);
M4R_API int m4r_upscale(m4r_context* ctx, VkCommandBuffer cb, const m4r_upscale_desc* desc);
/* M4R_DUMP_RR=<dir> (+ M4R_DUMP_RR_COUNT, default 6 frames): creating the file <dir>/trigger starts a capture; writes the raw texels of the Ray Reconstruction inputs and the
 * frame's parameters (rr-NN-<name>.raw, rr-NN.txt) for offline work on the denoiser. Does nothing when unset. 0 = ok. */
M4R_API int m4r_denoise_capture(m4r_context* ctx, VkCommandBuffer cb, const m4r_denoise_desc* desc);
M4R_API int m4r_interpolate_init(m4r_context* ctx, uint32_t out_w, uint32_t out_h, uint32_t in_w, uint32_t in_h);
M4R_API int m4r_interpolate(m4r_context* ctx, VkCommandBuffer cb, const m4r_interpolate_desc* desc);
/* Returns 1 when the generated frame was written to desc->backbuffer: present it, then call m4r_framegen_restore with
 * the next backbuffer to put the real frame back. 0 = nothing generated (first frame or reset), the backbuffer is
 * unchanged. Negative = error (m4r_last_error). Loads the fg plan for the backbuffer and depth sizes on first use. */
M4R_API int m4r_framegen_capture(m4r_context* ctx, VkCommandBuffer cb, const m4r_framegen_desc* desc);
/* writes the frame given to the last m4r_framegen_capture into backbuffer */
M4R_API int m4r_framegen_restore(m4r_context* ctx, VkCommandBuffer cb, const m4r_image* backbuffer);
M4R_API const char* m4r_last_error(m4r_context* ctx);

#ifdef __cplusplus
}
#endif
