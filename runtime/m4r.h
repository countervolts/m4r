/* m4r: MetalFX ML temporal upscaling and frame interpolation for Vulkan callers (OptiScaler backend, harnesses).
 *
 * The caller owns the Vulkan device; m4r records its passes into the caller's command buffer. Images are passed as
 * VkImage + format + size and must be in VK_IMAGE_LAYOUT_GENERAL when the recorded commands execute (vkd3d-proton
 * keeps D3D12 resources in GENERAL). Prepared plans (tools/prepare_bbr.py, tools/prepare_fg.py) are read from
 * <data dir>/bbr-<out w>x<out h>-<in w>x<in h> and <data dir>/fg-<out w>x<out h>-<in w>x<in h>.
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
    VkImage image;      /* VK_NULL_HANDLE = not provided */
    VkFormat format;
    uint32_t width, height;
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

/* get_instance_proc_addr resolves every Vulkan entry point on the caller's instance/device */
M4R_API int m4r_create(m4r_context** out, PFN_vkGetInstanceProcAddr get_instance_proc_addr, VkInstance instance,
                       VkPhysicalDevice physical, VkDevice device, uint32_t queue_family, const char* data_dir);
M4R_API void m4r_destroy(m4r_context* ctx);
/* 0 on success; a non-zero return means the configuration is not prepared or not supported (see m4r_last_error) */
M4R_API int m4r_upscale_init(m4r_context* ctx, uint32_t out_w, uint32_t out_h, uint32_t max_in_w, uint32_t max_in_h);
M4R_API int m4r_upscale(m4r_context* ctx, VkCommandBuffer cb, const m4r_upscale_desc* desc);
M4R_API int m4r_interpolate_init(m4r_context* ctx, uint32_t out_w, uint32_t out_h, uint32_t in_w, uint32_t in_h);
M4R_API int m4r_interpolate(m4r_context* ctx, VkCommandBuffer cb, const m4r_interpolate_desc* desc);
M4R_API const char* m4r_last_error(m4r_context* ctx);

#ifdef __cplusplus
}
#endif
