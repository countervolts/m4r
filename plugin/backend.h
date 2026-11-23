// The upscaler behind the OptiScaler plugin (asi.cpp, framegen.cpp): m4r, MetalFX. The plugin's other files are shared with
// anf4r (github.com/countervolts/anf4r, plugin/), whose copy of this header names anf4r's API; the two have the same shape.
// SPDX-License-Identifier: GPL-3.0-only
#pragma once
#include "m4r.h"

#define BE_NAME "m4r"                  // <plugin dir>\m4r\ holds the data, the log is m4r.log there
#define BE_DATA "plans"                // the backend's data dir under it (be_create's last argument)
#define BE_UPSCALER "MetalFX"          // provider names in OptiScaler's lists
#define BE_FG "MetalFX FG"
#define BE_VERSION_ID 0x6d34720001ull  // "m4r" 1
#define BE_VERSION_NAME "1.0.0 (MetalFX, m4r)"

typedef m4r_context be_context;
typedef m4r_upscale_desc be_upscale_desc;
typedef m4r_interpolate_desc be_interpolate_desc;
#define be_create m4r_create
#define be_destroy m4r_destroy
#define be_last_error m4r_last_error
#define be_upscale_init m4r_upscale_init
#define be_upscale m4r_upscale
#define be_interpolate_init m4r_interpolate_init
#define be_interpolate m4r_interpolate
// Ray Reconstruction (rr.cpp): MetalFX's denoiser and its upscaling network
#define BE_RR "MetalFX"
typedef m4r_denoise_desc be_denoise_desc;
#define be_denoise_init m4r_denoise_init
#define be_denoise m4r_denoise
