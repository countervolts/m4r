// OpenCL C port of the GLSL pass of the same name (see glsl_port.h).
// SPDX-License-Identifier: GPL-3.0-only
#include "glsl_port.h"
static inline float3v unorm3(uint v) { return (float3v)((float)(v & 255u), (float)((v >> 8) & 255u), (float)((v >> 16) & 255u)) * (1.0f / 255.0f); }
PORT_KERNEL {
    const uint W = a->c[0], H = a->c[1], px = GID_X, py = GID_Y;
    if (px >= W || py >= H) return;
    const uint i = py * W + px;
    const float3v s = unorm3(BUF(const uint, 0)[i]) + unorm3(BUF(const uint, 1)[i]);
    BUF(half4v, 2)[i] = h4c((float3v)(g_clamp(s.x, 0.02f, 2.0f), g_clamp(s.y, 0.02f, 2.0f), g_clamp(s.z, 0.02f, 2.0f)), 1.0f);
}
