// OpenCL C port of the GLSL pass of the same name (see glsl_port.h).
// SPDX-License-Identifier: GPL-3.0-only
#include "glsl_port.h"
typedef struct { float r0, r1, strength; } Par;
PORT_KERNEL {
    const int W = (int)a->c[0], H = (int)a->c[1], px = (int)GID_X, py = (int)GID_Y;
    if (px >= W || py >= H) return;
    global const Par* par = BUF(const Par, 3);
    global const half* rough = BUF(const half, 2);
    float r = 0.0f;
    for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx)
            r += (float)rough[g_clampi(py + dy, 0, H - 1) * W + g_clampi(px + dx, 0, W - 1)];
    r = g_div(r, 9.0f);
    const float t = g_clamp(g_div(r - par->r0, g_max(par->r1 - par->r0, 1e-4f)), 0.0f, 1.0f);
    const float mirror = (1.0f - t * t * (3.0f - 2.0f * t)) * par->strength;
    if (mirror <= 0.0f) return;
    const int i = py * W + px;
    const float4v f = f4(BUF(const half4v, 0)[i]), g = f4(BUF(const half4v, 1)[i]);
    BUF(half4v, 0)[i] = h4c(mix3(f.xyz, g.xyz, mirror), f.w);
}
