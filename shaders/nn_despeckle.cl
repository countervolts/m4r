// OpenCL C port of the GLSL pass of the same name (see glsl_port.h).
// SPDX-License-Identifier: GPL-3.0-only
#include "glsl_port.h"
PORT_KERNEL {
    const int W = (int)a->c[0], H = (int)a->c[1], px = (int)GID_X, py = (int)GID_Y;
    if (px >= W || py >= H) return;
    global const half4v* src = BUF(const half4v, 0);
    global half4v* dst = BUF(half4v, 1);
    const float4v c = f4(src[py * W + px]);
    float t0 = 0.0f, t1 = 0.0f, t2 = 0.0f;
    for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx) {
            if (dx == 0 && dy == 0) continue;
            const int qx = g_clampi(px + dx, 0, W - 1), qy = g_clampi(py + dy, 0, H - 1);
            const float l = lum3(rgb(src[qy * W + qx]));
            if (l > t0) { t2 = t1; t1 = t0; t0 = l; }
            else if (l > t1) { t2 = t1; t1 = l; }
            else if (l > t2) t2 = l;
        }
    const float l = lum3(c.xyz), cap = __builtin_fmaf(t2, 1.5f, 1e-4f);
    const float s = l > cap ? g_div(cap, l) : 1.0f;
    dst[py * W + px] = l > cap ? h4c((float3v)(c.x * s, c.y * s, c.z * s), c.w) : h4(c);
}
