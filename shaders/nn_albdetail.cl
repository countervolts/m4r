// OpenCL C port of the GLSL pass of the same name (see glsl_port.h).
// SPDX-License-Identifier: GPL-3.0-only
#include "glsl_port.h"
#define R 3
PORT_KERNEL {
    const int W = (int)a->c[0], H = (int)a->c[1], px = (int)GID_X, py = (int)GID_Y;
    if (px >= W || py >= H) return;
    global half4v* col = BUF(half4v, 0);
    global const half4v* alb = BUF(const half4v, 1);
    const int i = py * W + px;
    const float ac = lum3(rgb(alb[i]));
    if (!(ac < 1.5f)) return;
    float num = 0.0f, den = 0.0f, wsum = 0.0f;
    for (int dy = -R; dy <= R; ++dy)
        for (int dx = -R; dx <= R; ++dx) {
            const int qx = g_clampi(px + dx, 0, W - 1), qy = g_clampi(py + dy, 0, H - 1);
            const float w = g_exp(-0.5f * (float)(dx * dx + dy * dy));
            const float aq = lum3(rgb(alb[qy * W + qx]));
            const float v = aq < 1.5f ? 1.0f : 0.0f;
            num = __builtin_fmaf(w * aq, v, num); den = __builtin_fmaf(w, v, den); wsum += w;
        }
    const float blur = g_div(num, g_max(den, 1e-3f));
    float ratio = g_clamp(g_div(ac, g_max(blur, 1e-3f)), 0.7f, 1.4f);
    ratio = __builtin_fmaf(ratio - 1.0f, g_clamp(g_div(g_div(den, wsum) - 0.85f, 0.15f), 0.0f, 1.0f), 1.0f);
    const float4v c = f4(col[i]);
    col[i] = h4c((float3v)(c.x * ratio, c.y * ratio, c.z * ratio), c.w);
}
