// OpenCL C port of the GLSL pass of the same name (see glsl_port.h).
// SPDX-License-Identifier: GPL-3.0-only
#include "glsl_port.h"
PORT_KERNEL {
    const int W = (int)a->c[0], H = (int)a->c[1], iw = (int)a->c[2], ih = (int)a->c[3], px = (int)GID_X, py = (int)GID_Y;
    if (px >= W || py >= H) return;
    global const half4v* v3 = BUF(const half4v, 0);
    global half4v* bb = BUF(half4v, 1);
    global const half* rough = BUF(const half, 2);
    float3v lo = (float3v)(65504.0f), hi = (float3v)(-65504.0f);
    for (int dy = -2; dy <= 2; ++dy)
        for (int dx = -2; dx <= 2; ++dx) {
            const int qx = g_clampi(px + dx, 0, W - 1), qy = g_clampi(py + dy, 0, H - 1);
            const float3v c = rgb(v3[qy * W + qx]);
            lo = (float3v)(g_min(lo.x, c.x), g_min(lo.y, c.y), g_min(lo.z, c.z));
            hi = (float3v)(g_max(hi.x, c.x), g_max(hi.y, c.y), g_max(hi.z, c.z));
        }
    const float3v m = (hi - lo) * 0.25f;
    const int i = py * W + px;
    const float4v b = f4(bb[i]);
    const float3v bc = (float3v)(g_clamp(b.x, lo.x - m.x, hi.x + m.x), g_clamp(b.y, lo.y - m.y, hi.y + m.y), g_clamp(b.z, lo.z - m.z, hi.z + m.z));
    const int rx = g_clampi((int)g_div(((float)px + 0.5f) * (float)iw, (float)W), 0, iw - 1);
    const int ry = g_clampi((int)g_div(((float)py + 0.5f) * (float)ih, (float)H), 0, ih - 1);
    float r = 0.0f;
    for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx)
            r += (float)rough[g_clampi(ry + dy, 0, ih - 1) * iw + g_clampi(rx + dx, 0, iw - 1)];
    const float wb = g_smoothstep(0.10f, 0.25f, r * (1.0f / 9.0f)) * g_clamp(BUF(const float, 3)[0], 0.0f, 1.0f);
    bb[i] = h4c(mix3(rgb(v3[i]), bc, wb), b.w);
}
