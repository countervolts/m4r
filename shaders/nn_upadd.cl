// OpenCL C port of the GLSL pass of the same name (see glsl_port.h).
// SPDX-License-Identifier: GPL-3.0-only
#include "glsl_port.h"
PORT_KERNEL {
    const uint W = a->c[0], H = a->c[1], C4 = a->c[2] / 4u, S4 = (a->c[3] != 0u ? a->c[3] : a->c[2]) / 4u;
    const uint t = GID_X, px = t / C4, g = t % C4, py = GID_Y;
    if (px >= W || py >= H) return;
    global const half4v* src = BUF(const half4v, 0);
    global const half4v* skip = BUF(const half4v, 1);
    global half4v* dst = BUF(half4v, 2);
    const uint Wi = W / 2u, Hi = H / 2u;
    const float fx = (float)px * 0.5f - 0.25f, fy = (float)py * 0.5f - 0.25f;
    const float x0f = g_floor(fx), y0f = g_floor(fy), tx = fx - x0f, ty = fy - y0f;
    const int x0 = (int)x0f, y0 = (int)y0f;
    const uint xa = (uint)g_clampi(x0, 0, (int)Wi - 1), xb = (uint)g_clampi(x0 + 1, 0, (int)Wi - 1);
    const uint ya = (uint)g_clampi(y0, 0, (int)Hi - 1), yb = (uint)g_clampi(y0 + 1, 0, (int)Hi - 1);
    const float4v v00 = f4(src[(ya * Wi + xa) * C4 + g]), v10 = f4(src[(ya * Wi + xb) * C4 + g]);
    const float4v v01 = f4(src[(yb * Wi + xa) * C4 + g]), v11 = f4(src[(yb * Wi + xb) * C4 + g]);
    const float4v s = f4(skip[(py * W + px) * S4 + g]);
    float4v u;
    u.x = g_mix(g_mix(v00.x, v10.x, tx), g_mix(v01.x, v11.x, tx), ty) + s.x;
    u.y = g_mix(g_mix(v00.y, v10.y, tx), g_mix(v01.y, v11.y, tx), ty) + s.y;
    u.z = g_mix(g_mix(v00.z, v10.z, tx), g_mix(v01.z, v11.z, tx), ty) + s.z;
    u.w = g_mix(g_mix(v00.w, v10.w, tx), g_mix(v01.w, v11.w, tx), ty) + s.w;
    dst[(py * W + px) * C4 + g] = h4(u);
}
