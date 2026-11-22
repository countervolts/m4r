// OpenCL C port of the GLSL pass of the same name (see glsl_port.h).
// SPDX-License-Identifier: GPL-3.0-only
#include "glsl_port.h"
PORT_KERNEL {
    const uint W = a->c[0], H = a->c[1], C4 = a->c[2] / 4u;
    const uint t = GID_X, px = t / C4, g = t % C4, py = GID_Y;
    if (px >= W || py >= H) return;
    global const half4v* src = BUF(const half4v, 0);
    global half4v* dst = BUF(half4v, 1);
    const uint Wi = W * 2u, i0 = ((py * 2u) * Wi + px * 2u) * C4 + g;
    const float4v v0 = f4(src[i0]), v1 = f4(src[i0 + C4]), v2 = f4(src[i0 + Wi * C4]), v3 = f4(src[i0 + Wi * C4 + C4]);
    float4v m;
    m.x = g_max(g_max(v0.x, v1.x), g_max(v2.x, v3.x)); m.y = g_max(g_max(v0.y, v1.y), g_max(v2.y, v3.y));
    m.z = g_max(g_max(v0.z, v1.z), g_max(v2.z, v3.z)); m.w = g_max(g_max(v0.w, v1.w), g_max(v2.w, v3.w));
    dst[(py * W + px) * C4 + g] = h4(m);
}
