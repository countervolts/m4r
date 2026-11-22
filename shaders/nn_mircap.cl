// OpenCL C port of the GLSL pass of the same name (see glsl_port.h).
// SPDX-License-Identifier: GPL-3.0-only
#include "glsl_port.h"
PORT_KERNEL {
    const uint Wt = a->c[0], Wv = a->c[2], Hv = a->c[3], iw = a->c[4], ih = a->c[5], px = GID_X, py = GID_Y;
    if (px >= Wv || py >= Hv) return;
    global const float* g = BUF(const float, 2);   /* translation gate, cap */
    if (!(g[0] > 0.0f)) return;
    global half* t = BUF(half, 0);
    global const half* rough = BUF(const half, 1);
    const uint rx = g_minu(iw - 1u, (uint)g_div(((float)px + 0.5f) * (float)iw, (float)Wv));
    const uint ry = g_minu(ih - 1u, (uint)g_div(((float)py + 0.5f) * (float)ih, (float)Hv));
    const float s = (1.0f - g_smoothstep(0.08f, 0.20f, (float)rough[ry * iw + rx])) * g[0];
    if (!(s > 0.0f)) return;
    const float cap = g_mix(1.0f, g[1], s);
    const uint b = (py * Wt + px) * 11u;
    for (uint c = 3u; c < 7u; ++c) t[b + c] = (half)g_min((float)t[b + c], cap);
}
