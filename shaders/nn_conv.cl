// OpenCL C port of the GLSL pass of the same name (see glsl_port.h).
// SPDX-License-Identifier: GPL-3.0-only
#include "glsl_port.h"
PORT_KERNEL {
    const uint Wo = a->c[0], Ho = a->c[1], Cin = a->c[2], Cout = a->c[3], Wi = a->c[4], Hi = a->c[5];
    const uint k = a->c[6] & 255u, st = (a->c[6] >> 8) & 255u, pad = a->c[6] >> 16, act = a->c[7];
    const uint px = GID_X, py = GID_Y, co0 = GID_Z * 8u;
    if (px >= Wo || py >= Ho || co0 >= Cout) return;
    global const half* src = BUF(const half, 0);
    global half* dst = BUF(half, 1);
    global const half* wt = BUF(const half, 2);
    global const half* bias = BUF(const half, 3);
    const uint n = g_minu(8u, Cout - co0);
    float acc[8];
    for (uint i = 0u; i < 8u; ++i) acc[i] = i < n ? (float)bias[co0 + i] : 0.0f;
    for (uint ky = 0u; ky < k; ++ky) {
        const int iy = (int)(py * st + ky) - (int)pad;
        if (iy < 0 || iy >= (int)Hi) continue;
        for (uint kx = 0u; kx < k; ++kx) {
            const int ix = (int)(px * st + kx) - (int)pad;
            if (ix < 0 || ix >= (int)Wi) continue;
            const uint ib = ((uint)iy * Wi + (uint)ix) * Cin, wb = (ky * k + kx) * Cin * Cout + co0;
            for (uint ci = 0u; ci < Cin; ++ci) {
                const float x = (float)src[ib + ci];
                const uint wo = wb + ci * Cout;
                for (uint i = 0u; i < 8u; ++i)
                    if (i < n) acc[i] = __builtin_fmaf(x, (float)wt[wo + i], acc[i]);
            }
        }
    }
    const uint ob = (py * Wo + px) * Cout + co0;
    for (uint i = 0u; i < n; ++i) {
        float v = acc[i];
        if (act == 1u) v = g_max(v, 0.0f);
        else if (act == 2u) v = __builtin_amdgcn_rcpf(1.0f + g_exp(-v));
        dst[ob + i] = (half)v;
    }
}
