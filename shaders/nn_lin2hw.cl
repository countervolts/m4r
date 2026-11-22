// OpenCL C port of the GLSL pass of the same name (see glsl_port.h).
// SPDX-License-Identifier: GPL-3.0-only
#include "glsl_port.h"
typedef struct { float nearZ, m23, m33; uint isFloat; float sky, m22, m32; } Par;
PORT_KERNEL {
    const uint px = GID_X, py = GID_Y;
    if (px >= a->c[0] || py >= a->c[1]) return;
    const uint i = py * a->c[0] + px;
    global const Par* par = BUF(const Par, 2);
    const float lin = par->isFloat != 0u ? BUF(const float, 0)[i] : (float)BUF(const half, 0)[i];
    float d = 0.0f;
    if (a->c[2] == 1u) {
        d = g_clamp(par->m22, 0.0f, 1.0f);
        if (lin > 0.0f && lin < par->sky) {
            const float w = __builtin_fmaf(lin, par->m23, par->m33);
            if (w > 0.0f) d = g_clamp(g_div(__builtin_fmaf(lin, par->m22, par->m32), w), 0.0f, 1.0f);
        }
    } else if (lin > 0.0f && lin < par->sky) {
        const float w = __builtin_fmaf(lin, par->m23, par->m33);
        if (w > 0.0f) d = g_div(par->nearZ, w);
    }
    BUF(float, 1)[i] = d;
}
