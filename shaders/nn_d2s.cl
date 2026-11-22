// OpenCL C port of the GLSL pass of the same name (see glsl_port.h).
// SPDX-License-Identifier: GPL-3.0-only
#include "glsl_port.h"
PORT_KERNEL {
    const uint W = a->c[0], H = a->c[1], qx = GID_X, qy = GID_Y;
    if (qx >= 2u * W || qy >= 2u * H) return;
    global const half* src = BUF(const half, 0);
    global half* dst = BUF(half, 1);
    const uint sb = ((qy >> 1) * W + (qx >> 1)) * 44u + (qy & 1u) * 2u + (qx & 1u);
    const uint db = (qy * (W * 2u) + qx) * 11u;
    const bool sig = a->c[2] != 0u;
    for (uint c = 0u; c < 11u; ++c) {
        const float v = (float)src[sb + c * 4u];
        dst[db + c] = (half)(sig ? __builtin_amdgcn_rcpf(1.0f + g_exp(-v)) : v);
    }
}
