// OpenCL C port of the GLSL pass of the same name (see glsl_port.h).
// SPDX-License-Identifier: GPL-3.0-only
#include "glsl_port.h"
typedef struct { uint colorFmt, expFmt; } Par;
static inline float ufloat(uint bits, uint mb) {   // unsigned 11 / 10 bit floats (5 exponent bits)
    const uint m = bits & ((1u << mb) - 1u), e = bits >> mb;
    if (e == 0u) return g_div((float)m, (float)(1u << mb)) * __builtin_amdgcn_exp2f(-14.0f);
    if (e == 31u) return 65504.0f;
    return (1.0f + g_div((float)m, (float)(1u << mb))) * __builtin_amdgcn_exp2f((float)e - 15.0f);
}
PORT_KERNEL {
    const uint px = GID_X, py = GID_Y;
    if (px >= a->c[0] || py >= a->c[1]) return;
    const uint i = py * a->c[0] + px;
    global const Par* par = BUF(const Par, 2);
    const float e = par->expFmt == 1u ? (float)BUF(const half, 3)[0] : BUF(const float, 3)[0];
    if (!(e > 0.0f) && a->c[2] == 1u) return;
    const float scale = e > 0.0f ? e : 1.0f;
    if (a->c[2] == 0u) {
        float3v c;
        if (par->colorFmt == 1u) {
            const uint w = BUF(const uint, 0)[i];
            c = (float3v)(ufloat(w & 0x7ffu, 6u), ufloat((w >> 11) & 0x7ffu, 6u), ufloat((w >> 22) & 0x3ffu, 5u));
        } else {
            c = rgb(BUF(const half4v, 0)[i]);
        }
        BUF(half4v, 1)[i] = h4c((float3v)(g_clamp(c.x * scale, 0.0f, 65504.0f), g_clamp(c.y * scale, 0.0f, 65504.0f), g_clamp(c.z * scale, 0.0f, 65504.0f)), 1.0f);
    } else {
        const float4v v = f4(BUF(const half4v, 0)[i]);
        BUF(half4v, 1)[i] = h4c((float3v)(g_clamp(g_div(v.x, scale), 0.0f, 65504.0f), g_clamp(g_div(v.y, scale), 0.0f, 65504.0f),
                                          g_clamp(g_div(v.z, scale), 0.0f, 65504.0f)), v.w);
    }
}
