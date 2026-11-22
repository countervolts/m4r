// OpenCL C port of the GLSL pass of the same name (see glsl_port.h).
// SPDX-License-Identifier: GPL-3.0-only
#include "glsl_port.h"
static inline float3v nz(float3v v) { return v * g_rsqrt(g_max(v.x * v.x + v.y * v.y + v.z * v.z, 1e-12f)); }
PORT_KERNEL {
    const int W = (int)a->c[0], H = (int)a->c[1], STEP = (int)a->c[2], px = (int)GID_X, py = (int)GID_Y;
    if (px >= W || py >= H) return;
    global const half4v* src = BUF(const half4v, 0);
    global half4v* dst = BUF(half4v, 1);
    global const half4v* nrm = BUF(const half4v, 2);
    global const float* dep = BUF(const float, 3);
    global const half2v* mv = BUF(const half2v, 4);
    global const float2v* ms = BUF(const float2v, 5);
    const int i = py * W + px;
    const float4v c = f4(src[i]), np = f4(nrm[i]);
    const float r = np.w;
    const float mx = (float)mv[i].x * ms->x, my = (float)mv[i].y * ms->y;
    const float s = (1.0f - g_smoothstep(0.15f, 0.30f, r)) * g_smoothstep(1.0f, 3.0f, g_sqrt(mx * mx + my * my));
    if (!(s > 0.0f) || !(dep[i] > 0.0f)) { dst[i] = h4(c); return; }
    const float3v n = nz(np.xyz);
    const float z = dep[i];
    float ref = 0.0f;
    for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx) {
            const int qx = g_clampi(px + dx, 0, W - 1), qy = g_clampi(py + dy, 0, H - 1);
            ref += g_log(lum3(rgb(src[qy * W + qx])) + 1e-3f);
        }
    ref *= 1.0f / 9.0f;
    const float B[5] = {1.0f / 16.0f, 4.0f / 16.0f, 6.0f / 16.0f, 4.0f / 16.0f, 1.0f / 16.0f};
    float3v acc = (float3v)(0.0f);
    float wsum = 0.0f;
    for (int ky = 0; ky < 5; ++ky)
        for (int kx = 0; kx < 5; ++kx) {
            const int qx = g_clampi(px + (kx - 2) * STEP, 0, W - 1), qy = g_clampi(py + (ky - 2) * STEP, 0, H - 1);
            const int j = qy * W + qx;
            const float3v cq = rgb(src[j]);
            const float4v nq = f4(nrm[j]);
            const float lq = g_log(lum3(cq) + 1e-3f) - ref;
            const float3v nn = nz(nq.xyz);
            float w = B[ky] * B[kx] * g_pow(g_max(n.x * nn.x + n.y * nn.y + n.z * nn.z, 0.0f), 32.0f);
            w *= g_exp(-g_div(g_abs(1.0f - g_div(dep[j], z)), 0.01f * (float)STEP) - g_div(g_abs(nq.w - r), 0.03f) - 0.5f * (lq * lq));
            acc += w * cq; wsum += w;
        }
    const float3v f = wsum > 1e-8f ? (float3v)(g_div(acc.x, wsum), g_div(acc.y, wsum), g_div(acc.z, wsum)) : c.xyz;
    dst[i] = h4c(mix3(c.xyz, f, s), c.w);
}
