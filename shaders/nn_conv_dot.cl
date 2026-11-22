// m4r: NHWC fp16 convolution (zero padding) for GPUs without WMMA (RDNA2; any GPU with M4R_NN_DOT=1): the same layer as nn_conv_wmma.cl,
// as dot products. A lane is an output pixel (32 consecutive pixels of a row per wave) and accumulates NC output channels; the weights
// are the same for every lane, so they are scalar loads and the dot instruction reads them from SGPRs (v_dot2c_f32_f16 s, v: fp16
// pairs, fp32 accumulator, full rate, no cross-lane traffic). Every accumulator is the chain of dot2 over (tap, input channel block, k
// pair) in increasing order starting from 0, which is what v_wmma_f32_16x16x16_f16 computes (docs/mxu.md), so the output is
// bit-identical to nn_conv_wmma.cl. Zero-padded input channels are skipped (their products are exact zeros).
// -D: W H (output size) WI HI (input size) CIN COUT K ST (stride) PADV NC (output channels per wave, a multiple of 8 that divides COUT
//     rounded up to 8) MPX (output rows per lane) ACT (0 none, 1 relu, 2 sigmoid); built by tools/denoise_nets.py (dot_kernel)
// args: 0 input [HI][WI][CIN], 1 output [H][W][COUT], 2 weights packed as nn_conv_wmma.cl ([tap][cin block][cout block][16 cout][16 cin]),
//       3 bias [COUT]; launch: x = ceil(W/32) workgroups of 32, y = ceil(H/MPX), z = ceil(COUT/NC)
// SPDX-License-Identifier: GPL-3.0-only
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
typedef half half8v __attribute__((ext_vector_type(8)));
typedef uint uint4v __attribute__((ext_vector_type(4)));

#define CIB ((CIN + 15) / 16)
#define COB ((COUT + 15) / 16)
#define NPAIR(cb) ((CIN - (cb) * 16) >= 16 ? 8 : (CIN - (cb) * 16 + 1) / 2)   // k pairs of input channel block cb that are not padding

#include "nn_dot.h"

#ifndef MPX
#define MPX 1   // output rows per lane: each weight load used MPX times
#endif

__attribute__((amdgpu_flat_work_group_size(32, 32)))
kernel void nn_conv(global const half* src, global half* dst, constant const uint* wp, constant const half* bias) {
    const int l = __builtin_amdgcn_workitem_id_x();
    const int x = __builtin_amdgcn_workgroup_id_x() * 32 + l, y0 = __builtin_amdgcn_workgroup_id_y() * MPX;
    const int co0 = __builtin_amdgcn_workgroup_id_z() * NC;
    float acc[MPX][NC];
#pragma unroll
    for (int m = 0; m < MPX; ++m)
#pragma unroll
        for (int c = 0; c < NC; ++c) acc[m][c] = 0.0f;
    for (int ky = 0; ky < K; ++ky) {
        for (int kx = 0; kx < K; ++kx) {
            // loads from a clamped address, then zero where the tap is padding: no divergent branches in the loop
            const global half* px[MPX];
            bool ok[MPX];
#pragma unroll
            for (int m = 0; m < MPX; ++m) {
                const int yy = (y0 + m) * ST + ky - PADV, xx = x * ST + kx - PADV;
                ok[m] = yy >= 0 && yy < HI && xx >= 0 && xx < WI && x < W;
                const int yc = yy < 0 ? 0 : yy >= HI ? HI - 1 : yy, xc = xx < 0 ? 0 : xx >= WI ? WI - 1 : xx;
                px[m] = src + ((size_t)(yc * WI + xc)) * CIN;
            }
#pragma unroll
            for (int cb = 0; cb < CIB; ++cb) {
                uint a[MPX][8];
#pragma unroll
                for (int m = 0; m < MPX; ++m) {
#if CIN % 8 == 0
#pragma unroll
                    for (int q = 0; q < (NPAIR(cb) + 3) / 4; ++q) {
                        const uint4v v = *(const global uint4v*)(px[m] + cb * 16 + q * 8);
                        a[m][4 * q] = ok[m] ? v.x : 0u; a[m][4 * q + 1] = ok[m] ? v.y : 0u; a[m][4 * q + 2] = ok[m] ? v.z : 0u; a[m][4 * q + 3] = ok[m] ? v.w : 0u;
                    }
#else
#pragma unroll
                    for (int j = 0; j < NPAIR(cb); ++j) {
                        const int c = cb * 16 + 2 * j;
                        const uint h0 = __builtin_bit_cast(ushort, px[m][c]), h1 = c + 1 < CIN ? __builtin_bit_cast(ushort, px[m][c + 1]) : 0u;
                        a[m][j] = ok[m] ? (h0 | (h1 << 16)) : 0u;
                    }
#endif
                }
                constant const uint* wq = wp + ((size_t)(((ky * K + kx) * CIB + cb) * COB * 16 + co0)) * 8;
                DOT_BLOCK(acc, a, wq, NC, NPAIR(cb), MPX);
            }
        }
    }
    PIN(acc, NC, MPX);
#if COUT % 8
    // channel counts that are not a multiple of 8 (the UBF head, 44): one wave has all of them (NC >= COUT) and writes its 32 pixels as one
    // contiguous span through shared memory (lane-per-pixel stores of 2 bytes, COUT * 2 bytes apart, do not coalesce)
#if NC < COUT || COUT % 2
#error "COUT % 8 != 0 needs NC >= COUT and COUT even"
#endif
    local half stg[32 * COUT];
    const int x0 = x - l, npx = W - x0 < 32 ? W - x0 : 32;
#pragma unroll
    for (int m = 0; m < MPX; ++m) {
        const int y = y0 + m;
#pragma unroll
        for (int c = 0; c < COUT; ++c) {
            float v = acc[m][c] + (float)bias[c];
#if ACT == 1
            v = v > 0.0f ? v : 0.0f;
#elif ACT == 2
            v = 1.0f / (1.0f + native_exp(-v));
#endif
            stg[l * COUT + c] = (half)v;
        }
        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront"); __builtin_amdgcn_wave_barrier(); __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
        if (y < H) {
            global uint* q = (global uint*)(dst + ((size_t)(y * W + x0)) * COUT);
            for (int i = l; i < npx * COUT / 2; i += 32) q[i] = ((const local uint*)stg)[i];
        }
        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront"); __builtin_amdgcn_wave_barrier(); __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
    }
#else
    if (x >= W) return;
#pragma unroll
    for (int m = 0; m < MPX; ++m) {
        const int y = y0 + m;
        if (y >= H) continue;
        global half* q = dst + ((size_t)(y * W + x)) * COUT + co0;
        half o[NC];
#pragma unroll
        for (int c = 0; c < NC; ++c) {
            float v = acc[m][c] + (co0 + c < COUT ? (float)bias[co0 + c] : 0.0f);
#if ACT == 1
            v = v > 0.0f ? v : 0.0f;
#elif ACT == 2
            v = 1.0f / (1.0f + native_exp(-v));
#endif
            o[c] = (half)v;
        }
#pragma unroll
        for (int c8 = 0; c8 < NC / 8; ++c8)
            if (co0 + c8 * 8 < COUT)
                *(global half8v*)(q + c8 * 8) = (half8v)(o[c8 * 8], o[c8 * 8 + 1], o[c8 * 8 + 2], o[c8 * 8 + 3], o[c8 * 8 + 4], o[c8 * 8 + 5], o[c8 * 8 + 6], o[c8 * 8 + 7]);
    }
#endif
}
