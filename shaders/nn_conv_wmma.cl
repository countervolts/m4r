// m4r: NHWC fp16 convolution (stride 1, zero padding) for MetalFX's DBFNet as an implicit GEMM on RDNA3 WMMA
// (v_wmma_f32_16x16x16_f16: fp16 inputs, fp32 accumulation, like nn_conv.comp). One wave computes 16 consecutive pixels of a row for
// NT tiles of 16 output channels. Specialised per layer with -D W H CIN COUT K NT RELU; built by tools/denoise_nets.py (wmma_conv).
// args: 0 input [H][W][CIN], 1 output [H][W][COUT], 2 weights packed [tap][cin block][cout block][16 cout][16 cin] (zero padded),
//       3 bias [COUT]; launch: x = ceil(W/16) workgroups of 32, y = H, z = ceil(ceil(COUT/16)/NT)
// SPDX-License-Identifier: GPL-3.0-only
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
typedef half half16 __attribute__((ext_vector_type(16)));
typedef float float8 __attribute__((ext_vector_type(8)));

#define PAD (K / 2)
#define CIB ((CIN + 15) / 16)
#define COB ((COUT + 15) / 16)

__attribute__((amdgpu_flat_work_group_size(32, 32)))
kernel void nn_conv(global const half* src, global half* dst, global const half* wp, global const half* bias) {
    const uint l = __builtin_amdgcn_workitem_id_x();
    const int tx = __builtin_amdgcn_workgroup_id_x(), y = __builtin_amdgcn_workgroup_id_y(), ng = __builtin_amdgcn_workgroup_id_z();
    const int n = l & 15;
    const int x = tx * 16 + n;   // A row = pixel; D row of accumulator i is pixel 2 * i + (l >> 4)
    float8 acc[NT];
#pragma unroll
    for (int t = 0; t < NT; ++t) acc[t] = (float8)(0.0f);
#pragma unroll
    for (int ky = 0; ky < K; ++ky) {
#pragma unroll
        for (int kx = 0; kx < K; ++kx) {
            const int yy = y + ky - PAD, xx = x + kx - PAD;
            const bool ok = yy >= 0 && yy < H && xx >= 0 && xx < W;
            global const half* px = src + ((size_t)(yy * W + xx)) * CIN;
#pragma unroll
            for (int cb = 0; cb < CIB; ++cb) {
                half16 a = (half16)(0.0h);
                if (ok) {
#if CIN % 16 == 0
                    a = *(global const half16*)(px + cb * 16);
#else
#pragma unroll
                    for (int c = 0; c < 16; ++c)
                        if (cb * 16 + c < CIN) a[c] = px[cb * 16 + c];
#endif
                }
#pragma unroll
                for (int t = 0; t < NT; ++t) {
                    const int cob = ng * NT + t;
                    if (cob < COB) {
                        const half16 b = *(global const half16*)(wp + ((size_t)((((ky * K + kx) * CIB + cb) * COB + cob) * 16 + n)) * 16);
                        acc[t] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, acc[t]);
                    }
                }
            }
        }
    }
#pragma unroll
    for (int t = 0; t < NT; ++t) {
        const int co = (ng * NT + t) * 16 + n;
        if (co < COUT) {
            const float bs = (float)bias[co];
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int p = tx * 16 + 2 * i + (l >> 4);
                if (p < W) {
                    float v = acc[t][i] + bs;
#if RELU
                    v = v > 0.0f ? v : 0.0f;
#endif
                    dst[((size_t)(y * W + p)) * COUT + co] = (half)v;
                }
            }
        }
    }
}
