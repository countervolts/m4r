// m4r: one DBFNet cell per dispatch on RDNA3 WMMA (v_wmma_f32_16x16x16_f16: fp16 inputs, fp32 accumulation), NHWC fp16:
//   input transform (none / 2x2 max pool / bilinear x2 upsample + skip) -> 3x3 conv + relu -> 1x1 conv (+ relu) [-> 1x1 conv]
// Same arithmetic as nn_pool.comp + nn_upadd.comp + nn_conv_wmma.cl in sequence (every accumulator sees the same WMMAs in the same
// order: taps, then input-channel blocks; intermediates rounded to fp16 where those kernels stored them), so the output is bit-identical,
// without the intermediate tensors in memory and in 1 dispatch instead of 3-5.
// A workgroup computes a TX x TY tile of output pixels: the transformed input tile with a 1 pixel border goes to shared memory once,
// each wave computes MP groups of 16 pixels of a row for all output channels; the 3x3 result goes through shared memory (the WMMA
// result layout is pixel-per-row, the next WMMA wants pixel-per-lane) into the 1x1, whose result goes out through shared memory as
// 16-byte stores.
// -D: W H (cell size), MODE (0 plain: input [H][W] with SIN halfs per pixel; 1 pool: input [2H][2W][CIN]; 2 upadd: input [H/2][W/2][CIN]
//     + skip [H][W][CIN]), CIN, C1 (3x3 out), C2 (1x1 out), C3 (second 1x1 out, 0 none), SOUT (halfs per output pixel), TX TY MP,
//     UPMIX (formula of the bilinear blend; 1 = RADV's: fma(t, b - a, a)), built by tools/denoise_nets.py (cell_kernel)
// args: 0 in, 1 skip, 2 out, 3 w1, 4 b1, 5 w2, 6 b2, 7 w3, 8 b3 (weights packed as nn_conv_wmma.cl: [tap][cin block][cout block][16][16])
// launch: x = ceil(W/TX) * (TX/16*TY/MP) * 32 threads, y = ceil(H/TY)
// DOT=1 (GPUs without WMMA: RDNA2, or any GPU with M4R_NN_DOT=1): the convolutions as dot products, lane = pixel (TX = 32: wave = one
//     MPX rows of the tile, TY / MPX such waves, WC waves per row group splitting the output channels), weights as scalar loads used
//     by MPX pixels per lane (shaders/nn_dot.h). Each accumulator is the chain of dot2 over (tap, channel block, k pair) that the WMMA
//     path computes, so the output is the same bits. launch: x = ceil(W/32) * TY / MPX * WC * 32 threads, y = ceil(H/TY)
// SPDX-License-Identifier: GPL-3.0-only
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma clang fp contract(off)
typedef half half16 __attribute__((ext_vector_type(16)));
typedef half half8v __attribute__((ext_vector_type(8)));
typedef float float8 __attribute__((ext_vector_type(8)));

#define CIB ((CIN + 15) / 16)
#define CINP (CIB * 16)
#define CB1 ((C1 + 15) / 16)
#define CB2 ((C2 + 15) / 16)
#define CB3 ((C3 + 15) / 16)
#define GX (TX / 16)
#ifndef WC
#define WC 1                           // waves per pixel group, splitting the output channel blocks (small cells: more waves in flight)
#endif
#ifndef DOT
#define DOT 0
#endif
#ifndef SOFTW
#define SOFTW 0                        // software WMMA (DPP dot products) for GPUs without WMMA, see WMMA_T
#endif
#if DOT
#ifndef MPX
#define MPX 1                          // DOT: pixels per lane (rows of the tile), each weight load used MPX times
#endif
#define PWAVES (TY / MPX)              // TX = 32: one wave per MPX tile rows
#else
#define PWAVES (GX * TY / MP)
#endif
#define WAVES (PWAVES * WC)
#define NTH (WAVES * 32)
#define TW (TX + 2)
#define TH (TY + 2)
#define PS (CINP + 8)                  // halfs per pixel in the input tile (16 bytes of padding against bank conflicts)
#define MAXC (C1 > C2 ? (C1 > C3 ? C1 : C3) : (C2 > C3 ? C2 : C3))
#define MPS (((MAXC + 15) / 16) * 16 + 8)   // halfs per pixel of the shared-memory staging of results
#define TILE_HALFS (TH * TW * PS)
#define MID_HALFS (TX * TY * MPS)
#if DOT
#define XPS (C1 + 8)                   // halfs per pixel of the channel-split exchange of the 3x3 result
#define XCH_DOT_HALFS ((MPX > 1 || WC > 1) ? TX * TY * XPS : 0)   // the exchange reuses the input tile
#define LDS_HALFS (TILE_HALFS > XCH_DOT_HALFS ? TILE_HALFS : XCH_DOT_HALFS)
#elif LDSSTAGE
#define LDS_HALFS (TILE_HALFS > MID_HALFS ? TILE_HALFS : MID_HALFS)
#else
#define XCH_HALFS (WC > 1 ? PWAVES * MP * CB1 * 256 : 0)   // channel-split exchange of the 3x3 result (pixel-major blocks)
#define LDS_HALFS (TILE_HALFS + XCH_HALFS)
#endif

#ifndef TAPU
#define TAPU 1                         // unroll of the 3x3 tap loop (full unrolling hoists every weight load: 250 VGPRs and spills)
#endif
#ifndef CBU
#define CBU 1                          // unroll of the input-channel-block loops
#endif
#ifndef PIPE
#define PIPE 1                         // software-pipelined 3x3 weight loads
#endif
#ifndef PIPEU
#define PIPEU 4                        // unroll of the pipelined step loop (6: spills on the 64-channel decoder)
#endif
#ifndef FILLU
#define FILLU 8                        // unroll of the fill's channel-chunk loop (8: all chunks of a 64-channel pixel in flight)
#endif
#ifndef UPMIX
#define UPMIX 1
#endif
// u + s rounded once, straight to fp16: what RADV compiles nn_upadd.comp's f16vec4(u + vec4(skip)) to (v_fma_mixlo_f16 1.0, u, s);
// (half)(u + s) rounds twice (to fp32, then fp16) and differs in rare ties
// the same for two channels at once into one packed word (v_fma_mixlo_f16 + v_fma_mixhi_f16, the halves of s2 as the skips)
typedef uint uint4v __attribute__((ext_vector_type(4)));
static inline uint add_round_once2(float u0, float u1, uint s2) {
    uint r;
    __asm("v_fma_mixlo_f16 %0, 1.0, %1, %3 op_sel_hi:[0,0,1]\n\tv_fma_mixhi_f16 %0, 1.0, %2, %3 op_sel:[0,0,1] op_sel_hi:[0,0,1]"
          : "=&v"(r) : "v"(u0), "v"(u1), "v"(s2));
    return r;
}
static inline half add_round_once(float u, half s) {
    uint r = 0;
    const uint sb = (uint)__builtin_bit_cast(ushort, s);
    __asm volatile("v_fma_mixlo_f16 %0, 1.0, %1, %2 op_sel_hi:[0,0,1]" : "+v"(r) : "v"(u), "v"(sb));
    return __builtin_bit_cast(half, (ushort)(r & 0xffff));
}
#if C3 > 0
#define COUTN C3
#else
#define COUTN C2
#endif
#ifdef EXP_SAMEW
#define EXP_TAPIDX 0
#define EXP_CBIDX 0
#else
#define EXP_TAPIDX tap
#define EXP_CBIDX cb
#endif
#ifndef LDSSTAGE
#define LDSSTAGE 0
#endif
#define BARRIER() do { __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup"); __builtin_amdgcn_s_barrier(); __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup"); } while (0)
static inline int clampi(int v, int lo, int hi) { return v < lo ? lo : v > hi ? hi : v; }
static inline float lerpf(float a, float b, float t) {
#if UPMIX == 0
    return a * (1.0f - t) + b * t;
#elif UPMIX == 1
    return __builtin_fmaf(t, b - a, a);
#elif UPMIX == 2
    return a + t * (b - a);
#else
    return __builtin_fmaf(a, 1.0f - t, b * t);
#endif
}

// one pixel's 16-channel block of a WMMA result (lane: channel l & 15, pixels 2i + (l >> 4)) after bias and activation into the staging
static inline void stage(local half* mid, int j, float8 acc, const global half* bias, int cb, int cout, int relu) {
    const uint l = __builtin_amdgcn_workitem_id_x() & 31;
    const int co = cb * 16 + (l & 15);
    const float bs = co < cout ? (float)bias[co] : 0.0f;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        float v = acc[i] + bs;
        if (relu) v = v > 0.0f ? v : 0.0f;
        mid[(j * 16 + 2 * i + (l >> 4)) * MPS + co] = (half)v;
    }
}

// transposed result (weights as A, pixels as B): lane = pixel l & 15, rows = channels 2i + (l >> 4)
static inline void stage_t(local half* mid, int j, float8 acc, const global half* bias, int cb, int cout, int relu) {
    const uint l = __builtin_amdgcn_workitem_id_x() & 31;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int co = cb * 16 + 2 * i + (l >> 4);
        float v = acc[i] + (co < cout ? (float)bias[co] : 0.0f);
        if (relu) v = v > 0.0f ? v : 0.0f;
        mid[(j * 16 + (l & 15)) * MPS + co] = (half)v;
    }
}
#ifdef SWAP
#define WMMA1(a, b, c) __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(b, a, c)
#define STAGE1 stage_t
#else
#define WMMA1(a, b, c) __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c)
#define STAGE1 stage
#endif


// ---- register path (default): every convolution computes the transposed product (weights as A, pixels as B; swapping the operands gives
// the same bits), so a 16x16 result has pixel l & 15 in lane l and channels 2i + (l >> 4) in its 8 values. bias + activation + fp16, then
// one exchange between the two half-waves (v_permlanex16) gives each lane all 16 channels of its pixel: the next layer's B operand, or
// 16 bytes per lane to store. No shared memory between the layers.
typedef uint uint8v __attribute__((ext_vector_type(8)));
static inline half16 to_pixel_major(float8 acc, const global half* bias, int cb, int cout, int relu) {
    const uint l = __builtin_amdgcn_workitem_id_x() & 31, h = l >> 4;
    half v[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int co = cb * 16 + 2 * i + (int)h;
        float x = acc[i] + (co < cout ? (float)bias[co] : 0.0f);
        if (relu) x = x > 0.0f ? x : 0.0f;
        v[i] = (half)x;
    }
    uint8v d;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const uint own = (uint)__builtin_bit_cast(ushort, v[2 * k]) | ((uint)__builtin_bit_cast(ushort, v[2 * k + 1]) << 16);   // channels 4k+h, 4k+2+h
        const uint oth = __builtin_amdgcn_permlanex16(own, own, 0x76543210u, 0xfedcba98u, false, false);           // the other half's
        const uint E = h ? oth : own, O = h ? own : oth;                                                            // even / odd channels
        d[2 * k] = __builtin_amdgcn_perm(O, E, 0x05040100u);       // channels 4k, 4k+1
        d[2 * k + 1] = __builtin_amdgcn_perm(O, E, 0x07060302u);   // channels 4k+2, 4k+3
    }
    return __builtin_bit_cast(half16, d);
}
typedef half half4v __attribute__((ext_vector_type(4)));
static inline void st_tex(constant const char* rec, int x, int y, half4v v) {   // storage descriptor at +32 of the texture record
    __builtin_amdgcn_image_store_2d_v4f16_i32(v, 15, x, y, *(constant const __amdgpu_texture_t*)(rec + 32), 0, 0);
}
#ifndef OUTTEX
#define OUTTEX 0
#endif
static inline half16 ldw(const global half* p) { return __builtin_bit_cast(half16, *(const global uint8v*)p); }   // 32 bytes as 8 dwords
#if SOFTW
// GPUs without WMMA (RDNA2): the same products from DPP dot products (shaders/nn_dot.h wmma_soft, same bits); the weights (A) are loaded
// with the upper half-wave rotated by one row
#define WMMA_T(w, x, c) wmma_soft(w, x, c)
#define WROW(n) wsoft_row(l)
#else
#define WMMA_T(w, x, c) __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(w, x, c)
#define WROW(n) (n)
#endif

#if DOT || SOFTW
#include "nn_dot.h"
#endif
#if DOT
#define WARG constant const half*      // weights and biases: scalar loads
static inline uint pack2(float a, float b) {
    return (uint)__builtin_bit_cast(ushort, (half)a) | ((uint)__builtin_bit_cast(ushort, (half)b) << 16);
}
static inline float relu(float v) { return v > 0.0f ? v : 0.0f; }
#else
#define WARG global const half*
#endif

__attribute__((amdgpu_flat_work_group_size(NTH, NTH)))
kernel void nn_cell(global const half* src, global const half* skip, global half* dst,
                    WARG w1, WARG b1, WARG w2, WARG b2, WARG w3, WARG b3,
                    constant const char* texFeat, constant const char* texNoise, constant const char* texTw) {   // constant: scalar descriptor loads
    local half lds[LDS_HALFS];
    const uint tid = __builtin_amdgcn_workitem_id_x();
    const int wave = tid >> 5, l = tid & 31, n = l & 15;
    const int x0 = __builtin_amdgcn_workgroup_id_x() * TX, y0 = __builtin_amdgcn_workgroup_id_y() * TY;
#ifdef EXP_EMPTY
    if (x0 >= 0) return;
#endif

    // 1. the transformed input tile with a 1 pixel border (zero outside the image = the convolution's padding), CINP channels
#if DOT
#define FILL_LOOP _Pragma("unroll 1")      // DOT: the fill's loads would otherwise set the kernel's VGPR count (occupancy)
#else
#define FILL_LOOP
#endif
#if MODE == 0
    FILL_LOOP
    for (int i = tid; i < TH * TW * (CINP / 8); i += NTH) {
        const int c8 = i % (CINP / 8), p = i / (CINP / 8);
        const int tx = p % TW, ty = p / TW;
        const int gx = x0 + tx - 1, gy = y0 + ty - 1;
        half8v v = (half8v)(0.0h);
        if (gx >= 0 && gx < W && gy >= 0 && gy < H) {
            const global half* px = src + (size_t)(gy * W + gx) * SIN;
#if SIN % 8 == 0
            v = *(const global half8v*)(px + c8 * 8);   // padded rows (the unused channels are zero)
#else
#pragma unroll
            for (int c = 0; c < 8; ++c)
                if (c8 * 8 + c < CIN) v[c] = px[c8 * 8 + c];
#endif
        }
        *(local half8v*)(lds + p * PS + c8 * 8) = v;
    }
#else
    // one tile pixel per iteration, its channel chunks inside: the pixel's geometry and addresses once, the chunks at immediate offsets
    FILL_LOOP
    for (int p = tid; p < TH * TW; p += NTH) {
        const int tx = p % TW, ty = p / TW;
        const int gx = x0 + tx - 1, gy = y0 + ty - 1;
        local half* lp = lds + p * PS;
        if (!(gx >= 0 && gx < W && gy >= 0 && gy < H)) {
#pragma unroll
            for (int c8 = 0; c8 < CINP / 8; ++c8) *(local half8v*)(lp + c8 * 8) = (half8v)(0.0h);
            continue;
        }
#if MODE == 1
        const int W2 = 2 * W;
        const global half* p0 = src + ((size_t)(2 * gy) * W2 + 2 * gx) * CIN;
        const global half* p2 = p0 + (size_t)W2 * CIN;
#pragma unroll FILLU
        for (int c8 = 0; c8 < CINP / 8; ++c8) {
            const half8v a = *(const global half8v*)(p0 + c8 * 8), b = *(const global half8v*)(p0 + CIN + c8 * 8);
            const half8v c = *(const global half8v*)(p2 + c8 * 8), d = *(const global half8v*)(p2 + CIN + c8 * 8);
            half8v v;
#pragma unroll
            for (int k = 0; k < 8; ++k)
                v[k] = (half)__builtin_fmaxf(__builtin_fmaxf((float)a[k], (float)b[k]), __builtin_fmaxf((float)c[k], (float)d[k]));
            *(local half8v*)(lp + c8 * 8) = v;
        }
#else
        const int Wi = W / 2, Hi = H / 2;
        const float fx = (float)gx * 0.5f - 0.25f, fy = (float)gy * 0.5f - 0.25f;
        const float x0f = __builtin_floorf(fx), y0f = __builtin_floorf(fy);
        const float txf = fx - x0f, tyf = fy - y0f;
        const int xi = (int)x0f, yi = (int)y0f;
        const int xa = clampi(xi, 0, Wi - 1), xb = clampi(xi + 1, 0, Wi - 1), ya = clampi(yi, 0, Hi - 1), yb = clampi(yi + 1, 0, Hi - 1);
        const global half* q00 = src + ((size_t)ya * Wi + xa) * CIN;
        const global half* q10 = src + ((size_t)ya * Wi + xb) * CIN;
        const global half* q01 = src + ((size_t)yb * Wi + xa) * CIN;
        const global half* q11 = src + ((size_t)yb * Wi + xb) * CIN;
        const global half* qs = skip + ((size_t)gy * W + gx) * CIN;
#pragma unroll FILLU
        for (int c8 = 0; c8 < CINP / 8; ++c8) {
            const half8v v00 = *(const global half8v*)(q00 + c8 * 8), v10 = *(const global half8v*)(q10 + c8 * 8);
            const half8v v01 = *(const global half8v*)(q01 + c8 * 8), v11 = *(const global half8v*)(q11 + c8 * 8);
            const uint4v sp = *(const global uint4v*)(qs + c8 * 8);   // the skip, two halves per word
            uint4v o;
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                const float u0 = lerpf(lerpf((float)v00[2 * k], (float)v10[2 * k], txf), lerpf((float)v01[2 * k], (float)v11[2 * k], txf), tyf);
                const float u1 = lerpf(lerpf((float)v00[2 * k + 1], (float)v10[2 * k + 1], txf), lerpf((float)v01[2 * k + 1], (float)v11[2 * k + 1], txf), tyf);
                o[k] = add_round_once2(u0, u1, sp[k]);
            }
            *(local uint4v*)(lp + c8 * 8) = o;
        }
#endif
    }
#endif
    BARRIER();

#ifdef EXP_ONLYFILL
    if (lds[tid] == (half)12345.0f) dst[tid] = lds[tid + 1];
    return;
#endif
#if DOT
#if TX != 32 || TY % MPX || (C1 % (WC * CH)) || (C2 % (WC * CH)) || (WC > 1 && C3 > 0) || (CIN % 16 && CIB > 1)
#error "DOT: TX 32, MPX dividing TY, channel split dividing C1 and C2 in chunks, no split with a second 1x1, CIN a multiple of 16 or one block"
#endif
#define NC1 (C1 / WC)
#define NC2 (C2 / WC)
#define NP1 (CIN % 16 ? (CIN + 1) / 2 : 8)   // k pairs per input channel block that are not padding
    // 2. 3x3 convolution CIN -> C1: output channels wc * NC1 .. of pixels (x0 + l, y0 + wp * MPX + m), m < MPX
    const int wv = __builtin_amdgcn_readfirstlane(wave);
    const int wp = wv / WC, wc = wv % WC;
    float acc[MPX][NC1];
#pragma unroll
    for (int m = 0; m < MPX; ++m)
#pragma unroll
        for (int c = 0; c < NC1; ++c) acc[m][c] = 0.0f;
#ifndef DOTU
#define DOTU 1
#endif
#pragma unroll DOTU
    for (int st = 0; st < 9 * CIB; ++st) {
        const int tap = st / CIB, cb = st % CIB, ky = tap / 3, kx = tap % 3;
        uint a[MPX][8];
#pragma unroll
        for (int m = 0; m < MPX; ++m) {
            const local uint4v* ap = (const local uint4v*)(lds + ((wp * MPX + m + ky) * TW + l + kx) * PS + cb * 16);
            const uint4v a0 = ap[0], a1 = ap[1];
            a[m][0] = a0.x; a[m][1] = a0.y; a[m][2] = a0.z; a[m][3] = a0.w; a[m][4] = a1.x; a[m][5] = a1.y; a[m][6] = a1.z; a[m][7] = a1.w;
        }
#ifdef EXP_SAMEW1
        constant const uint* wq = (constant const uint*)w1 + ((size_t)(st & EXP_SAMEW1) * CB1 * 16 + wc * NC1) * 8;
#else
        constant const uint* wq = (constant const uint*)w1 + ((size_t)st * CB1 * 16 + wc * NC1) * 8;
#endif
        DOT_BLOCK(acc, a, wq, NC1, NP1, MPX);
    }
    // bias + relu + fp16 (two channels per word): the 1x1's input
#define XCH (MPX > 1 || WC > 1)        // through shared memory (all C1 channels of MPX pixels do not fit in registers next to the 1x1's sums)
#if XCH
    BARRIER();                         // the exchange reuses the input tile
#pragma unroll
    for (int m = 0; m < MPX; ++m) {
        local uint* xp = (local uint*)(lds + ((wp * MPX + m) * TX + l) * XPS);
#pragma unroll
        for (int j = 0; j < NC1 / 2; ++j) {
            const int co = wc * NC1 + 2 * j;
            xp[co / 2] = pack2(relu(acc[m][2 * j] + (float)b1[co]), relu(acc[m][2 * j + 1] + (float)b1[co + 1]));
        }
    }
    BARRIER();
#else
    uint x1[1][C1 / 2];
#pragma unroll
    for (int j = 0; j < C1 / 2; ++j) x1[0][j] = pack2(relu(acc[0][2 * j] + (float)b1[2 * j]), relu(acc[0][2 * j + 1] + (float)b1[2 * j + 1]));
#endif
    // 3. 1x1 convolution C1 -> C2 (output channels wc * NC2 ..)
    float acc2[MPX][NC2];
#pragma unroll
    for (int m = 0; m < MPX; ++m)
#pragma unroll
        for (int c = 0; c < NC2; ++c) acc2[m][c] = 0.0f;
#pragma unroll
    for (int cb = 0; cb < CB1; ++cb) {
        uint a[MPX][8];
#pragma unroll
        for (int m = 0; m < MPX; ++m) {
#if XCH
            const local uint4v* xp = (const local uint4v*)(lds + ((wp * MPX + m) * TX + l) * XPS + cb * 16);
            const uint4v a0 = xp[0], a1 = xp[1];
            a[m][0] = a0.x; a[m][1] = a0.y; a[m][2] = a0.z; a[m][3] = a0.w; a[m][4] = a1.x; a[m][5] = a1.y; a[m][6] = a1.z; a[m][7] = a1.w;
#else
#pragma unroll
            for (int j = 0; j < 8; ++j) a[m][j] = x1[0][cb * 8 + j];
#endif
        }
        constant const uint* wq = (constant const uint*)w2 + ((size_t)cb * CB2 * 16 + wc * NC2) * 8;
        DOT_BLOCK(acc2, a, wq, NC2, 8, MPX);
    }
    PIN(acc2, NC2, MPX);
#if C3 > 0
    // 4. second 1x1 convolution C2 -> C3, no activation
#define NC3 (((C3 + CH - 1) / CH) * CH)   // the packed weights are zero up to 16 output channels
    float acc3[MPX][NC3];
#pragma unroll
    for (int m = 0; m < MPX; ++m)
#pragma unroll
        for (int c = 0; c < NC3; ++c) acc3[m][c] = 0.0f;
#pragma unroll
    for (int cb = 0; cb < CB2; ++cb) {
        uint a[MPX][8];
#pragma unroll
        for (int m = 0; m < MPX; ++m)
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int co = cb * 16 + 2 * j;
                a[m][j] = pack2(relu(acc2[m][co] + (float)b2[co]), relu(acc2[m][co + 1] + (float)b2[co + 1]));
            }
        constant const uint* wq = (constant const uint*)w3 + ((size_t)cb * CB3 * 16) * 8;
        DOT_BLOCK(acc3, a, wq, NC3, 8, MPX);
    }
    PIN(acc3, NC3, MPX);
#define NCO C3
#define CO0 0
#else
#define NCO NC2
#define CO0 (wc * NC2)
#endif
    // 5. out
#pragma unroll
    for (int m = 0; m < MPX; ++m) {
        half o[NCO];
#pragma unroll
        for (int c = 0; c < NCO; ++c)
#if C3 > 0
            o[c] = (half)(acc3[m][c] + (float)b3[c]);
#else
            o[c] = (half)relu(acc2[m][c] + (float)b2[CO0 + c]);
#endif
        const int X = x0 + l, Y = y0 + wp * MPX + m;
#if OUTTEX
        // as the WMMA path: channels 1-4 -> features (inf / nan -> 0), 0 and 5 -> noise (-> 0.5; z, w = 1), 6 -> temporal weight (-> 0.5)
        if (X < TXW && Y < TXH) {
            half v[7];
#pragma unroll
            for (int c = 0; c < 7; ++c) {
                const bool bad = (__builtin_bit_cast(ushort, o[c]) & 0x7c00) == 0x7c00;
                v[c] = bad ? ((c >= 1 && c <= 4) ? (half)0.0f : (half)0.5f) : o[c];
            }
            st_tex(texFeat, X, Y, (half4v)(v[1], v[2], v[3], v[4]));
            st_tex(texNoise, X, Y, (half4v)(v[0], v[5], (half)1.0f, (half)1.0f));
            st_tex(texTw, X, Y, (half4v)(v[6], (half)1.0f, (half)1.0f, (half)1.0f));
        }
#else
        if (X < W && Y < H) {
            global half* q = dst + ((size_t)Y * W + X) * SOUT + CO0;
#if (SOUT % 8) == 0 && (NCO % 8) == 0
#pragma unroll
            for (int c8 = 0; c8 < NCO / 8; ++c8)
                *(global half8v*)(q + c8 * 8) = (half8v)(o[c8 * 8], o[c8 * 8 + 1], o[c8 * 8 + 2], o[c8 * 8 + 3], o[c8 * 8 + 4], o[c8 * 8 + 5], o[c8 * 8 + 6], o[c8 * 8 + 7]);
#else
#pragma unroll
            for (int c = 0; c < NCO; ++c) q[c] = o[c];
#endif
        }
#endif
    }
#elif !LDSSTAGE
    // 2. 3x3 convolution CIN -> C1 (transposed: weights A, pixels B)
    // with WC > 1 this wave computes the output blocks t = tt * WC + wc of its pixel groups (wp)
    const int wp = wave / WC, wc = wave % WC;
#define NT1 (CB1 / WC)
#define NT2 ((CB2 + WC - 1) / WC)
#if (CB1 % WC) != 0 || (WC > 1 && C3 > 0)
#error "channel split needs CB1 % WC == 0 and no second 1x1"
#endif
    float8 acc[MP][NT1];
#pragma unroll
    for (int j = 0; j < MP; ++j)
#pragma unroll
        for (int t = 0; t < NT1; ++t) acc[j][t] = (float8)(0.0f);
#if PIPE
    // software pipelined: the weights of step s + 1 (step = tap * CIB + cb, the same order as below) load while step s's WMMAs run
    const global half* wb = w1 + ((size_t)((wc) * 16 + WROW(n))) * 16;
    half16 w[NT1];
#pragma unroll
    for (int tt = 0; tt < NT1; ++tt) w[tt] = ldw(wb + (size_t)(tt * WC) * 256);
#pragma unroll PIPEU
    for (int st = 0; st < 9 * CIB; ++st) {
        const int tap = st / CIB, cb = st % CIB, ky = tap / 3, kx = tap % 3;
        half16 wn[NT1];
        const int sn = st + 1 < 9 * CIB ? st + 1 : st;   // the last step reloads its own (harmless, keeps the loop uniform)
#pragma unroll
        for (int tt = 0; tt < NT1; ++tt) wn[tt] = ldw(wb + ((size_t)sn * CB1 + tt * WC) * 256);
#pragma unroll
        for (int j = 0; j < MP; ++j) {
            const int g = wp * MP + j, gy = g / GX, gx = g % GX;
            const half16 a = *(const local half16*)(lds + ((gy + ky) * TW + gx * 16 + n + kx) * PS + cb * 16);
#pragma unroll
            for (int tt = 0; tt < NT1; ++tt) acc[j][tt] = WMMA_T(w[tt], a, acc[j][tt]);
        }
#pragma unroll
        for (int tt = 0; tt < NT1; ++tt) w[tt] = wn[tt];
    }
#else
#pragma unroll TAPU
    for (int tap = 0; tap < 9; ++tap) {
        const int ky = tap / 3, kx = tap % 3;
#pragma unroll CBU
        for (int cb = 0; cb < CIB; ++cb) {
            half16 w[NT1];
#pragma unroll
            for (int tt = 0; tt < NT1; ++tt)
                w[tt] = ldw(w1 + ((size_t)(((EXP_TAPIDX * CIB + EXP_CBIDX) * CB1 + tt * WC + wc) * 16 + WROW(n))) * 16);
#pragma unroll
            for (int j = 0; j < MP; ++j) {
                const int g = wp * MP + j, gy = g / GX, gx = g % GX;
                const half16 a = *(const local half16*)(lds + ((gy + ky) * TW + gx * 16 + n + kx) * PS + cb * 16);
#pragma unroll
                for (int tt = 0; tt < NT1; ++tt) acc[j][tt] = WMMA_T(w[tt], a, acc[j][tt]);
            }
        }
    }
#endif
    // 3. 1x1 convolution C1 -> C2
    float8 acc2[MP][NT2];
#if WC > 1
    local half* xch = lds + TILE_HALFS;   // [pixel group][cb][16 pixels][16 channels]
#pragma unroll
    for (int j = 0; j < MP; ++j)
#pragma unroll
        for (int tt = 0; tt < NT1; ++tt) {
            const half16 x = to_pixel_major(acc[j][tt], b1, tt * WC + wc, C1, 1);
            if (l < 16) *(local half16*)(xch + (((wp * MP + j) * CB1 + tt * WC + wc) * 16 + n) * 16) = x;
        }
    BARRIER();
#endif
#pragma unroll
    for (int j = 0; j < MP; ++j) {
        half16 x[CB1];
#if WC > 1
#pragma unroll
        for (int cb = 0; cb < CB1; ++cb) x[cb] = *(const local half16*)(xch + (((wp * MP + j) * CB1 + cb) * 16 + n) * 16);
#else
#pragma unroll
        for (int cb = 0; cb < CB1; ++cb) x[cb] = to_pixel_major(acc[j][cb], b1, cb, C1, 1);
#endif
#pragma unroll
        for (int t = 0; t < NT2; ++t) acc2[j][t] = (float8)(0.0f);
#pragma unroll
        for (int cb = 0; cb < CB1; ++cb)
#pragma unroll
            for (int tt = 0; tt < NT2; ++tt)
                if (tt * WC + wc < CB2)
                    acc2[j][tt] = WMMA_T(ldw(w2 + ((size_t)((cb * CB2 + tt * WC + wc) * 16 + WROW(n))) * 16), x[cb], acc2[j][tt]);
    }
#if C3 > 0
    // 4. second 1x1 convolution C2 -> C3, no activation
    float8 accO[MP][CB3];
#pragma unroll
    for (int j = 0; j < MP; ++j) {
        half16 x[CB2];
#pragma unroll
        for (int cb = 0; cb < CB2; ++cb) x[cb] = to_pixel_major(acc2[j][cb], b2, cb, C2, 1);
#pragma unroll
        for (int t = 0; t < CB3; ++t) accO[j][t] = (float8)(0.0f);
#pragma unroll
        for (int cb = 0; cb < CB2; ++cb)
#pragma unroll
            for (int t = 0; t < CB3; ++t)
                accO[j][t] = WMMA_T(ldw(w3 + ((size_t)((cb * CB3 + t) * 16 + WROW(n))) * 16), x[cb], accO[j][t]);
    }
#define CBO CB3
#define BO b3
#define RO 0
#else
#define accO acc2
#define CBO NT2
#define BO b2
#define RO 1
#endif
    // 5. out: lanes 0-15 store channels 0-7 of their pixel, lanes 16-31 channels 8-15
#pragma unroll
    for (int j = 0; j < MP; ++j) {
        const int g = wp * MP + j, gy = g / GX, gx = g % GX;
        const int X = x0 + gx * 16 + n, Y = y0 + gy, hh = l >> 4;
#pragma unroll
        for (int tt = 0; tt < CBO; ++tt) {
            const int t = tt * WC + wc;   // the output block (WC = 1: tt)
            if (C3 == 0 && t >= CB2) continue;
            const half16 o = to_pixel_major(accO[j][tt], BO, t, COUTN, RO);
#if OUTTEX
            // the network's output straight into MetalFX's three textures, as dbfnetv1_output_channels (NHWC) writes them: channels 1-4 ->
            // features (inf / nan -> 0), 0 and 5 -> noise (-> 0.5; z, w = 1), 6 -> temporal weight (-> 0.5); texture size TXW x TXH
            if (hh == 0 && X < TXW && Y < TXH) {
                half v[7];
#pragma unroll
                for (int c = 0; c < 7; ++c) {
                    const bool bad = (__builtin_bit_cast(ushort, o[c]) & 0x7c00) == 0x7c00;
                    v[c] = bad ? ((c >= 1 && c <= 4) ? (half)0.0f : (half)0.5f) : o[c];
                }
                st_tex(texFeat, X, Y, (half4v)(v[1], v[2], v[3], v[4]));
                st_tex(texNoise, X, Y, (half4v)(v[0], v[5], (half)1.0f, (half)1.0f));
                st_tex(texTw, X, Y, (half4v)(v[6], (half)1.0f, (half)1.0f, (half)1.0f));
            }
            continue;
#endif
            if (X < W && Y < H) {
                global half* q = dst + ((size_t)Y * W + X) * SOUT + t * 16 + hh * 8;
#if (SOUT % 8) == 0 && (COUTN % 16) == 0
                *(global half8v*)q = hh ? o.s89abcdef : o.s01234567;
#else
#pragma unroll
                for (int c = 0; c < 8; ++c)
                    if (t * 16 + hh * 8 + c < COUTN) q[c] = hh ? o[8 + c] : o[c];
#endif
            }
        }
    }
#else
    // 2. 3x3 convolution CIN -> C1
    float8 acc[MP][CB1];
#pragma unroll
    for (int j = 0; j < MP; ++j)
#pragma unroll
        for (int t = 0; t < CB1; ++t) acc[j][t] = (float8)(0.0f);
#pragma unroll TAPU
    for (int tap = 0; tap < 9; ++tap) {
        const int ky = tap / 3, kx = tap % 3;
        {
#pragma unroll CBU
            for (int cb = 0; cb < CIB; ++cb) {
                half16 b[CB1];
#pragma unroll
                for (int t = 0; t < CB1; ++t)
                    b[t] = ldw(w1 + ((size_t)((((ky * 3 + kx) * CIB + cb) * CB1 + t) * 16 + n)) * 16);
#pragma unroll
                for (int j = 0; j < MP; ++j) {
                    const int g = wave * MP + j, gy = g / GX, gx = g % GX;
                    const half16 a = *(const local half16*)(lds + ((gy + ky) * TW + gx * 16 + n + kx) * PS + cb * 16);
#ifndef EXP_NOCONV
#pragma unroll
                    for (int t = 0; t < CB1; ++t) acc[j][t] = WMMA1(a, b[t], acc[j][t]);
#else
                    acc[j][0][0] += (float)a[0] + (float)b[0][1];
#endif
                }
            }
        }
    }
    BARRIER();   // everyone is done with the input tile; the staging reuses it
    local half* mid = lds + wave * MP * 16 * MPS;   // this wave's MP * 16 pixels
#pragma unroll
    for (int j = 0; j < MP; ++j)
#pragma unroll
        for (int t = 0; t < CB1; ++t) STAGE1(mid, j, acc[j][t], b1, t, C1, 1);
    BARRIER();

    // 3. 1x1 convolution C1 -> C2
    float8 acc2[MP][CB2];
#pragma unroll
    for (int j = 0; j < MP; ++j)
#pragma unroll
        for (int t = 0; t < CB2; ++t) acc2[j][t] = (float8)(0.0f);
#pragma unroll CBU
    for (int cb = 0; cb < CB1; ++cb) {
        half16 b[CB2];
#pragma unroll
        for (int t = 0; t < CB2; ++t) b[t] = ldw(w2 + ((size_t)((cb * CB2 + t) * 16 + n)) * 16);
#pragma unroll
        for (int j = 0; j < MP; ++j) {
            const half16 a = *(const local half16*)(mid + (j * 16 + n) * MPS + cb * 16);
#pragma unroll
            for (int t = 0; t < CB2; ++t) acc2[j][t] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b[t], acc2[j][t]);
        }
    }
    BARRIER();
#pragma unroll
    for (int j = 0; j < MP; ++j)
#pragma unroll
        for (int t = 0; t < CB2; ++t) stage(mid, j, acc2[j][t], b2, t, C2, 1);
    BARRIER();
#if C3 > 0
    // 4. second 1x1 convolution C2 -> C3, no activation (the network's output layer)
    float8 acc3[MP][CB3];
#pragma unroll
    for (int j = 0; j < MP; ++j)
#pragma unroll
        for (int t = 0; t < CB3; ++t) acc3[j][t] = (float8)(0.0f);
#pragma unroll CBU
    for (int cb = 0; cb < CB2; ++cb) {
        half16 b[CB3];
#pragma unroll
        for (int t = 0; t < CB3; ++t) b[t] = ldw(w3 + ((size_t)((cb * CB3 + t) * 16 + n)) * 16);
#pragma unroll
        for (int j = 0; j < MP; ++j) {
            const half16 a = *(const local half16*)(mid + (j * 16 + n) * MPS + cb * 16);
#pragma unroll
            for (int t = 0; t < CB3; ++t) acc3[j][t] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b[t], acc3[j][t]);
        }
    }
    BARRIER();
#pragma unroll
    for (int j = 0; j < MP; ++j)
#pragma unroll
        for (int t = 0; t < CB3; ++t) stage(mid, j, acc3[j][t], b3, t, C3, 0);
    BARRIER();
#define COUT C3
#else
#define COUT C2
#endif
    // 5. out: SOUT halfs per pixel (the first COUT of them written)
#if (COUT % 8) == 0 && (SOUT % 8) == 0
    for (int i = l; i < MP * 16 * (COUT / 8); i += 32) {
        const int c8 = i % (COUT / 8), q = i / (COUT / 8), j = q / 16, p = q % 16;
        const int g = wave * MP + j, gy = g / GX, gx = g % GX;
        const int X = x0 + gx * 16 + p, Y = y0 + gy;
        if (X < W && Y < H) *(global half8v*)(dst + ((size_t)Y * W + X) * SOUT + c8 * 8) = *(const local half8v*)(mid + q * MPS + c8 * 8);
    }
#else
    for (int i = l; i < MP * 16 * COUT; i += 32) {
        const int c = i % COUT, q = i / COUT, j = q / 16, p = q % 16;
        const int g = wave * MP + j, gy = g / GX, gx = g % GX;
        const int X = x0 + gx * 16 + p, Y = y0 + gy;
        if (X < W && Y < H) dst[((size_t)Y * W + X) * SOUT + c] = mid[q * MPS + c];
    }
#endif
#endif
}
