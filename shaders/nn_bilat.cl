// m4r: MetalFX's denoiser bilateral (a-trous) pass, BFNet's `bilateral` kernel with bfnet1_enabled and no specular hit distance, rewritten
// for RDNA3 in its FastMath form (Metal's fast:: exp / log / sqrt / pow as the raw hardware instructions, as the translated kernel runs with
// M4R_FAST_MATH). Same formula, inputs, sampling and accumulation order as Apple's kernel; what differs is how the 5x5 taps reach a thread.
// Apple's kernel samples four corner taps per lane and passes the other 21 between lanes with simd shuffles and selects (about a third of
// its instructions); here a workgroup loads its footprint once into shared memory and every thread reads its 25 taps from there.
//
// A workgroup is an 8x8 lattice of pixels at the pass's step s = 2^IT (pixels base + (i, j) * s), so its 5x5 footprint is a 12x12 lattice:
// colour+variance and normal+roughness at clamped texel positions (Apple: nearest, clamp to edge, pixel coordinates), learned features
// sampled bilinearly at the tap's (pixel + 0.5) / size like Apple's (half resolution, linear, clamp to edge).
// -D W H (image size), IT (iteration: step 2^IT). Textures: 0 colour/variance (RGBA16F, read as float), 1 normal/roughness, 2 out,
// 3 shininess, 4 noise/bandwidth (half res), 5 learned features (half res), 6 depth. Samplers: __m4r_sampler_0 linear normalised,
// __m4r_sampler_1 nearest normalised. Launch: groups ceil(W / 8s) * s x ceil(H / 8s) * s of 8x8.
// SPDX-License-Identifier: GPL-3.0-only
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
typedef half half4 __attribute__((ext_vector_type(4)));
typedef float float4v __attribute__((ext_vector_type(4)));
typedef int int4v __attribute__((ext_vector_type(4)));
typedef __amdgpu_texture_t tex_t;

__attribute__((visibility("protected"))) global int4v __m4r_sampler_0 = (int4v)(0);   // linear, clamp to edge, normalised
__attribute__((visibility("protected"))) global int4v __m4r_sampler_1 = (int4v)(0);   // nearest, clamp to edge, normalised

#define S (1 << IT)
#define TL 12                          // footprint lattice side (8 + 4)
#ifndef POWSQ
#define POWSQ 0
#endif
#ifndef LDSF
#define LDSF 0                         // 1: the footprint in shared memory as f32 (the same values: fp16 -> f32 is exact)
#endif
#if LDSF
typedef float4v lds_t;
#define TOL(v) ((float4v)((float)(v).x, (float)(v).y, (float)(v).z, (float)(v).w))
#else
typedef half4 lds_t;
#define TOL(v) (v)
#endif
#define BARRIER() do { __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup"); __builtin_amdgcn_s_barrier(); __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup"); } while (0)

// texture records through the constant address space: scalar loads (a record is uniform), not a waterfall loop per image access
static inline tex_t T(constant const char* rec) { return *(constant const tex_t*)rec; }
static inline float4v ldf(constant const char* rec, int x, int y) { return __builtin_amdgcn_image_load_2d_v4f32_i32(15, x, y, T(rec), 0, 0); }
static inline half4 ldh(constant const char* rec, int x, int y) { return __builtin_amdgcn_image_load_2d_v4f16_i32(15, x, y, T(rec), 0, 0); }
static inline half4 smp(constant const char* rec, int4v s, float u, float v) {
    return __builtin_amdgcn_image_sample_lz_2d_v4f16_f32(15, u, v, T(rec), s, false, 0, 0);
}
static inline int clampi(int v, int lo, int hi) { return v < lo ? lo : v > hi ? hi : v; }
// Metal fast:: functions as the translator lowers them with M4R_FAST_MATH (air/intrinsics.py fast_exp / fast_log / fpow / fast_sqrt)
static inline float fexp(float x) { return __builtin_amdgcn_exp2f(x * 1.44269502162933349609375f); }
static inline float flog(float x) { return __builtin_amdgcn_logf(x) * 0.693147182464599609375f; }
static inline float fpow(float x, float y) { return __builtin_amdgcn_exp2f(__builtin_amdgcn_logf(__builtin_fabsf(x)) * y); }
static inline float clampf(float x, float lo, float hi) { return __builtin_fminf(__builtin_fmaxf(x, lo), hi); }
static inline float rcp(float x) { return __builtin_amdgcn_rcpf(x); }
#define EPS 9.99999974737875163555145263671875e-06f   // f0x3727C5AC

__attribute__((amdgpu_flat_work_group_size(64, 64)))
kernel void nn_bilat(constant const char* colTex, constant const char* nrTex, constant const char* outTex, constant const char* shinTex,
                     constant const char* noiseTex, constant const char* featTex, constant const char* depthTex) {
    local lds_t lc[TL * TL], ln[TL * TL], lf[TL * TL];
    const int lx = (int)__builtin_amdgcn_workitem_id_x(), ly = (int)__builtin_amdgcn_workitem_id_y(), tid = ly * 8 + lx;
    const int gx = __builtin_amdgcn_workgroup_id_x(), gy = __builtin_amdgcn_workgroup_id_y();
    const int bx = (gx / S) * 8 * S + gx % S, by = (gy / S) * 8 * S + gy % S;   // the lattice's first pixel
    const int4v linS = __m4r_sampler_0, nearS = __m4r_sampler_1;
    const float iw = rcp((float)W), ih = rcp((float)H);
    for (int i = tid; i < TL * TL; i += 64) {
        const int qx = bx + (i % TL - 2) * S, qy = by + (i / TL - 2) * S;
        const int cx = clampi(qx, 0, W - 1), cy = clampi(qy, 0, H - 1);
        const float4v c = ldf(colTex, cx, cy);
        lc[i] = TOL(((half4)((half)c.x, (half)c.y, (half)c.z, (half)c.w)));   // RGBA16F texels: exact
        ln[i] = TOL(ldh(nrTex, cx, cy));
        lf[i] = TOL(smp(featTex, linS, ((float)qx + 0.5f) * iw, ((float)qy + 0.5f) * ih));
    }
    BARRIER();
    const int x = bx + lx * S, y = by + ly * S;
    if (x >= W || y >= H) return;
    const int ci = (ly + 2) * TL + lx + 2;
    const lds_t n0h = ln[ci];
    const float n0x = (float)n0h.x, n0y = (float)n0h.y, n0z = (float)n0h.z;
    const lds_t c0h = lc[ci];
    const float4v c0 = (float4v)((float)c0h.x, (float)c0h.y, (float)c0h.z, (float)c0h.w);
    const bool valid = (n0x * n0x + n0y * n0y + n0z * n0z) > 0.0f && ldh(depthTex, x, y).x != (half)0.0f;
    if (!valid) { __builtin_amdgcn_image_store_2d_v4f32_i32(c0, 15, x, y, *(constant const tex_t*)(outTex + 32), 0, 0); return; }
    const float u = ((float)x + 0.5f) * iw, v = ((float)y + 0.5f) * ih;
    const half4 nz = smp(noiseTex, linS, u, v);
    const float sA = 0.98f * rcp(fexp(-(float)nz.x) + 1.0f) + 9.99999e-03f;
    const float sB = 1.98f * rcp(fexp(-(float)nz.y) + 1.0f) + 9.99999e-03f;
    const float wA = __builtin_fmaxf(sA, 9.99999e-03f), wB = __builtin_fmaxf(sB, 9.99999e-03f);
    const lds_t f0h = lf[ci];
    const float4v f0 = (float4v)((float)f0h.x, (float)f0h.y, (float)f0h.z, (float)f0h.w);
    const half r0 = (half)n0h.w;
    const half stepH = (half)(float)S;
    const float sig = __builtin_fmaxf((float)smp(shinTex, nearS, u, v).x * 40.0f, 0.0f) + EPS;
    const float isig = rcp(sig), iwB = rcp(wB), ivar = rcp(c0.w + EPS);
    float ar = EPS, ag = EPS, ab = EPS, aw = EPS;
#pragma unroll
    for (int r = -2; r <= 2; ++r) {
#pragma unroll
        for (int c = -2; c <= 2; ++c) {
            float w, pr, pg, pb;
            if (r == 0 && c == 0) {
                w = (float)(half)(1.0f - clampf(wA, 0.0f, 1.0f));
                pr = w * c0.x; pg = w * c0.y; pb = w * c0.z;
            } else {
                const int ti = (ly + 2 + r) * TL + lx + 2 + c;
                const lds_t nt = ln[ti], ct = lc[ti], ft = lf[ti];
                const float dy = (float)((half)(float)r * stepH), dx = (float)((half)(float)c * stepH);
                const float d = (float)(half)__builtin_amdgcn_sqrtf(dy * dy + dx * dx);
                const float q = d * isig;
                w = fexp(-(q * q)) * wA;
                const half th = (half)clampf((float)__builtin_fabsf((half)nt.w - r0) * 1.11111116409301757812f - 0.1f, 0.0f, 1.0f);
                const float t2 = (float)(th * th), tf = (float)th;
                w = w * clampf(1.0f - (3.0f - tf * 2.0f) * t2, 0.0f, 1.0f);
                const float nd = (float)nt.x * n0x + (float)nt.y * n0y + (float)nt.z * n0z;
#if POWSQ
                // x^128 as seven squarings instead of exp2(log2(x) * 128) (not bit-exact: about 1e-5 relative)
                float p = __builtin_fmaxf(nd, 0.01f);
                p *= p; p *= p; p *= p; p *= p; p *= p; p *= p; p *= p;
                w = w * p;
#else
                w = w * fpow(__builtin_fmaxf(nd, 0.01f), 128.0f);
#endif
                const float ex = (float)ft.x - f0.x, ey = (float)ft.y - f0.y, ez = (float)ft.z - f0.z, ew = (float)ft.w - f0.w;
                const float e = fexp(-(ex * ex + ey * ey + ez * ez + ew * ew) * iwB);
                if ((__builtin_bit_cast(uint, e) & 0x7f800000u) != 0x7f800000u) w = clampf(e, 1e-3f, 1.0f) * w;
                w = __builtin_fmaxf(1.0f - __builtin_fabsf(flog((float)ct.w * ivar + EPS)), 1e-3f) * w;
                pr = w * (float)ct.x; pg = w * (float)ct.y; pb = w * (float)ct.z;
            }
            ar = pr + ar; ag = pg + ag; ab = pb + ab; aw = w + aw;
        }
    }
    const float iaw = rcp(aw);
    __builtin_amdgcn_image_store_2d_v4f32_i32((float4v)(ar * iaw, ag * iaw, ab * iaw, c0.w), 15, x, y, *(constant const tex_t*)(outTex + 32), 0, 0);
}
