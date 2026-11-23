// m4r: the stages around MetalFX's denoiser that are not MetalFX's, as kernels that read and write images directly (no staging
// copies; the runtime hands a kernel the texture's descriptor record: sampled descriptor at +0, storage descriptor at +32).
// Three kernels from this source, selected with -D:
//   PREP   before the denoiser: roughness = normals.w (R16F), colour * exposure (RGBA16F), linear depth -> reversed hardware depth and, for the
//          upscaler that follows, depth in the game's projection convention (R32F). Replaces shaders/nn_rough, nn_expose (mode 0), nn_lin2hw.
//   TAIL1  after the denoiser: mirror gate = the temporal-only result blended in on mirror-like pixels (shaders/nn_mirror.comp semantics).
//   TAIL2  despeckle (shaders/nn_despeckle.comp) and the exposure undo (nn_expose mode 1) on TAIL1's image, stored into the denoiser's output
//          image and, when asked, straight into the image the upscaler reads.
// Every stage rounds to fp16 where the buffer-based version stored fp16, so the results are the same bits.
// SPDX-License-Identifier: GPL-3.0-only
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
typedef half half4 __attribute__((ext_vector_type(4)));
typedef float float4v __attribute__((ext_vector_type(4)));
typedef __amdgpu_texture_t tex_t;
#pragma clang fp contract(off)   // every fused multiply-add below is written as fma(): the buffer-based shaders' rounding is reproduced exactly

// texture records and parameters come through the constant address space: scalar loads (uniform), no per-lane waterfall loops
// no OpenCL library is linked: GLSL's clamp (min(max(x, lo), hi)) and fmax as compiler builtins
static inline float clampf(float x, float lo, float hi) { return __builtin_fminf(__builtin_fmaxf(x, lo), hi); }
static inline int clampi(int x, int lo, int hi) { return x < lo ? lo : (x > hi ? hi : x); }

static inline float4v ld(constant const char* rec, int x, int y) {   // sampled descriptor at +0, float result (exact for fp16 / packed formats)
    return __builtin_amdgcn_image_load_2d_v4f32_i32(15, x, y, *(constant const tex_t*)rec, 0, 0);
}
static inline void st16(constant const char* rec, int x, int y, half4 v) { __builtin_amdgcn_image_store_2d_v4f16_i32(v, 15, x, y, *(constant const tex_t*)(rec + 32), 0, 0); }
static inline void st16x(constant const char* rec, int x, int y, half v) { __builtin_amdgcn_image_store_2d_v4f16_i32((half4)(v, (half)0, (half)0, (half)0), 1, x, y, *(constant const tex_t*)(rec + 32), 0, 0); }
static inline void st32x(constant const char* rec, int x, int y, float v) { __builtin_amdgcn_image_store_2d_f32_i32(v, 1, x, y, *(constant const tex_t*)(rec + 32), 0, 0); }

#if defined(PREP) || defined(LOWRES)
typedef struct { uint flags; float nearZ, m23, m33, sky, m22, m32, m23k, m33k; uint mw, mh; } PrepPar;   // flags: 1 roughness, 2 expose, 4 linear depth, 8 depth for K, 16 no int_motion copy (LOWRES); mw, mh: motion texture size (LOWRES)
#endif
#if defined(PREP)
__attribute__((amdgpu_flat_work_group_size(64, 64)))
kernel void nn_prep(constant const char* normals, constant const char* roughOut, constant const char* colorGame, constant const char* exposure,
                    constant const char* colorExp, constant const char* depthGame, constant const char* depthHw, constant const char* depthK,
                    constant const PrepPar* par) {
    const int x = __builtin_amdgcn_workgroup_id_x() * 8 + (int)__builtin_amdgcn_workitem_id_x();
    const int y = __builtin_amdgcn_workgroup_id_y() * 8 + (int)__builtin_amdgcn_workitem_id_y();
    if (x >= W || y >= H) return;
    const uint fl = par->flags;
    if (fl & 1u) st16x(roughOut, x, y, (half)ld(normals, x, y).w);
    if (fl & 2u) {
        const float e = ld(exposure, 0, 0).x;
        const float scale = e > 0.0f ? e : 1.0f;
        const float4v c = ld(colorGame, x, y);
        half4 o = (half4)((half)clampf(c.x * scale, 0.0f, 65504.0f), (half)clampf(c.y * scale, 0.0f, 65504.0f), (half)clampf(c.z * scale, 0.0f, 65504.0f), (half)1.0f);
        st16(colorExp, x, y, o);
    }
    if (fl & 4u) {
        const float lin = ld(depthGame, x, y).x;
        float d = 0.0f;
        if (lin > 0.0f && lin < par->sky) {
            const float w = fma(lin, par->m23, par->m33);
            if (w > 0.0f) d = DIV(par->nearZ, w);
        }
        st32x(depthHw, x, y, d);
        if (fl & 8u) {
            float dk = clampf(par->m22, 0.0f, 1.0f);
            if (lin > 0.0f && lin < par->sky) {
                const float w = fma(lin, par->m23k, par->m33k);
                if (w > 0.0f) dk = clampf(DIV(fma(lin, par->m22, par->m32), w), 0.0f, 1.0f);
            }
            st32x(depthK, x, y, dk);
        }
    }
}
#endif

#if defined(TAIL1)
typedef struct { uint flags; float r0, r1, strength; } Tail1Par;   // flags: 1 mirror gate on
__attribute__((amdgpu_flat_work_group_size(64, 64)))
kernel void nn_tail1(constant const char* den, constant const char* denT, constant const char* rough, constant const char* mirOut, constant const Tail1Par* par) {
    const int x = __builtin_amdgcn_workgroup_id_x() * 8 + (int)__builtin_amdgcn_workitem_id_x();
    const int y = __builtin_amdgcn_workgroup_id_y() * 8 + (int)__builtin_amdgcn_workitem_id_y();
    if (x >= W || y >= H) return;
    const float4v f = ld(den, x, y);
    float mirror = 0.0f;
    if (par->flags & 1u) {
        float r = 0.0f;
        for (int dy = -1; dy <= 1; ++dy)
            for (int dx = -1; dx <= 1; ++dx)
                r += (float)(half)ld(rough, clampi(x + dx, 0, W - 1), clampi(y + dy, 0, H - 1)).x;
        r = DIV9(r);
        const float t = clampf(DIVT(r - par->r0, __builtin_fmaxf(par->r1 - par->r0, 1e-4f)), 0.0f, 1.0f);
        mirror = SMOOTHSTEP1(t) * par->strength;
    }
    half4 o = (half4)((half)f.x, (half)f.y, (half)f.z, (half)f.w);
    if (mirror > 0.0f) {
        const float4v g = ld(denT, x, y);
        o = (half4)((half)MIX(f.x, g.x, mirror), (half)MIX(f.y, g.y, mirror), (half)MIX(f.z, g.z, mirror), (half)f.w);
    }
    st16(mirOut, x, y, o);
}
#endif

#if defined(TAIL2)
typedef struct { uint flags; } Tail2Par;   // flags: 1 despeckle, 2 exposure undo, 4 also store into outK
static inline float lum(float4v c) { return (c.x + c.y + c.z) * (1.0f / 3.0f); }
__attribute__((amdgpu_flat_work_group_size(64, 64)))
kernel void nn_tail2(constant const char* mir, constant const char* exposure, constant const char* out0, constant const char* outK, constant const Tail2Par* par) {
    const int x = __builtin_amdgcn_workgroup_id_x() * 8 + (int)__builtin_amdgcn_workitem_id_x();
    const int y = __builtin_amdgcn_workgroup_id_y() * 8 + (int)__builtin_amdgcn_workitem_id_y();
    if (x >= W || y >= H) return;
    const uint fl = par->flags;
    const float4v c = ld(mir, x, y);
    float3 rgb = (float3)(c.x, c.y, c.z);
    if (fl & 1u) {
        float top0 = 0.0f, top1 = 0.0f, top2 = 0.0f;   // three largest neighbour luminances, descending
        for (int dy = -1; dy <= 1; ++dy)
            for (int dx = -1; dx <= 1; ++dx) {
                if (dx == 0 && dy == 0) continue;
                const float l = lum(ld(mir, clampi(x + dx, 0, W - 1), clampi(y + dy, 0, H - 1)));
                if (l > top0) { top2 = top1; top1 = top0; top0 = l; }
                else if (l > top1) { top2 = top1; top1 = l; }
                else if (l > top2) top2 = l;
            }
        const float l = lum(c), cap = FMA(top2, 1.5f, 1e-4f);
        if (l > cap) rgb = rgb * DIV(cap, l);
    }
    half3 h = (half3)((half)rgb.x, (half)rgb.y, (half)rgb.z);   // the buffer between the stages held fp16
    float3 v = (float3)((float)h.x, (float)h.y, (float)h.z);
    if (fl & 2u) {
        const float e = ld(exposure, 0, 0).x;
        if (e > 0.0f) {
            v = (float3)(clampf(DIV(v.x, e), 0.0f, 65504.0f), clampf(DIV(v.y, e), 0.0f, 65504.0f), clampf(DIV(v.z, e), 0.0f, 65504.0f));
            h = (half3)((half)v.x, (half)v.y, (half)v.z);
        }
    }
    const half4 o = (half4)(h.x, h.y, h.z, (half)c.w);
    st16(out0, x, y, o);
    if (fl & 4u) st16(outK, x, y, o);
}
#endif

#if defined(TAILF)
// TAILF: everything after the bilateral filter in one pass. MetalFX's remodulation (denoiser_between_processing, as specialised in m4r's
// plans: bfnet1_enabled, no str mask, no compose buffer) of the temporal blend (for the mirror gate) and of the filtered result, then TAIL1's
// mirror gate and TAIL2's despeckle and exposure undo; the gated image of the tile and a 1 pixel border stays in shared memory for the
// despeckle's 3x3 neighbourhood. Bit-identical to the four passes it replaces: the remodulation follows the instructions of the translated
// Apple kernel (albedo read as fp16 like its d16 loads, products rounded once to fp16 like its v_fma_mixlo_f16, fp16 arithmetic, LLVM's fp16
// division, the fast-math reassociation (4 t) / L its compiler made).
typedef struct { uint flags; float r0, r1, strength, acc, accAll, msx, msy; } TailFPar;   // flags: 1 mirror gate, 2 despeckle, 4 exposure undo, 8 also store into outK,
                                                                   // 16 not into out0 (the caller upscales from outK and nothing reads out0),
                                                                   // 32 mirror median (with 1): gated pixels toward their MEDK x MEDK median
                                                                   // 64 mirror accumulation (with 1; camera still): gated pixels averaged with
                                                                   //    mhist, the previous frames' result at the same pixel, with weight acc
                                                                   //    (acc 1: the first still frame, only stored)
                                                                   // 128 still accumulation (camera still): every pixel whose motion vector is
                                                                   //    below a quarter pixel (msx, msy: motion scale) averaged the same way,
                                                                   //    weight max(acc, accAll); mhist then holds every pixel
                                                                   // 512 temporal outlier clamp (camera still, history warm): a pixel's luminance
                                                                   //    kept within its 3x3 neighbours' running means (pxPrev) x [0.7, 1.4];
                                                                   //    pxCur gets the running mean (always written)
// the median of nine values (a 19-exchange sorting network; Paeth / Devillard's opt_med9)
#define MSORT(a, b) { const float lo_ = __builtin_fminf(p[a], p[b]); p[b] = __builtin_fmaxf(p[a], p[b]); p[a] = lo_; }
static inline float med9(float* p) {
    MSORT(1, 2) MSORT(4, 5) MSORT(7, 8) MSORT(0, 1) MSORT(3, 4) MSORT(6, 7) MSORT(1, 2) MSORT(4, 5) MSORT(7, 8) MSORT(0, 3)
    MSORT(5, 8) MSORT(4, 7) MSORT(3, 6) MSORT(1, 4) MSORT(2, 5) MSORT(4, 7) MSORT(4, 2) MSORT(6, 4) MSORT(4, 2)
    return p[4];
}
// the median of 25 values (Devillard's opt_med25, 99 exchanges; checked against numpy on 600000 random sets with ties)
static inline float med25(float* p) {
    MSORT(0, 1) MSORT(3, 4) MSORT(2, 4) MSORT(2, 3) MSORT(6, 7) MSORT(5, 7) MSORT(5, 6) MSORT(9, 10) MSORT(8, 10) MSORT(8, 9)
    MSORT(12, 13) MSORT(11, 13) MSORT(11, 12) MSORT(15, 16) MSORT(14, 16) MSORT(14, 15) MSORT(18, 19) MSORT(17, 19) MSORT(17, 18)
    MSORT(21, 22) MSORT(20, 22) MSORT(20, 21) MSORT(23, 24) MSORT(2, 5) MSORT(3, 6) MSORT(0, 6) MSORT(0, 3) MSORT(4, 7)
    MSORT(1, 7) MSORT(1, 4) MSORT(11, 14) MSORT(8, 14) MSORT(8, 11) MSORT(12, 15) MSORT(9, 15) MSORT(9, 12) MSORT(13, 16)
    MSORT(10, 16) MSORT(10, 13) MSORT(20, 23) MSORT(17, 23) MSORT(17, 20) MSORT(21, 24) MSORT(18, 24) MSORT(18, 21) MSORT(19, 22)
    MSORT(8, 17) MSORT(9, 18) MSORT(0, 18) MSORT(0, 9) MSORT(10, 19) MSORT(1, 19) MSORT(1, 10) MSORT(11, 20) MSORT(2, 20)
    MSORT(2, 11) MSORT(12, 21) MSORT(3, 21) MSORT(3, 12) MSORT(13, 22) MSORT(4, 22) MSORT(4, 13) MSORT(14, 23) MSORT(5, 23)
    MSORT(5, 14) MSORT(15, 24) MSORT(6, 24) MSORT(6, 15) MSORT(7, 16) MSORT(7, 19) MSORT(13, 21) MSORT(15, 23) MSORT(7, 13)
    MSORT(7, 15) MSORT(1, 9) MSORT(3, 11) MSORT(5, 17) MSORT(11, 17) MSORT(9, 17) MSORT(4, 10) MSORT(6, 12) MSORT(7, 14)
    MSORT(4, 6) MSORT(4, 7) MSORT(12, 14) MSORT(10, 14) MSORT(6, 7) MSORT(10, 12) MSORT(6, 10) MSORT(6, 17) MSORT(12, 17)
    MSORT(7, 17) MSORT(7, 10) MSORT(12, 18) MSORT(7, 12) MSORT(10, 18) MSORT(12, 20) MSORT(10, 20) MSORT(10, 12)
    return p[12];
}
#define BARRIER() do { __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup"); __builtin_amdgcn_s_barrier(); __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup"); } while (0)
static inline half4 ld16(constant const char* rec, int x, int y) {
    return __builtin_amdgcn_image_load_2d_v4f16_i32(15, x, y, *(constant const tex_t*)rec, 0, 0);
}
static inline half mul_round_once(float a, float b) {   // v_fma_mixlo_f16 a, b, 0: the exact product rounded to fp16
    uint r = 0;
    __asm volatile("v_fma_mixlo_f16 %0, %1, %2, 0" : "+v"(r) : "v"(a), "v"(b));
    return __builtin_bit_cast(half, (ushort)(r & 0xffff));
}
static inline half4 remodulate(float4v s, half4 dif, half4 spe) {
    const half cx = mul_round_once(s.x, __builtin_fmaxf((float)dif.x, 1e-3f) + __builtin_fmaxf((float)spe.x, 1e-3f));
    const half cy = mul_round_once(s.y, __builtin_fmaxf((float)dif.y, 1e-3f) + __builtin_fmaxf((float)spe.y, 1e-3f));
    const half cz = mul_round_once(s.z, __builtin_fmaxf((float)dif.z, 1e-3f) + __builtin_fmaxf((float)spe.z, 1e-3f));
    const half L = (half)__builtin_fmaf16((_Float16)(cz + cx), (_Float16)0.25f, (_Float16)(cy * (half)0.5f));
    const half m = __builtin_fminf16(__builtin_fmaxf16(L, (half)1.0f), __builtin_bit_cast(half, (ushort)0x41ff));
    const half q = ((half)-1.0f - m) / (m + (half)-3.0f);
    const half t = L >= (half)1.0f ? q : L;
    const half k = __builtin_fabsf16(L) >= __builtin_bit_cast(half, (ushort)0x0002) ? ((half)4.0f * t) / L : (half)4.0f;
    const half lim = __builtin_bit_cast(half, (ushort)0x70e2);
    return (half4)(__builtin_fminf16(k * cx, lim), __builtin_fminf16(k * cy, lim), __builtin_fminf16(k * cz, lim), (half)1.0f);
}
static inline float lumf(float4v c) { return (c.x + c.y + c.z) * (1.0f / 3.0f); }
#ifndef TS
#define TS 16                          // tile side: TS x TS threads; the gated image of the tile + MB pixels, the roughness of the tile + MB + 1
#endif
#ifndef ACCG
#define ACCG 0.35f                     // the mirror accumulation guard: history and current 3x3 median luminance differing by more
#endif                                 // (relative) start following the current frame
#ifndef TCLO
#define TCLO 0.7f                     // the temporal outlier clamp's range (flag 512): [TCLO x min, TCHI x max] of the neighbours' running means
#endif
#ifndef TCHI
#define TCHI 1.4f
#endif
#ifndef BSK
#define BSK 1.33f                     // the strict sparkle removal (flag 1024): brighter than BSK x the second brightest neighbour
#endif
#ifndef DSK
#define DSK 0.75f                     // the dark despeckle's threshold (flag 256): darker than DSK x the second darkest neighbour
#endif
#ifndef MEDK
#define MEDK 3                         // the mirror median's window (flag 32): 3 or 5
#endif
#define MB (MEDK == 5 ? 2 : 1)         // border of the gated image in shared memory (the despeckle needs 1, the 5x5 median 2)
#define TB (TS + 2 * MB)
#define TR (TS + 2 * MB + 2)
__attribute__((amdgpu_flat_work_group_size(TS * TS, TS * TS)))
kernel void nn_tailf(constant const char* tempSrc, constant const char* finSrc, constant const char* specular, constant const char* diffuse,
                     constant const char* rough, constant const char* exposure, constant const char* out0, constant const char* outK,
                     constant const char* mhist, constant const char* motion, constant const char* pxPrev, constant const char* pxCur,
                     constant const TailFPar* par) {
    local half4 mir[TB * TB];
    local float rgh[TR * TR];
    local half mwt[TB * TB];   // the mirror gate's weight (flag 32)
    const int lx = (int)__builtin_amdgcn_workitem_id_x(), ly = (int)__builtin_amdgcn_workitem_id_y(), tid = ly * TS + lx;
    const int x0 = __builtin_amdgcn_workgroup_id_x() * TS, y0 = __builtin_amdgcn_workgroup_id_y() * TS;
    const uint fl = par->flags;
    if (fl & 1u) {
        for (int i = tid; i < TR * TR; i += TS * TS)   // roughness around the border pixels' clamped coordinates
            rgh[i] = (float)(half)ld(rough, clampi(x0 + i % TR - MB - 1, 0, W - 1), clampi(y0 + i / TR - MB - 1, 0, H - 1)).x;
        BARRIER();
    }
    for (int i = tid; i < TB * TB; i += TS * TS) {   // the tile and its border, at clamped coordinates (the despeckle clamps its neighbours)
        const int x = clampi(x0 + i % TB - MB, 0, W - 1), y = clampi(y0 + i / TB - MB, 0, H - 1);
        const half4 dif = ld16(diffuse, x, y), spe = ld16(specular, x, y);
        const half4 fh = remodulate(ld(finSrc, x, y), dif, spe);
        const float4v f = (float4v)((float)fh.x, (float)fh.y, (float)fh.z, (float)fh.w);
        float mirror = 0.0f;
        if (fl & 1u) {
            float r = 0.0f;
            for (int dy = -1; dy <= 1; ++dy)
                for (int dx = -1; dx <= 1; ++dx) {   // roughness at clamp(x + dx), clamp(y + dy): its index in the tile (x, y already clamped)
                    const int rx = clampi(x + dx, 0, W - 1) - x0 + MB + 1, ry = clampi(y + dy, 0, H - 1) - y0 + MB + 1;
                    r += rgh[ry * TR + rx];
                }
            r = DIV9(r);
            const float t = clampf(DIVT(r - par->r0, __builtin_fmaxf(par->r1 - par->r0, 1e-4f)), 0.0f, 1.0f);
            mirror = SMOOTHSTEP1(t) * par->strength;
        }
        half4 o = fh;
        if (mirror > 0.0f) {
            const half4 gh = remodulate(ld(tempSrc, x, y), dif, spe);
            o = (half4)((half)MIX(f.x, (float)gh.x, mirror), (half)MIX(f.y, (float)gh.y, mirror), (half)MIX(f.z, (float)gh.z, mirror), fh.w);
        }
        mir[i] = o;
        mwt[i] = (half)mirror;
    }
    BARRIER();
    const int x = x0 + lx, y = y0 + ly;
    if (x >= W || y >= H) return;
    const half4 ch = mir[(ly + MB) * TB + lx + MB];
    const float4v c = (float4v)((float)ch.x, (float)ch.y, (float)ch.z, (float)ch.w);
    float3 rgb = (float3)(c.x, c.y, c.z);
    // mirror median (m4r, not MetalFX): the gate keeps reflections sharp by skipping the spatial filter, which also keeps the ray noise
    // (single dark / bright pixels on water); a 5x5 median (MEDK) of the gated image removes it and keeps edges, weighted by the gate
    const float mw = (float)mwt[(ly + MB) * TB + lx + MB];
    // mirror accumulation (m4r, not MetalFX; flag 64, while the camera is still): the gated pixels average over frames at the same pixel
    // instead of being filtered spatially, which removes the ray noise and keeps the reflection sharp. The history follows the current
    // frame where it is far from the current 3x3 median (something moved in the reflection). Without it (camera moving) the median below.
    bool accumulated = false;
    // still accumulation (m4r, not MetalFX; flag 128, camera still): MetalFX trusts its history at most about 80% per frame, which leaves
    // low-frequency shading noise boiling on large soft surfaces (Control's characters at 853x480); pixels that do not move (motion vector
    // under a quarter pixel, fading out by half a pixel) average over more frames at the same pixel, with the same guard. Temporal only:
    // no neighbour is mixed in, so texture detail stays as it is.
    float wAll = 0.0f;
    if (fl & 128u) {
        const half4 mv = ld16(motion, x, y);
        const float px2 = (float)mv.x * par->msx, py2 = (float)mv.y * par->msy, mlen = __builtin_sqrtf(px2 * px2 + py2 * py2);
        wAll = clampf(DIV(0.5f - mlen, 0.25f), 0.0f, 1.0f);
    }
    const float mwa = (fl & 64u) ? mw : 0.0f;
    const float wAcc = __builtin_fmaxf(mwa, wAll);
    if (((fl & 64u) && mw > 0.0f) || (fl & 128u)) {
        // the guard compares like with like: the current 3x3 median luminance against the one stored with the history (its alpha),
        // so single noisy pixels in either do not count as a change
        float pc[9];
        for (int k = 0; k < 9; ++k) {
            const half4 nh = mir[(ly + MB - 1 + k / 3) * TB + lx + MB - 1 + k % 3];
            pc[k] = ((float)nh.x + (float)nh.y + (float)nh.z) * (1.0f / 3.0f);
        }
        const float lr = med9(pc);
        // non-mirror pixels (flag 128) accumulate only the 3x3 mean: the denoised image still carries the frame's jitter (the upscaler resolves
        // it), so averaging whole pixels over frames would also average sub-pixel-shifted detail (softer textures); the boiling is coarse,
        // so the 3x3 mean is averaged over frames and the frame's own detail (pixel minus its 3x3 mean) is kept as it is
        const bool lfOnly = (fl & 128u) && mwa <= 0.0f;
        float3 cur = rgb, nlo = rgb, nhi = rgb;
        if (lfOnly) {
            float3 s = (float3)(0.0f, 0.0f, 0.0f);
            for (int k = 0; k < 9; ++k) {
                const half4 nh = mir[(ly + MB - 1 + k / 3) * TB + lx + MB - 1 + k % 3];
                const float3 q = (float3)((float)nh.x, (float)nh.y, (float)nh.z);
                s += q; nlo = __builtin_elementwise_min(nlo, q); nhi = __builtin_elementwise_max(nhi, q);
            }
            cur = s * (1.0f / 9.0f);
        }
        float3 acc3 = cur;
        if (par->acc < 1.0f && wAcc > 0.0f) {   // acc 1: the first still frame, which only seeds the history
            const half4 hh = ld16(mhist, x, y);
            const float lh = (float)hh.w;
            const float dev = DIV(__builtin_fabsf(lh - lr), __builtin_fmaxf(__builtin_fmaxf(lr, lh), 1e-4f));
            const float tg = clampf(DIV(dev - ACCG, 0.4f), 0.0f, 1.0f);
            // mirror pixels: weight acc (down to 0.1); the others: at least accAll (0.2), a shorter memory for lighting that changes slowly
            const float base = lfOnly ? __builtin_fmaxf(par->acc, par->accAll) : par->acc;
            const float a = MIX(base, 1.0f, tg * tg * (3.0f - 2.0f * tg));
            acc3 = (float3)(MIX((float)hh.x, cur.x, a), MIX((float)hh.y, cur.y, a), MIX((float)hh.z, cur.z, a));
            // the corrected pixel stays within its 3x3 neighbourhood of this frame: at silhouettes the 3x3 mean moves with the jitter and an
            // unclamped correction over- or undershoots (dark dots along edges)
            const float3 m = lfOnly ? __builtin_elementwise_min(__builtin_elementwise_max(rgb + (acc3 - cur), nlo), nhi) : acc3;
            rgb = (float3)(MIX(rgb.x, __builtin_fmaxf(m.x, 0.0f), wAcc), MIX(rgb.y, __builtin_fmaxf(m.y, 0.0f), wAcc), MIX(rgb.z, __builtin_fmaxf(m.z, 0.0f), wAcc));
            accumulated = mw > 0.0f;
        }
        const float3 sv = lfOnly ? acc3 : rgb;   // the history: the accumulated 3x3 mean (flag 128) or the result (mirror pixels)
        st16(mhist, x, y, (half4)((half)sv.x, (half)sv.y, (half)sv.z, (half)lr));
    }
    if ((fl & 32u) && mw > 0.0f && !accumulated) {
        float pr[MEDK * MEDK], pg[MEDK * MEDK], pb[MEDK * MEDK];
        for (int k = 0; k < MEDK * MEDK; ++k) {
            const half4 nh = mir[(ly + MB - MEDK / 2 + k / MEDK) * TB + lx + MB - MEDK / 2 + k % MEDK];
            pr[k] = (float)nh.x; pg[k] = (float)nh.y; pb[k] = (float)nh.z;
        }
#if MEDK == 5
        rgb = (float3)(MIX(rgb.x, med25(pr), mw), MIX(rgb.y, med25(pg), mw), MIX(rgb.z, med25(pb), mw));
#else
        rgb = (float3)(MIX(rgb.x, med9(pr), mw), MIX(rgb.y, med9(pg), mw), MIX(rgb.z, med9(pb), mw));
#endif
    }
    if (fl & 2u) {
        float top0 = 0.0f, top1 = 0.0f, top2 = 0.0f;
        for (int dy = -1; dy <= 1; ++dy)
            for (int dx = -1; dx <= 1; ++dx) {
                if (dx == 0 && dy == 0) continue;
                const half4 nh = mir[(ly + MB + dy) * TB + lx + MB + dx];
                const float l = lumf((float4v)((float)nh.x, (float)nh.y, (float)nh.z, (float)nh.w));
                if (l > top0) { top2 = top1; top1 = top0; top0 = l; }
                else if (l > top1) { top2 = top1; top1 = l; }
                else if (l > top2) top2 = l;
            }
        const float l = lumf(c), cap = FMA(top2, 1.5f, 1e-4f);
        if (l > cap) rgb = rgb * DIV(cap, l);
        // strict sparkle removal (m4r; flag 1024): single bright pixels (floors, grain behind glass) capped at BSK x the second brightest
        // neighbour; pixels on thin bright lines have two bright neighbours and stay
        if (fl & 1024u) {
            const float l2 = (rgb.x + rgb.y + rgb.z) * (1.0f / 3.0f), cap2 = FMA(top1, BSK, 1e-4f);
            if (l2 > cap2) rgb = rgb * DIV(cap2, l2);
        }
    }
    if (fl & 256u) {
        // dark despeckle (m4r, not MetalFX; flag 256): single dark dropouts (hair at low render sizes: strand normals stop the bilateral
        // filter, so pixels keep the ray noise) lifted to their second darkest neighbour. Pixels on thin dark lines have two dark neighbours
        // and stay; only a pixel darker than DSK (0.75) x its second darkest neighbour changes, nothing else is touched (no filter).
        float bot0 = 3.0e38f, bot1 = 3.0e38f;
        for (int dy = -1; dy <= 1; ++dy)
            for (int dx = -1; dx <= 1; ++dx) {
                if (dx == 0 && dy == 0) continue;
                const half4 nh = mir[(ly + MB + dy) * TB + lx + MB + dx];
                const float l = ((float)nh.x + (float)nh.y + (float)nh.z) * (1.0f / 3.0f);
                if (l < bot0) { bot1 = bot0; bot0 = l; }
                else if (l < bot1) bot1 = l;
            }
        const float l = (rgb.x + rgb.y + rgb.z) * (1.0f / 3.0f);
        if (l < DSK * bot1) {
            if (l > 1.0e-6f) rgb = rgb * DIV(bot1, l);
            else rgb = (float3)(bot1, bot1, bot1);
        }
    }
    {
        // temporal outlier clamp (m4r, not MetalFX; flag 512, camera still and the pixel not moving): pixels that jump for one frame (dark
        // clusters on skin and cloth, single sparkles on floors, grain behind glass) on a scene that does not change. The pixel's luminance
        // is kept within [0.7, 1.4] x the range of its 3x3 neighbours' running means (TCLO, TCHI): a jitter-shifted edge or texture pixel takes values
        // its neighbours have, so it stays inside; a one-frame outlier does not. Nothing is filtered. A real change (the 3x3 median of the
        // frame far from the median of the running means: light, shadow, movement) follows the frame and restarts the means.
        float l = (rgb.x + rgb.y + rgb.z) * (1.0f / 3.0f);
        float mp[9], cl[9];
        float lo = 3.0e38f, hi = 0.0f;
        for (int k = 0; k < 9; ++k) {
            const int qx = clampi(x + k % 3 - 1, 0, W - 1), qy = clampi(y + k / 3 - 1, 0, H - 1);
            mp[k] = ld(pxPrev, qx, qy).x;
            lo = __builtin_fminf(lo, mp[k]); hi = __builtin_fmaxf(hi, mp[k]);
            const half4 nh = mir[(ly + MB - 1 + k / 3) * TB + lx + MB - 1 + k % 3];
            cl[k] = ((float)nh.x + (float)nh.y + (float)nh.z) * (1.0f / 3.0f);
        }
        const float mprev = mp[4];
        const float medP = med9(mp), medC = med9(cl);
        const float dev = DIV(__builtin_fabsf(medP - medC), __builtin_fmaxf(__builtin_fmaxf(medP, medC), 1e-4f));
        const bool warm = (fl & 512u) && wAll > 0.0f && dev < ACCG && mprev > 0.0f;
        if (warm) {
            const float a = TCLO * lo, b = TCHI * hi;
            const float lc = clampf(l, a, b);
            if (lc != l) {
                if (l > 1.0e-6f) rgb = rgb * DIV(lc, l);
                else rgb = (float3)(lc, lc, lc);
                l = lc;
            }
        }
        st32x(pxCur, x, y, warm ? MIX(mprev, l, 0.25f) : l);
    }
    half3 h = (half3)((half)rgb.x, (half)rgb.y, (half)rgb.z);
    float3 v = (float3)((float)h.x, (float)h.y, (float)h.z);
    if (fl & 4u) {
        const float e = ld(exposure, 0, 0).x;
        if (e > 0.0f) {
            v = (float3)(clampf(DIV(v.x, e), 0.0f, 65504.0f), clampf(DIV(v.y, e), 0.0f, 65504.0f), clampf(DIV(v.z, e), 0.0f, 65504.0f));
            h = (half3)((half)v.x, (half)v.y, (half)v.z);
        }
    }
    const half4 o = (half4)(h.x, h.y, h.z, ch.w);
    if (!(fl & 16u)) st16(out0, x, y, o);
    if (fl & 8u) st16(outK, x, y, o);
}
#endif

#if defined(LOWRES)
// LOWRES: PREP and MetalFX's dbfnetv1_lowres_signals (is_tensor_layout_nchw false, spec_hit_dist_enabled false, new_overshoot_clamp false)
// in one pass. Per tensor pixel, its 2x2 render pixels: PREP's colour * exposure (stored for reproject_warp) and linear -> hardware depth (the
// int_depth Apple's kernel copies the depth into, and the game-convention depth for DLSS), then Apple's signals: range-reduced colour and
// reprojected history, specular albedo, normal, roughness, max-pooled over the 2x2 block into the 13-channel NHWC tensor; motion copied into
// int_motion (Apple's zero mirror mask: see below). Formulas and fp16 steps as the translated kernel (LLVM's fp16 division for its fast-math half divides).
typedef struct { float ms[2], pj[2], cj[2]; int rowBytes; ushort th, tw; uchar reset; } DbfPar;   // dbfnet_v1_build_tensors_t
static inline half4 ld16l(constant const char* rec, int x, int y) { return __builtin_amdgcn_image_load_2d_v4f16_i32(15, x, y, *(constant const tex_t*)rec, 0, 0); }
static inline half rr(half c) {   // MetalFX's range reduction: c >= 1 -> 4m / (m + 1) - 1 with m = max(c, 1)
    const half m = __builtin_fmaxf16(c, (half)1.0f);
    const half t = (m * (half)4.0f) / (m + (half)1.0f) + (half)-1.0f;
    return c >= (half)1.0f ? t : c;
}
typedef int int4v __attribute__((ext_vector_type(4)));
__attribute__((visibility("protected"))) global int4v __m4r_sampler_0 = (int4v)(0);   // linear, clamp to edge, normalised (history)
__attribute__((amdgpu_flat_work_group_size(64, 64)))
kernel void nn_lowres(constant const char* normals, constant const char* motion, constant const char* colorGame, constant const char* exposure,
                      constant const char* colorExp, constant const char* depthGame, constant const char* intDepth, constant const char* depthK,
                      constant const char* intMotion, constant const char* specular, constant const char* hist, constant const char* mirror,
                      constant const char* roughTex, constant const char* nrmFix, constant const char* nrmPx,
                      constant const PrepPar* par, constant const DbfPar* dp, constant const half* expo, constant const float* preexp,
                      global half* tensor) {
    const int gx = __builtin_amdgcn_workgroup_id_x() * 8 + (int)__builtin_amdgcn_workitem_id_x();
    const int gy = __builtin_amdgcn_workgroup_id_y() * 8 + (int)__builtin_amdgcn_workitem_id_y();
    const int TW = dp->tw, TH = dp->th;
    if (gx >= TW || gy >= TH) return;
    const uint fl = par->flags;
    const float e = ld(exposure, 0, 0).x, scale = e > 0.0f ? e : 1.0f;
    const half E = expo[0], P = (half)preexp[0];
    const float msx = dp->ms[0], msy = dp->ms[1];
    const float imw = __builtin_amdgcn_rcpf((float)par->mw), imh = __builtin_amdgcn_rcpf((float)par->mh);   // 1 / motion texture size
    const float iuw = __builtin_amdgcn_rcpf((float)(2 * TW)), iuh = __builtin_amdgcn_rcpf((float)(2 * TH));
    const int4v hs = __m4r_sampler_0;
    // the four render pixels stage by stage (all loads, then the arithmetic, then the stores), so that the loads of all four are in flight
    // together; flag tests are uniform branches around whole stages, and pixels outside the image are masked rather than branched around
    int px[4], py[4]; bool in[4];
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        px[k] = 2 * gx + (k >> 1); py[k] = 2 * gy + (k & 1);
        in[k] = px[k] < W && py[k] < H;
        px[k] = px[k] < W ? px[k] : W - 1; py[k] = py[k] < H ? py[k] : H - 1;
    }
    half3 c[4];
    if (fl & 2u) {
        float4v cg[4];
#pragma unroll
        for (int k = 0; k < 4; ++k) cg[k] = ld(colorGame, px[k], py[k]);
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            c[k] = (half3)((half)clampf(cg[k].x * scale, 0.0f, 65504.0f), (half)clampf(cg[k].y * scale, 0.0f, 65504.0f), (half)clampf(cg[k].z * scale, 0.0f, 65504.0f));
            if (in[k]) st16(colorExp, px[k], py[k], (half4)(c[k].x, c[k].y, c[k].z, (half)1.0f));
        }
    } else {
#pragma unroll
        for (int k = 0; k < 4; ++k) c[k] = ld16l(colorGame, px[k], py[k]).xyz;
    }
    float d[4];
#pragma unroll
    for (int k = 0; k < 4; ++k) d[k] = ld(depthGame, px[k], py[k]).x;
    if (fl & 4u) {
        const float nearZ = par->nearZ, m23 = par->m23, m33 = par->m33, sky = par->sky;
        const float m22 = par->m22, m32 = par->m32, m23k = par->m23k, m33k = par->m33k;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const float lin = d[k];
            float dh = 0.0f;
            if (lin > 0.0f && lin < sky) {
                const float w = fma(lin, m23, m33);
                if (w > 0.0f) dh = DIV(nearZ, w);
            }
            d[k] = dh;
            if (fl & 8u) {
                float dk = clampf(m22, 0.0f, 1.0f);
                if (lin > 0.0f && lin < sky) {
                    const float w = fma(lin, m23k, m33k);
                    if (w > 0.0f) dk = clampf(DIV(fma(lin, m22, m32), w), 0.0f, 1.0f);
                }
                if (in[k]) st32x(depthK, px[k], py[k], dk);
            }
        }
    }
    half rough[4]; half3 spec[4], nrm[4]; half4 mv[4];
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        rough[k] = ld16l(roughTex, px[k], py[k]).x; spec[k] = ld16l(specular, px[k], py[k]).xyz;
        const half4 n4 = ld16l(normals, px[k], py[k]);
        nrm[k] = n4.xyz;
        mv[k] = ld16l(motion, px[k], py[k]);
        // normals that are not unit length (Control: ~40% of pixels at length 0.72-0.78) made unit length (m4r, not MetalFX). MetalFX's
        // reproject_warp normalises only lengths in (0.99, 1.005) and compares pow(dot(n, n_prev), 128): two 0.75-long normals give at most
        // 0.56^128 = 0, so every such pixel threw its history away every frame. Lengths in that range (and zero normals) pass through
        // unchanged, so games with unit normals are untouched. reproject_warp and the network both read the result.
        const float lx = (float)nrm[k].x, ly = (float)nrm[k].y, lz = (float)nrm[k].z, l2 = lx * lx + ly * ly + lz * lz;
        if (l2 > 1.0e-8f && !(l2 > 0.99f * 0.99f && l2 < 1.005f * 1.005f)) {
            const float il = __builtin_amdgcn_rsqf(l2);
            nrm[k] = (half3)((half)(lx * il), (half)(ly * il), (half)(lz * il));
        }
#if NSMOOTH
        // the normal reproject_warp compares and stores (m4r, not MetalFX): the 3x3 mean of the unit normals at the same depth (within 2%),
        // normalised. At low render sizes the game's normal maps change with the jitter from frame to frame (Control at 853x480: 26% of the
        // character's and 52% of the floor's pixels failed MetalFX's pow(dot, 128) test on a still camera, so they kept no history and
        // flickered); the mean keeps silhouettes and creases (other depth) apart. The network still reads the per-pixel normal.
        {
            const float dc = ld(depthGame, px[k], py[k]).x, tol = 0.02f * __builtin_fabsf(dc);
            float sx = 0.0f, sy = 0.0f, sz = 0.0f;
#pragma unroll
            for (int j = 0; j < 9; ++j) {
                int qx = px[k] + j % 3 - 1, qy = py[k] + j / 3 - 1;
                qx = qx < 0 ? 0 : qx >= W ? W - 1 : qx; qy = qy < 0 ? 0 : qy >= H ? H - 1 : qy;
                const half3 q = ld16l(normals, qx, qy).xyz;
                const float ax = (float)q.x, ay = (float)q.y, az = (float)q.z, a2 = ax * ax + ay * ay + az * az;
                const float w = (a2 > 1.0e-8f && __builtin_fabsf(ld(depthGame, qx, qy).x - dc) <= tol) ? __builtin_amdgcn_rsqf(a2) : 0.0f;
                sx = fma(ax, w, sx); sy = fma(ay, w, sy); sz = fma(az, w, sz);
            }
            const float s2 = sx * sx + sy * sy + sz * sz;
            half3 ns = nrm[k];
            if (s2 > 1.0e-8f) { const float is = __builtin_amdgcn_rsqf(s2); ns = (half3)((half)(sx * is), (half)(sy * is), (half)(sz * is)); }
            if (in[k]) st16(nrmFix, px[k], py[k], (half4)(ns.x, ns.y, ns.z, n4.w));
            // the bilateral passes stop at normal edges: they get the per-pixel unit normal and the roughness (what reproject_warp stores in
            // normal_roughness_tex for them), not the smoothed one (that blurred normal-mapped detail such as Control's floor plates)
            if (in[k]) st16(nrmPx, px[k], py[k], (half4)(nrm[k].x, nrm[k].y, nrm[k].z, rough[k]));
        }
#else
        if (in[k]) st16(nrmFix, px[k], py[k], (half4)(nrm[k].x, nrm[k].y, nrm[k].z, n4.w));
#endif
    }
    half3 hs4[4];
    if (!dp->reset) {
        const float pjx = dp->pj[0], pjy = dp->pj[1], cjx = dp->cj[0], cjy = dp->cj[1];
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const float u = fma(fma(msx, (float)mv[k].x, pjx) - cjx, imw, ((float)(2 * gx + (k >> 1)) + 0.5f) * iuw);
            const float v = fma(fma(msy, (float)mv[k].y, pjy) - cjy, imh, ((float)(2 * gy + (k & 1)) + 0.5f) * iuh);
            hs4[k] = ((half4)__builtin_amdgcn_image_sample_lz_2d_v4f16_f32(15, u, v, *(constant const tex_t*)hist, hs, false, 0, 0)).xyz;   // LLVM 24 returns a GCC vector
        }
    }
    half3 aN = (half3)(-INFINITY), aC = (half3)(-INFINITY), aH = (half3)(-INFINITY), aS = (half3)(-INFINITY);
    half aR = -INFINITY;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const half3 cr = __builtin_elementwise_min((E * c[k]) / P, (half3)(half)1.0e4f);
        const half3 crr = (half3)(rr(cr.x), rr(cr.y), rr(cr.z));
        const half3 h = dp->reset ? crr : (half3)(rr(hs4[k].x), rr(hs4[k].y), rr(hs4[k].z));
        if (in[k]) {
            aC = __builtin_elementwise_max(aC, crr); aS = __builtin_elementwise_max(aS, spec[k]); aR = __builtin_fmaxf16(aR, rough[k]);
            aN = __builtin_elementwise_max(aN, nrm[k]); aH = __builtin_elementwise_max(aH, h);
            st32x(intDepth, px[k], py[k], d[k]);
            if (!(fl & 16u)) st16(intMotion, px[k], py[k], (half4)(mv[k].x, mv[k].y, mv[k].y, mv[k].y));   // 16: int_motion is the game's motion
            // Apple's zero mirror mask is not stored: nothing else writes mm0/mm1 and the plan clears every texture once
        }
    }
    #define FIN3(a) ((((__builtin_bit_cast(ushort, a.x) & 0x7c00) != 0x7c00) && ((__builtin_bit_cast(ushort, a.y) & 0x7c00) != 0x7c00) && ((__builtin_bit_cast(ushort, a.z) & 0x7c00) != 0x7c00)) ? a : (half3)(half)0.0f)
    const half3 n = FIN3(aN), cc = FIN3(aC), hh = FIN3(aH), ss = FIN3(aS);
    const half r = (__builtin_bit_cast(ushort, aR) & 0x7c00) == 0x7c00 ? (half)0.0f : aR;
#ifdef TSTRIDE
    // m4r's layout for its own network kernels: TSTRIDE halves per pixel (13 used, the rest zero), two aligned 16-byte stores
    typedef half half8t __attribute__((ext_vector_type(8)));
    global half8t* t8 = (global half8t*)(tensor + ((size_t)gy * TW + gx) * TSTRIDE);
    t8[0] = (half8t)(n.x, n.y, n.z, cc.x, cc.y, cc.z, hh.x, hh.y);
    t8[1] = (half8t)(hh.z, ss.x, ss.y, ss.z, r, (half)0.0f, (half)0.0f, (half)0.0f);
#else
    global half* t = tensor + ((size_t)gy * TW + gx) * (dp->rowBytes / 2);
    t[0] = n.x; t[1] = n.y; t[2] = n.z; t[3] = cc.x; t[4] = cc.y; t[5] = cc.z; t[6] = hh.x; t[7] = hh.y; t[8] = hh.z;
    t[9] = ss.x; t[10] = ss.y; t[11] = ss.z; t[12] = r;
#endif
}
#endif
