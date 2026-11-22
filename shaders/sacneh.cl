// m4r: OpenCL C port of shaders/sacneh.comp for native Windows (HIP cannot run SPIR-V): the same filter, taps, order and fp16
// roundings, so the same bits. One kernel argument: the argument block's address (8 x u64 slots, 8 x u32 constants, grid), as the
// GLSL pass reads it through its push constant.
// SPDX-License-Identifier: GPL-3.0-only
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma clang fp contract(off)   // RADV (ACO) fuses GLSL's acc += t * v into one fma on RDNA: written as fma() here

#define BARRIER() do { __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup"); __builtin_amdgcn_s_barrier(); __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup"); } while (0)

typedef struct { ulong slot[8]; uint c[8]; uint grid[4]; } Args;

#define TW 32
#define TH 32
#define R 9
#define SW (TW + 2 * R)
#define SH (TH + 2 * R)

__attribute__((amdgpu_flat_work_group_size(256, 256)))
kernel void sacneh(constant const Args* a) {
    local half src_t[SH * SW];
    local half xf[SH * TW];
    local float tx[19], ty[19];
    const uint w = a->c[0], h = a->c[1], pitch = a->c[2], plane = a->c[3];
    const uint lx = __builtin_amdgcn_workitem_id_x(), ly = __builtin_amdgcn_workitem_id_y();
    const uint gx = __builtin_amdgcn_workgroup_id_x(), gy = __builtin_amdgcn_workgroup_id_y(), gz = __builtin_amdgcn_workgroup_id_z();
    const uint idx = gx * 32 + lx, idz = gz;
    const bool zok = idz < a->c[4];
    global const half* src = (global const half*)a->slot[0];
    global half* dst = (global half*)a->slot[1];
    global const half* tapx = (global const half*)a->slot[2];
    global const half* tapy = (global const half*)a->slot[3];
    const int g0x = (int)(gx * TW) - R, g0y = (int)(gy * TH) - R;
    const uint li = ly * 32 + lx;
    for (uint i = li; i < SH * SW; i += 256u) {
        const int x = g0x + (int)(i % SW), y = g0y + (int)(i / SW);
        src_t[i] = (zok && x >= 0 && y >= 0 && x < (int)w && y < (int)h) ? src[idz * plane + (uint)y * pitch + (uint)x] : (half)0.0f;
    }
    if (li < 19u) { tx[li] = (float)tapx[li]; ty[li] = (float)tapy[li]; }
    BARRIER();
    if (li < SH * (TW / 8u)) {
        const uint r = li / (TW / 8u), c0 = (li % (TW / 8u)) * 8u;
        float v[26];
#pragma unroll
        for (uint i = 0u; i < 26u; ++i) v[i] = (float)src_t[r * SW + c0 + i];
#pragma unroll
        for (uint o = 0u; o < 8u; ++o) {
            float acc = 0.0f;
#pragma unroll
            for (uint t = 0u; t < 19u; ++t) acc = __builtin_fmaf(tx[t], v[o + t], acc);
            xf[r * TW + c0 + o] = (half)acc;
        }
    }
    BARRIER();
    if (!zok || idx >= w) return;
    const uint oy0 = gy * TH + ly * 4u;
    float u[22];
#pragma unroll
    for (uint i = 0u; i < 22u; ++i) {
        const int y = (int)oy0 + (int)i - 9;
        u[i] = (y >= 0 && y < (int)h) ? (float)xf[(ly * 4u + i) * TW + lx] : 0.0f;
    }
#pragma unroll
    for (uint k = 0u; k < 4u; ++k) {
        const uint oy = oy0 + k;
        if (oy >= h) break;
        float acc = 0.0f;
#pragma unroll
        for (uint t = 0u; t < 19u; ++t) acc = __builtin_fmaf(ty[t], u[k + t], acc);
        dst[idz * plane + oy * pitch + idx] = (half)acc;
    }
}
