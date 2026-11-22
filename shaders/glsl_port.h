// m4r: shared part of the OpenCL C ports of m4r's GLSL passes (shaders/nn_*.cl, for native Windows where HIP cannot run SPIR-V).
// A port reads the same argument block as its GLSL pass (8 x u64 slots, 8 x u32 constants, grid) through its single kernel argument,
// and does its arithmetic the way RADV lowers the GLSL: division as a multiply by the reciprocal, exp/log/pow through the hardware's
// exp2/log2, mix as one fma, inversesqrt and sqrt as the hardware instructions. No OpenCL library is linked.
// SPDX-License-Identifier: GPL-3.0-only
#pragma once
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma clang fp contract(off)

typedef half half2v __attribute__((ext_vector_type(2)));
typedef half half4v __attribute__((ext_vector_type(4)));
typedef float float2v __attribute__((ext_vector_type(2)));
typedef float float3v __attribute__((ext_vector_type(3)));
typedef float float4v __attribute__((ext_vector_type(4)));

typedef struct { ulong slot[8]; uint c[8]; uint grid[4]; } Args;

#ifndef KNAME
#error KNAME (the kernel name, the plan's glsl name) is set by air/m4r_air.py glsl_port
#endif
#define PORT_KERNEL __attribute__((amdgpu_flat_work_group_size(64, 64))) kernel void KNAME(constant const Args* a)
// gl_GlobalInvocationID for the 8 x 8 x 1 workgroups every nn_*.comp declares
#define GID_X ((uint)__builtin_amdgcn_workgroup_id_x() * 8u + (uint)__builtin_amdgcn_workitem_id_x())
#define GID_Y ((uint)__builtin_amdgcn_workgroup_id_y() * 8u + (uint)__builtin_amdgcn_workitem_id_y())
#define GID_Z ((uint)__builtin_amdgcn_workgroup_id_z())

#define BUF(T, i) ((global T*)a->slot[i])

static inline float g_min(float x, float y) { return __builtin_fminf(x, y); }
static inline float g_max(float x, float y) { return __builtin_fmaxf(x, y); }
static inline float g_clamp(float x, float lo, float hi) { return __builtin_fminf(__builtin_fmaxf(x, lo), hi); }
static inline uint g_minu(uint x, uint y) { return x < y ? x : y; }
static inline int g_clampi(int x, int lo, int hi) { return x < lo ? lo : (x > hi ? hi : x); }
static inline float g_div(float x, float y) { return x * __builtin_amdgcn_rcpf(y); }
static inline float g_exp(float x) { return __builtin_amdgcn_exp2f(x * 1.44269502f); }
static inline float g_log(float x) { return __builtin_amdgcn_logf(x) * 0.693147182f; }
static inline float g_pow(float x, float y) { return __builtin_amdgcn_exp2f(y * __builtin_amdgcn_logf(x)); }
static inline float g_rsqrt(float x) { return __builtin_amdgcn_rsqf(x); }
static inline float g_sqrt(float x) { return __builtin_amdgcn_sqrtf(x); }
static inline float g_mix(float x, float y, float t) { return __builtin_fmaf(t, y - x, x); }
static inline float g_smoothstep(float e0, float e1, float x) {
    const float t = g_clamp(g_div(x - e0, e1 - e0), 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}
static inline float g_floor(float x) { return __builtin_floorf(x); }
static inline float g_abs(float x) { return __builtin_fabsf(x); }
static inline float3v rgb(half4v v) { return (float3v)((float)v.x, (float)v.y, (float)v.z); }
static inline float4v f4(half4v v) { return (float4v)((float)v.x, (float)v.y, (float)v.z, (float)v.w); }
static inline half4v h4(float4v v) { return (half4v)((half)v.x, (half)v.y, (half)v.z, (half)v.w); }
static inline half4v h4c(float3v c, float w) { return (half4v)((half)c.x, (half)c.y, (half)c.z, (half)w); }
static inline float lum3(float3v c) { return (c.x + c.y + c.z) * (1.0f / 3.0f); }
static inline float3v mix3(float3v x, float3v y, float t) { return (float3v)(g_mix(x.x, y.x, t), g_mix(x.y, y.y, t), g_mix(x.z, y.z, t)); }
