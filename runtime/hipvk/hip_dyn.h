// m4r: the HIP runtime functions m4r uses on Windows, loaded at run time from AMD's driver (amdhip64_7.dll, else amdhip64_6.dll).
// The declarations repeat the ABI of the ROCm 6/7 headers (hip_runtime_api.h, driver_types.h, texture_types.h) so the DLL builds
// without the HIP SDK and never imports the HIP runtime at load time (d4r: loading it next to OptiScaler's hooks before the game's
// first frame can hang the GPU).
// SPDX-License-Identifier: GPL-3.0-only
#pragma once
#include <cstddef>
#include <cstdint>

namespace hipdyn {

typedef int hipError_t;   // 0 = hipSuccess
typedef struct ihipStream_t* hipStream_t;
typedef struct ihipModule_t* hipModule_t;
typedef struct ihipModuleSymbol_t* hipFunction_t;
typedef struct ihipEvent_t* hipEvent_t;
typedef void* hipExternalMemory_t;
typedef void* hipExternalSemaphore_t;
typedef struct __hip_texture* hipTextureObject_t;

enum hipMemcpyKind { hipMemcpyHostToHost = 0, hipMemcpyHostToDevice = 1, hipMemcpyDeviceToHost = 2, hipMemcpyDeviceToDevice = 3, hipMemcpyDefault = 4 };
enum hipChannelFormatKind { hipChannelFormatKindSigned = 0, hipChannelFormatKindUnsigned = 1, hipChannelFormatKindFloat = 2, hipChannelFormatKindNone = 3 };
struct hipChannelFormatDesc { int x, y, z, w; hipChannelFormatKind f; };
enum hipResourceType { hipResourceTypeArray = 0, hipResourceTypeMipmappedArray = 1, hipResourceTypeLinear = 2, hipResourceTypePitch2D = 3 };
struct hipResourceDesc {
    hipResourceType resType;
    union {
        struct { void* array; } array;
        struct { void* mipmap; } mipmap;
        struct { void* devPtr; hipChannelFormatDesc desc; size_t sizeInBytes; } linear;
        struct { void* devPtr; hipChannelFormatDesc desc; size_t width, height, pitchInBytes; } pitch2D;
    } res;
};
enum hipTextureAddressMode { hipAddressModeWrap = 0, hipAddressModeClamp = 1, hipAddressModeMirror = 2, hipAddressModeBorder = 3 };
enum hipTextureFilterMode { hipFilterModePoint = 0, hipFilterModeLinear = 1 };
enum hipTextureReadMode { hipReadModeElementType = 0, hipReadModeNormalizedFloat = 1 };
struct hipTextureDesc {
    hipTextureAddressMode addressMode[3];
    hipTextureFilterMode filterMode;
    hipTextureReadMode readMode;
    int sRGB;
    float borderColor[4];
    int normalizedCoords;
    unsigned int maxAnisotropy;
    hipTextureFilterMode mipmapFilterMode;
    float mipmapLevelBias, minMipmapLevelClamp, maxMipmapLevelClamp;
};
enum hipExternalMemoryHandleType { hipExternalMemoryHandleTypeOpaqueWin32 = 2, hipExternalMemoryHandleTypeD3D12Heap = 4, hipExternalMemoryHandleTypeD3D12Resource = 5 };
struct hipExternalMemoryHandleDesc {
    hipExternalMemoryHandleType type;
    union { int fd; struct { void* handle; const void* name; } win32; const void* nvSciBufObject; } handle;
    unsigned long long size;
    unsigned int flags;
    unsigned int reserved[16];
};
struct hipExternalMemoryBufferDesc { unsigned long long offset, size; unsigned int flags; unsigned int reserved[16]; };
enum hipExternalSemaphoreHandleType { hipExternalSemaphoreHandleTypeD3D12Fence = 4, hipExternalSemaphoreHandleTypeTimelineSemaphoreWin32 = 10 };
struct hipExternalSemaphoreHandleDesc {
    hipExternalSemaphoreHandleType type;
    union { int fd; struct { void* handle; const void* name; } win32; const void* NvSciSyncObj; } handle;
    unsigned int flags;
    unsigned int reserved[16];
};
struct hipExternalSemaphoreSignalParams {
    struct { struct { unsigned long long value; } fence; union { void* fence; unsigned long long reserved; } nvSciSync;
             struct { unsigned long long key; } keyedMutex; unsigned int reserved[12]; } params;
    unsigned int flags;
    unsigned int reserved[16];
};
struct hipExternalSemaphoreWaitParams {
    struct { struct { unsigned long long value; } fence; union { void* fence; unsigned long long reserved; } nvSciSync;
             struct { unsigned long long key; unsigned int timeoutMs; } keyedMutex; unsigned int reserved[10]; } params;
    unsigned int flags;
    unsigned int reserved[16];
};

constexpr size_t kDevicePropBytes = 1472, kGcnArchNameOffset = 1160;   // hipDeviceProp_tR0600 (ROCm 6.0 - 7.x)
constexpr uint32_t hipExtAnyOrderLaunch = 1, hipStreamNonBlocking = 1, hipEventDisableTiming = 2;

#define HIPDYN_FUNCS(X) \
    X(hipInit, hipError_t, (unsigned)) \
    X(hipGetDeviceCount, hipError_t, (int*)) \
    X(hipSetDevice, hipError_t, (int)) \
    X(hipGetDevicePropertiesR0600, hipError_t, (void*, int)) \
    X(hipDeviceGetPCIBusId, hipError_t, (char*, int, int)) \
    X(hipGetErrorString, const char*, (hipError_t)) \
    X(hipMalloc, hipError_t, (void**, size_t)) \
    X(hipFree, hipError_t, (void*)) \
    X(hipMemcpy, hipError_t, (void*, const void*, size_t, hipMemcpyKind)) \
    X(hipMemcpyAsync, hipError_t, (void*, const void*, size_t, hipMemcpyKind, hipStream_t)) \
    X(hipMemcpy2DAsync, hipError_t, (void*, size_t, const void*, size_t, size_t, size_t, hipMemcpyKind, hipStream_t)) \
    X(hipMemsetAsync, hipError_t, (void*, int, size_t, hipStream_t)) \
    X(hipMemsetD32Async, hipError_t, (void*, int, size_t, hipStream_t)) \
    X(hipHostMalloc, hipError_t, (void**, size_t, unsigned)) \
    X(hipHostFree, hipError_t, (void*)) \
    X(hipStreamCreateWithFlags, hipError_t, (hipStream_t*, unsigned)) \
    X(hipStreamDestroy, hipError_t, (hipStream_t)) \
    X(hipStreamSynchronize, hipError_t, (hipStream_t)) \
    X(hipDeviceSynchronize, hipError_t, ()) \
    X(hipEventCreateWithFlags, hipError_t, (hipEvent_t*, unsigned)) \
    X(hipEventDestroy, hipError_t, (hipEvent_t)) \
    X(hipEventRecord, hipError_t, (hipEvent_t, hipStream_t)) \
    X(hipEventSynchronize, hipError_t, (hipEvent_t)) \
    X(hipEventQuery, hipError_t, (hipEvent_t)) \
    X(hipEventElapsedTime, hipError_t, (float*, hipEvent_t, hipEvent_t)) \
    X(hipModuleLoadData, hipError_t, (hipModule_t*, const void*)) \
    X(hipModuleUnload, hipError_t, (hipModule_t)) \
    X(hipModuleGetFunction, hipError_t, (hipFunction_t*, hipModule_t, const char*)) \
    X(hipModuleGetGlobal, hipError_t, (void**, size_t*, hipModule_t, const char*)) \
    X(hipModuleLaunchKernel, hipError_t, (hipFunction_t, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, hipStream_t, void**, void**)) \
    X(hipExtModuleLaunchKernel, hipError_t, (hipFunction_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, size_t, hipStream_t, void**, void**, hipEvent_t, hipEvent_t, uint32_t)) \
    X(hipCreateTextureObject, hipError_t, (hipTextureObject_t*, const hipResourceDesc*, const hipTextureDesc*, const void*)) \
    X(hipDestroyTextureObject, hipError_t, (hipTextureObject_t)) \
    X(hipImportExternalMemory, hipError_t, (hipExternalMemory_t*, const hipExternalMemoryHandleDesc*)) \
    X(hipExternalMemoryGetMappedBuffer, hipError_t, (void**, hipExternalMemory_t, const hipExternalMemoryBufferDesc*)) \
    X(hipDestroyExternalMemory, hipError_t, (hipExternalMemory_t)) \
    X(hipImportExternalSemaphore, hipError_t, (hipExternalSemaphore_t*, const hipExternalSemaphoreHandleDesc*)) \
    X(hipSignalExternalSemaphoresAsync, hipError_t, (const hipExternalSemaphore_t*, const hipExternalSemaphoreSignalParams*, unsigned, hipStream_t)) \
    X(hipWaitExternalSemaphoresAsync, hipError_t, (const hipExternalSemaphore_t*, const hipExternalSemaphoreWaitParams*, unsigned, hipStream_t)) \
    X(hipDestroyExternalSemaphore, hipError_t, (hipExternalSemaphore_t))

struct Api {
#define HIPDYN_MEMBER(name, ret, args) ret (*name) args = nullptr;
    HIPDYN_FUNCS(HIPDYN_MEMBER)
#undef HIPDYN_MEMBER
    void* module = nullptr;
    const char* err(hipError_t e) const { return hipGetErrorString ? hipGetErrorString(e) : "HIP error"; }
};

// Loads the runtime once (thread-safe). Null when no HIP runtime is installed or a function is missing; *why says which.
const Api* load(const char** why = nullptr);
// The LLVM target of HIP device 0 ("gfx1101"), without feature suffixes (":sramecc-:xnack-"); empty on failure.
const char* deviceArch();

} // namespace hipdyn
