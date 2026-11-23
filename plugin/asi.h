// The OptiScaler plugin's shared state (asi.cpp: upscaler, framegen.cpp: frame generation; backend.h: which upscaler).
// SPDX-License-Identifier: GPL-3.0-only
#pragma once
#include "dx_interop.h"
#include "optiscaler_plugin.h"
#include "ffx_api/ffx_api.h"
#include "backend.h"
#include <mutex>
#include <string>
#include <vector>


namespace m4rasi {

// The second part of a split game list, reused once the GPU has retired it.
struct Part {
    ID3D12CommandAllocator* allocator = nullptr;
    ID3D12GraphicsCommandList* list = nullptr;
    ID3D12Fence* retire = nullptr;
    uint64_t retireValue = 0;
    bool busy = false;
    // what the HIP work between the parts runs (filled when the list is split, read when it executes)
    be_upscale_desc upscale{};
    be_interpolate_desc interpolate{};
#ifdef BE_RR
    be_denoise_desc denoise{};
#endif
};

struct Parts {
    std::vector<Part*> parts;
    Part* get(ID3D12Device* device);
    void drain();   // waits until the GPU has retired every part
};

struct Context {
    enum Kind { Upscale, FrameGen, SwapChain } kind;
    void* impl;
};

struct Global {
    std::mutex mutex;   // m4r, hipvk and the interop are used under it only
    const OptiPluginApi* api = nullptr;
    OptiPlugin* plugin = nullptr;
    std::string dir;    // <plugin dir>\m4r
    be_context* m4r = nullptr;
    bool failed = false;
    dxi::Interop dx;
    VkCommandBuffer cb = VK_NULL_HANDLE;   // hipvk command buffer for the work between the parts
    uint64_t splitValue = 0;
};
extern Global g;

void log(const char* fmt, ...);
bool ready(ID3D12Device* device);   // g.mutex held
Part* split(Parts& parts, ID3D12GraphicsCommandList* list, void (*work)(void*, ID3D12CommandQueue*));   // g.mutex held

// rr.cpp (backends with BE_RR): NGX Ray Reconstruction exports
void* rrGetProc(void*, const char* name);

// framegen.cpp
void* fgGetProc(void*, const char* name);
ffxReturnCode_t fgCreateContext(ffxContext* ctx, ffxCreateContextDescHeader* desc);
ffxReturnCode_t fgDestroyContext(ffxContext* ctx);
ffxReturnCode_t fgConfigure(ffxContext* ctx, const ffxConfigureDescHeader* desc);
ffxReturnCode_t fgQuery(ffxContext* ctx, ffxQueryDescHeader* desc);
ffxReturnCode_t fgDispatch(ffxContext* ctx, const ffxDispatchDescHeader* desc);
// all4r's frame generation router: the generator alone (see framegen.cpp)
void* fgGenCreate(ID3D12Device* device, uint32_t flags, uint32_t displayW, uint32_t displayH, uint32_t maxRenderW, uint32_t maxRenderH);
int fgGenPrepare(void* f, const void* prepareDesc);
int fgGenGenerate(void* f, const void* generateDesc);
void fgGenDestroy(void* f);

} // namespace m4rasi
