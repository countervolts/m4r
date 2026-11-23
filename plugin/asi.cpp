// m4r.asi: MetalFX in OptiScaler on native Windows (AMD RDNA2/3/4), as an OptiScaler plugin (plugins folder, LoadAsiPlugins=true).
//
// It registers MetalFX with OptiScaler's plugin API (optiscaler_plugin.h, version 3) as providers OptiScaler already knows how to
// drive: an FFX upscaler ("MetalFX" in the upscaler list) and FFX frame generation ("MetalFX FG"), see framegen.cpp. No OptiScaler
// patch is needed. m4r's kernels run on AMD's HIP runtime (runtime/hipvk) inside the game's frame: dx_interop.h moves the inputs
// into shared buffers, d4r's command-list split puts HIP's work between the part of the game's list before the upscaler and the
// part after it (GPU fences, no CPU wait).
//
// Files: <plugin dir>\m4r\plans (install.py), <plugin dir>\m4r\m4r.log.
// SPDX-License-Identifier: GPL-3.0-only
#include "asi.h"
#include "ffx_api/ffx_upscale.h"
#include "ffx_api/ffx_framegeneration.h"
#include "ffx_api/dx12/ffx_api_dx12.h"
#include <cmath>
#include <cstdarg>
#include <cstring>

#include "d4r_d3d12_split.h"   // defines the split's stubs: included in this one translation unit

namespace m4rasi {

Global g;

void log(const char* fmt, ...) {
    char buf[1024];
    va_list ap;
    va_start(ap, fmt);
    std::vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    static FILE* f = nullptr;
    static std::mutex m;
    std::lock_guard<std::mutex> lock(m);
    if (!f && !g.dir.empty()) f = _fsopen((g.dir + "\\" BE_NAME ".log").c_str(), "w", _SH_DENYNO);
    if (f) { std::fprintf(f, "%s\n", buf); std::fflush(f); }
    if (g.api && OPTI_API_HAS(g.api, Log)) g.api->Log(g.plugin, OptiLog_Info, buf);
}

// m4r, HIP and the interop, once per process, on the first context (not at load: HIP must not load before the game's first frame)
bool ready(ID3D12Device* device) {
    if (g.m4r) return true;
    if (g.failed) return false;
    const char* why = nullptr;
    if (!d4r_split::install(device, &why)) { log("command-list split unavailable: %s", why ? why : "?"); g.failed = true; return false; }
    if (be_create(&g.m4r, nullptr, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, 0, (g.dir + "\\" BE_DATA).c_str()) != 0) {
        log("%s: %s", BE_NAME, g.m4r ? be_last_error(g.m4r) : "no context");
        if (g.m4r) be_destroy(g.m4r);
        g.m4r = nullptr; g.failed = true;
        return false;
    }
    if (!g.dx.init(device)) { log("interop: %s", g.dx.error.c_str()); be_destroy(g.m4r); g.m4r = nullptr; g.failed = true; return false; }
    VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    ai.commandBufferCount = 1;
    vkAllocateCommandBuffers(VK_NULL_HANDLE, &ai, &g.cb);
    log("%s ready on %s (HIP)", BE_NAME, hipvk::gpuTarget().c_str());
    return true;
}

// ---- second parts of split lists ---------------------------------------------------------------------------------------------

Part* Parts::get(ID3D12Device* device) {
    for (Part* p : parts)
        if (!p->busy && p->retire->GetCompletedValue() >= p->retireValue) {
            if (FAILED(p->allocator->Reset()) || FAILED(p->list->Reset(p->allocator, nullptr))) continue;
            p->busy = true;
            return p;
        }
    auto* p = new Part;
    if (FAILED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&p->allocator))) ||
        FAILED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, p->allocator, nullptr, IID_PPV_ARGS(&p->list))) ||
        FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&p->retire)))) {
        delete p;
        return nullptr;
    }
    p->busy = true;
    parts.push_back(p);
    return p;
}

void Parts::drain() {
    for (Part* p : parts) {
        if (p->retire && p->retire->GetCompletedValue() < p->retireValue) {
            HANDLE e = CreateEventW(nullptr, FALSE, FALSE, nullptr);
            if (e && SUCCEEDED(p->retire->SetEventOnCompletion(p->retireValue, e))) WaitForSingleObject(e, 2000);
            if (e) CloseHandle(e);
        }
    }
}

void partDone(void* owner, bool) { static_cast<Part*>(owner)->busy = false; }

// Splits `list` here: what is recorded into it from now on runs after `work(part, queue)` (called between the parts on the
// executing thread, under g.mutex). Returns the part (its `job` holds the caller's state) or null.
Part* split(Parts& parts, ID3D12GraphicsCommandList* list, void (*work)(void*, ID3D12CommandQueue*)) {
    Part* p = parts.get(g.dx.device());
    if (!p) return nullptr;
    const uint64_t v = ++g.splitValue;
    if (!d4r_split::begin(list, p->list, g.dx.splitFence(), v, p->retire, ++p->retireValue, &partDone, p, work, p)) {
        p->list->Close();
        p->busy = false;
        --p->retireValue;
        return nullptr;
    }
    return p;
}

// ---- the FFX upscaler ---------------------------------------------------------------------------------------------------------

struct Upscaler {
    uint32_t flags = 0;
    FfxApiDimensions2D maxRender{}, maxOutput{};
    bool initialised = false, broken = false;
    dxi::Shared color, depth, motion, exposure, output;
    Parts parts;
    uint32_t outW = 0, outH = 0;
};

static void runUpscale(void* owner, ID3D12CommandQueue* queue) {
    Part* p = static_cast<Part*>(owner);
    std::lock_guard<std::mutex> lock(g.mutex);
    vkBeginCommandBuffer(g.cb, nullptr);
    if (be_upscale(g.m4r, g.cb, &p->upscale) != 0) {
        log("be_upscale: %s", be_last_error(g.m4r));
        vkEndCommandBuffer(g.cb);
        vkBeginCommandBuffer(g.cb, nullptr);   // an empty submission keeps the fence handoff intact
    }
    vkEndCommandBuffer(g.cb);
    if (!g.dx.handoff(queue, g.cb)) log("%s", g.dx.error.c_str());
}

static float ratioOf(uint32_t mode) {
    switch (mode) {
    case FFX_UPSCALE_QUALITY_MODE_NATIVEAA: return 1.0f;
    case FFX_UPSCALE_QUALITY_MODE_QUALITY: return 1.5f;
    case FFX_UPSCALE_QUALITY_MODE_BALANCED: return 1.7f;
    case FFX_UPSCALE_QUALITY_MODE_PERFORMANCE: return 2.0f;
    default: return 3.0f;
    }
}

static float halton(int i, int b) {
    float f = 1.0f, r = 0.0f;
    while (i > 0) { f /= float(b); r += f * float(i % b); i /= b; }
    return r;
}

constexpr uint64_t kVersionId = BE_VERSION_ID;
static const char* kVersionName = BE_VERSION_NAME;

static ffxReturnCode_t query(ffxContext* ctx, ffxQueryDescHeader* desc) {
    for (auto* h = desc; h; h = h->pNext) {
        switch (h->type) {
        case FFX_API_QUERY_DESC_TYPE_GET_VERSIONS: {
            auto* q = reinterpret_cast<ffxQueryDescGetVersions*>(h);
            if (q->createDescType != FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE) { if (q->outputCount) *q->outputCount = 0; break; }
            if (q->outputCount && *q->outputCount >= 1) {
                if (q->versionIds) q->versionIds[0] = kVersionId;
                if (q->versionNames) q->versionNames[0] = kVersionName;
            }
            if (q->outputCount) *q->outputCount = 1;
            break;
        }
        case FFX_API_QUERY_DESC_TYPE_GET_PROVIDER_VERSION: {
            auto* q = reinterpret_cast<ffxQueryGetProviderVersion*>(h);
            q->versionId = kVersionId; q->versionName = kVersionName;
            break;
        }
        case FFX_API_QUERY_DESC_TYPE_UPSCALE_GETUPSCALERATIOFROMQUALITYMODE: {
            auto* q = reinterpret_cast<ffxQueryDescUpscaleGetUpscaleRatioFromQualityMode*>(h);
            if (q->pOutUpscaleRatio) *q->pOutUpscaleRatio = ratioOf(q->qualityMode);
            break;
        }
        case FFX_API_QUERY_DESC_TYPE_UPSCALE_GETRENDERRESOLUTIONFROMQUALITYMODE: {
            auto* q = reinterpret_cast<ffxQueryDescUpscaleGetRenderResolutionFromQualityMode*>(h);
            const float r = ratioOf(q->qualityMode);
            if (q->pOutRenderWidth) *q->pOutRenderWidth = uint32_t(float(q->displayWidth) / r);
            if (q->pOutRenderHeight) *q->pOutRenderHeight = uint32_t(float(q->displayHeight) / r);
            break;
        }
        case FFX_API_QUERY_DESC_TYPE_UPSCALE_GETJITTERPHASECOUNT: {
            auto* q = reinterpret_cast<ffxQueryDescUpscaleGetJitterPhaseCount*>(h);
            const float r = q->renderWidth ? float(q->displayWidth) / float(q->renderWidth) : 1.0f;
            if (q->pOutPhaseCount) *q->pOutPhaseCount = int32_t(8.0f * r * r);
            break;
        }
        case FFX_API_QUERY_DESC_TYPE_UPSCALE_GETJITTEROFFSET: {
            auto* q = reinterpret_cast<ffxQueryDescUpscaleGetJitterOffset*>(h);
            const int n = q->phaseCount > 0 ? q->phaseCount : 1;
            const int i = ((q->index % n) + n) % n + 1;
            if (q->pOutX) *q->pOutX = halton(i, 2) - 0.5f;
            if (q->pOutY) *q->pOutY = halton(i, 3) - 0.5f;
            break;
        }
        case FFX_API_QUERY_DESC_TYPE_UPSCALE_GPU_MEMORY_USAGE: {
            auto* q = reinterpret_cast<ffxQueryDescUpscaleGetGPUMemoryUsage*>(h);
            if (q->gpuMemoryUsageUpscaler) *q->gpuMemoryUsageUpscaler = {};
            break;
        }
        default:
            if (ctx && *ctx && (h->type & FFX_API_EFFECT_MASK) == FFX_API_EFFECT_ID_FRAMEGENERATION) return fgQuery(ctx, h);
            return FFX_API_RETURN_ERROR_UNKNOWN_DESCTYPE;
        }
    }
    return FFX_API_RETURN_OK;
}

static ffxReturnCode_t createContext(ffxContext* ctx, ffxCreateContextDescHeader* desc, const ffxAllocationCallbacks*) {
    if (!ctx || !desc) return FFX_API_RETURN_ERROR_PARAMETER;
    const ffxCreateContextDescUpscale* up = nullptr;
    ID3D12Device* device = nullptr;
    for (auto* h = desc; h; h = h->pNext) {
        if (h->type == FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE) up = reinterpret_cast<const ffxCreateContextDescUpscale*>(h);
        else if (h->type == FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_DX12) device = reinterpret_cast<const ffxCreateBackendDX12Desc*>(h)->device;
    }
    if (!up) return fgCreateContext(ctx, desc);
    if (!device) return FFX_API_RETURN_ERROR_PARAMETER;
    std::lock_guard<std::mutex> lock(g.mutex);
    if (!ready(device)) return FFX_API_RETURN_ERROR_RUNTIME_ERROR;
    auto* u = new Upscaler;
    u->flags = up->flags; u->maxRender = up->maxRenderSize; u->maxOutput = up->maxUpscaleSize;
    log("upscaler context: render %ux%u -> %ux%u, flags 0x%x", u->maxRender.width, u->maxRender.height, u->maxOutput.width, u->maxOutput.height, u->flags);
    *ctx = new Context{Context::Upscale, u};
    return FFX_API_RETURN_OK;
}

static bool initUpscaler(Upscaler& u, uint32_t outW, uint32_t outH) {
    if (u.initialised && u.outW == outW && u.outH == outH) return true;
    g.dx.release(u.color); g.dx.release(u.depth); g.dx.release(u.motion); g.dx.release(u.exposure); g.dx.release(u.output);
    const uint32_t iw = u.maxRender.width, ih = u.maxRender.height;
    if (be_upscale_init(g.m4r, outW, outH, iw, ih) != 0) { log("be_upscale_init %ux%u -> %ux%u: %s", iw, ih, outW, outH, be_last_error(g.m4r)); return false; }
    if (!g.dx.create(u.color, iw, ih, dxi::Layout::RGBA16F) || !g.dx.create(u.depth, iw, ih, dxi::Layout::R32F) ||
        !g.dx.create(u.motion, iw, ih, dxi::Layout::RG16F) || !g.dx.create(u.exposure, 1, 1, dxi::Layout::R32F) ||
        !g.dx.create(u.output, outW, outH, dxi::Layout::RGBA16F)) {
        log("interop buffers: %s", g.dx.error.c_str());
        return false;
    }
    u.outW = outW; u.outH = outH; u.initialised = true;
    return true;
}

static ffxReturnCode_t dispatch(ffxContext* ctx, const ffxDispatchDescHeader* desc) {
    if (!ctx || !*ctx || !desc) return FFX_API_RETURN_ERROR_PARAMETER;
    auto* c = static_cast<Context*>(*ctx);
    if (c->kind != Context::Upscale) return fgDispatch(ctx, desc);
    if (desc->type == FFX_API_DISPATCH_DESC_TYPE_UPSCALE_GENERATEREACTIVEMASK) return FFX_API_RETURN_OK;   // MetalFX has no reactive input
    if (desc->type != FFX_API_DISPATCH_DESC_TYPE_UPSCALE) return FFX_API_RETURN_ERROR_UNKNOWN_DESCTYPE;
    auto& d = *reinterpret_cast<const ffxDispatchDescUpscale*>(desc);
    auto& u = *static_cast<Upscaler*>(c->impl);
    auto* list = static_cast<ID3D12GraphicsCommandList*>(d.commandList);
    auto* color = static_cast<ID3D12Resource*>(d.color.resource);
    auto* depth = static_cast<ID3D12Resource*>(d.depth.resource);
    auto* motion = static_cast<ID3D12Resource*>(d.motionVectors.resource);
    auto* out = static_cast<ID3D12Resource*>(d.output.resource);
    auto* exposure = static_cast<ID3D12Resource*>(d.exposure.resource);
    if (!list || !color || !depth || !motion || !out) return FFX_API_RETURN_ERROR_PARAMETER;
    std::lock_guard<std::mutex> lock(g.mutex);
    if (u.broken) return FFX_API_RETURN_ERROR_RUNTIME_ERROR;
    const D3D12_RESOURCE_DESC od = out->GetDesc();
    const uint32_t ow = d.upscaleSize.width ? d.upscaleSize.width : uint32_t(od.Width);
    const uint32_t oh = d.upscaleSize.height ? d.upscaleSize.height : od.Height;
    if (!initUpscaler(u, ow, oh)) { u.broken = true; return FFX_API_RETURN_ERROR_RUNTIME_ERROR; }
    const uint32_t rw = std::min(d.renderSize.width ? d.renderSize.width : u.maxRender.width, u.maxRender.width);
    const uint32_t rh = std::min(d.renderSize.height ? d.renderSize.height : u.maxRender.height, u.maxRender.height);
    const bool hiresMv = (u.flags & FFX_UPSCALE_ENABLE_DISPLAY_RESOLUTION_MOTION_VECTORS) != 0;
    // inputs -> shared buffers (game's list, before the split)
    const D3D12_RESOURCE_STATES read = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    const D3D12_RESOURCE_STATES cs = dxi::d3dState(d.color.state), ds = dxi::d3dState(d.depth.state), ms = dxi::d3dState(d.motionVectors.state),
                                es = dxi::d3dState(d.exposure.state), os = dxi::d3dState(d.output.state);
    dxi::transition(list, color, cs, read); dxi::transition(list, depth, ds, read); dxi::transition(list, motion, ms, read);
    if (exposure) dxi::transition(list, exposure, es, read);
    g.dx.pack(list, color, u.color, rw, rh);
    g.dx.pack(list, depth, u.depth, rw, rh);
    if (hiresMv) g.dx.packResampled(list, motion, u.motion, rw, rh, ow, oh);
    else g.dx.pack(list, motion, u.motion, rw, rh);
    if (exposure) g.dx.pack(list, exposure, u.exposure, 1, 1);
    dxi::transition(list, color, read, cs); dxi::transition(list, depth, read, ds); dxi::transition(list, motion, read, ms);
    if (exposure) dxi::transition(list, exposure, read, es);
    Part* p = split(u.parts, list, &runUpscale);
    if (!p) { log("the command list could not be split"); return FFX_API_RETURN_ERROR_RUNTIME_ERROR; }
    be_upscale_desc& m = p->upscale;
    m = {};
    m.color = {u.color.image, VK_FORMAT_R16G16B16A16_SFLOAT, u.color.w, u.color.h, VK_IMAGE_LAYOUT_GENERAL};
    m.depth = {u.depth.image, VK_FORMAT_R32_SFLOAT, u.depth.w, u.depth.h, VK_IMAGE_LAYOUT_GENERAL};
    m.motion = {u.motion.image, VK_FORMAT_R16G16_SFLOAT, u.motion.w, u.motion.h, VK_IMAGE_LAYOUT_GENERAL};
    if (exposure) m.exposure = {u.exposure.image, VK_FORMAT_R32_SFLOAT, 1, 1, VK_IMAGE_LAYOUT_GENERAL};
    m.output = {u.output.image, VK_FORMAT_R16G16B16A16_SFLOAT, ow, oh, VK_IMAGE_LAYOUT_GENERAL};
    m.content_width = rw; m.content_height = rh;
    m.jitter_x = d.jitterOffset.x; m.jitter_y = d.jitterOffset.y;
    m.motion_scale_x = d.motionVectorScale.x; m.motion_scale_y = d.motionVectorScale.y;
    if (hiresMv) { m.motion_scale_x *= float(rw) / float(ow); m.motion_scale_y *= float(rh) / float(oh); }
    m.pre_exposure = d.preExposure > 0.0f ? d.preExposure : 1.0f;
    m.reset = d.reset ? 1 : 0;
    m.depth_reversed = (u.flags & FFX_UPSCALE_ENABLE_DEPTH_INVERTED) ? 1 : 0;
    // result -> game's output (recorded into the second part)
    dxi::transition(list, out, os, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    g.dx.unpack(list, u.output, out, ow, oh);
    dxi::transition(list, out, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, os);
    return FFX_API_RETURN_OK;
}

static ffxReturnCode_t configure(ffxContext* ctx, const ffxConfigureDescHeader* desc) {
    if (ctx && *ctx && static_cast<Context*>(*ctx)->kind != Context::Upscale) return fgConfigure(ctx, desc);
    return FFX_API_RETURN_OK;   // global debug and FSR's tuning keys: nothing to set for MetalFX
}

static ffxReturnCode_t destroyContext(ffxContext* ctx, const ffxAllocationCallbacks*) {
    if (!ctx || !*ctx) return FFX_API_RETURN_ERROR_PARAMETER;
    auto* c = static_cast<Context*>(*ctx);
    if (c->kind != Context::Upscale) return fgDestroyContext(ctx);
    std::lock_guard<std::mutex> lock(g.mutex);
    auto* u = static_cast<Upscaler*>(c->impl);
    u->parts.drain();
    hipvk::finish();
    g.dx.release(u->color); g.dx.release(u->depth); g.dx.release(u->motion); g.dx.release(u->exposure); g.dx.release(u->output);
    delete u;
    delete c;
    *ctx = nullptr;
    return FFX_API_RETURN_OK;
}

void* getProc(void*, const char* name) {
    if (!std::strcmp(name, "ffxCreateContext")) return reinterpret_cast<void*>(&createContext);
    if (!std::strcmp(name, "ffxDestroyContext")) return reinterpret_cast<void*>(&destroyContext);
    if (!std::strcmp(name, "ffxConfigure")) return reinterpret_cast<void*>(&configure);
    if (!std::strcmp(name, "ffxQuery")) return reinterpret_cast<void*>(&query);
    if (!std::strcmp(name, "ffxDispatch")) return reinterpret_cast<void*>(&dispatch);
    return nullptr;
}

} // namespace m4rasi

using namespace m4rasi;

extern "C" __declspec(dllexport) bool OptiPluginInit(const OptiPluginApi* api, OptiPlugin* plugin) {
    g.api = api; g.plugin = plugin;
    if (!OPTI_API_HAS(api, RegisterProvider)) { log("OptiScaler is too old for provider plugins (plugin API version %u)", api ? api->version : 0); return false; }
    const bool up = api->RegisterProvider(plugin, BE_UPSCALER, OptiProvider_Ffx, &getProc, nullptr);
    const bool fg = api->RegisterProvider(plugin, BE_FG, OptiProvider_FfxFg, &fgGetProc, nullptr);
    bool rr = false;
#ifdef BE_RR
    rr = api->RegisterProvider(plugin, BE_RR, OptiProvider_Dlssd, &rrGetProc, nullptr);
    log("registered: %s ray reconstruction %s", BE_RR, rr ? "yes" : "no");
#endif
    log("registered: %s %s, %s %s", BE_UPSCALER, up ? "yes" : "no", BE_FG, fg ? "yes" : "no");
    return up || fg || rr;
}

BOOL WINAPI DllMain(HINSTANCE module, DWORD reason, LPVOID) {
    if (reason == DLL_PROCESS_ATTACH) {
        DisableThreadLibraryCalls(module);
        char path[MAX_PATH] = {};
        GetModuleFileNameA(module, path, MAX_PATH);
        std::string dir = path;
        dir = dir.substr(0, dir.find_last_of("\\/"));
        g.dir = dir + "\\" BE_NAME;
    }
    return TRUE;
}
