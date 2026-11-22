// m4r.asi: MetalFX frame interpolation as OptiScaler's FFX frame generation provider ("MetalFX FG").
//
// FFX frame generation is two things: the swapchain (frame pacing, presenting real and generated frames) and the frame generation
// context that makes the in-between frame. OptiScaler drives both through the provider's ffx* functions; it creates the swapchain
// and the context, configures them, sends a prepare dispatch (depth, motion vectors) per frame, and the swapchain asks for each
// generated frame through OptiScaler's frame generation callback, which dispatches the context (FFX_API_DISPATCH_DESC_TYPE_
// FRAMEGENERATION). Everything except the frame itself goes to AMD's own FFX frame generation (the amd_fidelityfx_*dx12.dll that
// ships with OptiScaler or the game): its swapchain, its pacing. The two dispatches are MetalFX's: prepare packs depth and motion
// into shared buffers, the generation dispatch packs the frame and runs m4r's MetalFX interpolation between it and the previous
// frame on HIP, in the list the swapchain executes (split there, like the upscaler).
// SPDX-License-Identifier: GPL-3.0-only
#include "asi.h"
#include "ffx_api/ffx_framegeneration.h"
#include "ffx_api/dx12/ffx_api_dx12.h"
#include <cstring>

namespace m4rasi {

namespace {

struct Amd {
    HMODULE dll = nullptr;
    PfnFfxCreateContext create = nullptr;
    PfnFfxDestroyContext destroy = nullptr;
    PfnFfxConfigure configure = nullptr;
    PfnFfxQuery query = nullptr;
    PfnFfxDispatch dispatch = nullptr;
    bool tried = false;
} amd;

// AMD's FFX frame generation dll, from OptiScaler's folder (the plugin's parent), the plugin's folder or the game's
bool loadAmd() {
    if (amd.tried) return amd.dll != nullptr;
    amd.tried = true;
    const std::string pluginDir = g.dir.substr(0, g.dir.find_last_of('\\'));
    const std::string optiDir = pluginDir.substr(0, pluginDir.find_last_of('\\'));
    char exe[MAX_PATH] = {};
    GetModuleFileNameA(nullptr, exe, MAX_PATH);
    std::string gameDir = exe;
    gameDir = gameDir.substr(0, gameDir.find_last_of('\\'));
    for (const std::string& dir : {optiDir, pluginDir, gameDir})
        for (const char* name : {"amd_fidelityfx_framegeneration_dx12.dll", "amd_fidelityfx_dx12.dll", "amd_fidelityfx_loader_dx12.dll"}) {
            const std::string path = dir + "\\" + name;
            if (GetFileAttributesA(path.c_str()) == INVALID_FILE_ATTRIBUTES) continue;
            HMODULE m = LoadLibraryA(path.c_str());
            if (!m) continue;
            amd.create = reinterpret_cast<PfnFfxCreateContext>(GetProcAddress(m, "ffxCreateContext"));
            amd.destroy = reinterpret_cast<PfnFfxDestroyContext>(GetProcAddress(m, "ffxDestroyContext"));
            amd.configure = reinterpret_cast<PfnFfxConfigure>(GetProcAddress(m, "ffxConfigure"));
            amd.query = reinterpret_cast<PfnFfxQuery>(GetProcAddress(m, "ffxQuery"));
            amd.dispatch = reinterpret_cast<PfnFfxDispatch>(GetProcAddress(m, "ffxDispatch"));
            if (amd.create && amd.destroy && amd.configure && amd.query && amd.dispatch) {
                amd.dll = m;
                log("frame generation: swapchain and pacing from %s", path.c_str());
                return true;
            }
            FreeLibrary(m);
        }
    log("frame generation: no AMD FFX frame generation dll (amd_fidelityfx_dx12.dll) next to OptiScaler or the game");
    return false;
}

struct FrameGen {
    ffxContext amdCtx = nullptr;   // AMD's context: the swapchain is linked to it
    ID3D12Device* device = nullptr;
    uint32_t flags = 0;
    FfxApiDimensions2D display{}, maxRender{};
    bool initialised = false, broken = false;
    // per frame (by frameID parity): the prepare dispatch's depth and motion; the frames themselves
    dxi::Shared depth[2], motion[2], color[2], output;
    uint32_t renderW = 0, renderH = 0;
    float motionScale[2] = {1, 1}, nearZ = 0.1f, farZ = 1000.0f, deltaMs = 16.6f;
    uint64_t preparedId = ~0ull, lastGenerated = ~0ull;
    bool havePrev = false;
    Parts parts;
};

struct Ctx { Context::Kind kind; ffxContext amdCtx; FrameGen* fg; };   // the plugin's handle for swapchain and frame generation contexts

ffxContext* amdOf(ffxContext* ctx) { return &static_cast<Ctx*>(*ctx)->amdCtx; }

bool initFrameGen(FrameGen& f, uint32_t renderW, uint32_t renderH) {
    if (f.initialised && f.renderW == renderW && f.renderH == renderH) return true;
    for (auto* s : {&f.depth[0], &f.depth[1], &f.motion[0], &f.motion[1], &f.color[0], &f.color[1], &f.output}) g.dx.release(*s);
    const uint32_t ow = f.display.width, oh = f.display.height;
    if (be_interpolate_init(g.m4r, ow, oh, renderW, renderH) != 0) {
        log("be_interpolate_init %ux%u -> %ux%u: %s", renderW, renderH, ow, oh, be_last_error(g.m4r));
        return false;
    }
    for (int i = 0; i < 2; ++i)
        if (!g.dx.create(f.depth[i], renderW, renderH, dxi::Layout::R32F) || !g.dx.create(f.motion[i], renderW, renderH, dxi::Layout::RG16F) ||
            !g.dx.create(f.color[i], ow, oh, dxi::Layout::RGBA16F)) {
            log("interop buffers: %s", g.dx.error.c_str());
            return false;
        }
    if (!g.dx.create(f.output, ow, oh, dxi::Layout::RGBA16F)) { log("interop buffers: %s", g.dx.error.c_str()); return false; }
    f.renderW = renderW; f.renderH = renderH; f.initialised = true; f.havePrev = false;
    return true;
}

void runInterpolate(void* owner, ID3D12CommandQueue* queue) {
    Part* p = static_cast<Part*>(owner);
    std::lock_guard<std::mutex> lock(g.mutex);
    vkBeginCommandBuffer(g.cb, nullptr);
    if (be_interpolate(g.m4r, g.cb, &p->interpolate) != 0) {
        log("be_interpolate: %s", be_last_error(g.m4r));
        vkBeginCommandBuffer(g.cb, nullptr);
    }
    vkEndCommandBuffer(g.cb);
    if (!g.dx.handoff(queue, g.cb)) log("%s", g.dx.error.c_str());
}

// FFX_API_DISPATCH_DESC_TYPE_FRAMEGENERATION_PREPARE: this frame's depth and motion vectors
ffxReturnCode_t prepare(FrameGen& f, const ffxDispatchDescFrameGenerationPrepare& d) {
    auto* list = static_cast<ID3D12GraphicsCommandList*>(d.commandList);
    auto* depth = static_cast<ID3D12Resource*>(d.depth.resource);
    auto* motion = static_cast<ID3D12Resource*>(d.motionVectors.resource);
    if (!list || !depth || !motion) return FFX_API_RETURN_ERROR_PARAMETER;
    std::lock_guard<std::mutex> lock(g.mutex);
    if (f.broken) return FFX_API_RETURN_OK;
    const uint32_t rw = d.renderSize.width ? d.renderSize.width : f.maxRender.width;
    const uint32_t rh = d.renderSize.height ? d.renderSize.height : f.maxRender.height;
    if (!initFrameGen(f, f.maxRender.width ? f.maxRender.width : rw, f.maxRender.height ? f.maxRender.height : rh)) { f.broken = true; return FFX_API_RETURN_OK; }
    const int slot = int(d.frameID & 1);
    const bool hires = (f.flags & FFX_FRAMEGENERATION_ENABLE_DISPLAY_RESOLUTION_MOTION_VECTORS) != 0;
    const D3D12_RESOURCE_STATES ds = dxi::d3dState(d.depth.state), ms = dxi::d3dState(d.motionVectors.state);
    dxi::transition(list, depth, ds, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    dxi::transition(list, motion, ms, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    g.dx.pack(list, depth, f.depth[slot], rw, rh);
    if (hires) g.dx.packResampled(list, motion, f.motion[slot], rw, rh, f.display.width, f.display.height);
    else g.dx.pack(list, motion, f.motion[slot], rw, rh);
    dxi::transition(list, depth, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, ds);
    dxi::transition(list, motion, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, ms);
    f.motionScale[0] = d.motionVectorScale.x * (hires ? float(rw) / float(f.display.width) : 1.0f);
    f.motionScale[1] = d.motionVectorScale.y * (hires ? float(rh) / float(f.display.height) : 1.0f);
    f.nearZ = d.cameraNear; f.farZ = d.cameraFar; f.deltaMs = d.frameTimeDelta;
    f.preparedId = d.frameID;
    return FFX_API_RETURN_OK;
}

// FFX_API_DISPATCH_DESC_TYPE_FRAMEGENERATION: the frame between the previous one and presentColor, into outputs[0]
ffxReturnCode_t generate(FrameGen& f, const ffxDispatchDescFrameGeneration& d) {
    auto* list = static_cast<ID3D12GraphicsCommandList*>(d.commandList);
    auto* present = static_cast<ID3D12Resource*>(d.presentColor.resource);
    auto* out = static_cast<ID3D12Resource*>(d.outputs[0].resource);
    if (!list || !present || !out) return FFX_API_RETURN_ERROR_PARAMETER;
    std::lock_guard<std::mutex> lock(g.mutex);
    if (f.broken || !f.initialised) return FFX_API_RETURN_ERROR_RUNTIME_ERROR;
    const uint32_t ow = f.display.width, oh = f.display.height;
    const int slot = int(d.frameID & 1);
    const D3D12_RESOURCE_STATES ps = dxi::d3dState(d.presentColor.state), os = dxi::d3dState(d.outputs[0].state);
    dxi::transition(list, present, ps, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    g.dx.pack(list, present, f.color[slot], ow, oh);
    dxi::transition(list, present, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, ps);
    const bool reset = d.reset || !f.havePrev || f.lastGenerated + 1 != d.frameID || f.preparedId != d.frameID;
    f.havePrev = true;
    f.lastGenerated = d.frameID;
    if (reset) {   // nothing to interpolate from: the generated frame repeats the current one
        dxi::transition(list, out, os, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        g.dx.unpack(list, f.color[slot], out, ow, oh);
        dxi::transition(list, out, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, os);
        return FFX_API_RETURN_OK;
    }
    Part* p = split(f.parts, list, &runInterpolate);
    if (!p) { log("frame generation: the command list could not be split"); return FFX_API_RETURN_ERROR_RUNTIME_ERROR; }
    be_interpolate_desc& m = p->interpolate;
    m = {};
    m.color = {f.color[slot].image, VK_FORMAT_R16G16B16A16_SFLOAT, ow, oh, VK_IMAGE_LAYOUT_GENERAL};
    m.prev_color = {f.color[slot ^ 1].image, VK_FORMAT_R16G16B16A16_SFLOAT, ow, oh, VK_IMAGE_LAYOUT_GENERAL};
    m.depth = {f.depth[slot].image, VK_FORMAT_R32_SFLOAT, f.renderW, f.renderH, VK_IMAGE_LAYOUT_GENERAL};
    m.motion = {f.motion[slot].image, VK_FORMAT_R16G16_SFLOAT, f.renderW, f.renderH, VK_IMAGE_LAYOUT_GENERAL};
    m.output = {f.output.image, VK_FORMAT_R16G16B16A16_SFLOAT, ow, oh, VK_IMAGE_LAYOUT_GENERAL};
    m.motion_scale_x = f.motionScale[0]; m.motion_scale_y = f.motionScale[1];
    m.delta_time = f.deltaMs * 1e-3f;
    m.near_plane = f.nearZ; m.far_plane = f.farZ;
    m.depth_reversed = (f.flags & FFX_FRAMEGENERATION_ENABLE_DEPTH_INVERTED) ? 1 : 0;
    m.reset = 0;
    dxi::transition(list, out, os, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    g.dx.unpack(list, f.output, out, ow, oh);
    dxi::transition(list, out, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, os);
    // several generated frames per real one: MetalFX makes the middle one; the others repeat it
    for (uint32_t i = 1; i < d.numGeneratedFrames && i < 4; ++i)
        if (auto* o = static_cast<ID3D12Resource*>(d.outputs[i].resource)) {
            const D3D12_RESOURCE_STATES s = dxi::d3dState(d.outputs[i].state);
            dxi::transition(list, o, s, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            g.dx.unpack(list, f.output, o, ow, oh);
            dxi::transition(list, o, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, s);
        }
    return FFX_API_RETURN_OK;
}

ffxReturnCode_t createImpl(ffxContext* ctx, ffxCreateContextDescHeader* desc, const ffxAllocationCallbacks* cb) {
    if (!ctx || !desc) return FFX_API_RETURN_ERROR_PARAMETER;
    if (!loadAmd()) return FFX_API_RETURN_NO_PROVIDER;
    const ffxCreateContextDescFrameGeneration* fgd = nullptr;
    ID3D12Device* device = nullptr;
    for (auto* h = desc; h; h = h->pNext) {
        if (h->type == FFX_API_CREATE_CONTEXT_DESC_TYPE_FRAMEGENERATION) fgd = reinterpret_cast<const ffxCreateContextDescFrameGeneration*>(h);
        else if (h->type == FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_DX12) device = reinterpret_cast<const ffxCreateBackendDX12Desc*>(h)->device;
    }
    ffxContext inner = nullptr;
    const ffxReturnCode_t r = amd.create(&inner, desc, cb);
    if (r != FFX_API_RETURN_OK) return r;
    auto* c = new Ctx{fgd ? Context::FrameGen : Context::SwapChain, inner, nullptr};
    if (fgd) {
        auto* f = new FrameGen;
        f->amdCtx = inner; f->device = device; f->flags = fgd->flags; f->display = fgd->displaySize; f->maxRender = fgd->maxRenderSize;
        std::lock_guard<std::mutex> lock(g.mutex);
        if (!device || !ready(device)) f->broken = true;
        log("frame generation context: render %ux%u -> %ux%u, flags 0x%x%s", f->maxRender.width, f->maxRender.height, f->display.width,
            f->display.height, f->flags, f->broken ? " (m4r unavailable: AMD's frame generation runs instead)" : "");
        c->fg = f;
    }
    *ctx = c;
    return FFX_API_RETURN_OK;
}

ffxReturnCode_t createEntry(ffxContext* ctx, ffxCreateContextDescHeader* desc, const ffxAllocationCallbacks* cb) { return createImpl(ctx, desc, cb); }

ffxReturnCode_t destroyEntry(ffxContext* ctx, const ffxAllocationCallbacks* cb) {
    if (!ctx || !*ctx) return FFX_API_RETURN_ERROR_PARAMETER;
    auto* c = static_cast<Ctx*>(*ctx);
    const ffxReturnCode_t r = amd.destroy(&c->amdCtx, cb);
    if (FrameGen* f = c->fg) {
        std::lock_guard<std::mutex> lock(g.mutex);
        f->parts.drain();
        hipvk::finish();
        for (auto* s : {&f->depth[0], &f->depth[1], &f->motion[0], &f->motion[1], &f->color[0], &f->color[1], &f->output}) g.dx.release(*s);
        delete f;
    }
    delete c;
    *ctx = nullptr;
    return r;
}

ffxReturnCode_t configureEntry(ffxContext* ctx, const ffxConfigureDescHeader* desc) {
    if (!ctx || !*ctx) return amd.dll ? amd.configure(ctx, desc) : FFX_API_RETURN_ERROR_PARAMETER;
    return amd.configure(amdOf(ctx), desc);
}

ffxReturnCode_t queryEntry(ffxContext* ctx, ffxQueryDescHeader* desc) {
    if (!loadAmd()) return FFX_API_RETURN_NO_PROVIDER;
    if (!ctx || !*ctx) return amd.query(ctx, desc);
    return amd.query(amdOf(ctx), desc);
}

ffxReturnCode_t dispatchEntry(ffxContext* ctx, const ffxDispatchDescHeader* desc) {
    if (!ctx || !*ctx || !desc) return FFX_API_RETURN_ERROR_PARAMETER;
    auto* c = static_cast<Ctx*>(*ctx);
    if (FrameGen* f = c->fg; f && !f->broken) {
        if (desc->type == FFX_API_DISPATCH_DESC_TYPE_FRAMEGENERATION_PREPARE)
            return prepare(*f, *reinterpret_cast<const ffxDispatchDescFrameGenerationPrepare*>(desc));
        if (desc->type == FFX_API_DISPATCH_DESC_TYPE_FRAMEGENERATION) {
            const ffxReturnCode_t r = generate(*f, *reinterpret_cast<const ffxDispatchDescFrameGeneration*>(desc));
            if (r == FFX_API_RETURN_OK) return r;
            log("frame generation: MetalFX failed, AMD's frame generation takes over");
            f->broken = true;
        }
    }
    return amd.dispatch(&c->amdCtx, desc);
}

} // namespace

void* fgGetProc(void*, const char* name) {
    if (!std::strcmp(name, "ffxCreateContext")) return reinterpret_cast<void*>(&createEntry);
    if (!std::strcmp(name, "ffxDestroyContext")) return reinterpret_cast<void*>(&destroyEntry);
    if (!std::strcmp(name, "ffxConfigure")) return reinterpret_cast<void*>(&configureEntry);
    if (!std::strcmp(name, "ffxQuery")) return reinterpret_cast<void*>(&queryEntry);
    if (!std::strcmp(name, "ffxDispatch")) return reinterpret_cast<void*>(&dispatchEntry);
    return nullptr;
}

// The upscaler provider's entry points route frame generation descriptors here (a dll that serves both kinds)
ffxReturnCode_t fgCreateContext(ffxContext* ctx, ffxCreateContextDescHeader* desc) { return createImpl(ctx, desc, nullptr); }
ffxReturnCode_t fgDestroyContext(ffxContext* ctx) { return destroyEntry(ctx, nullptr); }
ffxReturnCode_t fgConfigure(ffxContext* ctx, const ffxConfigureDescHeader* desc) { return configureEntry(ctx, desc); }
ffxReturnCode_t fgQuery(ffxContext* ctx, ffxQueryDescHeader* desc) { return queryEntry(ctx, desc); }
ffxReturnCode_t fgDispatch(ffxContext* ctx, const ffxDispatchDescHeader* desc) { return dispatchEntry(ctx, desc); }

} // namespace m4rasi
