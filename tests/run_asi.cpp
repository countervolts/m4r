// Drives m4r.asi the way OptiScaler does, on native D3D12: loads the plugin, calls OptiPluginInit with a minimal plugin API and
// runs a fixture sequence through one of its providers, recorded into an ordinary D3D12 command list (the plugin splits it and
// runs HIP between the parts). Same fixtures and outputs as run_bbr / run_fg, so tests/compare.py / compare_fg.py check them.
//   run_asi up <m4r.asi> <fixture dir> <frames> <out dir>   "MetalFX": the FFX upscaler (tests/scene_motion.py fixtures)
//   run_asi fg <m4r.asi> <fixture dir> <frames> <out dir>   "MetalFX FG": the prepare and frame generation dispatches, called
//       directly as the swapchain's frame generation callback would (tests/scene_fg.py fixtures; needs AMD's FFX frame generation
//       dlls next to the plugin, as in a game)
// Sizes: 640x360 -> 1280x720. Plans: <asi dir>\m4r\plans.
// SPDX-License-Identifier: GPL-3.0-only
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <d3d12.h>
#include "optiscaler_plugin.h"
#include "ffx_api/ffx_upscale.h"
#include "ffx_api/ffx_framegeneration.h"
#include "ffx_api/dx12/ffx_api_dx12.h"
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

static OptiGetProcFn g_getProc[8] = {};

static bool registerProvider(OptiPlugin*, const char* name, OptiProviderKind kind, OptiGetProcFn getProc, void*) {
    std::printf("provider: %s (kind %d)\n", name, int(kind));
    const char* want = std::getenv("RUN_ASI_PROVIDER");   // a plugin with several backends (all4r.asi): the provider to test
    if (unsigned(kind) < 8 && (!want || !std::strcmp(want, name) || (std::string(want) + " FG") == name)) g_getProc[kind] = getProc;
    return true;
}
static void logFn(OptiPlugin*, OptiLogLevel, const char* m) { std::printf("[plugin] %s\n", m); }
static bool registerTab(OptiPlugin*, const char* name, const char*, OptiDrawTabFn, void*) { std::printf("tab: %s\n", name); return true; }
static bool getString(OptiPlugin*, const char*, char* buf, size_t n, const char* def) { std::snprintf(buf, n, "%s", def); return false; }
static bool getBool(OptiPlugin*, const char*, bool def) { return def; }

static std::vector<uint8_t> readAll(const std::string& p) {
    std::ifstream f(p, std::ios::binary);
    return std::vector<uint8_t>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

static double halton(int i, int b) {
    double v = 0, s = 1;
    while (i) { s /= b; v += (i % b) * s; i /= b; }
    return v - 0.5;
}

#define CHECK(x) do { HRESULT hr_ = (x); if (FAILED(hr_)) { std::fprintf(stderr, "%s failed: 0x%08lx (line %d)\n", #x, hr_, __LINE__); return 1; } } while (0)

struct Tex { ID3D12Resource* res = nullptr; DXGI_FORMAT fmt; uint32_t w, h, bpp; };
const auto kRead = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;

static uint32_t pitchOf(const Tex& t) { return (t.w * t.bpp + 255) & ~255u; }

static void barrier(ID3D12GraphicsCommandList* l, ID3D12Resource* r, D3D12_RESOURCE_STATES a, D3D12_RESOURCE_STATES b) {
    D3D12_RESOURCE_BARRIER x{};
    x.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    x.Transition = {r, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, a, b};
    l->ResourceBarrier(1, &x);
}

static FfxApiResource ffxRes(const Tex& t, uint32_t state) {
    FfxApiResource r{};
    r.resource = t.res;
    r.description.width = t.w; r.description.height = t.h; r.description.depth = 1; r.description.mipCount = 1;
    r.state = state;
    return r;
}

// A device, a queue and one command list the test records each frame into ("the game's"), plus upload/readback staging.
struct Gpu {
    ID3D12Device* dev = nullptr;
    ID3D12CommandQueue* queue = nullptr;
    ID3D12CommandAllocator* alloc = nullptr;
    ID3D12GraphicsCommandList* list = nullptr;
    ID3D12Fence* fence = nullptr;
    uint64_t fenceValue = 0;
    HANDLE event = nullptr;
    ID3D12Resource *upload = nullptr, *readback = nullptr;
    uint8_t* up = nullptr;
    uint64_t upUsed = 0;

    int init() {
        CHECK(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&dev)));
        D3D12_COMMAND_QUEUE_DESC qd{D3D12_COMMAND_LIST_TYPE_DIRECT};
        CHECK(dev->CreateCommandQueue(&qd, IID_PPV_ARGS(&queue)));
        CHECK(dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&alloc)));
        CHECK(dev->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, alloc, nullptr, IID_PPV_ARGS(&list)));
        list->Close();
        CHECK(dev->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)));
        event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        upload = buffer(D3D12_HEAP_TYPE_UPLOAD, 64ull << 20, D3D12_RESOURCE_STATE_GENERIC_READ);
        readback = buffer(D3D12_HEAP_TYPE_READBACK, 16ull << 20, D3D12_RESOURCE_STATE_COPY_DEST);
        if (!upload || !readback) { std::fprintf(stderr, "staging buffers\n"); return 1; }
        CHECK(upload->Map(0, nullptr, reinterpret_cast<void**>(&up)));
        return 0;
    }
    ID3D12Resource* buffer(D3D12_HEAP_TYPE type, uint64_t size, D3D12_RESOURCE_STATES state) {
        D3D12_HEAP_PROPERTIES hp{type};
        D3D12_RESOURCE_DESC rd{};
        rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER; rd.Width = size; rd.Height = 1; rd.DepthOrArraySize = 1; rd.MipLevels = 1;
        rd.SampleDesc.Count = 1; rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        ID3D12Resource* r = nullptr;
        dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd, state, nullptr, IID_PPV_ARGS(&r));
        return r;
    }
    bool texture(Tex& t, DXGI_FORMAT fmt, uint32_t w, uint32_t h, uint32_t bpp, bool uav, D3D12_RESOURCE_STATES state) {
        D3D12_HEAP_PROPERTIES hp{D3D12_HEAP_TYPE_DEFAULT};
        D3D12_RESOURCE_DESC rd{};
        rd.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D; rd.Width = w; rd.Height = h; rd.DepthOrArraySize = 1; rd.MipLevels = 1;
        rd.Format = fmt; rd.SampleDesc.Count = 1; rd.Flags = uav ? D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS : D3D12_RESOURCE_FLAG_NONE;
        t = {nullptr, fmt, w, h, bpp};
        return SUCCEEDED(dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd, state, nullptr, IID_PPV_ARGS(&t.res)));
    }
    int begin() {
        CHECK(alloc->Reset());
        CHECK(list->Reset(alloc, nullptr));
        upUsed = 0;
        return 0;
    }
    // records the upload of tightly packed `data` into `t` (in state `state`, left in it)
    void upload2d(const Tex& t, const void* data, D3D12_RESOURCE_STATES state) {
        const uint32_t pitch = pitchOf(t), row = t.w * t.bpp;
        for (uint32_t y = 0; y < t.h; ++y) std::memcpy(up + upUsed + uint64_t(y) * pitch, static_cast<const uint8_t*>(data) + size_t(y) * row, row);
        barrier(list, t.res, state, D3D12_RESOURCE_STATE_COPY_DEST);
        D3D12_TEXTURE_COPY_LOCATION dst{t.res, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX}, src{upload, D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT};
        dst.SubresourceIndex = 0;
        src.PlacedFootprint = {upUsed, {t.fmt, t.w, t.h, 1, pitch}};
        list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
        barrier(list, t.res, D3D12_RESOURCE_STATE_COPY_DEST, state);
        upUsed = (upUsed + uint64_t(pitch) * t.h + 511) & ~511ull;
    }
    void readback2d(const Tex& t, D3D12_RESOURCE_STATES state) {
        barrier(list, t.res, state, D3D12_RESOURCE_STATE_COPY_SOURCE);
        D3D12_TEXTURE_COPY_LOCATION dst{readback, D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT}, src{t.res, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX};
        dst.PlacedFootprint = {0, {t.fmt, t.w, t.h, 1, pitchOf(t)}};
        src.SubresourceIndex = 0;
        list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
        barrier(list, t.res, D3D12_RESOURCE_STATE_COPY_SOURCE, state);
    }
    // closes, executes and waits; *ms = submit to completion
    int run(double* ms) {
        CHECK(list->Close());
        const auto t0 = std::chrono::steady_clock::now();
        ID3D12CommandList* lists[] = {list};
        queue->ExecuteCommandLists(1, lists);
        CHECK(queue->Signal(fence, ++fenceValue));
        CHECK(fence->SetEventOnCompletion(fenceValue, event));
        if (WaitForSingleObject(event, 10000) != WAIT_OBJECT_0) { std::fprintf(stderr, "GPU timeout\n"); return 1; }
        *ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        return 0;
    }
    int save(const Tex& t, const std::string& path) {
        uint8_t* rb = nullptr;
        CHECK(readback->Map(0, nullptr, reinterpret_cast<void**>(&rb)));
        std::vector<uint8_t> out(size_t(t.w) * t.h * t.bpp);
        for (uint32_t y = 0; y < t.h; ++y) std::memcpy(out.data() + size_t(y) * t.w * t.bpp, rb + uint64_t(y) * pitchOf(t), size_t(t.w) * t.bpp);
        readback->Unmap(0, nullptr);
        std::ofstream(path, std::ios::binary).write(reinterpret_cast<char*>(out.data()), out.size());
        return 0;
    }
};

struct Ffx { PfnFfxCreateContext create; PfnFfxDispatch dispatch; PfnFfxDestroyContext destroy; };
static bool ffxOf(OptiGetProcFn gp, Ffx& f) {
    if (!gp) return false;
    f.create = reinterpret_cast<PfnFfxCreateContext>(gp(nullptr, "ffxCreateContext"));
    f.dispatch = reinterpret_cast<PfnFfxDispatch>(gp(nullptr, "ffxDispatch"));
    f.destroy = reinterpret_cast<PfnFfxDestroyContext>(gp(nullptr, "ffxDestroyContext"));
    return f.create && f.dispatch && f.destroy;
}

constexpr uint32_t iw = 640, ih = 360, ow = 1280, oh = 720;

static int upscale(Gpu& gpu, const Ffx& ffx, const std::string& fx, int frames, const std::string& outDir) {
    ffxCreateBackendDX12Desc backend{};
    backend.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_DX12;
    backend.device = gpu.dev;
    ffxCreateContextDescUpscale cd{};
    cd.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE;
    cd.header.pNext = &backend.header;
    cd.flags = FFX_UPSCALE_ENABLE_DEPTH_INVERTED | FFX_UPSCALE_ENABLE_HIGH_DYNAMIC_RANGE;
    cd.maxRenderSize = {iw, ih}; cd.maxUpscaleSize = {ow, oh};
    ffxContext ctx = nullptr;
    if (ffx.create(&ctx, &cd.header, nullptr) != FFX_API_RETURN_OK) { std::fprintf(stderr, "ffxCreateContext failed (see m4r.log)\n"); return 1; }
    Tex color, depth, motion, exposure, output;
    if (!gpu.texture(color, DXGI_FORMAT_R16G16B16A16_FLOAT, iw, ih, 8, false, kRead) ||
        !gpu.texture(depth, DXGI_FORMAT_R32_FLOAT, iw, ih, 4, false, kRead) ||
        !gpu.texture(motion, DXGI_FORMAT_R16G16_FLOAT, iw, ih, 4, false, kRead) ||
        !gpu.texture(exposure, DXGI_FORMAT_R32_FLOAT, 1, 1, 4, false, kRead) ||
        !gpu.texture(output, DXGI_FORMAT_R16G16B16A16_FLOAT, ow, oh, 8, true, D3D12_RESOURCE_STATE_UNORDERED_ACCESS)) {
        std::fprintf(stderr, "textures\n"); return 1;
    }
    double total = 0;
    for (int t = 0; t < frames; ++t) {
        char nm[64];
        std::snprintf(nm, sizeof nm, "/input-%02d.rgba16f", t); auto c = readAll(fx + nm);
        std::snprintf(nm, sizeof nm, "/depth-%02d.r32f", t); auto d = readAll(fx + nm);
        std::snprintf(nm, sizeof nm, "/mvlo-%02d.rg16f", t); auto mv = readAll(fx + nm);
        if (c.size() != size_t(iw) * ih * 8 || d.size() != size_t(iw) * ih * 4 || mv.size() != size_t(iw) * ih * 4) {
            std::fprintf(stderr, "frame %d: fixture size mismatch\n", t); return 1;
        }
        const float one = 1.0f;
        if (gpu.begin()) return 1;
        // the "game's" work before the upscaler
        gpu.upload2d(color, c.data(), kRead); gpu.upload2d(depth, d.data(), kRead); gpu.upload2d(motion, mv.data(), kRead);
        gpu.upload2d(exposure, &one, kRead);
        ffxDispatchDescUpscale dd{};
        dd.header.type = FFX_API_DISPATCH_DESC_TYPE_UPSCALE;
        dd.commandList = gpu.list;
        dd.color = ffxRes(color, FFX_API_RESOURCE_STATE_PIXEL_COMPUTE_READ);
        dd.depth = ffxRes(depth, FFX_API_RESOURCE_STATE_PIXEL_COMPUTE_READ);
        dd.motionVectors = ffxRes(motion, FFX_API_RESOURCE_STATE_PIXEL_COMPUTE_READ);
        dd.exposure = ffxRes(exposure, FFX_API_RESOURCE_STATE_PIXEL_COMPUTE_READ);
        dd.output = ffxRes(output, FFX_API_RESOURCE_STATE_UNORDERED_ACCESS);
        dd.jitterOffset = {float(halton(t + 1, 2)), float(halton(t + 1, 3))};   // run_bbr's "-1" jitter sign, in m4r's convention
        dd.motionVectorScale = {1.0f, 1.0f};
        dd.renderSize = {iw, ih}; dd.upscaleSize = {ow, oh};
        dd.preExposure = 1.0f;
        dd.reset = t == 0;
        dd.frameTimeDelta = 16.6f;
        if (ffx.dispatch(&ctx, &dd.header) != FFX_API_RETURN_OK) { std::fprintf(stderr, "frame %d: ffxDispatch failed (see m4r.log)\n", t); return 1; }
        // the "game's" work after it
        gpu.readback2d(output, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        double ms = 0;
        if (gpu.run(&ms)) return 1;
        if (t > 0) total += ms;
        std::snprintf(nm, sizeof nm, "/out-%02d.rgba16f", t);
        if (gpu.save(output, outDir + nm)) return 1;
        std::printf("frame %d: %.3f ms (submit to completion)\n", t, ms);
    }
    if (frames > 1) std::printf("average after the first frame: %.3f ms\n", total / (frames - 1));
    if (ffx.destroy(&ctx, nullptr) != FFX_API_RETURN_OK) { std::fprintf(stderr, "ffxDestroyContext failed\n"); return 1; }
    return 0;
}

static int framegen(Gpu& gpu, const Ffx& ffx, const std::string& fx, int frames, const std::string& outDir) {
    ffxCreateBackendDX12Desc backend{};
    backend.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_DX12;
    backend.device = gpu.dev;
    ffxCreateContextDescFrameGeneration cd{};
    cd.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_FRAMEGENERATION;
    cd.header.pNext = &backend.header;
    cd.flags = FFX_FRAMEGENERATION_ENABLE_DEPTH_INVERTED | FFX_FRAMEGENERATION_ENABLE_HIGH_DYNAMIC_RANGE;
    cd.displaySize = {ow, oh}; cd.maxRenderSize = {iw, ih};
    cd.backBufferFormat = FFX_API_SURFACE_FORMAT_R16G16B16A16_FLOAT;
    ffxContext ctx = nullptr;
    const ffxReturnCode_t cr = ffx.create(&ctx, &cd.header, nullptr);
    if (cr != FFX_API_RETURN_OK) { std::fprintf(stderr, "ffxCreateContext (frame generation) failed: %u (see m4r.log)\n", cr); return 1; }
    Tex present, depth, motion, output;
    if (!gpu.texture(present, DXGI_FORMAT_R16G16B16A16_FLOAT, ow, oh, 8, false, kRead) ||
        !gpu.texture(depth, DXGI_FORMAT_R32_FLOAT, iw, ih, 4, false, kRead) ||
        !gpu.texture(motion, DXGI_FORMAT_R16G16_FLOAT, iw, ih, 4, false, kRead) ||
        !gpu.texture(output, DXGI_FORMAT_R16G16B16A16_FLOAT, ow, oh, 8, true, D3D12_RESOURCE_STATE_UNORDERED_ACCESS)) {
        std::fprintf(stderr, "textures\n"); return 1;
    }
    double total = 0;
    for (int t = 0; t < frames; ++t) {
        char nm[64];
        std::snprintf(nm, sizeof nm, "/color-%02d.rgba16f", t); auto c = readAll(fx + nm);
        std::snprintf(nm, sizeof nm, "/depth-%02d.r32f", t); auto d = readAll(fx + nm);
        std::snprintf(nm, sizeof nm, "/mv-%02d.rg16f", t); auto mv = readAll(fx + nm);
        if (c.size() != size_t(ow) * oh * 8 || d.size() != size_t(iw) * ih * 4 || mv.size() != size_t(iw) * ih * 4) {
            std::fprintf(stderr, "frame %d: fixture size mismatch\n", t); return 1;
        }
        if (gpu.begin()) return 1;
        gpu.upload2d(present, c.data(), kRead); gpu.upload2d(depth, d.data(), kRead); gpu.upload2d(motion, mv.data(), kRead);
        // the game's prepare dispatch (OptiScaler sends it with the frame's depth and motion vectors)
        ffxDispatchDescFrameGenerationPrepare pd{};
        pd.header.type = FFX_API_DISPATCH_DESC_TYPE_FRAMEGENERATION_PREPARE;
        pd.frameID = uint64_t(t);
        pd.commandList = gpu.list;
        pd.renderSize = {iw, ih};
        pd.motionVectorScale = {1.0f, 1.0f};
        pd.frameTimeDelta = 1000.0f / 60.0f;
        pd.cameraNear = 0.1f; pd.cameraFar = 1000.0f; pd.cameraFovAngleVertical = 1.0f; pd.viewSpaceToMetersFactor = 1.0f;
        pd.depth = ffxRes(depth, FFX_API_RESOURCE_STATE_PIXEL_COMPUTE_READ);
        pd.motionVectors = ffxRes(motion, FFX_API_RESOURCE_STATE_PIXEL_COMPUTE_READ);
        if (ffx.dispatch(&ctx, &pd.header) != FFX_API_RETURN_OK) { std::fprintf(stderr, "frame %d: prepare failed\n", t); return 1; }
        // the swapchain's frame generation callback: the frame between the previous one and this one
        ffxDispatchDescFrameGeneration gd{};
        gd.header.type = FFX_API_DISPATCH_DESC_TYPE_FRAMEGENERATION;
        gd.commandList = gpu.list;
        gd.presentColor = ffxRes(present, FFX_API_RESOURCE_STATE_PIXEL_COMPUTE_READ);
        gd.outputs[0] = ffxRes(output, FFX_API_RESOURCE_STATE_UNORDERED_ACCESS);
        gd.numGeneratedFrames = 1;
        gd.generationRect = {0, 0, int32_t(ow), int32_t(oh)};
        gd.frameID = uint64_t(t);
        if (ffx.dispatch(&ctx, &gd.header) != FFX_API_RETURN_OK) { std::fprintf(stderr, "frame %d: frame generation failed (see m4r.log)\n", t); return 1; }
        gpu.readback2d(output, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        double ms = 0;
        if (gpu.run(&ms)) return 1;
        if (t > 1) total += ms;
        if (t > 0) {
            std::snprintf(nm, sizeof nm, "/interp-%02d.rgba16f", t);
            if (gpu.save(output, outDir + nm)) return 1;
        }
        std::printf("frame %d: %.3f ms (submit to completion)\n", t, ms);
    }
    if (frames > 2) std::printf("average from frame 2: %.3f ms\n", total / (frames - 2));
    if (ffx.destroy(&ctx, nullptr) != FFX_API_RETURN_OK) { std::fprintf(stderr, "ffxDestroyContext failed\n"); return 1; }
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 6) { std::fprintf(stderr, "usage: %s up|fg <m4r.asi> <fixture dir> <frames> <out dir>\n", argv[0]); return 2; }
    const std::string mode = argv[1], asi = argv[2], fx = argv[3], outDir = argv[5];
    const int frames = std::atoi(argv[4]);
    Gpu gpu;
    if (gpu.init()) return 1;
    HMODULE m = LoadLibraryA(asi.c_str());
    if (!m) { std::fprintf(stderr, "LoadLibrary %s: %lu\n", asi.c_str(), GetLastError()); return 1; }
    auto init = reinterpret_cast<PFN_OptiPluginInit>(GetProcAddress(m, OPTI_PLUGIN_INIT_NAME));
    if (!init) { std::fprintf(stderr, "no OptiPluginInit\n"); return 1; }
    OptiPluginApi api{};
    api.size = sizeof api; api.version = OPTI_PLUGIN_API_VERSION;
    api.Log = &logFn; api.RegisterProvider = &registerProvider; api.RegisterTab = &registerTab; api.GetString = &getString; api.GetBool = &getBool;
    if (!init(&api, reinterpret_cast<OptiPlugin*>(&api))) { std::fprintf(stderr, "OptiPluginInit failed\n"); return 1; }
    Ffx ffx{};
    const OptiProviderKind kind = mode == "fg" ? OptiProvider_FfxFg : OptiProvider_Ffx;
    if (!ffxOf(g_getProc[kind], ffx)) { std::fprintf(stderr, "the plugin registered no %s provider\n", mode.c_str()); return 1; }
    const int rc = mode == "fg" ? framegen(gpu, ffx, fx, frames, outDir) : upscale(gpu, ffx, fx, frames, outDir);
    if (rc == 0) std::printf("ok\n");
    return rc;
}
