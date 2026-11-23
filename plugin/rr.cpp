// Ray Reconstruction (DLSS-RR's NGX feature, OptiScaler's "RRUpscaler" list, provider kind Dlssd) on the backend's denoiser
// (backend.h BE_RR: m4r's MetalFX denoiser followed by MetalFX's own upscaling, m4r_denoise).
//
// OptiScaler loads an Dlssd provider like an nvngx_dlssd.dll: NVSDK_NGX_D3D12_Init(_Ext), CreateFeature, EvaluateFeature,
// ReleaseFeature with its own NVSDK_NGX_Parameter objects. Those are a C++ interface with overloaded virtual Set/Get in MSVC's
// vtable order, which this file (built with clang-cl) declares as it is. The evaluate packs every input the game gives into shared
// buffers (dx_interop.h), splits the game's list there and runs the denoiser on HIP between the parts, like the upscaler; the
// inputs and their meaning are those of m4r's earlier OptiScaler patch (docs/dlssd-inputs.md).
// SPDX-License-Identifier: GPL-3.0-only
#include "asi.h"
#ifdef BE_RR
#include <algorithm>
#include <cstring>

namespace m4rasi {
namespace {

using NgxResult = unsigned int;
constexpr NgxResult kOk = 0x1, kFail = 0xBAD00000u, kFailFeatureNotSupported = 0xBAD00001u, kFailInvalidParameter = 0xBAD00005u;

struct NgxHandle { unsigned int Id; };
struct NgxParameter {   // NVSDK_NGX_Parameter (nvsdk_ngx_defs.h), MSVC vtable order
    virtual void Set(const char* name, unsigned long long value) = 0;
    virtual void Set(const char* name, float value) = 0;
    virtual void Set(const char* name, double value) = 0;
    virtual void Set(const char* name, unsigned int value) = 0;
    virtual void Set(const char* name, int value) = 0;
    virtual void Set(const char* name, struct ID3D11Resource* value) = 0;
    virtual void Set(const char* name, ID3D12Resource* value) = 0;
    virtual void Set(const char* name, void* value) = 0;
    virtual NgxResult Get(const char* name, unsigned long long* value) const = 0;
    virtual NgxResult Get(const char* name, float* value) const = 0;
    virtual NgxResult Get(const char* name, double* value) const = 0;
    virtual NgxResult Get(const char* name, unsigned int* value) const = 0;
    virtual NgxResult Get(const char* name, int* value) const = 0;
    virtual NgxResult Get(const char* name, struct ID3D11Resource** value) const = 0;
    virtual NgxResult Get(const char* name, ID3D12Resource** value) const = 0;
    virtual NgxResult Get(const char* name, void** value) const = 0;
    virtual void Reset() = 0;
};
struct NgxFeatureRequirement { int FeatureSupported; unsigned int MinHWArchitecture; char MinOSVersion[255]; };

// DLSS create flags (NVSDK_NGX_DLSS_Feature_Flags)
constexpr int kMvLowRes = 1 << 1, kDepthInverted = 1 << 3;

ID3D12Resource* res(const NgxParameter* p, const char* key) {
    ID3D12Resource* r = nullptr;
    if (p->Get(key, &r) != kOk || !r) { void* v = nullptr; if (p->Get(key, &v) == kOk) r = static_cast<ID3D12Resource*>(v); }
    return r;
}
uint32_t u32(const NgxParameter* p, const char* key, uint32_t def = 0) {
    unsigned int v = 0; int i = 0;
    if (p->Get(key, &v) == kOk) return v;
    if (p->Get(key, &i) == kOk) return uint32_t(i);
    return def;
}
float f32(const NgxParameter* p, const char* key, float def) { float v = 0; return p->Get(key, &v) == kOk ? v : def; }

// one game input: its shared buffer and what m4r is told about it
struct Input { const char* keys[2]; dxi::Layout layout; bool display; dxi::Shared shared; };

struct Denoiser {
    unsigned int id = 0;
    uint32_t renderW = 0, renderH = 0, outW = 0, outH = 0;
    int flags = 0;
    bool broken = false;
    dxi::Shared output;
    // order: color, depth, motion, diffuse albedo, specular albedo, normals, roughness, specular hit, diffuse hit, colour before
    // particles, exposure
    Input in[11] = {
        {{"Color", nullptr}, dxi::Layout::RGBA16F, false, {}},
        {{"Depth", nullptr}, dxi::Layout::R32F, false, {}},
        {{"MotionVectors", nullptr}, dxi::Layout::RG16F, false, {}},
        {{"DLSS.Input.DiffuseAlbedo", "GBuffer.DiffuseAlbedo"}, dxi::Layout::RGBA8, false, {}},   // as games give them; m4r's albedo
        {{"DLSS.Input.SpecularAlbedo", "GBuffer.SpecularAlbedo"}, dxi::Layout::RGBA8, false, {}},  // detail needs RGBA8
        {{"GBuffer.Normals", nullptr}, dxi::Layout::RGBA16F, false, {}},
        {{"GBuffer.Roughness", nullptr}, dxi::Layout::R32F, false, {}},
        {{"DLSSD.SpecularHitDistance", "DLSSD.SpecularRayDirectionHitDistance"}, dxi::Layout::RGBA32F, false, {}},
        {{"DLSSD.DiffuseHitDistance", nullptr}, dxi::Layout::R32F, false, {}},
        {{"DLSSD.ColorBeforeParticles", nullptr}, dxi::Layout::RGBA16F, false, {}},
        {{"ExposureTexture", nullptr}, dxi::Layout::R32F, false, {}},
    };
    Parts parts;
};

std::mutex g_featuresMutex;
std::vector<Denoiser*> g_features;
unsigned int g_nextId = 1;
ID3D12Device* g_device = nullptr;

Denoiser* find(const NgxHandle* h) {
    std::lock_guard<std::mutex> lock(g_featuresMutex);
    for (Denoiser* d : g_features) if (h && d->id == h->Id) return d;
    return nullptr;
}

void release(Denoiser& d) {
    d.parts.drain();
    hipvk::finish();
    for (auto& i : d.in) g.dx.release(i.shared);
    g.dx.release(d.output);
}

void runDenoise(void* owner, ID3D12CommandQueue* queue) {
    Part* p = static_cast<Part*>(owner);
    std::lock_guard<std::mutex> lock(g.mutex);
    vkBeginCommandBuffer(g.cb, nullptr);
    if (be_denoise(g.m4r, g.cb, &p->denoise) != 0) {
        log("ray reconstruction: %s", be_last_error(g.m4r));
        vkBeginCommandBuffer(g.cb, nullptr);   // an empty submission keeps the fence handoff intact
    }
    vkEndCommandBuffer(g.cb);
    if (!g.dx.handoff(queue, g.cb)) log("%s", g.dx.error.c_str());
}

NgxResult init(unsigned long long, const wchar_t*, ID3D12Device* device) {
    if (!device) return kFailInvalidParameter;
    std::lock_guard<std::mutex> lock(g.mutex);
    g_device = device;
    return ready(device) ? kOk : kFail;
}

NgxResult __cdecl Init(unsigned long long app, const wchar_t* path, ID3D12Device* device, const void*, int) { return init(app, path, device); }
NgxResult __cdecl InitExt(unsigned long long app, const wchar_t* path, ID3D12Device* device, int, const void*) { return init(app, path, device); }
NgxResult __cdecl Shutdown() { return kOk; }
NgxResult __cdecl Shutdown1(ID3D12Device*) { return kOk; }
NgxResult __cdecl GetScratchBufferSize(int, const NgxParameter*, size_t* out) { if (out) *out = 0; return kOk; }
NgxResult __cdecl PopulateParameters(NgxParameter* p) {
    if (!p) return kFailInvalidParameter;
    p->Set("SuperSamplingDenoising.Available", 1);
    p->Set("SuperSamplingDenoising.FeatureInitResult", 1);
    p->Set("SuperSamplingDenoising.NeedsUpdatedDriver", 0);
    return kOk;
}
NgxResult __cdecl GetFeatureRequirements(void*, const void*, NgxFeatureRequirement* out) {
    if (!out) return kFailInvalidParameter;
    out->FeatureSupported = 0;   // NVSDK_NGX_FeatureSupportResult_Supported
    out->MinHWArchitecture = 0;
    std::strcpy(out->MinOSVersion, "10.0.10240.16384");
    return kOk;
}

NgxResult __cdecl CreateFeature(ID3D12GraphicsCommandList*, int feature, NgxParameter* p, NgxHandle** out) {
    if (!p || !out) return kFailInvalidParameter;
    if (feature != 13) return kFailFeatureNotSupported;   // NVSDK_NGX_Feature_RayReconstruction
    auto* d = new Denoiser;
    d->renderW = u32(p, "Width"); d->renderH = u32(p, "Height");
    d->outW = u32(p, "OutWidth"); d->outH = u32(p, "OutHeight");
    d->flags = int(u32(p, "DLSS.Feature.Create.Flags"));
    if (!d->renderW || !d->renderH || !d->outW || !d->outH) { delete d; return kFailInvalidParameter; }
    {
        std::lock_guard<std::mutex> lock(g.mutex);
        if (!g.m4r || be_denoise_init(g.m4r, d->outW, d->outH, d->renderW, d->renderH) != 0) {
            log("ray reconstruction %ux%u -> %ux%u: %s", d->renderW, d->renderH, d->outW, d->outH, g.m4r ? be_last_error(g.m4r) : "not initialised");
            delete d;
            return kFail;
        }
        if (!g.dx.create(d->output, d->outW, d->outH, dxi::Layout::RGBA16F)) { log("interop: %s", g.dx.error.c_str()); delete d; return kFail; }
    }
    log("ray reconstruction: render %ux%u -> %ux%u, flags 0x%x", d->renderW, d->renderH, d->outW, d->outH, d->flags);
    std::lock_guard<std::mutex> lock(g_featuresMutex);
    d->id = g_nextId++;
    g_features.push_back(d);
    *out = new NgxHandle{d->id};   // the caller keeps it until ReleaseFeature
    return kOk;
}

NgxResult __cdecl ReleaseFeature(NgxHandle* h) {
    Denoiser* d = find(h);
    if (!d) return kFailInvalidParameter;
    {
        std::lock_guard<std::mutex> lock(g_featuresMutex);
        g_features.erase(std::find(g_features.begin(), g_features.end(), d));
    }
    std::lock_guard<std::mutex> lock(g.mutex);
    release(*d);
    delete d;
    delete h;
    return kOk;
}

m4r_image image(const dxi::Shared& s) { return {s.image, dxi::layoutFormat(s.layout), s.w, s.h, VK_IMAGE_LAYOUT_GENERAL}; }

NgxResult __cdecl EvaluateFeature(ID3D12GraphicsCommandList* list, const NgxHandle* h, NgxParameter* p, void*) {
    Denoiser* d = find(h);
    if (!d || !list || !p) return kFailInvalidParameter;
    ID3D12Resource* out = res(p, "Output");
    if (!out) return kFailInvalidParameter;
    std::lock_guard<std::mutex> lock(g.mutex);
    if (d->broken) return kFail;
    const uint32_t rw = std::min(u32(p, "DLSS.Render.Subrect.Dimensions.Width", d->renderW), d->renderW);
    const uint32_t rh = std::min(u32(p, "DLSS.Render.Subrect.Dimensions.Height", d->renderH), d->renderH);
    const bool lowResMv = (d->flags & kMvLowRes) != 0;
    ID3D12Resource* r[11] = {};
    for (int i = 0; i < 11; ++i) {
        Input& in = d->in[i];
        r[i] = res(p, in.keys[0]);
        if (!r[i] && in.keys[1]) r[i] = res(p, in.keys[1]);
        if (!r[i]) continue;
        const bool one = i == 10;   // the 1x1 exposure
        const uint32_t w = one ? 1 : d->renderW, hh = one ? 1 : d->renderH;
        if (!in.shared && !g.dx.create(in.shared, w, hh, in.layout)) { log("interop: %s", g.dx.error.c_str()); d->broken = true; return kFail; }
        if (one) g.dx.pack(list, r[i], in.shared, 1, 1);
        else if (i == 2 && !lowResMv) g.dx.packResampled(list, r[i], in.shared, rw, rh, d->outW, d->outH);   // display-size vectors
        else g.dx.pack(list, r[i], in.shared, rw, rh);
    }
    if (!r[0] || !r[1] || !r[2]) { log("ray reconstruction: colour, depth or motion missing"); return kFailInvalidParameter; }
    Part* part = split(d->parts, list, &runDenoise);
    if (!part) { log("ray reconstruction: the command list could not be split"); return kFail; }
    be_denoise_desc& m = part->denoise;
    m = {};
    m4r_image* slots[11] = {&m.color, &m.depth, &m.motion, &m.diffuse_albedo, &m.specular_albedo, &m.normals, &m.roughness,
                            &m.specular_hit, &m.diffuse_hit, &m.color_before_particles, &m.exposure};
    for (int i = 0; i < 11; ++i)
        if (r[i]) *slots[i] = image(d->in[i].shared);
    m.output = image(d->output);
    void *world = nullptr, *proj = nullptr;
    p->Get("WorldToViewMatrix", &world);
    p->Get("ViewToClipMatrix", &proj);
    if (world) std::memcpy(m.world_to_view, world, sizeof m.world_to_view);
    if (proj) std::memcpy(m.view_to_clip, proj, sizeof m.view_to_clip);
    m.content_width = rw; m.content_height = rh;
    m.jitter_x = f32(p, "Jitter.Offset.X", 0.0f); m.jitter_y = f32(p, "Jitter.Offset.Y", 0.0f);
    m.motion_scale_x = f32(p, "MV.Scale.X", 1.0f); m.motion_scale_y = f32(p, "MV.Scale.Y", 1.0f);
    if (!lowResMv) { m.motion_scale_x *= float(rw) / float(d->outW); m.motion_scale_y *= float(rh) / float(d->outH); }
    m.pre_exposure = f32(p, "DLSS.Pre.Exposure", 1.0f);
    if (!(m.pre_exposure > 0.0f)) m.pre_exposure = 1.0f;
    m.reset = int(u32(p, "Reset"));
    m.depth_reversed = (d->flags & kDepthInverted) ? 1 : 0;
    m.hardware_depth = int(u32(p, "DLSS.Use.HW.Depth", 1));
    m.roughness_in_normals_w = u32(p, "DLSS.Roughness.Mode") == 1;
    m.motion_low_res = 1;   // the vectors were brought to render size above
    m.specular_hit_in_w = !res(p, "DLSSD.SpecularHitDistance") && r[7];
    static const bool flip = std::getenv("M4R_FLIP_JITTER") && std::atoi(std::getenv("M4R_FLIP_JITTER")) != 0;
    m.flip_upscale_jitter = flip ? 1 : 0;
    m.fast_math = 0;
    g.dx.unpack(list, d->output, out, d->outW, d->outH);
    return kOk;
}

} // namespace

void* rrGetProc(void*, const char* name) {
    static const struct { const char* name; void* fn; } table[] = {
        {"NVSDK_NGX_D3D12_Init", reinterpret_cast<void*>(&Init)},
        {"NVSDK_NGX_D3D12_Init_Ext", reinterpret_cast<void*>(&InitExt)},
        {"NVSDK_NGX_D3D12_Shutdown", reinterpret_cast<void*>(&Shutdown)},
        {"NVSDK_NGX_D3D12_Shutdown1", reinterpret_cast<void*>(&Shutdown1)},
        {"NVSDK_NGX_D3D12_GetScratchBufferSize", reinterpret_cast<void*>(&GetScratchBufferSize)},
        {"NVSDK_NGX_D3D12_CreateFeature", reinterpret_cast<void*>(&CreateFeature)},
        {"NVSDK_NGX_D3D12_ReleaseFeature", reinterpret_cast<void*>(&ReleaseFeature)},
        {"NVSDK_NGX_D3D12_EvaluateFeature", reinterpret_cast<void*>(&EvaluateFeature)},
        {"NVSDK_NGX_D3D12_GetFeatureRequirements", reinterpret_cast<void*>(&GetFeatureRequirements)},
        {"NVSDK_NGX_D3D12_PopulateParameters_Impl", reinterpret_cast<void*>(&PopulateParameters)},
    };
    for (const auto& e : table)
        if (!std::strcmp(name, e.name)) return e.fn;
    return nullptr;
}

} // namespace m4rasi
#endif // BE_RR
