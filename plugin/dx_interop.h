// m4r plugin: the D3D12 side of running HIP work inside a game's frame (native Windows, "d3dhip").
//
// A game texture reaches HIP through a shared, pitch-linear D3D12 buffer: a small compute shader recorded into the game's command
// list converts the texture (any format the game uses) into the buffer as RGBA16F / RG16F / R32F rows of 256-byte aligned pitch,
// HIP maps the same memory (hipvk::importD3D12) and wraps it as an image (hipvk::wrapLinear). Results go back the other way: HIP
// writes a shared buffer and a shader writes it into the game's texture (a UAV). The shaders are compiled at run time with
// d3dcompiler_47 (part of Windows), so the plugin needs no shader toolchain.
//
// Ordering is GPU-side (d4r's d3dhip handoff): the game's list is split where HIP's work goes (d4r_d3d12_split.h); between the
// parts the queue signals a shared fence, HIP waits for it, runs and signals it again, and the queue waits for that before the
// second part.
// SPDX-License-Identifier: GPL-3.0-only
#pragma once
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <d3d12.h>
#include <d3dcompiler.h>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>
#include "hipvk/hipvk.h"
#include "vkfn.h"

namespace dxi {

enum class Layout : uint32_t { RGBA16F = 0, R32F = 1, RG16F = 2, RGBA32F = 3, RGBA8 = 4 };   // RGBA8: unorm
inline uint32_t layoutBytes(Layout l) { return l == Layout::RGBA16F ? 8 : l == Layout::RGBA32F ? 16 : 4; }
inline VkFormat layoutFormat(Layout l) {
    switch (l) {
    case Layout::RGBA16F: return VK_FORMAT_R16G16B16A16_SFLOAT;
    case Layout::R32F: return VK_FORMAT_R32_SFLOAT;
    case Layout::RG16F: return VK_FORMAT_R16G16_SFLOAT;
    case Layout::RGBA8: return VK_FORMAT_R8G8B8A8_UNORM;
    default: return VK_FORMAT_R32G32B32A32_SFLOAT;
    }
}

// A view format a shader can read (SRV) or write (UAV) for a game resource's format (typeless and depth formats resolved).
inline DXGI_FORMAT readable(DXGI_FORMAT f) {
    switch (f) {
    case DXGI_FORMAT_R32_TYPELESS: case DXGI_FORMAT_D32_FLOAT: return DXGI_FORMAT_R32_FLOAT;
    case DXGI_FORMAT_R24G8_TYPELESS: case DXGI_FORMAT_D24_UNORM_S8_UINT: return DXGI_FORMAT_R24_UNORM_X8_TYPELESS;
    case DXGI_FORMAT_R32G8X24_TYPELESS: case DXGI_FORMAT_D32_FLOAT_S8X24_UINT: return DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS;
    case DXGI_FORMAT_R16_TYPELESS: case DXGI_FORMAT_D16_UNORM: return DXGI_FORMAT_R16_UNORM;
    case DXGI_FORMAT_R16G16B16A16_TYPELESS: return DXGI_FORMAT_R16G16B16A16_FLOAT;
    case DXGI_FORMAT_R32G32B32A32_TYPELESS: return DXGI_FORMAT_R32G32B32A32_FLOAT;
    case DXGI_FORMAT_R32G32_TYPELESS: return DXGI_FORMAT_R32G32_FLOAT;
    case DXGI_FORMAT_R16G16_TYPELESS: return DXGI_FORMAT_R16G16_FLOAT;
    case DXGI_FORMAT_R10G10B10A2_TYPELESS: return DXGI_FORMAT_R10G10B10A2_UNORM;
    case DXGI_FORMAT_R8G8B8A8_TYPELESS: return DXGI_FORMAT_R8G8B8A8_UNORM;
    case DXGI_FORMAT_B8G8R8A8_TYPELESS: return DXGI_FORMAT_B8G8R8A8_UNORM;
    case DXGI_FORMAT_R16_FLOAT: case DXGI_FORMAT_R32_FLOAT: default: return f;
    }
}
// sRGB views cannot be UAVs: their linear twin (the shader then writes encoded values itself: not needed for HDR upscaler outputs)
inline DXGI_FORMAT writable(DXGI_FORMAT f) {
    f = readable(f);
    if (f == DXGI_FORMAT_R8G8B8A8_UNORM_SRGB) return DXGI_FORMAT_R8G8B8A8_UNORM;
    if (f == DXGI_FORMAT_B8G8R8A8_UNORM_SRGB) return DXGI_FORMAT_B8G8R8A8_UNORM;
    return f;
}

// Pack: game texture (t0) -> linear buffer (u0). Unpack: linear buffer (t1) -> game texture (u1).
// Constants: width, height, pitch (bytes), layout, source x/y offset, scale, mode flags (1: alpha = 1 on unpack)
static const char kShaders[] = R"(
cbuffer C : register(b0) { uint w, h, pitch, layout, ox, oy; float scale; uint flags; };
Texture2D<float4> src : register(t0);
RWByteAddressBuffer dst : register(u0);
ByteAddressBuffer srcBuf : register(t1);
RWTexture2D<float4> dstTex : register(u1);
uint pack2(float a, float b) { return f32tof16(a) | (f32tof16(b) << 16); }
[numthreads(8, 8, 1)] void pack(uint3 id : SV_DispatchThreadID) {
    if (id.x >= w || id.y >= h) return;
    uint2 p = id.xy + uint2(ox, oy);
    if (flags & 2) p = uint2((float2(id.xy) + 0.5) * float2(ox, oy) / float2(w, h));   // nearest from an ox x oy source region
    float4 v = src.Load(int3(p, 0)) * scale;
    uint off = id.y * pitch;
    if (layout == 0) dst.Store2(off + id.x * 8, uint2(pack2(v.x, v.y), pack2(v.z, v.w)));
    else if (layout == 1) dst.Store(off + id.x * 4, asuint(v.x));
    else if (layout == 2) dst.Store(off + id.x * 4, pack2(v.x, v.y));
    else if (layout == 4) { uint4 q = uint4(round(saturate(v) * 255.0)); dst.Store(off + id.x * 4, q.x | (q.y << 8) | (q.z << 16) | (q.w << 24)); }
    else dst.Store4(off + id.x * 16, asuint(v));
}
[numthreads(8, 8, 1)] void unpack(uint3 id : SV_DispatchThreadID) {
    if (id.x >= w || id.y >= h) return;
    uint off = (id.y + oy) * pitch;
    float4 v;
    if (layout == 0) { uint2 d = srcBuf.Load2(off + (id.x + ox) * 8); v = float4(f16tof32(d.x), f16tof32(d.x >> 16), f16tof32(d.y), f16tof32(d.y >> 16)); }
    else if (layout == 1) v = float4(asfloat(srcBuf.Load(off + (id.x + ox) * 4)), 0, 0, 1);
    else if (layout == 2) { uint d = srcBuf.Load(off + (id.x + ox) * 4); v = float4(f16tof32(d), f16tof32(d >> 16), 0, 1); }
    else if (layout == 4) { uint d = srcBuf.Load(off + (id.x + ox) * 4); v = float4(d & 255, (d >> 8) & 255, (d >> 16) & 255, d >> 24) / 255.0; }
    else v = asfloat(srcBuf.Load4(off + (id.x + ox) * 16));
    if (flags & 1) v.w = 1;
    dstTex[id.xy] = v * scale;
}
)";

// FFX_API_RESOURCE_STATE_* (ffx_api_types.h) -> D3D12
inline D3D12_RESOURCE_STATES d3dState(uint32_t ffx) {
    if (ffx == 0 || ffx == (1u << 0)) return D3D12_RESOURCE_STATE_COMMON;
    if (ffx == (1u << 1)) return D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    if (ffx == (1u << 7)) return D3D12_RESOURCE_STATE_PRESENT;
    if (ffx == (1u << 8)) return D3D12_RESOURCE_STATE_RENDER_TARGET;
    if (ffx == (1u << 5)) return D3D12_RESOURCE_STATE_COPY_DEST;
    if (ffx == (1u << 6)) return D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT;
    D3D12_RESOURCE_STATES s = D3D12_RESOURCE_STATE_COMMON;
    if (ffx & (1u << 2)) s |= D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    if (ffx & (1u << 3)) s |= D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
    if (ffx & (1u << 4)) s |= D3D12_RESOURCE_STATE_COPY_SOURCE;
    return s;
}

// Moves `r` from `from` to `to` (nothing if a shader read is already allowed / the states are equal).
inline void transition(ID3D12GraphicsCommandList* list, ID3D12Resource* r, D3D12_RESOURCE_STATES from, D3D12_RESOURCE_STATES to) {
    if (from == to) return;
    if (to == D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE && (from & D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE)) return;
    D3D12_RESOURCE_BARRIER b{};
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Transition = {r, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, from, to};
    list->ResourceBarrier(1, &b);
}

struct Consts { uint32_t w, h, pitch, layout, ox, oy; float scale; uint32_t flags; };

// A pitch-linear buffer shared between D3D12 and HIP, seen by HIP as an image.
struct Shared {
    ID3D12Resource* buffer = nullptr;
    hipvk::External ext;
    VkImage image = VK_NULL_HANDLE;
    Layout layout = Layout::RGBA16F;
    uint32_t w = 0, h = 0, pitch = 0;
    explicit operator bool() const { return image != VK_NULL_HANDLE; }
};

class Interop {
public:
    std::string error;

    bool init(ID3D12Device* device) {
        if (device_) return true;
        device_ = device;
        HMODULE dc = LoadLibraryA("d3dcompiler_47.dll");
        auto compile = dc ? reinterpret_cast<pD3DCompile>(GetProcAddress(dc, "D3DCompile")) : nullptr;
        if (!compile) return fail("d3dcompiler_47.dll is missing");
        ID3DBlob *pack = nullptr, *unpack = nullptr, *err = nullptr;
        if (FAILED(compile(kShaders, sizeof kShaders - 1, "dx_interop", nullptr, nullptr, "pack", "cs_5_0", D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, &pack, &err)) ||
            FAILED(compile(kShaders, sizeof kShaders - 1, "dx_interop", nullptr, nullptr, "unpack", "cs_5_0", D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, &unpack, &err))) {
            std::string m = err ? std::string(static_cast<const char*>(err->GetBufferPointer()), err->GetBufferSize()) : "?";
            return fail("interop shaders: " + m);
        }
        D3D12_DESCRIPTOR_RANGE srv{D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0, 0, 0};
        D3D12_DESCRIPTOR_RANGE uav{D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 1, 0, 0};
        D3D12_ROOT_PARAMETER p[5] = {};
        p[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS; p[0].Constants = {0, 0, 8};
        p[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE; p[1].DescriptorTable = {1, &srv};
        p[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE; p[2].DescriptorTable = {1, &uav};
        p[3].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV; p[3].Descriptor = {0, 0};
        p[4].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV; p[4].Descriptor = {1, 0};
        D3D12_ROOT_SIGNATURE_DESC rd{5, p, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE};
        ID3DBlob* sig = nullptr;
        if (FAILED(D3D12SerializeRootSignature(&rd, D3D_ROOT_SIGNATURE_VERSION_1, &sig, &err)) ||
            FAILED(device->CreateRootSignature(0, sig->GetBufferPointer(), sig->GetBufferSize(), IID_PPV_ARGS(&root_))))
            return fail("interop root signature");
        D3D12_COMPUTE_PIPELINE_STATE_DESC pd{};
        pd.pRootSignature = root_;
        pd.CS = {pack->GetBufferPointer(), pack->GetBufferSize()};
        if (FAILED(device->CreateComputePipelineState(&pd, IID_PPV_ARGS(&pack_)))) return fail("pack pipeline");
        pd.CS = {unpack->GetBufferPointer(), unpack->GetBufferSize()};
        if (FAILED(device->CreateComputePipelineState(&pd, IID_PPV_ARGS(&unpack_)))) return fail("unpack pipeline");
        D3D12_DESCRIPTOR_HEAP_DESC hd{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, kHeap, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0};
        if (FAILED(device->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&heap_)))) return fail("descriptor heap");
        inc_ = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        if (FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_SHARED, IID_PPV_ARGS(&front_)))) return fail("shared fence");
        HANDLE h = nullptr;
        if (FAILED(device->CreateSharedHandle(front_, nullptr, GENERIC_ALL, nullptr, &h))) return fail("fence handle");
        frontHip_ = hipvk::importD3D12Fence(h, error);
        CloseHandle(h);
        if (!frontHip_) return fail("HIP fence import: " + error);
        if (FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&splitFence_)))) return fail("split fence");
        return true;
    }

    ID3D12Device* device() const { return device_; }
    ID3D12Fence* splitFence() const { return splitFence_; }

    bool create(Shared& s, uint32_t w, uint32_t h, Layout layout) {
        release(s);
        s.layout = layout; s.w = w; s.h = h;
        s.pitch = (w * layoutBytes(layout) + 255) & ~255u;
        D3D12_RESOURCE_DESC rd{};
        rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER; rd.Width = uint64_t(s.pitch) * h; rd.Height = 1; rd.DepthOrArraySize = 1; rd.MipLevels = 1;
        rd.SampleDesc.Count = 1; rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR; rd.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        D3D12_HEAP_PROPERTIES hp{D3D12_HEAP_TYPE_DEFAULT};
        if (FAILED(device_->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_SHARED, &rd, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&s.buffer))))
            return fail("shared buffer");
        const uint64_t bytes = device_->GetResourceAllocationInfo(0, 1, &rd).SizeInBytes;
        HANDLE handle = nullptr;
        if (FAILED(device_->CreateSharedHandle(s.buffer, nullptr, GENERIC_ALL, nullptr, &handle))) return fail("buffer handle");
        const bool ok = hipvk::importD3D12(handle, bytes, true, s.ext, error);
        CloseHandle(handle);
        if (!ok) return fail("HIP import: " + error);
        s.image = hipvk::wrapLinear(s.ext.pointer, s.pitch, layoutFormat(layout), w, h);
        return s.image ? true : fail("HIP image over the shared buffer");
    }

    void release(Shared& s) {
        if (s.image) vkDestroyImage(VK_NULL_HANDLE, s.image, nullptr);
        hipvk::release(s.ext);
        if (s.buffer) s.buffer->Release();
        s = Shared{};
    }

    // Records the conversion of `tex` (subresource 0; [ox, ox + w) x [oy, oy + h)) into `s` into `list`.
    void pack(ID3D12GraphicsCommandList* list, ID3D12Resource* tex, const Shared& s, uint32_t w, uint32_t h, uint32_t ox = 0, uint32_t oy = 0,
              float scale = 1.0f, uint32_t flags = 0) {
        bind(list);
        const D3D12_RESOURCE_DESC td = tex->GetDesc();
        D3D12_SHADER_RESOURCE_VIEW_DESC sd{};
        sd.Format = readable(td.Format);
        sd.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
        sd.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        sd.Texture2D.MipLevels = 1;
        const UINT slot = next();
        device_->CreateShaderResourceView(tex, &sd, cpu(slot));
        list->SetComputeRootDescriptorTable(1, gpu(slot));
        list->SetComputeRootUnorderedAccessView(3, s.buffer->GetGPUVirtualAddress());
        const Consts c{w, h, s.pitch, uint32_t(s.layout), ox, oy, scale, flags};
        list->SetComputeRoot32BitConstants(0, 8, &c, 0);
        list->SetPipelineState(pack_);
        list->Dispatch((w + 7) / 8, (h + 7) / 8, 1);
    }

    // pack() of a texture larger than `s` (display-size motion vectors for a render-size buffer): the nearest texel of the srcW x srcH
    // region at each pixel centre
    void packResampled(ID3D12GraphicsCommandList* list, ID3D12Resource* tex, const Shared& s, uint32_t w, uint32_t h, uint32_t srcW, uint32_t srcH) {
        pack(list, tex, s, w, h, srcW, srcH, 1.0f, 2);
    }

    // Records the write of `s` ([ox, ox + w) x [oy, oy + h)) into `tex` (a UAV-capable texture, subresource 0) into `list`.
    void unpack(ID3D12GraphicsCommandList* list, const Shared& s, ID3D12Resource* tex, uint32_t w, uint32_t h, uint32_t ox = 0, uint32_t oy = 0,
                bool opaque = false) {
        bind(list);
        const D3D12_RESOURCE_DESC td = tex->GetDesc();
        D3D12_UNORDERED_ACCESS_VIEW_DESC ud{};
        ud.Format = writable(td.Format);
        ud.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
        const UINT slot = next();
        device_->CreateUnorderedAccessView(tex, nullptr, &ud, cpu(slot));
        list->SetComputeRootDescriptorTable(2, gpu(slot));
        list->SetComputeRootShaderResourceView(4, s.buffer->GetGPUVirtualAddress());
        const Consts c{w, h, s.pitch, uint32_t(s.layout), ox, oy, 1.0f, opaque ? 1u : 0u};
        list->SetComputeRoot32BitConstants(0, 8, &c, 0);
        list->SetPipelineState(unpack_);
        list->Dispatch((w + 7) / 8, (h + 7) / 8, 1);
    }

    // Between the parts of a split list (d4r_split's beforeSecond, on the thread executing it): the queue signals the shared
    // fence, `cb` (recorded on hipvk) runs once the first part is done, and the queue waits for it before the second part.
    bool handoff(ID3D12CommandQueue* queue, VkCommandBuffer cb) {
        const uint64_t a = ++frontValue_, b = ++frontValue_;
        std::string why;
        if (FAILED(queue->Signal(front_, a)) || !hipvk::submit(cb, frontHip_, a, frontHip_, b, why) || FAILED(queue->Wait(front_, b))) {
            front_->Signal(b);   // never leave the queue waiting for a signal that will not come
            error = "handoff: " + (why.empty() ? std::string("a fence operation failed") : why);
            return false;
        }
        return true;
    }

private:
    static constexpr UINT kHeap = 4096;
    ID3D12Device* device_ = nullptr;
    ID3D12RootSignature* root_ = nullptr;
    ID3D12PipelineState *pack_ = nullptr, *unpack_ = nullptr;
    ID3D12DescriptorHeap* heap_ = nullptr;
    UINT inc_ = 0, next_ = 0;
    ID3D12Fence *front_ = nullptr, *splitFence_ = nullptr;
    void* frontHip_ = nullptr;
    uint64_t frontValue_ = 0;

    bool fail(const std::string& m) { error = m; return false; }
    UINT next() { return next_++ % kHeap; }   // a slot is reused after kHeap more views: many frames later
    D3D12_CPU_DESCRIPTOR_HANDLE cpu(UINT i) const { auto h = heap_->GetCPUDescriptorHandleForHeapStart(); h.ptr += SIZE_T(i) * inc_; return h; }
    D3D12_GPU_DESCRIPTOR_HANDLE gpu(UINT i) const { auto h = heap_->GetGPUDescriptorHandleForHeapStart(); h.ptr += UINT64(i) * inc_; return h; }
    void bind(ID3D12GraphicsCommandList* list) {
        list->SetDescriptorHeaps(1, &heap_);
        list->SetComputeRootSignature(root_);
    }
};

} // namespace dxi
