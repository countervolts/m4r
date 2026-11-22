// m4r: the part of Vulkan m4r uses, on AMD's HIP runtime (see hipvk.h).
// SPDX-License-Identifier: GPL-3.0-only
#include "hipvk.h"
#include "hip_dyn.h"
#include "hipvk_table.h"   // the project's Vulkan function table, which install() fills
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <deque>
#include <fstream>
#include <map>
#include <set>

using namespace hipdyn;

namespace hipvk {
namespace {

const Api* H = nullptr;
hipStream_t g_stream = nullptr;
std::string g_arch, g_name, g_kernelDir;
hipEvent_t g_epoch = nullptr, g_submitBegin = nullptr, g_submitEnd = nullptr;
bool g_timed = false;

constexpr uint32_t kRowAlign = 256;   // pitch of every image row (also D3D12's footprint row alignment)

bool ok(hipError_t e, const char* what, std::string* why = nullptr) {
    if (!e) return true;
    if (why) *why = std::string(what) + ": " + H->err(e);
    else std::fprintf(stderr, "hipvk: %s: %s\n", what, H->err(e));
    return false;
}

// ---- memory -----------------------------------------------------------------------------------------------------------------

struct Memory {
    uint8_t* dev = nullptr;
    size_t size = 0;
    uint8_t* shadow = nullptr;            // host-visible: the CPU's copy (MEM_WRITE_WATCH pages), uploaded where written
    hipExternalMemory_t ext = nullptr;    // imported (not freed with hipFree)
};
std::map<uintptr_t, Memory*> g_mems;      // by device address (address -> host pointer lookups)
std::set<Memory*> g_hostMems;

Memory* memoryAt(uint64_t addr) {
    auto it = g_mems.upper_bound(uintptr_t(addr));
    if (it == g_mems.begin()) return nullptr;
    --it;
    return addr < it->first + it->second->size ? it->second : nullptr;
}

// pinned staging for uploads of CPU-written pages; a region is reused once the submission that read it has finished
struct Staging {
    uint8_t* base = nullptr;
    size_t size = 0, head = 0;
    struct Use { hipEvent_t ev; size_t lo, hi; };
    std::deque<Use> uses;
    std::vector<hipEvent_t> freeEvents;
    hipEvent_t event() {
        if (!freeEvents.empty()) { auto e = freeEvents.back(); freeEvents.pop_back(); return e; }
        hipEvent_t e = nullptr;
        H->hipEventCreateWithFlags(&e, hipEventDisableTiming);
        return e;
    }
    void drain(size_t lo, size_t hi) {   // waits for every pending use overlapping [lo, hi)
        while (!uses.empty()) {
            bool overlap = false;
            for (auto& u : uses) overlap |= u.lo < hi && lo < u.hi;
            if (!overlap) return;
            H->hipEventSynchronize(uses.front().ev);
            freeEvents.push_back(uses.front().ev);
            uses.pop_front();
        }
    }
    uint8_t* alloc(size_t n, size_t& off) {
        n = (n + 255) & ~size_t(255);
        if (n > size) {
            H->hipStreamSynchronize(g_stream);
            for (auto& u : uses) freeEvents.push_back(u.ev);
            uses.clear();
            if (base) H->hipHostFree(base);
            size = std::max(n, size_t(16) << 20);
            base = nullptr;
            if (H->hipHostMalloc(reinterpret_cast<void**>(&base), size, 0)) { size = 0; return nullptr; }
            head = 0;
        }
        if (head + n > size) head = 0;
        drain(head, head + n);
        off = head;
        head += n;
        return base + off;
    }
    size_t mark = 0;   // start of this submission's uploads
    void close(size_t lo, size_t hi) {
        if (hi <= lo) return;
        hipEvent_t e = event();
        H->hipEventRecord(e, g_stream);
        uses.push_back({e, lo, hi});
    }
} g_staging;

// uploads every page the CPU wrote since the last call (stream-ordered before the submission's commands)
void uploadDirty() {
    std::vector<void*> pages;
    for (Memory* m : g_hostMems) {
        ULONG_PTR count = (m->size + 4095) / 4096;
        pages.resize(count);
        DWORD gran = 0;
        if (GetWriteWatch(WRITE_WATCH_FLAG_RESET, m->shadow, m->size, pages.data(), &count, &gran) != 0 || !count) continue;
        std::sort(pages.begin(), pages.begin() + count);
        for (ULONG_PTR i = 0; i < count;) {
            ULONG_PTR j = i + 1;
            while (j < count && static_cast<uint8_t*>(pages[j]) == static_cast<uint8_t*>(pages[j - 1]) + gran) ++j;
            const size_t lo = static_cast<uint8_t*>(pages[i]) - m->shadow;
            const size_t hi = std::min(m->size, size_t(static_cast<uint8_t*>(pages[j - 1]) - m->shadow) + gran);
            size_t off = 0;
            if (uint8_t* s = g_staging.alloc(hi - lo, off)) {
                std::memcpy(s, m->shadow + lo, hi - lo);
                H->hipMemcpyAsync(m->dev + lo, s, hi - lo, hipMemcpyHostToDevice, g_stream);
                g_staging.close(off, off + hi - lo);
            }
            i = j;
        }
    }
}

// ---- objects ----------------------------------------------------------------------------------------------------------------

struct Buffer { size_t size = 0; Memory* mem = nullptr; size_t offset = 0; uint8_t* dev() const { return mem ? mem->dev + offset : nullptr; } };

struct FormatInfo { uint32_t bytes; hipChannelFormatDesc desc; hipTextureReadMode read; bool bgra; };
bool formatInfo(VkFormat f, FormatInfo& fi) {
    auto d = [](int x, int y, int z, int w, hipChannelFormatKind k) { return hipChannelFormatDesc{x, y, z, w, k}; };
    const auto U = hipChannelFormatKindUnsigned, S = hipChannelFormatKindSigned, F = hipChannelFormatKindFloat;
    const auto E = hipReadModeElementType, N = hipReadModeNormalizedFloat;
    switch (f) {
    case VK_FORMAT_R8_UNORM: fi = {1, d(8, 0, 0, 0, U), N, false}; return true;
    case VK_FORMAT_R8_UINT: fi = {1, d(8, 0, 0, 0, U), E, false}; return true;
    case VK_FORMAT_R8G8_UNORM: fi = {2, d(8, 8, 0, 0, U), N, false}; return true;
    case VK_FORMAT_R8G8B8A8_UNORM: case VK_FORMAT_R8G8B8A8_SRGB: fi = {4, d(8, 8, 8, 8, U), N, false}; return true;
    case VK_FORMAT_B8G8R8A8_UNORM: case VK_FORMAT_B8G8R8A8_SRGB: fi = {4, d(8, 8, 8, 8, U), N, true}; return true;
    case VK_FORMAT_R8G8B8A8_SNORM: fi = {4, d(8, 8, 8, 8, S), N, false}; return true;
    case VK_FORMAT_R16_SFLOAT: fi = {2, d(16, 0, 0, 0, F), E, false}; return true;
    case VK_FORMAT_R16_UNORM: case VK_FORMAT_D16_UNORM: fi = {2, d(16, 0, 0, 0, U), N, false}; return true;
    case VK_FORMAT_R16_UINT: fi = {2, d(16, 0, 0, 0, U), E, false}; return true;
    case VK_FORMAT_R16G16_SFLOAT: fi = {4, d(16, 16, 0, 0, F), E, false}; return true;
    case VK_FORMAT_R16G16_UNORM: fi = {4, d(16, 16, 0, 0, U), N, false}; return true;
    case VK_FORMAT_R16G16_SNORM: fi = {4, d(16, 16, 0, 0, S), N, false}; return true;
    case VK_FORMAT_R16G16_UINT: fi = {4, d(16, 16, 0, 0, U), E, false}; return true;
    case VK_FORMAT_R16G16_SINT: fi = {4, d(16, 16, 0, 0, S), E, false}; return true;
    case VK_FORMAT_R16G16B16A16_SINT: fi = {8, d(16, 16, 16, 16, S), E, false}; return true;
    case VK_FORMAT_R16G16B16A16_SFLOAT: fi = {8, d(16, 16, 16, 16, F), E, false}; return true;
    case VK_FORMAT_R16G16B16A16_UNORM: fi = {8, d(16, 16, 16, 16, U), N, false}; return true;
    case VK_FORMAT_R16G16B16A16_UINT: fi = {8, d(16, 16, 16, 16, U), E, false}; return true;
    case VK_FORMAT_R32_SFLOAT: case VK_FORMAT_D32_SFLOAT: fi = {4, d(32, 0, 0, 0, F), E, false}; return true;
    case VK_FORMAT_R32_UINT: fi = {4, d(32, 0, 0, 0, U), E, false}; return true;
    case VK_FORMAT_R32G32_SFLOAT: fi = {8, d(32, 32, 0, 0, F), E, false}; return true;
    case VK_FORMAT_R32G32_UINT: fi = {8, d(32, 32, 0, 0, U), E, false}; return true;
    case VK_FORMAT_R32G32B32A32_SFLOAT: fi = {16, d(32, 32, 32, 32, F), E, false}; return true;
    case VK_FORMAT_R32G32B32A32_UINT: fi = {16, d(32, 32, 32, 32, U), E, false}; return true;
    default: return false;
    }
}

struct Image {
    VkFormat format = VK_FORMAT_UNDEFINED;
    FormatInfo fi{};
    uint32_t w = 0, h = 0, layers = 1, pitch = 0;
    uint8_t* dev = nullptr;
    hipTextureObject_t tex = nullptr;
    uint32_t srd[8] = {};
    bool external = false;
    size_t bytes() const { return size_t(pitch) * h * layers; }
};

bool makeSrd(Image& im, std::string* why) {
    hipResourceDesc r{};
    r.resType = hipResourceTypePitch2D;
    r.res.pitch2D.devPtr = im.dev;
    r.res.pitch2D.desc = im.fi.desc;
    r.res.pitch2D.width = im.w;
    r.res.pitch2D.height = size_t(im.h) * im.layers;   // array images are not used by m4r's plans: layers stack as rows
    r.res.pitch2D.pitchInBytes = im.pitch;
    hipTextureDesc td{};
    td.addressMode[0] = td.addressMode[1] = td.addressMode[2] = hipAddressModeClamp;
    td.filterMode = hipFilterModePoint;
    td.readMode = im.fi.read;
    td.sRGB = im.format == VK_FORMAT_R8G8B8A8_SRGB || im.format == VK_FORMAT_B8G8R8A8_SRGB;
    if (!ok(H->hipCreateTextureObject(&im.tex, &r, &td, nullptr), "hipCreateTextureObject", why)) return false;
    // a HIP texture object is a device pointer to the image resource (8 dwords) followed by the sampler (dword 12)
    if (!ok(H->hipMemcpy(im.srd, reinterpret_cast<void*>(im.tex), 32, hipMemcpyDeviceToHost), "texture descriptor", why)) return false;
    if (im.fi.bgra) {   // DST_SEL_X/Z (word 3 bits 0-2 and 6-8) swapped
        const uint32_t x = im.srd[3] & 7, z = (im.srd[3] >> 6) & 7;
        im.srd[3] = (im.srd[3] & ~0x1c7u) | z | (x << 6);
    }
    return true;
}

struct View { Image* image = nullptr; uint32_t srd[8] = {}; };
struct Sampler { uint32_t desc[4] = {}; };

struct Pipeline {
    hipModule_t module = nullptr;
    hipFunction_t fn = nullptr;
    KernelAbi abi = KernelAbi::Kernarg;
    uint32_t tg[3] = {1, 1, 1}, kernarg = 0, lds = 0;
};

struct Cmd {
    enum Kind { Dispatch, Barrier, CopyBI, CopyIB, CopyII, Clear, Blit, Timestamp, ResetQuery, Fill, Update, CopyBB } kind;
    Pipeline* pipe = nullptr;
    uint64_t push = 0;
    uint32_t n[3] = {};
    Buffer *buf = nullptr, *buf2 = nullptr;   // Fill / Update / CopyBB: buf is the destination, buf2 the copy's source
    uint64_t offset = 0, srcOffset = 0, size = 0;
    uint32_t value = 0;
    std::vector<uint8_t> bytes;                // Update's data
    Image *src = nullptr, *dst = nullptr;
    VkBufferImageCopy bic{};
    VkImageCopy ic{};
    uint8_t texel[16] = {};
    struct QueryPool* pool = nullptr;
    uint32_t query = 0, count = 0;
};

struct CommandBuffer { std::vector<Cmd> cmds; Pipeline* pipe = nullptr; uint64_t push = 0; };
struct Fence { hipEvent_t ev = nullptr; bool pending = false; };
struct QueryPool { std::vector<hipEvent_t> ev; std::vector<char> written; };

// ---- the format-converting blit kernel (hipvk_blit.cl) ----------------------------------------------------------------------

hipModule_t g_blitModule = nullptr;
hipFunction_t g_blitFn = nullptr;
bool g_blitTried = false;
uint8_t* g_blitRecords = nullptr;   // pinned ring of image records for blits (64 bytes: source descriptor, destination descriptor)
size_t g_blitHead = 0;
constexpr size_t kBlitRing = 1024;

bool loadBlit() {
    if (g_blitTried) return g_blitFn != nullptr;
    g_blitTried = true;
    std::ifstream f(g_kernelDir + "/hipvk_blit.hsaco", std::ios::binary);
    std::vector<char> co((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    if (co.empty() || H->hipModuleLoadData(&g_blitModule, co.data()) || H->hipModuleGetFunction(&g_blitFn, g_blitModule, "hipvk_blit")) {
        std::fprintf(stderr, "hipvk: %s/hipvk_blit.hsaco missing or unusable: format-converting blits are skipped\n", g_kernelDir.c_str());
        g_blitFn = nullptr;
        return false;
    }
    return !H->hipHostMalloc(reinterpret_cast<void**>(&g_blitRecords), kBlitRing * 64, 0);
}

// ---- replay -----------------------------------------------------------------------------------------------------------------

uint8_t* imageAt(Image* im, int32_t x, int32_t y, uint32_t layer) {
    return im->dev + size_t(layer) * im->pitch * im->h + size_t(y) * im->pitch + size_t(x) * im->fi.bytes;
}

bool replay(CommandBuffer* cb) {
    bool ordered = true;   // the next dispatch waits for everything before it (a barrier or a non-dispatch command came first)
    for (Cmd& c : cb->cmds) {
        switch (c.kind) {
        case Cmd::Barrier: ordered = true; break;
        case Cmd::Dispatch: {
            Pipeline* p = c.pipe;
            if (!p) break;
            std::vector<uint8_t> args;
            if (p->abi == KernelAbi::ArgPointer) {
                args.resize(8);
                std::memcpy(args.data(), &c.push, 8);
            } else {
                Memory* m = memoryAt(c.push);
                if (!m || !m->shadow) { std::fprintf(stderr, "hipvk: argument block outside host-visible memory\n"); return false; }
                args.assign(m->shadow + (c.push - uintptr_t(m->dev)), m->shadow + (c.push - uintptr_t(m->dev)) + p->kernarg);
            }
            size_t sz = args.size();
            void* cfg[] = {reinterpret_cast<void*>(1), args.data(), reinterpret_cast<void*>(2), &sz, reinterpret_cast<void*>(3)};   // HIP_LAUNCH_PARAM_*
            hipError_t e = H->hipExtModuleLaunchKernel(p->fn, c.n[0] * p->tg[0], c.n[1] * p->tg[1], c.n[2] * p->tg[2], p->tg[0], p->tg[1], p->tg[2],
                                                       p->lds, g_stream, nullptr, cfg, nullptr, nullptr, ordered ? 0 : hipExtAnyOrderLaunch);
            if (!ok(e, "launch")) return false;
            ordered = false;
            break;
        }
        case Cmd::CopyBI: case Cmd::CopyIB: {
            const VkBufferImageCopy& r = c.bic;
            Image* im = c.kind == Cmd::CopyBI ? c.dst : c.src;
            const size_t row = size_t(r.imageExtent.width) * im->fi.bytes;
            const size_t bpitch = size_t(r.bufferRowLength ? r.bufferRowLength : r.imageExtent.width) * im->fi.bytes;
            const size_t bimg = bpitch * (r.bufferImageHeight ? r.bufferImageHeight : r.imageExtent.height);
            for (uint32_t l = 0; l < r.imageSubresource.layerCount; ++l) {
                uint8_t* ip = imageAt(im, r.imageOffset.x, r.imageOffset.y, r.imageSubresource.baseArrayLayer + l);
                uint8_t* bp = c.buf->dev() + r.bufferOffset + l * bimg;
                hipError_t e = c.kind == Cmd::CopyBI
                    ? H->hipMemcpy2DAsync(ip, im->pitch, bp, bpitch, row, r.imageExtent.height, hipMemcpyDeviceToDevice, g_stream)
                    : H->hipMemcpy2DAsync(bp, bpitch, ip, im->pitch, row, r.imageExtent.height, hipMemcpyDeviceToDevice, g_stream);
                if (!ok(e, "buffer/image copy")) return false;
                if (c.kind == Cmd::CopyIB && c.buf->mem->shadow) {   // a readback: the CPU sees it after the fence
                    uint8_t* hp = c.buf->mem->shadow + c.buf->offset + r.bufferOffset + l * bimg;
                    H->hipMemcpy2DAsync(hp, bpitch, bp, bpitch, row, r.imageExtent.height, hipMemcpyDeviceToHost, g_stream);
                }
            }
            ordered = true;
            break;
        }
        case Cmd::CopyII: {
            const VkImageCopy& r = c.ic;
            for (uint32_t l = 0; l < r.srcSubresource.layerCount; ++l) {
                hipError_t e = H->hipMemcpy2DAsync(imageAt(c.dst, r.dstOffset.x, r.dstOffset.y, r.dstSubresource.baseArrayLayer + l), c.dst->pitch,
                                                   imageAt(c.src, r.srcOffset.x, r.srcOffset.y, r.srcSubresource.baseArrayLayer + l), c.src->pitch,
                                                   size_t(r.extent.width) * c.src->fi.bytes, r.extent.height, hipMemcpyDeviceToDevice, g_stream);
                if (!ok(e, "image copy")) return false;
            }
            ordered = true;
            break;
        }
        case Cmd::Clear: {
            Image* im = c.dst;
            const size_t row = size_t(im->w) * im->fi.bytes;
            size_t off = 0;
            uint8_t* s = g_staging.alloc(row, off);
            if (!s) return false;
            for (uint32_t x = 0; x < im->w; ++x) std::memcpy(s + x * im->fi.bytes, c.texel, im->fi.bytes);
            for (uint32_t y = 0; y < im->h * im->layers; ++y)
                H->hipMemcpyAsync(im->dev + size_t(y) * im->pitch, s, row, hipMemcpyHostToDevice, g_stream);
            g_staging.close(off, off + row);
            ordered = true;
            break;
        }
        case Cmd::Blit: {
            Image *s = c.src, *d = c.dst;
            const uint32_t w = std::min(s->w, d->w), h = std::min(s->h, d->h);
            if (s->format == d->format || (s->fi.bytes == d->fi.bytes && !loadBlit())) {
                H->hipMemcpy2DAsync(d->dev, d->pitch, s->dev, s->pitch, size_t(w) * s->fi.bytes, h, hipMemcpyDeviceToDevice, g_stream);
            } else if (loadBlit()) {
                uint8_t* rec = g_blitRecords + (g_blitHead++ % kBlitRing) * 64;
                std::memcpy(rec, s->srd, 32);
                std::memcpy(rec + 32, d->srd, 32);
                struct { void* rec; uint32_t w, h; } a{rec, w, h};
                size_t sz = sizeof a;
                void* cfg[] = {reinterpret_cast<void*>(1), &a, reinterpret_cast<void*>(2), &sz, reinterpret_cast<void*>(3)};
                if (!ok(H->hipExtModuleLaunchKernel(g_blitFn, (w + 7) / 8 * 8, (h + 7) / 8 * 8, 1, 8, 8, 1, 0, g_stream, nullptr, cfg, nullptr, nullptr, 0),
                        "blit")) return false;
            }
            ordered = true;
            break;
        }
        case Cmd::Fill: {
            if (!ok(H->hipMemsetD32Async(c.buf->dev() + c.offset, int(c.value), size_t(c.size / 4), g_stream), "fill")) return false;
            ordered = true;
            break;
        }
        case Cmd::Update: {
            size_t off = 0;
            uint8_t* s = g_staging.alloc(c.bytes.size(), off);
            if (!s) return false;
            std::memcpy(s, c.bytes.data(), c.bytes.size());
            if (!ok(H->hipMemcpyAsync(c.buf->dev() + c.offset, s, c.bytes.size(), hipMemcpyHostToDevice, g_stream), "update")) return false;
            g_staging.close(off, off + c.bytes.size());
            ordered = true;
            break;
        }
        case Cmd::CopyBB: {
            uint8_t* d = c.buf->dev() + c.offset;
            if (!ok(H->hipMemcpyAsync(d, c.buf2->dev() + c.srcOffset, size_t(c.size), hipMemcpyDeviceToDevice, g_stream), "buffer copy")) return false;
            if (c.buf->mem->shadow)   // a readback: the CPU sees it after the fence
                H->hipMemcpyAsync(c.buf->mem->shadow + c.buf->offset + c.offset, d, size_t(c.size), hipMemcpyDeviceToHost, g_stream);
            ordered = true;
            break;
        }
        case Cmd::Timestamp:
            if (c.query < c.pool->ev.size()) { H->hipEventRecord(c.pool->ev[c.query], g_stream); c.pool->written[c.query] = 1; }
            break;
        case Cmd::ResetQuery:
            for (uint32_t i = c.query; i < c.query + c.count && i < c.pool->written.size(); ++i) c.pool->written[i] = 0;
            break;
        }
    }
    return true;
}

bool submitOne(CommandBuffer* cb, hipExternalSemaphore_t wait, uint64_t waitValue, hipExternalSemaphore_t signal, uint64_t signalValue,
               hipEvent_t fence, std::string* why) {
    if (wait) {
        hipExternalSemaphoreWaitParams p{};
        p.params.fence.value = waitValue;
        if (!ok(H->hipWaitExternalSemaphoresAsync(&wait, &p, 1, g_stream), "fence wait", why)) return false;
    }
    if (g_timed) H->hipEventRecord(g_submitBegin, g_stream);
    uploadDirty();
    if (!replay(cb)) { if (why) *why = "replay failed"; return false; }
    if (g_timed) H->hipEventRecord(g_submitEnd, g_stream);
    if (signal) {
        hipExternalSemaphoreSignalParams p{};
        p.params.fence.value = signalValue;
        if (!ok(H->hipSignalExternalSemaphoresAsync(&signal, &p, 1, g_stream), "fence signal", why)) return false;
    }
    if (fence) H->hipEventRecord(fence, g_stream);
    return true;
}

// ---- the Vulkan entry points ------------------------------------------------------------------------------------------------

template <class T> T* obj(uint64_t h) { return reinterpret_cast<T*>(uintptr_t(h)); }
template <class T> T* obj(void* h) { return reinterpret_cast<T*>(h); }
template <class H_, class T> H_ handle(T* p) { return (H_)(uintptr_t)(p); }

struct Dummy { int x; };
Dummy g_instance, g_physical, g_device, g_queue;

VKAPI_ATTR VkResult VKAPI_CALL CreateInstance(const VkInstanceCreateInfo*, const VkAllocationCallbacks*, VkInstance* out) {
    *out = reinterpret_cast<VkInstance>(&g_instance); return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL DestroyInstance(VkInstance, const VkAllocationCallbacks*) {}
VKAPI_ATTR VkResult VKAPI_CALL EnumeratePhysicalDevices(VkInstance, uint32_t* n, VkPhysicalDevice* out) {
    if (out && *n) out[0] = reinterpret_cast<VkPhysicalDevice>(&g_physical);
    *n = 1;
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL GetPhysicalDeviceFormatProperties(VkPhysicalDevice, VkFormat f, VkFormatProperties* p) {
    *p = {};
    FormatInfo fi;
    if (formatInfo(f, fi))
        p->optimalTilingFeatures = p->linearTilingFeatures = VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT | VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT |
            VK_FORMAT_FEATURE_TRANSFER_SRC_BIT | VK_FORMAT_FEATURE_TRANSFER_DST_BIT | VK_FORMAT_FEATURE_BLIT_SRC_BIT | VK_FORMAT_FEATURE_BLIT_DST_BIT;
}
VKAPI_ATTR void VKAPI_CALL GetPhysicalDeviceMemoryProperties(VkPhysicalDevice, VkPhysicalDeviceMemoryProperties* p) {
    *p = {};
    p->memoryHeapCount = 1;
    p->memoryHeaps[0] = {VkDeviceSize(8) << 30, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT};
    p->memoryTypeCount = 2;
    p->memoryTypes[0] = {VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0};
    p->memoryTypes[1] = {VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 0};
}
void fillProps(VkPhysicalDeviceProperties& p) {
    p = {};
    p.apiVersion = VK_API_VERSION_1_3;
    p.vendorID = 0x1002;
    p.deviceType = VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU;
    std::snprintf(p.deviceName, sizeof p.deviceName, "%s (HIP %s)", g_name.c_str(), g_arch.c_str());
    p.limits.timestampPeriod = 1.0f;   // timestamps are nanoseconds (hipvk query pools)
    p.limits.timestampComputeAndGraphics = VK_TRUE;
}
VKAPI_ATTR void VKAPI_CALL GetPhysicalDeviceProperties(VkPhysicalDevice, VkPhysicalDeviceProperties* p) { fillProps(*p); }
VKAPI_ATTR void VKAPI_CALL GetPhysicalDeviceProperties2(VkPhysicalDevice, VkPhysicalDeviceProperties2* p) {
    fillProps(p->properties);
    for (auto* s = static_cast<VkBaseOutStructure*>(p->pNext); s; s = s->pNext)
        if (s->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_BUFFER_PROPERTIES_EXT) {
            auto* d = reinterpret_cast<VkPhysicalDeviceDescriptorBufferPropertiesEXT*>(s);
            d->sampledImageDescriptorSize = 32; d->storageImageDescriptorSize = 32; d->samplerDescriptorSize = 16;
        }
}
VKAPI_ATTR void VKAPI_CALL GetPhysicalDeviceQueueFamilyProperties(VkPhysicalDevice, uint32_t* n, VkQueueFamilyProperties* out) {
    if (out && *n) out[0] = {VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT, 1, 64, {1, 1, 1}};
    *n = 1;
}
VKAPI_ATTR VkResult VKAPI_CALL CreateDevice(VkPhysicalDevice, const VkDeviceCreateInfo*, const VkAllocationCallbacks*, VkDevice* out) {
    *out = reinterpret_cast<VkDevice>(&g_device); return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL DestroyDevice(VkDevice, const VkAllocationCallbacks*) {}
VKAPI_ATTR void VKAPI_CALL GetDeviceQueue(VkDevice, uint32_t, uint32_t, VkQueue* q) { *q = reinterpret_cast<VkQueue>(&g_queue); }
VKAPI_ATTR VkResult VKAPI_CALL DeviceWaitIdle(VkDevice) { return H->hipStreamSynchronize(g_stream) ? VK_ERROR_DEVICE_LOST : VK_SUCCESS; }

VKAPI_ATTR VkResult VKAPI_CALL AllocateMemory(VkDevice, const VkMemoryAllocateInfo* ai, const VkAllocationCallbacks*, VkDeviceMemory* out) {
    auto* m = new Memory;
    m->size = size_t(ai->allocationSize);
    if (H->hipMalloc(reinterpret_cast<void**>(&m->dev), std::max<size_t>(m->size, 256))) { delete m; return VK_ERROR_OUT_OF_DEVICE_MEMORY; }
    if (ai->memoryTypeIndex == 1) {
        m->shadow = static_cast<uint8_t*>(VirtualAlloc(nullptr, m->size, MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH, PAGE_READWRITE));
        if (!m->shadow) { H->hipFree(m->dev); delete m; return VK_ERROR_OUT_OF_HOST_MEMORY; }
        H->hipMemsetAsync(m->dev, 0, m->size, g_stream);   // fresh VirtualAlloc pages are zero: so is the device copy
        ULONG_PTR count = 0; DWORD gran = 0;
        GetWriteWatch(WRITE_WATCH_FLAG_RESET, m->shadow, m->size, nullptr, &count, &gran);
        g_hostMems.insert(m);
    }
    g_mems[uintptr_t(m->dev)] = m;
    *out = handle<VkDeviceMemory>(m);
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL FreeMemory(VkDevice, VkDeviceMemory h, const VkAllocationCallbacks*) {
    Memory* m = obj<Memory>(uint64_t(h));
    if (!m) return;
    H->hipStreamSynchronize(g_stream);
    g_mems.erase(uintptr_t(m->dev));
    g_hostMems.erase(m);
    if (m->shadow) VirtualFree(m->shadow, 0, MEM_RELEASE);
    if (m->ext) H->hipDestroyExternalMemory(m->ext);
    else H->hipFree(m->dev);
    delete m;
}
VKAPI_ATTR VkResult VKAPI_CALL MapMemory(VkDevice, VkDeviceMemory h, VkDeviceSize off, VkDeviceSize, VkMemoryMapFlags, void** out) {
    Memory* m = obj<Memory>(uint64_t(h));
    if (!m->shadow) return VK_ERROR_MEMORY_MAP_FAILED;
    *out = m->shadow + off;
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL UnmapMemory(VkDevice, VkDeviceMemory) {}   // the host copy stays: writes are picked up at the next submission

VKAPI_ATTR VkResult VKAPI_CALL CreateBuffer(VkDevice, const VkBufferCreateInfo* ci, const VkAllocationCallbacks*, VkBuffer* out) {
    auto* b = new Buffer; b->size = size_t(ci->size); *out = handle<VkBuffer>(b); return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL DestroyBuffer(VkDevice, VkBuffer h, const VkAllocationCallbacks*) { delete obj<Buffer>(uint64_t(h)); }
VKAPI_ATTR void VKAPI_CALL GetBufferMemoryRequirements(VkDevice, VkBuffer h, VkMemoryRequirements* r) {
    r->size = (obj<Buffer>(uint64_t(h))->size + 255) & ~VkDeviceSize(255); r->alignment = 256; r->memoryTypeBits = 3;
}
VKAPI_ATTR VkResult VKAPI_CALL BindBufferMemory(VkDevice, VkBuffer h, VkDeviceMemory m, VkDeviceSize off) {
    Buffer* b = obj<Buffer>(uint64_t(h)); b->mem = obj<Memory>(uint64_t(m)); b->offset = size_t(off); return VK_SUCCESS;
}
VKAPI_ATTR VkDeviceAddress VKAPI_CALL GetBufferDeviceAddress(VkDevice, const VkBufferDeviceAddressInfo* ai) {
    return VkDeviceAddress(uintptr_t(obj<Buffer>(uint64_t(ai->buffer))->dev()));
}

VKAPI_ATTR VkResult VKAPI_CALL CreateImage(VkDevice, const VkImageCreateInfo* ci, const VkAllocationCallbacks*, VkImage* out) {
    auto* im = new Image;
    if (!formatInfo(ci->format, im->fi)) { delete im; return VK_ERROR_FORMAT_NOT_SUPPORTED; }
    im->format = ci->format; im->w = ci->extent.width; im->h = ci->extent.height; im->layers = std::max(1u, ci->arrayLayers);
    im->pitch = (im->w * im->fi.bytes + kRowAlign - 1) / kRowAlign * kRowAlign;
    *out = handle<VkImage>(im);
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL DestroyImage(VkDevice, VkImage h, const VkAllocationCallbacks*) {
    Image* im = obj<Image>(uint64_t(h));
    if (!im) return;
    if (im->tex) H->hipDestroyTextureObject(im->tex);
    delete im;
}
VKAPI_ATTR void VKAPI_CALL GetImageMemoryRequirements(VkDevice, VkImage h, VkMemoryRequirements* r) {
    r->size = obj<Image>(uint64_t(h))->bytes(); r->alignment = 256; r->memoryTypeBits = 1;
}
VKAPI_ATTR VkResult VKAPI_CALL BindImageMemory(VkDevice, VkImage h, VkDeviceMemory m, VkDeviceSize off) {
    Image* im = obj<Image>(uint64_t(h));
    im->dev = obj<Memory>(uint64_t(m))->dev + off;
    return makeSrd(*im, nullptr) ? VK_SUCCESS : VK_ERROR_INITIALIZATION_FAILED;
}
VKAPI_ATTR VkResult VKAPI_CALL CreateImageView(VkDevice, const VkImageViewCreateInfo* ci, const VkAllocationCallbacks*, VkImageView* out) {
    auto* v = new View;
    v->image = obj<Image>(uint64_t(ci->image));
    std::memcpy(v->srd, v->image->srd, 32);
    if (ci->components.r == VK_COMPONENT_SWIZZLE_A) {   // alpha in every channel (Context::wrap's alphaToAll)
        const uint32_t a = (v->srd[3] >> 9) & 7;
        v->srd[3] = (v->srd[3] & ~0xfffu) | a | (a << 3) | (a << 6) | (a << 9);
    }
    *out = handle<VkImageView>(v);
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL DestroyImageView(VkDevice, VkImageView h, const VkAllocationCallbacks*) { delete obj<View>(uint64_t(h)); }

VKAPI_ATTR VkResult VKAPI_CALL CreateSampler(VkDevice, const VkSamplerCreateInfo* ci, const VkAllocationCallbacks*, VkSampler* out) {
    // the descriptor HIP builds for the same modes, read from a texture object over a 1x1 float image
    static void* backing = nullptr;
    if (!backing && H->hipMalloc(&backing, 256)) return VK_ERROR_OUT_OF_DEVICE_MEMORY;
    auto mode = [](VkSamplerAddressMode m) {
        switch (m) {
        case VK_SAMPLER_ADDRESS_MODE_REPEAT: return hipAddressModeWrap;
        case VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT: return hipAddressModeMirror;
        case VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER: return hipAddressModeBorder;
        default: return hipAddressModeClamp;
        }
    };
    hipResourceDesc r{};
    r.resType = hipResourceTypePitch2D;
    r.res.pitch2D.devPtr = backing;
    r.res.pitch2D.desc = {32, 0, 0, 0, hipChannelFormatKindFloat};
    r.res.pitch2D.width = 1; r.res.pitch2D.height = 1; r.res.pitch2D.pitchInBytes = 256;
    hipTextureDesc td{};
    td.addressMode[0] = mode(ci->addressModeU); td.addressMode[1] = mode(ci->addressModeV); td.addressMode[2] = mode(ci->addressModeW);
    td.filterMode = ci->magFilter == VK_FILTER_LINEAR ? hipFilterModeLinear : hipFilterModePoint;
    td.mipmapFilterMode = ci->mipmapMode == VK_SAMPLER_MIPMAP_MODE_LINEAR ? hipFilterModeLinear : hipFilterModePoint;
    td.readMode = hipReadModeElementType;
    td.normalizedCoords = ci->unnormalizedCoordinates ? 0 : 1;
    td.maxMipmapLevelClamp = ci->maxLod;
    hipTextureObject_t t = nullptr;
    if (H->hipCreateTextureObject(&t, &r, &td, nullptr)) return VK_ERROR_INITIALIZATION_FAILED;
    uint32_t words[16];
    hipError_t e = H->hipMemcpy(words, reinterpret_cast<void*>(t), 64, hipMemcpyDeviceToHost);
    H->hipDestroyTextureObject(t);
    if (e) return VK_ERROR_INITIALIZATION_FAILED;
    auto* s = new Sampler;
    std::memcpy(s->desc, words + 12, 16);
    *out = handle<VkSampler>(s);
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL DestroySampler(VkDevice, VkSampler h, const VkAllocationCallbacks*) { delete obj<Sampler>(uint64_t(h)); }

VKAPI_ATTR void VKAPI_CALL GetDescriptorEXT(VkDevice, const VkDescriptorGetInfoEXT* gi, size_t size, void* out) {
    std::memset(out, 0, size);
    if (gi->type == VK_DESCRIPTOR_TYPE_SAMPLER) { std::memcpy(out, obj<Sampler>(uint64_t(*gi->data.pSampler))->desc, std::min<size_t>(size, 16)); return; }
    const VkDescriptorImageInfo* ii = gi->type == VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE ? gi->data.pSampledImage : gi->data.pStorageImage;
    if (ii && ii->imageView) std::memcpy(out, obj<View>(uint64_t(ii->imageView))->srd, std::min<size_t>(size, 32));
}

VKAPI_ATTR VkResult VKAPI_CALL CreatePipelineLayout(VkDevice, const VkPipelineLayoutCreateInfo*, const VkAllocationCallbacks*, VkPipelineLayout* out) {
    static Dummy layout; *out = handle<VkPipelineLayout>(&layout); return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL DestroyPipelineLayout(VkDevice, VkPipelineLayout, const VkAllocationCallbacks*) {}
VKAPI_ATTR VkResult VKAPI_CALL CreateShaderModule(VkDevice, const VkShaderModuleCreateInfo*, const VkAllocationCallbacks*, VkShaderModule*) {
    return VK_ERROR_FEATURE_NOT_PRESENT;   // SPIR-V cannot run here: m4r's GLSL passes load their OpenCL ports (Runtime::loadGlsl)
}
VKAPI_ATTR void VKAPI_CALL DestroyShaderModule(VkDevice, VkShaderModule, const VkAllocationCallbacks*) {}
VKAPI_ATTR VkResult VKAPI_CALL CreateComputePipelines(VkDevice, VkPipelineCache, uint32_t, const VkComputePipelineCreateInfo*, const VkAllocationCallbacks*, VkPipeline*) {
    return VK_ERROR_FEATURE_NOT_PRESENT;
}
VKAPI_ATTR void VKAPI_CALL DestroyPipeline(VkDevice, VkPipeline h, const VkAllocationCallbacks*) {
    Pipeline* p = obj<Pipeline>(uint64_t(h));
    if (!p) return;
    H->hipStreamSynchronize(g_stream);
    if (p->module) H->hipModuleUnload(p->module);
    delete p;
}

VKAPI_ATTR VkResult VKAPI_CALL CreateCommandPool(VkDevice, const VkCommandPoolCreateInfo*, const VkAllocationCallbacks*, VkCommandPool* out) {
    static Dummy pool; *out = handle<VkCommandPool>(&pool); return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL DestroyCommandPool(VkDevice, VkCommandPool, const VkAllocationCallbacks*) {}
VKAPI_ATTR VkResult VKAPI_CALL AllocateCommandBuffers(VkDevice, const VkCommandBufferAllocateInfo* ai, VkCommandBuffer* out) {
    for (uint32_t i = 0; i < ai->commandBufferCount; ++i) out[i] = reinterpret_cast<VkCommandBuffer>(new CommandBuffer);
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL FreeCommandBuffers(VkDevice, VkCommandPool, uint32_t n, const VkCommandBuffer* cbs) {
    for (uint32_t i = 0; i < n; ++i) delete reinterpret_cast<CommandBuffer*>(cbs[i]);
}
CommandBuffer* cbOf(VkCommandBuffer cb) { return reinterpret_cast<CommandBuffer*>(cb); }
VKAPI_ATTR VkResult VKAPI_CALL BeginCommandBuffer(VkCommandBuffer cb, const VkCommandBufferBeginInfo*) {
    cbOf(cb)->cmds.clear(); cbOf(cb)->pipe = nullptr; return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL EndCommandBuffer(VkCommandBuffer) { return VK_SUCCESS; }
VKAPI_ATTR void VKAPI_CALL CmdBindPipeline(VkCommandBuffer cb, VkPipelineBindPoint, VkPipeline p) { cbOf(cb)->pipe = obj<Pipeline>(uint64_t(p)); }
VKAPI_ATTR void VKAPI_CALL CmdPushConstants(VkCommandBuffer cb, VkPipelineLayout, VkShaderStageFlags, uint32_t off, uint32_t size, const void* v) {
    if (off == 0 && size >= 8) std::memcpy(&cbOf(cb)->push, v, 8);
}
VKAPI_ATTR void VKAPI_CALL CmdDispatch(VkCommandBuffer cb, uint32_t x, uint32_t y, uint32_t z) {
    Cmd c{Cmd::Dispatch};
    c.pipe = cbOf(cb)->pipe; c.push = cbOf(cb)->push; c.n[0] = x; c.n[1] = y; c.n[2] = z;
    cbOf(cb)->cmds.push_back(c);
}
VKAPI_ATTR void VKAPI_CALL CmdDispatchBase(VkCommandBuffer cb, uint32_t bx, uint32_t by, uint32_t bz, uint32_t x, uint32_t y, uint32_t z) {
    // HIP launches have no workgroup base: Encoder dispatches the whole grid with the first band instead (kernel.cpp)
    if (!bx && !by && !bz) CmdDispatch(cb, x, y, z);
    else std::fprintf(stderr, "hipvk: vkCmdDispatchBase with a base is not supported\n");
}
VKAPI_ATTR void VKAPI_CALL CmdPipelineBarrier(VkCommandBuffer cb, VkPipelineStageFlags, VkPipelineStageFlags, VkDependencyFlags, uint32_t,
                                              const VkMemoryBarrier*, uint32_t, const VkBufferMemoryBarrier*, uint32_t, const VkImageMemoryBarrier*) {
    if (cbOf(cb)->cmds.empty() || cbOf(cb)->cmds.back().kind != Cmd::Barrier) cbOf(cb)->cmds.push_back(Cmd{Cmd::Barrier});
}
VKAPI_ATTR void VKAPI_CALL CmdCopyBufferToImage(VkCommandBuffer cb, VkBuffer b, VkImage im, VkImageLayout, uint32_t n, const VkBufferImageCopy* r) {
    for (uint32_t i = 0; i < n; ++i) {
        Cmd c{Cmd::CopyBI}; c.buf = obj<Buffer>(uint64_t(b)); c.dst = obj<Image>(uint64_t(im)); c.bic = r[i];
        cbOf(cb)->cmds.push_back(c);
    }
}
VKAPI_ATTR void VKAPI_CALL CmdCopyImageToBuffer(VkCommandBuffer cb, VkImage im, VkImageLayout, VkBuffer b, uint32_t n, const VkBufferImageCopy* r) {
    for (uint32_t i = 0; i < n; ++i) {
        Cmd c{Cmd::CopyIB}; c.buf = obj<Buffer>(uint64_t(b)); c.src = obj<Image>(uint64_t(im)); c.bic = r[i];
        cbOf(cb)->cmds.push_back(c);
    }
}
VKAPI_ATTR void VKAPI_CALL CmdCopyImage(VkCommandBuffer cb, VkImage s, VkImageLayout, VkImage d, VkImageLayout, uint32_t n, const VkImageCopy* r) {
    for (uint32_t i = 0; i < n; ++i) {
        Cmd c{Cmd::CopyII}; c.src = obj<Image>(uint64_t(s)); c.dst = obj<Image>(uint64_t(d)); c.ic = r[i];
        cbOf(cb)->cmds.push_back(c);
    }
}
VKAPI_ATTR void VKAPI_CALL CmdBlitImage(VkCommandBuffer cb, VkImage s, VkImageLayout, VkImage d, VkImageLayout, uint32_t, const VkImageBlit*, VkFilter) {
    Cmd c{Cmd::Blit}; c.src = obj<Image>(uint64_t(s)); c.dst = obj<Image>(uint64_t(d));   // m4r blits whole images of equal size
    cbOf(cb)->cmds.push_back(c);
}
VKAPI_ATTR void VKAPI_CALL CmdFillBuffer(VkCommandBuffer cb, VkBuffer b, VkDeviceSize off, VkDeviceSize size, uint32_t value) {
    Cmd c{Cmd::Fill}; c.buf = obj<Buffer>(uint64_t(b)); c.offset = off; c.value = value;
    c.size = size == VK_WHOLE_SIZE ? (c.buf->size - off) & ~VkDeviceSize(3) : size;
    cbOf(cb)->cmds.push_back(c);
}
VKAPI_ATTR void VKAPI_CALL CmdUpdateBuffer(VkCommandBuffer cb, VkBuffer b, VkDeviceSize off, VkDeviceSize size, const void* data) {
    Cmd c{Cmd::Update}; c.buf = obj<Buffer>(uint64_t(b)); c.offset = off;
    c.bytes.assign(static_cast<const uint8_t*>(data), static_cast<const uint8_t*>(data) + size);
    cbOf(cb)->cmds.push_back(std::move(c));
}
VKAPI_ATTR void VKAPI_CALL CmdCopyBuffer(VkCommandBuffer cb, VkBuffer s, VkBuffer d, uint32_t n, const VkBufferCopy* r) {
    for (uint32_t i = 0; i < n; ++i) {
        Cmd c{Cmd::CopyBB}; c.buf = obj<Buffer>(uint64_t(d)); c.buf2 = obj<Buffer>(uint64_t(s));
        c.offset = r[i].dstOffset; c.srcOffset = r[i].srcOffset; c.size = r[i].size;
        cbOf(cb)->cmds.push_back(c);
    }
}

uint16_t toHalf(float f) {
    uint32_t x; std::memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000, e = (x >> 23) & 0xff;
    uint32_t m = x & 0x7fffff;
    if (e == 0xff) return uint16_t(sign | 0x7c00 | (m ? 0x200 : 0));
    int ex = int(e) - 127 + 15;
    if (ex >= 31) return uint16_t(sign | 0x7c00);
    if (ex <= 0) {
        if (ex < -10) return uint16_t(sign);
        m |= 0x800000;
        const int shift = 14 - ex;
        uint32_t hm = m >> shift;
        const uint32_t rem = m & ((1u << shift) - 1), half = 1u << (shift - 1);
        if (rem > half || (rem == half && (hm & 1))) ++hm;
        return uint16_t(sign | hm);
    }
    uint32_t hm = m >> 13;
    const uint32_t rem = m & 0x1fff;
    uint32_t h = sign | (uint32_t(ex) << 10) | hm;
    if (rem > 0x1000 || (rem == 0x1000 && (hm & 1))) ++h;
    return uint16_t(h);
}

VKAPI_ATTR void VKAPI_CALL CmdClearColorImage(VkCommandBuffer cb, VkImage im, VkImageLayout, const VkClearColorValue* v, uint32_t, const VkImageSubresourceRange*) {
    Cmd c{Cmd::Clear};
    c.dst = obj<Image>(uint64_t(im));
    const FormatInfo& fi = c.dst->fi;
    const int ch = (fi.desc.x > 0) + (fi.desc.y > 0) + (fi.desc.z > 0) + (fi.desc.w > 0);
    const int bits = fi.desc.x;
    for (int i = 0; i < ch; ++i) {
        const int src = fi.bgra && i != 3 ? 2 - i : i;
        if (fi.desc.f == hipChannelFormatKindFloat) {
            if (bits == 32) std::memcpy(c.texel + 4 * i, &v->float32[src], 4);
            else { uint16_t hv = toHalf(v->float32[src]); std::memcpy(c.texel + 2 * i, &hv, 2); }
        } else if (fi.read == hipReadModeNormalizedFloat) {
            const float f = std::clamp(v->float32[src], fi.desc.f == hipChannelFormatKindSigned ? -1.0f : 0.0f, 1.0f);
            const double mx = fi.desc.f == hipChannelFormatKindSigned ? (bits == 8 ? 127 : 32767) : (bits == 8 ? 255 : 65535);
            const int32_t q = int32_t(std::lround(f * mx));
            if (bits == 8) c.texel[i] = uint8_t(q); else { uint16_t s = uint16_t(q); std::memcpy(c.texel + 2 * i, &s, 2); }
        } else {
            const uint32_t u = v->uint32[src];
            std::memcpy(c.texel + (bits / 8) * i, &u, bits / 8);
        }
    }
    cbOf(cb)->cmds.push_back(c);
}

VKAPI_ATTR VkResult VKAPI_CALL CreateQueryPool(VkDevice, const VkQueryPoolCreateInfo* ci, const VkAllocationCallbacks*, VkQueryPool* out) {
    auto* p = new QueryPool;
    p->ev.resize(ci->queryCount); p->written.assign(ci->queryCount, 0);
    for (auto& e : p->ev) H->hipEventCreateWithFlags(&e, 0);
    *out = handle<VkQueryPool>(p);
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL DestroyQueryPool(VkDevice, VkQueryPool h, const VkAllocationCallbacks*) {
    QueryPool* p = obj<QueryPool>(uint64_t(h));
    if (!p) return;
    H->hipStreamSynchronize(g_stream);
    for (auto e : p->ev) H->hipEventDestroy(e);
    delete p;
}
VKAPI_ATTR void VKAPI_CALL CmdResetQueryPool(VkCommandBuffer cb, VkQueryPool p, uint32_t first, uint32_t n) {
    Cmd c{Cmd::ResetQuery}; c.pool = obj<QueryPool>(uint64_t(p)); c.query = first; c.count = n; cbOf(cb)->cmds.push_back(c);
}
VKAPI_ATTR void VKAPI_CALL CmdWriteTimestamp(VkCommandBuffer cb, VkPipelineStageFlagBits, VkQueryPool p, uint32_t q) {
    Cmd c{Cmd::Timestamp}; c.pool = obj<QueryPool>(uint64_t(p)); c.query = q; cbOf(cb)->cmds.push_back(c);
}
VKAPI_ATTR VkResult VKAPI_CALL GetQueryPoolResults(VkDevice, VkQueryPool h, uint32_t first, uint32_t n, size_t, void* data, VkDeviceSize stride, VkQueryResultFlags flags) {
    QueryPool* p = obj<QueryPool>(uint64_t(h));
    for (uint32_t i = 0; i < n; ++i) {
        const uint32_t q = first + i;
        if (q >= p->ev.size() || !p->written[q] || H->hipEventQuery(p->ev[q])) return VK_NOT_READY;
        float ms = 0;
        H->hipEventElapsedTime(&ms, g_epoch, p->ev[q]);
        const uint64_t ns = uint64_t(double(ms) * 1e6);
        uint8_t* dst = static_cast<uint8_t*>(data) + i * stride;
        if (flags & VK_QUERY_RESULT_64_BIT) std::memcpy(dst, &ns, 8);
        else { uint32_t v = uint32_t(ns); std::memcpy(dst, &v, 4); }
    }
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL CreateFence(VkDevice, const VkFenceCreateInfo*, const VkAllocationCallbacks*, VkFence* out) {
    auto* f = new Fence;
    H->hipEventCreateWithFlags(&f->ev, hipEventDisableTiming);
    *out = handle<VkFence>(f);
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL DestroyFence(VkDevice, VkFence h, const VkAllocationCallbacks*) {
    Fence* f = obj<Fence>(uint64_t(h));
    if (!f) return;
    H->hipEventSynchronize(f->ev);
    H->hipEventDestroy(f->ev);
    delete f;
}
VKAPI_ATTR VkResult VKAPI_CALL WaitForFences(VkDevice, uint32_t n, const VkFence* fs, VkBool32, uint64_t) {
    for (uint32_t i = 0; i < n; ++i) {
        Fence* f = obj<Fence>(uint64_t(fs[i]));
        if (f->pending && H->hipEventSynchronize(f->ev)) return VK_ERROR_DEVICE_LOST;
        f->pending = false;
    }
    return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL QueueSubmit(VkQueue, uint32_t n, const VkSubmitInfo* si, VkFence fence) {
    for (uint32_t i = 0; i < n; ++i)
        for (uint32_t j = 0; j < si[i].commandBufferCount; ++j)
            if (!submitOne(cbOf(si[i].pCommandBuffers[j]), nullptr, 0, nullptr, 0, nullptr, nullptr)) return VK_ERROR_DEVICE_LOST;
    if (fence) { Fence* f = obj<Fence>(uint64_t(fence)); H->hipEventRecord(f->ev, g_stream); f->pending = true; }
    return VK_SUCCESS;
}

PFN_vkVoidFunction lookup(const char* name);
VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL GetDeviceProcAddr(VkDevice, const char* name) { return lookup(name); }
VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL GetInstanceProcAddr(VkInstance, const char* name) { return lookup(name); }

PFN_vkVoidFunction lookup(const char* name) {
    static const std::map<std::string, PFN_vkVoidFunction> table = {
#define E(f) {"vk" #f, reinterpret_cast<PFN_vkVoidFunction>(&f)}
        E(CreateInstance), E(DestroyInstance), E(EnumeratePhysicalDevices), E(GetPhysicalDeviceFormatProperties), E(GetPhysicalDeviceMemoryProperties),
        E(GetPhysicalDeviceProperties), E(GetPhysicalDeviceProperties2), E(GetPhysicalDeviceQueueFamilyProperties), E(CreateDevice), E(DestroyDevice),
        E(GetDeviceQueue), E(DeviceWaitIdle), E(AllocateMemory), E(FreeMemory), E(MapMemory), E(UnmapMemory), E(CreateBuffer), E(DestroyBuffer),
        E(GetBufferMemoryRequirements), E(BindBufferMemory), E(GetBufferDeviceAddress), E(CreateImage), E(DestroyImage), E(GetImageMemoryRequirements),
        E(BindImageMemory), E(CreateImageView), E(DestroyImageView), E(CreateSampler), E(DestroySampler), E(GetDescriptorEXT), E(CreatePipelineLayout),
        E(DestroyPipelineLayout), E(CreateShaderModule), E(DestroyShaderModule), E(CreateComputePipelines), E(DestroyPipeline), E(CreateCommandPool),
        E(DestroyCommandPool), E(AllocateCommandBuffers), E(FreeCommandBuffers), E(BeginCommandBuffer), E(EndCommandBuffer), E(CmdBindPipeline),
        E(CmdPushConstants), E(CmdDispatch), E(CmdDispatchBase), E(CmdPipelineBarrier), E(CmdCopyBufferToImage), E(CmdCopyImageToBuffer), E(CmdCopyImage),
        E(CmdBlitImage), E(CmdClearColorImage), E(CmdFillBuffer), E(CmdUpdateBuffer), E(CmdCopyBuffer), E(CreateQueryPool), E(DestroyQueryPool), E(CmdResetQueryPool), E(CmdWriteTimestamp),
        E(GetQueryPoolResults), E(CreateFence), E(DestroyFence), E(WaitForFences), E(QueueSubmit), E(GetDeviceProcAddr), E(GetInstanceProcAddr),
#undef E
    };
    auto it = table.find(name);
    return it == table.end() ? nullptr : it->second;
}

bool g_installed = false;

} // namespace

bool install(std::string& why) {
    if (g_installed) return true;
    const char* w = nullptr;
    H = load(&w);
    if (!H) { why = w ? w : "HIP runtime"; return false; }
    g_arch = deviceArch();
    {
        alignas(16) unsigned char props[kDevicePropBytes + 1024] = {};
        if (!H->hipGetDevicePropertiesR0600(props, 0)) g_name.assign(reinterpret_cast<const char*>(props), strnlen(reinterpret_cast<const char*>(props), 256));
    }
    if (g_arch.rfind("gfx10", 0) && g_arch.rfind("gfx11", 0) && g_arch.rfind("gfx12", 0)) { why = "not an RDNA GPU (" + g_arch + ")"; return false; }
    if (!ok(H->hipStreamCreateWithFlags(&g_stream, hipStreamNonBlocking), "hipStreamCreate", &why)) return false;
    H->hipEventCreateWithFlags(&g_epoch, 0);
    H->hipEventRecord(g_epoch, g_stream);
    H->hipEventCreateWithFlags(&g_submitBegin, 0);
    H->hipEventCreateWithFlags(&g_submitEnd, 0);
    g_timed = true;
#define SET(f) f = reinterpret_cast<PFN_##f>(lookup(#f));
    HIPVK_TABLE(SET)
#undef SET
    g_installed = true;
    return true;
}

bool installed() { return g_installed; }
PFN_vkVoidFunction procAddr(const char* name) { return lookup(name); }
bool imageViewSize(VkImageView view, uint32_t& w, uint32_t& h) {
    const View* v = obj<View>(uint64_t(view));
    if (!v || !v->image) return false;
    w = v->image->w; h = v->image->h;
    return true;
}
std::string gpuTarget() { return g_arch; }
void setKernelDir(const std::string& dir) { g_kernelDir = dir; g_blitTried = false; }

VkPipeline createKernel(const std::vector<uint8_t>& elf, const std::string& name, KernelAbi abi, const uint32_t tg[3], uint32_t kernargBytes,
                        uint32_t dynamicLds, const std::vector<SamplerGlobal>& samplers, std::string& why) {
    auto* p = new Pipeline;
    p->abi = abi; p->kernarg = abi == KernelAbi::ArgPointer ? 8 : kernargBytes; p->lds = dynamicLds;
    std::memcpy(p->tg, tg, sizeof p->tg);
    if (!ok(H->hipModuleLoadData(&p->module, elf.data()), ("load " + name).c_str(), &why) ||
        !ok(H->hipModuleGetFunction(&p->fn, p->module, name.c_str()), ("kernel " + name).c_str(), &why)) {
        if (p->module) H->hipModuleUnload(p->module);
        delete p;
        return VK_NULL_HANDLE;
    }
    for (const auto& s : samplers) {
        void* g = nullptr; size_t bytes = 0;
        if (H->hipModuleGetGlobal(&g, &bytes, p->module, s.symbol.c_str())) continue;   // optimised away
        if (!ok(H->hipMemcpy(g, s.descriptor, std::min<size_t>(16, bytes), hipMemcpyHostToDevice), "sampler global", &why)) {
            H->hipModuleUnload(p->module); delete p; return VK_NULL_HANDLE;
        }
    }
    return handle<VkPipeline>(p);
}

bool importD3D12(void* ntHandle, uint64_t bytes, bool resource, External& out, std::string& why) {
    hipExternalMemoryHandleDesc d{};
    d.type = resource ? hipExternalMemoryHandleTypeD3D12Resource : hipExternalMemoryHandleTypeD3D12Heap;
    d.handle.win32.handle = ntHandle;
    d.size = bytes;
    d.flags = resource ? 1 : 0;   // hipExternalMemoryDedicated
    hipExternalMemory_t m = nullptr;
    if (!ok(H->hipImportExternalMemory(&m, &d), "hipImportExternalMemory", &why)) return false;
    hipExternalMemoryBufferDesc b{};
    b.size = bytes;
    void* p = nullptr;
    if (!ok(H->hipExternalMemoryGetMappedBuffer(&p, m, &b), "hipExternalMemoryGetMappedBuffer", &why)) { H->hipDestroyExternalMemory(m); return false; }
    out.memory = m; out.pointer = p; out.bytes = bytes;
    return true;
}

void release(External& e) {
    if (e.memory) { H->hipStreamSynchronize(g_stream); H->hipDestroyExternalMemory(static_cast<hipExternalMemory_t>(e.memory)); }
    e = External{};
}

VkImage wrapLinear(void* pointer, uint32_t pitch, VkFormat format, uint32_t w, uint32_t h) {
    auto* im = new Image;
    if (!formatInfo(format, im->fi)) { delete im; return VK_NULL_HANDLE; }
    im->format = format; im->w = w; im->h = h; im->pitch = pitch; im->dev = static_cast<uint8_t*>(pointer); im->external = true;
    if (!makeSrd(*im, nullptr)) { delete im; return VK_NULL_HANDLE; }
    return handle<VkImage>(im);
}

void* importD3D12Fence(void* ntHandle, std::string& why) {
    hipExternalSemaphoreHandleDesc d{};
    d.type = hipExternalSemaphoreHandleTypeD3D12Fence;
    d.handle.win32.handle = ntHandle;
    hipExternalSemaphore_t s = nullptr;
    if (!ok(H->hipImportExternalSemaphore(&s, &d), "hipImportExternalSemaphore", &why)) return nullptr;
    return s;
}

void releaseFence(void* fence) { if (fence) H->hipDestroyExternalSemaphore(static_cast<hipExternalSemaphore_t>(fence)); }

bool submit(VkCommandBuffer cb, void* wait, uint64_t waitValue, void* signal, uint64_t signalValue, std::string& why) {
    return submitOne(cbOf(cb), static_cast<hipExternalSemaphore_t>(wait), waitValue, static_cast<hipExternalSemaphore_t>(signal), signalValue, nullptr, &why);
}

void finish() { if (H && g_stream) H->hipStreamSynchronize(g_stream); }

float lastMilliseconds() {
    float ms = 0;
    if (g_timed && !H->hipEventQuery(g_submitEnd)) H->hipEventElapsedTime(&ms, g_submitBegin, g_submitEnd);
    return ms;
}

} // namespace hipvk
