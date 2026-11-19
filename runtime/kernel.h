// m4r: translated Metal kernels as Vulkan compute pipelines (vkhip), bound the way Metal binds them.
// SPDX-License-Identifier: GPL-3.0-only
#pragma once
#include "vk.h"
#include "vkhip/vkhip.h"
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace m4r {

struct Arg {
    enum Kind { Buffer, Texture, Sampler, Threadgroup } kind = Buffer;
    int location = -1;       // Metal [[buffer(n)]] / [[texture(n)]] index
    uint32_t offset = 0;     // byte offset in the kernel-argument block
    std::string name;
    bool readonly = false;   // the kernel only reads it (manifest access read / sample)
};

struct Kernel {
    std::string name;
    uint32_t tg[3] = {1, 1, 1};
    uint32_t kernargSize = 0, gridOffset = 0;
    std::vector<Arg> args;
    struct SamplerGlobal { std::string global; uint64_t state[2]; };
    std::vector<SamplerGlobal> samplers;
    vkhip::Module module;
    vkhip::Kernel kd;
    Buffer data;              // the code object's globals (constexpr sampler descriptors live here)
    VkPipeline pipeline = VK_NULL_HANDLE;
    std::vector<VkSampler> keep;
    // m4r's own GLSL kernels (passes that are not Metal kernels, e.g. MPSGraph parts): argument block is
    // 8 x u64 (buffer addresses / texture records by binding index 0-7), then u32 constants from setBytes 8-15
    bool glsl = false;
    uint32_t staticLds = 0;   // the code object's own LDS; dynamic threadgroup memory follows it
};

// Metal sampler state words (as stored in AIR constexpr samplers) -> Vulkan sampler. Bit layout inferred from
// MetalFX's own kernels, see docs/air-translation.md; unsupported modes return false.
bool decodeSamplerState(uint64_t w0, VkSamplerCreateInfo& ci, std::string& why);

class Runtime {
public:
    explicit Runtime(Context& vk) : vk_(vk) {}
    bool init();
    void destroy();
    // loads <base>.hsaco + <base>.args; kernels are cached by base path
    Kernel* load(const std::string& base, uint32_t extraLds = 0);   // extraLds: dynamic threadgroup memory
    // a SPIR-V compute shader using the same push-constant ABI (one 64-bit argument-block pointer)
    Kernel* loadGlsl(const std::string& spv, uint32_t tx, uint32_t ty, uint32_t tz);
    Context& vk() { return vk_; }
    VkPipelineLayout layout() const { return layout_; }
    // transient per-frame memory for argument blocks and setBytes data
    VkDeviceAddress transient(const void* data, size_t bytes, size_t align = 16);
    void resetTransient() { ringUsed_ = 0; }
    std::string error;

private:
    Context& vk_;
    vkhip::Device hip_;
    VkPipelineLayout layout_ = VK_NULL_HANDLE;
    std::map<std::string, std::unique_ptr<Kernel>> kernels_;
    uint32_t nextId_ = 1;
    Buffer ring_;
    size_t ringUsed_ = 0;
};

// A compute encoder with Metal's binding model: bindings persist across setKernel and dispatches.
class Encoder {
public:
    Encoder(Runtime& rt, VkCommandBuffer cb) : rt_(rt), cb_(cb) {}
    void setKernel(Kernel* k) { k_ = k; }
    void setBuffer(VkDeviceAddress a, uint32_t index) { buffers_[index] = a; }
    void setBuffer(const Buffer& b, uint32_t index, VkDeviceSize offset = 0) { buffers_[index] = b.address + offset; }
    void setBytes(const void* data, size_t bytes, uint32_t index) {
        buffers_[index] = rt_.transient(data, bytes);
        raw_[index].assign(static_cast<const uint8_t*>(data), static_cast<const uint8_t*>(data) + bytes);
    }
    void setTexture(const Texture* t, uint32_t index) { textures_[index] = t ? t->record : 0; }
    void setThreadgroupMemory(uint32_t bytes, uint32_t index) { lds_[index] = bytes; }
    void setBarrier(bool b) { barrier_ = b; }   // memory barrier before the next dispatch (default: always)
    bool dispatchThreads(uint32_t x, uint32_t y = 1, uint32_t z = 1);       // Metal dispatchThreads (grid in threads)
    bool dispatchThreadgroups(uint32_t x, uint32_t y = 1, uint32_t z = 1);  // Metal dispatchThreadgroups
    std::string error;

private:
    bool dispatch(uint32_t gx, uint32_t gy, uint32_t gz, uint32_t nx, uint32_t ny, uint32_t nz);
    Runtime& rt_;
    VkCommandBuffer cb_;
    Kernel* k_ = nullptr;
    std::map<uint32_t, VkDeviceAddress> buffers_, textures_;
    std::map<uint32_t, uint32_t> lds_;
    std::map<uint32_t, std::vector<uint8_t>> raw_;
    bool barrier_ = true;
};

} // namespace m4r
