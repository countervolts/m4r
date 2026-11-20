// m4r: translated Metal kernels as Vulkan compute pipelines (see kernel.h).
// SPDX-License-Identifier: GPL-3.0-only
#include "kernel.h"
#include <cstdio>
#include <fstream>
#include <random>
#include <sstream>

namespace m4r {

static std::vector<uint8_t> readFile(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    return std::vector<uint8_t>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

bool decodeSamplerState(uint64_t w0, VkSamplerCreateInfo& ci, std::string& why) {
    // bits 0-2/3-5/6-8: s/t/r address mode (0 clamp_to_zero, 1 clamp_to_edge, 2 repeat, 3 mirrored_repeat,
    // 4 clamp_to_border: assumed MSL enum order), 9-10 mag filter, 11-12 min filter (0 nearest, 1 linear),
    // 13-14 mip filter, 15 coord::pixel. The upper words hold LOD clamps (0x7bff = half max) and are ignored:
    // every MetalFX texture has one mip level.
    ci = VkSamplerCreateInfo{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    auto addr = [&](uint32_t v, VkSamplerAddressMode& m) {
        switch (v) {
        case 0: m = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER; return true;
        case 1: m = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE; return true;
        case 2: m = VK_SAMPLER_ADDRESS_MODE_REPEAT; return true;
        case 3: m = VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT; return true;
        case 4: m = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER; return true;
        default: return false;
        }
    };
    if (!addr(w0 & 7, ci.addressModeU) || !addr((w0 >> 3) & 7, ci.addressModeV) || !addr((w0 >> 6) & 7, ci.addressModeW)) {
        why = "address mode"; return false;
    }
    const uint32_t mag = (w0 >> 9) & 3, min = (w0 >> 11) & 3, mip = (w0 >> 13) & 3;
    // filter 2 = bicubic (MetalFX's V3 history warp): the translator turns those samples into Catmull-Rom from 9 linear taps
    // (air/intrinsics.py sample_tex_bicubic), so the Vulkan sampler is linear
    if (mag > 2 || min > 2) { why = "filter mode " + std::to_string(mag) + "/" + std::to_string(min); return false; }
    ci.magFilter = mag ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
    ci.minFilter = min ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
    ci.mipmapMode = mip == 2 ? VK_SAMPLER_MIPMAP_MODE_LINEAR : VK_SAMPLER_MIPMAP_MODE_NEAREST;
    ci.borderColor = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK;
    ci.maxLod = 0.0f;
    ci.unnormalizedCoordinates = (w0 >> 15) & 1;
    if (ci.unnormalizedCoordinates) {
        ci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        if (ci.magFilter != ci.minFilter) { why = "pixel coordinates need equal filters"; return false; }
    }
    return true;
}

bool Runtime::init() {
    // vkhip donors are keyed by their specialisation (including the pipeline id); RADV's on-disk shader cache can
    // hand back a pipeline patched by an earlier process for a repeated key. Fresh ids per process avoid that.
    nextId_ = uint32_t(std::random_device{}()) | 1u;
    if (!hip_.load(vk_.device, vkGetDeviceProcAddr)) { error = "vkhip device functions"; return false; }
    VkPushConstantRange pc{VK_SHADER_STAGE_COMPUTE_BIT, 0, 8};
    VkPipelineLayoutCreateInfo li{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    li.pushConstantRangeCount = 1; li.pPushConstantRanges = &pc;
    if (vkCreatePipelineLayout(vk_.device, &li, nullptr, &layout_) != VK_SUCCESS) { error = "pipeline layout"; return false; }
    return vk_.createBuffer(ring_, 16 << 20, true);
}

void Runtime::destroy() {
    for (auto& [n, k] : kernels_) {
        if (k->pipeline) vkDestroyPipeline(vk_.device, k->pipeline, nullptr);
        for (auto s : k->keep) vkDestroySampler(vk_.device, s, nullptr);
        vk_.destroyBuffer(k->data);
    }
    kernels_.clear();
    vk_.destroyBuffer(ring_);
    if (layout_) vkDestroyPipelineLayout(vk_.device, layout_, nullptr);
}

VkDeviceAddress Runtime::transient(const void* data, size_t bytes, size_t align) {
    ringUsed_ = (ringUsed_ + align - 1) & ~(align - 1);
    if (ringUsed_ + bytes > ring_.size) ringUsed_ = 0;  // callers reset per frame; wrap as a last resort
    std::memcpy(static_cast<uint8_t*>(ring_.map) + ringUsed_, data, bytes);
    VkDeviceAddress a = ring_.address + ringUsed_;
    ringUsed_ += bytes;
    return a;
}

Kernel* Runtime::load(const std::string& base, uint32_t extraLds) {
    auto it = kernels_.find(base);
    if (it != kernels_.end()) return it->second.get();
    auto k = std::make_unique<Kernel>();
    std::ifstream af(base + ".args");
    if (!af) { error = base + ".args missing"; return nullptr; }
    std::string line;
    while (std::getline(af, line)) {
        std::istringstream ls(line);
        std::string w;
        ls >> w;
        if (w == "kernel") ls >> k->name;
        else if (w == "tg") ls >> k->tg[0] >> k->tg[1] >> k->tg[2];
        else if (w == "kernarg") ls >> k->kernargSize;
        else if (w == "grid") ls >> k->gridOffset;
        else if (w == "arg") {
            std::string kind;
            Arg a;
            std::string acc;
            ls >> kind >> a.location >> a.offset >> a.name >> acc;
            a.readonly = acc == "r";
            a.kind = kind == "buffer" ? Arg::Buffer : kind == "texture" ? Arg::Texture : kind == "sampler" ? Arg::Sampler : Arg::Threadgroup;
            k->args.push_back(a);
        } else if (w == "sampler") {
            Kernel::SamplerGlobal s;
            ls >> s.global >> s.state[0] >> s.state[1];
            k->samplers.push_back(s);
        }
    }
    auto elf = readFile(base + ".hsaco");
    if (!k->module.parse(elf)) { error = base + ": " + k->module.error; return nullptr; }
    auto kd = k->module.kernels.find(k->name);
    if (kd == k->module.kernels.end()) { error = base + ": kernel symbol missing"; return nullptr; }
    k->kd = kd->second;
    k->staticLds = k->kd.lds;
    k->kd.lds = ((k->kd.lds + 15) & ~15u) + extraLds;
    uint64_t dataAddr = 0;
    if (k->module.dataHi > k->module.dataLo) {
        if (!vk_.createBuffer(k->data, k->module.dataHi - k->module.dataLo, true)) { error = "data buffer"; return nullptr; }
        std::memcpy(k->data.map, k->module.dataInit.data(), k->module.dataInit.size());
        dataAddr = k->data.address;
        for (auto& s : k->samplers) {
            auto g = k->module.globals.find(s.global);
            if (g == k->module.globals.end()) continue;  // optimised away
            VkSamplerCreateInfo ci;
            std::string why;
            if (!decodeSamplerState(s.state[0], ci, why)) { error = base + ": sampler " + why; return nullptr; }
            uint8_t desc[16];
            VkSampler smp;
            if (!vk_.samplerDescriptor(ci, desc, smp)) { error = "sampler"; return nullptr; }
            k->keep.push_back(smp);
            std::memcpy(static_cast<uint8_t*>(k->data.map) + (g->second.va - k->module.dataLo), desc, 16);
        }
    }
    vkhip::Launch launch{k->tg[0], k->tg[1], k->tg[2]};
    std::string err;
    k->pipeline = vkhip::createPipeline(hip_, layout_, k->module, k->kd, launch, dataAddr, nextId_++, err);
    if (!k->pipeline) { error = err; return nullptr; }
    Kernel* raw = k.get();
    kernels_[base] = std::move(k);
    return raw;
}

Kernel* Runtime::loadGlsl(const std::string& spv, uint32_t tx, uint32_t ty, uint32_t tz) {
    auto it = kernels_.find(spv);
    if (it != kernels_.end()) return it->second.get();
    auto code = readFile(spv);
    if (code.empty() || code.size() % 4) { error = spv + " missing"; return nullptr; }
    VkShaderModuleCreateInfo smi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smi.codeSize = code.size(); smi.pCode = reinterpret_cast<const uint32_t*>(code.data());
    VkShaderModule mod;
    if (vkCreateShaderModule(vk_.device, &smi, nullptr, &mod) != VK_SUCCESS) { error = spv + ": module"; return nullptr; }
    VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    ci.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT, mod, "main", nullptr};
    ci.layout = layout_;
    auto k = std::make_unique<Kernel>();
    k->glsl = true; k->name = spv; k->tg[0] = tx; k->tg[1] = ty; k->tg[2] = tz;
    VkResult r = vkCreateComputePipelines(vk_.device, VK_NULL_HANDLE, 1, &ci, nullptr, &k->pipeline);
    vkDestroyShaderModule(vk_.device, mod, nullptr);
    if (r != VK_SUCCESS) { error = spv + ": pipeline"; return nullptr; }
    Kernel* raw = k.get();
    kernels_[spv] = std::move(k);
    return raw;
}

bool Encoder::dispatchThreads(uint32_t x, uint32_t y, uint32_t z) {
    if (!k_) { error = "no kernel"; return false; }
    return dispatch(x, y, z, (x + k_->tg[0] - 1) / k_->tg[0], (y + k_->tg[1] - 1) / k_->tg[1], (z + k_->tg[2] - 1) / k_->tg[2]);
}

bool Encoder::dispatchThreadgroups(uint32_t x, uint32_t y, uint32_t z) {
    if (!k_) { error = "no kernel"; return false; }
    return dispatch(x * k_->tg[0], y * k_->tg[1], z * k_->tg[2], x, y, z);
}

bool Encoder::dispatch(uint32_t gx, uint32_t gy, uint32_t gz, uint32_t nx, uint32_t ny, uint32_t nz) {
    if (!nx || !ny || !nz) return true;
    if (k_->glsl) {
        uint8_t block[8 * 8 + 8 * 4 + 16] = {};
        for (uint32_t i = 0; i < 8; ++i) {
            uint64_t v = 0;
            auto b = buffers_.find(i);
            auto t = textures_.find(i);
            if (b != buffers_.end()) v = b->second;
            else if (t != textures_.end()) v = t->second;
            std::memcpy(block + 8 * i, &v, 8);
        }
        for (uint32_t i = 8; i < 16; ++i) {
            auto r = raw_.find(i);
            if (r != raw_.end()) std::memcpy(block + 64 + 4 * (i - 8), r->second.data(), std::min<size_t>(4, r->second.size()));
        }
        const uint32_t grid[3] = {gx, gy, gz};
        std::memcpy(block + 96, grid, 12);
        VkDeviceAddress args = rt_.transient(block, sizeof(block), 64);
        if (barrier_) rt_.vk().barrier(cb_);
        barrier_ = true;
        vkCmdBindPipeline(cb_, VK_PIPELINE_BIND_POINT_COMPUTE, k_->pipeline);
        vkCmdPushConstants(cb_, rt_.layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, &args);
        vkCmdDispatch(cb_, nx, ny, nz);
        return true;
    }
    std::vector<uint8_t> block(k_->kernargSize, 0);
    uint32_t ldsOffset = (k_->staticLds + 15) & ~15u;
    for (const auto& a : k_->args) {
        uint64_t v = 0;
        if (a.kind == Arg::Buffer) {
            auto it = buffers_.find(a.location);
            v = it == buffers_.end() ? 0 : it->second;
        } else if (a.kind == Arg::Texture) {
            auto it = textures_.find(a.location);
            v = (it == textures_.end() || !it->second) ? rt_.vk().nullRecord() : it->second;
        } else if (a.kind == Arg::Threadgroup) {
            uint32_t off = ldsOffset;
            ldsOffset += (lds_[a.location] + 15) & ~15u;
            std::memcpy(block.data() + a.offset, &off, 4);
            continue;
        }
        std::memcpy(block.data() + a.offset, &v, 8);
    }
    const uint32_t grid[3] = {gx, gy, gz};
    std::memcpy(block.data() + k_->gridOffset, grid, 12);
    VkDeviceAddress args = rt_.transient(block.data(), block.size(), 64);
    if (barrier_) rt_.vk().barrier(cb_);
    barrier_ = true;
    vkCmdBindPipeline(cb_, VK_PIPELINE_BIND_POINT_COMPUTE, k_->pipeline);
    vkCmdPushConstants(cb_, rt_.layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, &args);
    vkCmdDispatch(cb_, nx, ny, nz);
    return true;
}

} // namespace m4r
