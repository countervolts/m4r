// From d4r (github.com/countervolts/d4r, engine/vkhip), GPL-3.0-only.
// vkhip: run native AMDGPU (HIP code object) kernels as Vulkan compute pipelines on RADV, unchanged.
//
// m4r: RDNA2 (gfx10.3) and RDNA4 (gfx12) too: the instruction encodings vkhip patches come from a per-generation table and
// gfx12's getpc sequence (with s_sext_i32_i16 and s_wait_alu) is recognised. Only gfx1101 is tested on hardware.
// A HIP kernel on gfx11 starts with its kernarg pointer in s[0:1] (two user SGPRs; on gfx1100/gfx1102 LLVM pads
// them to 16 user + system SGPRs for a wave32 init bug, which the kernel's own rsrc2 keeps), the workgroup ids after
// them, packed thread ids in v0, and reaches its globals PC-relative (s_getpc_b64 + s_add_u32 + s_addc_u32).
// A RADV compute shader whose only interface is one 64-bit push constant gets exactly that SGPR layout
// (gfx11 compute declares no ring offsets; the push constant is inlined into the first user SGPRs). So:
//   1. compile a donor pipeline with the kernel's workgroup size and LDS size,
//   2. read it back through the pipeline cache (core Vulkan; RADV stores the shader as radv_shader_binary_legacy:
//      config + info + code, no checksum),
//   3. replace its code with the kernel's .text (behind one s_branch to the entry) and its register config
//      (rsrc1/rsrc2/rsrc3, VGPRs, scratch) with the kernel descriptor's,
//   4. create the pipeline again from the patched cache data.
// The globals live in a Vulkan buffer: each s_getpc/s_add/s_addc triple that targets .data/.bss becomes two
// s_mov_b32 of the buffer address plus s_nop (same size, same instruction count). Every other instruction is
// the code object's own, so the arithmetic is bit-identical to the HIP launch.
#pragma once
#include <vulkan/vulkan.h>
#include <cstdint>
#include <cstring>
#include <map>
#include <string>
#include <vector>

namespace vkhip {

struct Kernel {
    std::string name;
    uint64_t entry = 0;  // virtual address
    uint32_t lds = 0, scratch = 0, kernarg = 0, rsrc1 = 0, rsrc2 = 0, rsrc3 = 0, props = 0, preload = 0;
    uint32_t kernargSgpr = 0;   // first SGPR of the kernarg pointer (4 on RDNA2, after the private segment buffer)
};

struct Global { uint64_t va = 0, size = 0; };

// A parsed AMDGPU code object: its .text, its writable data image and its kernels.
struct Module {
    std::vector<uint8_t> text;
    uint64_t textVa = 0;
    uint32_t mach = 0;                // EF_AMDGPU_MACH_* (machOf)
    int family = 0;                   // 10 (RDNA2), 11 (RDNA3, RDNA 3.5) or 12 (RDNA4)
    uint64_t dataLo = 0, dataHi = 0;  // the .data/.bss range
    std::vector<uint8_t> dataInit;    // initial contents of [dataLo, dataHi)
    std::map<std::string, Kernel> kernels;
    std::map<std::string, Global> globals;
    struct Site { uint32_t word = 0, sdst = 0; uint64_t target = 0; uint32_t words = 5; };   // the getpc sequence: first word, length
    std::vector<Site> sites;           // getpc triples that address data
    uint32_t textGetpc = 0;            // getpc triples that address code (left alone)
    std::string error;

    bool parse(const std::vector<uint8_t>& elf);
    // .text with every data site redirected to dataAddress + (target - dataLo)
    std::vector<uint32_t> patchedText(uint64_t dataAddress) const;
};

// The ELF machine of code objects for a KFD gfx_target_version (110001 = gfx1101, 100300 = gfx1030, 120001 = gfx1201);
// 0 outside RDNA2, RDNA3, RDNA 3.5 and RDNA4. gfx11Mach is the old name.
uint32_t machOf(uint32_t gfxTargetVersion);
uint32_t gfx11Mach(uint32_t gfxTargetVersion);
// 10, 11 or 12 for an RDNA2/3/4 code-object machine, 0 otherwise
int machFamily(uint32_t mach);
// RDNA 3.5 (gfx1150-gfx1153) code-object machines
bool gfx115Mach(uint32_t mach);

// Vulkan entry points (device-level) the injector needs.
struct Device {
    VkDevice device = VK_NULL_HANDLE;
    PFN_vkCreatePipelineCache createCache = nullptr;
    PFN_vkDestroyPipelineCache destroyCache = nullptr;
    PFN_vkGetPipelineCacheData cacheData = nullptr;
    PFN_vkCreateComputePipelines createCompute = nullptr;
    PFN_vkDestroyPipeline destroyPipeline = nullptr;
    PFN_vkCreateShaderModule createModule = nullptr;
    PFN_vkDestroyShaderModule destroyModule = nullptr;
    bool load(VkDevice d, PFN_vkGetDeviceProcAddr proc);
};

struct Launch { uint32_t bx = 32, by = 1, bz = 1; };

// Creates the pipeline running `kernel` with workgroup size `launch`. `layout` must have exactly one push
// constant range of 8 bytes (compute stage) and no descriptor sets. `id` must differ between all pipelines
// created on one device (it keys the donor). Returns VK_NULL_HANDLE and sets `error` on failure.
VkPipeline createPipeline(const Device& vk, VkPipelineLayout layout, const Module& module, const Kernel& kernel,
                          Launch launch, uint64_t dataAddress, uint32_t id, std::string& error);

} // namespace vkhip
