// m4r: the part of Vulkan m4r uses, implemented on AMD's HIP runtime (native Windows, where RADV's vkhip cannot run).
//
// m4r's runtime reaches Vulkan only through the function table of vkfn.h. hipvk::install() fills that table with functions that
// keep m4r's model working on HIP:
//   - buffers are HIP device memory; host-visible ones also have a host copy made with MEM_WRITE_WATCH, and the pages the CPU wrote
//     since the last submission are uploaded at the start of the next one (through a pinned staging ring), so writes through
//     vkMapMemory pointers and the per-frame transient ring keep working;
//   - images are pitch-linear HIP memory (256-byte rows); their descriptors (the 32-byte image resource the translated kernels read
//     from a texture record) come from HIP texture objects over that memory, samplers' 16-byte descriptors likewise;
//   - a command buffer is a list of commands replayed on a HIP stream at submission: translated kernels launch with their argument
//     block as kernarg bytes (the push-constant address points into a host-visible buffer), m4r's own kernels (the OpenCL ports of
//     its GLSL passes) with the 8-byte address of their argument block; a dispatch with no barrier recorded before it launches with
//     hipExtAnyOrderLaunch, the rest in stream order; copies are 2D memcpys, same-size blits with format conversion a small kernel.
// Pipeline barriers and image layouts have no meaning here (one stream, linear memory).
// SPDX-License-Identifier: GPL-3.0-only
#pragma once
#ifndef VK_NO_PROTOTYPES
#define VK_NO_PROTOTYPES
#endif
#include <vulkan/vulkan.h>
#include <cstdint>
#include <string>
#include <vector>

namespace hipvk {

// Loads HIP and fills m4r's Vulkan function table (vkfn.h). Idempotent. False: *why says why (no HIP runtime, no AMD GPU).
bool install(std::string& why);
bool installed();
// Any hipvk entry point by name (also those outside the project's table, e.g. vkGetDescriptorEXT); null if hipvk has none.
PFN_vkVoidFunction procAddr(const char* name);
// The size of the image a view shows (for kernels that ask for it, e.g. GLSL textureSize). False for an unknown view.
bool imageViewSize(VkImageView view, uint32_t& w, uint32_t& h);
// "gfx1101" for the HIP device (install() first).
std::string gpuTarget();
// <dir>/hipvk_blit.hsaco: the format-converting blit kernel for this GPU (built by install.py next to the plans). Without it only
// blits between formats of equal texel size work (as copies).
void setKernelDir(const std::string& dir);

enum class KernelAbi { Kernarg, ArgPointer };   // translated Metal kernels / m4r's own kernels (one 8-byte argument-block address)
struct SamplerGlobal { std::string symbol; uint8_t descriptor[16]; };
// A compute "pipeline" running `name` of the code object `elf` with workgroup size tg and `dynamicLds` bytes of dynamic LDS;
// `samplers` are written into the module's globals. VK_NULL_HANDLE and *why on failure.
VkPipeline createKernel(const std::vector<uint8_t>& elf, const std::string& name, KernelAbi abi, const uint32_t tg[3],
                        uint32_t kernargBytes, uint32_t dynamicLds, const std::vector<SamplerGlobal>& samplers, std::string& why);

// Memory owned elsewhere: a D3D12 resource shared as an NT handle (the handle is not closed), mapped into HIP as a buffer.
struct External { void* memory = nullptr; void* pointer = nullptr; uint64_t bytes = 0; };
bool importD3D12(void* ntHandle, uint64_t bytes, bool resource, External& out, std::string& why);
void release(External& e);
// An image over pitch-linear memory owned elsewhere (rows of `pitch` bytes, 256-aligned); destroyed with vkDestroyImage.
VkImage wrapLinear(void* pointer, uint32_t pitch, VkFormat format, uint32_t w, uint32_t h);

// A D3D12 fence shared as an NT handle (not closed), for GPU-side ordering with the game's queue.
void* importD3D12Fence(void* ntHandle, std::string& why);
void releaseFence(void* fence);
// Submits cb like vkQueueSubmit, without blocking: the stream first waits until `wait` reaches waitValue (if wait), and `signal`
// is set to signalValue once cb's work is done (if signal).
bool submit(VkCommandBuffer cb, void* wait, uint64_t waitValue, void* signal, uint64_t signalValue, std::string& why);
// Blocks until everything submitted has finished.
void finish();
// GPU milliseconds of the last submit() (0 before the first).
float lastMilliseconds();

} // namespace hipvk
