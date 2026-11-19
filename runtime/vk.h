// m4r: minimal Vulkan context for running translated MetalFX kernels (Linux harness and in-game backend).
// SPDX-License-Identifier: GPL-3.0-only
#pragma once
#include "vkfn.h"
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace m4r {

struct Buffer {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceSize size = 0;
    VkDeviceAddress address = 0;
    void* map = nullptr;  // host-visible buffers only
};

// A texture as the translated kernels see it: a 64-byte record (sampled-image descriptor, then storage-image
// descriptor) stored in device memory; `record` is its device address.
struct Texture {
    VkImage image = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkFormat format = VK_FORMAT_UNDEFINED;
    uint32_t width = 0, height = 0, layers = 1;
    VkDeviceAddress record = 0;
    bool owned = true;
};

// The LLVM GPU target (gfx1030 ... gfx1201) of a RADV device, from the chip name RADV puts in the device name
// ("AMD Radeon RX 7700 XT (RADV NAVI32)"); M4R_GPU overrides it. Empty for anything else.
std::string gpuTarget(const char* deviceName);

struct Context {
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physical = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    uint32_t queueFamily = 0;
    VkCommandPool pool = VK_NULL_HANDLE;
    VkPhysicalDeviceMemoryProperties memProps{};
    VkPhysicalDeviceDescriptorBufferPropertiesEXT descProps{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_BUFFER_PROPERTIES_EXT};
    PFN_vkGetDescriptorEXT getDescriptor = nullptr;
    bool ownsDevice = false;
    std::string gpu;             // LLVM target of the device (gfx1101, ...; gpuTarget); empty if not recognised
    std::string error;

    // Descriptor records live in one host-visible arena.
    Buffer records;
    VkDeviceSize recordsUsed = 0;

    bool create();               // own instance and device (harness)
    // use an existing device (inside a game); no work is submitted to its queues
    bool adopt(VkInstance inst, VkPhysicalDevice phys, VkDevice dev, uint32_t family);
    // a view and descriptor record for an image owned by someone else (not destroyed by destroyTexture's owner check)
    bool wrap(Texture& t, VkImage image, VkFormat format, uint32_t w, uint32_t h);
    void destroy();

    uint32_t memoryType(uint32_t bits, VkMemoryPropertyFlags want) const;
    bool createBuffer(Buffer& b, VkDeviceSize size, bool hostVisible, VkBufferUsageFlags extra = 0);
    void destroyBuffer(Buffer& b);
    bool createTexture(Texture& t, VkFormat format, uint32_t w, uint32_t h, uint32_t layers = 1);
    void destroyTexture(Texture& t);
    bool writeRecord(Texture& t);   // (re)writes the descriptor record for t
    VkDeviceAddress nullRecord();   // a record of zeros (Metal nil texture)
    bool samplerDescriptor(const VkSamplerCreateInfo& ci, uint8_t out[16], VkSampler& keep);

    VkCommandBuffer begin();
    bool submit(VkCommandBuffer cb);  // submit and wait
    // whole-image transitions for harness images (kernels use VK_IMAGE_LAYOUT_GENERAL throughout)
    void toGeneral(VkCommandBuffer cb, Texture& t);
    void barrier(VkCommandBuffer cb);  // compute/transfer -> compute/transfer, all memory
    bool upload(Texture& t, const void* data, size_t bytes);
    bool download(Texture& t, void* data, size_t bytes);

private:
    VkDeviceAddress nullRecord_ = 0;
};

uint32_t formatBytes(VkFormat f);

} // namespace m4r
