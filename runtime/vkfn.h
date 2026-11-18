// m4r: Vulkan entry points. On Linux (harness) the loader's prototypes are used; with M4R_DYNAMIC_VK (the Windows
// DLL inside a game) every function is a pointer loaded from the game's own instance/device.
// SPDX-License-Identifier: GPL-3.0-only
#pragma once
#ifdef M4R_DYNAMIC_VK
#define VK_NO_PROTOTYPES
#endif
#include <vulkan/vulkan.h>

#define M4R_VK_INSTANCE_FUNCS(X) X(vkCreateDevice) X(vkEnumeratePhysicalDevices) X(vkGetPhysicalDeviceFormatProperties) \
    X(vkGetPhysicalDeviceMemoryProperties) X(vkGetPhysicalDeviceProperties) X(vkGetPhysicalDeviceProperties2) \
    X(vkGetPhysicalDeviceQueueFamilyProperties) X(vkDestroyInstance) X(vkGetDeviceProcAddr)
#define M4R_VK_DEVICE_FUNCS(X) X(vkAllocateCommandBuffers) X(vkAllocateMemory) X(vkBeginCommandBuffer) X(vkBindBufferMemory) \
    X(vkBindImageMemory) X(vkCmdBindPipeline) X(vkCmdClearColorImage) X(vkCmdCopyBufferToImage) X(vkCmdCopyImageToBuffer) \
    X(vkCmdDispatch) X(vkCmdPipelineBarrier) X(vkCmdPushConstants) X(vkCreateBuffer) X(vkCreateCommandPool) \
    X(vkCreateComputePipelines) X(vkCreateFence) X(vkCreateImage) X(vkCreateImageView) X(vkCreatePipelineLayout) \
    X(vkCreateSampler) X(vkCreateShaderModule) X(vkDestroyBuffer) X(vkDestroyCommandPool) X(vkDestroyDevice) X(vkDestroyFence) \
    X(vkDestroyImage) X(vkDestroyImageView) X(vkDestroyPipeline) X(vkDestroyPipelineLayout) X(vkDestroySampler) \
    X(vkDestroyShaderModule) X(vkDeviceWaitIdle) X(vkEndCommandBuffer) X(vkFreeCommandBuffers) X(vkFreeMemory) \
    X(vkGetBufferDeviceAddress) X(vkGetBufferMemoryRequirements) X(vkGetDeviceQueue) X(vkGetImageMemoryRequirements) \
    X(vkMapMemory) X(vkQueueSubmit) X(vkWaitForFences)

#ifdef M4R_DYNAMIC_VK
#define M4R_VK_DECLARE(f) extern PFN_##f f;
extern PFN_vkCreateInstance vkCreateInstance;
M4R_VK_INSTANCE_FUNCS(M4R_VK_DECLARE)
M4R_VK_DEVICE_FUNCS(M4R_VK_DECLARE)
#undef M4R_VK_DECLARE
#endif

namespace m4r {
// Loads the table (no-op without M4R_DYNAMIC_VK). Returns false if a function is missing.
bool loadVulkan(PFN_vkGetInstanceProcAddr gipa, VkInstance instance, VkDevice device);
}
