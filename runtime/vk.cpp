// m4r: minimal Vulkan context (see vk.h).
// SPDX-License-Identifier: GPL-3.0-only
#include "vk.h"
#include <cstdlib>
#include <cstdio>
#include <string>

namespace m4r {

uint32_t formatBytes(VkFormat f) {
    switch (f) {
    case VK_FORMAT_R8_UNORM: case VK_FORMAT_R8_UINT: return 1;
    case VK_FORMAT_R16_SFLOAT: case VK_FORMAT_R16_UINT: case VK_FORMAT_R8G8_UNORM: case VK_FORMAT_R16_UNORM: return 2;
    case VK_FORMAT_R32_SFLOAT: case VK_FORMAT_R32_UINT: case VK_FORMAT_R16G16_SFLOAT: case VK_FORMAT_R16G16_UINT:
    case VK_FORMAT_R8G8B8A8_UNORM: case VK_FORMAT_B8G8R8A8_UNORM: case VK_FORMAT_A2B10G10R10_UNORM_PACK32:
    case VK_FORMAT_B10G11R11_UFLOAT_PACK32: case VK_FORMAT_R16G16_SNORM: case VK_FORMAT_R16G16_UNORM: return 4;
    case VK_FORMAT_R16G16B16A16_SFLOAT: case VK_FORMAT_R32G32_SFLOAT: case VK_FORMAT_R16G16B16A16_UINT:
    case VK_FORMAT_R32G32_UINT: return 8;
    case VK_FORMAT_R32G32B32A32_SFLOAT: case VK_FORMAT_R32G32B32A32_UINT: return 16;
    default: return 0;
    }
}

bool Context::create() {
    if (!loadVulkan(nullptr, VK_NULL_HANDLE, VK_NULL_HANDLE)) { error = "vulkan entry points"; return false; }
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "m4r";
    app.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
    if (vkCreateInstance(&ici, nullptr, &instance) != VK_SUCCESS) { error = "vkCreateInstance"; return false; }
    uint32_t n = 0;
    vkEnumeratePhysicalDevices(instance, &n, nullptr);
    std::vector<VkPhysicalDevice> devs(n);
    vkEnumeratePhysicalDevices(instance, &n, devs.data());
    for (auto d : devs) {
        VkPhysicalDeviceProperties p;
        vkGetPhysicalDeviceProperties(d, &p);
        if (p.vendorID == 0x1002 && p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) { physical = d; break; }
        if (p.vendorID == 0x1002 && !physical) physical = d;
    }
    if (!physical) { error = "no AMD GPU"; return false; }
    uint32_t qn = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physical, &qn, nullptr);
    std::vector<VkQueueFamilyProperties> qf(qn);
    vkGetPhysicalDeviceQueueFamilyProperties(physical, &qn, qf.data());
    for (uint32_t i = 0; i < qn; ++i)
        if (qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { queueFamily = i; break; }
    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = queueFamily; qci.queueCount = 1; qci.pQueuePriorities = &prio;

    VkPhysicalDeviceDescriptorBufferFeaturesEXT db{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_BUFFER_FEATURES_EXT};
    db.descriptorBuffer = VK_TRUE;
    VkPhysicalDeviceVulkan13Features v13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES, &db};
    v13.subgroupSizeControl = VK_TRUE; v13.computeFullSubgroups = VK_TRUE; v13.synchronization2 = VK_TRUE;
    VkPhysicalDeviceVulkan12Features v12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, &v13};
    v12.bufferDeviceAddress = VK_TRUE; v12.shaderFloat16 = VK_TRUE; v12.storageBuffer8BitAccess = VK_TRUE;
    VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &v12};
    f2.features.shaderInt64 = VK_TRUE; f2.features.shaderInt16 = VK_TRUE;
    const char* exts[] = {VK_EXT_DESCRIPTOR_BUFFER_EXTENSION_NAME};
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, &f2};
    dci.queueCreateInfoCount = 1; dci.pQueueCreateInfos = &qci;
    dci.enabledExtensionCount = 1; dci.ppEnabledExtensionNames = exts;
    if (vkCreateDevice(physical, &dci, nullptr, &device) != VK_SUCCESS) { error = "vkCreateDevice"; return false; }
    ownsDevice = true;
    vkGetDeviceQueue(device, queueFamily, 0, &queue);
    vkGetPhysicalDeviceMemoryProperties(physical, &memProps);
    VkPhysicalDeviceProperties2 p2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &descProps};
    vkGetPhysicalDeviceProperties2(physical, &p2);
    gpu = gpuTarget(p2.properties.deviceName);
    getDescriptor = reinterpret_cast<PFN_vkGetDescriptorEXT>(vkGetDeviceProcAddr(device, "vkGetDescriptorEXT"));
    VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT; pci.queueFamilyIndex = queueFamily;
    vkCreateCommandPool(device, &pci, nullptr, &pool);
    if (!getDescriptor) { error = "vkGetDescriptorEXT"; return false; }
    // RDNA2 and older: RADV's sampled-image descriptor is 64 bytes (the image descriptor, then FMASK's); the kernels read the
    // first 32 (the image resource), so records keep those
    if (descProps.sampledImageDescriptorSize > 64 || descProps.storageImageDescriptorSize > 64 || descProps.samplerDescriptorSize > 16 ||
        descProps.sampledImageDescriptorSize < 32 || descProps.storageImageDescriptorSize < 32) {
        error = "unsupported descriptor sizes (sampled image " + std::to_string(descProps.sampledImageDescriptorSize) + ", storage image " +
                std::to_string(descProps.storageImageDescriptorSize) + ", sampler " + std::to_string(descProps.samplerDescriptorSize) + ")";
        return false;
    }
    return createBuffer(records, 1 << 20, true);
}

std::string gpuTarget(const char* deviceName) {
    if (const char* o = std::getenv("M4R_GPU")) return o;
    static const struct { const char* chip; const char* gfx; } chips[] = {
        {"NAVI21", "gfx1030"}, {"NAVI22", "gfx1031"}, {"NAVI23", "gfx1032"}, {"VANGOGH", "gfx1033"}, {"NAVI24", "gfx1034"},
        {"REMBRANDT", "gfx1035"}, {"RAPHAEL_MENDOCINO", "gfx1036"},
        {"NAVI31", "gfx1100"}, {"NAVI32", "gfx1101"}, {"NAVI33", "gfx1102"}, {"GFX1103_R1", "gfx1103"}, {"GFX1103_R2", "gfx1103"},
        {"PHOENIX", "gfx1103"}, {"GFX1150", "gfx1150"}, {"GFX1151", "gfx1151"}, {"GFX1152", "gfx1152"}, {"GFX1153", "gfx1153"},
        {"GFX1200", "gfx1200"}, {"NAVI44", "gfx1200"}, {"GFX1201", "gfx1201"}, {"NAVI48", "gfx1201"},
    };
    const std::string n = deviceName;
    const size_t r = n.find("(RADV ");
    if (r == std::string::npos) return {};
    std::string chip = n.substr(r + 6);
    chip = chip.substr(0, chip.find(')'));
    for (const auto& c : chips)
        if (chip == c.chip) return c.gfx;
    return {};
}

bool Context::adopt(VkInstance inst, VkPhysicalDevice phys, VkDevice dev, uint32_t family) {
    instance = inst; physical = phys; device = dev; queueFamily = family; ownsDevice = false;
    vkGetPhysicalDeviceMemoryProperties(physical, &memProps);
    VkPhysicalDeviceProperties2 p2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &descProps};
    vkGetPhysicalDeviceProperties2(physical, &p2);
    gpu = gpuTarget(p2.properties.deviceName);
    getDescriptor = reinterpret_cast<PFN_vkGetDescriptorEXT>(vkGetDeviceProcAddr(device, "vkGetDescriptorEXT"));
    if (!getDescriptor) { error = "VK_EXT_descriptor_buffer is not enabled on the game's device"; return false; }
    // RDNA2 and older: RADV's sampled-image descriptor is 64 bytes (the image descriptor, then FMASK's); the kernels read the
    // first 32 (the image resource), so records keep those
    if (descProps.sampledImageDescriptorSize > 64 || descProps.storageImageDescriptorSize > 64 || descProps.samplerDescriptorSize > 16 ||
        descProps.sampledImageDescriptorSize < 32 || descProps.storageImageDescriptorSize < 32) {
        error = "unsupported descriptor sizes (sampled image " + std::to_string(descProps.sampledImageDescriptorSize) + ", storage image " +
                std::to_string(descProps.storageImageDescriptorSize) + ", sampler " + std::to_string(descProps.samplerDescriptorSize) + ")";
        return false;
    }
    return createBuffer(records, 1 << 20, true);
}

bool Context::wrap(Texture& t, VkImage image, VkFormat format, uint32_t w, uint32_t h) {
    t = Texture{};
    VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    const bool isDepth = format == VK_FORMAT_D32_SFLOAT || format == VK_FORMAT_D32_SFLOAT_S8_UINT ||
                         format == VK_FORMAT_D24_UNORM_S8_UINT || format == VK_FORMAT_D16_UNORM || format == VK_FORMAT_X8_D24_UNORM_PACK32;
    vci.image = image; vci.viewType = VK_IMAGE_VIEW_TYPE_2D; vci.format = format;
    vci.subresourceRange = {VkImageAspectFlags(isDepth ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT), 0, 1, 0, 1};
    if (vkCreateImageView(device, &vci, nullptr, &t.view) != VK_SUCCESS) { error = "image view"; return false; }
    t.image = image; t.format = format; t.width = w; t.height = h; t.owned = false;
    return writeRecord(t);
}

void Context::destroy() {
    if (!device) return;
    vkDeviceWaitIdle(device);
    destroyBuffer(records);
    if (pool) vkDestroyCommandPool(device, pool, nullptr);
    if (ownsDevice) vkDestroyDevice(device, nullptr);
    if (ownsDevice && instance) vkDestroyInstance(instance, nullptr);
    device = VK_NULL_HANDLE;
}

uint32_t Context::memoryType(uint32_t bits, VkMemoryPropertyFlags want) const {
    for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i)
        if ((bits & (1u << i)) && (memProps.memoryTypes[i].propertyFlags & want) == want) return i;
    return UINT32_MAX;
}

bool Context::createBuffer(Buffer& b, VkDeviceSize size, bool hostVisible, VkBufferUsageFlags extra) {
    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size = size;
    bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | extra;
    if (vkCreateBuffer(device, &bci, nullptr, &b.buffer) != VK_SUCCESS) return false;
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(device, b.buffer, &req);
    VkMemoryAllocateFlagsInfo fl{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
    fl.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, &fl};
    mai.allocationSize = req.size;
    uint32_t type = UINT32_MAX;
    if (hostVisible) {
        type = memoryType(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                                   VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        if (type == UINT32_MAX)
            type = memoryType(req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    } else {
        type = memoryType(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    }
    mai.memoryTypeIndex = type;
    if (type == UINT32_MAX || vkAllocateMemory(device, &mai, nullptr, &b.memory) != VK_SUCCESS) return false;
    vkBindBufferMemory(device, b.buffer, b.memory, 0);
    VkBufferDeviceAddressInfo ai{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
    ai.buffer = b.buffer;
    b.address = vkGetBufferDeviceAddress(device, &ai);
    b.size = size;
    if (hostVisible) vkMapMemory(device, b.memory, 0, VK_WHOLE_SIZE, 0, &b.map);
    return true;
}

void Context::destroyBuffer(Buffer& b) {
    if (b.buffer) vkDestroyBuffer(device, b.buffer, nullptr);
    if (b.memory) vkFreeMemory(device, b.memory, nullptr);
    b = Buffer{};
}

bool Context::createTexture(Texture& t, VkFormat format, uint32_t w, uint32_t h, uint32_t layers) {
    VkFormatProperties fp;
    vkGetPhysicalDeviceFormatProperties(physical, format, &fp);
    const bool storage = fp.optimalTilingFeatures & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT;
    VkImageCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ci.imageType = VK_IMAGE_TYPE_2D; ci.format = format; ci.extent = {w, h, 1}; ci.mipLevels = 1; ci.arrayLayers = layers;
    ci.samples = VK_SAMPLE_COUNT_1_BIT; ci.tiling = VK_IMAGE_TILING_OPTIMAL;
    ci.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
               (storage ? VK_IMAGE_USAGE_STORAGE_BIT : 0);
    if (vkCreateImage(device, &ci, nullptr, &t.image) != VK_SUCCESS) return false;
    VkMemoryRequirements req;
    vkGetImageMemoryRequirements(device, t.image, &req);
    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mai.allocationSize = req.size;
    mai.memoryTypeIndex = memoryType(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (vkAllocateMemory(device, &mai, nullptr, &t.memory) != VK_SUCCESS) return false;
    vkBindImageMemory(device, t.image, t.memory, 0);
    VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vci.image = t.image; vci.viewType = layers > 1 ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D; vci.format = format;
    vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, layers};
    if (vkCreateImageView(device, &vci, nullptr, &t.view) != VK_SUCCESS) return false;
    t.format = format; t.width = w; t.height = h; t.layers = layers;
    return writeRecord(t);
}

bool Context::writeRecord(Texture& t) {
    if (!t.record) {
        if (recordsUsed + 64 > records.size) { error = "descriptor arena full"; return false; }
        t.record = records.address + recordsUsed;
        recordsUsed += 64;
    }
    uint8_t* dst = static_cast<uint8_t*>(records.map) + (t.record - records.address);
    std::memset(dst, 0, 64);
    VkDescriptorImageInfo ii{VK_NULL_HANDLE, t.view, VK_IMAGE_LAYOUT_GENERAL};
    VkDescriptorGetInfoEXT gi{VK_STRUCTURE_TYPE_DESCRIPTOR_GET_INFO_EXT};
    gi.type = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE; gi.data.pSampledImage = &ii;
    // a record holds the 32-byte image resource of each descriptor (RDNA2's 64-byte sampled descriptor appends FMASK's)
    uint8_t tmp[64];
    getDescriptor(device, &gi, descProps.sampledImageDescriptorSize, tmp);
    std::memcpy(dst, tmp, 32);
    VkFormatProperties fp;
    vkGetPhysicalDeviceFormatProperties(physical, t.format, &fp);
    if (fp.optimalTilingFeatures & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT) {
        gi.type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; gi.data.pStorageImage = &ii;
        getDescriptor(device, &gi, descProps.storageImageDescriptorSize, tmp);
        std::memcpy(dst + 32, tmp, 32);
    }
    return true;
}

VkDeviceAddress Context::nullRecord() {
    if (!nullRecord_) {
        nullRecord_ = records.address + recordsUsed;
        std::memset(static_cast<uint8_t*>(records.map) + recordsUsed, 0, 64);
        recordsUsed += 64;
    }
    return nullRecord_;
}

bool Context::samplerDescriptor(const VkSamplerCreateInfo& ci, uint8_t out[16], VkSampler& keep) {
    if (vkCreateSampler(device, &ci, nullptr, &keep) != VK_SUCCESS) return false;
    VkDescriptorGetInfoEXT gi{VK_STRUCTURE_TYPE_DESCRIPTOR_GET_INFO_EXT};
    gi.type = VK_DESCRIPTOR_TYPE_SAMPLER; gi.data.pSampler = &keep;
    std::memset(out, 0, 16);
    getDescriptor(device, &gi, descProps.samplerDescriptorSize, out);
    return true;
}

void Context::destroyTexture(Texture& t) {
    if (!t.owned && t.view) vkDestroyImageView(device, t.view, nullptr);
    if (t.owned) {
        if (t.view) vkDestroyImageView(device, t.view, nullptr);
        if (t.image) vkDestroyImage(device, t.image, nullptr);
        if (t.memory) vkFreeMemory(device, t.memory, nullptr);
    }
    t = Texture{};
}

VkCommandBuffer Context::begin() {
    VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    ai.commandPool = pool; ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; ai.commandBufferCount = 1;
    VkCommandBuffer cb;
    vkAllocateCommandBuffers(device, &ai, &cb);
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cb, &bi);
    return cb;
}

bool Context::submit(VkCommandBuffer cb) {
    vkEndCommandBuffer(cb);
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1; si.pCommandBuffers = &cb;
    VkFence fence;
    VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    vkCreateFence(device, &fci, nullptr, &fence);
    VkResult r = vkQueueSubmit(queue, 1, &si, fence);
    if (r == VK_SUCCESS) r = vkWaitForFences(device, 1, &fence, VK_TRUE, 10ull * 1000 * 1000 * 1000);
    vkDestroyFence(device, fence, nullptr);
    vkFreeCommandBuffers(device, pool, 1, &cb);
    if (r != VK_SUCCESS) { error = "submit/wait failed: " + std::to_string(r); return false; }
    return true;
}

void Context::toGeneral(VkCommandBuffer cb, Texture& t) {
    VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    b.srcAccessMask = 0; b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED; b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = t.image; b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, t.layers};
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                         0, 0, nullptr, 0, nullptr, 1, &b);
}

void Context::barrier(VkCommandBuffer cb) {
    VkMemoryBarrier m{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    m.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    m.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    const VkPipelineStageFlags s = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT;
    vkCmdPipelineBarrier(cb, s, s, 0, 1, &m, 0, nullptr, 0, nullptr);
}

bool Context::upload(Texture& t, const void* data, size_t bytes) {
    Buffer s;
    if (!createBuffer(s, bytes, true)) return false;
    std::memcpy(s.map, data, bytes);
    VkCommandBuffer cb = begin();
    toGeneral(cb, t);
    VkBufferImageCopy r{};
    r.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, t.layers};
    r.imageExtent = {t.width, t.height, 1};
    vkCmdCopyBufferToImage(cb, s.buffer, t.image, VK_IMAGE_LAYOUT_GENERAL, 1, &r);
    barrier(cb);
    bool ok = submit(cb);
    destroyBuffer(s);
    return ok;
}

bool Context::download(Texture& t, void* data, size_t bytes) {
    Buffer s;
    if (!createBuffer(s, bytes, true)) return false;
    VkCommandBuffer cb = begin();
    barrier(cb);
    VkBufferImageCopy r{};
    r.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, t.layers};
    r.imageExtent = {t.width, t.height, 1};
    vkCmdCopyImageToBuffer(cb, t.image, VK_IMAGE_LAYOUT_GENERAL, s.buffer, 1, &r);
    bool ok = submit(cb);
    if (ok) std::memcpy(data, s.map, bytes);
    destroyBuffer(s);
    return ok;
}

} // namespace m4r
