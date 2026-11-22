// SPDX-License-Identifier: GPL-3.0-only
#include "vkfn.h"
#ifdef M4R_HIPVK
#include "hipvk/hipvk.h"
#include <cstdio>
#endif

#ifdef M4R_DYNAMIC_VK
#define M4R_VK_DEFINE(f) PFN_##f f = nullptr;
PFN_vkCreateInstance vkCreateInstance = nullptr;
M4R_VK_INSTANCE_FUNCS(M4R_VK_DEFINE)
M4R_VK_DEVICE_FUNCS(M4R_VK_DEFINE)
#undef M4R_VK_DEFINE
#endif

namespace m4r {
bool loadVulkan(PFN_vkGetInstanceProcAddr gipa, VkInstance instance, VkDevice device) {
#ifdef M4R_HIPVK
    // native Windows: the table is hipvk's (Vulkan on AMD's HIP runtime); the arguments are unused
    (void)gipa; (void)instance; (void)device;
    std::string why;
    if (!hipvk::install(why)) { std::fprintf(stderr, "m4r: %s\n", why.c_str()); return false; }
    return true;
#elif defined(M4R_DYNAMIC_VK)
    bool ok = true;
#define LI(f) f = reinterpret_cast<PFN_##f>(gipa(instance, #f)); ok &= f != nullptr;
    M4R_VK_INSTANCE_FUNCS(LI)
#undef LI
#define LD(f) f = reinterpret_cast<PFN_##f>(vkGetDeviceProcAddr(device, #f)); ok &= f != nullptr;
    M4R_VK_DEVICE_FUNCS(LD)
#undef LD
    return ok;
#else
    (void)gipa; (void)instance; (void)device;
    return true;
#endif
}
}
