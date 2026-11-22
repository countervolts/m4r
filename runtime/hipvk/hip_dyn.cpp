// m4r: run-time loading of AMD's HIP runtime (see hip_dyn.h).
// SPDX-License-Identifier: GPL-3.0-only
#include "hip_dyn.h"
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <cstring>
#include <mutex>
#include <string>

namespace hipdyn {

static Api g_api;
static const Api* g_loaded = nullptr;
static std::string g_why, g_arch;

const Api* load(const char** why) {
    static std::once_flag once;
    std::call_once(once, [] {
        // the driver installs both; amdhip64_7 first (ROCm 7 runtime), amdhip64_6 for older drivers. M4R_HIP_DLL overrides.
        const char* names[] = {getenv("M4R_HIP_DLL"), "amdhip64_7.dll", "amdhip64_6.dll"};
        HMODULE m = nullptr;
        for (const char* n : names)
            if (n && *n && (m = LoadLibraryA(n)) != nullptr) break;
        if (!m) { g_why = "no HIP runtime (amdhip64_7.dll / amdhip64_6.dll) found: install a current AMD Software driver"; return; }
        g_api.module = m;
#define HIPDYN_LOAD(name, ret, args) \
        g_api.name = reinterpret_cast<ret (*) args>(GetProcAddress(m, #name)); \
        if (!g_api.name) { g_why = std::string("the HIP runtime has no ") + #name; return; }
        HIPDYN_FUNCS(HIPDYN_LOAD)
#undef HIPDYN_LOAD
        hipError_t e = g_api.hipInit(0);
        int n = 0;
        if (e || g_api.hipGetDeviceCount(&n) || n < 1) { g_why = std::string("hipInit: ") + g_api.err(e) + " (no HIP device)"; return; }
        if ((e = g_api.hipSetDevice(0)) != 0) { g_why = std::string("hipSetDevice: ") + g_api.err(e); return; }
        alignas(16) unsigned char props[kDevicePropBytes + 1024] = {};
        if ((e = g_api.hipGetDevicePropertiesR0600(props, 0)) != 0) { g_why = std::string("hipGetDeviceProperties: ") + g_api.err(e); return; }
        const char* a = reinterpret_cast<const char*>(props + kGcnArchNameOffset);
        g_arch.assign(a, strnlen(a, 255));
        g_arch = g_arch.substr(0, g_arch.find(':'));
        g_loaded = &g_api;
    });
    if (why) *why = g_why.c_str();
    return g_loaded;
}

const char* deviceArch() { return load() ? g_arch.c_str() : ""; }

} // namespace hipdyn
