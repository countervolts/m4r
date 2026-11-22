// m4r's Vulkan function table (vkfn.h), as hipvk::install() fills it. hipvk.cpp is shared with anf4r and x4r, each with its own
// copy of this header naming its table.
// SPDX-License-Identifier: GPL-3.0-only
#pragma once
#include "../vkfn.h"

#define HIPVK_TABLE(X) X(vkCreateInstance) M4R_VK_INSTANCE_FUNCS(X) M4R_VK_DEVICE_FUNCS(X)
