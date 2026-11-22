// OpenCL C port of the GLSL pass of the same name (see glsl_port.h).
// SPDX-License-Identifier: GPL-3.0-only
#include "glsl_port.h"
PORT_KERNEL {
    const uint px = GID_X, py = GID_Y;
    if (px >= a->c[0] || py >= a->c[1]) return;
    const uint i = py * a->c[0] + px;
    BUF(half, 1)[i] = BUF(const half, 0)[i * 4u + 3u];
}
