#!/usr/bin/env bash
# Builds tests/d3d12_harness.cpp with clang-cl + the xwin MSVC CRT/SDK (needs external/optiscaler for the NGX headers).
# SPDX-License-Identifier: GPL-3.0-only
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
X="${XWIN:-$HOME/.xwin}"
mkdir -p "$ROOT/build-harness"
clang-cl --target=x86_64-pc-windows-msvc -fms-compatibility-version=19.44 \
  /imsvc"$X/crt/include" /imsvc"$X/sdk/include/ucrt" /imsvc"$X/sdk/include/um" /imsvc"$X/sdk/include/shared" \
  /EHsc /O2 /std:c++20 /MT -D_CRT_SECURE_NO_WARNINGS -I"$ROOT/external/optiscaler/external/nvngx_dlss_sdk" \
  "$ROOT/tests/d3d12_harness.cpp" /Fo"$ROOT/build-harness/" /Fe"$ROOT/build-harness/d3d12_harness.exe" -fuse-ld=lld-link \
  /link /libpath:"$X/crt/lib/x86_64" /libpath:"$X/sdk/lib/um/x86_64" /libpath:"$X/sdk/lib/ucrt/x86_64" d3d12.lib dxgi.lib
