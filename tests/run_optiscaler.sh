#!/usr/bin/env bash
# End-to-end test: the D3D12 harness calls OptiScaler's NGX entry points under Proton, OptiScaler's upscaler is MetalFX
# (m4r). Everything is staged in a fresh app directory; the Proton install is not modified.
# usage: tests/run_optiscaler.sh PROTON_DIR OPTISCALER_DLL M4R_DLL PLAN_DIR FIXTURE_DIR [FRAMES IN_W IN_H OUT_W OUT_H]
# M4R_FG_PLAN=<fg plan dir>: also turns on MetalFX frame generation (FGInput=upscaler, FGOutput=metalfx), gives the harness a
# swapchain and writes m4r's frame generation parameters to <app>/m4r.log (compare with tests/check_framegen.sh).
# M4R_FG_DUMP=1 additionally saves the depth and motion vectors m4r received to <app>/dumps.
# SPDX-License-Identifier: GPL-3.0-only
set -euo pipefail
if [[ $# -lt 5 ]]; then
  sed -n 4p "$0" >&2
  exit 2
fi
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROTON_DIR="$(realpath "$1")"; OPTI="$(realpath "$2")"; M4R="$(realpath "$3")"; PLAN="$(realpath "$4")"; FIX="$(realpath "$5")"
FRAMES="${6:-8}"; IW="${7:-640}"; IH="${8:-360}"; OW="${9:-1280}"; OH="${10:-720}"
HARNESS="${M4R_HARNESS:-$ROOT/build-harness/d3d12_harness.exe}"
APP="${M4R_APP_DIR:-$ROOT/build-harness/app}"
COMPAT="${M4R_COMPAT_DIR:-$ROOT/build-harness/compat}"

rm -rf "$APP"
mkdir -p "$APP/m4r/plans" "$APP/out" "$COMPAT"
cp -f "$HARNESS" "$APP/d3d12_harness.exe"
cp -f "$OPTI" "$APP/dxgi.dll"
cp -f "$M4R" "$APP/m4r/m4r.dll"
ln -s "$PLAN" "$APP/m4r/plans/bbr-${OW}x${OH}-${IW}x${IH}"
ln -s "$FIX" "$APP/fixture"
FGEN=false; FGIN=nofg; FGOUT=nofg
if [[ -n "${M4R_FG_PLAN:-}" ]]; then
  ln -s "$(realpath "$M4R_FG_PLAN")" "$APP/m4r/plans/fg-${OW}x${OH}-${IW}x${IH}"
  FGEN=true; FGIN=upscaler; FGOUT=metalfx
  export M4R_HARNESS_SWAPCHAIN=1 M4R_HARNESS_REVERSED_DEPTH=1 M4R_LOG=m4r.log
  if [[ -n "${M4R_FG_DUMP:-}" ]]; then mkdir -p "$APP/dumps"; export M4R_DUMP_FG=dumps; fi   # depth/motion as m4r saw them
fi
cat > "$APP/OptiScaler.ini" <<EOF
[Upscalers]
Dx12Upscaler=metalfx
[FrameGen]
Enabled=$FGEN
FGInput=$FGIN
FGOutput=$FGOUT
[Spoofing]
Dxgi=false
[Plugins]
LoadAsiPlugins=false
[Menu]
OverlayMenu=false
[Log]
LogToFile=true
LogLevel=${M4R_OPTI_LOGLEVEL:-1}
EOF

export STEAM_COMPAT_CLIENT_INSTALL_PATH="${STEAM_COMPAT_CLIENT_INSTALL_PATH:-$HOME/.local/share/Steam}"
export STEAM_COMPAT_DATA_PATH="$COMPAT"
export WINEDLLOVERRIDES="dxgi=n,b"
export WINEDEBUG="${WINEDEBUG:--all}"
cd "$APP"
"$PROTON_DIR/proton" run "$APP/d3d12_harness.exe" dxgi.dll fixture out "$FRAMES" "$IW" "$IH" "$OW" "$OH" || true
cat "$APP/harness.log"
