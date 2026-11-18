#!/usr/bin/env bash
# End-to-end test: the D3D12 harness calls OptiScaler's NGX entry points under Proton, OptiScaler's upscaler is MetalFX
# (m4r). Everything is staged in a fresh app directory; the Proton install is not modified.
# usage: tests/run_optiscaler.sh PROTON_DIR OPTISCALER_DLL M4R_DLL PLAN_DIR FIXTURE_DIR [FRAMES IN_W IN_H OUT_W OUT_H]
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
cat > "$APP/OptiScaler.ini" <<EOF
[Upscalers]
Dx12Upscaler=metalfx
[FrameGen]
Enabled=false
FGInput=nofg
FGOutput=nofg
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
