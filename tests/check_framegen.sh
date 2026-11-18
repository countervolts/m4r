#!/usr/bin/env bash
# Replays the frame generation of a harness run (M4R_FG_PLAN=<fg plan> tests/run_optiscaler.sh ...) through m4r's C API on a
# native Vulkan device and compares: restored real frames, generated frames (see tests/check_framegen.py).
# usage: tests/check_framegen.sh <app dir> <fixture dir> <fg plan dir> <frames> [IN_W IN_H OUT_W OUT_H]
# MID_DIR=<dir with mid-NN.rgba16f (tests/scene_fg.py)>: also scores the generated frames against the true in-between frames.
# BBR_PLAN=<bbr plan dir>: the replay also runs the upscaler on every frame (as the harness does), so the frame generation
# kernels run after the upscaler's kernels in both.
# SPDX-License-Identifier: GPL-3.0-only
set -euo pipefail
if [[ $# -lt 4 ]]; then
  sed -n 4p "$0" >&2
  exit 2
fi
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP="$(realpath "$1")"; FIX="$(realpath "$2")"; PLAN="$(realpath "$3")"; FRAMES="$4"
IW="${5:-640}"; IH="${6:-360}"; OW="${7:-1280}"; OH="${8:-720}"
rm -rf "$APP/ref" "$APP/refplans"
mkdir -p "$APP/ref" "$APP/refplans"
ln -s "$PLAN" "$APP/refplans/fg-${OW}x${OH}-${IW}x${IH}"
if [[ -n "${BBR_PLAN:-}" ]]; then ln -s "$(realpath "$BBR_PLAN")" "$APP/refplans/bbr-${OW}x${OH}-${IW}x${IH}"; export REF_UPSCALE=1; fi
"$ROOT/build/ref_framegen" "$APP/refplans" "$APP/out" "$FIX" "$APP/m4r.log" "$FRAMES" "$APP/ref" >/dev/null
python3 -I "$ROOT/tests/check_framegen.py" "$APP" "$FRAMES" ${MID_DIR:+"$MID_DIR"}
