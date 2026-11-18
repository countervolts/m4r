#!/usr/bin/env bash
# Applies m4r's patches to an OptiScaler checkout: MetalFX in the upscaler dropdown and MetalFX FG in the frame
# generation menu. m4r ships these patches only, no OptiScaler build.
# usage: scripts/apply_optiscaler_patches.sh <OptiScaler git checkout> [--clang-cl]
#   --clang-cl   also apply optiscaler/clang-cl.patch (only needed to build OptiScaler with clang-cl on Linux)
# SPDX-License-Identifier: GPL-3.0-only
set -euo pipefail
if [[ $# -lt 1 ]]; then
  sed -n 2,6p "$0" >&2
  exit 2
fi
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIR="$(realpath "$1")"; shift
PIN=fd3a45a   # OptiScaler commit the patches were made against ("Display asi's version")

[[ -d "$DIR/.git" ]] || { echo "$DIR is not a git checkout of OptiScaler" >&2; exit 1; }
HEAD="$(git -C "$DIR" rev-parse --short=7 HEAD)"
if [[ "$HEAD" != "$PIN" ]]; then
  echo "warning: checkout is at $HEAD, the patches were made against $PIN; they may not apply" >&2
fi
if ! git -C "$DIR" diff --quiet; then
  echo "$DIR has local changes to tracked files; use a clean checkout" >&2
  exit 1
fi

patches=("$ROOT"/patches/optiscaler/*.patch)
[[ "${1:-}" == "--clang-cl" ]] && patches+=("$ROOT/optiscaler/clang-cl.patch")

# in order (later patches build on earlier ones); a failure takes back what was applied
applied=()
for p in "${patches[@]}"; do
  if ! git -C "$DIR" apply "$p"; then
    echo "does not apply: $p" >&2
    for ((i = ${#applied[@]} - 1; i >= 0; i--)); do git -C "$DIR" apply -R "${applied[i]}"; done
    exit 1
  fi
  applied+=("$p")
done
echo "applied ${#patches[@]} patches to $DIR"
