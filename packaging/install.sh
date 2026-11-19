#!/usr/bin/env bash
# m4r installer: run it in the game folder the release was extracted into (next to OptiScaler's dxgi.dll).
# Gets the MetalFX files (macOS 27.2.1), checks their SHA-256 hashes and builds m4r's plans for this GPU and the
# connected displays from them. MetalFX files and plans are kept in ~/.cache/m4r, so the next game copies them.
# SPDX-License-Identifier: GPL-3.0-only
set -euo pipefail
METALFX_URL='https://cdn.ayois.gay/sk-ant-api03-vtAgsUCb9K8raWCNjC1Mg2HK4u69uxOBVvRphTYVECoNNsHXRpi5LJL5_h2qSda_Uz7aeK8rijh0c8snFqVAFIELZE7zgAA'   # metalfx-27.2.1.zip
METALFX_ZIP_SHA256=5be6ad36084c7b0b5ea46fc6ecd0238530981c00b80d41e131f6a66e9815a775
METALFX_VERSION=27.2.1
METALFX_FILES=(
  "MetalFX f8d89ddde419975af7c1fd72ecea345da3efd6ffd87c9737e6c075b08a41fbc8"
  "Resources/default.metallib da0225cc853c3dac172913805449a53057186e1beafeaac018f274977520be56"
  "Resources/emit_sacneh_constants.dat 4e308ad9fc8450ee8b016226ec5980fc3f1d6366632e2dfd544868fd612171e1"
  "Resources/emit_sbbrnet_nhwc_q_constants.dat 4a268b0cf3722f2e46aff771557f8fa9426bcf621be692cd18231e1f21c3f665"
  "Resources/emit_tbbrnet_nhwc_q_constants.dat 9250a45553b53a0d16c10e04683d884ffe4e99b3abeb98fb0c3ef99b98064bf0"
)

fail() { printf 'install.sh: %s\n' "$*" >&2; exit 1; }
usage() {
  printf '%s\n' 'Usage: bash install.sh [options] [game folder]' \
    'Default folder: the one containing this script.' \
    '  --metalfx PATH     the MetalFX zip (metalfx-27.2.1.zip) or its extracted folder instead of downloading it' \
    '  --display WxH      display resolution to build plans for (repeatable; default: the connected displays)' \
    '  --input WxH        extra render resolution (dynamic resolution, custom scaling; repeatable)' \
    '  --modes LIST       quality modes (default quality,balanced,performance,ultra; also dlaa, ultra-quality)' \
    '  --gpu gfxNNNN      GPU to build for (default: detected)' \
    '  --no-fg            no frame generation plans' \
    '  --line-locking     upscaler plans with MetalFX line locking (about 0.06 ms more)' \
    '  --rebuild          build the plans again even when cached ones fit' \
    '  -y, --yes          accept every default (also the behaviour without a terminal)' \
    'Requires python3 with numpy and LLVM 23 or newer with lld (llvm-as, opt, llc, ld.lld, llvm-nm, llvm-objdump);' \
    'curl to download the MetalFX files. No sudo needed. M4R_CACHE_DIR moves the cache.'
}
YES=0 REBUILD=0 METALFX='' PASS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    -y|--yes) YES=1; shift ;;
    --rebuild) REBUILD=1; shift ;;
    --metalfx) [[ $# -ge 2 ]] || { usage >&2; exit 2; }; METALFX=$2; shift 2 ;;
    --display|--input|--modes|--gpu) [[ $# -ge 2 ]] || { usage >&2; exit 2; }; PASS+=("$1" "$2"); shift 2 ;;
    --no-fg|--line-locking) PASS+=("$1"); shift ;;
    -*) usage >&2; exit 2 ;;
    *) break ;;
  esac
done
[[ $# -le 1 ]] || { usage >&2; exit 2; }
SELF=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
GAME=$(cd -- "${1:-$SELF}" && pwd)
RUNTIME="$GAME/m4r"
TOOLS="$SELF/m4r/tools"
[[ -f "$RUNTIME/m4r.dll" ]] || fail "Extract the release ZIP into the game folder first; no m4r/m4r.dll in $GAME"
[[ -f "$GAME/dxgi.dll" ]] || fail "No dxgi.dll (OptiScaler) in $GAME; extract the whole release ZIP there."
[[ -f "$TOOLS/scripts/install.py" ]] || fail "missing $TOOLS (extract the whole release ZIP)"
CACHE="${M4R_CACHE_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/m4r}"
interactive() { (( YES == 0 )) && [[ -t 0 && -t 1 ]]; }
ask() {   # ask "QUESTION" DEFAULT(y|n)
  local answer
  if ! interactive; then [[ $2 == y ]]; return; fi
  while true; do
    read -r -p "$1 [$([[ $2 == y ]] && echo Y/n || echo y/N)] " answer || { echo; answer=''; }
    case "${answer,,}" in
      '') [[ $2 == y ]]; return ;;
      y*) return 0 ;;
      n*) return 1 ;;
      *) echo '  Please answer y or n (or press Enter for the default).' ;;
    esac
  done
}
TMP=$(mktemp -d)
trap 'rm -rf -- "$TMP"' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
command -v sha256sum >/dev/null || fail 'Missing sha256sum (install coreutils).'
sha() { sha256sum -- "$1" | cut -d' ' -f1; }

# --- tools ------------------------------------------------------------------------------------------------------
# distributions that keep a versioned LLVM outside PATH (Debian/Ubuntu apt.llvm.org: /usr/lib/llvm-23/bin)
for d in $(ls -d /usr/lib/llvm-[0-9]*/bin /usr/lib64/llvm[0-9]*/bin 2>/dev/null | sort -V -r); do
  if ! command -v llc >/dev/null || [[ $("$d/llc" --version 2>/dev/null | sed -n 's/.*LLVM version \([0-9]*\).*/\1/p') -gt \
        $(llc --version | sed -n 's/.*LLVM version \([0-9]*\).*/\1/p') ]]; then
    [[ -x "$d/llc" ]] && PATH="$d:$PATH"
  fi
done
export PATH
missing=()
command -v python3 >/dev/null || missing+=(python3)
python3 -c 'import numpy' 2>/dev/null || missing+=(numpy)
for t in llvm-as llvm-dis opt llc llvm-nm llvm-objdump ld.lld; do command -v "$t" >/dev/null || missing+=("$t"); done
LLVM=$(llc --version 2>/dev/null | sed -n 's/.*LLVM version \([0-9]*\).*/\1/p' || true)
if (( ${#missing[@]} )) || [[ -z "$LLVM" || "$LLVM" -lt 23 ]]; then
  (( ${#missing[@]} )) && echo "Missing: ${missing[*]}" >&2
  [[ -z "$LLVM" || "$LLVM" -ge 23 ]] || echo "LLVM $LLVM found; m4r needs LLVM 23 or newer (it reads Apple's Metal bitcode)." >&2
  echo 'Install them, then run install.sh again:' >&2
  echo '  Arch, CachyOS, SteamOS dev mode:  sudo pacman -S llvm lld python-numpy' >&2
  echo '  Fedora:                           sudo dnf install llvm lld python3-numpy' >&2
  echo '  Debian, Ubuntu:                   https://apt.llvm.org (llvm.sh 23), then sudo apt install lld-23 python3-numpy' >&2
  exit 1
fi

# --- MetalFX files ----------------------------------------------------------------------------------------------
MFX="$CACHE/metalfx/$METALFX_VERSION"
files_ok() {   # files_ok DIR: every file present with its hash
  local f name sum
  for f in "${METALFX_FILES[@]}"; do
    read -r name sum <<<"$f"
    [[ -f "$1/$name" && $(sha "$1/$name") == "$sum" ]] || return 1
  done
}
from_zip() {   # from_zip ZIP: extract into $TMP/mfx and check
  rm -rf -- "$TMP/mfx"
  python3 -I -c 'import os, sys, zipfile
z = zipfile.ZipFile(sys.argv[1]); out = sys.argv[2]
for n in z.namelist():
    parts = n.split("/")
    if n.endswith("/") or ".." in parts:
        continue
    rel = "/".join(parts[1:]) if len(parts) > 1 and parts[1] in ("MetalFX", "Resources") else n
    if rel == "MetalFX" or rel.startswith("Resources/"):
        os.makedirs(os.path.dirname(os.path.join(out, rel)) or out, exist_ok=True)
        open(os.path.join(out, rel), "wb").write(z.read(n))' "$1" "$TMP/mfx" || return 1
  files_ok "$TMP/mfx"
}
keep_mfx() {   # keep_mfx DIR: copy the checked files into the cache
  local f
  rm -rf -- "$MFX.tmp"; mkdir -p -- "$MFX.tmp/Resources"
  for f in "${METALFX_FILES[@]}"; do cp -- "$1/${f%% *}" "$MFX.tmp/${f%% *}"; done
  rm -rf -- "$MFX"; mv -- "$MFX.tmp" "$MFX"
}
get_metalfx() {
  local z
  if [[ -n "$METALFX" ]]; then
    if [[ -d "$METALFX" ]]; then
      files_ok "$METALFX" || fail "$METALFX does not hold the MetalFX $METALFX_VERSION files (hashes differ or files missing)."
      keep_mfx "$METALFX"
    else
      from_zip "$METALFX" || fail "$METALFX is not the MetalFX $METALFX_VERSION zip (hashes differ)."
      keep_mfx "$TMP/mfx"
    fi
    return
  fi
  if files_ok "$MFX"; then echo "MetalFX $METALFX_VERSION: verified copy in $MFX."; return; fi
  # a zip downloaded by hand: next to this script, in the game folder, in Downloads or on the Desktop
  for z in "$SELF"/metalfx-*.zip "$GAME"/metalfx-*.zip "$HOME"/Downloads/metalfx-*.zip "$HOME"/Desktop/metalfx-*.zip; do
    [[ -f "$z" ]] || continue
    if from_zip "$z"; then echo "MetalFX $METALFX_VERSION: using $z."; keep_mfx "$TMP/mfx"; return; fi
  done
  [[ -n "$METALFX_URL" ]] || fail "No MetalFX files found. Put metalfx-$METALFX_VERSION.zip next to install.sh (or pass --metalfx PATH)."
  echo "m4r runs Apple's MetalFX upscaler and frame interpolation from the macOS $METALFX_VERSION MetalFX files (~1.2 MiB)."
  ask "Download them now (once: later installs use $MFX)?" y ||
    fail "Not downloaded. Put metalfx-$METALFX_VERSION.zip next to install.sh or pass --metalfx PATH."
  command -v curl >/dev/null || fail 'Missing curl. Install it and run this script again.'
  curl --fail --location --show-error --retry 3 --connect-timeout 30 --progress-bar \
    --proto '=https' --proto-redir '=https' --output "$TMP/metalfx.zip" "$METALFX_URL" ||
    fail "Download failed: $METALFX_URL. Run the installer again to retry."
  [[ $(sha "$TMP/metalfx.zip") == "$METALFX_ZIP_SHA256" ]] || fail 'SHA-256 mismatch for the MetalFX download; nothing installed.'
  from_zip "$TMP/metalfx.zip" || fail 'The MetalFX download does not hold the expected files; nothing installed.'
  keep_mfx "$TMP/mfx"
  echo "MetalFX $METALFX_VERSION: downloaded and verified."
}
get_metalfx

# --- displays ---------------------------------------------------------------------------------------------------
if ! printf '%s\n' "${PASS[@]}" | grep -qx -- --display; then
  # each connected output's preferred mode (the first one the kernel lists)
  displays=$(for s in /sys/class/drm/card*-*/status; do
    [[ $(cat "$s" 2>/dev/null) == connected ]] && head -n1 "$(dirname "$s")/modes" 2>/dev/null
  done | grep -E '^[0-9]+x[0-9]+$' | sort -u || true)
  if [[ -z "$displays" ]]; then
    displays=$'1920x1080\n2560x1440\n3840x2160'
    echo 'No connected display found; building plans for 1920x1080, 2560x1440 and 3840x2160.'
  else
    echo "Displays: $(echo $displays)"
  fi
  for d in $displays; do PASS+=(--display "$d"); done
fi

# --- plans ------------------------------------------------------------------------------------------------------
# One set of plans per GPU, MetalFX files, tools and options serves every game: built once into the cache.
GPU=$(python3 -I -c 'import importlib.util, sys
s = importlib.util.spec_from_file_location("i", sys.argv[1]); i = importlib.util.module_from_spec(s); s.loader.exec_module(i)
g = i.detect_gpu(); print(g[0] if g else "")' "$TOOLS/scripts/install.py")
for ((i = 0; i < ${#PASS[@]}; i++)); do [[ ${PASS[i]} != --gpu ]] || GPU=${PASS[i+1]}; done
if [[ -z "$GPU" ]]; then
  interactive && read -r -p 'GPU not detected. LLVM target (e.g. gfx1101 for RX 7700/7800, gfx1100 for RX 7900): ' GPU || true
  [[ -n "$GPU" ]] || fail 'Could not detect the GPU; pass --gpu (e.g. --gpu gfx1101).'
  PASS+=(--gpu "$GPU")
fi
KEY=$({ echo "gpu=$GPU"; printf '%s\n' "${METALFX_FILES[@]}"
        (cd "$TOOLS" && find . -type f ! -name '*.pyc' -print0 | LC_ALL=C sort -z | xargs -0 sha256sum); } | sha256sum | cut -c1-16)
PLANS="$CACHE/plans/$KEY"
(( REBUILD == 0 )) || rm -rf -- "$PLANS"
mkdir -p -- "$PLANS"
echo "Building the plans for $GPU (a few seconds each, once: kept in $PLANS)..."
python3 -I "$TOOLS/scripts/install.py" --resources "$MFX/Resources" --binary "$MFX/MetalFX" --plans "$PLANS" "${PASS[@]}" ||
  fail 'Some plans were not built (reason above).'

# the plans this run asked for, copied into the game (a plan the game had is replaced)
mkdir -p -- "$RUNTIME/plans"
n=0
while read -r name; do
  [[ -f "$PLANS/$name/plan.txt" ]] || continue
  rm -rf -- "$RUNTIME/plans/$name.new"
  cp -r --reflink=auto -- "$PLANS/$name" "$RUNTIME/plans/$name.new"
  rm -rf -- "$RUNTIME/plans/$name"
  mv -- "$RUNTIME/plans/$name.new" "$RUNTIME/plans/$name"
  n=$((n + 1))
done < <(ls "$PLANS")
echo "$n plans in $RUNTIME/plans."
echo
echo 'Done. In Steam: Properties > General > Launch Options:  WINEDLLOVERRIDES="dxgi=n,b" %command%'
echo 'Then select DLSS (or FSR/XeSS) in the game: OptiScaler runs MetalFX instead. Insert opens its menu.'
echo 'Frame generation: on in OptiScaler.ini ([FrameGen] Enabled=true); MetalFX FG makes the extra frames.'
echo 'Plans contain Apple'"'"'s translated kernels and weights: keep them to yourself.'
