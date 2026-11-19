#!/usr/bin/env bash
# Builds the release zip: extract it into a game folder, run install.sh. Holds no Apple material: install.sh fetches the
# MetalFX files and builds the plans on the user's machine.
#   dxgi.dll, OptiScaler.ini      OptiScaler at fd3a45a with patches/optiscaler (build-opti, clang-cl + xwin)
#   m4r/m4r.dll                   build-win
#   m4r/tools/                    the plan builder (air/, tools/, scripts/install.py, shaders/sacneh.{comp,spv})
#   install.sh, uninstall.sh, M4R_README.txt
# usage: scripts/package.sh [out dir (default dist)]
# SPDX-License-Identifier: GPL-3.0-only
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$(mkdir -p "${1:-$ROOT/dist}" && cd "${1:-$ROOT/dist}" && pwd)"
VERSION="$(cat "$ROOT/packaging/VERSION")"
OPTI="$ROOT/external/optiscaler"
PIN=fd3a45a

# the OptiScaler checkout must be the pinned commit with exactly the shipped patches (and clang-cl.patch) applied
[[ "$(git -C "$OPTI" rev-parse --short=7 HEAD)" == "$PIN" ]] || { echo "external/optiscaler is not at $PIN" >&2; exit 1; }
CHECK="$(mktemp -d)"
trap 'git -C "$OPTI" worktree remove --force "$CHECK/src" 2>/dev/null || true; rm -rf -- "$CHECK"' EXIT
git -C "$OPTI" worktree add --detach -q "$CHECK/src" "$PIN"
"$ROOT/scripts/apply_optiscaler_patches.sh" "$CHECK/src" --clang-cl >/dev/null
bad=0
while read -r f; do
  [[ -L "$OPTI/$f" && ! -e "$CHECK/src/$f" ]] && continue   # optiscaler/casefix.py's symlinks
  cmp -s "$OPTI/$f" "$CHECK/src/$f" || { echo "external/optiscaler/$f differs from the patches" >&2; bad=1; }
done < <({ git -C "$OPTI" diff --name-only HEAD; git -C "$OPTI" ls-files --others --exclude-standard
           git -C "$CHECK/src" diff --name-only HEAD; git -C "$CHECK/src" ls-files --others --exclude-standard; } | sort -u)
(( bad == 0 )) || { echo 'fold the changes into patches/optiscaler (scripts/sync_optiscaler_patches.py) first' >&2; exit 1; }

cmake --build "$ROOT/build-opti"
cmake --build "$ROOT/build-win"

STAGE="$CHECK/m4r-$VERSION"
mkdir -p "$STAGE/m4r/tools/scripts" "$STAGE/m4r/tools/shaders"
cp "$ROOT/build-opti/OptiScaler.dll" "$STAGE/dxgi.dll"
cp "$ROOT/build-win/m4r.dll" "$STAGE/m4r/m4r.dll"
# OptiScaler.ini: the patched one with MetalFX selected
python3 -I - "$OPTI/OptiScaler.ini" "$STAGE/OptiScaler.ini" <<'EOF'
import sys
want = {'Upscalers': {'Dx12Upscaler': 'metalfx'}, 'FrameGen': {'FGInput': 'upscaler', 'FGOutput': 'metalfx'}}
sec, out, seen = None, [], set()
for line in open(sys.argv[1], encoding='utf-8-sig').read().split('\n'):
    s = line.strip()
    if s.startswith('[') and s.endswith(']'):
        sec = s[1:-1]
    elif '=' in s and not s.startswith(';'):
        k = s.split('=', 1)[0].strip()
        if k in want.get(sec, {}):
            line = f'{k}={want[sec][k]}'
            seen.add((sec, k))
    out.append(line)
assert seen == {(s, k) for s, d in want.items() for k in d}, seen
open(sys.argv[2], 'w', newline='\r\n').write('\n'.join(out))
EOF
cp -r "$ROOT/air" "$ROOT/tools" "$STAGE/m4r/tools/"
cp "$ROOT/scripts/install.py" "$STAGE/m4r/tools/scripts/"
cp "$ROOT/shaders/sacneh.comp" "$STAGE/m4r/tools/shaders/"
glslangValidator -V --target-env vulkan1.3 "$ROOT/shaders/sacneh.comp" -o "$STAGE/m4r/tools/shaders/sacneh.spv" >/dev/null
find "$STAGE/m4r/tools" \( -name __pycache__ -o -name '*.pyc' \) -prune -exec rm -rf {} +
cp "$ROOT/packaging/install.sh" "$ROOT/packaging/uninstall.sh" "$STAGE/"
cp "$ROOT/packaging/M4R_README.txt" "$ROOT/LICENSE" "$STAGE/"
cp "$ROOT/LICENSE" "$STAGE/m4r/LICENSE"
chmod 755 "$STAGE/install.sh" "$STAGE/uninstall.sh"
# uninstall.sh removes what the zip put there
(cd "$STAGE" && find . -type f ! -path ./m4r/files.txt | sed 's|^\./||' | LC_ALL=C sort > "$CHECK/files" && mv "$CHECK/files" m4r/files.txt)
# no Apple material: nothing from a MetalFX dump may be in the zip
if grep -rlI --exclude=install.sh -e 'emit_tbbrnet' "$STAGE" | grep -v '\.py$' | grep -q .; then
  echo 'unexpected MetalFX data in the release' >&2; exit 1
fi
find "$STAGE" -exec touch -h -d "@$(git -C "$ROOT" log -1 --format=%ct)" {} +
rm -f "$OUT/m4r-$VERSION.zip"
(cd "$STAGE" && LC_ALL=C find . -mindepth 1 | sed 's|^\./||' | LC_ALL=C sort | zip -X -9 -q "$OUT/m4r-$VERSION.zip" -@)
(cd "$OUT" && sha256sum "m4r-$VERSION.zip" > "m4r-$VERSION.zip.sha256")
echo "$OUT/m4r-$VERSION.zip ($(du -h "$OUT/m4r-$VERSION.zip" | cut -f1))"
