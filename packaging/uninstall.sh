#!/usr/bin/env bash
# Removes m4r from the game folder: the files the release zip put there (m4r/files.txt), the plans and OptiScaler's log.
# The cache (~/.cache/m4r: MetalFX files and plans for other games) stays unless --cache is given.
# SPDX-License-Identifier: GPL-3.0-only
set -euo pipefail
CACHE_TOO=0
[[ "${1:-}" != --cache ]] || { CACHE_TOO=1; shift; }
GAME=$(cd -- "${1:-$(dirname -- "${BASH_SOURCE[0]}")}" && pwd)
LIST="$GAME/m4r/files.txt"
[[ -f "$LIST" ]] || { echo "uninstall.sh: no m4r/files.txt in $GAME; nothing removed" >&2; exit 1; }
while read -r f; do
  [[ -n "$f" && "$f" != /* && "$f" != *..* ]] || continue
  rm -f -- "$GAME/$f"
done < "$LIST"
rm -rf -- "$GAME/m4r"
rm -f -- "$GAME/OptiScaler.log"
echo "m4r removed from $GAME. Remove WINEDLLOVERRIDES=\"dxgi=n,b\" from the game's Steam launch options."
if (( CACHE_TOO )); then
  rm -rf -- "${M4R_CACHE_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/m4r}"
  echo 'Cache removed.'
fi
