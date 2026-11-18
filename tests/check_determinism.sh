#!/usr/bin/env bash
# Runs the BBR plan several times on the same input and checks that every output frame is bit-identical between runs.
# (A kernel whose result depends on timing or on what ran before it fails this; see docs/mxu.md, "LDS slots".)
# usage: tests/check_determinism.sh <plan dir> <fixture dir> [frames=8] [runs=3]
# SPDX-License-Identifier: GPL-3.0-only
set -euo pipefail
if [[ $# -lt 2 ]]; then
  sed -n 5p "$0" >&2
  exit 2
fi
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PLAN="$1"; FIX="$2"; FRAMES="${3:-8}"; RUNS="${4:-3}"
first=""
for ((r = 1; r <= RUNS; r++)); do
  "$ROOT/build/run_bbr" "$PLAN" "$FIX" "$FRAMES" -1 >/dev/null
  sum="$(cd "$PLAN" && md5sum out-*.rgba16f | awk '{print $1}' | md5sum | awk '{print $1}')"
  echo "run $r: $sum"
  if [[ -z "$first" ]]; then first="$sum"; elif [[ "$sum" != "$first" ]]; then echo "NOT DETERMINISTIC" >&2; exit 1; fi
done
echo "PASS: $RUNS runs, $FRAMES frames each, identical"
