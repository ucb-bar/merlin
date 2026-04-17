#!/usr/bin/env bash
set -euo pipefail

BUDDY_OPT="${BUDDY_OPT:-$HOME/work/buddy-mlir/build/bin/buddy-opt}"
IN="$1"
OUTDIR="${2:-experiments/gemmini/logs}"
mkdir -p "$OUTDIR"

base="$(basename "$IN")"
LOG="$OUTDIR/${base}.print-after-all.mlir"

"$BUDDY_OPT" "$IN" \
  --convert-linalg-to-gemmini \
  --lower-gemmini \
  --mlir-print-ir-after-all \
  2> "$LOG" > /dev/null

echo "Wrote: $LOG"
grep -n "gemmini\\." "$LOG" | head -n 20 || true
