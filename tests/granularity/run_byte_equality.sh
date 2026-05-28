#!/usr/bin/env bash
# Byte-equality test driver for kernel embedding.
#
# Usage:
#   tests/granularity/run_byte_equality.sh <baseline.vmfb> <embedded.vmfb> [--input <spec>...]
#
# Runs both VMFBs through iree-run-module with the same inputs, captures
# stdout (the standard shaped-buffer output format), and compares md5
# checksums. Exit 0 on match, 1 on mismatch.

set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 <baseline.vmfb> <embedded.vmfb> [--input <spec>...]" >&2
  exit 2
fi

BASELINE="$1"; shift
EMBEDDED="$1"; shift

IREE_RUN_MODULE="${IREE_RUN_MODULE:-build/host-vanilla-release/tools/iree-run-module}"
DEVICE_FLAG="${DEVICE_FLAG:---device=local-task}"

if [[ ! -x "$IREE_RUN_MODULE" ]]; then
  echo "error: iree-run-module not found at $IREE_RUN_MODULE" >&2
  echo "  set IREE_RUN_MODULE=<path> or build host-vanilla-release first" >&2
  exit 2
fi

OUT_BASELINE="$(mktemp)"
OUT_EMBEDDED="$(mktemp)"
trap 'rm -f "$OUT_BASELINE" "$OUT_EMBEDDED"' EXIT

"$IREE_RUN_MODULE" --module="$BASELINE" "$DEVICE_FLAG" "$@" > "$OUT_BASELINE"
"$IREE_RUN_MODULE" --module="$EMBEDDED" "$DEVICE_FLAG" "$@" > "$OUT_EMBEDDED"

BASELINE_MD5="$(md5sum < "$OUT_BASELINE" | awk '{print $1}')"
EMBEDDED_MD5="$(md5sum < "$OUT_EMBEDDED" | awk '{print $1}')"

echo "baseline md5: $BASELINE_MD5"
echo "embedded md5: $EMBEDDED_MD5"

if [[ "$BASELINE_MD5" == "$EMBEDDED_MD5" ]]; then
  echo "[OK] outputs are byte-equal"
  exit 0
fi
echo "[FAIL] output mismatch"
diff -u "$OUT_BASELINE" "$OUT_EMBEDDED" | head -50 >&2
exit 1
