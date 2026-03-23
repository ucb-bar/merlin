#!/usr/bin/env bash
# Copyright 2026 UCB-BAR
#
# Show semantic diff between forked cuda_tile files and upstream CUDA HAL.
# Re-applies the mechanical rename to upstream, then diffs against our version.
# Only real semantic changes are shown.
#
# Usage: ./tools/diff_cuda_fork.sh [file_name]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

SRC_DIR="$REPO_ROOT/third_party/iree_bar/runtime/src/iree/hal/drivers/cuda"
DST_DIR="$REPO_ROOT/runtime/src/iree/hal/drivers/cuda_tile"

apply_renames() {
  sed \
    -e 's/iree_hal_cuda_nccl_/iree_hal_cuda_tile_nccl_/g' \
    -e 's/IREE_HAL_CUDA_NCCL_/IREE_HAL_CUDA_TILE_NCCL_/g' \
    -e 's/iree_hal_cuda_/iree_hal_cuda_tile_/g' \
    -e 's/IREE_HAL_CUDA_/IREE_HAL_CUDA_TILE_/g' \
    -e 's|iree/hal/drivers/cuda/|iree/hal/drivers/cuda_tile/|g' \
    -e 's/IREE_HAL_DRIVERS_CUDA_/IREE_HAL_DRIVERS_CUDA_TILE_/g'
}

# File mapping (same as fork_cuda_hal.sh).
declare -A FILE_MAP=(
  ["cuda_driver.c"]="cuda_tile_driver.c"
  ["cuda_device.c"]="cuda_tile_device.c"
  ["cuda_device.h"]="cuda_tile_device.h"
  ["stream_command_buffer.c"]="cuda_tile_stream_command_buffer.c"
  ["stream_command_buffer.h"]="cuda_tile_stream_command_buffer.h"
  ["cuda_allocator.c"]="cuda_tile_allocator.c"
  ["cuda_allocator.h"]="cuda_tile_allocator.h"
  ["cuda_buffer.c"]="cuda_tile_buffer.c"
  ["cuda_buffer.h"]="cuda_tile_buffer.h"
  ["nop_executable_cache.c"]="cuda_tile_nop_executable_cache.c"
  ["nop_executable_cache.h"]="cuda_tile_nop_executable_cache.h"
  ["event_semaphore.c"]="cuda_tile_event_semaphore.c"
  ["event_semaphore.h"]="cuda_tile_event_semaphore.h"
  ["event_pool.c"]="cuda_tile_event_pool.c"
  ["event_pool.h"]="cuda_tile_event_pool.h"
  ["timepoint_pool.c"]="cuda_tile_timepoint_pool.c"
  ["timepoint_pool.h"]="cuda_tile_timepoint_pool.h"
  ["cuda_dynamic_symbols.c"]="cuda_tile_dynamic_symbols.c"
  ["cuda_dynamic_symbols.h"]="cuda_tile_dynamic_symbols.h"
  ["cuda_dynamic_symbol_table.h"]="cuda_tile_dynamic_symbol_table.h"
  ["cuda_status_util.c"]="cuda_tile_status_util.c"
  ["cuda_status_util.h"]="cuda_tile_status_util.h"
  ["cuda_headers.h"]="cuda_tile_headers.h"
  ["api.h"]="api.h"
  ["registration/driver_module.c"]="registration/driver_module.c"
  ["registration/driver_module.h"]="registration/driver_module.h"
)

total_changes=0
for src_name in "${!FILE_MAP[@]}"; do
  dst_name="${FILE_MAP[$src_name]}"
  src_path="$SRC_DIR/$src_name"
  dst_path="$DST_DIR/$dst_name"

  # Filter to specific file if requested.
  if [[ -n "${1:-}" ]] && [[ "$dst_name" != *"$1"* ]]; then
    continue
  fi

  if [[ ! -f "$src_path" ]] || [[ ! -f "$dst_path" ]]; then
    continue
  fi

  # Apply renames to upstream, diff against our version.
  renamed=$(apply_renames < "$src_path")
  diff_output=$(diff -u --label "upstream+rename/$dst_name" \
    --label "cuda_tile/$dst_name" \
    <(echo "$renamed") "$dst_path" 2>/dev/null || true)

  if [[ -n "$diff_output" ]]; then
    echo "=== $dst_name ==="
    echo "$diff_output"
    echo ""
    total_changes=$((total_changes + 1))
  fi
done

echo "--- $total_changes file(s) with semantic changes ---"
