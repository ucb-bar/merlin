#!/usr/bin/env bash
# Copyright 2026 UCB-BAR
#
# Fork CUDA HAL driver files for cuda_tile, applying mechanical renames.
# Usage: ./tools/fork_cuda_hal.sh [--dry-run]
#
# Copies core CUDA HAL files and renames all identifiers from
# iree_hal_cuda_ to iree_hal_cuda_tile_. Preserves existing custom files
# (native_executable.c/h) in the target directory.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

SRC_DIR="$REPO_ROOT/third_party/iree_bar/runtime/src/iree/hal/drivers/cuda"
DST_DIR="$REPO_ROOT/runtime/src/iree/hal/drivers/cuda_tile"
DRY_RUN=false

if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=true
  echo "=== DRY RUN ==="
fi

# Files to copy (relative to SRC_DIR).
# Format: source_name -> target_name
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

# Files to preserve (not overwritten by the fork).
PRESERVE=(
  "native_executable.c"
  "native_executable.h"
  "CMakeLists.txt"
)

apply_renames() {
  # Apply all mechanical renames to the content of a file.
  local content="$1"

  # Order matters: longest prefixes first to avoid partial matches.
  # Identifier renames.
  content="${content//iree_hal_cuda_nccl_/iree_hal_cuda_tile_nccl_}"
  content="${content//IREE_HAL_CUDA_NCCL_/IREE_HAL_CUDA_TILE_NCCL_}"
  content="${content//iree_hal_cuda_/iree_hal_cuda_tile_}"
  content="${content//IREE_HAL_CUDA_/IREE_HAL_CUDA_TILE_}"

  # Include path renames.
  content="${content//iree\/hal\/drivers\/cuda\//iree\/hal\/drivers\/cuda_tile\/}"

  # Header guard renames: IREE_HAL_DRIVERS_CUDA_ -> IREE_HAL_DRIVERS_CUDA_TILE_
  content="${content//IREE_HAL_DRIVERS_CUDA_/IREE_HAL_DRIVERS_CUDA_TILE_}"

  echo "$content"
}

SRC_COMMIT="$(cd "$SRC_DIR" && git rev-parse --short HEAD)"

echo "Forking CUDA HAL from: $SRC_DIR"
echo "Source commit: $SRC_COMMIT"
echo "Target dir:    $DST_DIR"
echo ""

mkdir -p "$DST_DIR/registration"

FORKED_FILES=()

for src_name in "${!FILE_MAP[@]}"; do
  dst_name="${FILE_MAP[$src_name]}"
  src_path="$SRC_DIR/$src_name"
  dst_path="$DST_DIR/$dst_name"

  # Check if target is a preserved file.
  base_dst="$(basename "$dst_name")"
  skip=false
  for p in "${PRESERVE[@]}"; do
    if [[ "$base_dst" == "$p" && "$dst_name" == "$p" ]]; then
      skip=true
      break
    fi
  done
  if $skip; then
    echo "SKIP (preserved): $dst_name"
    continue
  fi

  if [[ ! -f "$src_path" ]]; then
    echo "WARN: source not found: $src_name"
    continue
  fi

  echo "FORK: $src_name -> $dst_name"
  FORKED_FILES+=("$src_name -> $dst_name")

  if ! $DRY_RUN; then
    content="$(cat "$src_path")"
    renamed="$(apply_renames "$content")"
    echo "$renamed" > "$dst_path"
  fi
done

# Generate FORK_MANIFEST.md
if ! $DRY_RUN; then
  cat > "$DST_DIR/FORK_MANIFEST.md" << MANIFEST_EOF
# CUDA Tile HAL Driver — Fork Manifest

Forked from IREE CUDA HAL driver.

- **Source**: \`third_party/iree_bar/runtime/src/iree/hal/drivers/cuda/\`
- **Source commit**: \`$SRC_COMMIT\`
- **Fork date**: $(date -u +%Y-%m-%d)
- **Fork script**: \`tools/fork_cuda_hal.sh\`

## Forked Files

| Source (cuda/) | Target (cuda_tile/) |
|---|---|
$(for f in "${FORKED_FILES[@]}"; do
  src="${f%% -> *}"
  dst="${f##* -> }"
  echo "| \`$src\` | \`$dst\` |"
done)

## Custom Files (not forked)

| File | Description |
|---|---|
| \`native_executable.c\` | CTL1 FlatBuffer reader (custom) |
| \`native_executable.h\` | CTL1 executable API (custom) |
| \`CMakeLists.txt\` | Build configuration (custom) |

## Skipped Files (see FUTURE.md)

- \`graph_command_buffer.c/h\` — CUDA graph capture
- \`nccl_channel.c/h\` + \`nccl_dynamic_symbols.c/h\` — Multi-GPU collectives
- \`memory_pools.c/h\` — Async memory pool allocation
MANIFEST_EOF
fi

echo ""
echo "Done. Forked ${#FORKED_FILES[@]} files."
echo "Run 'tools/diff_cuda_fork.sh' to see semantic changes after manual patches."
