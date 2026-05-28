#!/usr/bin/env bash
# Re-run the SaturnOPU kernel embed pipeline on the synthetic test inputs and
# copy curated phase MLIR files into per-kernel snapshot directories. This
# mirrors the IREE canonical sample at
# `third_party/iree_bar/samples/custom_dispatch/cpu/embedded/`, which keeps
# `example_transform.mlir`, `example_stream.mlir`, and `example_hal.mlir` as
# reference artifacts showing the same custom-dispatch flow at successive
# IR levels.
#
# Usage:
#   ./refresh_phase_dumps.sh        # refresh both kernels
#   ./refresh_phase_dumps.sh add    # only add_f32
#   ./refresh_phase_dumps.sh linear # only linear_f32
#
# Output (one subdir per kernel):
#   benchmarks/SaturnOPU/kernels/phase_dumps/<kernel>/
#     0_input.mlir              — user MLIR (the synthetic op to be matched)
#     1_transform_spec.mlir     — auto-generated kernels_cache/transform_spec.mlir
#     2_after_preprocessing.mlir — phase 3 (flow.dispatch rewrite landed)
#     3_flow.mlir                — phase 6 (dispatch outlined; util.call into wrapper)
#     4_stream.mlir              — phase 7 (stream-dialect form; mirror of IREE's example_stream.mlir)
#     5_hal.mlir                 — phase 11 (hal-dialect form; mirror of IREE's example_hal.mlir)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
DUMP_ROOT="$REPO_ROOT/benchmarks/SaturnOPU/kernels/phase_dumps"
BUILD_ROOT="$REPO_ROOT/build/saturnopu_phase_dumps"

run_kernel() {
  local key="$1"
  local input_mlir="$2"
  local extra_args="${3:-}"
  local out_dir="$DUMP_ROOT/$key"
  local build_dir="$BUILD_ROOT/$key"

  rm -rf "$build_dir"
  echo ">>> compiling $key from $input_mlir"
  # shellcheck disable=SC2086 # extra_args is intentionally word-split
  (cd "$REPO_ROOT" && ./merlin compile "$input_mlir" \
    --target spacemit_x60 --hw RVV \
    --kernels-dir benchmarks/SaturnOPU/kernels \
    --dump-phases \
    --output-dir "$build_dir" \
    $extra_args) >/dev/null

  mkdir -p "$out_dir"
  local stem
  stem=$(basename "${input_mlir%.mlir}")
  # File stem in /phases/ uses dot-stripped name (`dronet.q.int8` -> `dronet_q_int8`).
  local phase_stem
  phase_stem=${stem//./_}
  cp "$REPO_ROOT/$input_mlir"                                      "$out_dir/0_input.mlir"
  cp "$build_dir/kernels_cache/transform_spec.mlir"                "$out_dir/1_transform_spec.mlir"
  cp "$build_dir/phases/${phase_stem}.3.preprocessing.mlir"        "$out_dir/2_after_preprocessing.mlir"
  cp "$build_dir/phases/${phase_stem}.6.flow.mlir"                 "$out_dir/3_flow.mlir"
  cp "$build_dir/phases/${phase_stem}.7.stream.mlir"               "$out_dir/4_stream.mlir"
  cp "$build_dir/phases/${phase_stem}.11.hal.mlir"                 "$out_dir/5_hal.mlir"
  # Coverage summary: for each kernel matcher, count occurrences in flow phase.
  {
    echo "Source: $input_mlir"
    echo "Compiled: $build_dir/$(basename "${input_mlir%.mlir}").vmfb"
    echo
    echo "--- Kernel call sites in flow phase (one line per actual rewrite) ---"
    grep -oE "util\.call @call_[a-zA-Z0-9_]+" "$out_dir/3_flow.mlir" \
      | sort | uniq -c | sort -rn || true
    echo
    echo "--- Unmatched linalg ops (still in dispatch-creation) ---"
    grep -oE "linalg\.(generic|matmul|fill|reduce|map|transpose|broadcast|conv_[a-z0-9_]+|softmax)" \
      "$build_dir/phases/${phase_stem}.5.dispatch-creation.mlir" 2>/dev/null \
      | sort | uniq -c | sort -rn || true
  } > "$out_dir/COVERAGE.txt"
  echo "    snapshots updated under $out_dir"
  echo "    coverage:"
  sed 's/^/      /' "$out_dir/COVERAGE.txt"
}

case "${1:-all}" in
  add)
    run_kernel add_f32    "tests/granularity/fixtures/embed_pipeline/add_input.mlir"
    ;;
  linear)
    run_kernel linear_f32 "tests/granularity/fixtures/embed_pipeline/matmul_input.mlir"
    ;;
  dronet)
    # Disable data tiling so linalg.matmul stays a plain named op (encoding
    # ops would otherwise wrap matmuls and the match.mlir pattern wouldn't
    # apply). Use the f32 variant — the i8 variant needs i8 kernels.
    run_kernel dronet_partial \
      "models/dronet/dronet.mlir" \
      "--iree-compile-arg=--iree-opt-data-tiling=false"
    ;;
  all|"")
    run_kernel add_f32    "tests/granularity/fixtures/embed_pipeline/add_input.mlir"
    run_kernel linear_f32 "tests/granularity/fixtures/embed_pipeline/matmul_input.mlir"
    run_kernel dronet_partial \
      "models/dronet/dronet.mlir" \
      "--iree-compile-arg=--iree-opt-data-tiling=false"
    ;;
  *)
    echo "usage: $0 [add|linear|dronet|all]" >&2
    exit 1
    ;;
esac
