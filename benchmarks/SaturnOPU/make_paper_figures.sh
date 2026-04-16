#!/usr/bin/env bash
# Regenerate Saturn OPU model-analysis CSVs and paper figures.

set -euo pipefail

BENCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$BENCH_DIR/../.." && pwd)"
RUN=(conda run -n merlin-dev uv run)

cd "$REPO_ROOT"

REFRESH=()
if [ "${1:-}" = "--refresh" ]; then
	REFRESH=(--refresh)
fi

echo "[1/3] analyze_model_paths.py"
"${RUN[@]}" python "$BENCH_DIR/analyze_model_paths.py" "${REFRESH[@]}"

echo
echo "[2/3] plot_optimization_journey.py"
"${RUN[@]}" python "$BENCH_DIR/plot_optimization_journey.py"

echo
echo "[3/3] plot_model_decomposition.py"
"${RUN[@]}" python "$BENCH_DIR/plot_model_decomposition.py"

echo
echo "Generated Saturn OPU artifacts:"
ls -la "$BENCH_DIR"/model_dispatch_decomposition.csv \
	"$BENCH_DIR"/model_layer_decomposition.csv \
	"$BENCH_DIR"/opu_path_opcode_summary.csv \
	"$BENCH_DIR"/per_model_summary.csv \
	"$BENCH_DIR"/figures/
