#!/usr/bin/env bash
# Stage a CLEAN baseline sandbox containing ONLY public materials. Does NOT launch an agent.
# After this, launch your agent runtime with cwd = the sandbox and TASK.md as the prompt.
set -euo pipefail
REPO=/scratch/agustin/projects/oscar-merlin
WS="${1:-/scratch/agustin/agent_bench/baseline_ws}"
CHIPYARD="${MERLIN_CHIPYARD:-/scratch2/agustin/chipyard}"
GEM="$CHIPYARD/generators/gemmini/software/libgemmini"

rm -rf "$WS"; mkdir -p "$WS/submission" "$WS/docs"
cp -r "$REPO/bench_contract" "$WS/bench_contract"                       # public contract + examples
cp "$REPO/experiments/agent_bench/TASK_baseline.md" "$WS/TASK.md"
cp "$REPO/experiments/agent_bench/grade.sh" "$WS/grade.sh"; chmod +x "$WS/grade.sh"

# public docs: Gemmini ISA headers + the MLIR out-of-tree template
for h in gemmini.h gemmini_params.h; do
  [ -f "$GEM/$h" ] && cp "$GEM/$h" "$WS/docs/" || echo "WARN: $GEM/$h not found" >&2
done
cp -r "$REPO/third_party/llvm-project/mlir/examples/standalone" "$WS/docs/standalone_template" 2>/dev/null \
  || echo "WARN: standalone template not found" >&2

# explicitly NOT copied: the Merlin source tree, generated_targets/ reference packages, hidden/.
echo "baseline sandbox staged at: $WS"
echo "contents:"; ls -1 "$WS"
echo
echo "LAUNCH (operator): point your agent runtime at this dir with TASK.md as the prompt, e.g."
echo "  (cwd=$WS) prompt=TASK.md  -- the agent writes into submission/ and iterates via ./grade.sh"
