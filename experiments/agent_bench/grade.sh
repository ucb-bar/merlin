#!/usr/bin/env bash
# Grade a submission package against PUBLIC or HIDDEN examples via oot_runner.
# Usage: grade.sh <submission_dir> <public|hidden|g0|g1|g2> [spike|verilator]
# Agents use this against PUBLIC only to iterate; HIDDEN scoring is operator-only.
set -euo pipefail
REPO=/scratch/agustin/projects/oscar-merlin
SUB="${1:?usage: grade.sh <submission_dir> <public|hidden|g0|g1|g2> [spike|verilator]}"
SET="${2:-public}"
SIM="${3:-spike}"

case "$SET" in
  public) NAMES="g0_matmul g1_relu g2_acc_scale"; BASE="$REPO/bench_contract/examples";;
  hidden) NAMES="h0_matmul h1_relu h2_acc_scale"; BASE="$REPO/experiments/agent_bench/hidden";;
  g0)     NAMES="g0_matmul"; BASE="$REPO/bench_contract/examples";;
  g1)     NAMES="g1_relu";   BASE="$REPO/bench_contract/examples";;
  g2)     NAMES="g2_acc_scale"; BASE="$REPO/bench_contract/examples";;
  *) echo "unknown set: $SET" >&2; exit 2;;
esac

RUNS="$(readlink -f "$SUB")/../grade_runs_${SET}"
rc=0
for n in $NAMES; do
  echo "### grading $n ($SIM)"
  "$REPO/.venv/bin/python" -m merlin.targetgen.oot_runner \
    --package "$SUB" --input "$BASE/$n.interface.mlir" \
    --run-id "${SET}_${n}_${SIM}" --simulator "$SIM" --runs-root "$RUNS" || rc=1
done
exit $rc
