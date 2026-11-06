#!/usr/bin/env bash
# Build the Verilator simulator of Iris (IrisNoTraceConfig = IrisConfig minus its Tacit trace fragment)
# from a full chipyard iris-cy tree at pin chipyard_iris.
#
#   build_sim.sh [--tree D] [--jobs N] [--threads N] [--conda-env D]
#
#   --tree       the full iris-cy tree to build in (default out/build/ext/chipyard_iris_rtl). Created when
#                missing: cloned, checked out at the pin, and only the submodules IrisConfig elaborates are
#                fetched. Never the setup's $MERLIN_CHIPYARD_IRIS, which is a sparse checkout for reading
#                facts and is not modified.
#   --jobs       make -j (default 16); the build runs under nice -n 10
#   --threads    VERILATOR_THREADS of the simulator (default 4)
#   --conda-env  a chipyard conda env with verilator, java, sbt, firtool and riscv-tools (libfesvr)
#                (default $MERLIN_CHIPYARD/.conda-env)
#
# The pinned tree does not compile as it is; iris-cy-b4542fa.patch (beside this script) is applied:
#   * IrisConfigs.scala: the Tacit trace fragment names chipyard.WithTacitParallelEncoder, which is defined
#     nowhere, and tacit's chipyard glue needs a `traceParams` field the pinned Shuttle tile lacks, so the
#     trace encoder and sinks are left out (generators/tacit is not initialised) and IrisNoTraceConfig
#     names the result. Cores, vector units, VME, caches, NoC, memory and serial-TL are IrisConfig's.
#   * ChipletConfigs.scala (unrelated chiplet examples): wrap translationParams in Some(...) to match the
#     pinned testchipip.
# Measured: about 11 min on a shared 48-core host at -j16 (sbt ~2, elaboration + firtool ~3, Verilator ~6).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
PY="${PY:-$REPO/.venv/bin/python}"
TREE="$REPO/out/build/ext/chipyard_iris_rtl"
JOBS=16
THREADS=4
CONDA_ENV="${MERLIN_CHIPYARD:+$MERLIN_CHIPYARD/.conda-env}"
CONFIG=IrisNoTraceConfig
PATCH="$HERE/iris-cy-b4542fa.patch"
SUBMODULES=(
  generators/rocket-chip generators/saturn generators/shuttle generators/testchipip
  generators/rocket-chip-blocks generators/rocket-chip-inclusive-cache generators/boom generators/pmu
  generators/icenet generators/constellation generators/bar-fetchers generators/rerocc generators/hardfloat
  generators/diplomacy tools/cde tools/dsptools tools/fixedpoint tools/rocket-dsp-utils tools/firrtl2
  tools/DRAMSim2 sims/firesim
)
while [ $# -gt 0 ]; do
  case "$1" in
    --tree)      TREE="$2"; shift ;;
    --jobs)      JOBS="$2"; shift ;;
    --threads)   THREADS="$2"; shift ;;
    --conda-env) CONDA_ENV="$2"; shift ;;
    *) sed -n '2,24p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 2 ;;
  esac
  shift
done

read -r COMMIT URL BRANCH < <(cd "$REPO" && "$PY" -c "
from merlin.common.provenance import pin
p = pin('chipyard_iris'); print(p.commit, p.repo_canonical, p.branch)")
if [ ! -d "$TREE/.git" ] && [ ! -f "$TREE/.git" ]; then
  echo "== fetching chipyard $BRANCH @ ${COMMIT:0:12} into $TREE"
  git clone --filter=blob:none --no-checkout "$URL" "$TREE"
  git -C "$TREE" checkout -B "$BRANCH" "$COMMIT"
fi
HEAD="$(git -C "$TREE" rev-parse HEAD)"
if [ "$HEAD" != "$COMMIT" ]; then
  echo "error: $TREE is at ${HEAD:0:12}, pin chipyard_iris is ${COMMIT:0:12}; the patch is for the pin" >&2
  exit 1
fi
if git -C "$TREE" apply --reverse --check "$PATCH" 2>/dev/null; then
  echo "== $(basename "$PATCH") already applied"
else
  git -C "$TREE" apply "$PATCH"
fi
git -C "$TREE" submodule update --init --jobs 8 "${SUBMODULES[@]}"
# A full clone would drag in tacit too; its chipyard glue does not compile against this tree (see above).
if [ -e "$TREE/generators/tacit/.git" ]; then git -C "$TREE" submodule deinit -f generators/tacit; fi

if [ -z "$CONDA_ENV" ] || [ ! -x "$CONDA_ENV/bin/verilator" ]; then
  echo "error: need a chipyard conda env with verilator (--conda-env, or \$MERLIN_CHIPYARD/.conda-env)" >&2
  exit 1
fi
if ! command -v conda >/dev/null; then echo "error: conda is not on PATH (chipyard's env needs activating)" >&2; exit 1; fi
eval "$(conda shell.bash hook)"
set +u; conda activate "$CONDA_ENV"; set -u   # its activate.d scripts set JAVA_HOME and the compilers
export RISCV="$CONDA_ENV/riscv-tools" PATH="$CONDA_ENV/riscv-tools/bin:$PATH"
started=$(date +%s)
nice -n 10 make -C "$TREE/sims/verilator" -j"$JOBS" CONFIG="$CONFIG" VERILATOR_THREADS="$THREADS" \
  JAVA_HEAP_SIZE="${JAVA_HEAP_SIZE:-24G}"
SIM="$TREE/sims/verilator/simulator-chipyard.harness-$CONFIG"
echo "built $SIM in $(( ($(date +%s) - started) / 60 )) min"
echo "run.sh rtl-sim run finds it there; elsewhere: export MERLIN_IRIS_VERILATOR=$SIM"
