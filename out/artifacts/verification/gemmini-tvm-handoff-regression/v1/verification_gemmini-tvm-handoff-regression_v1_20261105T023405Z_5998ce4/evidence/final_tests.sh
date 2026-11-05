#!/bin/bash
# Final regression suites for the TVM handoff; logs go into the verification product.
set -u
cd /scratch/grammy/blast/merlin-tvm-tuning
source /scratch/grammy/blast/tvm-tuning-env.sh
V=$PWD/out/artifacts/verification/gemmini-tvm-handoff-regression/v1/verification_gemmini-tvm-handoff-regression_v1_20261105T023405Z_5998ce4
PIN=$BASELINE/tvm-f2-pin-f3cfd1a7d
TIP=$BASELINE/tvm-direct-intrinsics
C=$MERLIN_DELIVERY/examples/gemmini/comparisons/tvm
export TVM_NUM_THREADS=1

run() {  # name, cwd, tvm-root, pytest args...
  local name=$1 cwd=$2 root=$3; shift 3
  ( cd "$cwd" && env PYTHONPATH="$MERLIN_DELIVERY/src:$root/python:$root/apps/gemmini:$C" TVM_SOURCE="$root" \
      timeout 1800 nice -n 10 "$PY" -m pytest -q -rf -p no:cacheprovider -n 8 -W ignore "$@" 2>&1 \
      | grep -v 'Warning: ' > "$V/$name.log"
    echo "exit ${PIPESTATUS[0]}" >> "$V/$name.log" )
}

true \
  merlin/tests/gemmini/test_tvm_firesim*.py merlin/tests/gemmini/test_tvm_kernel_benchmarks.py \
  merlin/tests/gemmini/test_tvm_shared_*.py merlin/tests/gemmini/test_tvm_exported_chunks.py \
  merlin/tests/gemmini/test_tvm_resnet_*.py merlin/tests/gemmini/test_tvm_target.py \
  merlin/tests/gemmini/test_tvm_package_session.py merlin/tests/gemmini/test_tvm_prepare_accumulator_bias.py \
  merlin/tests/gemmini/test_verify_firesim_payload.py
true \
  merlin/tests/gemmini/test_tvm_*.py merlin/tests/gemmini/test_verify_firesim_payload.py \
  merlin/tests/gemmini/test_tinyllama_session.py merlin/tests/gemmini/test_smolvla_session.py \
  merlin/tests/gemmini/test_prepare_firesim_queue_view.py
run tvm-tip-gemmini-tests "$TIP" "$TIP" $(cd "$TIP" && ls tests/python/relax/test_*gemmini*.py)
run tvm-pin-gemmini-tests "$PIN" "$PIN" $(cd "$PIN" && ls tests/python/relax/test_*gemmini*.py)

for f in merlin-f2-path-tests merlin-tvm-all-tests tvm-tip-gemmini-tests tvm-pin-gemmini-tests; do
  echo "$f: $(grep -E '[0-9]+ (passed|failed)' "$V/$f.log" | tail -1) | $(tail -1 "$V/$f.log")"
done
