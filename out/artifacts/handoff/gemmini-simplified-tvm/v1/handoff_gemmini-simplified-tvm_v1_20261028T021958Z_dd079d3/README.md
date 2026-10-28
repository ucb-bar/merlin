# Simplified TVM source, operator checks and FireSim preparation

This is a portable **preparation release**, not a FireSim performance result or
full-model qualification. Use the current repository's
[FIRESIM_HANDOFF.md](../../../../../../examples/gemmini/comparisons/tvm/FIRESIM_HANDOFF.md)
and [adapter contract](../../../../../../examples/gemmini/comparisons/tvm/ADAPTER_CONTRACT.md).

## Restore compiler and inputs

Set RELEASE to this directory and TVM to a separate checkout of the user TVM fork.
From that checkout, these commands preserve the existing branch:

```sh
git fetch https://github.com/grammy117/tvm.git gemmini/device-ops
git bundle verify "$RELEASE/source/tvm.bundle"
git fetch "$RELEASE/source/tvm.bundle" refs/heads/gemmini/direct-intrinsics
git worktree add --detach "$TVM" 619c7c202ff1fe133eceb8fddf3cc9081a6c7e8a
```

The bundle needs base 42b3c1ef27abe878f930e48b335b24956beb0e56. The full patch
independently reconstructs tree 6e7f980eae5a062b4b5416daa0623c3bbae8b983 from that base;
it is an alternative to the bundle, not an additional patch to apply afterward.
Build/load TVM using its apps/gemmini README. No C++ source changed in this series;
the local test library's exact hash is recorded, not distributed as a portable binary.

Restore the **already tracked** shared inputs using
`out/artifacts/handoff/gemmini-shared-model-chunks/v1/latest/restore.py` and its README.
Use the restored directory as INPUTS. Keep its programs, weights/scales, inputs,
expected outputs and checksums unchanged. Older packed graph plans are superseded.

```sh
export PYTHONPATH="$TVM/python:$TVM/apps/gemmini:$MERLIN/examples/gemmini/comparisons/tvm"
python3 "$RELEASE/reextract_chunks.py" --input-root "$INPUTS" --output-dir "$NEW_INDEX"
```

Choose a new NEW_INDEX directory. Bind the remote native compiler and Python source
hashes; local absolute paths in receipts are historical evidence, not remote paths.
Compare the 47 full structural workloads, not just short hash strings. Set each
campaign program/fixture path relative to INPUTS when building the remote study.
Use the existing prepare/compile commands in FIRESIM_HANDOFF.md with prepacking
and bias folding disabled. Rebuild all model ELFs with this compiler before tuning.
Do not use the earlier example ELF as a new-compiler model ELF.

## Contents and verified scope

- Source bundle, exact patch and commit list for TVM 619c7c202.
- 71 unchanged mathematical fixtures (51 baseline plus 20 device extensions),
  their generated source/package, valid and wrong-oracle RV64 ELFs, and diagnostic
  Spike receipts. Valid exit 0; wrong oracle exit 22; neither ELF has prohibited
  instruction hits. Each receipt binds 496 source/tool/package entries.
- Fresh extraction of 36 chunks, 47 distinct contractions, 26 owners. No old
  workload key matches. Per-model counts are in index/summary.json.
- Focused test logs and validation-summary.json. Suite totals overlap; do not add
  them into one claimed unique-test count. Source patch reconstruction was checked.

For an independent local rebuild, run `prepare_operator_checks.py --help`, supply
an independently bound adapter receipt and RV64 GCC, then use TVM's
`verify_tensorir.py run --bundle NEW_BUNDLE --backend spike --simulator-receipt RECEIPT`.
Run the same command with `--negative` for the wrong-reference control. This local
check is not a mandatory sequential preflight for remote tuning candidates.

## Remote run policy

Keep stock MetaSchedule EvolutionarySearch/XGB. Three real successful baseline
jobs calibrate each workload; then freeze each job's timeout at admission:
`min(ceiling, max(3*baseline_median, 3*median(latest nine correct uncached jobs)))`.
Before nine candidates, use the baseline median. The pilot supplies explicit
bootstrap and ceiling; a ceiling below the baseline floor refuses search.
Completed slow measurements train the cost model. Timeout is an error, never a
made-up cycle label. Collect warm target cycles and convert to seconds only at
the TVM runner boundary using the documented target counter frequency.

Charge actual occupied slots, including calibration/failures, against ResNet4h,
TinyLlama1h, SmolVLA1h. Close admissions at budget; finish active jobs under their
frozen watchdogs and report overrun. Final baseline/tuned E2E is separate:
ResNet timeout1h, TinyLlama/SmolVLA6h each. Tune Tiny16/1; final chain32+32/cache64.
Protocol8ec99308 is authoritative; FP quality is report-only under that protocol.

There are 63/36/42 baseline jobs before search. At an illustrative one minute
per job, at most 177/24/18 further jobs remain, before connected confirmation.
Use the pilot occupancy to allocate actual trials, not these illustrative counts.
Search stock fusion/CSE only after the published fusion2/CSEoff starting point,
within the same budget; prepack/bias transforms remain removed.

## Required before paid model execution

The remote agent qualifies fixed-warm cycles, wrong-reference rejection, actual
wall-time cancellation and safe slot reuse under adapter v4. No live active-cycle
observer is required. Unconfirmed remote termination quarantines the slot.
Admit only exact declared/proved software-spec paths with selected hardware facts,
source/provider proofs and emitted-program bounds. A matcher or these small tests
does not discharge those obligations. Missing spec-pinned oracle provenance is
still explicit: diagnostic Spike is a repaired local RNE build, not pinned fce5cae.

Complete connected-chunk checks and the full selected model builders before final
E2E. Tiny16/1 probes do not cover attention/cache/LM-head32+32; Smol capsules do not
establish its entire prefix/expert/flow session. Never replace carried candidate
state with reference state or sum isolated minima into E2E cycles. No E2E cycles
are supplied by this release. Nine inherited Merlin module-size gate failures are
source-maintenance failures; they are separate from the passing runtime tests.
