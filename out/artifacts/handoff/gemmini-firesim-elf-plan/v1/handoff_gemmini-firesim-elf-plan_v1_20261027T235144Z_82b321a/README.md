# FireSim handoff

Start with [the current handoff instructions](../../../../../../examples/gemmini/comparisons/tvm/FIRESIM_HANDOFF.md).
Use the Merlin commit containing this publication; record its `git rev-parse HEAD`.
The checked-in Merlin scripts already include the protocol guard, ELF builder,
graph options and finish-active-runs mode. Do not apply an old Merlin patch over them.

For TVM, use grammy117/tvm `gemmini/device-ops` at
42b3c1ef27abe878f930e48b335b24956beb0e56 in an isolated checkout. Review and apply
`tvm-device-ops.patch` with `git apply --check` followed by `git apply`. This patch
contains the audited default schedule/source-binding fixes; it is not a native
TVM binary. Use a matching TVM/LLVM build and the recorded target adapter.

The approved tuning allowances are four hours for ResNet and one hour each for
TinyLlama/SmolVLA, across each model's chunks and graph plans. Stop admitting new
work at the limit and finish active measurements. Record extra completion time;
do not start queued jobs during it or renew the tuning allowance. Final E2E is
separate. The remote worker owns the deployment, aggregate occupancy ledger,
FireSim-only validation and explicit per-run hang watchdog.

TinyLlama tunes the frozen 16/1 row probes and runs E2E at 32/32. Begin at published
fusion=2/prepacking/bias preload/CSE-off, then search existing graph options within
the same budgets. The sole model protocol is 8ec99308; its linked numerics has only
the approved 128 -> 32 prefill correction. Calibration remains 128 prefill and
1024 decode samples. FP quality is report-only for functional_v1; all declared
compiled-versus-quantized arrays remain strictly checked.

## Contents and checks

- `firesim-plan.json`, `user-decisions.json`, `WORKLOAD_ROSTER.md`: current scope,
  36 ready chunks, 47 initial workloads, 26 owners and the excluded diagnostic.
- `baseline-build-commands.json`: build-only argv templates for all 36 chunks.
  Restore the committed gemmini-shared-model-chunks publication first; those
  payloads are not duplicated here. Substitute worker paths as argv, not eval.
- `example-elf/`: one newly linked SmolVLA action-input chunk, positive and
  wrong-oracle ELFs with memory/instruction checks. Neither ELF was executed.
- `run-contract/`, correction patch and protocol decisions: frozen selected
  protocol and all linked checker inputs. The checker passes 3 recipes, 46 calls,
  133 arrays. These counts do not establish model capture or execution.
- `tinyllama-seed/`: checked input/calibration/range seed and model2MLIR source
  bundle for full 32+32 preparation; full checkpoint/exports/references are absent.
- `audit-evidence/`: earlier software-spec and Spike observations under their
  original identities, including the unavailable spec-pinned oracle limitation.
- `publication-tests-final.log`: 94 focused tests passed. `drain-final.log`:
  36 affected tests passed again after final lock-release cleanup.
- `validation-summary.json`: readiness and the nine inherited module-size failures.

Verify this delivery with `sha256sum -c SHA256SUMS` from this directory. This
publication keeps the previous chunk delivery and historical execution evidence
unchanged. The existing local FireSim tuner still has sequential Spike preflight;
the remote worker must use its FireSim-only adapter and aggregate budget meter.
The finish-active mode is reusable, not a claim that deployment integration ran.

## Complete model ELFs

See the handoff for the exact build-only connected-chunk command and complete
session linker interface. One chunk pair is built here; the other 35 have verified
inputs/build commands. Complete TinyLlama/SmolVLA selected releases and all 99/33
reference arrays remain preparation work. Published probes do not substitute for
full attention/cache or prefix/recurrent-state coverage.

For TinyLlama set PROTOCOL to this package's
`run-contract/examples/gemmini/protocol/experiment.yaml`. Initialize MERLIN_M2M_DIR
from `tinyllama-seed/source/model2mlir.bundle` at
5fec4a2dc106668e4e74605eeaa54729df36e1b4; use a qualified capture interpreter.
On the allocated CPU worker run prepare_remote_tinyllama.py with
`--phase bootstrap --protocol "$PROTOCOL" --seed-dir "$PACKAGE/tinyllama-seed"`,
then prefill, decode and references phases with the same protocol and the printed
`--release` directory. Prefer the verified checkpoint via TINYLLAMA_CHECKPOINT_FILE;
bootstrap otherwise downloads it. Preserve all source, input and scale hashes.
