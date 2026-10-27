# Shared FireSim tuning chunks: all three models

Fetch `comparisons/tvm-workload-tuning` from `https://github.com/ucb-bar/merlin`.
This delivery freezes the same programs, weights, boundary inputs and expected
outputs for every runner. Restore them; do not recalibrate or regenerate inputs.
SmolVLA chunk publication/tuning is back in scope under the latest user request.
Full SmolVLA E2E admission and quality remain unfinished.

```bash
export MERLIN="$PWD"  # root of the published Merlin checkout
export DELIVERY="$MERLIN/out/artifacts/handoff/gemmini-shared-model-chunks/v1/latest"
(cd "$DELIVERY" && sha256sum -c SHA256SUMS)
python3 "$DELIVERY/restore.py" --output-dir "$DELIVERY/unpacked"
export INPUTS="$DELIVERY/unpacked"
```

Python 3.12+ is sufficient for restoration. The helper verifies the archive
parts, every payload member and the existing committed ResNet dependency,
then writes relocated proof copies. No original workspace or checkpoint
download is needed to use the chunks. Read `catalog.json`, `task-owners.json`,
`campaign-catalog.json`, `host-checks.json` and `verification.json` after restore.
All program/fixture/proof paths in the catalog are relative to `$INPUTS`.

| Model | Frozen chunks | Ready for tuning |
| --- | --- | --- |
| ResNet50 | 18: stem, all 16 complete bottleneck blocks, classifier | All 18 |
| TinyLlama | 10: Q, K and V projections plus complete MLPs of widths 5728 and 6096, each at M=16 and M=1 | All 10 |
| SmolVLA | 9: action input projection; time/action embedding; self QKV; self-attention core; cross QKV; cross-attention core; expert MLP; action projection/Euler update; diagnostic final-norm/action-update | 8; diagnostic final-norm/action-update is blocked |

TinyLlama retains full reduction/output dimensions and frozen scales. MLP
chunks include RMSNorm, gate/up, SiLU/product, down projection and residual.
The smaller row profiles are checked row-local specializations of captured
32-row source regions. **M=1 is a representative row-local probe, not an
exported complete decode session.** Attention QK/PV, cache/RoPE and LM-head
coverage are still missing from this set. Final E2E remains 32/32 with cache 64;
M=16 traces cannot be assumed to match M=32 prefill.

SmolVLA retains all 50 action rows, full feature/reduction dimensions, all 15
query heads and full 163-key self / 113-key cross context. The attention core
boundaries are after the original host RoPE work; QKV projections are separate
chunks. They retain the source's actual integer planes, softmax, output
projection and residual. This covers representative expert/flow computation,
not prefix vision/VLM or all 16 layers and ten flow steps.

All 19 newly published TinyLlama/SmolVLA references replay from their saved
Torch exports with checked independent NumPy INT64 contractions and zero
output difference. TVM host execution passes 36 of 37 chunks at the unchanged
rtol=1e-5 / atol=2e-6. `smolvla-action-update` fails its connected host check
(maximum absolute error 0.000223577); it includes final normalization. Preserve
it for diagnosis and exclude it from the ready queue. The smaller
`smolvla-action-projection-update`, with normalized input captured at its
boundary, passes. No tolerance or compiler math was changed.

## Compiler and tuning methodology

Use `https://github.com/grammy117/tvm`, branch `gemmini/device-ops`, exact commit
`42b3c1ef27abe878f930e48b335b24956beb0e56`. Both repositories are published.
The new chunk preparation adds no TVM source changes. Local replay used
Python 3.12, Torch 2.10.0 and NumPy 1.26.4. Use the existing pinned TVM/LLVM,
RISC-V toolchain and Gemmini adapter setup from the runner handoff.

1. Import each ready `program.json` and its exact `fixture.npz`. Keep graph
   fusion at **2**, constant prepacking and proved accumulator-bias preload.
   Extract the existing contraction tasks. CPU host work remains explicit.
2. Use `task-owners.json` / `campaign-catalog.json`: tune each distinct
   structurally equal TensorIR workload once, with the recorded owner and its
   workload keys. K/V and repeated blocks can share records only when this
   structural check succeeds. Shapes alone are insufficient.
3. Use the existing **stock TVM MetaSchedule EvolutionarySearch + XGB** search
   over legal schedule/tiling choices. Initial proposals provide measurements;
   the learned cost model ranks candidates as data accumulates. This is not a
   static parameter sweep. Do not implement proposed CPU scheduling, producer
   placement or input-packing knobs.
4. Use the existing remote FireSim runner and guest correctness checks. Do not
   add sequential Spike runs before remote candidates or long sessions.
   The older local `tune_firesim_chunks.py` still contains Spike preflight;
   do not invoke it unchanged as a FireSim-only runner. Have the remote runner
   retain guest `DSE_PASS`/full-output checks and a wrong-oracle qualification
   control. This publication does not replace an already qualified remote runner.
5. Batch/parallelize across actual assigned physical FireSim slots, one driver
   per slot, common slot locks and proposal ledger. Prepare/build concurrently
   within measured RAM headroom. Keep the manager alive between ELFs. Memory
   is already configured by the bitstream; **DDR3 is expected**. No slot count,
   clock, job duration or extra hardware capacity is assumed.
6. Reuse the **existing shared deadline and remaining proposal budget**, at most
   four hours. This catalog adds no budget and starts no jobs. Establish a
   baseline for each exact task/input/hardware/window. Abort a candidate once
   measured active, non-startup cycles reach/exceed its current best, refreshing
   the incumbent in flight. Exclude warmup/startup; do not substitute quantized
   driver totals. Keep timed-out candidates as censored trials, not timings.
7. Replay winner traces only through exact task/target matching. Confirm full
   connected chunks on FireSim, then baseline versus tuned E2E. Do not sum
   operator minima into an E2E claim. FireSim correctness, full selected-session
   references and model quality are separate gates; none is claimed here.

To repeat the optional independent CPU reference check after restoration:

```bash
export PYTHONPATH="$MERLIN/examples/gemmini/comparisons/tvm:$PYTHONPATH"
python3 "$MERLIN/examples/gemmini/comparisons/tvm/check_shared_references.py" \
  --root "$INPUTS/chunks" --output "$INPUTS/reference-recheck.json" --threads 2
```

This checks the 19 new chunks. ResNet's independent reference sources and
original qualification controls are in the restored `resnet-release` dependency.
Source preparation lineages retain parent hashes; full TinyLlama/SmolVLA parent
exports and checkpoints are deliberately not duplicated in this chunk release.
The frozen small exports are sufficient to recalculate these chunk outputs.
