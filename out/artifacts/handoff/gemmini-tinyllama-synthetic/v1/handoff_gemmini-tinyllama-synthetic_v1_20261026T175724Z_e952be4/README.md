# Local synthetic TinyLlama tuning inputs

**Diagnostic only — fails selected EL4 protocol conformance.** See the
[input audit](../../../../../../examples/gemmini/comparisons/tvm/EL4_INPUT_AUDIT.md).
The source changes calibration, recipe transforms/planes, cache ABI, host math
and reference producer beyond the authorized synthetic input substitution.
Do not enqueue these as selected-profile campaigns or promote their schedule
records to the selected E2E model. No new numerical profile or paid run is
selected here. The code, frozen data and ELFs remain reproducible diagnostics.

The user explicitly authorized local synthetic inputs for this experiment.
This is a separate numerical identity from the team's v00/WikiText release.
Real weights come from TinyLlama 1.1B Chat revision
`fe8a4ea1ffedaf415f4da2f062534de366a451e6`; checkpoint SHA256:
`6e6001da2106d4757498752a021df6c2bdc332c650aae4bae6b0c004dcf14933`.

**Do not implement the proposed optimization knobs.** Use existing TVM/Gemmini
contraction DSE with graph fusion fixed at 2. No new host scheduling, producer
placement, panelized packing, hardware capability or silent CPU matmul fallback.
Existing permitted CPU work stays explicit. Unsupported hardware stays unsupported.

## Prepared and checked

- Three synthetic 64-token calibration sequences, seed 20261026. A separate
  64-token measurement sequence uses seed 20261027. The actual IDs are frozen
  in `inputs.json`; no shared team input or calibration input is required.
- Weights use signed INT8, zero point0, static scales shared by 16 output channels.
  Activations use one frozen scale per tensor, range -128..127. RNE, saturation,
  INT32 accumulation; this checkpoint has no linear biases.
- All 22 layers are retained. The source session executes 32-token prefill plus 32
  teacher-forced decode calls, cache 64, all 99 output arrays. Cache bytes are INT8,
  shape [22,2,4,64,64]; logits are FP32 [1,1,32000]. Candidate state is carried.
- Both full prefill/decode exports were generated locally and imported by the
  existing TVM frontend: **1,563 INT8 x INT8 -> INT32 matmuls in each phase**.
  The saved99 arrays were reproduced exactly after source export compatibility
  adjustments. No TVM source or hardware change was made.
- Fourteen small layer0 chunks cover prefill/decode QKV, O, connected MLP,
  full-output down, QK, PV and connected attention. Full N/K dimensions remain;
  all 32 query heads and 4 KV heads are present in connected attention.
- All 14 independent chunk checks and all14 LLVM checks have **zero error**.
  There are 18 structurally distinct contraction workloads. The three initial
  campaigns own 9/18; that is workload coverage, not schedule-space coverage.
- The connected decode MLP and attention baseline ELFs passed positive and
  wrong-oracle Spike controls. Their matmul functions all use Gemmini, with no
  CPU matmul fallback or forbidden hardware-loop instructions.

These checks do not establish a new FireSim result or WikiText quality.
`verification.json` records the exact boundary. The older reported full E2E
result at TVM c50c4e2 remains evidence for its original program, not this new
synthetic/quantized source identity.

## Restore without recalibrating

```bash
export DELIVERY="$MERLIN/out/artifacts/handoff/gemmini-tinyllama-synthetic/v1/latest"
cd "$DELIVERY"
sha256sum -c SHA256SUMS
mkdir inputs
tar -xzf synthetic-inputs.tar.gz -C inputs
export INPUTS="$DELIVERY/inputs"
(cd "$INPUTS" && sha256sum -c PAYLOAD_SHA256SUMS)
```

The 21,826,168-byte archive contains frozen calibration, token IDs, all 99
reference arrays, real activation/cache boundaries, first/middle/last decode
contexts, the 14 independent fixtures, producer-source snapshots and host/task
checks. Full checkpoint weights and the two approximately 1.3 GiB PT2 exports
are not committed. The generator reproduces exports from the pinned checkpoint
and frozen scales. The actual model config is committed as `config.json`.

Set `CHECKPOINT` to the matching local safetensors file. Reuse its existing
model cache; the transfer manifest gives the exact SHA. Use the pinned TVM
fork and preparation environment: Python 3.12, Torch 2.10.0, NumPy 1.26.4,
safetensors, the existing TVM 0.19 source/native library and CPU XGBoost 2.1.4
for DSE. No Transformers loader is needed.

```bash
export PYTHONPATH="$MERLIN/examples/gemmini/comparisons/tvm:$MERLIN/src:$TVM/python:$TVM/apps/gemmini:$PYTHONPATH"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TVM_NUM_THREADS=1
"$PY" "$MERLIN/examples/gemmini/comparisons/tvm/prepare_synthetic_tinyllama.py" \
  --checkpoint "$CHECKPOINT" --config "$DELIVERY/config.json" \
  --replay-bundle "$INPUTS" --output-dir "$DELIVERY/generated-chunks" \
  --skip-full-export --threads 2
"$PY" "$DELIVERY/check_chunks.py" --input-dir "$DELIVERY/generated-chunks" \
  --output-dir "$DELIVERY/rechecked"
```

Replay copies each frozen fixture unchanged. It does not recalibrate or replace
golden outputs with another host's calculation. It verifies source agreement
using the existing float tolerance and exact integers. The initial producer
used independent NumPy INT64 contractions and floor/parity ties-to-even encoding.
Do not sweep the seed, calibration or scales as performance knobs.

For a completely fresh local experiment, omit `--replay-bundle`. That mode
generates/calibrates a new source release and computes new references; it is not
the default remote handoff path.

## Agreed DSE and ELF controls

The directly committed `ELFS/tinyllama-decode-mlp.elf` and
`ELFS/tinyllama-decode-attention.elf` require only the existing ELF launcher.
They are default connected-chunk controls, not full-model or tuned guests.
DDR3 is expected and already configured by the bitstream. No hardware changes
or F2 specification is assumed.

The original selected-profile TinyLlama study rows remain pending. A local
`compose_study.py` preparation helper is retained as unqualified evidence and
is not published as an execution entry point. Do not use it to replace those
rows with this diagnostic source. The selected study still has one shared
four-hour maximum and proposal cap 96; no extra campaign is added here.

Before paid DSE, qualify live measured-window pruning and remote cancellation.
Abort at active_cycles >= the same workload's incumbent, tighten in flight and
exclude startup/init/warmup. Total-cycle caps or completed-result filtering do
not satisfy this. An unavailable observer is a reported limitation; do not
invent capability or modify the bitstream to hide it.

## Diagnostic full-source reconstruction

The following reconstructs this diagnostic source, not the selected EL4 model.
It does not authorize a paid E2E run or waive the input audit's blockers.

Run the generator again in a fresh output directory **without**
`--skip-full-export` to emit both full PT2 files as well as the chunks.
Replay uses the stored session input/reference arrays; it need not re-run
calibration. Compile baseline/tuned stage packages using the existing
`compile_package.py`, graph level 2, the same frozen source/host-math policy,
and only freshly admitted structurally matching tuning records.

After compiling the two packages:

```bash
"$PY" "$DELIVERY/make_session_inputs.py" --inputs "$INPUTS" \
  --generated "$FULL_GENERATED" --prefill-package "$PREFILL_PACKAGE" \
  --decode-package "$DECODE_PACKAGE" --output-dir "$DELIVERY/session-inputs"
```

This writes `stages.json`, `route.json`, `fixture.npz` and source-bound
`fixture-provenance.json` for the existing `build_package_session.py`.
The route feeds actual prior cache/position outputs to each decode; only tokens
come from input fixtures. All99 expected arrays are retained.
Use the existing session builder, actual qualified loader/address map and
positive/wrong-oracle controls before FireSim. Do not guess an F2 memory span
or claim that frontend intake proves the complete target package fits/runs.

Compare new-source baseline/tuned complete windows. Never compare a sum of
operator minima to E2E or attribute a changed numerical release to scheduling.
Full FireSim execution, complete target session qualification and matched
WikiText-2 test PPL <= 1.10x FP32 remain NOT_RUN.
