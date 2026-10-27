# FireSim ELF preparation and tuning plan

This handoff extends `comparisons/tvm-workload-tuning` base
**82b321a89a8a646877484f4bd09637040ce0fa26** with the published audit, build-only
and DSE completion changes. Use the commit containing this handoff. TVM is `gemmini/device-ops` at
**42b3c1ef27abe878f930e48b335b24956beb0e56**, plus the recorded schedule/source-binding
patch. Earlier worktrees and published evidence are preserved.

Our deliverable is ELFs or exact instructions to build them after handoff.
**The remote worker owns deployment, manager configuration and physical slots.**
Do not ask the user to supply deployment details to finish this preparation.
No simulator job is started by the build commands below.

## Approved decisions

- Tune all models concurrently, provisionally one slot per model. Budgets are
  **ResNet 4 hours; TinyLlama 1 hour; SmolVLA 1 hour**, total across each model's
  chunks and graph configurations. Charge actual tuning slot occupancy; exclude
  CPU preparation and final E2E. The remote worker implements this ledger and
  resumes it without renewing spent time. Allocation can be revised by the user.
- **TinyLlama tuning uses the published 16/1 row probes; E2E uses 32/32.**
  Do not replace the tuning roster with newly derived 32-row chunks.
- Start with published graph settings: fusion=2, constant prepacking on,
  proved accumulator-bias preload on, CSE off. Then search the existing graph
  options within the **same** per-model budgets. No new compiler passes.
- Use stock MetaSchedule **EvolutionarySearch + XGB**, existing TensorIR schedules
  and registered TensorIntrins. Share records through exact structural equality.
- Remote candidate validation uses FireSim guest checks. **No sequential Spike
  preflight** is added before candidates or long sessions. Preserve independent
  guest wrong-oracle qualification and every-output checks.

The publication's shared-four-hour/remaining-proposal wording and fixed-fusion-only
instructions describe its original campaign; they do not override these decisions.
The full-model protocol and its numerical choices remain fixed during tuning.

## Contract and quality

The sole full-model authority is [protocol at 8ec99308](https://github.com/ucb-bar/merlin/tree/8ec99308cb1182b9b896a6075300312266fadf6f/examples/gemmini/protocol).
Use the frozen `run-contract/` in the preparation product. The user approved one
correction to its linked numerics: TinyLlama prefill 128 -> 32. Decode is 32 and
cache capacity is 64. Protocol/recipe bytes are unchanged; the correction patch
and both numerical-file identities are retained. The pinned checker passes
**3 recipes, 46 calls and 133 arrays**; that is definition consistency only.

`functional_v1` checks every declared output, carried state and padded channel
against the sealed selected quantized reference. Shape, dtype, arity and integer/
bool values are exact; floats are finite with atol=2e-6 and rtol=1e-5. Use the
pinned checker without tolerance overrides. **FP quality is report-only** under
this profile. Earlier custom PPL/top-1/cosine gates are superseded. Independent
exact-kernel and eligible-device-placement checks still apply.

The selected target is full Gemmini/Rocket U250 at 30 MHz; hardware facts, image,
clock and memory compatibility are verified by the remote worker. Local ELF link
and instruction checks do not establish execution on that deployment.

## Frozen chunks, workloads and owners

Restore the publication at
`out/artifacts/handoff/gemmini-shared-model-chunks/v1/latest/`. Its catalog SHA256
is `839fad75688a9c2a87b975842f072f217a1f0fbc0ff3ba26e9b1707600e37c2a`.
Restoration verified 171 shared payload members, the 182-member ResNet dependency,
and relocated reference proofs. All 36 ready chunks' build inputs passed identity
checks here. Published host/reference results remain attributed to the publication:
36 host comparisons pass; all 19 new saved-export references replay exactly.

| Model | Ready connected chunks | Initial distinct contractions | Initial owners | Coverage |
| --- | --- | --- | --- | --- |
| ResNet50 | 18 | 21 | 10 | Stem, 16 complete bottleneck blocks, classifier. |
| TinyLlama | 10 | 12 | 8 | Q/K/V projections and complete MLPs of widths 5728/6096, each at 16 and 1 rows. K/V share structurally matching records. |
| SmolVLA | 8 | 14 | 8 | Action input projection, time/action embedding, self QKV, self-attention core, cross QKV, cross-attention core, expert MLP, action projection/Euler update. |
| Total | 36 | 47 | 26 | Counts apply to the published graph configuration. |

A **workload** is an extracted TensorIR contraction; an **owner** supplies one
representative exact task/fixture for shared tuning. A **chunk** includes connected
host and device work and may contain several workloads. Thus 47 workloads are
neither 47 E2E models nor 47 total schedule candidates.

Keep `smolvla-action-update` excluded: its final-normalization-plus-update host
comparison fails (max absolute error 0.000223577). The smaller
`smolvla-action-projection-update` passes with normalized input at its explicit
boundary. Do not widen tolerances or rename the failing diagnostic as ready.

TinyLlama M=1 is a row-local probe, not a complete decode session. These chunks
omit full QK/PV, cache, RoPE and LM head. M=16 records do not automatically match
M=32 prefill. SmolVLA chunks include 50 action rows, 15 query heads, 163 self keys
or 113 cross keys; attention starts after RoPE. Prefix vision/VLM, all layers,
full normalization and ten-step state carry are not established by these probes.
The supplied frozen chunk weights/scales/references are sufficient to tune these
chunks; their source policy files remain provenance, not a replacement E2E protocol.

## Knobs: searchable versus fixed

| Layer | Existing choices | Boundary |
| --- | --- | --- |
| Graph preparation | Fusion 0/1/2/3; constant prepacking on/off; bias preload on/off; CSE on/off | Published configuration first, then legal distinct alternatives. 32 combinations before deduplication; not 32 mandatory FireSim sweeps. |
| Intrinsic geometry | Shape-derived legal M/N/K rectangles and interior/tail or K-padding alternatives | Hardware dimensions are 1–16, but current sketch generation uses selected divisor/tail templates, not every arbitrary triple. |
| Reuse/loop order | Sampled M/N/K outer/reuse factors; `ij`/`ji` traversal | Stock `sample_perfect_tile` and mutators; reuse groups bounded at 64 tiles and checked against actual capacity. |
| Dataflow | Primitive OS or WS | Preload/compute commands; hardware loop-FSM instructions remain forbidden. |
| Movement placement | Operand hoisting, tile/root staging, legal two-slot software pipeline | Capacity, buffer lifetimes and host hazards must remain valid. |
| Epilogue/loops | Tile/root epilogue placement, existing unroll choices (0/16/64/512; compact space 0/16) | Only legal existing schedule alternatives; no new CPU optimization search. |

Physical SRAM addresses, allocation correctness, hazard fences, final completion,
config shadowing and compatible adjacent-DMA coalescing are fixed lowering rules.
They are not additional free optimizer dimensions. Tiling/reuse/staging decisions
are exposed through ordinary TensorIR schedules and MetaSchedule traces. General
cross-function SRAM residency, arbitrary DMA coalescing search and new host packing
passes are not implemented. The FireSim campaign extracts contractions; it does
not separately tune every secondary movement/reduction/pool task. Their actual
cost remains in connected-chunk and E2E timing.

Use `apps/gemmini/audit_graph_plans.py` to inspect/deduplicate existing graph plans
before measuring them. Re-extract tasks after changing graph settings: published
47-workload keys and 26-owner assignments cannot simply be reused for a different
graph. Record hits/misses and fresh measurements for the exact task, target and
measurement window. Reuse traces, not incompatible old cycle labels.

## Build connected-chunk ELFs

On the remote CPU worker, select the recorded patched Merlin/TVM sources and
matching TVM/LLVM environment. Set PY, TVM_SOURCE, RISCV_GCC and ADAPTER_RECEIPT to
its installed paths; COMPARISON is this directory. Use bounded compiler workers.
The new entry point below uses the existing `compile_package.py` and
`chunk_measure.build_chunk`; it adds no compiler passes and invokes no simulator.

```sh
export DELIVERY="$MERLIN/out/artifacts/handoff/gemmini-shared-model-chunks/v1/latest"
# WORK is a new directory under the worker's authorized artifact/build root.
python3 "$DELIVERY/restore.py" --output-dir "$WORK/shared-chunks"
export INPUTS="$WORK/shared-chunks"
export COMPARISON="$MERLIN/examples/gemmini/comparisons/tvm"
export PYTHONPATH="$MERLIN/src:$TVM_SOURCE/python:$PYTHONPATH"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TVM_NUM_THREADS=1

"$PY" "$COMPARISON/build_shared_chunk_elf.py" \
  --tvm-source "$TVM_SOURCE" --restored-root "$INPUTS" \
  --chunk smolvla-action-input-projection \
  --output-dir "$WORK/baseline/smolvla-action-input-projection" \
  --riscv-gcc "$RISCV_GCC" --adapter-receipt "$ADAPTER_RECEIPT" \
  --host-math-policy portable_neural_f32 --graph-fusion-level 2
```

`baseline-build-commands.json` in the preparation product enumerates this command
for **all 36 ready chunks**, with their selected host math: ResNet `native`,
TinyLlama `portable_exp_trig`, SmolVLA `portable_neural_f32`. The guest must still
validate those compiled outputs; a host reference pass is not a target pass.
Each build emits `package/`, `guest/valid.elf`, `guest/bad-oracle.elf`, instruction/
memory audits, exact reference bindings and hashes. No Spike receipt is required.
The connected harness currently links within its 256 MiB region and refuses an
oversized image; a full-model session uses the separate memory planner below.

For tuned builds use a fresh output directory, add `--tuning-database "$DATABASE"`
and the selected graph flags (`--graph-fusion-level`, `--graph-cse`,
`--no-prepack-weights`, `--no-bias-preload`). The existing package compiler uses
stock JSONDatabase/MetaScheduleApplyDatabase and rejects an explicit database with
zero structural hits. Record partially tuned coverage; do not force a mismatch.

One small build was performed here: `smolvla-action-input-projection`, both ELFs
202,576 bytes, no forbidden instructions, 494,272 mapped bytes, 16 KiB stack.
The positive ELF SHA256 is
`e16f10a9ce19c93188bf54f2404e2fa97c9a95b5599532cd63ff11928cdf94b0`.
**Neither ELF has run.** The other 35 ready chunks have verified inputs and build
commands, not newly built ELFs.

## Remote tuning sequence

1. Establish a valid guest baseline for each selected distinct task as budget
   permits. Rotate owners across chunks; prioritize measured connected cost,
   not guessed FLOP cost. Record any uncovered tasks if time runs out.
2. Run stock EvolutionarySearch/XGB over the existing schedules. Then explore
   deduplicated graph alternatives within the same budget, with new task extraction.
   No hard per-chunk minute split is assumed. One ledger per model covers all
   chunks, graph plans and candidate validation/measurement occupancy.
3. Use the remote runner's guest checks and wrong-oracle qualification. Current
   local `tune_firesim_chunks.py` still runs a Spike preflight and uses elapsed
   deadlines including CPU preparation. **Do not execute it unchanged** as this
   campaign's remote runner. Its graph-option forwarding and new
   `--finish-active-runs --active-run-timeout-seconds SECONDS` mode are reusable;
   the remote worker supplies a separate per-run hang watchdog. Admission closes
   at the budget boundary, active slot jobs can finish under that watchdog, and
   `occupancy.json` records completion time beyond the admission deadline.
   The watchdog is not a new tuning allowance. Remote occupancy accounting and
   FireSim-only validation remain the remote agent's work.
4. Stop a candidate when live active measured cycles reach the current best for
   the same exact task/input/hardware/window. Refresh incumbents in flight;
   retain failures/censored trials. Exclude startup, loading and warmup from the
   performance cycle window; include actual host work, DMA and completion.
5. Replay matching winners and confirm each promoted connected chunk, including
   those sharing owners. Connected confirmation consumes the same tuning allowance.
   Cover all 36 as budget permits; report unconfirmed chunks and retain untuned
   schedules where promotion lacks evidence. Keep baseline/tuned timing identities. Stop tuning
   at the approved budget or recorded search exhaustion. No quantitative plateau
   rule has been selected; do not label a short trial cap as convergence.
6. Assemble complete baseline and tuned model sessions and measure E2E afterward,
   outside tuning time. Do not add operator minima to manufacture E2E cycles.

Prior FASED results may suggest candidate traces, not FireSim costs: the connected
TinyLlama K/V CSE observation improved 622,531 -> 460,958 warm cycles in its own
30/30/16/16 memory setup. Keep that and older magic-memory measurements separately
attributed. Fresh FireSim measurements select winners for this deployment.

## Complete E2E preparation after handoff

The 36 chunks do not form complete TinyLlama/SmolVLA sessions. Preserve selected
checkpoints, recipes, scales and input identities; no recalibration to make a
schedule match. Full selected releases/references still need producing or binding.

- **ResNet:** use the restored parent export and independently bound selected
  input/reference, then rebuild the whole model baseline and tuned. Check one
  1000-logit array; retain all 53 convolutions and one linear, host residual/pool
  work and the selected 32 requant/ReLU plus 22 raw exits.
- **TinyLlama:** use the packaged selected seed and `prepare_remote_tinyllama.py`
  with explicit `--protocol "$PROTOCOL"`; bootstrap, prefill, decode, then
  references. Keep the selected operands2 recipe: ordinary QK has two planes per
  operand (four products). Full prefill is 32 tokens; follow with 32 teacher-forced
  decode calls, capacity 64, all 22 layers, and 99 output and state arrays. M=16/M=1
  probe records transfer only with exact structural matches. Report misses; no
  automatic retuning outside the one-hour allowance is implied.
- **SmolVLA:** bind or reproduce the selected one-camera ALOHA episode 0/frame 100,
  noise seed 1000 observation (SHA256
  `e4863c12cb7f9ae9bdf9f20ec48d6788083a81f29bb680738f420daed5405414`). Capture the
  exact selected `prefix_encode`, `flow_denoise`, `action_decode` stages and recipe,
  then use `calculate_selected_references.py --model smolvla` with the frozen
  protocol/models. Prefix + ten denoise calls + decode yields 12 calls/33 arrays.
  The unrelated `prepare_smolvla_observation.py` selects two-camera SO100/LeRobot
  0.5.1 and is unsuitable. `prepare_selected_inputs.py` still records a different
  historical processed-input hash; do not treat its output as selected without
  the exact protocol hash check. The needed selected full-session preparation is
  not a completed artifact in this chunk delivery.

For each model compile its actual stage programs with `compile_package.py`, the
selected host math and matching winning database. Construct `stages.json` as
`{"version":1,"stages":{"stage_name":"relative/package/path"}}`. Construct
`route.json` as version 1 calls with explicit fixture or prior-output edges;
carry candidate cache/actions/time between calls. Bind `input_N`/`output_N` arrays
and the exact route hash in independent fixture provenance. Never insert reference
state to replace a candidate's state. Then link through the existing interface:

```sh
"$PY" "$COMPARISON/build_package_session.py" \
  --tvm-source "$TVM_SOURCE" --stages "$STAGES" --route "$ROUTE" \
  --fixture "$SESSION_FIXTURE" --fixture-provenance "$SESSION_PROOF" \
  --output-dir "$NEW_SESSION_DIR" --riscv-gcc "$RISCV_GCC" \
  --memory-bytes 17179869184 --rtol 1e-5 --atol 2e-6 --readback
```

Here STAGES/ROUTE/SESSION_FIXTURE/SESSION_PROOF must be real completed assets;
this is not a runnable placeholder session. Use the selected 16 GiB memory map;
the existing memory planner must verify the actual weights, references, arena,
code and stack. Supply explicit weights/arena bases if its absolute-layout path
is needed. This command builds, audits and hashes positive/wrong-oracle complete
session ELFs; it does not execute them. Preserve one warm and one measured chain,
reset between repetitions, and use uninstrumented builds for headline cycles.

## Operation evidence and remaining limits

Read [TENSORIR_OP_AUDIT.md](TENSORIR_OP_AUDIT.md): exact 8ec software-spec declarations
and obligations govern support, not similar op names. The 51 baseline and 20
extension Spike fixtures passed the separately recorded diagnostic oracle, but
that is not blanket all-dtype/shape support. The spec-pinned rounding-fixed
Spike/libgemmini identity remains unavailable; retain that provenance limitation
without blocking this simulator-free ELF handoff or inventing an oracle identity.
The declaration inventory is not an implemented compiler admission gate. Bind
actual selected lowering/proof and placement/instruction evidence before claiming
full spec-qualified execution. No hardware results or E2E cycles are claimed here.

Current preparation product: [plan](../../../../out/artifacts/handoff/gemmini-firesim-elf-plan/v1/handoff_gemmini-firesim-elf-plan_v1_20261027T235144Z_82b321a/firesim-plan.json), [all chunk rows](../../../../out/artifacts/handoff/gemmini-firesim-elf-plan/v1/handoff_gemmini-firesim-elf-plan_v1_20261027T235144Z_82b321a/WORKLOAD_ROSTER.md), [36 build commands](../../../../out/artifacts/handoff/gemmini-firesim-elf-plan/v1/handoff_gemmini-firesim-elf-plan_v1_20261027T235144Z_82b321a/baseline-build-commands.json).

At the per-model tuning limit, stop admitting new work and **finish active runs**.
Do not start queued jobs during this completion period. Record the time beyond
budget separately; preserve cycle-pruning and failure/hang safeguards. This adds
no new tuning allowance and does not renew a continuation's budget.
