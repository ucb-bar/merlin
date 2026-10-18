# Bounded EL4 Phase 2 exploration

**Stopped 2026-10-18 21:18 UTC:** the user clarified that optimization must run
through the automatic sandboxed Phase 2 experiment. This directory preserves
the earlier manual exploration and its audit; it is **not** a launched Phase 2
campaign. The measurement/model entrypoints refuse further manual jobs while
the recorded stop exists. No candidate is promoted.

The task now includes `shared-scheduling` at `608976eaa`. Its source release and
isolated retained-policy integration are preparation infrastructure; no automatic
accelerator campaign has been launched. The original compiler remains unchanged.
Starting Phase 2 from an older compiler does not require another Phase 1 authoring
run. Adding a newer library requires an explicit Phase 2 release and baseline
revalidation; the ordinary inherited-profile route still needs that connection.
See [the integration guide](../../../../docs/guides/shared_pass_phase2.md).

The operator-only automatic selection is in this task's
`inputs/phase2/experiment.yaml`, using the existing `measured_claims` adapter.
It remains `kind: template`, with unavailable owner inputs explicitly pending.
The original functional admission check refused because the selected frozen
run has no `qa_loop_summary.yaml`; its manifest also says
`formal_grade_complete: false`. The new performance declarations are not an
admitted materialized corpus. Do not manufacture either handoff or a timing
certificate. The existing automatic runner owns authoring, sandboxing, brokered
feedback, immutable candidate jobs and selection once admission succeeds.

This example is a task-local operator adapter over Merlin's existing corpus,
compiler, reference, linker, admission, simulator, roofline and archive APIs.
Target schedules live in the companion gemmini-mlir candidate packages.

The original immutable budget starts at **2026-10-18 19:57:49.333 UTC** and ends four
hours later. Final verification reserves 75 minutes. This example deliberately
does not create a formal incumbent, qualify Spike timing, change numerical
contracts, or seal the provisional Phase 1 corpus.

Inspect preserved evidence from the Merlin task checkout, using the existing Python environment:

```sh
export TMPDIR=/work/merlin/tasks/phase2-el4-explore-20261018/.tmp
export PYTHONDONTWRITEBYTECODE=1
P2PY=/work/merlin/tasks/phase1-harness-v2/merlin/.venv/bin/python
P2DRIVER=examples/gemmini/phase2/el4_20261018
$P2PY $P2DRIVER/report.py --watch
```

Generated records use the task's `out/artifacts/perf-studies/gemmini/el4-20261018`
and `out/build/phase2-campaign` roots. `common.py status` prints active jobs and
the fixed deadline; `LIVE.log` refreshes every 30 seconds during native work.
Only CPUs 6 and 7 are admitted through the existing shared CPU/gSIM slot roots.

## Selecting capsules

`audit.py` joins the pinned hardware facts, compiler-supported operations and
explicit numeric/readout contracts with **each** Phase 0 deferred declaration
and the selected full models' source-derived accelerator groups. The output is
`capsule-manifest.json`, `coverage-matrix.json` and `full-model-groups.json`.
Identity, source digest, derived geometry, operation requirements, pressure,
floor availability, baseline/refusal and evaluation-cost availability are
recorded per member. Unsupported members remain in the manifest.

The 183 deferred performance declarations come from a failed parent generation.
They remain declarations, not an admitted cohort. Five cheap functional guards
are a separate cohort. The older performance policy came from small development
applications and cannot establish full-model connected-layout coverage.
Exact shape witnesses do not prove GQA, KV reuse, readout or connected-boundary
equivalence. The coverage matrix preserves these gaps explicitly. `stress.py`
adds independent signed-input capacity/tail/live-residency checks; these do not
change the frozen performance objective or its declared 0.8 attainment target.

The audit and stress materializers can be rerun as follows. Preserve the original
manifest and its digest before regenerating; do not replace a search objective.

```sh
$P2PY $P2DRIVER/audit.py
$P2PY $P2DRIVER/stress.py
```

## Measurements and candidates

Historical invocation (now blocked by the recorded user stop):

```sh
$P2PY $P2DRIVER/evaluate.py \
  --members layers/G_matmul_m17k16n32_raw \
  --candidate ../gemmini-mlir/phase2_candidates/rhs_panel_stream/package \
  --label rhs_panel_stream --engines spike,gsim --repeats 3 --repeat-offset 100
```

The driver retains compiler bytes, source/recipe/input/ELF hashes, actual CPU
grants, simulator identities, complete output comparisons, failures and costs.
The linked harness reads `mcycle` immediately around kernel entry and completion;
the completion fence belongs to that window. This includes issue and waits.
It does not measure array busy time or use simulator walltime as target cycles.
Packing/layout/transfers inside the kernel are included; full-model host compute
is a separate boundary. Untuned Spike counters are functional diagnostics only.

`selected_engine_full_values_v2` uses the selected engine's existing full-value
readback policy (coherent gSIM dump, serial Spike readback). Historical
`legacy_serial_b64_v1` measurements use a different linked harness. Reports keep
these profiles separate. Repeated cycles across simulator restarts establish
repeatability in this domain, not a hardware distribution or cache-warmth claim.
`protocol.py` inspects actual linked counter windows and instruction sites and
pins selected Rocket/Gemmini source evidence. Static command sites are not
dynamic traffic or accelerator utilization counters.

Rooflines use existing staged-bound derivation and worst-repeat priorities.
Unknown host/issue/overlap terms remain unknown; partial floors cannot establish
roofline convergence. Only correct comparable measured results support retention.

## Calibration and final model regression

The current offline path is `calibrate_offline.py`. It calls the existing
archive inspect/prepare/attach, design-check, fit and validation APIs and launches
no simulator. Version 2 bindings preserve each repeat's original counter-review
receipt, retain all declared cases and keep missing reviews unavailable. The
selected archive yields 483 labels, 313 execution cases in 87 source groups,
473 incomplete native reports and ten missing reports. The measured held design
has 41 separated pairs across three families; native prediction coverage remains
zero. No empirical fit or native timing qualification has been earned.

`timing_config.py` connects the existing resource-mapping implementation to the
selected facts and verifies cache geometry against the selected DTS/L2 JSON.
It writes a separate partial device profile and native-model configuration.
It derives load resources `(capacity=2, queue=6)`, store resources `(2, 2)`,
32 KiB eight-way L1I/L1D and 512 KiB eight-way L2, all with 64-byte lines.
Other resource fields and unpriced costs remain unknown. Spike's existing LRU
and coherence policy are stated approximations, not RTL protocol equivalence.
The receipt pins the gSIM memory adapter: `mm_magic_t`, not a FireSim DRAM bridge.
The original observations still bind the original count-only model. They cannot
be relabeled as predictions from this new configuration.

Both commands retain existing outputs and require a fresh directory on replay:

```sh
$P2PY $P2DRIVER/calibrate_offline.py --output-dir "$TMPDIR/offline-intake-review"
$P2PY $P2DRIVER/timing_config.py --output-dir "$TMPDIR/spike-config-review"
```

Completing the native model needs reviewed CPU/execute/config resource mapping,
remaining provider/observer semantics, price fitting, saved-ELF Spike replay and
the original held-out gates. This does not require recapturing the models or
rerunning Phase 1 compiler authoring. No new gSIM run is part of these commands.

`calibrate.py` verifies the original public archive and attempts the documented
v3 intake route without inventing missing evidence or dropping repeats. Its
status and capacity-validation plan are artifacts. No prediction is authorized
until the existing held-out gates and domain tests pass. In this campaign the
original untimed reports and incomplete counter reviews prevent qualification.

`model_regression.py` recompiles only the saved device interfaces, verifies the
logical command buffers, and reuses the original host objects/weights read-only.
It relinks each bound input through `model_check.full_build`, runs the complete
selected chain on functional Spike, and compares **every output byte** against
the saved compiled baseline chain. Full values stay operator-local.

```sh
$P2PY $P2DRIVER/model_regression.py --model tiny_llama \
  --candidate ../gemmini-mlir/phase2_candidates/rhs_panel_stream/package
$P2PY $P2DRIVER/model_regression.py --model smolvla \
  --candidate ../gemmini-mlir/phase2_candidates/rhs_panel_stream/package
```

This is a compiler schedule regression check. It does not close TinyLlama's
pending 32/32 quantized-reference comparison, SmolVLA's unresolved internal
divergence, or the selected ResNet readout admission. The Phase 1 owner retains
those dependencies, public capture/readout closure and formal qualification.

Final accounting uses the latest cumulative author usage once, records missing
interrupted/operator/helper usage explicitly, and separates compilation,
simulation, calibration and verification costs without summing overlapping
walltimes. No push or merge is part of this example.
