# Bounded EL4 Phase 2 exploration

This directory retains the original manual experiment drivers and their historical
status below. Current automatic source development uses the ordinary sandboxed
author and retained-revision queue described in
[capsule-qualified source development](../../../../docs/guides/phase2_capsule_source_development.md).
That explicit diagnostic route preserves incomplete full-model and timing gates;
it does not admit a scored performance campaign. Its new runs have separate clocks
and evidence, and do not reopen the stopped manual experiment.

## Automatic development at 2026-10-20 19:05 UTC

The current bounded run is `phase2-loop-20261020T190502Z` under this task's
`out/artifacts/perf-studies/gemmini/el4-20261018`. Its persisted deadline is
23:05:02 UTC; new target jobs stop by 22:20:02 UTC, preserving 45 minutes for
verification. CPU admission uses the shared slot pool on CPUs 24–31, at most
eight CPUs across operator and helpers. Phase 1 trees and both frozen compiler
packages remain unchanged. This run uses Spike and existing gSIM evidence;
it launches no new gSIM jobs or prohibited hardware-loop instructions.

From the task root, `.tmp/phase2-loop-status-current` prints current status.
The run retains `status.json`, `LIVE.md` and a 30-second heartbeat log.
`parallel-work-assignments.json` records operator/helper responsibilities.

The original retained-revision queue completed an unchanged control, complete
outputs, fresh Spike observations, parent feedback validation and cleanup.
Both arms matched all 544 values and produced 125 count features. The retained
RHS candidate `85f47996…` changed the actual LLVM and ELF while preserving the
logical command buffer, and
its native numerical screen also matched all 544 values. Its V1 source checker
refused a first nonaccumulating accumulator write. That refusal remains intact;
the candidate is unpromoted. Both jobs completed feedback and UNKNOWN timing
decisions; the enclosing diagnostic exited nonzero because candidate source
qualification did not pass.

The separately selected accumulator-overwrite V2 checks are documented in
gemmini-mlir's `docs/v26/accumulator_first_product_overwrite.md`. V1 remains the
default. The new effect requires original live selected/arithmetic owners,
checked RTL/receiver write semantics and complete product/readout induction.
Host controls and original metadata construction pass. Fresh paired target
execution remains a separate requirement; these checks do not establish overlap,
completion latency or timing qualification.

The development ledger keeps all 243 declarations, 90 coverage obligations,
original 183-declaration lineage and every refusal. This run added original
writer evidence for guards 54, 20 and 21. Under a separate source epoch containing
only the lossless direct-output renderer change, objective 175
(`M=64, K=256, N=64`) also passed original source/native/writer checks: all 4,096
INT32 outputs and 1,048,576 products, with no host answer stores. Its complete
trace is 63,994,343 bytes, below the unchanged 64 MiB ceiling. Original runtime,
headers, logical output type, references and numerical policy are unchanged;
actual completed values select the existing checked two-byte wire encoding.
This is no timing claim or sealed performance cohort. Earlier guards are not
silently recertified under the new renderer epoch.

Archive audit now accounts for all 788 count reports (454 unique executions).
Separately, 33 retained manual public ELFs passed fresh complete numerics and
counts, joining 84 historical gSIM records and 58,308 output values. All 33 have
one flush; their TRAIN activated-readout count reaches only six. Across the
Phase 1 archives, TRAIN activated readouts reach 32. These facts do not repair
the current flush-free V26 timing gap or the held readout-33 support failure.
Existing source-group splits and all failed intake attempts remain recorded.
Model `dc413b…` retains its passing archived domain; current unsupported kernels,
capacity and attention/KV regimes remain UNKNOWN. Neither the model nor its
qualification gates were changed to make a compiler candidate win.

The metadata profile and indexed inventory check are host-cost evidence only.
Indexing lexical ancestry recomputes its sets on every validation; it preserves
fresh source/hash, alias, ownership and qualification checks. The integrated
runtime/adapter/inventory controls passed 221 focused host tests. Complete
roofline attainment remains UNKNOWN, and no incumbent or plateau is established.

## Historical manual experiment

**Stopped 2026-10-18 21:18 UTC:** the user clarified that optimization must run
through the automatic sandboxed Phase 2 experiment. This directory preserves
the earlier manual exploration and its audit; it is **not** a launched Phase 2
campaign. The measurement/model entrypoints refuse further manual jobs while
the recorded stop exists. No candidate is promoted.

At that checkpoint the task included `shared-scheduling` at `608976eaa`. Its source release and
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

## Infrastructure continuation after the closed campaign

The original measured campaign remains closed. Infrastructure work does not
restart optimization or promote a compiler; immutable Phase 1 inputs are retained.

## Structural model connection

`timing_config.py` checks the pinned facts, original observer device profile,
elaborated cache metadata and gSIM memory-source receipt. It writes a separate
native model with only supported resource mappings and cache geometry. Version 2
also derives `reset_accumulator_activation=0` from the selected FIRRTL's
`StoreController` register reset, using the generic `merlin_timing.capacities`
API, and copies the resulting hardware facts into the actual native model.
It leaves the independent, uninitialized execute activation register alone.

```sh
python examples/gemmini/phase2/el4_20261018/timing_config.py \
  --output-dir "$TASK_DIR/out/build/phase2-campaign/spike-structural-config-v2"
```

Use the task's existing Python and selected package paths. The output directory
must be fresh. No simulator is launched by this command. The selected gSIM
binary uses `mm_magic_t`, with AXI backpressure and response queuing; its labels
do not belong to a FireSim DRAM bridge or DRAMSim domain.

Prepare an independently retained replay request with the generic tool:

```sh
spike-timing archive-replay-plan --plan observation-plan.json \
  --model-artifact spike-structural-config-v2/model.json \
  --spike-executable /selected/annotated-spike \
  --provider-library /selected/libgemmini.so \
  --provider-library /selected/libtiming_provider.so --output replay.json
```

The existing Phase 1 evidence module
`merlin_experiments.phase1.annotated_spike_evidence` supplies `review_elf` and
`observe` for explicitly selected, independently pinned runtime invocations.
Replay only public archived ELFs, under shared CPU admission, with their exact
input and counter-window bindings. Never change an old report's model digest.
The generic `archive-attach` command consumes fresh reports and the separately
saved `observation_plan` nested in the replay request.

A bounded infrastructure replay on public `G_matmul_m1k16n16_raw` executed the
same archived ELF under three model files. All runs exited successfully and
produced byte-identical consoles. The structural model removed missing load and
store resource diagnostics; adding the derived reset removed the unconfigured
accumulator activation diagnostic, reducing semantic-unknown events from one to
zero. No independent numerical comparison was repeated, and native event timing
remained incomplete. This is an exercised model connection, with no latency or
compiler-performance claim.

Remaining native inputs include CPU issue/execution resources, execute/config
resource semantics, instruction/memory/target prices, initial-work drain timing,
and ISA/counter-taint eligibility. Observer implementation changes require an
editable managed simulator repository; the preserved Phase 1 `.tmp` source is
read-only to this task.
