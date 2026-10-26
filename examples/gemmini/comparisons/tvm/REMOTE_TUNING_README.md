# Remote handoff: TVM/Gemmini tuning audit

Use [FIRESIM_ROUND2_HANDOFF.md](FIRESIM_ROUND2_HANDOFF.md) for the current
portable inputs and parallel DSE launcher. Historical reproduction follows below.

New runs follow [CURRENT_RUN_POLICY.md](../../numerics/CURRENT_RUN_POLICY.md).
The October 26 raw F2 archive has now been audited; use the new
[profile-directed ResNet plan](TUNING_PLAN.md#next-resnet-dse-from-the-raw-f2-profile).
Existing TinyLlama probe timings are historical dynamic-row diagnostics.
Before more paid DSE, demonstrate [live active-cycle pruning](ADAPTER_CONTRACT.md).
The previous remote adapter's total-cycle cap/completion filter did not satisfy
that policy; its enforcement booleans must not be reused with the current pool.

Start with [BRANCH_HANDOFF.md](BRANCH_HANDOFF.md). The prepared ELFs are now
committed on the Merlin branch and run without a TVM checkout. Source/fixture
transfers are only needed for generating further tuning candidates.

## Current next-machine FireSim delivery

Follow [FIRESIM_CHUNK_TUNING.md](FIRESIM_CHUNK_TUNING.md). The run order is:
TinyLlama baseline plus **ResNet baseline and current tuned ResNet**, then
TinyLlama chunk DSE, then tuned TinyLlama E2E. Independent jobs may overlap on
available slots. Full TinyLlama ELFs must be supplied/built on the new machine.

The supplied bitstream already sets the memory model; **DDR3 is expected**.
Deliver the committed executables from `ELFS/` in
`out/artifacts/handoff/gemmini-firesim-elfs/v1/latest/` to the existing
ELF launcher. No bitstream setup or inferred F2 specifications are required.

DSE is stock TVM **EvolutionarySearch + XGBoost**, with measured feedback and
JSONDatabase, not a static sweep. Abort a candidate when its active measured
cycles reach the current best; exclude startup/loading/initialization/warmup,
and tighten limits as faster valid results arrive. Initial batches/proposal
counts are search budgets, not runtime predictions. Use as much real parallel
capacity as available within one shared four-hour maximum DSE deadline.
The earlier invented per-job caps and requested 30-cycle FireSim setup are removed.

The delivery retains full ResNet baseline/tuned/profiled guests, three short
trained-weight TinyLlama probes, independent fixtures, Spike checks and source
pins. Prefer exact remote EL4 slices for tuning and validate full-shape reuse
before E2E. The completed local Verilator campaign below is historical evidence.

## Historical local audit context

The completed audit and stock TVM follow-up used the verified **Verilator
30/30-cycle, 16/16-outstanding latency-pipe model**. Move its retained source,
inputs and evidence as historical hints; the new requested studies use FireSim
as described above. gSIM setup must not delay the handoff.

Read [TUNING_PLAN.md](TUNING_PLAN.md) for the findings and priorities,
[GENERATION.md](GENERATION.md) for compiler ownership, and
[FASED_SWEEPS.md](FASED_SWEEPS.md) for the earlier study. The latest F2 report
was supplied in chat; its raw manager receipts and original `.pt2` exports are
not in this local handoff. The newly prepared full ResNet has its own identity.

## Completed bounded local DSE policy

The current product is
`out/artifacts/perf-studies/gemmini-four-hour-dse/v1/perf-studies_gemmini-four-hour-dse_v1_20261025T235038Z_431485b/`.
It completed at **6:55 PM PDT, October 25, 2026**, before its immutable
**8:50 PM PDT** deadline, including setup. All three campaigns exited zero,
with 24 proposals each. The previous multi-day supervisors below are stopped; their `exit.json`
files are authoritative. Their last `status.json` files are historical snapshots.

Use `tvm-dse-budgeted`, its complete `tvm-source.patch`, the matching corrected
native library, and this product's final Merlin source snapshot. The new launch
uses `--bounded-dse` and the frozen `policy.json`, with 24 proposals and six
workers per selected task. `campaigns.json` scopes actual ResNet classifier,
synthetic TinyLlama decode K/V and synthetic SmolVLA action output. Remaining
tasks and full EL4 coverage are deferred. Read [TUNING_PLAN.md](TUNING_PLAN.md)
for the precise compact space and measurement policy.

`monitor-build-v3/build_receipt.json` binds a host observer linked against the
unchanged generated RTL and latency-pipe objects. `build_dse_cycle_monitor.py`
is its reproducible builder; preserve its referenced original object/archive
closure when transferring. Positive/negative controls, same-ELF cycle agreement
and a live in-flight cutoff test passed in `controls-v3/`. Earlier monitor
prototypes and a failed pre-measurement launch are retained separately; they
are not admitted measurement sources.

The guest wrapper marks the measured second call through `mscratch`. The host
reads the actual cycle CSR and a per-task atomic cutoff file. It refreshes live
bounds during simulation, so a faster validated candidate tightens other jobs.
The simulator must confirm the observer and all four memory-model parameters;
failed, dominated or pruned proposals are not valid timings. No target interrupt
or per-loop watchdog instrumentation changes the scheduled operator.

Watch the new `status.json`, each campaign's `incumbents.json`, baseline records,
`trials.jsonl`, `validated-baselines/` and `database/`. If the deadline interrupts
a batch, keep its raw completed measurements alongside the last committed stock
database. `live-campaign.json` records running identity; it is not a completed
campaign admission. Validate records and complete missing comparison/coverage
during the FireSim handoff before deployment claims. Do not rename an unfinished
manifest to bypass `tuning_library.py`'s checks.

Do not restart the stopped old products or change the current frozen deadline.
A later remote continuation needs a fresh product and explicit budget, relocated
paths, the same numerical identities and qualified hardware receipts. CSR-marker
ELFs are local measurement wrappers; full FireSim sessions use the existing
package/session exporter and their normal correctness/deployment gates.

## Previous parallel DSE launch — historical

The active local product is
`out/artifacts/perf-studies/gemmini-stock-tvm-dse/v1/perf-studies_gemmini-stock-tvm-dse_v1_20261025T222757Z_431485b/`.
It contains `start.sh`, `local-env.sh`, `launch.json`, `source-pins.json`, full
`tvm-source.patch`/`merlin-source.patch`, a toolchain JSON, exact ResNet input,
per-campaign databases/logs, host admission events and periodic status/history.
The patches include the earlier audit changes; apply them to the stated base
once. Do not apply both the earlier audit patch and the full follow-up patch.
The local TVM source is the isolated `tvm-dse-throughput` worktree at `c50c4e2`.
Its matching native library is unchanged from the corrected audit build.

`launch_dse.py` supervises external processes. TVM's stock gradient task
scheduler, EvolutionarySearch, XGB and JSONDatabase perform optimization. The
launcher contains no schedule selection or model-specific tile policy.
Default graph preparation is fixed. The real ResNet campaign extracts all
eligible tasks; TinyLlama decode and SmolVLA expert use explicitly synthetic
full-dimension operator programs because their actual selected exports are
missing locally. Never report these as full-model or F2 coverage.

After reconstructing sources and the environment below, map the receipt paths
in `TOOLCHAIN_JSON` to the remote machine and use a fresh output product:

```bash
"$PY" "$MERLIN/examples/gemmini/comparisons/tvm/launch_dse.py" \
  --tvm "$TVM" --python "$PY" --toolchain "$TOOLCHAIN_JSON" \
  --resnet "$PROGRAM" --resnet-fixture "$RESNET_INPUT_NPZ" \
  --cpus 18 --build-slots 2 --reserve-gib 4 --parallel 8 \
  --trials-per-task 32 --rtl-seconds 7200
```

`RESNET_INPUT_NPZ` is the supplied `resnet-input.npz` (`input_0` contains the
exact captured visible image). Preserve the source `visible.npz` and hashes.
The example CPU count assumes 20 available CPUs and 16 GiB RAM; adjust to leave
room for the remote host. Set OMP/OpenBLAS/MKL/TVM thread counts to one for each
worker. CPU counts describe concurrent host jobs; the guest remains one hart.

Watch `status.json` for live descendants, RSS, trial counts and valid timings;
`host-history.jsonl` retains the history. `host-pool/config.json` holds the shared
CPU roster, builder slots and memory floor. Updating it changes future
admissions, preserving running measurements. `host-pool/events.jsonl` records
wait/start/finish events. At 4 GiB available RAM new external jobs wait; below
1 GiB the supervisor terminates only its owned campaign process groups.
These are conservative admission measures, not a guarantee against unrelated
host activity exhausting memory.

Create `STOP` in the product root to stop all owned campaigns. Do not delete a
STOP marker to restart that product. For an ordinary process interruption
without an explicit stop, the same launch with `--work-dir EXISTING_PRODUCT`
reuses exact-IR database records and bound measurement caches. Preserve source,
input seed, numerical settings, binaries and memory identity. Resume may consume
a fresh proposal allowance; it is not proof of additional distinct samples.
Keep failed trial records and inspect whether failures came from build limits,
validation, cycle caps or wall caps before increasing the budget.
Stock AddToDatabase retains failed proposals with its large penalty value
(`1e10` in the first returned batch). This is not measured cycles. Use
`trials.jsonl` errors and passing `measurement.json`/ELF evidence to identify
valid timings; validate deployment records through `tuning_library.py`.

Baselines are deferred until after search. Candidate correctness remains
mandatory, and comparison claims require matching completed baselines. Inspect
distinct validated ELF hashes and several batches after XGB warmup; proposal
count alone is not search depth. The earlier one-task eight-trial command below
is a historical pilot recipe, not the active campaign configuration.

The broad ResNet campaign is delayed while generating the space for a large
fused quantization/im2col task after registering 35 workloads. The parallel
focused product is
`out/artifacts/perf-studies/gemmini-resnet-contraction-dse/v1/perf-studies_gemmini-resnet-contraction-dse_v1_20261025T224144Z_431485b/`.
Its `campaigns.json` selects all 21 contraction tasks from the same actual
extraction. TVM distributes its 672 proposals and selects schedules. The other
46 eligible tasks remain unresolved; this is not complete model tuning.
The focused supervisor joins the original pool, preserving the global limit.
Carry both products, including their failed diagnostic evidence.

To reproduce that scoped run, use the launch command above with
`--campaign-manifest "$SCOPED_PRODUCT/campaigns.json"` and
`--execution-pool "$BROAD_PRODUCT/host-pool"`. Remap the program and fixture
paths inside the campaign manifest as well as the toolchain receipt paths.
When joining an existing pool, the launcher reads its configuration without
replacing it. Each supervisor's STOP marker affects only its owned campaigns;
stop both products to stop the whole study. Do not start a second supervisor
on an already running product.

Use the focused product's full `merlin-source.patch` for the updated launcher
that supports campaign manifests and joining a pool. The broad product's
launch snapshot records the original three-campaign supervisor. A later
`merlin-final-handoff.patch` in the broad product includes the final documentation
and launcher; it replaces other Merlin patches when reconstructing that state.
Apply one complete patch to the stated base, not several overlapping patches.

## 1. Freeze and carry the sources

| Source | Revision used here | Local location |
| --- | --- | --- |
| Merlin comparison | `431485bbbd6e7d09c3544b4d6da3d9632c2667ac` + local audit/profiling changes | `$BLAST_ROOT/merlin-tvm-tuning` |
| Merlin memory-model helper | `b44479ce53f95aabe0135981d59a43db5cc650a3` | `$BLAST_ROOT/merlin-fased-memory` |
| TVM for fresh real exports | `c50c4e26992eb79ea4fc1c693d3aa16f62aaea43` + local audit/profiling changes | `merlin-tvm-gemmini/out/build/baselines/tvm-gemmini/tvm-f2-followup` |
| model2MLIR for the fresh local ResNet | `7915e23475c6db446a3c404847b11e8bc72c8a27` | `merlin-tvm-gemmini/out/build/baselines/tvm-gemmini/model2mlir-source` |

The older `tvm-workload-tuning` worktree is at `f1e411b1f`; the completed new
synthetic schedule pilot used that source and `compiler-v5`. Keep that evidence
under its original attribution. Use the corrected `c50c4e2` source and matching
native library for the fresh real-export follow-up. Do not combine their costs.

Git worktree directories contain `.git` pointers to another checkout. Copying
those directories alone is not a complete source delivery. Carry Git bundles or
clone the repositories at the exact hashes, then apply the **uncommitted** source
patches. The audit product below contains `merlin-audit.patch` and
`tvm-c50-audit.patch`, including newly added source files. Applying a patch does
not reproduce any prior hardware qualification.

For an offline transfer, create bundles before moving. Choose a fresh private
`HANDOFF` directory and use the actual source paths above:

```bash
git -C "$MERLIN_DELIVERY" bundle create "$HANDOFF/merlin.bundle" \
  HEAD feat/verilator-fased-memory-model
git -C "$TVM" bundle create "$HANDOFF/tvm.bundle" HEAD
git -C "$M2M" bundle create "$HANDOFF/model2mlir.bundle" HEAD
```

The memory-model branch is needed because the comparison branch does not
provide `mem_perturb.pinned_latency_pipe`. Keep this as an explicit dependency;
this audit does not cherry-pick or push it into a delivery branch.

On the remote machine, clone/fetch the bundles, check out the exact hashes in
separate directories/worktrees, and apply the matching patches with
`git apply --check` before `git apply`. Recreate the memory worktree separately.
Do not reuse a moving branch tip as a pin.

## 2. Carry the inputs, tools and evidence

Paths below are relative to the chosen workspace root (`$BLAST_ROOT`). Preserve complete products,
including failed trials, source snapshots, logs, manifests and checksums.

| Product | Path |
| --- | --- |
| Audit, graph equivalence results, source patches and small profiling controls | `merlin-tvm-tuning/out/artifacts/audits/gemmini-effective-dse/v1/audits_gemmini-effective-dse_v1_20261025T054556Z_431485b/` |
| Completed eight-proposal synthetic action-output study | `merlin-tvm-tuning/out/artifacts/perf-studies/gemmini-dse-followup/v1/perf-studies_gemmini-dse-followup_v1_20261025T055037Z_431485b/` |
| Fresh full ResNet inputs and capture | `merlin-tvm-tuning/out/artifacts/handoff/gemmini-local-resnet-tuning-input/v1/handoff_gemmini-local-resnet-tuning-input_v1_20261025T055124Z_431485b/` |
| Real ResNet classifier campaign, including capped outcomes | `merlin-tvm-tuning/out/artifacts/perf-studies/gemmini-resnet-real-tuning/v1/perf-studies_gemmini-resnet-real-tuning_v1_20261025T060908Z_431485b/` |
| Full diagnostic ResNet package and positive/wrong-oracle session ELFs | `merlin-tvm-tuning/out/artifacts/handoff/gemmini-resnet-profile-session/v1/handoff_gemmini-resnet-profile-session_v1_20261025T061349Z_431485b/` |
| Earlier study, toolchain recipe and original controls | `merlin-tvm-tuning/out/artifacts/perf-studies/gemmini-fased-workload-sweeps/v1/perf-studies_gemmini-fased-workload-sweeps_v1_20261024T073102Z_36899e8/` |

For the ResNet, use `resnet-capture-v2/program.pt2`, its `meta.json`,
`integer-reference.json`, `inputs.json`, `calibration.npz` and `visible.npz`.
All 260 calibration source hashes were checked, and the `v00` crop matches the
committed visible input. The first capture attempt's missing-dependency failure
is retained separately. The second capture is complete and opaque-free.

Also retain or rebuild the dependency closure named by each toolchain receipt:

- LLVM **18.1.3** SDK, RV64/lp64d GCC, Spike and its matching Gemmini plugin.
- The qualified adapter object and its exact hardware headers.
- The pinned Chipyard/RTL source/build tree and base Verilator executable.
- `merlin-fased-memory/out/build/rtl_engines/gemmini/verilator_mem_models_v1/`,
  including the latency-pipe emulator and build receipt.
- The selected hardware pin registry, configuration and memory-map evidence.
- The official ResNet checkpoint and Imagenette archive if recapture is needed.
  These are under `merlin-tvm-gemmini/out/artifacts/cache/resnet50/` locally.

Resolve artifact symlink targets when making a transfer archive; some linked
session inputs point back into their package. Do not copy credentials, SSH
keys or `.aws` configuration. Rebuild Python environments rather than copying
virtual environments. The audit's `python-environments.json` records the local
dependency inventories; those overlays are not a validated portable lockfile.

## 3. Establish the remote environment

Use Linux/x86-64 for these existing native binaries. Start with Python 3.12,
the recorded Torch/export dependencies, NumPy, ONNX and XGBoost 2.1.4. Check
against `python-environments.json`; preparation and compiler environments have
different dependency needs. Keep model preparation separate from measurement.
The existing [HOST_SETUP.md](HOST_SETUP.md) and companion TVM
`apps/gemmini/{README,TUNING,WORKLOAD_TUNING}.md` describe the full dependencies.

Define paths for the **remote** checkouts, SDK and fresh build directory:

```bash
export MERLIN_REPO_ROOT="$MERLIN_DELIVERY"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONPATH="$MERLIN_MEMORY/src:$MERLIN_DELIVERY/src:$TVM/python:$TVM/apps/gemmini"
export LD_LIBRARY_PATH="$LLVM18/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

A clean full TVM rebuild can use the same relevant options as the local build:

```bash
cmake -S "$TVM" -B "$TVM_BUILD" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DUSE_LLVM="$LLVM18/bin/llvm-config --link-shared" \
  -DUSE_OPENMP=OFF -DUSE_LIBBACKTRACE=OFF \
  -DUSE_FALLBACK_STL_MAP=OFF -DUSE_HOST_ONLY_AUTO_COPY_GUARD=ON
cmake --build "$TVM_BUILD" --target tvm tvm_runtime -j2
export TVM_LIBRARY_PATH="$TVM_BUILD"
```

Initialize the TVM submodules required by its build first. The local follow-up
library is at `merlin-tvm-tuning/out/build/baselines/tvm-f2-followup/compiler/`;
its receipt records the two changed native units and a byte-identical control
relink of the previous library. Copying it does not qualify a different host.

Use a Python environment with the required dependencies installed, or add its
explicit, recorded overlay directories. After building, confirm the actual imports:

```bash
"$PY" - <<'PY'
import tvm
from merlin.targetgen import mem_perturb
print(tvm.__file__)
print(tvm._ffi.base._LIB._name)
print(mem_perturb.__file__)
assert hasattr(mem_perturb, "pinned_latency_pipe")
PY
```

## 4. Rebind execution on the remote host

Old receipts contain absolute paths and identify historical executions. Keep
them unchanged. If the old absolute layout cannot be preserved, rebuild/re-run
the relevant tools and produce **new remote receipts**. Do not edit old hashes
or paths to make validation pass. Receipt relocation/requalification remains
an explicit migration task; this audit does not implement a universal rewriter.

Start from the earlier study's `memory/toolchain.json`. For new runs supply the
remote paths for `gcc`, `adapter_receipt`, `spike_receipt`, `hardware_receipt`,
`memory_variant_receipt`, `memory_pin_registry`, and keep:

```json
{
  "firesim_memory_artifact": "firesim_gemmini_rocket_u250_30mhz",
  "timing_policy": "fixed-warm"
}
```

This is the provisional memory-parameter source, not a claim that the F2 image
is the U250 image. Verify the base simulator/adapter/Spike controls, the memory
variant's control relink, and a positive/wrong-oracle pair before a campaign.
The console must confirm `latency_pipe` and **30/30/16/16**. The existing
`verify_matmul.py`, `verify_matmul_runtime.py`, `qualify_package.py` and retained
build/qualification transcripts are the starting points for those controls.

Check the complete memory span before executing the full ResNet profile ELF:
its weights are at `0x2C0000000`, arena at `0x3C0000000`, inside a 16-GiB map
beginning at `0x80000000`. The kernel-tuning guests instead use the local
256-MiB map. The experimental local gSIM harness supports only that smaller map.

## 5. Continue the audit on real stages

Allocate fresh products with `merlin.common.artifacts.new_product`; do not
resume a local directory after changing source, paths, compiler or engine.

```bash
"$PY" "$TVM/apps/gemmini/tune_models.py" \
  --program "$PROGRAM" --work-dir "$RUN/extraction" \
  --extract-only --graph-fusion-level 2

"$PY" "$TVM/apps/gemmini/audit_graph_plans.py" \
  --program "$PROGRAM" --output-dir "$RUN/graph-audit"
```

Set `PROGRAM` to an actual frozen stage archive. The local fresh full ResNet
has 68 unique tasks, of which 67 contain an eligible region; eligibility does
not imply that the whole function is accelerator work. Preserve all CPU work
and session call-count weights in the inventory.

Profile the complete stage before selecting a larger search budget. Compile
with `compile_package.py --profile-operators` and link with
`build_package_session.py --profile-stages`. The full local positive and
wrong-oracle sessions pass on Spike with all 1,000 logits checked; the valid
session records 128 operator calls in each phase. **Spike's counters are instruction
counts, not RTL performance labels.** Obtain target-cycle attribution on the
selected timing engine/image. The saved package/session `commands.json` is the
concrete build recipe; use fresh output directories when reproducing it.

For a bounded initial operator campaign, select actual eligible task indices
from that extraction and use the validated remote receipts:

```bash
"$PY" "$TVM/apps/gemmini/tune_models.py" \
  --program "$PROGRAM" --work-dir "$RUN/operators" --tasks "$TASK_INDEX" \
  --graph-fusion-level 2 --trials-per-task 8 --trials-per-iteration 2 \
  --parallel 1 --builders 1 --baseline-workers 1 \
  --timing-policy fixed-warm --max-cycles 6000000 \
  --cpu-max-cycles 500000 --validation-overhead-cycles 1000000 \
  --rtl-seconds 1200 --seed 20261025 \
  --gcc "$GCC" --adapter-receipt "$ADAPTER_RECEIPT" \
  --spike-receipt "$SPIKE_RECEIPT" --hardware-receipt "$RTL_RECEIPT" \
  --memory-variant-receipt "$MEMORY_VARIANT_RECEIPT" \
  --memory-pin-registry "$MEMORY_PIN_REGISTRY" \
  --firesim-memory-artifact firesim_gemmini_rocket_u250_30mhz
```

These are starting bounds, not convergence evidence. The simulator cap covers
startup, both invocations and validation; the reported inference interval does
not. Increase bounds only after inspecting retained capped logs. Never use a
timeout or an unvalidated timing line as a speedup denominator.

Validate/merge records with `tuning_library.py`, check exact structural coverage
against the same stage, then compile with that explicit library. Keep compiler
graph choices, engine and numerical identities separate. Connected sweeps now
measure one representative of identical prepared graphs. A final performance
comparison needs uninstrumented default/winner sessions on the same deployment.

## Optional gSIM evidence

A local native gSIM build was made from public commit
`f38de1704dac25b540d1552fbe1d23fc48463212`, with the same neutral latency-pipe
factory sources added to its harness. The small positive and wrong-oracle
controls passed; positive warm cycles matched Verilator. Both larger projection
controls exhausted their 120-second wall limits. No gSIM tuning cost or general
speedup was admitted, and the TVM runner remains a Verilator runner.

The optional source, native build and control evidence are respectively under
`merlin-tvm-tuning/out/build/baselines/gsim-local-source/`,
`merlin-tvm-tuning/out/build/rtl_engines/gemmini/gsim-latpipe-native-v1/`, and
`merlin-tvm-tuning/out/artifacts/verification/gemmini-gsim-latpipe/`.
The receipt-writing environment issue and its successful recovery are retained.
Archive these if desired; further gSIM setup is outside this audit's next step.
