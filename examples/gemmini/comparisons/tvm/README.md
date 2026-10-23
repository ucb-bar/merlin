# TVM comparison: ResNet50, TinyLlama and SmolVLA FireSim handoff

The goal is repeatable complete inference runs of **all three EL4 workloads**.
The workload authority is Merlin EL4 at
[`0cc289de52a10bed256f32c90cfbb54a21b3c81f`](https://github.com/ucb-bar/merlin/tree/0cc289de52a10bed256f32c90cfbb54a21b3c81f/examples/gemmini/protocol).
Its protocol already specifies model revisions, quantization recipes, calibration,
input shapes, token counts, recurrence and output checks. The repo also includes
stored visible inputs. The remaining task is to bind these definitions to a
shared numerical release, complete compiled sessions and verified FireSim runs.
This handoff supersedes the earlier two-model 16+1 preliminary selection.

The 2026-10-23 fetch and pull found `origin/feat/el4-experiment` still at `0cc289de5`;
this handoff already contains it, with no merge conflicts. The selected protocol,
three base recipes and committed input definitions are unchanged from the user's
earlier `db2f52415fe9f7d2cd7f028e122f8c9b039a925c` reference. Runtime tooling has
advanced: EL4 now has a complete-chain image builder, compact all-element
comparison, and strict or lightweight boot-file provenance. The TVM session
bridge still needs an adapter to those newer manifest/transport contracts.

The companion TVM fork owns mathematical lowering, schedules and static
packages. Merlin owns the workload contract, independent references, session
routing and execution evidence. Read [AGENTS.md](AGENTS.md) for the continuation
sequence and [EL4_ASSETS.md](EL4_ASSETS.md) for the shared input/release checklist.

## What is already standardized, per workload

One selected `visible/v00` input per model is the functional benchmark cohort.
Warmup and measured repetitions can reuse that input with a full state reset;
record their counts in the run manifest. Larger quality cohorts are separate.
The definitions below come from the pinned
[functional profile](https://github.com/ucb-bar/merlin/blob/0cc289de52a10bed256f32c90cfbb54a21b3c81f/examples/gemmini/protocol/experiment.yaml#L24-L41).

| Workload | Model and standardized input | Complete inference pass and outputs | Remaining workload preparation |
| --- | --- | --- | --- |
| ResNet50 v1.5 | torchvision `IMAGENET1K_V2`, 50 layers, batch 1. The committed crop for `v00` is normalized to FP32 `[1,3,224,224]`; its derived NPZ hash is pinned. | One call; all `[1,1000]` logits. | Publish the selected quantized weights/scales/IR and independent logits reference, then compile and qualify the complete model. The input definition is already usable. |
| TinyLlama | `TinyLlama-1.1B-Chat-v1.0` at `fe8a4ea1ffedaf415f4da2f062534de366a451e6`; all 22 layers, hidden 2048, intermediate 5632, 32 attention heads / 4 KV heads. The committed `v00` contains 160 WikiText2 test token IDs; this chain consumes the first 64. | Prefill 32 tokens, then 32 teacher-forced single-token calls: 33 calls and 99 arrays. Each call returns logits `[1,32000]`, FP32 KV `[2,22,1,4,64,64]` and i64 positions `[22]`. | Bind the prescribed calibration, SmoothQuant/outlier transforms and `portable_exp_trig` host math to the capture path; publish both stage programs, scales and all-output references. Link the complete recurrence with cache capacity 64. |
| SmolVLA | `lerobot/smolvla_base` at `c83c3163b8ca9b7e67c509fffd9121e66cb96205`, with the pinned SmolVLM2 backbone; 16 selected VLM layers and 16 expert layers. One camera, image `[1,3,512,512]`, 48 language slots, state `[1,32]`. `v00` fixes episode 0/frame 100/noise seed 1000. | Prefix once, denoise ten times, action decode once: 12 calls and 33 arrays. Carry actions/cache/time; compare all `[1,50,32]` final action values, including padding. | Reproduce the pinned processed observation from the committed raw frame/state/task/seed using the selected processor environment. Publish all three prepared programs and references with `portable_neural_f32`, then qualify the full recurrence. |

Sources: [ResNet50 contract](https://github.com/ucb-bar/merlin/blob/0cc289de52a10bed256f32c90cfbb54a21b3c81f/examples/gemmini/protocol/experiment.yaml#L77-L102),
[TinyLlama contract](https://github.com/ucb-bar/merlin/blob/0cc289de52a10bed256f32c90cfbb54a21b3c81f/examples/gemmini/protocol/experiment.yaml#L103-L147),
[SmolVLA contract](https://github.com/ucb-bar/merlin/blob/0cc289de52a10bed256f32c90cfbb54a21b3c81f/examples/gemmini/protocol/experiment.yaml#L148-L202),
[stored input formats](https://github.com/ucb-bar/merlin/blob/0cc289de52a10bed256f32c90cfbb54a21b3c81f/examples/gemmini/numerics/README.md#L91-L127).

**ResNet's 260 images are calibration data, not 260 benchmark passes.** The
recipe uses 26 training images per Imagenette class to determine quantization
scales during model preparation. Once those scales are frozen, each benchmark
pass processes one selected validation image. Likewise, TinyLlama's 128 prefill
and 1,024 decode calibration samples are distinct from its measured 32+32 chain;
SmolVLA selects one calibration sample per program. The historical stored cohort
contains eight visible cases per model, but the selected functional profile uses
`v00`. SmolVLA's final output is normalized policy-space actions; its six semantic
channels can be reported separately, but physical-unit postprocessing is not
part of the selected chain. These choices are specified in the contracts above.

## Freeze one numerical release for the team

A previous capture is **not inherently needed to create a new repeatable run**.
The recipes define how to prepare the quantized model; calibration produces the
actual scale and weight bytes. Prepare once, admit the independent references,
then give every team member the same versioned package and hashes. Fresh
calibration creates a new numerical identity and cannot silently replace the
old selected scales or inherit its qualification. Exact reproduction of the
historical frozen result requires its bytes or a verified identical rebuild.
See the pinned [quantization authority](https://github.com/ucb-bar/merlin/blob/0cc289de52a10bed256f32c90cfbb54a21b3c81f/examples/gemmini/protocol/experiment.yaml#L61-L76)
and [baseline workflow](https://github.com/ucb-bar/merlin/blob/0cc289de52a10bed256f32c90cfbb54a21b3c81f/examples/gemmini/protocol/baselines.md).

The repo supplies definitions, input bytes, recipes and producers/checkers. It
does not supply a portable, complete selected capture/reference release with a
team-accessible payload location. Publish that release with checkpoint and
environment pins, prepared inputs, calibration selection, scales/constants,
integerized IR, all-output references and the resolved runtime configuration.
Keep heavyweight/protected payloads in shared artifact storage; a checked-in
manifest must let another member fetch and verify them. Historical absolute
machine paths alone do not satisfy this requirement. Do not substitute the older
comparison factories' quantization recipes for the selected EL4 recipe.

| Owner | Responsibility for this run |
| --- | --- |
| EL4 protocol and capture preparation | Checkpoint, preprocessing, calibration, INT8 recipe, scale derivation, attention planes and rounding; independent numerical reference |
| TVM | Import that quantized program; choose legal packing, tiling, OS/WS dataflow, software pipelining, SRAM allocation/reuse and CPU placement |
| Merlin session runtime | Route every stage and carried state, own persistent tensor buffers, reset each repetition, check all outputs and collect complete-session timing |

Cache residency within an operator is a TVM scheduling decision. KV state
across separate calls is an explicit runtime tensor: the existing scheduler does
not promise that the whole model's KV cache remains in Gemmini SRAM across calls.

The generic capture worker accepts `--save-exported-program`. It saves
`program.pt2` from the **same prepared conversion** used for MLIR, without a second
export or different quantizer, and records its SHA256 in `meta.json`. In a
multi-program capture, each `stages/<name>/` receives its own archive. TVM's
`compile_package.py --program` and `tune_models.py --program` accept these archives.
Only successful, opaque-free, precision-preserving, integerization-checked
conversions are saved. An archive alone does not establish full-model correctness.

Existing delivery evidence: a small convolution/ReLU/pool/linear graph using the
ResNet EL4 recipe passes TVM host comparison against the independent PT2E integer
reference at `atol=2e-6, rtol=1e-5`. Its capture compiles into a static RV64/Gemmini
package and passes Spike positive and wrong-oracle controls. A two-stage capture
round-trip verifies both archives exactly. TinyLlama explicit-cache tests cover
the earlier 16+1 diagnostic and repeated reset. These controls do not qualify
full checkpoints, the canonical 32+32 chain or FireSim execution.

## Preparation gaps and current process stage

These findings were rechecked against fetched EL4 `0cc289de5` and the delivered
TVM code. A specified policy, an implemented tool and a qualified model run are
different milestones:

| Finding | Action before a shared run |
| --- | --- |
| Default `numerics/models.yaml` resolves host math to `native`; the protocol requires `portable_exp_trig` for TinyLlama and `portable_neural_f32` for SmolVLA. | Publish a resolved models configuration carrying these existing choices and use it for references/execution. The policy is specified; its runtime binding is incomplete. [Model defaults](https://github.com/ucb-bar/merlin/blob/0cc289de52a10bed256f32c90cfbb54a21b3c81f/src/merlin/model_check/models.py#L35-L42). |
| SmolVLA input preparation reads `source.local_root` instead of the supplied data root. The generic setup script installs LeRobot 0.5.1, while the protocol selects 0.6 processors. | Make the data path portable and pin the matching processor environment; verify the processed observation hash. [Input preparation](https://github.com/ucb-bar/merlin/blob/0cc289de52a10bed256f32c90cfbb54a21b3c81f/src/merlin/model_check/inputs.py#L357-L365), [setup script](https://github.com/ucb-bar/merlin/blob/0cc289de52a10bed256f32c90cfbb54a21b3c81f/build_tools/scripts/setup_model2mlir.sh#L23-L40). |
| EL4 now builds complete native guest chains from finished `baremetal_model.json` program builds. It can compare every output element in the guest and send compact results. | Reuse its contracts and checks. This is implemented for Merlin's native program ABI; TVM's `model_run` packages and `tvm_merlin_package_session_v1` manifest are not accepted as native build evidence. The remaining work is a TVM adapter, not inventing a complete-chain runtime. [Chain-image guide](https://github.com/ucb-bar/merlin/blob/0cc289de52a10bed256f32c90cfbb54a21b3c81f/docs/reference/chain_images.md). |
| The FireSim selection is a template with `manager_contract: null` and empty model asset lists. | Bind actual compiler freeze, complete-chain ELFs, inputs/references and manager identity, then prepare the request. [Selection template](https://github.com/ucb-bar/merlin/blob/0cc289de52a10bed256f32c90cfbb54a21b3c81f/examples/gemmini/validation/firesim_workhorse.yaml). |
| Strict version-pinned upload/manager helpers and lightweight boot-file verification are implemented. The checked-in template selects strict version fields; the installed legacy helper has different behavior. | Choose one documented mode and supply its actual evidence. Strict helper installation is a separate deployment step; lightweight mode avoids that upgrade but still requires a compact complete-chain guest. Neither mode currently accepts the TVM session unchanged. [Strict integration](https://github.com/ucb-bar/merlin/blob/0cc289de52a10bed256f32c90cfbb54a21b3c81f/examples/gemmini/validation/manager-provenance.md), [lightweight mode](https://github.com/ucb-bar/merlin/blob/0cc289de52a10bed256f32c90cfbb54a21b3c81f/examples/gemmini/validation/booted-elf-provenance.md). |
| TVM imports prepared `.pt2` programs, but its importer does not automatically apply Merlin's selected portable host-math policy. The session bridge records an aggregate warm/measured interval, without per-stage counters. | Check full-model agreement with the independent portable references; implement any required arithmetic compatibility without widening tolerances. Add stage counters while retaining the complete-chain interval, including CPU work, transfers and recurrence. [TVM intake](https://github.com/grammy117/tvm/blob/5b4c36da713234e48a0deb35365b42a26c196707/apps/gemmini/program_io.py), [session runtime](package_session.py). |

**Current stage:** workload definitions and input/reference tooling exist;
portable numerical-release assembly and complete-model integration/qualification
remain. The V26 selection records a frozen generated compiler, 152 public numeric
passes plus one expected refusal and eight hidden aggregate passes, followed by
an independent **capsule-only** qualification. It still records
`full_model_validation: NOT_RUN for V26`, no qualified release, and no bundled
portable capsule receipt; original-source replay is required because four harness
files differ. Those facts neither require a new authoring run nor qualify a TVM
baseline. See the pinned [V26 status](https://github.com/ucb-bar/merlin/blob/0cc289de52a10bed256f32c90cfbb54a21b3c81f/examples/gemmini/phase2/el4_20261018/selected-phase1-v26.json#L40-L79).
This delivery's small TVM/Spike controls also do not establish three-model
FireSim completion.

The current audit passed the protocol checker (3 models, 46 calls, 133 arrays),
283 EL4 input/numerics/chain/transport tests, 218 TVM backend/CLI tests, and two
explicit prepared-capture round-trip tests. The broader comparison run had
256 passes, three failures and five skips: two failures are pinned AVX512 SiLU
environment refusals; one is the LeRobot 0.5.1/diffusers/TorchAO import conflict.
Three skips also require AVX512; the two capture skips passed when rerun with
their explicit interpreter. These are unresolved environment coverage limits,
not an all-green full-model result. Input derivations and the small EL4 recipe
probe are recorded in [EL4_ASSETS.md](EL4_ASSETS.md); no full-model FireSim run
or SmolVLA processed-input equality is claimed.

For the TVM capture bridge, the tested model2MLIR pin is
[`ucb-bar/model2MLIR`](https://github.com/ucb-bar/model2MLIR) at
`7915e23475c6db446a3c404847b11e8bc72c8a27`. The capture probe used Python 3.12,
Torch 2.10.0, TorchAO 0.16.0, xDSL 0.68.0 and NumPy 1.26.4; TinyLlama's selected
cache adapter additionally uses Transformers 5.4.0. These are delivery probe
versions, not evidence that all EL4 captures can already be rebuilt in that
environment. The [pinned TinyLlama loader](https://github.com/ucb-bar/model2MLIR/blob/7915e23475c6db446a3c404847b11e8bc72c8a27/workloads/tiny_llama/loader.py)
has no EL4 WikiText calibration hook or SmoothQuant/outlier preparation;
[Merlin's recipe worker](https://github.com/ucb-bar/merlin/blob/0cc289de52a10bed256f32c90cfbb54a21b3c81f/src/merlin/targetgen/_m2m_capture_worker.py#L1208-L1237)
expects the loader to supply calibration inputs. This integration remains unfinished. Implement
the specified transforms before accepting its full-model export; do not ask for
new model-policy choices. Use the environment setup below and create outputs
through Merlin's artifact helper; `RUN` is that fresh directory.

## Apply the learned TVM settings

The companion branch ships `apps/gemmini/presets/rtl-v1/`: all **nine measured
operator winners** as stock MetaSchedule workload/trace records, plus capability,
payload and original measurement hashes. Use this preset for each prepared stage:

```bash
"$PY" "$TVM/apps/gemmini/compile_package.py" \
  --program "$RUN/resnet-capture/program.pt2" \
  --tuning-preset rtl-v1 --tensor-memory-bytes 8589934592 \
  --adapter-receipt "$ADAPTER_RECEIPT" --riscv-gcc "$GCC" \
  --output-dir "$RUN/resnet-package"

# After the full EL4 TinyLlama capture has passed its numerical checks:
for stage in prefill decode; do
  "$PY" "$TVM/apps/gemmini/compile_package.py" \
    --program "$RUN/tinyllama-capture/stages/$stage/program.pt2" \
    --tuning-preset rtl-v1 --tensor-memory-bytes 8589934592 \
    --adapter-receipt "$ADAPTER_RECEIPT" --riscv-gcc "$GCC" \
    --output-dir "$RUN/tinyllama-$stage-package"
done
```

The tensor-memory argument is a compilation ceiling, not a measured allocation
or a complete device-memory reservation. Inspect package and final session maps.
The traces include tile/reuse factors, traversal, OS/WS, operand hoisting,
pipelining, epilogue/storage placement and unrolling. Config shadowing, DMA
coalescing, first-K overwrite, lifetime allocation, fences and proved packing/bias
folding remain compiler passes. No user-selected per-model SRAM tile sizes are
required.

Stock structural matching applies only compatible records. A 32-row prefill
winner does not prove coverage of the selected full program: EL4 epilogues and
other graph differences can change matches. `placement.json` and `package.json` identify
matched functions and unmeasured defaults; a preset permits zero matches and
reports that truthfully. `--tuning-database` retains its stricter zero-match error.
Do not call all full-model operators tuned merely because the preset is installed.
The nine RTL measurements are not FireSim or E2E timing; remeasure the full chain.

Compile SmolVLA's `prefix_encode`, `flow_denoise` and `action_decode` archives in
the same way after their capture/reference gates pass. Finish the independent
full-model output gates, construct the explicit routes and fixtures, and follow
the package-session and direct FireSim sections below. The selected manager's
driver, memory map and loader still need concrete bindings and qualification.

The delivered operator studies are complete at their stated scope. They do not
qualify those full-model FireSim sessions. Start with [AGENTS.md](AGENTS.md) and
[UPSTREAM_AUDIT.md](UPSTREAM_AUDIT.md). [EL4_ASSETS.md](EL4_ASSETS.md) identifies the
committed inputs, required shared release files and known hashes. The
historical host, cutout and legacy BYOC
reports remain in [README_HISTORICAL.md](README_HISTORICAL.md); their original
inputs, tolerances, failures and attribution remain unchanged.

## Source checkout and documentation map

The delivery branch `comparisons/tvm-el4-handoff` contains fetched EL4
`0cc289de52a10bed256f32c90cfbb54a21b3c81f`. Record the actual handoff compiler/runtime
checkout separately from the EL4 workload pin. To refresh a clean handoff checkout:

```bash
git fetch origin
git log --oneline HEAD..origin/feat/el4-experiment
git diff --stat HEAD...origin/feat/el4-experiment
git merge --no-edit origin/feat/el4-experiment
```

Review incoming changes and resolve conflicts before testing. Preserve dirty
work in its existing worktree; do not reset it. In the separate TVM checkout,
fetch/pull `grammy117/tvm` branch `gemmini/curated-handoff` with `--ff-only`.
Record both resulting hashes and refresh this audit when code changes. This is
a repeatable update procedure, not an automatic background synchronization job.
Original comparison work was on
`comparisons/tvm-gemmini-byoc` at `1bd601422fc1defc4343bacbf4227ca17e8aefef`
plus the saved working changes. The original comparison branch and dirty tree
are preserved. This delivery adds comparison files/tests/docs to EL4 and does
not rewrite its compiler/campaign code. A generic optional capture export
now preserves the exact prepared Torch program for external compiler intake.

TVM's pullable delivery is
[`grammy117/tvm:gemmini/curated-handoff`](https://github.com/grammy117/tvm/tree/gemmini/curated-handoff)
with tested compiler/runtime source at `5b4c36da713234e48a0deb35365b42a26c196707`;
this re-audit updates its handoff documentation without changing those sources.
It is based on Apache v0.19.0
`c4dc0c29ff81ddae688da24625a603da3b4a2c0e`. Its historical measured source is
`87f07883b65ee55dbd1b93f665aaab87cd8a1e50` on `gemmini/metaschedule-dse`.
See the final handoff artifact for delivery commit IDs, bundles, hashes and tests.
TVM is a separate repository, not a Merlin submodule. Inside its source tree:

| Document | Authority |
| --- | --- |
| `apps/gemmini/README.md` | Supported entrypoints and qualification boundary |
| `apps/gemmini/OWNERSHIP.md` | Stock/supplied/external responsibilities, all custom passes, optimization ownership chart, upstream changes |
| `apps/gemmini/TUNING.md` | Stock MetaSchedule flow, measurement policy and extension priorities |
| `apps/gemmini/FIRESIM.md` | Static ABI, independent fixtures, final linking/deployment gates |
| `apps/gemmini/AGENTS.md` | Compiler continuation instructions |

The retained `target.yaml`, `HOST_SETUP.md` and `byoc/` describe the older
`6b376cc8` bringup profile. They are reproducibility references, not the current
TensorIR configuration. The new compiler uses its source-bound
`GemminiCapabilities`, primitive adapter receipt and `firesim_target.json`.
Do not install the legacy snapshot over a new development checkout.

## First session: establish an isolated environment

Read root `AGENT.md`/`CLAUDE.md`. Work in separate checkouts, set the actual paths
below and use a fresh artifact root. Do not copy another server's absolute paths
or source-epoch receipts as if they qualified this server.

```bash
export MERLIN=/absolute/merlin-el4-checkout
export TVM=/absolute/tvm-curated-checkout
export TVM_BUILD=/absolute/matching-tvm-build
export PY=/absolute/python-with-torch-onnx-tvm-dependencies
export M2M=/absolute/model2mlir-checkout
export GCC=/absolute/riscv64-unknown-elf-gcc
export MERLIN_REPO_ROOT="$MERLIN"
export PYTHONPATH="$MERLIN/src:$MERLIN/packages/merlin-experiments/src:$TVM/python:$M2M"
export TVM_LIBRARY_PATH="$TVM_BUILD"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
cd "$MERLIN"
"$PY" - <<'PY'
import tvm
from tvm._ffi.base import _LIB
from merlin.common.artifacts import new_product
print('TVM Python:', tvm.__file__)
print('TVM native:', _LIB._name)
p = new_product('compare', version=1, target='gemmini-tvm-el4',
                notes='Actual workload intake; no hardware qualification inferred')
print('Set RUN to:', p.path)
PY
export RUN=/absolute/path/printed/above
```

Add only the explicit dependency roots required by that environment. Record
Python, Torch, ONNX, NumPy, TVM, LLVM, GCC and source hashes. The completed local
study used LLVM18/native compiler-v5 and a separate Python3.12 environment; a
clean history does not require recompiling byte-identical native sources.
Any subsequent C++ change does. Confirm the native source identity before reuse.
The TinyLlama cache/session integration requires the selected Transformers 5.4.0
overlay; the older SmolVLA diagnostic environment's 4.57.1 cache API is incompatible.
Source-exact SiLU diagnostics additionally require AVX512 and remain unqualified
on this AVX2 host; distinguish those from the selected EL4 portable policies.
The handoff artifact includes a local-path environment example; it is not portable.

## Use the selected numerical contract

The current EL4 model roster is
[examples/gemmini/numerics/models.yaml](../../numerics/models.yaml), with source
and numerical policy in [the protocol](../../protocol/README.md). Its selected
run configuration and input manifests are authoritative if they override defaults.
[Phase 0 workload patterns](../../target/phase0-workloads.yaml) are independently
authored development inputs, not actual pretrained model checkpoints. The old
`el4_20261018/common.py` embeds server-specific paths and a stopped campaign;
do not rerun it to discover a new model workflow.

EL4's default numerical policy names all declared outputs at every step,
finite values and exact integer/boolean comparison. The roster's default
functional thresholds are `atol=2e-6`, `rtol=1e-5`; its quality thresholds are a
separate contract (e.g. TinyLlama KL/top-one agreement), report-only in the selected
functional profile. Read the **selected frozen
run policy** before adopting these defaults. TVM comparison CLI defaults differ
and do not override EL4. Preserve historical recipes as historical; do not silently
replace a full-model contract with the old 128+32 host study or an INT8 cutout.

Checkpoint/data hashes, factory arguments and session manifests must identify:
model revision, original dtype, all quantization scales/zero points/rounding,
input order/layout, source preprocessing, calibration/development/held-out roles,
sequence/observation counts, cache capacity, recurrence/reset behavior, output
meaning and accuracy policy. Protect held-out inputs/answers from tuning and
public compiler-development artifacts. Preserve the already committed visible
inputs; new heavyweight captures and protected reference payloads stay outside Git.

## Comparison frontend controls and semantic-program intake

The primary EL4 intake is the prepared `program.pt2` saved by the capture worker
above. The older comparison factories below remain useful for frontend/session
diagnostics, but their quantizers and output interfaces do not automatically
implement the selected EL4 recipe. Their exports are not substitutes for the
frozen EL4 prepared program, even when the checkpoint and token counts match.

The reusable [session verifier](verify_session.py) follows model2MLIR's
`ExternalRuntimeSession`; factories supply model-specific contracts. It supports
source preservation, explicit routes, signature limits and repeated resets.
`--save-programs` persists `stage_N/semantic.json` with a source hash for the same
program input used by TVM tuning and compilation. This is opt-in because constants
can be large. Import-only mode executes native source stages to build real stage
inputs but does not build or execute the TVM candidate.

For TinyLlama, prepare local factory arguments (JSON) with at least
`checkpoint_dir`, `model2mlir_root`, `corpus_path`, `prefill_tokens: 32`,
`decode_tokens: 32`, and the diagnostic's explicit `precision`/`quantized` policy.
The defaults in [tinyllama_session.py](tinyllama_session.py) are diagnostic.
For SmolVLA use [smolvla_session.py](smolvla_session.py): `checkpoint_dir`,
`backbone_dir`, `model2mlir_root`, `input_npz`, `input_manifest`, `input_source`,
selected cameras and the declared original precision/quantization policy. Select
`fixture_kind="dataset"`; its default `fixture_kind="synthetic"` is a diagnostic. Verify the processor and physical action dimension.

```bash
"$PY" examples/gemmini/comparisons/tvm/import_models.py \
  --model tinyllama --tvm-source "$TVM" --tvm-build "$TVM_BUILD" \
  --model2mlir-root "$M2M" --factory-arguments "$RUN/tinyllama-arguments.json" \
  --save-programs --output-dir "$RUN/tinyllama-import"

# Repeat for smolvla with its own real-input arguments and directory.
# ResNet intake accepts --model resnet50 --checkpoint /absolute/checkpoint.
```

For host candidate/reference checks of that diagnostic, use the verifier:

```bash
"$PY" examples/gemmini/comparisons/tvm/verify_session.py \
  --model2mlir-root "$M2M" --tvm-source "$TVM" --tvm-build "$TVM_BUILD" \
  --factory examples/gemmini/comparisons/tvm/tinyllama_session.py \
  --factory-arguments "$RUN/tinyllama-arguments.json" \
  --onnx-exporter dynamo --opset 22 --graph-mode optimized --repeats 2 \
  --rtol 0.00001 --atol 0.000002 --save-programs \
  --output-dir "$RUN/tinyllama-host"
```

This diagnostic uses the roster tolerances; passing it does not admit the EL4
quantized program. Native BF16 source reproduction is
not established by generic FP32 fused export. An import/refusal or host mismatch
must be resolved and recorded before any device performance claim.

For ResNet, [prepare_resnet_inputs.py](prepare_resnet_inputs.py) can create distinct
streams from the selected local image archive; it never downloads the dataset.
The comparison verifier's `--quantized --calibration-inputs … --inputs … --labels …`
path requires explicit sources/preprocessing and accepts a frozen quantization
plan for held-out evaluation. `--export-baremetal` also saves
`integer_semantic.json`, the full integer recipe including its declared outputs.
Its Imagenette development stream is not a substitute for an EL4 selected dataset
or canonical ImageNet accuracy. See `verify_resnet.py --help` for that comparison's
controls. In the selected EL4 functional profile, source-versus-quantized quality
is reported separately; compiled-versus-quantized agreement is the functional gate.

## Extract, tune and compile exactly those programs

```bash
export PROGRAM="$RUN/tinyllama-capture/stages/prefill/program.pt2"
"$PY" "$TVM/apps/gemmini/tune_models.py" --program "$PROGRAM" \
  --work-dir "$RUN/stage0-extraction" --extract-only
```

Read `extraction.json`: all extracted tasks, call weights, eligible tuning indices
and CPU exclusions are explicit. Compare it with the entire stage/session roster.
A stage imported once may execute 32 or ten times; IR-local extraction does not
know the outer session cadence. `--model tinyllama-decode` and the other built-in
TVM model-shaped programs use synthetic constants and are only pilot workloads.

```bash
"$PY" "$TVM/apps/gemmini/tune_models.py" --program "$PROGRAM" \
  --work-dir "$RUN/stage0-tuning" --trials-per-task 16 \
  --trials-per-iteration 4 --parallel 4 --builders 2 \
  --gcc "$GCC" --adapter-receipt "$ADAPTER_RECEIPT" \
  --spike-receipt "$SPIKE_RECEIPT" --hardware-receipt "$RTL_RECEIPT"

"$PY" "$TVM/apps/gemmini/compile_package.py" --program "$PROGRAM" \
  --output-dir "$RUN/stage0-package" --riscv-gcc "$GCC" \
  --adapter-receipt "$ADAPTER_RECEIPT" \
  --tensor-memory-bytes "$STAGE_TENSOR_BUDGET" \
  --tuning-database "$RUN/stage0-tuning/database"
"$PY" "$TVM/apps/gemmini/compile_package.py" --verify "$RUN/stage0-package"
```

Set the receipt and budget variables to verified actual artifacts, not example
filenames. The tuning command measures on local Verilator, not FireSim. The
budget is a starting experiment setting, not evidence of convergence. Repeat for
all signatures; preserve CPU/default baselines and all failed/capped trials.
Inspect each package's placement, constant/workspace plan and archive audit.
No function matching a database is not the same as complete database coverage.
`--tuning-result` and the older custom controller are retired.

## Build the full session and functional controls

A TVM package exports `model_run(inputs, outputs, workspace, workspace_bytes)`.
The caller owns aligned, correctly sized, nonaliasing buffers. It must route every
stage and state update, preserve immutable constants and execute the selected
reset/repeat contract. Multiple archives export the same generic symbols: link
through an explicit namespaced-stage plan or the existing `baremetal.export_graphs`
API, with hashes binding each entrypoint. Do not blindly link several `model_run`
definitions, nor treat a TVM pointer-array ABI as a gemmini-mlir command-buffer ABI.

[`build_package_session.py`](build_package_session.py) now links static packages
through Merlin's existing startup, HTIF, linker, warm-profile generator and
OUT_BIN readback. It namespaces each stage's defined symbols and shares one
runtime/adapter. [`package_session.py`](package_session.py) validates explicit
routes, carries candidate state, resets input/output/workspace storage, guards
buffers and checks every output plus constant/input immutability. The result
includes positive and wrong-oracle ELFs, memory/instruction/symbol audits, source
bindings and a Merlin UART validation policy. It does not deploy a job.

Use the session verifier's existing route/fixture format, with real compiled
packages and independent references from the **selected** quantized captures:

```sh
export SESSION="$RUN/session-inputs"  # populated stage/route/fixture/provenance files
"$PY" examples/gemmini/comparisons/tvm/build_package_session.py \
  --tvm-source "$TVM" --stages "$SESSION/stages.json" \
  --route "$SESSION/route.json" --fixture "$SESSION/fixture.npz" \
  --fixture-provenance "$SESSION/reference.json" \
  --riscv-gcc "$GCC" --output-dir "$RUN/linked-session" \
  --rtol 1e-5 --atol 2e-6 --readback
```

`stages.json` is `{"version":1,"stages":{"prefill":"prefill-package","decode":"decode-package"}}`;
paths are relative to that file. Each route call selects a stage and lists input
edges as `{"fixture":N}` or `{"output":[earlier_call,output_index]}`.
`fixture.npz` contains dense `input_N` and `output_N` arrays, outputs flattened
in invocation/output order. `reference.json` binds `fixture_sha256`, the route
path/SHA, `reference_contract` and `reference_sources` (path/SHA records).
Use independently admitted references; a provenance string alone cannot prove
their independence. Read tolerances from the selected numerical policy.

The bridge has local host tests and a real two-package Spike control: four
calls/eight outputs, independent NumPy reference, warm/reset/measured execution,
full readback, and wrong-oracle rejection after successful warmup. The same
control passes with constants/references at 9 GiB and session buffers at 13 GiB,
using an explicit 16 GiB guest span starting at `0x80000000`. This is a
small recurrent integration proof, not model or FireSim qualification. The v2
handoff artifact retains the exact source, ELF, simulator and reference hashes.

For large models, select `--weights-base ADDRESS --arena-base ADDRESS` together
with `--memory-bytes BYTES`. Both bases must be page aligned, inside the declared
map and leave room for code/stack. `session_memory.py` places stage constants and
input/reference blobs in Merlin's existing `weights_blob.o` region, and reserves
guarded buffers/workspace in an explicit NOLOAD arena. The final link checks
code/stack bounds and exact weight extent; ELF audits include the arena. The plan
rejects overlap, overflow and unsupported older packages before linking.

Absolute mode requires packages whose graph manifests declare
`constants_binding.absolute_address_macro=TVM_MODEL_CONSTANTS_ADDRESS`. It
recompiles only their generated `model.c` wrappers with literal constant bases;
the original scheduled `operators.o` and constant bytes are reused. Near mode
continues to link the packaged objects directly. Package hashing is streamed and
final ELF audits memory-map the image instead of copying multi-GB blobs into
Python. Record the selected bases and qualification; the small high-address
control does not establish full-model capacity or FireSim loader support.

Current limits: all invocation outputs and their references remain in memory
for validation outside timing; the report explicitly accounts for this cost.
The default near-code layout admits at most 2 GiB; larger maps require the
explicit absolute layout above. Stages must use the same capability,
runtime and adapter. Float16 is refused; FP32, FP64, BF16 and integer/boolean
storage are supported. Full-model capture-to-package intake, actual large-model linking,
selected-reference correctness and hardware execution remain unfinished.
`prepare_firesim.py` still stages **isolated block** ELFs only.

Use TVM `qualify_package.py` for an independently referenced stage and
`verify_tensorir.py run` for Spike/RTL positive and wrong-oracle controls. Then
verify the entire linked session on the selected functional engine. Check every
output and state at every invocation, not just a final cosine. Retain the preserved
SmolVLA diagnostic failure and establish correctness for the selected EL4 capture.
Keep source/quantization quality and compiler
agreement as separate verdicts. Spike gives functional evidence, not hardware cost.

After execution, export complete UART outputs for the shared offline checker:

```sh
"$PY" examples/gemmini/comparisons/tvm/read_package_session.py \
  --session "$RUN/linked-session/session.json" --uart "$RUN/uart.log" \
  --validation-policy "$RUN/linked-session/validation-policy.json" \
  --output-dir "$RUN/readback"
```

[`read_package_session.py`](read_package_session.py) reuses Merlin's OUT_BIN
decoder and warm/measured UART validator. It checks checksums, completeness,
storage ABI, successful exit and the policy's session-manifest hash before
writing one NPZ per invocation with `out0`, `out1`, etc. `readback.json` records
the stage name, zero-based stage invocation, array metadata and file hashes.
It preserves scalar shapes, integer widths and floating bit patterns; BF16 uses
explicit `raw_uint16_bfloat16` storage, which needs a BF16-aware consumer.
The selected EL4 interfaces use FP32/int64/bool and need no BF16 conversion.

Use these `steps` entries as the actual-array records in the
[baseline output manifest](../../protocol/baselines.md), pairing each with its
independently admitted quantized reference and binding the selected inputs,
protocol and numerical policy. Run `python -m merlin.model_check.baseline_numerics`
as described there. Export alone does not compare references, identify the
executed ELF/device, or qualify timing; retain the original execution receipts.
The exporter has local malformed-log tests and exact all-output comparisons
against the two-stage near/absolute Spike controls, not full EL4 model evidence.

## Bind the actual FireSim device and run complete chains

The pinned EL4 workload selects the full one-hart RV64GC U250 design, with no RVV.
Retain its hardware identity separately from the compiler and numerical release:

| Artifact | Selected identity / remaining gate |
| --- | --- |
| Hardware pin | `firesim_gemmini_rocket_u250_30mhz` in `merlin/contract/hardware_pins.yaml` |
| Legacy queue hw_config | `alveo_u250_firesim_gemmini_rocket_30mhz`; only relevant to that deployment |
| Bitstream archive SHA-256 | `0af6f26e1cbad1c03f9e5873217505a5100bf2d0d068205cfa486522b24fc603` |
| Elaborated design SHA-256 | `7b6280272c7d8db3e013c7e5fd07592d2fd317a42c669e8b3b08f29c32944d08` |
| Params header SHA-256 | `3758ae967af3a179497660970201093a7fb624be00173990ce33d3f5c38da924` |
| DTS SHA-256 | `c4a2009a405945ea98f731713e0f4e7dc483031ab8cd5156d871e4eb7594f1df` |
| Matching driver | Bind and hash the actual manager driver; the delivery's TVM target record is null |
| DRAM | Declared 16 GiB starting at `0x80000000`; qualify the actual manager/loader address map |

Header agreement alone does not establish whole-SoC or memory-timing equivalence.
If the execution manager uses F2 or another image, record a distinct hardware
identity and qualify its loader, memory map and source correspondence. Do not
transfer old U250 cycle claims to it. FPGA host MHz and simulation wall time are
not the architectural target-clock denominator. See the pinned
[hardware selection](https://github.com/ucb-bar/merlin/blob/0cc289de52a10bed256f32c90cfbb54a21b3c81f/examples/gemmini/protocol/experiment.yaml#L42-L50)
and [FireSim clock guidance](https://github.com/ucb-bar/merlin/blob/0cc289de52a10bed256f32c90cfbb54a21b3c81f/examples/gemmini/validation/README.md#L154-L159).

The current [post-freeze FireSim workflow](https://github.com/ucb-bar/merlin/blob/0cc289de52a10bed256f32c90cfbb54a21b3c81f/examples/gemmini/validation/README.md)
uses direct ELF transport through locked SSM, serialized one complete model at a
time. It does not require Merlin's old queue or per-model registration. Its modes
have different evidence requirements; choose explicitly before building the guest:

| Mode | Implemented path | Remaining deployment/integration work |
| --- | --- | --- |
| Strict version-pinned provenance | `prepare` returns `staging_module`, then `elf_upload` uploads and re-downloads versioned ELF/request objects and writes an **unsent** SSM plan. The manager records the actual boot file before/after execution and the runtime/cleanup outcome. | Install and pin the supplied manager wrapper and SSM document version. The supplied manager-result publisher requires compact `all_element_comparison_v1`; a full-array request passing prepare is not evidence that this publisher supports it. |
| Lightweight `booted_elf_hash_v1` | `prepare` returns the installed `fpga-run-elf ELF NAME` launch arguments. Collection verifies an actual runtime-host boot-file hash, simulator `+prog0` path/argv, UART and compact roster. | Retain the real `booted-elf.json` plus SSM/upload evidence. Requires compact complete-chain transport; no strict-manager installation or synthetic `result_manifest.json` is required. |
| Legacy direct full-array transport | Without strict version pins or lightweight selection, the old helper and manager output-file manifest remain usable. | Produce the source-bound chain manifest, complete arrays and manager records; separately check timing and teardown. This mode does not establish the newer boot-file custody guarantee. |

Sources: [strict manager integration](https://github.com/ucb-bar/merlin/blob/0cc289de52a10bed256f32c90cfbb54a21b3c81f/examples/gemmini/validation/manager-provenance.md),
[lightweight boot-file contract](https://github.com/ucb-bar/merlin/blob/0cc289de52a10bed256f32c90cfbb54a21b3c81f/examples/gemmini/validation/booted-elf-provenance.md),
[publisher's compact-only admission](https://github.com/ucb-bar/merlin/blob/0cc289de52a10bed256f32c90cfbb54a21b3c81f/packages/merlin-experiments/src/merlin_experiments/post_freeze/manager_result.py#L215-L216).

**TVM-specific gap:** the existing builder emits `tvm_merlin_package_session_v1`
and `TVM_SESSION_*`/warm-profile output, optionally with full `OUT_BIN` arrays.
The newer chain path expects `merlin.post_freeze.firesim.chain_image.v1`, retained
native build/ABI evidence and `CHAIN_*` compact comparisons. Renaming the schema
does not make the pointer-array TVM ABI into Merlin's native memref ABI. Extend
the existing comparison bridge to bind its actual source, packages, routes,
references and ELF to the transport contract, and add compatible compact
comparison or explicitly integrate the supported full-array mode. No such
adapter is delivered yet. The native builder and its checks are in
[chain_images.md](https://github.com/ucb-bar/merlin/blob/0cc289de52a10bed256f32c90cfbb54a21b3c81f/docs/reference/chain_images.md);
the current TVM artifacts are generated by [build_package_session.py](build_package_session.py)
and [package_session.py](package_session.py).

Compact validation still checks **every element** of all 133 declared arrays
against admitted references; it sends errors/counts/digests instead of every
tensor byte. A digest alone is not the numerical check. This avoids making large
KV readback through UART a prerequisite for each benchmark pass.

1. Copy [firesim_workhorse.yaml](../../validation/firesim_workhorse.yaml) to the
   run directory and fill its compiler freeze, manager contract and per-model
   assets. Each model needs a complete-chain ELF, matching `chain_manifest`,
   input, independent reference and build/source/placement receipts. For TVM,
   bind the actual frozen TVM compiler and its freeze record; the generated V26
   compiler's capsule receipt does not qualify this baseline.
2. Audit the final ELF and memory map. Include code/stack, weights and packed
   duplicates, input and output buffers, KV caches, host scratch, workspace and
   retained outputs and references. Qualify the required high addresses through the
   actual loader. The small Spike high-address control is not that qualification.
3. Prepare the request and resolve every reported blocker:

   ```bash
   "$PY" -m merlin_experiments.post_freeze.firesim prepare \
     --selection "$RUN/selected-firesim.yaml" --out "$RUN/request.json"
   ```

4. Use the selected mode. For strict provenance, stage with the existing tool:

   ```bash
   "$PY" -m merlin_experiments.post_freeze.elf_upload "$RUN/request.json" \
     --model resnet50 --output "$RUN/resnet50-upload" --upload
   ```

   This command **does write to S3**; prepare/collect do not. Staging does not
   send its SSM plan. Submit that exact plan through the installed locked path
   only after the model and deployment gates pass. For lightweight/legacy mode,
   use the request's `launch_argv`. Keep original upload/SSM and full UART
   evidence, plus the selected mode's actual boot/manager records. The installed
   old helper's truncated stdout, unbounded error polling and incomplete shell
   failure/cleanup reporting remain documented limitations; the strict replacement
   already implements additional execution/cleanup checks. Local tests do not
   prove that replacement is installed on a manager.
5. Import the complete result for each model:

   ```bash
   "$PY" -m merlin_experiments.post_freeze.firesim collect \
     --request "$RUN/request.json" --result-dir "$RUN/model-result" \
     --ssm-invocation "$RUN/ssm-invocation.json" \
     --ssm-command "$RUN/send-command.json" --elf-upload "$RUN/elf-upload.json" \
     --out "$RUN/observation.json"
   ```

6. For full arrays, use `read_package_session.py` and the shared baseline checker;
   map those files into the actual manager roster. For compact transport, retain
   the admitted reference bindings and complete guest comparison transcript.
   In both cases, retain independent instruction/placement/exact-kernel checks.
   `collect` intentionally leaves `model_validation_complete: false`, including
   after compact comparison succeeds. Combine evidence in the final report.

The session must execute the entire chain in one guest run, carrying its own
computed state. The TVM bridge currently supplies one warm and one measured
complete-session invocation with reset and aggregate counters; add per-stage
counters and explicit repetition support for any different selected run policy.
Conversely, native `CHAIN_STEP` windows exclude input transfers and descriptor
preparation: summing them is not a full-chain E2E measurement. Preserve both
whole-chain and stage windows, separating initialization and output publication.
No larger calibration or quality dataset is required just to repeat these
inference passes. The E2E scope includes CPU work, transfers and all recurrent
calls, as specified by the
[performance contract](https://github.com/ucb-bar/merlin/blob/0cc289de52a10bed256f32c90cfbb54a21b3c81f/examples/gemmini/protocol/experiment.yaml#L204-L212).

The older `merlin.perf.firesim_checkpoint` queue workflow remains documented in
[the FireSim guide](../../../../docs/guides/firesim.md) for deployments explicitly
using it. Its queue lock and receipt rules apply there; do not mix them into the
direct workhorse workflow or claim that a direct observation is a queue receipt.

## Completion criteria for the next agent

For each actual workload retain checkpoint/input/export/program/package/ELF hashes,
source and quantized references, full output, state and reset checks, unchanged quality
metrics, a meaningful wrong-oracle control, peak memory accounting, hardware and
driver identities, job ID, submitted argv, client/daemon/UART logs and sealed
receipt. Declare exactly what the timing window includes: all stages, CPU work,
packing, state updates, completion fences and repetition policy. Do not report
summed operator minima as measured E2E latency. Preserve failed cases and CPU work
in the report. Compare only matching hardware and numerical/timing contracts.

Optional later tuning priorities are real-stage coverage, more post-XGB-training
search batches, CPU-only task tuning, a FireSim measurement adapter, full-session
memory/residency planning and independently tested new legal sketch families.
These are not prerequisites for collecting the first correct complete-chain runs.
The detailed ownership/optimization chart is in TVM `apps/gemmini/OWNERSHIP.md`;
the extension plan and cost-model caveats are in its `TUNING.md`.
