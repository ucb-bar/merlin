---
title: Set up Phase 0 and Phase 1
kind: guide
status: current
owner: experiments
last_verified: 2026-10-14
related: [gsim_worker_setup, neutral_runtime_tooling, phase0_specification, target_resolution]
code_refs: [src/merlin/runtime/backends/rocc_selection.py, src/merlin/runtime/backends/chipyard_rocc.py, src/merlin/targetgen/generalization_prompt.py, packages/merlin-experiments/src/merlin_experiments/cli.py, packages/merlin-experiments/src/merlin_experiments/corpus/preparation.py, packages/merlin-experiments/src/merlin_experiments/phase1/timing.py, packages/merlin-experiments/src/merlin_experiments/phase1/feedback/private_source_demand.py]
---

# Set up Phase 0 and Phase 1

Select and record an exact Merlin revision before preparing inputs. Install core
and `merlin-experiments` from that revision, then qualify the selected tools and
runtime on the execution worker. See [gSIM worker setup](gsim_worker_setup.md)
for the simulator build and deployment procedure.

Use the existing catalog, corpus release, native readiness, preflight and canary
paths. No additional component qualification ledger is a launch prerequisite.
Do not use a handwritten compiler, compiler-bearing support, its vendor kernel
headers, final-validation programs or performance notes to prepare author inputs.

## 1. Review the execution declaration and pin its actual files

The selected capability contract must describe hardware capabilities and software
semantics independently of a solution. The retained target examples are
diagnostic prototypes. Do not copy their legacy harness includes, issue order,
padding or preferred lowering into a fresh compiler experiment.

The [deployment template](../../examples/gemmini/phase0/execution-declaration.template.yaml)
names the planned `GemminiGsimFireSimAlignedConfig` for the capsule TestHarness.
This new Chipyard configuration must be implemented and elaborated before use;
the template is deliberately incomplete:
replace every marked field and attach freshly derived capability declarations.
Its facts must describe that exact configuration; the diagnostic example's
`GemminiRocketConfig` elaboration cannot qualify it. The checked-in templates
contain no compiler routines and cannot execute with their placeholder inputs.

Select the shared execution family and the complete logical ABI:

```yaml
runner:
  backend: chipyard_rocc
  chipyard_rocc:
    version: 1
    # config, toolchain and engines are explicit selections described below.
harness_abi:
  version: 2
  kind: logical_pointer
  entry_symbol: kernel_entry
  fence_symbol: kernel_complete
  tensor_alignment: 64
  byte_order: little
  main_convention: void
  readback_transport: out_b64_v1
```

This fragment is a calling convention, not a complete execution declaration.
Review each of these required fields before preparing the contract:

| Block | Required explicit selection |
| --- | --- |
| Contract identity | `name` matching the target and selected facts |
| Capabilities | A nonempty reviewed `compute_units` block, derived dtypes/operation declarations, and `endpoint_kind: inline_asm_insn`; see the [contract schema](../../merlin/schemas/target_contract.schema.yaml) |
| Configuration | `runner.chipyard_rocc.config` matching the facts' source and consistency records |
| Toolchain | `compiler` and `link_script` file pins; `runtime_units` and `headers` core IDs; `load_address`, `cflags`, `ldflags`, `kernel_stack_frame` |
| Engine roster | Only the selected `spike`, `gsim` or `verilator` engines; each has `binary`, `argv`, `environment`, `cwd`, `max_timeout_s`, `max_console_bytes`, `failure_markers` |
| gSIM | A `receipt` file pin binding that exact executable and FIRRTL through the existing strict-v3 receipt |
| Verilator | A `receipt` file pin using `merlin.elaborated_rtl_build.v1`, joining actual observed FIRRTL, model, transport and final executable build stages |
| Spike | `runner.spike_extension` with `extension_name`, `extlib` and its file hash; explicit `--isa=` in engine argv |
| Operand interpretation | `rocc_operand_roles` v1: `word_bits`, and instruction rows containing `funct`, `class`, `operands` |
| Instruction policy | The selected decoded instruction names and public endpoint role declarations must resolve the experiment's prohibited roles; missing role evidence is unmeasured, never an empty successful scan |
| RTL checks | Explicit `rtl_checks`; only supported hardware-legality checks, never lowering strategy checks |

An instruction operand selects `rs1` or `rs2` with `{bundle: EXACT_RTL_BUNDLE}`.
The selected `register_bundle_layouts` supplies its offsets and widths. Missing,
ambiguous or overlapping layouts refuse. Do not invent a layout when extraction
cannot resolve it. `legal_funct` additionally needs a complete decoder taxonomy;
a partial observation cannot become complete through a declaration. Unknown
source consistency likewise remains unknown after pinning.

Core runtime IDs are `startup`, `htif_console`, `printf`, `libc`; header IDs are
`htif`, `out_b64`, `out_bin`, `out_bin_memory`. Choose the actual dependencies;
B64 requires `startup`, `htif_console`, `htif`, `out_b64`. Select `libc` for its
scalar primitives; do not also select `printf`, which defines overlapping symbols.
Core resources supply
startup, console and scalar library primitives only. They supply no device
implementation. Engine argv contains exactly one whole `{elf}` token. Select
flags and budgets explicitly; the maximum engine deadline is 600 seconds.

Use absolute canonical paths in the deployment declaration. File pins have
`{path: /absolute/file}` and may include an already selected `sha256`. Core
resource rows have `{id: IDENTIFIER}` and may include `sha256`. Only missing
hashes are filled; existing hashes are checked, never overwritten:

```sh
python -m merlin.runtime.backends.rocc_selection \
  --target "$TARGET" --contract "$DEPLOYMENT_DECLARATION" \
  --facts "$RTL_FACTS" --output "$PREPARED_CONTRACT"
export MERLIN_TARGET_CONTRACT="$PREPARED_CONTRACT"
export MERLIN_RTL_FACTS="$RTL_FACTS"
unset MERLIN_TARGET_PATH
```

`PREPARED_CONTRACT` must be a fresh file beneath the configured
`out/artifacts` root. Review its exact bytes before using it. Release preparation
grants the staged capability contract publicly, including its execution block:
it must contain no credentials, private answers or compiler code. Binaries,
receipt files, native probe packages and private validation artifacts remain
operator-owned; their paths in data do not grant filesystem access.
Preparation performs file/receipt checks only and reports `native_executed: false`.
It does not run a compiler, simulator, source extraction or capture.

## 2. Select one verified Phase 0 input closure

Prepare an operator-owned definition selecting all of the following together:

Start from the [ordinary verified wiring template](../../examples/gemmini/phase0/verified-experiment.template.yaml),
replace its paths and selections, and review the evidence below. It is outside
the catalog and supplies no captures, private answers or readiness verdict. Keep
the checked-in template inert. In the operator-owned reviewed copy, set
`kind: experiment` after reviewing the actual closure; the runner refuses to
execute a definition that still says `kind: template`.

The descriptor's `hardware_spec.target_contract` must name the complete prepared
deployment contract before derivation. The frozen release preserves this exact
selection; execution fields cannot be added after sealing. Add only reviewed
public primitive ISA/facts data to its
hardware grants; there is no vendor kernel-header requirement. Release preparation
rewrites that declaration to the staged contract automatically.

| Input | Required evidence |
| --- | --- |
| RTL and facts | One source selection, configuration and freshly checked extraction bytes |
| Execution contract | The reviewed neutral contract from step 1, used for derivation and grading |
| Software spec and host semantics | Reviewed operation/numerical/placement declarations, with unknowns explicit |
| Model2MLIR | An exact source revision and selected capture runtime |
| Iteration programs | Independent public programs and their complete capture, source, input and weight closure |
| Conformance and synthesis | Requirements and synthesis profile derived from that same selected closure |
| Hidden cohort | An operator-owned private selection, outside public derivation and author grants |
| Phase 0 mode | `evidence_mode: verified`, after the input evidence is established |

Use the existing [sealed capture and requirement derivation
procedure](../../examples/gemmini/phase0/README.md#3-capture-the-independent-iteration-workloads).
Select every iteration roster label and verify its original attestation; a
diagnostic copied model or an API availability check is insufficient. The public
roster may exercise the same operation families as validation, but must not contain
validation programs, exact held-out layer shapes or their derived implementation
tips. Keep private validation capture/weight closure available only to the evaluator.

The definition uses `capsule_derivation` for Phase 0 and `capsule_bench` for
Phase 1. Select absolute reviewed paths for the recipe, software spec, hardware
spec, capability contract, RTL facts, conformance spec, synthesis profile and
hidden profile. The CLI's `--phase0-*` selections may bind these fields explicitly;
use the identical selections through inspect, preflight and run. A bare change
from `diagnostic` to `verified` does not repair missing evidence.

## 3. Derive, review and seal the release

Set `SPEC` to that reviewed definition, `CONFORMANCE` to its exact requirements,
and the other variables to fresh generated destinations. Run these commands on
AWS, after its inputs are prepared:

```sh
merlin experiment inspect "$SPEC" --phase 0 --run-dir "$P0"
merlin experiment preflight "$SPEC" --phase 0 --run-dir "$P0"
merlin experiment run "$SPEC" --phase 0 --run-dir "$P0"
merlin experiment corpus coverage "$P0" --spec "$CONFORMANCE"
merlin experiment corpus prepare "$P0" --generated-only \
  --private-baseline "$PRIVATE_HIDDEN_ROOT" --output "$RELEASE"
merlin experiment corpus inspect "$RELEASE"
```

Resolve mandatory missing obligations, writer failures and omissions before
review. Inspect functional and performance coverage separately; a large member
count does not prove either. Preserve the exact source/capture identities and
reviewed cohort separation. Preparation must regenerate this release's descriptor
and bundles; never select an older example bundle or a vendor library include.

Seal only after reviewing the inspected digest:

```sh
merlin experiment corpus seal "$RELEASE" \
  --expected-digest "$REVIEW_DIGEST" \
  --reviewed-by "$OPERATOR" --review-note "$REVIEW_NOTE"
```

Preparation stages the selected contract and facts into the release. Select
those paths before Phase 1 inspection, probe, preflight and canary:

```sh
export MERLIN_TARGET_CONTRACT="$RELEASE/payload/experiment/contracts/target_contract.yaml"
export MERLIN_RTL_FACTS="$RELEASE/payload/experiment/rtl_facts/facts.json"
```

Keep the Phase 0 definition and its original selections for Phase 0 only. Phase 1
uses the regenerated descriptor and bundle. An ambient original contract path
conflicts with that descriptor even if its bytes match; the refusal is intentional.
Native startup later restores facts from its verified frozen bundle snapshot.

## 4. Run the independent native probe and startup checks

On the same worker and engine, select an operator-only primitive probe written
from the public semantics and ISA. It must not import the golden compiler or
serve as an author seed. Exercise the existing native grade and linked-ELF policy:

- Complete nonzero correct outputs pass the original numerical comparison.
- An incorrect output fails that same comparison.
- A forbidden hardware-loop instruction in the linked ELF is rejected.
- A genuine cert-tier execution produces the target/config/engine/binary-bound
  oracle timing record; gSIM and Verilator also bind their selected receipt and
  original FIRRTL through the explicit `MERLIN_RTL_FACTS` selection.

Follow [native readiness and timing](gsim_worker_setup.md#prepare-catalog-experiments).
Spike observations are numerical evidence only. Source tests, successful linking,
a copied console or a hand-written cycle number cannot replace native observations.

Use the focused existing oracle check with explicit private probe selections:

```sh
MERLIN_REPO_ROOT="$OPERATOR_ROOT" \
MERLIN_TARGET_EXPERIMENT="$RELEASE/payload/experiment/target_experiment.yaml" \
"$OPERATOR_PYTHON" "$OPERATOR_ROOT/merlin/experiments/capsule_bench/harness/readiness_check.py" \
  --oracle-probe-only --reference-backend "$POSITIVE_PACKAGE" \
  --probe-capsules-root "$PROBE_ROOT" \
  --screen-probe "$SCREEN_NAME" --timing-probe "$TIMING_NAME" \
  --incorrect-output-backend "$INCORRECT_PACKAGE" \
  --prohibited-instruction-backend "$PROHIBITED_PACKAGE" \
  --prohibited-probe "$POLICY_NAME" --probe-timeout-s "$PROBE_TIMEOUT" \
  --oracle-timing-output "$ORACLE_TIMING"
```

Each package uses the existing `mlir_oot_target_backend` manifest/tool ABI and
`integrity_exempt: false`; the positive package may implement only these primitive
probes. Capsule names select direct members of `PROBE_ROOT`. Every selected
capsule must declare the same nonempty candidate instruction policy, including
the campaign's prohibited hardware-loop role. The numerical negative grades the
same timing capsule on the same cert engine and must fail with actual numerical
mismatches and no missing outputs. A build failure, unsupported kernel or empty
output cannot satisfy it. The policy negative must contain a prohibited
instruction in the linked ELF and fail the existing whole-program policy.

Choose `PROBE_TIMEOUT` in 1–600 seconds, within every selected engine's declared
budget. This existing gate runs its screen on Spike and its timing on the selected
cert engine; select both execution inputs for this route. The parent of `ORACLE_TIMING`
must already exist beneath an operator
artifact directory; the file must be absent and outside the released descriptor
resources. This prevents timing publication from changing the sealed payload or
overwriting an old observation. The command uses the existing timing writer only
after the correct cert grade, screen and both negative controls succeed. Its
`ORACLE PROBES PASS` verdict covers this finite gate only, not full launch readiness.

When the grading cohort contains model capsules, select an absolute executable
host Python with Torch, NumPy and safetensors compatible with the capture runtime:

```sh
export MERLIN_M2M_PYTHON="$MODEL_PYTHON"
```

Set `MODEL_PYTHON` to the reviewed model runtime on this worker. Phase 1 probes
and records that interpreter before authoring. `MERLIN_COMPILER_PYTHON` separately
selects the interpreter for upstream host MLIR bindings; both selections must be
explicit when the deployment uses both roles. The explicit scalar
[stock tool transport](llvm_toolchain.md#scalar-compilation-without-python-mlir-bindings)
can compile host IR without those bindings. It refuses Python-runner-only options
and does not change the candidate compiler's own interpreter or qualification.

The RTL authoring tools also require an executable named `FileCheck` on the host
and sandbox PATH. Some LLVM distributions omit it or install only a versioned
launcher. Select `FILECHECK_BIN` as the directory containing the actual unversioned
binary, then keep it on PATH for preflight and authoring:

```sh
export PATH="$FILECHECK_BIN:$PATH"
```

For the Gemmini six-program full-model gate, the authored descriptor now selects
the full FireSim board even when the program executes on Spike. Prepare a **new**
private model specification using that board, its FireSim DTS and the newly
frozen board catalog. Before launch, run the operator alignment check on the
FireSim source checkout selected for the pinned U250 bitstream:

```sh
python -m merlin_experiments.phase1.spike_alignment \
  --contract "$PREPARED_CONTRACT" \
  --chipyard-root "$FIRESIM_CHIPYARD_ROOT" \
  --machines examples/gemmini/phase2/whole-model-machines.yaml \
  --hardware-pins merlin/contract/hardware_pins.yaml \
  --board-catalog examples/gemmini/platform/board-catalog.yaml \
  --private-model-spec "$PRIVATE_MODEL_SPEC"
```

Require `spike_aligned: true`. This checks the exact Spike executable and Gemmini
extension, their source commits, full-design ABI header, FireSim DTS and 16 GiB
board declaration, and every model row's board/DTS/catalog pins. It does not
prove gSIM/FireSim RTL equivalence or that the FPGA can use the entire declared
memory span; qualify those independently. A missing pinned checkout or DTS is a
failed selection, not a reason to reuse the Verilator DTS.

For capsule execution, select `GemminiGsimFireSimAlignedConfig` in the Phase 0
execution contract and derive facts from its exact TestHarness FIRRTL. Build it
from the [pinned configuration source](../../build_tools/chipyard/GemminiGsimFireSimAlignedConfig.scala)
added unchanged to a clean descendant of the FireSim-recorded Chipyard commit,
with the Gemmini submodule still at its recorded commit. The gSIM alignment
audit permits that one committed Chipyard source addition and no other source
changes. This configuration must preserve the gSIM harness
clock and serial adapter while matching FireSim's chip-visible 16 GiB memory
map, Rocket ISA and execution-relevant bus/peripheral settings. The older
`GemminiGsimSerialClkConfig` only fixes the harness clock path and is not enough.
Supply a reproduced source-to-FIRRTL elaboration
receipt to the RTL source selection, then build the gSIM emulator with a strict
v3 receipt. Before launch, check the complete chain:

```sh
python -m merlin_experiments.phase1.gsim_alignment \
  --contract "$PREPARED_CONTRACT" --rtl-facts "$PREPARED_FACTS" \
  --chipyard-root "$FIRESIM_CHIPYARD_ROOT" \
  --hardware-pins merlin/contract/hardware_pins.yaml \
  --board-catalog examples/gemmini/platform/board-catalog.yaml \
  --gsim-dts "$GSIM_DTS"
```

Require `gsim_source_aligned: true`. This checks the generated gSIM and FireSim
DTS memory maps and ISA as well as source/FIRRTL/emulator identities. The
current installed gSIM uses the older Gemmini source, and the available
`GemminiRocketConfig` DTS declares only 256 MiB, so it cannot satisfy this gate.
A passing source gate still needs an
explicit chip comparison and matched-ELF gSIM/FireSim checks before its cycles
can be treated as representative board timing.

Then run ordinary `--preflight-only` with the released descriptor, seal, regenerated
bundle and oracle timing. Run the fresh Codex client canary in the same sandbox
with the same provider/model/tool selections and a separate fresh run ID. See
[the exact preflight/canary commands](../../packages/merlin-experiments/README.md#sealed-fresh-codex-client-check).
Preflight must stop before authoring; the canary is only a bounded client check.

## 5. Author experiment start condition

Start Phase 1 only after the reviewed verified Phase 0 release, all four native
probe observations, ordinary native preflight and fresh-client canary pass on
the selected worker. Preserve their exact digests, selections and logs. A failure
requires fixing the relevant input or tool and re-running its check, not editing
a verdict. This permits a compiler author experiment; it does not guarantee its
functional success or final performance.

## Compiler completion requires full accelerator coverage

Every computation that can legally execute on the selected accelerator under the
original hardware and numerical contract must lower to it. Derive eligibility
from the original source independently of the candidate's placement and available
compiler forms. Missing compiler forms, large shapes, code-size limits and
timeouts are compiler gaps; they do not justify supported computation on the
host. Mixed groups may retain host orchestration and independently justified
unsupported postprocessing while their supported tensor computation executes
on the accelerator. Independently proved elimination may discharge source work.

The whole-model source gate records demanded, requested, lowered and covered
groups, justified host work and unresolved source work. Unknown eligibility or
missing accelerator work prevents completion. Linked coverage, source-bound
device evidence, numerical correctness and required structural checks are
separate requirements. Passing small public cases alone establishes neither
whole-model completion nor generality across unseen shapes and compositions.

Use compile-only checks for large and unseen shapes without allocating tensor
values or running simulations. Use bounded L2 execution for numerical checks
and bounded L3 cases for required native modes, layouts, transfers, accumulation
and tails. Whole-model L2 success remains distinct from whole-model RTL
qualification; select and report each required scope explicitly.

## Qualify the compiler before performance search

Keep host compilation in Merlin: source normalization, host arithmetic and packing,
intermediate buffers, dispatch, linking and runtime use the selected fixed host
provider. The candidate supplies device implementations and target constraints.
A mixed source function may need host-owned partitioning before its device regions
can lower. Check connected public programs through this production path, including
host/device dependencies, shared inputs and multiple outputs. A standalone device
reader's refusal is distinct from a failure of the composed model compiler.

Device-object builds use the same checked package lowering as capsule builds.
The command-buffer schema, logical dtypes and explicit refusal are checked before
an artifact reaches translation. An emitted LLVM stub or successful process exit
cannot override a declared refusal. Numerical execution and source accounting
remain necessary to detect incorrect accepted implementations.

Public authoring readiness and formal qualification are separate results. Freeze
the compiler before private validation. Require all declared validation programs,
their linked images and their existing numerical contracts; preserve instruction
policy, source placement and hardware qualification. Do not return private model
diagnostics to the authoring loop.

Normal retained-revision Phase 2 search reopens the original formal qualification
against the descriptor selected by Phase 1. It requires complete model builds and
execution of every required program, with no functional waivers or deferrals.
It reads existing evidence without launching another model run. Historical
capsule-only or explicitly waived diagnostics remain inspectable, but do not
provide this normal search baseline. A copied compiler bundle without its run
evidence is a diagnostic candidate, not a qualified handoff.

On the experiment worker, use the existing preflight and launch commands linked
above with a fresh run ID and the sealed release. After authoring, retain the
complete formal run directory, compiler digest and frozen host/tool profile.
Pass that exact run ID and digest to Phase 2. Compile-only checks, unit tests and
synthetic receipts do not substitute for an executed compiler qualification.

## Reserve CPU pools before the fresh launch

On a 32-CPU worker, an operator may reserve CPUs 8–30 for the experiment and
leave 0–7 and 31 for other workloads. Within that reservation, pin the manager
to 8–9, select `MERLIN_CODEX_CPUS=10,11,12,13`, and select
`MERLIN_BUILD_CPUS=14,15` with an absolute task-local `MERLIN_BUILD_SLOT_ROOT`.
Native builds acquire an exclusive CPU slot across processes and threads;
slot waits consume the existing build timeout. Codex subprocesses inherit the
author pool at startup, while the manager's launch thread restores its affinity.

Select `MERLIN_GSIM_CPUS=16,17,18,19,20,21,22,23,24,25,26,27,28,29,30`,
`MERLIN_GSIM_MAX_SLOTS=15`, and one shared `MERLIN_GSIM_SLOT_ROOT` for all runs
using these simulator CPUs. Set `MERLIN_GRADE_WORKERS=15`; the default public
self-check now delegates worker selection to the host, while an explicit
`--workers` request retains its meaning. Phase 1 plans freeze these selections
for preflight and resume. Finish or relocate older jobs in the reserved pool
before launching the fresh experiment, and confirm actual child affinities and
simulator concurrency under the final launcher.

A Phase 1 operator can select `--sim-per-capsule-timeout SECONDS` to set the
async certificate worker budget explicitly (1–86400 seconds). The default,
`0`, retains the existing selected-engine derivation. This setting does not
override the sealed native process ceiling or `--model-budget-s`; select those
resource bounds consistently before launching. Expired budgets remain
unmeasured and never supply numerical or hardware qualification.
