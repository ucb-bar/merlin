---
title: Prepare Phase 0 and Phase 1 on an AWS worker
kind: guide
status: current
owner: experiments
last_verified: 2026-10-10
related: [aws_gsim, neutral_runtime_tooling, phase0_specification, target_resolution]
code_refs: [src/merlin/runtime/backends/rocc_selection.py, src/merlin/runtime/backends/chipyard_rocc.py, packages/merlin-experiments/src/merlin_experiments/cli.py, packages/merlin-experiments/src/merlin_experiments/corpus/preparation.py, packages/merlin-experiments/src/merlin_experiments/phase1/timing.py]
---

# Prepare Phase 0 and Phase 1 on an AWS worker

Fetch the completed `feat/compiler-readiness` handoff and record the exact commit
before preparing inputs. Install core and `merlin-experiments` from that same
revision using the [AWS gSIM guide](aws_gsim.md). Repository delivery and worker
qualification are separate: these commands prepare and test the actual worker
after delivery. They do not require a compiler author experiment during repository
development.

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
selects the requested `FireSimGemminiRocketConfig`. It is deliberately incomplete:
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
  readback_transport: full_values_b64
```

This fragment is a calling convention, not a complete execution declaration.
Review each of these required fields before preparing the contract:

| Block | Required explicit selection |
| --- | --- |
| Contract identity | `name` matching the target and selected facts |
| Capabilities | A nonempty reviewed `compute_units` block, derived dtypes/operation declarations, and `endpoint_kind: inline_asm_insn`; see the [contract schema](../../merlin/schemas/target_contract.schema.yaml) |
| Configuration | `runner.chipyard_rocc.config` matching the facts' source and consistency records |
| Toolchain | `compiler` and `link_script` file pins; `runtime_units` and `headers` core IDs; `load_address`, `cflags`, `ldflags`, `kernel_stack_frame` |
| Engine roster | Only the selected `spike` and/or `gsim` engines; each has `binary`, `argv`, `environment`, `cwd`, `max_timeout_s`, `max_console_bytes`, `failure_markers` |
| gSIM | A `receipt` file pin binding that exact executable and FIRRTL through the existing strict-v3 receipt |
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
B64 requires `startup`, `htif_console`, `htif`, `out_b64`. Core resources supply
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

The descriptor's `hardware_spec.target_contract` must name the prepared contract
before derivation. Add only reviewed public primitive ISA/facts data to its
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
  oracle timing record; gSIM also binds its receipt and FIRRTL.

Follow [native readiness and timing](aws_gsim.md#prepare-the-catalog-phase-0-and-phase-1-handoff).
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

Then run ordinary `--preflight-only` with the released descriptor, seal, regenerated
bundle and oracle timing. Run the fresh Codex client canary in the same sandbox
with the same provider/model/tool selections and a separate fresh run ID. See
[the exact preflight/canary commands](../../packages/merlin-experiments/README.md#sealed-fresh-codex-client-check-aws-execution).
Preflight must stop before authoring; the canary is only a bounded client check.

## 5. Author experiment start condition

Start Phase 1 only after the reviewed verified Phase 0 release, all four native
probe observations, ordinary native preflight and fresh-client canary pass on
the selected worker. Preserve their exact digests, selections and logs. A failure
requires fixing the relevant input or tool and re-running its check, not editing
a verdict. This permits a compiler author experiment; it does not guarantee its
functional success or final performance.
