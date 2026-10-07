---
title: Compiling a whole model onto an accelerator
kind: guide
status: current
owner: compiler
last_verified: 2026-10-06
related: [compilation_strategies, targetgen, adding_a_target, gemmini_experiment, reproducing_whole_model_on_rtl, firesim]
code_refs: [src/merlin/compile_cli.py, src/merlin/compile/command.py, src/merlin/compile/baremetal_model.py, src/merlin/compile/model_execution_inputs.py, src/merlin/llvmlower/group_offload.py, src/merlin/llvmlower/device_build.py, src/merlin/targetgen/coverage_certificate.py, packages/merlin-experiments/src/merlin/targetgen/native_model_execution.py]
---

# Compiling a whole model onto an accelerator

A whole-model ELF, a correct whole-model output, and proof that source work ran on an accelerator
are three different results. The saved-model CLI can build and run a bare-metal ELF on a selected
board, including a matching native RTL engine. By default it builds the **host baseline**; it does
not silently choose an accelerator placement. Even a complete native output match is a numerical
diagnostic for that ELF, not certification of an agent-generated backend or proof that all operations
were accelerated.

| Workflow | What it exercises | Evidence boundary |
|---|---|---|
| `merlin-compile --model-build` | one byte-verified saved capture, selected host package and board | one host-baseline ELF by default; optional complete-output check |
| `compile_saved_model(..., device=routing)` | the same build with an explicit `DeviceRouting` | a static device sidecar and optional complete-output check; dispatch still unproved |
| `compile_model(..., run="mesh")` | Python-driven whole-model dispatch to per-layer oracles | interpreted mesh/host composition, not one bare-metal ELF |
| OOT capsule grader | candidate command buffer and emitted artifact against frozen corpus/oracles | independent eligibility, transform, execution and numerical verdicts |

The older [whole-model RTL reproduction](reproducing_whole_model_on_rtl.md) documents a specific
Gemmini experiment, including an unresolved spike-versus-Verilator numerical difference. Its historical
kernel and model measurements must not be promoted into a current saved-model or candidate certification.

## Phase 1 completion is more than capsule success

A target's `phase1_gates.private_full_models` declaration supplies the required
validation-model and program rosters. The operator separately selects complete
checkpoint-backed captures and their source-execution attestations; those inputs,
their MLIR and build results remain private to the formal grader. They are not
derivation workloads or authoring-agent feedback.

The build-only gate checks every required program against the same frozen candidate
compiler: capture identity, independently derived accelerator eligibility, exact
group routing, justified host placement, code generation and a linked ELF. A missing
program, dropped eligible group or failed build prevents Phase 1 completion even
if all public and hidden capsules pass. Passing this gate is bounded static build
and placement evidence for those captures, not full-model numerical execution or
a proof that every possible model will compile. Representative kernels still
require their independent L3 numerical grade.

The private selection also records input provenance. A complete pretrained network
captured with synthetic entry inputs can supply structural build evidence, but
must retain its synthetic-input and paper-readiness declarations. The gate does
not turn that capture into attributed-data validation, a paper-accuracy result,
or whole-model numerical equivalence. Those require separate executed evidence.

## Select and reuse the inputs

Select an existing capture bundle with a materialized, byte-verified `capture_receipt.json`; the build
does not recapture a workload. Select the host package separately from any candidate device package.
Its `knobs.yaml` flags must supply one `-march` and an ABI compatible with the runner-owned harness;
the board's byte-pinned elaborated DTS must describe that CPU ISA, DRAM and hart. The current executor
requires a bare-metal HTIF board, one hart, a code reserve and one inference. It checks the linked
ELF ISA against the DTS rather than assuming the build machine's ISA.

Select target support explicitly with `MERLIN_TARGET_PATH` pointing at the intended trusted provider.
The core's target name or a generated candidate package is not a substitute for that provider. For
native `gsim` or `verilator`, also select `--rtl-facts` from the same elaborated target/config as the
board and the selected L3 engine. The executor binds the facts to their FIRRTL bytes and revalidates
the concrete engine provenance. The provider must expose a public `run_elf` implementation for the
selected engine; merely having an emulator somewhere on disk is insufficient. A target with another
selected L3 engine cannot use these two native choices through this command.

Keep the saved bundle, host package, board catalog, DTS, RTL facts and provider selection stable
across diagnostics. Each `--output` must be a **fresh directory below `MERLIN_OUT_ROOT`**; the receipt
pins input tree hashes, selected engine, ELF hash and console. Reuse a prior artifact as evidence only
with its identity and provenance intact. In particular, rebuilding separately for gSIM and Verilator
does not establish a *same-ELF* comparison unless their ELF byte hashes agree. Do not recapture or
replace a reference between runs and present the results as one comparison.

## Host-baseline diagnostic: build, then optionally run

Set each variable to an explicitly selected path or value. `REFERENCE_FILE` is the basename of a
receipt-bound `.npy` file **inside** `MODEL_CAPTURE`, not an arbitrary external golden. Choose it for
the arithmetic actually built; a capture's default `golden.npy` is not necessarily the right W8A8
reference. A compile-only run needs no reference.

```sh
export MERLIN_TARGET_PATH=/absolute/path/to/trusted-target-support
export MERLIN_OUT_ROOT=/absolute/path/to/generated-out

merlin-compile --target "$TARGET" --model-build \
  --capture-bundle "$MODEL_CAPTURE" --package "$HOST_PACKAGE" \
  --board-catalog "$BOARD_CATALOG" --board "$BOARD_NAME" --host-dts "$HOST_DTS" \
  --arena-mb "$ARENA_MB" --output "$MERLIN_OUT_ROOT/model-build-only" \
  --run none --json
```

`--run none` produces `baremetal_model.json` with status `compiled` and an ELF. It claims neither
execution nor numerical agreement. It can build a model whose output exceeds the execution gate's
current 4096-element limit.

To run a **separate fresh build** on the selected native engine:

```sh
merlin-compile --target "$TARGET" --model-build \
  --capture-bundle "$MODEL_CAPTURE" --package "$HOST_PACKAGE" \
  --board-catalog "$BOARD_CATALOG" --board "$BOARD_NAME" --host-dts "$HOST_DTS" \
  --arena-mb "$ARENA_MB" --output "$MERLIN_OUT_ROOT/model-native-diagnostic" \
  --run "$RTL_ENGINE" --rtl-facts "$RTL_FACTS" \
  --reference-file "$REFERENCE_FILE" --timeout "$TIMEOUT_SECONDS" --json
```

Here `RTL_ENGINE` is the selected `gsim` or `verilator`; `spike` is also supported without
`--rtl-facts`, but is not RTL evidence. Native availability and completion depend on the exact
provider/engine, ELF and cycle/time budget; the presence of this API is not a claim that a particular
whole model has passed. Execution currently requires one finite float32 output of at most 4096
elements and checks the **entire** `OUT` against the explicit reference, plus completion and build
identity. Status `verified_complete_output` means exactly that for this ELF/reference/engine.
Partial or timed-out simulator output is retained as diagnostic bytes in a failed receipt and never
counts as a numerical pass. `--no-verify` is not an execution escape hatch.

Both commands above omit `device`, so their receipt says `execution_route: host_baseline`. A correct
host baseline is useful for isolating the host ABI, memory map and native runner. It says nothing
about device dispatch. The `--model-build` CLI has no device-placement flag and does not consume an
OOT candidate's command buffer. Its flag rules live in `merlin.compile.command`; to stop a build at a
named stage or keep its IR, see [compile_debugging](compile_debugging.md).

## Candidate model diagnostic is a separate artifact

A candidate device route must be an explicit input to the Python
`merlin.compile.baremetal_model.compile_saved_model` API via `device=DeviceRouting(...)`. Derive the
routing from a placement and the saved capture (for example, `routing_for_placement`), with a selected
device package and declared granularity; do not manufacture `select=lambda shape: True` to make a
coverage number. The build records a device sidecar, but labels it
`device_requested_dispatch_unverified`: static calls or nonzero opcodes do not prove a completed
accelerator dispatch. The host baseline ELF cannot be relabeled as a candidate ELF.

The OOT whole-program route is different again. The grader's candidate diagnostic links the
candidate-emitted whole-program LLVM and command buffer against the frozen capture, rather than
substituting a core-generated host baseline. Its `numeric_match_diagnostic` status records a native
numerical comparison, **not** a final grade or device-execution proof. Keep the candidate artifact,
capture, selected provider, board and facts byte-identical when comparing engines or resuming a run.
Use the reviewed corpus and grader workflow for a verdict; a manually invoked native run is only a
diagnostic.

## Grouping and fail-closed routing

`compile_model(..., offload=True, offload_granularity="contraction")` moves only the contraction;
bias, requantization, activation and other readout remain host work. `"group"` requests one closed
compute group per device call. It must be built from that group's stated program, not just its
`(M,N,K)` extents. These are different programs and must not share a coverage label.

The group planner inventories source operations and reports each group as accepted or declined by
name; `require_every_group_accounted` rejects an incomplete census. A saved capture's weight manifest
is needed where operand roles cannot otherwise be distinguished. Missing data, unsupported dtype or
transport, absent program entries, or a missing backend package must remain named refusals, not
implicit accelerator successes. A route plan is a prediction; only a build with the selected routing
can emit the calls, and only completed runtime dispatches can establish their lane.

For an OOT whole-program backend, preserve each source `prov.region_id` and map source operations
exactly once through `params.global_program_plan.tasks` (`task_index`, `kind`, source indices) into
LLVM operations tagged `merlin.global_task`. These declarations help the independent grader join
source regions, outlined symbols and completed dispatch ledger entries. They do not self-certify:
missing or mixed provenance, unexecuted eligible siblings, failed/fallback calls and unaccounted host
work remain vetoes. A complete output match, a static sidecar, or an instruction count cannot waive
those obligations. See the shared [OOT backend contract](../../merlin/contract/mlir_oot_backend_contract.yaml)
and [target generation guide](targetgen.md) for the authoring boundary.

## What a full claim needs

Treat the following as separate evidence, joined on the **same saved inputs and candidate bytes**:

1. The selected board, DTS, host package and trusted target provider agree on ISA, ABI, memory and
   concrete RTL engine; the ELF and execution receipts remain byte-bound.
2. The whole candidate model completes and its entire output passes the declared numerical policy.
3. The independent grader accounts for every eligible source operation through transform replay,
   exact source-region/outline identity and completed device dispatch, while naming host fallback and
   unsupported work. `NOT_RUN` is not a pass.
4. The required oracle tiers, corpus, release and review policy for the intended claim have passed.
   A diagnostic command or a pending human review cannot grant that authority.

Even a fully graded heterogeneous program need not accelerate every operation: a matmul unit does
not automatically execute norms or elementwise work. State which regions ran on which lane, which
were declined, and what numerical and RTL tier actually measured them. For the older experimental
Gemmini commands and their limitations, use the [reproduction guide](reproducing_whole_model_on_rtl.md);
do not use its historical spike result as an RTL result.
