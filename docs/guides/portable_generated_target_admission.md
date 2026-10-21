---
title: Portable generated-target instruction admission
kind: guide
status: current
owner: merlin-experiments
last_verified: 2026-10-20
related: [shared_pass_phase2, ../design/independent_instruction_policy]
code_refs:
  - src/merlin/targetgen/contract/elf_admission.py
  - src/merlin/targetgen/rtl/source_selection.py
  - src/merlin/targetgen/rtl/unit_correspondence.py
  - src/merlin/targetgen/rtl/source_relocations.py
  - src/merlin/targetgen/rocc/operand_selection_recipe.py
  - src/merlin/compile/route_before_build.py
  - src/merlin/transforms/dense_ir.py
  - src/merlin/targetgen/model_coverage.py
  - merlin/tests/gemmini/packing_execution_inputs.py
  - merlin/tests/gemmini/test_public_packing_execution.py
  - merlin/tests/gemmini/test_public_destination_execution.py
  - merlin/tests/gemmini/parameter_chain_execution_inputs.py
  - merlin/tests/gemmini/parameter_chain_reference.py
  - merlin/tests/gemmini/test_public_parameter_chain_execution.py
  - merlin/tests/ir/destination_forwarding_connected.mlir
  - packages/merlin-experiments/src/merlin_experiments/phase0/rtl_intake.py
  - packages/merlin-experiments/src/merlin_experiments/phase0/accessor_intake.py
  - packages/merlin-experiments/src/merlin_experiments/phase1/feedback/private_instruction_declaration.py
  - packages/merlin-experiments/src/merlin_experiments/phase1/feedback/private_instruction_coordinator.py
  - packages/merlin-experiments/src/merlin_experiments/phase1/feedback/instruction_admission_preparation.py
  - packages/merlin-experiments/src/merlin_experiments/phase1/feedback/instruction_admission_facts.py
---

# Portable generated-target instruction admission

A second server must issue its own live instruction-admission service from the
reviewed source inputs. Copying a successful report, approval or package does not
reconstruct `LinkedElfAdmissionService` authority. This guide describes source and
operator-input transport; it grants no compiler, runtime or timing qualification.
All commands below run in the receiving server's managed workspace with its own
resource admission, fresh output directories and temporary directory.

## Select public source before provisioning inputs

The original V26 Gemmini experiment has these published provenance pins. They
identify the original submission, rather than a Merlin release containing the
current source-optimization APIs below. Use the public qualification inputs in
the next section with an explicitly selected, reviewed Merlin library revision;
record that revision independently in every comparison.

| Repository | Commit |
| --- | --- |
| `https://github.com/ucb-bar/merlin` | `f8a43c657f11ee417f4c96d8920fa2f7e66f3f5b` |
| `https://github.com/ucb-bar/gemmini-mlir` archive | `8b8fd3629f4e8e1c3349176241620621048b96c7` |

The archive's `compiler/` contains 13 Python files and its manifest. Its original
16-file submission additionally contained two author reports and had tree digest
`29213759089b71de26f8a5ab26ca6ef1b8071dc0bb690041eb7de5d0efb53f24`.
Bind the actual received compiler directory independently. Do not attribute the
original tree's evidence to the 14-file projection by substituting its digest.
`python "$ARCHIVE/reproduce/verify_archive.py"` checks published archive bytes,
not native execution or certification. V26 does not declare `target_ir_consumer`
or `emit_target_bundle`; its analysis-bundle command does not promise to consume
an independently transformed target IR.

## Public qualification inputs

The table above records the original Phase 1 provenance. Receiving-server
integration is available at Merlin
`588d80a5192035c84ed9d2942ac196c5d9101c5d` and gemmini-mlir
`1616526ca76f94724485714f49302696e36104f4`. Selecting these sources does not
rewrite or recertify the original compiler. The target repository's
[`docs/v26/portable-runtime-and-compiler.md`](https://github.com/ucb-bar/gemmini-mlir/blob/1616526ca76f94724485714f49302696e36104f4/docs/v26/portable-runtime-and-compiler.md)
describes the exact original 16-file submission, matching public hardware and
software inputs, and externally provisioned runtime/tool requirements.

With `OOT_REPO` set to that managed target checkout, verify the received public
bytes before selecting capture or execution inputs:

```sh
cd "$OOT_REPO/examples/phase2/public_pass_inputs_v26"
python3 -B verify.py
cd "$OOT_REPO/docs/v26/inputs"
sha256sum -c SHA256SUMS
```

The fixtures provide a small signed integer contraction, a connected integer
GQA/KV program and a synthetic 3 MiB parameter chain, with public inputs, weights
and independent complete-output references. Integrity checks issue no capture,
execution admission or runtime qualification. Start with the small contraction
and freshly capture or bind the original trusted materialization through the
normal APIs. Preserve every declared output, input and carried-state check.

For producer-into-packing qualification, first check actual source applicability.
A connected program alone need not contain an eligible producer/permutation
chain; zero rewrites are an unchanged control. Use a separate, meaningful public
source case when needed, with its own independent reference and fresh capture.
Do not inject a redundant producer into an existing fixture to manufacture a
benefit. Compare baseline and candidate copies from one unchanged compiler with
identical inputs and numerical policy, and reopen the final linked ELF admission,
actual accelerator execution and transformed logical layout/consumer ABI.

V26 has no declared target-IR consumer. ABI-preserving source packing is a
separate route from device instruction rewrites; diagnostic target IR cannot
establish realization. Functional Spike checks and requested byte counts supply
no latency improvement, and this process leaves the incomplete original resource
certificate and formal Phase 2 admission open.

### Reproduce the complete public parameter-chain baseline

`test_public_parameter_chain_execution.py` compiles the unchanged public 3 MiB
parameter chain through the ordinary compiler path. It consumes a genuine sealed
capture through its original verifier and source reader, writes a new development
capsule, and uses the ordinary runtime-bundle helper for the pointer ABI. It does
not issue another capture, replay numerical capture, or copy its runtime content
store. The receiving machine must already have a valid original capture and its
complete required runtime inputs; a copied successful JSON report is insufficient.

Create a separate operator JSON with exactly these fields. File pins have the
same closed `path,sha256` form as the packing selection below:

| Field | Required input |
| --- | --- |
| `schema` | `merlin.test.public_parameter_chain_baseline.v1` |
| `execution` | Pin of the ordinary execution-selection JSON described below; its `trace_contract` may be null for this quiet test |
| `attestation` | Pin of the original sealed capture attestation, verified again at consumption |
| `source_member` | Basename of its original snapshotted Python loader |
| `public_case` | Seven file pins, keyed by the exact public fixture filenames listed below |
| `worker` | Explicit ABI-worker selection described below |
| `obligations_profile` | Pin of the original profile retaining all four required tiers, L0 through L3 |

The `public_case` files are `case.json`, `capsule.pytorch.py`,
`capsule.interface.mlir`, `capsule.weights.safetensors`,
`capsule.weights.safetensors.manifest.json`, `frontend-trace.json` and
`independent-reference.npz`. They must share the selected public case directory.
The `worker` has exactly `python`, `binary`, `config`, `torch_version` and
`torch_init`: the first is the selected virtual-environment interpreter path;
the others pin its resolved executable, `pyvenv.cfg`, Torch `version.py` and
Torch `__init__.py`. These selected bytes and the observed worker versions are
checked; they do not establish a hermetic runtime or a new numerical replay.
The existing bundle helper retains request/provenance, rather than a complete
worker invocation record and raw console streams.

Use a fresh pytest process with the selected compiler worker and toolchain. The
worker must support `respect_captured_quantization_scope`; preserve that feature
instead of removing it to accommodate an incompatible driver:

```sh
MERLIN_TEST_GEMMINI_PARAMETER_CHAIN_SELECTION="$PARAMETER_CHAIN_SELECTION_JSON" \
python -B -m pytest -q \
  merlin/tests/gemmini/test_public_parameter_chain_execution.py::test_original_public_parameter_chain_baseline \
  -p no:cacheprovider -o tmp_path_retention_policy=all \
  --basetemp "$TMPDIR/public-parameter-chain" \
  --junitxml "$TMPDIR/public-parameter-chain.xml"
```

Absent selection skips without qualification; malformed or changed selected
inputs fail. The test creates a fresh live linked-ELF admission service and
preserves the original 16-file compiler. It checks all 2,048 output values
against both the public reference and a separate ordered integer calculation,
and checks post-call bytes of all inputs and weights. The final `state` output
belongs to this single three-stage forward call; this test does not establish
recurrent state across invocations.

`baseline.json` records actual source, tools, compiler products, complete
readback and admission. This is a functional baseline, with ordinary pass
defaults, rather than an optimization comparison. Quiet execution supplies no
dynamic instruction counts. Timing, compiler peak memory, physical transfer
traffic and resource admission require separate evidence. All four formal tier
obligations remain pending, and end-of-call byte equality does not prove the
absence of intermediate writes.

### Explicit typed-source placement

The ordinary `plan_before_build` caller may select
`source_kind="typed_static_integer_v1"` for a public typed source without capture
provenance. This reads exact signed i8 × i8 → i32 rank-two generic contractions
with an explicit zero initializer: `linalg.fill` over `tensor.empty`, or a static
`tensor.splat` of an earlier local, exactly typed integer-zero constant. This
establishes the zero value; physical ownership and overwrite permission require
separate checks. It checks the full reduction body,
static dimensions and independent shape observations before asking the existing
router for placement. It preserves source bytes and operation order, assigns no
captured region IDs, and leaves other host computation outside that contraction
permission. Unknown reduction bodies, effects, iterator metadata and incompatible
geometry refuse.

The default remains `captured_provenance_v1`: losing a capture tag still refuses.
Typed intake grants neither capture nor numerical admission, device support or
an executable route. Select the hardware/software contract, original compiler
and live linked-ELF admission separately, then use the returned routing in the
ordinary build. Complete group admission remains a separate interface.

### Select source choices before preparation

An original host caller can fix choices before the prepared source exists:

```python
from merlin.llvmlower.source_optimization_policy import SourceOptimizationPolicy

policy = SourceOptimizationPolicy(
    producer_packing=True,
    padding_gather=False,
    maximum_added_operations=4096,
    permitted_passes=("automatic-host-permutations",),
    original_maximum_added_operations=4096,
)
result = compile_saved_model(**original_build_arguments, source_optimizations=policy)
```

The ordinary build snapshots this original object and its implementation before
preparation callbacks. After routing and projection, it binds the actual prepared
source once and passes an ordinary exact-SHA request to lowering. Its version-2
build receipt joins the original policy, derived request, consumed report,
compilation recipe and linked executable. Changed choices, permissions, ceilings,
source bytes or artifacts refuse. Noncanonical newline bytes currently refuse
rather than silently changing the source identity.

The construction ceiling applies separately to each pass, not to total storage
or scalar recomputation. Unsupported source patterns retain the existing route;
a selected pass is not proof of a rewrite. Existing exact-SHA selections and the
default unselected build remain available. This API grants no authoring access,
numerical permission, executable target capability or formal Phase 2 admission.
Those require their original independently selected contracts and owners.

### Reproduce connected packing with the public compiler

The opt-in regression
`merlin/tests/gemmini/test_public_packing_execution.py` uses the fixed public
`producer_packing_connected.mlir` source and its independent ordered integer
reference. It builds baseline and candidate copies of the same original
16-file compiler through `plan_before_build` and `compile_saved_model` with
full Spike readback. Before either arm executes, the original caller fixes a
bounded policy selecting `automatic-host-permutations`. The ordinary build binds
it once to its actual prepared lowering input and derives the existing exact-SHA
request. The candidate needs no source hash from the baseline. Every arm still
checks identical prepared input bytes. The ambient pass switch is disabled in
both arms, so it cannot account for a candidate rewrite. This exercises the
original-caller policy API, not a
formal Phase 2 authoring grant. The test reruns
`C.prepare(D.read(...))`; selected JSON pins cannot restore a service or issue
numerical permission.

Prepare the reviewed declaration and current quiet/trace runtime contracts
described in this guide. The trace contract must preserve every quiet-contract
member except the Spike `argv` prefix `[-l, --log-commits]` and its explicitly
selected console bound, which must be at most 32 MiB. Both contracts select the
same runtime, hardware, operand observer, tools and numerical policy. Quiet
execution supplies numeric readback; trace replay supplies retired instructions
from the same actual ELF. An absent selection or a null trace contract skips
the experiment before target builds and establishes no target qualification.

Construct the closed selection using original caller-selected, absolute,
canonical paths. File paths and their ancestors must not be symlinks. The
following uppercase variables denote those explicit receiving-host selections,
not paths supplied by a compiler candidate:

```python
import json
from pathlib import Path
from merlin.common.digest import sha256_file
from merlin.compile.model_execution_inputs import strict_tree_sha256

def file_pin(path):
    return {"path": str(path), "sha256": sha256_file(Path(path))}

def tree_pin(path):
    return {"path": str(path), **strict_tree_sha256(Path(path))}

selection = {
    "schema": "merlin.test.gemmini_packing_pair.v1",
    "instruction_declaration": file_pin(REVIEWED_DECLARATION),
    "quiet_contract": file_pin(QUIET_CONTRACT),
    "trace_contract": file_pin(TRACE_CONTRACT),
    "facts": file_pin(ORIGINAL_PUBLIC_FACTS),
    "board_catalog": file_pin(BOARD_CATALOG),
    "board": SELECTED_BOARD,
    "dts": file_pin(DTS),
    "host_package": tree_pin(ORIGINAL_HOST_PACKAGE),
    "public_compiler": tree_pin(ORIGINAL_PUBLIC_COMPILER),
}
Path(SELECTION_JSON).write_text(json.dumps(selection, indent=2) + "\n")
```

These are the complete accepted keys. Each file pin has exactly `path,sha256`;
each tree pin has exactly `path,sha256,n_files`. The compiler tree must be the
original 16-file public submission with digest `29213759089b71de26f8a5ab26ca6ef1b8071dc0bb690041eb7de5d0efb53f24`,
not the 14-file archive projection. Choose the ordinary host package, board
catalog and DTS matching the selected runtime. The JSON contains no source,
reference, command, callback, selector or approval override.

Run in a fresh pytest process with the installed Merlin/experiment dependencies
and the reviewed LLVM and RISC-V tools available. Retain the temporary outputs:

```sh
MERLIN_TEST_GEMMINI_PACKING_SELECTION="$SELECTION_JSON" \
python -B -m pytest -q \
  merlin/tests/gemmini/test_public_packing_execution.py::test_original_public_compiler_executes_connected_packing_pair \
  -p no:cacheprovider -o tmp_path_retention_policy=all \
  --basetemp "$TMPDIR/public-packing-pair" \
  --junitxml "$TMPDIR/public-packing-pair.xml"
```

The test creates a fresh live admission service and fresh compiler copies. Its
trusted source/ABI validator binds complete public synthetic runtime inputs;
it creates no capture receipt. Each arm performs three fresh ordinary builds,
feeding the previous actual target result into the next call's state input.
Every element of all three outputs is checked against the independent reference.
This is explicit external state carry, not a persistent in-process model session.
Both arms also request complete post-call bytes for the fixed public program's
readonly inputs and weights. The original driver reads the actual argument
addresses after `forward` returns, before any runtime state commit. Separate
named byte frames preserve the forward-result roster. Every selected byte must
match its original input, and the returned observation roster and budget must
match the original caller's selection.

`pair.json` retains output arrays, source/reference/helper identities, actual
invocation receipts, consumed host LLVM, packing reports, linked ELF identities
and native instruction-accessor observations. Every retired custom instruction
word and PC must match the executable bytes of that exact ELF. After the last
original callback, the test reopens all earlier artifacts and receipts before
publishing the pair. Unknown or refused admission does not fall back to a saved
approval or a handwritten provider.

Malloc call sites and executable-section bytes are static observations. The
combined wall time includes each build and execution. A separate compilation
wall time covers the ordinary model build and its selected compiler checks,
excluding capture and target execution. Compilation peak memory remains unknown.
Neither wall time measures accelerator latency. This bounded synthetic comparison
does not establish physical bus traffic, absence of intermediate input writes,
general-model numerical qualification,
the original resource certificate or formal Phase 2 admission. It changes no
frozen authoring grants or versioned pass inventory.

### Exercise destination forwarding between contractions

The separate public `destination_forwarding_connected.mlir` program computes
two integer contractions around a clamp, conversion and padded insertion. It
returns five complete results, including the original zero padding as an
independent observer. Its state input includes signed integer boundary cases;
the first executed call must actually overflow before wrapping and clamping.

This case additionally requires the original live borrowed-buffer effects owner
described in [checked device result identity](shared_source_lowering.md#expose-checked-device-result-identity).
The public V26 inputs do not supply that qualification. Without it, the test
explicitly skips before fresh capture/admission; this is an unresolved integration
prerequisite, not a passing destination optimization or a hardware limitation.

An original trusted pytest plugin may set
`config.merlin_public_destination_descriptor_effects` to its source-bound factory
during `pytest_configure`. That factory must return the live, independently
qualified `BorrowedBufferCalls` owner for the actual prepared source, kernel
objects, checked shim and selected runtime. No factory or authority is loaded
from the selection JSON. Once that integration exists, use the same original
selection and fresh runtime/admission provisioning:

```sh
MERLIN_TEST_GEMMINI_PACKING_SELECTION="$SELECTION_JSON" \
python -B -m pytest -q \
  -p "$ORIGINAL_EFFECTS_PLUGIN" \
  merlin/tests/gemmini/test_public_destination_execution.py::test_original_public_compiler_executes_destination_forwarding_pair \
  -p no:cacheprovider -o tmp_path_retention_policy=all \
  --basetemp "$TMPDIR/public-destination-pair" \
  --junitxml "$TMPDIR/public-destination-pair.xml"
```

This comparison uses the existing `automatic-host-destinations` optional-pass
switch, fixed independently for each arm before preparation. Packing and gather
are disabled in both arms. The two-choice source policy above does not select
destination forwarding. The test pins its selected algorithm, checker and parser
owners before original callbacks, checks the consumed report against those
owners, and reopens them after all six builds and executions. Both arms use the
same unchanged compiler and preserve complete results, original inputs, returned
state and final linked-ELF admission.

Both arms select the same checked descriptor bridge and borrowed-effects owner.
The test reopens the actually compiled shim C/object and invocation, consumed
lowering report and LLVM, original effects record and final completed compilation
recipe. The effects record preserves the original owner's premises and pins;
it cannot replace the live owner. A reduction in selected source operations alone
does not satisfy the test's strict final allocation-saving assertion.

The intended saving is host initialization, allocation and copying around real
device calls. This experiment does not establish a device-local destination
handoff, cross-dispatch residency, asynchronous lifetime extension or latency
improvement. A successful native-only test establishes only its host scope;
the opt-in target execution and independent artifact audit remain necessary.

### Observe readonly arguments in an ordinary build

An original caller can opt into the same observation independently of this test:

```python
result = compile_saved_model(
    **original_build_arguments,
    readback="full",
    readonly_input_indices=(0, 1, 2),
    max_readonly_input_bytes=1 << 20,
)
```

Indices refer to original capture arguments, including weights. The caller must
establish that those arguments are readonly during the call. A declared mutable
argument, known mutable weight alias, projected-away argument, invalid ordinal
or exceeded byte budget refuses. The runtime captures selected bytes before
subsequent state commits; a state input may therefore be observed when the call
itself preserves it. The observation does not declare arbitrary recurrent state
readonly.

The versioned `merlin.readonly_input_observation.v1` manifest pins argument
mapping, expected bytes, generated observer and original source inputs. The
build records the observer and generated headers alongside actual compilation
inputs. Full readback accepts the additional frames only with this exact
selection; missing, changed or unexpected frames fail. The result records the
observation separately from model outputs. Omitting the option retains ordinary
uninstrumented behavior.

This checks end-of-call byte equality. It proves neither the absence of
temporary writes nor a device's completion semantics. Completion must already
be established by the selected call interface. Snapshot storage and volatile
byte reads add runtime work, including work inside the existing total execution
counter window; compare equally instrumented arms and do not treat the counters
as accelerator latency.

## Obtain the complete operator input set

The archive's `reproduce/input-manifest.json` identifies separately owned inputs.
The following public RTL/source and reviewed operator data are needed for the
linked instruction service; private model checkpoints, golden arrays and hidden
capsules are not needed and must not be exported for this purpose.

| Input | Required content and origin |
| --- | --- |
| Hardware descriptor | Minimal reviewed target descriptor; V26 original SHA `9d331b293d3e6ebf0b8ae230607b3e6c57f8271c8acab1373f9ebcaac644877f`. |
| RTL source bundle | `source-selection.json`, selected FIRRTL, hierarchy, SoC HW and exact core HW closure; also selected FIR-to-HW tool and any explicitly selected elaboration receipt. Public elaborated source artifacts are operator-provided, not embedded in V26. |
| Command/predicate source | Public Gemmini checkout at `63f0b68a68f185a016a2993f2fb32e1661ea9e6c`: `GemminiISA.scala`, `LoopMatmul.scala`, `LoopConv.scala`. Keep tracked provenance, not extracted command tables. |
| Native accessor source | Public riscv-isa-sim checkout at `9c190a07c6838f6392bafa4ad83acea462c7f759`. The corresponding selected include tree must contain byte-identical tracked public headers. |
| Accessor specification | Reviewed type/member names only. Original SHA `09d2323b4c213f9f33e4f602df85e984efac7557edeec279ef2abd89dadc4e47`. |
| Software policy | Original SHA `3da7fd4266b1f48297b4cc4fbf211792137975b208bc6408f1fccc1b2786880b`: 15 prohibited `LOOP_WS*`/`LOOP_CONV_WS*` source symbols. It supplies no numeric command values. |
| Tools | Selected `firtool`, `circt-opt`, C++20 host compiler and their usable host dependencies. Original executable hashes are listed below; matching version strings are insufficient. |
| Reviewed declaration | `merlin.private_instruction_selection.v1`, current absolute canonical paths, exact file hashes, source commits, source span/predicates, decoder selection and explicit forbidden roots. |

The original accessor specification selects header `riscv/rocc.h`, union
`rocc_insn_union_t`, word member `i`, fields member `r`, conversion `insn_t`,
length member `length`, fields `opcode,rd,xs2,xs1,xd,rs1,rs2,funct`, and constants
`ROCC_OPCODE3,ROCC_OPCODE_MASK`. The decoder selects those two constants, member
`funct` and the explicitly reviewed RISC-V ELF machine value. The original command
span is `// funct values` through `CONFIG_EX`. The four source predicates are
`is_loop_run_cmd` and `is_loop_config_cmd` in each loop source, on
`cmd.bits.cmd.inst.funct`. Obtain the original files and review these selections;
this description is not a replacement approval.

The accessor issuer compiles its own fixed observer with `-std=c++20 -O2 -MMD`.
It checks **every discovered non-system header dependency** against the selected
tracked checkout and include tree. Provision the full matching include closure;
copying only `rocc.h` is insufficient. Host compiler/system-library dependency
closure and physical RTL equivalence remain explicitly unqualified by that test.

For re-elaboration rather than transfer of exact public artifacts, the original
Chipyard selection was `3e9bfb4bf58a7dfe6e1082496a9578a5dcd83831`, with Gemmini
as above, Saturn `e16a301d261fa4b5a374eb7069b78107a8c9aefb`, and FireSim
`b084672c2f8cf32e55d78f73a001074c23f8a2b8`. The selected
`GemminiGsimFireSimAlignedConfig.scala` content hash was
`5392120d1ed547b3a70c3786dd5a3d3d697f7817f9ce55e1ffc25b6c2ff6b2ec`.
A Git revision alone does not provision that configuration file or prove the old
elaboration; obtain its exact public source or select and review a new one.

## Relocate prospectively, then review the new declaration

Keep received original manifests/receipts byte-exact as historical provenance.
Place approved source inputs under a receiving-server input root. Do not symlink,
rewrite an old receipt's paths in place, or recreate the first server's directory
layout to masquerade as its owner. Record SHA-256 for each transferred file.

A portable route is to regenerate a **new** source selection with the existing
producer on the received public FIRRTL and hierarchy:

```python
from pathlib import Path
from merlin.targetgen.rtl.source_selection import produce_selection

source_selection = produce_selection(
    target="gemmini", firrtl=Path(FIR), hierarchy=Path(HIERARCHY),
    generator="gemmini", config="GemminiGsimFireSimAlignedConfig",
    core_root="Gemmini", firtool=Path(FIRTOOL), output=Path(FRESH_SOURCE_OUTPUT),
)
```

This runs the selected source tool, not a target simulation. It derives the
complete module closure and current producer attribution. Preserve any differing
output as a new selection; it does not inherit the old artifact's identity.
Select an elaboration receipt only if its original verifier accepts the actual
source/configuration on this host. Otherwise historical source-to-FIRRTL origin
stays UNKNOWN. The old `production.producer_source` is historical metadata:
current admission does not reopen that old path/hash. It pins current reader
modules, reruns the selected tool, and requires reproduced SoC/core HW bytes to
match the newly selected source bundle. Actual source/tool/output drift refuses.

Create a new closed declaration following
`private_instruction_declaration.read`'s schema. Rebind hardware descriptor and
source bundle, command checkout/source/tool, four predicate files, accessor
checkout/include/compiler/spec, policy and forbidden roots. Hash the complete new
declaration **after** path selection. A legitimate operator review must approve
those current choices and exclusions; retain the new approval separately. The
old selection hash `93c2102f37914cffb7e6a2c884b7d62d8a3149f0cc4ef72703a3ea026d265ec6`
and approval `20524cfe3f4447bea7d83a0668f626a3b093b84972d6d2e82d1fd1496fec7340`
identify first-server history and cannot approve a changed declaration.

### Relocate sources referenced inside unchanged facts

A facts document may retain source paths from its original server. Keep its
bytes unchanged. The source readers accept an explicit `source_relocations`
roster for recorded ISA and FIRRTL inputs:

```python
relocations = [{
    "original": {"path": ORIGINAL_SOURCE_PATH, "sha256": SOURCE_SHA256},
    "replacement": {"path": CURRENT_SOURCE_PATH, "sha256": SOURCE_SHA256},
}]
```

Each original pin must occur in the selected facts. Each replacement must be a
canonical ordinary file with exactly the recorded bytes. Unknown pins, changed
hashes, duplicate ownership, indirect paths and more than 16 moves refuse.
This changes only a detached reader view; it does not rewrite the facts or their
identity, alter any numerical or hardware value, or relocate admission receipts.

Pass this roster to `model_execution_inputs.selected_firrtl` when selecting the
facts' own configuration. To rebuild an operand observer, pass the same roster
to `operand_selection_recipe.derive` with the unchanged original facts bytes,
original `BoundSemantics` and explicitly selected recipe. The recipe rederives
the observations from the current sources. The resulting sidecar records the
roster and the **original facts hash**; its own hash is new. Select that new
sidecar prospectively in a new runtime contract, rather than assigning it an old
approval. Original observer binding rechecks the source files before returning.
The ordinary bound-runner facts reader obtains relocations only from its checked,
explicitly selected observer.

This roster does not relocate a different configuration's simulation
correspondence. That join retains its separate verification requirements. It
also supplies no source-to-binary, numerical, completion, execution or timing
qualification. Container-only `relocate_operand_protocol` still requires
byte-identical sidecar files; rebuilding an observer is a different selection.

## Check and prepare through the operator command

The existing experiment CLI exposes this fixed original path. Check mode is the
default and launches no issuer or tool:

```bash
merlin experiment instruction-admission \
  --target "$TARGET" --selection "$REVIEWED_DECLARATION" \
  --selection-sha256 "$REVIEWED_DECLARATION_SHA256" \
  --task-root "$TASK_DIR" --output "$FRESH_ADMISSION_OUTPUT"
```

It reopens the original closed declaration and its source pins, checks the
operator-selected hash, exclusions, and fresh disjoint output. Missing inputs or
stale selections report `REFUSED`; a check pass reports `CHECKED_NOT_PREPARED`,
not a service or qualification. It does not create the output directory.

Add `--prepare` to run only the original coordinator's host-side issuers. An
optional `--elf "$EXISTING_PUBLIC_ELF"` statically evaluates and revalidates that
existing artifact within the same actual live-owner lifetime; it never builds or
executes guest code. `operator_preparation.json` is a custody result, not a service
loader. The original coordinator's own records remain unchanged. The shared CPU
lease controls host-tool admission; select an external aggregate CPU/wall/RSS/disk
guard for the process tree. `--lease-timeout-s` bounds waiting for a lease, not the
whole preparation. No report grants numerical, hardware, model or timing credit.

An embedding operator uses the exact typed `InstructionAdmissionSelection` and
`prepare_live` API from `instruction_admission_preparation`, retaining the
returned original `PreparedInstructionSelection` object for ordinary linked
builds. No callback or serialized-success constructor is accepted. Selecting a
declaration hash expresses the operator's selection, not a new approval; perform
the separate source/tool/policy review described above first.

For an embedding build, bind its actual public and input facts in the same live
preparation lifetime:

```python
from merlin_experiments.phase1.feedback.instruction_admission_facts import (
    InstructionFactsSelection, prepare_for_facts,
)

facts = InstructionFactsSelection(
    Path(PUBLIC_FACTS), PUBLIC_FACTS_SHA256,
    Path(INPUT_FACTS), INPUT_FACTS_SHA256, EXECUTION_CONFIGURATION,
)
with prepare_for_facts(admission_selection, facts) as prepared:
    # Use this exact prepared.service in the original linked build API.
    # All numerical, source, runtime and execution gates remain required.
    ...
```

Both facts selections go through the original `selected_firrtl` reader, including
its explicit unit-correspondence path where selected. The context calls original
`prepared.require_facts` before and after use and reopens facts/FIRRTL bytes,
including on exceptional exit. It yields the original prepared object, not a new
authority wrapper. `selected_facts(prepared, facts)` provides the same checked
scope for an already-live original preparation. The coordinator's existing
`private_input` role name also accepts explicitly selected public input facts;
this API does not read private model metadata or replace the private gate's own
`selected_input_facts` context. No environment or configuration is inferred.

## Issue the original live service

With the receiving environment selected, invoke the fixed original coordinator:

```python
from pathlib import Path
from merlin_experiments.phase1.feedback import private_instruction_declaration as D
from merlin_experiments.phase1.feedback import private_instruction_coordinator as C

prepared = C.prepare(
    D.read(Path(REVIEWED_DECLARATION), target="gemmini"),
    target="gemmini", output=Path(FRESH_ADMISSION_OUTPUT),
)
prepared.verify()
service = prepared.service
service.verify("gemmini")
```

`FRESH_ADMISSION_OUTPUT` must be canonical, absent, disjoint from selected inputs
and exclusions, and owned by the receiving task. The constructor reruns hardware,
command, predicate, native accessor and policy issuers. Keep this live object;
neither the report nor a saved `status: accepted` can restore it.

Pass `service` as the original build API's `linked_elf_admission` selection. It
checks the actual final linked ELF before dispatch; refusal means no execution.
The build reopens the unchanged admission report and ELF before and after native
execution. A direct diagnostic can call `service.evaluate(elf=..., target=...,
evidence_root=...)` and `service.revalidate(...)`, but must retain all original
numerical, source/effect, runtime and timing gates. The structural
`model_check.admission.service` only records digest/declared ISA and is not a
replacement for this independent instruction-policy service.

## Preserve the two hardware identities and the timing boundary

The V26 facts hash is
`b8ff28edc096392cec356f9fececdf17ba70cd3e4b02e6c5931aedc43bb117af`.
Its embedded `facts.simulation_correspondence` records **110 Gemmini unit modules**:
110 exact FIRRTL bodies, 109 exact HW modules and 110 structurally identical HW
modules; the remaining text difference is an `sv.namehint`. Its target FIRRTL is
`b7d9826b1c9f16fd00bc757b4334a7f330a842c6a494e0c28608dc351b305bf0`;
the gSIM FIRRTL is
`6748b9756799c086a6fb06e6d80b4b052f2771f9eb1858c172e14dcf441ba149`.
They are **different artifacts**. `unit_correspondence.execution_binding` checks
the original facts' target-side identities and returns the selected simulation
side; it does not recompute the equivalence record. Reproduce or independently
admit this join on the receiving host. Do not fabricate identical FIR hashes to
satisfy `prepared.require_facts(actual_public_private_binding)`.

This unit comparison excludes SoC memory/cache, clocks and bridges. It is not
simulator fidelity or timing qualification. Original tool/runtime reference hashes:

| Artifact | SHA-256 |
| --- | --- |
| firtool | `cd4a5be2461207f2d1261f8cfba7ddb8c74648ddc3f2bf64a4a1a23abe11726a` |
| circt-opt | `41512929e7c37de1c25190dbc0e3888b2f230cf6a9ee8eeaa6ca68c5258883b3` |
| Host C++ compiler | `2360901d864cf10bfd6296e261cb2c14053552a80377761ab07146ec9ec9a2c0` |
| Functional Spike | `5df9c82ca3e921823da8606284a258b491541defbafee4586fb938639e03c6b9` |
| Functional extension | `2dbd8445376cdda0df63abcf217c7a19f974d34a16ada21b549ed82fffd718eb` |
| Original gSIM label producer | `76ddbfdc5b3beadef6b6b37768a08522122327057b92d71d3539a9529e4ea651` |

Different native tool bytes require explicit current selection/review and fresh
original issuer evidence. A compatible instruction-admission service proves only
the selected static prohibition over complete executable sections. It does not
complete V26's resource/capsule certificate, qualify shared transformations or
full models, or transfer the frozen Spike model's calibrated domain. The published
capsule-certified Phase 2 route still requires its original certificate and
publication; passing capsule counts cannot replace those inputs.
