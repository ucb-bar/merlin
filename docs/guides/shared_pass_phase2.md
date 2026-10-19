---
title: Use shared passes in compiler generation
kind: guide
status: current
owner: targetgen
last_verified: 2026-10-19
related: [shared_source_lowering, shared_offload_selection, ../reference/shared_compiler_passes, phase2_corpus_search]
code_refs:
  - src/merlin/targetgen/tool_profile.py
  - src/merlin/targetgen/tool_profile_shared.py
  - src/merlin/targetgen/common_transforms.py
  - src/merlin/compile/baremetal_model.py
  - src/merlin/compile/scheduling/compact_dense.py
  - src/merlin/compile/scheduling/configuration_ir.py
  - src/merlin/compile/scheduling/pointwise_source.py
  - src/merlin/compile/scheduling/row_softmax.py
  - src/merlin/llvmlower/source_dense.py
  - src/merlin/llvmlower/source_choices.py
  - src/merlin/targetgen/capsule_common.py
  - packages/merlin-experiments/src/merlin_experiments/phase2/source_choices_workflow.py
  - packages/merlin-experiments/src/merlin_experiments/phase2/ordinary_search_stage.py
---

# Use shared passes in compiler generation

Merlin's shared compiler library provides checked scheduling, storage and
composition algorithms for generated compilers. Select a library release before
Phase 1; Phase 2 inherits that same release and can tune the compiler's decisions
without changing the checks used to admit its output.

This guide covers the `merlin.common_compiler_substrate.v6` release profile.
Versions 1–5 retain their existing APIs and verification policies. An already
certified compiler keeps its frozen profile; changing Phase 2 options cannot
add library access to it.

## Inputs and backend requirements

The compiler needs a selected hardware configuration, a numerical policy, and
resource and code-growth budgets. Optional controls can restrict the supported
optimization choices.

| Input | What it establishes |
| --- | --- |
| Versioned RTL facts | Extracted dimensions, element widths, capacities, banks and address limits |
| Software and ISA specification | Arithmetic, layouts, conversions, instruction effects, ownership and completion rules |
| Numerical policy | Permitted arithmetic and the original admission thresholds |
| Budgets and controls | Bounds on compilation work, code growth and legal optimization choices |

A backend must also supply the applicable instruction implementations, physical
mapping, encoding and decoding, and independent execution qualification.
Structural RTL facts alone cannot establish arithmetic equivalence, completion
semantics or performance costs. Missing facts remain unknown.

Shared entries accept typed source or schedule IR. A compiler must call the
appropriate entry with its selected contracts and qualified backend bindings.
Ordinary model builds can select supported dense, pointwise and row families through
`shared_source_lowerings=(...)`; Merlin then discovers, schedules and rewrites
eligible regions. See [enable shared passes in model builds](shared_source_lowering.md)
for the integration interface, the `compile_saved_model` entry and fallback behavior. Backend bindings supply
implementations and ABI support. Importing the library alone does not change
routing or interpret arbitrary backend output. See the
[pass reference](../reference/shared_compiler_passes.md) for each entry's accepted
IR, required capabilities and fallback behavior.

## Select the right compiler boundary

Apply each pass while its required information is still available. Several
choices must precede target lowering even when their purpose is to improve the
target program.

| Choice | Required representation | Backend information required |
| --- | --- | --- |
| Layout and packing continuity | Connected typed operations and their layouts | Logical/physical layout mapping, mutability and ownership |
| Operand residency and prefetch | Loops or schedules with explicit transfers | Assigned banks/windows, aliases, launch/wait effects and qualified overlap |
| Exact output fusion | Complete reduction and original conversion graph | One supported physical readout with matching arithmetic, rounding and saturation |
| Convolution packing route | Original convolution, padding and full reduction domain | Qualified dense primitive, logical buffer ABI and actual allocation effects |
| CPU/offload selection | Complete source with both legal alternatives available | Actual complete-call costs, state/cache policy and current emitted artifacts |
| Redundant configuration removal | Typed target instructions and state effects | Persistent field semantics, resets, actual operand decoding and a consuming emitter |

A serialized command payload is an opaque barrier unless an independently
qualified public interface supplies those semantics. Names, planned commands
and an edited diagnostic target file cannot establish what the executable does.

For target-IR configuration cleanup, the package must explicitly declare:

```yaml
target_ir_consumer:
  schema: merlin.target_ir_consumer.v1
  command: emit_target_bundle
  input: selected_target_mlir
  output: command_buffer_and_target_artifact
```

The command uses `{input_mlir}` and `{output_json}` to consume the selected
target IR and return the actual command buffer and target artifact. The ordinary
interface-lowering path checks this handoff and retains both original and
selected identities. The declaration grants no instruction semantics: original
typed ISA/effects, execution qualification and final decoding remain necessary.
Without the contract, keep ordinary lowering; an explicitly requested target
rewrite reports the missing integration instead of emitting an unused rewrite.

## Prepare a library release

Choose the Merlin source tree, its license directory, and a fresh destination
for the release. `MERLIN_SOURCE_ROOT` must be the absolute canonical Python
source directory (`<checkout>/src`) used by the running Merlin installation;
`MERLIN_LICENSE_ROOT` is the directory containing `LICENSE` and `NOTICE`.

```sh
python -m merlin.targetgen.tool_profile_shared \
  --python-source-root "$MERLIN_SOURCE_ROOT" \
  --license-directory "$MERLIN_LICENSE_ROOT" \
  --output-dir "$TMPDIR/common-v6"
```

The command stages the shared library, verification support, licenses and a
manifest of source identities. Retain the release while a compiler uses it.
Staging files does not itself grant a candidate access to them; bundle creation
selects the permitted APIs and resources.

For experiments that expose additional authoring tools, repeat `--tool NAME`
for the selected registered tools. Each treatment's bundle still determines
which tools and resources that treatment may access.

Pass the prepared release to the normal bundle generator:

```text
--common-substrate merlin.common_compiler_substrate.v6
--python-source-root <the prepared common-v6 root>
```

Keep the hardware and software inputs, public corpus, numerical policy,
toolchain and qualification selections explicit. Phase 1 freezes the release
and its access profile. Run Phase 2 through the ordinary source-revision search
with that inherited profile.

## Choose optimization policies in Phase 2

The compiler can choose supported blocking, loop order, operand retention,
composition, copy elimination and configuration cleanup from the current
operations, shapes, layouts and resources. Changes are ordinary compiler source
revisions; each queued evaluation uses the frozen bytes of that revision.

V6 exposes the existing shared schedule entries; it does not expose every
original-caller model-building API. In particular,
`SourceDenseLowering.convolution_route`, `max_convolution_packing_bytes`,
`offload_plan` and `with_complete_offload_cost` currently require trusted host
orchestration. The separate opt-in source-choice workflow below can compile
predeclared alternatives through the broker. Ordinary authoring/search launchers
do not yet select it or construct a qualified generated-target adapter.
See [checked whole-model offload selection](shared_offload_selection.md) for
their executable scope and fallback rules.

Exposing another host service requires a fresh original selection before
authoring, with the same implementation and checks for every treatment. Its
bounded requests may choose legal alternatives; they cannot supply provider
callbacks, effect declarations, qualification or a cost verifier. A resumed
profile, saved hash or candidate manifest cannot create that authority. The
provider must implement the actual qualified generated compiler being tuned;
substituting a protected reference compiler would change the experiment.

Pass order is constrained by prerequisites, IR levels and invalidations. Current
entries validate explicit orders. A rewrite that changes instructions requires
new emission and program-size checks; spill insertion requires fresh allocation
and completion checks. Automated search over complete-model pass sequences is
not provided by the release.

Keep physical capacities, numerical policy, qualification and verification gates
fixed during tuning. Most `ScheduleBudgets.max_*` fields bound verification
work; increasing them is not a performance optimization. When costs are unknown,
use the entry's documented conservative policy or supply calibration evidence.

Inspect realized costs when choosing alternatives. Fusion can shorten traffic
while lengthening live ranges and increasing register pressure. Exact device
spill handling requires declared register classes and save/reload effects; the
native toolchain retains host CPU register allocation. Configuration cleanup
removes only writes proved redundant under declared persistence and reset rules.
Unknown instruction effects remain barriers.

### Evolve policies and passes

Parameter search selects legal control points. Heuristic search changes the
compiler policy that selects those points from current shapes, layouts,
resources and qualified costs. Both use the same frozen legality checks.

A candidate may also implement its own transformation where its original edit
contract permits it. That transformation must pass the independent checks for
its actual output; a proposed rule cannot certify itself. Unknown semantics or
unsupported proof coverage retain their refusal. Automatic discovery of general
passes is a separate research/evaluation claim, not provided by this service.

Promoting a transformation into the shared library requires a new reviewed
release and equal access for every compared treatment. Do not rewrite a frozen
library or verifier during a tuning run. Freeze the learned policy before
evaluating held-out shapes, configurations and complete models.

## An original-owned source-choice service

`SourceDenseChoices` binds one original source, its explicit entry, one
`SourceDenseLowering` configuration and a finite roster of permitted choices.
Each choice derives from that same configuration: it cannot replace resources,
numerical policy, qualifications, ABI or provider callbacks. Original source,
control, library or evidence changes require a new selection. A preissued
offload plan must match the resulting configuration and source; convolution
normalization cannot silently inherit an old plan.

For example, an original caller can declare two loop orders:

```python
from merlin.llvmlower.source_choices import SourceDenseChoice, SourceDenseChoices

choices = SourceDenseChoices(
    public_source,
    entry="forward",
    baseline=original_config,
    choices=(
        SourceDenseChoice("ordinary"),
        SourceDenseChoice("n-major", controls=(("outer_order", ("n", "m")),)),
    ),
    output_root=private_control / "source-products",
)
```

The host selects `source-choices-v1` through `select_workflow`, passes this exact
live owner as `source_choices`, and supplies `workflow.build_registry()` to the
existing `Broker`. The private receipt directory and output root must be
disjoint from the candidate; the output root must lie below the receipt
directory. Only the fixed `compile-source-choice` action is registered. No
candidate command or tool probe is inherited by this workflow.

The candidate writes a direct child file such as `choice.json` containing:

```json
{
  "schema": "merlin.source_dense_choice_request.v1",
  "source_sha256": "<SHA-256 from the original advertisement>",
  "choice_id": "n-major"
}
```

The normal broker client submits it with
`compile-source-choice request=choice.json`. Requests cannot supply source or
output paths, controls, callbacks, effects, costs or qualification. The broker
retains the actual request bytes privately and returns only bounded compilation
status and artifact identities. The original host retains `owner.results` for
complete product inspection and independent execution checks. Concurrent or
recursive compilation through one owner refuses rather than sharing mutable
provider state.

This route calls ordinary `lower_model_file` and preserves its current
host-domain-only restriction for shared source lowerings. It checks the actual
named entry, linked products, provider and completed recipe; compilation alone
supplies no numerical or runtime qualification. Broker admission deadlines are
checked before and after compilation and result verification. They do not
preempt an in-process Python callback or establish a new hard resource limit.

Select only original authorized public development sources and freeze the
service/checker closure before authoring, equally across treatments. Neither
the constructor nor a saved registry issues that authorization or fresh Phase 1
lineage. The ordinary authoring gate continues to reject this new workflow;
connecting it to agent search still requires the original launcher, generated
revision adapter and qualification contracts.

Reproduce the shared mechanism with
`merlin/tests/infra/test_source_choice_broker_model.py`: real broker requests,
ordinary compilation and complete native recurrent execution under two neutral
resource models. These tests establish no Gemmini execution qualification.

## Guarantees and limitations

The shared checks bind analyses to the current IR, hardware and software
selection, numerical policy and library version. Stale analyses and invalid
explicit ordering are rejected. Repairs use bounded budgets and recompute the
affected analyses.

A different selection cannot inherit an existing checked result merely because
its own files are valid. Dense, pointwise and complete-row results retain the
original selection in their analysis context and check it again before returning
images or emitting code. Editing a diagnostic report or recomputing a companion
selection digest does not authorize the old result under new contracts; compile
again with the new selection.

Emission checks operate on the representations each entry explicitly supports.
Where required, actual emitted instructions and effects must be reconstructed
independently. A selected plan or a successful link alone does not establish
correct execution. Unknown operations, effects and unsupported handoffs remain
barriers. Retain the original legal route when an optimization is unavailable;
otherwise propagate its diagnostic.

Capability requirements differ by pass. For example, device spills require
declared register classes and exact save/reload semantics; complete-row
reductions require the original reduction domain and numerical order. The
library does not infer addressable accumulators, asynchronous completion,
supported masks or new approximation policies from a target name or shape.

Neutral execution tests establish shared-mechanism behavior. They do not qualify
an accelerator backend or establish hardware latency. Consult the
[supported domains and limitations](../reference/shared_compiler_passes.md)
before selecting an entry.

Generality means running unchanged shared algorithms and checks with different
declared resource and execution models. It does not mean every pass is useful
on every target. Demonstrate performance transfer separately on held-out models
and configurations with independently qualified bindings. Phase 2 may tune
legal policies; it must not change the contracts to make a candidate pass.

## Compare compilers fairly

Every V6 treatment must receive the same shared algorithms and release identity.
V6 includes generic ISA parsing and generation, so comparisons must account for
that common access. Preserve the separate grants for target facts, software
specifications, runtime tools and independent qualification. Protected reference
implementations and evaluation answers remain outside authoring inputs.

Freeze tuned policies before evaluating unseen workloads. Use identical inputs
and unchanged numerical thresholds, checking complete outputs and carried state.
Measure complete models, including host arithmetic, packing, copies, layout
conversions, dispatch and accelerator work. Report compilation time, peak memory,
code size, commands and requested bytes separately from paired execution timing.
Functional Spike instruction counts are not accelerator latency cycles.
