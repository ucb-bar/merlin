---
title: Select checked whole-model offload alternatives
kind: guide
status: current
owner: core
last_verified: 2026-10-19
related: [../reference/shared_compiler_passes.md, shared_pass_phase2.md]
code_refs:
  - src/merlin/llvmlower/offload_plan.py
  - src/merlin/llvmlower/device_offload_selection.py
  - src/merlin/llvmlower/device_preparation.py
  - src/merlin/llvmlower/device_offload.py
  - src/merlin/llvmlower/device_build.py
  - src/merlin/runtime/backends/spike_model.py
  - src/merlin/llvmlower/source_dense.py
  - src/merlin/llvmlower/source_model.py
  - src/merlin/llvmlower/native_source_model.py
  - src/merlin/perf/offload_profitability.py
  - src/merlin/runtime/host_provider.py
---

# Select checked whole-model offload alternatives

Use this interface to retain some original CPU computations while offloading
other already-admitted dense operations. Selection happens before replacing the
typed source, while the CPU alternative still exists. The target provider keeps
ownership of instruction lowering and execution. Existing target-IR scheduling,
configuration cleanup and emission checks remain separate passes at their
appropriate IR levels.

## Required inputs

Start with an original `SourceDenseLowering` configuration: selected RTL facts,
software/ISA semantics, assigned storage, numerical policy, resource budgets,
independently qualified primitive/ABI effects, and provider preparation/build
callbacks. A plan does not provide any of those permissions.

`offload_context(config)` binds all source-lowering controls, logical ABI and
effects, selected contract bytes, adapter semantic identities and the versioned
source/cost checker owners. It excludes the plan and comparison themselves.
Actual callback implementations, tools, provider products and runtime behavior
still require their ordinary independent checks. This is source/context custody,
not a proof of implementation or a new authoring-access grant.

The initial source domain is a complete static, flat single-block tensor entry
with pure standard scalar/tensor/Linalg operations. The graph includes host glue.
Unknown effects, required placement, opaque attributes, calls and structured
control refuse explicit selection. Ordinary compilation without this option
keeps its existing behavior.

## Bind a source and an implementation plan

```python
from dataclasses import replace

from merlin.llvmlower.offload_plan import bind_source, realize_offload_plan
from merlin.llvmlower.source_dense import offload_context

context = offload_context(config)
source = bind_source(module, entry="forward", context=context)

# `plan` is an existing GlobalPlan covering every node of source.program.
# `device_alternatives` is its original, explicitly supplied alternative roster.
selection = realize_offload_plan(
    source, plan, device_alternatives=device_alternatives, context=context,
)
selected_config = replace(config, offload_plan=selection)
```

The source binding supplies unchanged singleton CPU alternatives with unknown
costs. The initial device domain is singleton exact signed integer contractions
using the unchanged logical tensor ABI and one selected endpoint. Device
implementation identities must match the original adapter qualification. No
implicit fusion, representation transition, resident handoff or allocation is
realized by this interface. Splitting an already-admitted numerical composition
produces a diagnostic.

Ordinary discovery first establishes which dense computations are legal under
the current facts, software contract, storage and effects. Selection can narrow
that set; it cannot add unsupported work. Retained CPU operations keep their
original bodies. Reparsed equal source binds to its current operation objects
through the complete source/graph identity and global ordinals. Equal shapes or
provenance labels cannot accidentally select both operations.

Pass prerequisites and source/context stamps use the existing pipeline session.
Changed source, graph, plan, numerical policy, packing controls or resources
requires replanning. A source normalization that changes the bound graph also
requires a new plan; this interface does not silently transfer a prior choice.

## Select with complete-cost evidence

The ordinary route is the current dense discovery/offload behavior with its
normal CPU fallback; it is not necessarily an all-CPU baseline.

Use `selected_config.with_complete_offload_cost(...)` to compare original
complete observations under explicit cold and warm costs, one unchanged unit,
and a live original verifier. It returns a configuration with its optional
`OffloadCostDecision` bound as `offload_comparison`. The factory derives exact
route identities:

- `ordinary_dense_route_identity(selection.stamp)` for the ordinary route;
- `selection.plan.digest` for this particular candidate.

The lower-level `compare_complete_offload_cost` interface also remains available
to an original caller that supplies those exact identities explicitly.

The original verifier must join the current source/context and route identities
to actual observed artifacts, inputs, state/cache policy, complete execution,
numerical/legality results and qualified timing or calibration. Artifact hashes
or a populated report alone do not establish those joins. Rebuilding a different
binary needs new measurement or independently justified transfer qualification.

A candidate is selected only when its complete upper cost is strictly below the
ordinary route's lower cost under the selected regime. Unknown, overlapping or
regressing evidence retains the ordinary route, including its normal CPU
fallback. It does not replace unavailable candidate work with an arbitrary new
route. Expired or changed evidence refuses the explicit selection.

Without a comparison, the supplied plan is an explicit structural choice and
makes no profitability claim. Nanoseconds stay nanoseconds; functional simulator
instruction counts never become accelerator cycles. Complete-call timings also
do not separately identify host and accelerator contributions.
Observed sample minima and maxima describe those samples; even identical
executable bytes do not turn them into bounds on future latency. Live evidence
validation has a compilation cost and may dominate a small build. Report that
cost alongside the observed execution measurements.

An original cost verifier checking several completed provider records can use
`revalidate_host_providers((original_record, candidate_record))` to check their
combined file set. The helper hashes each shared path once within that read-only
call, checks every record's expected identity and actual receipt, and rejects
changed original records. Each later call reads the files again. This supplies
link custody only; timing, numerical results and execution qualification remain
the original verifier's obligations. The caller must retain ownership against
concurrent writers; the helper does not create an atomic filesystem snapshot.
`max_records` bounds the batch before serialization or file access (default 4096).

## Phase 2 integration boundary

These original-caller APIs do not extend an inherited Phase 2 authoring profile.
Existing public dense scheduling alternatives remain available only through the
original selected profile and qualified capability contracts. Whole-source
offload plans, bounded convolution-panel controls and complete-cost comparisons
currently require trusted host orchestration; the ordinary Phase 2 services do
not yet expose these new choices.

The explicit `SourceDenseChoices` owner and separate `source-choices-v1` broker
workflow accept finite source-bound choices while retaining the original
resources, numerical/effect qualifications and cost verifier. See the
[source-choice service guide](shared_pass_phase2.md#an-original-owned-source-choice-service)
for original selection, private outputs and candidate request format. This
host-domain prototype does not extend an inherited profile or qualify a fresh
generated compiler. Its implementation and verification closure must be fixed
before authoring and identical across compared treatments. Importing host
providers or measurement internals into an inherited compiler sandbox is not
an integration mechanism.

## Compile and inspect the result

Pass `selected_config` through the ordinary `shared_source_lowerings` argument.
Source receipts distinguish the candidate plan, optional cost comparison and
actual selected route/operation ordinals. These records are intentions and
checked source correspondence; final emitted effects and complete execution
remain necessary evidence.

`lower_model` and `lower_model_file` also support this transaction for native
complete-model execution with `targets=("host",)` and explicitly host-domain
providers. The same source preparation, borrowed-buffer lifetime checks and
provider object/link custody apply. The native route records original inherited
dynamic imports, permits no additional provider imports, and requires an actual
no-undefined link. Dynamic-loader, external-library, provider-body and timer
qualification remain independent obligations.

This native path supports neutral executable tests of complete generated glue
and state. It does not execute or qualify accelerator instructions. Device-domain
providers require their existing target runtime and final-image admission.
Missing public target effects/decoding/consuming-emitter interfaces remain
integration requirements; Merlin does not infer them from opaque backend output.

For a paired experiment, use identical input and initial state, compare every
output and carried value against an independent reference, and reconstruct
actual helper calls, commands and transfer requests from emitted/executed
products. Report compile time/memory, code size, requested bytes and qualified
execution time separately. Reset invocation-owned state outside any measured
call, and retain the original cache/reuse policy for each timing regime.

With the existing selected native MLIR/compiler toolchain and a task-local
`TMPDIR`, reproduce source selection, invariant mutations, and complete native
execution with:

```sh
python -m pytest \
  merlin/tests/infra/test_source_offload_selection.py \
  merlin/tests/infra/test_source_offload_cost.py \
  merlin/tests/infra/test_native_offload_model.py
```

The native experiment exercises two rectangular resource descriptions and
different element widths. It records complete recurrent outputs, source/link
receipts, compilation wall time and sampled process-tree RSS, ELF sections, and
observed primitive/packing requests. These requests cover the instrumented
provider and bridge; retained CPU arithmetic and glue traffic are unmeasured.
The all-CPU control has no provider instrumentation: absence of provider calls
is established through source/link custody and separate complete execution.
This experiment alone does not supply whole-call execution timing.

`merlin/tests/infra/test_native_offload_profitability.py` adds a separate paired
warmed-call experiment. It retains raw nanosecond samples and primitive traces,
checks every timed output, consumes the cost decision through ordinary
compilation, and checks the complete selected state history. Timing reuse
requires exact selected/observed image identity; differing images produce a
transfer refusal. Cold execution and physical accelerator timing remain
unknown. Use `-o tmp_path_retention_policy=all` when retaining successful pytest
artifacts for an independent audit.

## Select an ordinary target-package route

`DeviceRouting.offload_selection` carries the same source/plan and complete-cost
checks into the ordinary package-backed device compiler. This is separate from
`SourceDenseLowering.offload_plan`: use the package route's own context and route
identities. It narrows operations already accepted by the original
`DeviceRouting.select`; it never grants the package additional capabilities.
CPU alternatives keep their original computation and logical buffers.

Start with an original routing configuration whose selected package, public
RTL/software facts, numerical policy, logical ABI and final linked-image policy
already support the original device route. The initial domain is one flat,
static, pure tensor entry, including its host glue, with singleton exact signed
integer contractions. Only one unchanged device endpoint and the declared
logical host-buffer kernel ABI, version 2, are supported. There are no resident
handoffs, representation changes or new fusion implementations in this choice.

Bind the exact prepared source **before** building the selected route. The
original caller must retain a private copy before routing rewrites the working
file in place. A deterministic preparation observation may supply that copy;
it does not qualify the target implementation. Recompilation must produce the
same complete prepared IR and graph. Normalization, changed dependencies or a
different entry require a new plan rather than an ordinal-only rebinding.

```python
from dataclasses import replace

from merlin.compile.baremetal_model import compile_saved_model
from merlin.llvmlower.device_offload_selection import bind_device_offload_source
from merlin.llvmlower.offload_plan import realize_offload_plan

owner = bind_device_offload_source(
    prepared_module,              # original retained, pre-routing snapshot
    routing=original_routing,
    entry="forward",
    configuration=original_build_options,  # tuple of explicit scalar fields
    source_pins=original_tool_runtime_pins, # tuple of (absolute path, SHA-256)
)
# Construct the complete plan and admitted alternatives from owner.source.
bound_plan = realize_offload_plan(
    owner.source,
    plan,                         # existing GlobalPlan covering every entry node
    device_alternatives=original_device_alternatives,
    context=owner.context,
)

# Original manual proposal for calibration, not a profitability decision.
proposal = owner.select(bound_plan, explicit=True)
routing = replace(original_routing, offload_selection=proposal)
receipt = compile_saved_model(**original_compile_options, device=routing)
```

Each device alternative must be an admitted singleton using
`placement=original_routing.device` and
`implementation=owner.implementation_identity`. The implementation identity binds
the actual package payload and tool; it is not an execution qualification. The
selected operations must be a subset of the original selector's exact accepted
operations. Complete source/graph identities bind ordinals to current operation
objects; equal shapes or labels do not select additional instances. Selector
outcomes must remain deterministic, and must return actual Booleans.

The owner binds the current package payload, public facts/contracts, ABI and
transport, numerical policy, declared budgets, original callback selection,
final-image policy and checker/library versions. Explicit tool/runtime pins and
build options extend this identity scope. They do not establish a dependency
closure or performance model. Host compiler flags, tools, runtime, input/state
ownership, cache/reset policy, completion and measurement details outside
`DeviceRouting` remain obligations of the original live cost verifier.

The package must already contain its tool and stay byte-immutable, including
its member roster and executable bits. Build outputs belong outside that frozen
payload; the checker does not exempt generated files by suffix. Default analysis
bounds are 4,096 source operations, 4,096 package members and 64 MiB of package
files. Original callers can select bounded limits explicitly. These are compiler
work budgets, not accelerator memory capacities or performance estimates.

### Consume complete costs without changing authority

`owner.select(bound_plan)` preserves the original selector route by default.
`explicit=True` instead chooses the original caller's unpriced subset for
calibration and reports `explicit_unpriced` with timing `UNKNOWN`. It cannot be
combined with a cost comparison. Neither mode supplies a speedup claim.

After obtaining independent complete observations, start from the ordinary
unpriced selection and attach their original owner:

```python
selection = owner.select(bound_plan).with_complete_cost(
    original=original_observation,
    candidate=candidate_observation,
    original_costs=original_cold_and_warm_terms,
    candidate_costs=candidate_cold_and_warm_terms,
    scope=original_measurement_scope,
    regime="warm",
    verify_original=original_live_verifier,
)
routing = replace(original_routing, offload_selection=selection)
```

The factory binds the exact prepared-source stamp,
`owner.original_selection_sha256` for the original accepted route, and
`bound_plan.plan.digest` for the candidate. Do not substitute
`ordinary_dense_route_identity` or a `SourceDenseLowering` context. The original
verifier must join those identities to the actual observed compilers, complete
artifacts, numerical results, identical inputs and carried state, dependencies
and timing scope. New binaries require new evidence or independently justified
transfer qualification. A saved measurement JSON or matching hashes alone
cannot issue that authority.

Only a strict complete-cost opportunity selects the candidate. Missing,
overlapping, unknown or regressing costs retain the original selector route,
which can itself include device computation. Stale evidence or inconsistent
source/package/plan identities refuse the explicit request. Retaining the
original route does not relax numerical admission or repair an illegal compiler.

### Check what reached the final program

Ordinary preparation freezes the selection before provider work. The rewrite
must route exactly the selected current operations, retain CPU bodies and record
actual source ordinals in the device sidecar. Trusted preparation products and
that realized sidecar retain their own file identities through compilation.
Package output and any selected target-IR cleanup continue through their existing
consuming emitter and final linked-image checks.

Later selection checks cannot refresh an earlier policy receipt. Finalization
retains the original accepted ELF result and report chain, reopens frozen report
bytes after policy validators, and reads the actual final ELF against its
original link hash last. Callback-free checks then retain the original
selection, current source/resource/numerical context and sidecar identities. These checks establish correspondence
and custody; they do not decode arbitrary backend output or prove the target's
numerical, ownership or completion behavior.

Each selection boundary obtains its record through one complete live check of
the actual routing. Original selector and cost callbacks still have their
required checks before and after execution. A lookup of the already-bound
source returns its current operations only after full validation; a reparsed
module is independently read and bound to its own operation objects. No prior
verdict or file hash is reused across these boundaries. For an explicit record
inspection against a routing copy, use `selection.record(routing=current)`;
changing that copy's numerical policy or resources requires replanning too.

This route refuses group routing, catalogs and `ExactOffloadSelection` mixing.
An opaque `prepared_model_transform` or later `post_offload_transform` ABI rewrite
is unsupported. An original `DeviceRouting.prepared_transform` may retain an
observation, but its resulting source must still match the prebound prepared
snapshot exactly. If the actual choice routes no device operations, an active
`target_configuration` also refuses because no target IR consumes that pass.
Existing selector, catalog and certified-source legality checks remain in force.
A plain selected device route now triggers normal preparation even when no host
optimization flags are selected; inert routes remain inert.

### Reproduce the package-path experiment

With the selected MLIR, RV64 compiler and functional simulator available, use a
task-local `TMPDIR` and retain successful artifacts:

```sh
python -m pytest \
  merlin/tests/infra/test_device_offload_package_model.py \
  -o tmp_path_retention_policy=all
```

The fixture defines an original and an explicitly narrowed route through an
actual neutral package's typed target IR, executable LLVM, device objects and
final RV64 ELF. Its independent reference covers all seven returned tensors
after three carried-state steps; primitive traces also check the intermediate
calls. The pair records compilation/execution observations, ELF sections, actual
primitive payloads and static command buffers. It does not supply a qualified
complete-cost decision, accelerator latency or total host/transfer traffic.
Functional simulator instruction counts are not accelerator cycles. Executing
this recipe and auditing its retained products is required before making any
result claim; the fixture itself is not hardware support.

This is an original-caller ordinary-build API. A fresh Phase 1 generated compiler
and a live Phase 2 broker are not connected to it by this guide or by the neutral
fixture. Existing authoring/evaluation boundaries remain unchanged. Any such
integration must carry the original qualified candidate and runtime/cost owners,
preregister the same versioned implementation and checks for all treatments,
and refuse missing public backend hooks. It must not supply a protected
reference compiler through a host service.
