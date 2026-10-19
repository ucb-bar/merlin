---
title: Select checked whole-model offload alternatives
kind: guide
status: current
owner: core
last_verified: 2026-10-19
related: [../reference/shared_compiler_passes.md]
code_refs:
  - src/merlin/llvmlower/offload_plan.py
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

A future host service must accept bounded source-bound choices while retaining
the original resources, numerical/effect qualifications and cost verifier. Its
implementation and verification closure must be fixed before authoring and
identical across compared treatments. Importing host providers or measurement
internals into an inherited compiler sandbox is not an integration mechanism.

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
