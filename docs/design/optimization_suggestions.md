# Selecting optimization suggestions

`merlin_experiments.phase1.optimization_suggestions` intersects the optimization
catalog with explicit selected capabilities, actual public API access and parsed
workload IR. This is advisory filtering, not a compiler legality check, pass order,
performance ranking or import grant.

The native joint prompt path writes `OPTIMIZATION_SUGGESTIONS.json`, the selected
catalog, an exact API list, and a standalone `optimization_suggest.py`. It reads
only the already public software declaration, the descriptor's declared contract,
and staged public RTL facts. Their digests accompany the derived traits. It does
not inspect private models, references or timing measurements.

Initial preparation has no workload IR. Once a source is selected:

```sh
python optimization_suggest.py --ir source.mlir --output suggestions.json
```

The CLI derives only narrow structural facts: a source graph, ranked tensor
results, graph fanout, multiple returned values, named convolutions and explicit
reduction iterators. These do not prove alias freedom, immutable weights,
profitable fusion, pipeline legality or repeated invocation. Unknown premises
remain discovery questions. The compiler's actual matchers still decide legality.

## Capability declarations

The optional `optimization_capabilities` mapping in a reviewed software
specification uses orthogonal namespaces: `host`, `submission`, `compute`,
`numeric`, `memory`, `scheduling`, `runtime`, `workload`, `compiler`. Values are
`supported`, `unsupported`, or `unknown`. Workload and compiler claims inside a
software declaration are ignored by the selector: workload evidence must come
from IR and public API access comes from the selected library.

For example, explicit `compute.vector_lanes`, `compute.predicate_masks` and
`compute.vector_register_files` support a vector design starter. They do not
imply a vendor runtime or supported instruction adapter. Likewise asynchronous
runtime advice requires submission queues, transfer, event and allocation
contracts separately. A floating scale field does not establish floating matrix
compute. Absence of a declaration does not establish unsupported hardware.

Current native extraction recognizes these exact facts without chip-name rules:

- Software operand/accumulator numerical types.
- Contract `endpoint_kind=inline_asm_insn` as CPU/instruction-issued submission.
- Contract `compute_units[].kind=systolic` as a compute mechanism.
- Positive-byte RTL `memory_address_spaces` with explicit scratchpad/accumulator
  roles as local storage. This gives no physical alias, coherence or overlap proof.

- Scheduling and memory capabilities from `capability_facts`, each recorded with
  the fact that settles it under `capability_provenance` (also in
  `OPTIMIZATION_CAPABILITIES.json`):
  - `scheduling.queues` / `finite_queues`: derived dispatch entries in
    `timing_structure.command_classes` (else the ordering record's queue names);
  - `scheduling.async_issue`: a queue releasing a younger command at the older
    one's issue, or a cross-queue pair issuing before completion;
  - `scheduling.multiple_outstanding`: a queue whose completion order is
    `out_of_order` (release at issue to a client with several transaction slots);
  - `scheduling.dependency_tokens`: a hardware-tracked cross-queue pair (waits for
    completion only when the rows overlap);
  - `scheduling.persistent_configuration`: a verified extent that reads a field a
    separate configuration command latches;
  - `memory.resident_operands`: a later command of another class reading local
    rows an earlier one wrote, ordered after its completion;
  - `memory.strided_local_rows`: a verified transfer-in extent covering column
    blocks at a local block stride.

  The ordering record is the run's completion-order selection
  (`MERLIN_COMPLETION_ORDER_SELECTION`), used only when its bytes, record digest,
  base facts and base contract match the staged inputs; otherwise a derived
  `facts.command_ordering`; otherwise these stay `unknown` with the reason.
  `unsupported` is recorded only from an excluding fact (every release waits for
  completion; every memory client holds one transaction).

More explicit reviewed traits can extend that set. The six extra tightly coupled
mechanisms and the mixed precision, vector and asynchronous starter sets remain
separate design proposals. Selecting one grants no implementation or runtime ABI.

## Registry and source access

Each concrete suggestion includes its existing module/symbol, catalog family,
source applicability, remaining adapter requirements and requested observations.
Actual route, prerequisites, outputs and invalidations belong to the concrete
registered `PassSpec`; the selector marks missing registry metadata unknown. It
never invents knobs or flattens a compact counted-loop route into a finite region.
The policy/registry implementation can bind those records to actual passes.

`optimization_api_extension_review.json` reviews all nine unavailable catalog
entries (eight distinct modules). The first narrow export candidates are exact
preparation CSE and padding/gather transforms. Several entire modules include
runtime, verification, dynamic registry or broad toolchain dependencies. No new
module grant is made by this review. Existing sparse selected-library edits remain
the normal route; new exact leaves require a reviewed immutable library epoch.

## Boundaries

Display is bounded to twelve supported suggestions and twelve discovery rows;
the full per-mechanism status remains in JSON, so nothing is silently discarded.
Catalog display order is not optimization priority or pass sequence. No target
schedule, tile constant, model/capsule branch, historical result or private kernel
is supplied. Numerical, offload and emitted-code checks remain independent.

## Abstractions for general compiler development

The generated `abstractions` map reports actual selected-module access per layer.
Use [typed IR parsing](../../src/merlin/common/mlir_query.py) and the
[contract dialect](../../src/merlin/xdsl_dialects/contract.py) to retain graph
semantics and numerical obligations. Existing restricted
[multi-output compilation](../../src/merlin/compile/scheduling/multioutput_fusion.py)
and [shared-input matching](../../src/merlin/compile/scheduling/shared_input_groups.py)
connect useful graph regions to lowering;
[prepared operand planning/install](../../src/merlin/llvmlower/prepared_operand_schedule.py)
separates ownership/lifetime analysis from actual reuse installation.

The existing [Kernel](../../src/merlin/sched/ir/kernel.py),
[compact dense schedule](../../src/merlin/compile/scheduling/compact_dense.py) and
[index-program emitter](../../src/merlin/compile/scheduling/llvm_program.py)
provide a counted-loop route. The existing
[finite schedule pipeline](../../src/merlin/compile/scheduling/schedule_pipeline.py)
is a different bounded-region route. Optional schedule/interface dialects are
available representations, with target integration still requiring evidence.
[Shared host codegen](../../src/merlin/llvmlower/host_tensor_codegen.py) and
[host optimizations](../../src/merlin/compile/host_optimizations.py) own generic
CPU work and boundary handling. None of these module grants proves a complete
production graph path, legal fusion, or realized performance.

Change the relevant sparse library implementation and adapter rather than build
an alternate compiler around a fixed workload. Retain unsupported operator and
numeric contracts explicitly. Generalize policies from IR, resources and measured
costs so additional models can use the same compiler after this experiment.
