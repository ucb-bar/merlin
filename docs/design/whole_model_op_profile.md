---
title: Whole-program operation profiling
kind: design
status: current
owner: core
last_verified: 2026-10-18
related: [normal_search_feedback, runtime_escape_audit, compile_debugging]
code_refs: [src/merlin/llvmlower/op_profile.py, src/merlin/llvmlower/op_profile_structural.py, src/merlin/llvmlower/lower.py, src/merlin/runtime/backends/spike_model.py, src/merlin/runtime/operation_profile.py, src/merlin/compile/baremetal_model.py, src/merlin/model_check/full_build.py, merlin/runtime/c/merlin_op_prof.c, build_tools/scripts/k1_op_profile.py]
---

# Whole-program operation profiling

Operation profiling attributes a selected function's execution to its top-level
IR operations. It complements a complete program timer; it does not establish
accelerator placement or a hardware-cycle cost by itself.

## Instrument a typed function

`merlin.llvmlower.op_profile.instrument(text, functions=("forward",),
structural=True)` returns instrumented MLIR and a marker table. The structural
route parses complete typed operation boundaries and supports exactly one
selected function. Unsupported syntax or multiple selected functions refuse.
The legacy text-based multi-function API remains separately available; it does
not acquire the typed route's scope.

The typed emitter checks the actual marker sequence and callback declaration,
then removes those additions from a private clone and compares the complete
remaining IR with its original source. Arithmetic, effects, typed metadata
(including signed zero), interfaces and dataflow must remain unchanged. The
table must describe the actual retained operations. This establishes source
correspondence, not runtime counter semantics or machine scheduling.

Marker calls sample the selected runtime counter and credit elapsed ticks to the
previous operation. A sentinel closes the final interval before return. The
runtime emits `PROF <id> <ticks> <hits>` observations that join to the same
build's marker table. Preserve the selected timer's units and scope; counter
ticks must not be relabeled as processor or accelerator cycles.

The table records operation, result type and available `prov.*` attributes.
`join_key` prefers `prov.fqn`, then `prov.region_id`, then the operation name.
These keys support correspondence; a matching key alone does not prove equal
semantics or a common execution boundary across compilers.

## Check perturbation and attribution

Build an uninstrumented control through the same source and option selection.
The profiling driver compares whole-program wall measurements against that
control using its declared perturbation threshold. Marker calls can alter
optimization, layout and scheduling, so a passing finite control is scoped
experimental evidence, not a proof that every operation is unaffected.

Allocation hoisting can move cost into another marker interval. Short operations
can fall below counter resolution, and nested work belongs to the enclosing
interval unless separately instrumented. Report aggregate coverage, missing
observations, repetitions and complete timer boundaries. Pipeline vectorization
intent is not evidence of the instructions actually emitted.

For the supported board workflow, use `build_tools/scripts/k1_op_profile.py`
with explicitly selected model inputs and runtime. Keep measured breakdowns in
run artifacts, including source/object/ELF identities and control observations.
Protected programs remain outside pre-freeze tuning feedback.

## Saved-model and functional-simulator builds

The ordinary `compile_saved_model` Python API accepts `op_profile=True` alongside
its existing capture, hardware, host-package, numerical and execution selection.
The default is false. Compile-only runs retain the actual
`build/op_profile_table.json` path and SHA256 in `baremetal_model.json`; this
makes no execution claim. Executed runs retain `operation_profile.json` after
the original complete-output gate. Full binary output frames are parsed as
frames before reading their text diagnostics, so payload bytes cannot create
false profiler records.

The functional-simulator builder now delegates marker insertion to
`lower_model(..., op_profile=True)`. Graph, automatic row and automatic
destination transforms run first, with their original selections and checks;
markers follow them before upstream preprocessing. `lower/operation-profile.json`
binds the source at that boundary, emitted instrumented source and operation
table by SHA256. The existing table beside the image contains identical bytes.
IR audits record `operation-profiled` after the source transformation stages.
An unsupported typed profiler request fails rather than silently dropping marks.

This ordering prevents marker calls from suppressing these source transforms.
It does not make profiling transparent to later bufferization, vectorization,
allocation, LLVM optimization or execution. Direct callers that instrument
before lowering, including the existing board/debug workflows, retain their
earlier boundary; compare their own transformation records. Moving those
workflows requires preserving their selected fallback and debug behavior.

For the existing build-once/relink model path, select
`BuildSettings(..., op_profile=True)` (or the explicit keyword to
`BuildSettings.of`). The flag participates in image reuse, and a missing,
changed or indirect operation table prevents reuse. This is a programmatic
diagnostic selection; it changes no model-check CLI or numerical admission.
Typed preprocessing retains the existing descriptor-interface route for shaped
functions and preserves private scalar callback ABIs. Explicit interfaces remain
as written. A private scalar marker must not acquire a wrapper ABI merely
because a model uses the typed scalar preprocessing route. This does not remove
the implicit descriptor interface used by existing routed buffer declarations.
When running or relinking those images separately, join their own table and
complete console explicitly:

```python
import json
from merlin.runtime.operation_profile import summarize
from merlin.runtime.out_bin import binary_console_diagnostics

table = json.loads(table_path.read_text())
report = summarize(table, binary_console_diagnostics(console_bytes).decode())
```

The closed reader supports a single selected straight-line function. Every
operation must have the same positive execution count; observations, marker
totals, accumulated ticks and the terminal DONE must agree. Overflow, duplicate
records and incomplete traces refuse. Between invocations, the sentinel
accumulates harness gaps; these ticks remain separate from operation categories.
Marker bookkeeping, startup, output and the final dump are not attributed to
operations. Equal hit counts check consistency, not recursive-call legality or
source-to-machine equivalence.

All reported values use `raw_counter_ticks`, with no frequency or milliseconds
conversion and no timing qualification. The bare-metal marker implementation
reads `mcycle`; its meaning depends on the selected executor. Functional Spike
counts do not establish hardware or accelerator latency. Actual calibrated
timer semantics require independent runtime evidence, not RTL structure alone.

Semantic category names retain their derivation (provenance, typed operation or
scalar body). They never assign CPU/device ownership. An opaque call stays
unclassified; packing, conversions, copies and dispatch nested inside it remain
unresolved. Hoisted allocations can move between intervals. Measure a paired
uninstrumented control, record compilation/memory/code growth separately, and
inspect actual pass reports before using a profile to prioritize optimization.
Instrumentation does not guarantee unchanged fusion or row scheduling.
