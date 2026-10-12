---
title: Whole-program operation profiling
kind: design
status: current
owner: core
last_verified: 2026-10-12
related: [normal_search_feedback, runtime_escape_audit, compile_debugging]
code_refs: [src/merlin/llvmlower/op_profile.py, src/merlin/llvmlower/op_profile_structural.py, merlin/runtime/c/merlin_op_prof.c, build_tools/scripts/k1_op_profile.py]
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
