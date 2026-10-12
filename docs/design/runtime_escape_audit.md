---
title: Audit runtime helper calls in emitted compute
kind: design
status: current
owner: compiler
last_verified: 2026-10-12
related: [normal_search_feedback, compiler_plane, compile_debugging]
code_refs: [src/merlin/kernels/escape_audit.py, packages/merlin-mining/src/merlin/mining/escape_sweep.py, src/merlin/llvmlower/selfcopy.py, build_tools/scripts/k1_escape_cost.py]
---

# Audit runtime helper calls in emitted compute

A runtime helper call can represent required work or avoidable lowering overhead.
The audit locates such calls in compiler-emitted functions, including available
loop context. It does not prove redundancy or authorize deleting a call.

## Bind scope and control flow

`merlin.kernels.escape_audit.audit(object_path, elf_path)` takes function scope
from symbols defined by the compiler object and control flow from the linked
ELF. Unrelocated object branches cannot substitute for linked control flow.
Exclude unrelated statically linked runtime functions from the compute scope.
Missing scope or disassembly remains unknown; an unresolved loop structure
cannot establish that helper calls are outside loops.

The report retains call sites, helper names, loop membership and
`depth_reliable`. Enclosing backward-branch spans do not establish nesting depth
when block layout prevents a containment chain. Site count alone cannot price
a call repeated inside a loop.

## Compare before optimizing

Vary independent sizes and precisions so a single-tile case cannot hide repeated
work. Compare unchanged source/ABI and numerical obligations across alternatives;
inspect actual object and ELF identities to confirm that a transformation reached
the executable.

Prove a copy or allocation redundant at the IR level before removing it. Alias,
effect, lifetime and output-publication requirements still apply. A structurally
clean report gives no measured speedup, and a helper call is not inherently a
compiler defect.

`merlin.rvvgen.escape_sweep` runs the optional mining screen across selected
inputs. `build_tools/scripts/k1_escape_cost.py` provides a board cost study for
its supported workflow. Preserve counter and timing brackets, independent
checks, held sizes and measurement variability. Record results under run
artifacts rather than adding tuning winners to this design.
