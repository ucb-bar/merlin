---
title: Checking LLVM region replacements
kind: reference
status: current
owner: compile
last_verified: 2026-10-21
related: [panel_transfers, optimization_pass_pipeline]
code_refs:
  - src/merlin/llvmlower/llvm_region_boundary.py
  - merlin/tests/ir/test_llvm_region_boundary.py
---

# Checking LLVM region replacements

`LLVMRegionBoundary` checks the structural boundary of a proposed replacement
inside literal LLVM source. A target adapter can use it to verify a small copy
loop while retaining surrounding arithmetic, conversions and unknown operations
byte for byte. It neither chooses the replacement nor assigns semantics to
instructions.

The checker reuses the public LLVM token spans and indexes an explicitly selected
void function. It checks direct control flow, unique local names, reachable blocks,
region entry and exit, dominance and instruction order. It can reject values that
escape a selected region and require exact replacement lines with unchanged
outside bytes and control/SSA frontiers.

## Using the boundary checker

Select actual function arguments and block names from the LLVM produced by the
compiler. For a function with pointer and integer arguments:

```python
from pathlib import Path
from merlin.llvmlower.llvm_region_boundary import LLVMRegionBoundary

source = Path("module.ll").read_text()
boundary = LLVMRegionBoundary(
    source, function_name="work", argument_types=("ptr", "i64")
)
nodes = boundary.closed_region("header", "exit")
boundary.require_no_liveouts(nodes)
proof_lines = boundary.dependency_union(nodes)
budget = boundary.proof_budget(proof_lines)
```

The consumer supplies any additional control facts through `include_lines`, and
must prove their connection to the changed operations. `require_precedes` checks
both dominance across blocks and ordering within one block.
`require_diamond_loop` checks the supported six-block guarded loop topology.
After producing a candidate, call `verify_outside(candidate, replacements)`, where
`replacements` maps original line indices to their exact replacement text.

## Connecting a shared pass to a target

For transfer coalescing, the existing shared panel planner chooses legal rectangles
from typed source/layout contracts and transfer limits. A target-owned adapter
decodes the actual copy and configuration operands, proves the required byte
coverage and pointer bounds, and realizes the selected rectangle. The boundary
checker constrains that adapter's changes to the selected LLVM region. RTL facts
can supply structural limits; software rules and independent qualification must
establish the physical mapping, completion and ownership semantics.

Phase 2 can propose replacements or tune a planner within those original limits.
It cannot use this structural check to authorize unknown instruction effects,
different numerical behavior or missing target capabilities. Unsupported cases
retain the original program when the caller supports fallback.
Availability in Merlin does not extend a frozen shared-library export. Phase 2
authoring needs an explicitly reviewed release containing the checker; compared
treatments must receive the same release and verification rules.

## Bounds and guarantees

Whole-source indexing defaults to at most 1 MiB, 32,768 lines and 4,096 blocks.
The aggregate proof union separately defaults to 64 KiB and 4,096 components,
including the complete indexed CFG. Callers may select stricter limits. Candidate
source retains the same scan limits; changed fragments are charged to the proof
budget. The semantic consumer separately owns its solver and execution deadlines.
These limits do not enlarge existing whole-module verification budgets.

The accepted function grammar is deliberately narrow: scalar integer/pointer
arguments, unquoted local names, direct branches and void returns. Unsupported
control, ambiguous names, metadata and poison/provenance assumptions in the
dependency domain refuse. LLVM verification remains required before and after
rewriting. Structural success does not prove arithmetic equivalence, opaque
effects, accessible memory, completion, hardware support or performance.

Tests cover different integer widths, control bypasses, side entries, liveouts,
dominance, poison flags, outside changes and budget boundaries. Native LLVM
assembler/disassembler checks validate the accepted examples' syntax; they do
not qualify an accelerator implementation.
