---
title: "Runtime loop generations and returned ownership"
kind: design
status: current
owner: core
last_verified: 2026-11-17
related: [execution_owned_results, llvm_pipeline_rationale]
code_refs: [src/merlin/llvmlower/dealloc_lifetime.py, src/merlin/llvmlower/dealloc_promotions.py, src/merlin/llvmlower/pipeline.py, src/merlin/llvmlower/owned_result_lowering.py, src/merlin/llvmlower/owned_result_program.py, merlin/tests/ir/test_promoted_loop_ownership.py, merlin/tests/runtime/test_resizing_owned_results.py]
---

# Runtime loop generations and returned ownership

The native lifetime checker now accepts runtime/zero-trip serial loops with
fresh direct carried allocations and explicitly promoted returned ownership.
The source loop, bounds, arithmetic, element computation, masks and allocations
remain executable. Result dimensions still come from the actual invocation;
neither shape prediction nor annotations establish ownership.

## Independent proof

Each replaced memory slot must yield a distinct direct body allocation. Its
previous generation has exactly one release, guarded by a loop flag that enters
false and yields true. Every synchronous read or view precedes that release;
fresh generations cannot escape, be freed, share owners or change result slots.
Complete nested operation/effect checking remains active. Initial roots must be
actual local allocations with closed uses, never guessed caller ownership.

The final buffer crosses an ownership promotion: on the true flag it is the
last fresh generation; on the false flag it is a complete independent copy of
the zero-trip seed. The result must transfer through the original function
return, without an intervening release or unknown consumer. Tuple promotions
retain every result coordinate and separate owner, including different storage
types and independently changing extents. A known positive constant step remains
required; runtime lower/upper bounds and zero iterations are admitted.

`bufferization.clone` itself permits aliasing, so its spelling supplies no
freshness fact. The normalizer materializes the upstream ranked identity-layout
alloc/dimension/copy implementation in the original false branch. Only then can
the proof replace aggregate seed/final-owner deallocation: seeds release once
after promotions, and returned fresh owners remain live for their caller.
Other candidate/retained aliases and ownership results remain checked; unproved
combinations retain their ordinary refusal.

## Liveness and ordinary guards

The pinned upstream liveness pass follows a conservative loop/branch alias union.
It can treat the seed as a returned alias and move a direct free after
`func.return`, producing invalid IR. Before that pass, a typed identity memref
view shields only the closing seed releases established by this proof. The
ordinary liveness optimization still runs for other allocations. The independent
checker follows the view and re-proves complete zero/positive-trip lifetimes
after the pass; a provenance marker cannot bypass it.

Both native runner variants forward every preparation/ownership callback across
the split stages. Allocation-product guards, count/byte budgets and all normal
ownership/deallocation checks remain active. Compiler products, saved programs
and selected frontend evidence pin the new proof implementation.

## Reused upstream work and measured scope

Source review used LLVM revision
`a47bddccec30255619bb8c37fa59700e661d4e66`: the ownership-based deallocation
implementation and `dealloc-loops.mlir`, ranked `CloneOpConversion` and its
static/dynamic/layout tests, and `OptimizeAllocationLiveness.cpp`. The local
proof adapts that executable ownership structure to the existing independent
checker; no LLVM/Triton build or target-specific ownership opcode is introduced.

Checks cover malformed flags, missing/wrong copies, early/duplicate releases,
hidden calls, escaped pointers and initial owners. Real native/saved/C programs
reuse one compiled model across changing shapes and trip counts. Two independently
resizing i32/f64 states retain exact signaling/quiet NaN, signed-zero and subnormal
words. Actual serial target images retain zero-trip, empty and resized results;
a negative count still fails through its original source assertion.

This closes the declared fresh-returned-generation form. General branch-selected
or permuted owners, arbitrary while/nested ownership, unknown alias/retained
state, dynamic steps and other result consumers need further proofs. It supplies
no physical view, effect, arithmetic-policy or accelerator permission. The
portable tensor computation and public result/package schemas are unchanged.
