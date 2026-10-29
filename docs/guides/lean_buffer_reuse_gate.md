---
title: Check private buffer reuse and segmentation with Lean
kind: guide
status: current
owner: verification
last_verified: 2026-10-29
related: [verification, lean_loop_transform_gate, lean_pass_gate]
code_refs:
  - src/merlin/verify/buffer_reuse_bridge.py
  - src/merlin/verify/buffer_reuse_intake.py
  - src/merlin/verify/buffer_reuse_emit.py
  - src/merlin/verify/buffer_reuse_gate.py
  - src/merlin/verify/lean/Merlin/GraphLoopsMemorySCFAfterEntriesChain.lean
  - src/merlin/verify/lean/Merlin/GraphLoopsMemorySCFAfterChain.lean
  - src/merlin/verify/lean/Merlin/GraphLoopsMemorySCFAfterUntiled.lean
  - src/merlin/verify/lean/Merlin/GraphLoopsMemorySCFAfterUntiledLocal.lean
  - src/merlin/verify/lean/Merlin/GraphLoopsShapeEpochLine.lean
  - src/merlin/verify/lean/Merlin/GraphLoopsShapeEpochLineLink.lean
  - src/merlin/verify/lean/Merlin/GraphLoopsMemorySCFAfterLineLocal.lean
  - src/merlin/verify/lean/Merlin/GraphLoopsMemorySCFAfterLineTiledLocal.lean
  - merlin/tests/ir/test_compiler_buffer_rank_one.py
---

# Check private buffer reuse and segmentation with Lean

This direct developer API checks a complete original→normalized→reused boundary,
with optional final segmentation. It reconstructs the memory plan from three
or four raw modules, independently
decodes the complete endpoints and generates Lean obligations linking actual
source syntax, descriptors, lifetimes, initialization, addresses and final code.
It does not install compiler-wide enforcement or numerical/hardware approval.

For untiled output, pass `(original, normalized, reused)` to `prepare_chain`.
The local initializer simulation links that exact actual target to the complete
memory-entry theorem. An auxiliary epoch hint supplies no delivered compiler
stage or acceptance authority. Three-buffer checked reports use
`merlin.buffer-reuse.checked.v3` and `actual_buffers`; the four-buffer report
remains v2 with `actual4`. Both routes support fresh replay and require the same
explicit total/private/unobservable allocator model. Neither proves full
error/world equality or native allocation behavior.

The [compiler stack guide](lean_compiler_stack.md) describes the separate
`lower_model` adapter, its `buffer_reuse` selector and its explicit memory-source
LLVM route. That adapter uses this same direct checker and requires its fresh
completion before compiler products are returned. The API below remains usable
independently of compiler enforcement.

Select the scoped buffer checks shown in the compiler stack examples. Default
`all` currently refuses this explicit-memory route: after storage transforms,
DCE can remove an unused index constant in a memref module, which the existing
tensor checker cannot model; leading memory CSE is also outside its recorded
tensor boundary. An actual untiled/tiled counterexample is retained in the
integration audit. A memory-context dead-constant proof and original memory-CSE
observation are being staged; neither is installed or qualified yet. Changed
DCE is not skipped. An actually observed unchanged CSE can use the existing
zero-job, zero-credit identity contract; a missing/skip-only record cannot.

Capture the three complete modules from the selected producer, adding the
fourth only when segmentation is requested, and record its revision/configuration.
The intake checks the actual IR relationship. Producer reexecution and source
attribution are separate qualification evidence. An unchanged normalization
prefix uses the separate complete-syntax identity checker and receives zero
positive normalization credit.

```python
from pathlib import Path
from merlin.verify.buffer_reuse_intake import ALLOCATOR, prepare_chain
from merlin.verify.buffer_reuse_gate import check, deliver, replay

paths = ["original.mlir", "normalized.mlir", "reused.mlir", "segmented.mlir"]
candidate = prepare_chain(tuple(Path(p).read_text() for p in paths), allocator=ALLOCATOR)
lean = Path("/path/to/selected/lean")
result = check(candidate, lean=lean, destination="out/build/buffer-example/check-1")
if result.status != "checked":
    raise RuntimeError(result.reason)

output = Path("out/artifacts/verification/buffer-example/after.mlir")
output.parent.mkdir(parents=True, exist_ok=True)
def consume(text):
    with output.open("x") as handle:
        handle.write(text)
deliver(result.completion, candidate, consume)

rechecked = replay(result.completion.report, lean=lean,
                   destination="out/build/buffer-example/replay-1")
assert rechecked.status == "checked", rechecked.reason
```

Each destination must be new. Preparation alone has no proof authority. A check
builds the selected project closure and every generated query module into a fresh
private tree, audits the complete root/type/import roster, and revalidates all
source, raw-input, tool, query, object and process-evidence bindings. Saved JSON,
copied tokens or caller-supplied success flags cannot authorize delivery. Replay
reconstructs the original request and builds fresh objects; it is evidence rather
than a way to deliver through a saved token.

The all-entry root is `GraphMemoryAfterEntries.every_definition`. For every
supported defined entry and arguments satisfying its domain, it relates the sets
of possible successful returned/public/caller-heap/user observations. The changed
kernel uses the source-derived actual-final chain; other entries must have checked
unchanged, call-closed bodies. A caller reaching the changed kernel requires the
separate Context route and is refused by this basic API. Declarations and arbitrary
nested/cyclic caller graphs are outside its domain.

This subset uses index64 and static normalized integer buffers with checked
sequential disjoint lifetimes and complete initialization. Its actual-final
initializer bridge supports the checked rank-two rectangular uniform-loop form
with i8, i16, i32 or i64 elements. All four widths have fresh-gate qualification;
broader element/rank profiles remain separate engineering.

An additive rank-one bridge now represents the actual single SCF fill loop,
its exact source/canonical SSA/configuration link and actual optional tiled
point loops. It derives local private-slot simulations and lifts them through
the same complete-entry root. All 33 installed qualification cases pass: 12 host
and 21 SDK/native cases. Actual untiled and tiled chains cover extents 1, 3 and
65,536; both routes have fresh-process replay and public compiler delivery.
The gates audit 250 canonical roots and 2,545 STD3 theorems. Target controls
include three genuine closed Lean refusals and seven explicit host refusals;
three additional initializer-link component mutations also refuse. Independent review
checks the exact source/product joins and all 24 finite native wrap observations
across the original, ordinary optimized and checked compiler routes.

For example, two nonoverlapping `memref<3xi32>` objects initialized by a captured
scalar seed can share one slot. The current rank-one bridge requires that sole
store body. Pure computed initializer prefixes and rank-three source bridges
remain safely unsupported, although the actual storage planner recognizes
additional forms. Their extension needs a typed pure-operation meaning and SSA
execution correspondence; relaxing the source parser alone would be unsound.
The storage producer reuses identical memref types/alignment, so equal-byte
objects with different shapes do not share a slot.
Typed arguments, caller descriptors outside the private partition and the stated
scalar/literal interpretation are explicit premises. The allocator is modeled as
total, private and unobservable.

Consequently this relation does not establish finite-capacity behavior, categorized
errors or terminal failure states, arbitrary pointer escape/aliasing, asynchronous
completion or native allocator fidelity. The retained 8-to-16-byte resource
counterexample rules out an unconditional finite-memory claim. This memory
relation differs from the scalar loop checker's return/error/terminal-world
equality. Source-bound successful executions establish meaningful inhabited
examples; they do not prove primitive fidelity for every native implementation.

Python/xDSL admission/projection and host transport remain trusted boundaries.
Actual qualification, independent specification review, numerical laws and model
coverage are separate evidence. Source-file comments inherited from earlier
uncompiled candidates do not replace the dated source-bound build/audit records.
