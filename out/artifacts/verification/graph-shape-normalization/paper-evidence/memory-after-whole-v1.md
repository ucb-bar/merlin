# Actual-final memory graph candidate — not yet a paper verification result

Current status: **UNCOMPILED / UNQUALIFIED**. The source-only implementation and
worked-example obligations exist; the runtime hold prevents any claim that these
new roots have passed Lean. Independent assumption/projection review is pending.

Proposed claim **only after successful exact-root checking and review**:

> For the checked static rectangular example and its admitted kernel entry,
> the composition of identical-view/static-dimension normalization, private
> buffer reuse and ordered initializer segmentation preserves possible successful
> observations: scalar/index returns, the complete caller heap and user state.
> Both independently projected boundary programs use the same operation
> interpretation. Allocation is modeled as total, private and unobservable;
> native descriptor/primitive fidelity and Python projection remain assumptions.

The claim does not cover every exported entry, all shared pass inputs, numerical
correctness, divergence, categorized failures, peak allocation, LLVM lowering or
hardware execution. `untouched` and the actual `touch` body are fully serialized;
the complete chain's selected entry remains `kernel.function.name`.

The exact implementation/evidence map is
`../after-binding-deadline-v1/final-memory-after-v1/whole-source-review-v1/README.md`.
`THEOREM-MAP.json` there records existing anchors and new candidate roots.
`source-whole-v1/QUERIES.json` records every exact raw boundary, query and root.

The worked example uses the pinned 4499 producer's retained `two_epochs` files:

- Before: `epoch-cases-v1/two_epochs/source.mlir`, two `2×3xi32` allocations,
  uniform initialization, ordered scalar updates and two actual observer calls.
- After: `epoch-cases-v1/two_epochs/memory/after.mlir`, one retained allocation,
  both logical epochs mapped to that slot, tile-2 outer loops, actual index add/min
  and dynamic point-loop bounds, one final free, the same observer order/returns.
- Positive query: the `positive` entry in `source-whole-v1/QUERIES.json`.
  It independently decodes raw source and final; finite goals bind the source
  plan, slot/call/return interfaces, conservative index extension and exact final.
- Negative query: the `buffer` entry inserts a separate same-type allocation and
  reads that buffer. The modified program passes xDSL verification; its complete
  independently decoded term differs. **Lean refusal is UNOBSERVED.**

Retained checker output currently consists of SDK-free preparation reports:
8 boundary tests and10 full-chain preparation tests passed. The Lean checker
has not run these queries. This supplies concrete paper example material and an
auditable mapping, but does **not** yet discharge the paper's requested retained
formal-checker acceptance output. Do not substitute these pytest results for it.

The candidate chain is `Merlin.GraphLoopsMemorySCF.After.checked_actual_chain`;
the generated example root is `GraphMemoryAfterWhole.complete_actual_chain`.
The remaining formal gate is compilation, exact-root type/std3 audit, observed
positive/negative checks, new frozen qualification and one replay in a newly
authorized unaffected runtime context. Original failures and all allocator,
native, model and review limits remain visible.
