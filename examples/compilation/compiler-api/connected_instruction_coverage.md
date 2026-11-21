# Connected instruction coverage and dense LLVM integration

The 2026-10-23 fresh rehearsal exposed a harness vocabulary mismatch. Connected
`owned_host_model_execution._dispatch_proof` collected raw decoder names, whereas
the ordinary core-capsule trace checker already reopened those instructions
through the explicitly selected operand protocol. A frozen semantic requirement
could therefore be reported missing even when the corresponding command existed.

`trace_check.instruction_class_coverage(trace, target=...)` now reuses that same
validated projection for connected dispatch. It returns raw classes, normalized
classes and their observed union separately. The input is the runner's structural
decode of the actual emitted module. Raw identities and decoded fields are
rechecked; configuration subtypes require actual resolved operand fields from the
selected source-bound protocol. There is no string-renaming table, implicit
protocol selection or inferred instruction subsumption. Conflicting meanings for
a label shared by the two vocabularies refuse. Unknown commands/selectors refuse.

The existing source/call/CFG, object/ELF custody, whole-ELF command prohibition,
complete numerical comparison and required execution tiers still apply. Class
coverage does not prove that an instruction ran, that a schedule is legal, or
that a complete result is correct. Per-kernel dispatch receipts retain both
vocabularies under `instruction_class_observations`.

## What this resolves, and what it does not

The retained response `resp_3_1792784256967592059_1e2b6f95d1dec599af98bd8b44ab5d60.json`
for `gemmini-author-rehearsal-20261023-v1` reports PB00 and PB01 incomplete before
numerical execution. Its candidate digest is
`2682d57c35e2a4291e6ac2c0dc199cb1214e93985450d8ccff987e9415bc84b1`.
The selected public facts contain raw `LOAD_CMD`, `STORE_CMD`, `PRELOAD_CMD` and
`COMPUTE_AND_FLIP_CMD`; the pinned operand protocol maps their exact selectors to
`MVIN`, `MVOUT`, `PRELOAD` and `COMPUTE_PRELOADED`. Configuration classification
additionally depends on emitted operands, not merely the `CONFIG_CMD` name.

The same response records no `FLUSH_CMD`. Its `FENCE` instruction is distinct from
the accelerator flush required by the frozen capsules. The fix must retain that
missing requirement. It neither inserts commands nor changes capsule policy.
The exact historical PB emitted artifacts were not retained with the final
responses; no complete historical normalized replay or new PB pass is claimed.
Re-evaluation must use an explicitly identified authored revision and ordinary
emission, linking and numerical execution. The source identity associated with a
request is not automatically its compiler-result identity.

Focused regression tests parse actual neutral LLVM command modules with opcode,
selector values and raw labels different from this target. They exercise selected
configuration discrimination, unresolved operands, unknown commands, vocabulary
collisions, unchanged traces, no-observer behavior and genuine missing flush.
These are harness-mechanism tests, not simulator or numerical qualification.

## Fresh-author integration with `DenseSourceResult.emit_llvm`

The shared LLVM bridge removes the need to manually rebuild the dense planner's
outer loops, dimension tails, reduction initialization/continuation and deferred
stores. It does not manufacture a qualified target instruction adapter from an
ISA encoding declaration. A fresh target author still supplies:

1. Selected source/readout and scheduling contracts supported by public hardware
   facts and explicit software semantics, plus a `DenseLoopAdapter` whose declared
   instruction meanings and qualification identity match those contracts.
2. A storage assignment and justified buffer-disjointness premise. Distinct tensor
   SSA names alone do not establish distinct runtime allocations.
3. Target callback lowering of each checked scheduled call using its supplied
   i64 SSA operands and logical pointer bindings. Encodings, pointer addressing,
   effects and completion remain target-owned. No additional outer loops, tail
   rounding, accumulator resets or early stores should be reconstructed there.
4. Normal function/module declarations, ABI integration, verification,
   translation and linking, followed by independent complete-output execution.

See `source_dense_llvm.py` and `source_dense_llvm.md` for the same offline/online
callable example. Its neutral native tests prove the emitted counted loops are
consumed, including tails and a structurally valid but numerically incorrect
callback. They provide no target seed or hardware performance claim. The current
bridge requires selected 64-bit indices, positive ascending loops and the existing
rectangular zero-initialized signed-integer contract. Bias initialization,
unqualified completion/effects, arbitrary asynchronous execution and mixed graph
composition require their corresponding explicit adapters rather than an assumed
successful dense lowering.
