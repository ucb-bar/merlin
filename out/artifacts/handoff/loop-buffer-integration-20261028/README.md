# Loop and private-buffer integration with DCE/CSE

The reviewed loop/buffer checkpoint from `feat/pass-verification` at d45415489
is merged into `feat/pass-verification-dce-cse`. Target base: 8215c0015;
merge: d3ffd69f1; compatibility fix: ca62a42b7. Original commit ancestry is
preserved. The primary checkout and three ongoing configuration workers were
not switched or rebased.

Only the generated docs index conflicted. Regenerating it with this branch's
generator preserves both sets of guides. Agustin's newer process runner uses
`common/owned_children.py`; the compatibility fix adds that actual dependency
to loop and buffer receipt fingerprints. Two regressions show that a changed
cleanup-helper source invalidates either request. This changes host binding,
not the Lean semantics, theorem statements, numerical policy or compiler gate
selection.

## Fresh combined-tree validation

- 147 focused loop/buffer/compiler-selector tests passed, plus 33 process
  infrastructure tests. Scoped lint passed.
- The actual loop paper example passed a fresh Lean check, a live exact-byte
  consumer, and a separate-Python fresh replay. Each check audited 44 roots and
  927 transitive theorems, standard three axioms only, zero violations.
- The actual baseline buffer chain passed a fresh Lean check, live exact-byte
  consumer, and separate-Python fresh replay with the new 192-source binding.
  Each audited 250 roots and 2,338 transitive theorems, standard three axioms
  only, zero violations.
- All five buffer cases reconstruct exactly the earlier query/part/root bytes.
  This is not a new five-case native qualification. The historical full F2
  campaign used 186 source fingerprints; its 191-source correction and this
  merged 192-source baseline/replay remain separate evidence.
- The structure guard still reports 16 problems. Every offending source is
  byte-identical to the target branch before this merge: two optional-package
  import violations and 14 module-size violations. No guard was weakened.

See [the exact integration result](../../verification/merlin-passes/dce-cse-loop-buffer-integration-20261028/RESULT.json),
its source/text manifest and four original native reports. SDKs and compiled
objects are excluded. Original absolute process paths are retained for attribution;
copied reports do not grant live authority. Overlapping theorem totals must not
be added.

## Scope available to callers

The existing opt-in compiler selector remains **DCE and CSE only**. This merge
does not register numerical policies or new loop/buffer compiler gates.
The scoped loop and buffer checks are direct developer APIs:

- [Loop gate](../../../../docs/guides/lean_loop_transform_gate.md): static
  index64 counted-loop unrolling followed by ordered one-dimensional
  segmentation, including a short final segment. Explicit common primitive and
  external-call premises; modeled return/error/terminal-world equality.
- [Buffer gate](../../../../docs/guides/lean_buffer_reuse_gate.md): the actual
  original→normalized→reused→segmented chain, with checked private lifetimes and
  initialization. Equality of possible successful public/caller observations
  under total, private, unobservable allocation.

These relations differ. General tiling, unroll tails, the newer carried
completion-token scheduling loops, asynchronous/fence behavior, finite memory,
arbitrary escaping aliases, native fidelity, whole-model coverage and independent
human approval are not established by this merge. The allocator and numerical
counterexamples remain valid. The separate Context candidate is not integrated.

Configuration-write elimination is ongoing in the other worker checkout and
is not part of this integration. Its interim receipts and fixes must not be
treated as approved compiler enforcement.
