---
title: Typed preprocessing correspondence
kind: reference
status: current
owner: ir
last_verified: 2026-10-06
related: [lowering_pipeline, architecture]
code_refs: [src/merlin/llvmlower/typed_preprocessing_correspondence.py, src/merlin/llvmlower/passes_xdsl.py, src/merlin/common/ir_audit.py]
---

# Typed preprocessing correspondence

`merlin.llvmlower.typed_preprocessing_correspondence.check_result_types` checks a
specific boundary: exact captured MLIR to the output of Merlin's xDSL
preprocessing. It takes both exact byte streams and the audited invocation's
`index.json`. Compact audits retain hashes and inspection views rather than the
executable IR, so callers must supply the original bytes separately.

The checker binds those bytes to the audit's `source-transform-map.json` receipt
and final `xdsl-c-interface` stage by SHA-256. It reparses both modules, requires
one defined single-block function, checks complete ownership of top-level
preprocessed operations, and compares every mapped source result's MLIR type to
the corresponding preprocessed result. Tensor dimensions and element types are
part of that type equality. Any missing, reordered, or mismatched mapping raises.
The preprocessing pass distinguishes function definitions from declarations with
explicit attribute dictionaries when attaching the C interface; an attribute
dictionary is not treated as a function body. A generically printed module is
handled through typed IR: only a public function definition gains the interface,
and private or external callbacks keep their original ABI. Only a defined
single-block function can supply this correspondence map.

The returned `claim` is `preprocessing_result_types_only`. Source operands are
observed in the source IR, but this checker does not establish their runtime
values. It does not compare operation outputs, check nested operations, map
results through later MLIR/LLVM passes, execute the lowered program, or certify
accelerator behavior. A final-output comparison against a saved golden cannot
establish every intermediate operation's value. This check therefore does not
discharge Phase 0 `support_lowering` obligations; those require a selected
compiler artifact with per-operation typed shape and value preservation evidence
through the relevant lowering and execution boundary.
