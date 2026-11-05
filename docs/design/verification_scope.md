---
title: Shared-pass verification scope and upstream trust boundary
kind: design
status: current
owner: verification
last_verified: 2026-11-04
related: [merlin_lean_verification, lean_compiler_stack, lean_bounded_pass_profiles, shared_compiler_passes]
code_refs:
  - src/merlin/verify/pass_validation.py
  - src/merlin/verify/compiler_memory_origin.py
  - src/merlin/verify/lean/plans.py
  - src/merlin/llvmlower/pipeline.py
  - merlin/lean/MerlinVerify/Check/SharedNormalize.lean
  - src/merlin/verify/lean/Merlin/HostGraphFusion.lean
---

# Shared-pass verification scope

Merlin verifies transformations that it owns and exposes to compiler authors.
Upstream MLIR and LLVM passes are trusted toolchain components. Calling an
upstream pass from Merlin does not make proving that pass a Merlin obligation.
The current registered compiler gates are listed in the
[stack guide](../guides/lean_compiler_stack.md); `all` selects those nine gates.

## Ownership determines what stays

| Component | Decision | Reason |
| --- | --- | --- |
| Tensor DCE, Q/DQ, MACC, ordered loop transforms, private buffer reuse, configuration elision/hoisting and host graph fusion | Keep | Actual Merlin producers/consumers have scoped proof obligations and delivery gates |
| Shared transformation broker, source projections, owned workers, replay, theorem/axiom auditing and resource/effect models | Keep | They bind Merlin proofs to the actual programs and explicitly selected contracts |
| MLIR operation semantics, typed SSA correspondence and exact normalization lemmas used by shared checks | Keep | Merlin transforms MLIR; proving its transformations needs a meaning for their input/output operations |
| Native differential evaluation of the semantic model | Keep | Tests the fidelity of the semantics used in Merlin proofs; it is finite evidence, not a proof of an upstream optimizer |
| Native-prefix gate for `canonicalize`, `cse`, `linalg-fuse-elementwise-ops` and `linalg-generalize-named-ops` | Remove | Every producer in that roster is an upstream MLIR pass |
| Leading-native-CSE gate and its PassManager interception/capture protocol | Remove | The checked producer is upstream CSE; the wrapper supplies no Merlin optimization |
| Bounded upstream elementwise-fusion matcher | Remove | Its only production registration served the upstream fusion comparison; shared tensor semantics do not need that matcher |
| Ordinary upstream compiler pipelines | Keep | They still compile programs; removing their verification does not remove the passes |

The two proof source trees have different consumers. `merlin/lean/MerlinVerify`
supplies tensor semantics, source support and typed refinement. The packaged
`src/merlin/verify/lean/Merlin` tree supplies shared graph, loop, memory,
configuration and resource proofs. Their folder names alone do not determine
whether a file is removable.

For example, `Rules/ParallelFusion.lean` and `Rules/ParallelFusionScalar.lean`
remain dependencies of the shared host-graph bridge and W8A8 reasoning. The
removed `Check/ParallelFusion*.lean` files were the upstream-specific matcher.
Likewise, common-expression and named-operation lemmas can support the typed
checker without granting any native CSE/generalization pass a compiler gate.

## Compiler behavior

`--validate-passes all` checks the registered Merlin boundaries. An explicit
`cse` selector is now rejected, like any unregistered selector. The Python
`verify_native_prefix` and internal `shared_pass_native` options were removed.
The shared-preprocessing and preparation-chain APIs remain available.

Native memory compilation still binds its actual source and runner to the
original shared-pass owner. It no longer splits upstream canonicalization/CSE,
records native CSE pairs, or requires an identity CSE observation. Those
upstream operations receive no proof credit and no Merlin boundary entry.

Existing handling of Merlin source-profile exclusions is unchanged: excluded
sources may compile with explicit unverified coverage; a submitted supported
obligation that fails blocks delivery. Numerical-policy approval and hardware
qualification remain separate from proof acceptance.

## Changes and historical evidence

The original submission design included an upstream C11 workstream. That scope
is retired. Older receipts and dated handoffs describe their original source
cuts; their gate counts do not describe the current registry. Do not rewrite
those historical results or treat them as qualification of a later checkout.

Before removing another proof component, identify its producer and all callers,
including generated Lean imports and packaged source lists. Remove only an
exclusive out-of-scope closure. Keep any shared dependency, or split the helper
from the unwanted adapter first. Delete only the corresponding retired theorem
locks, preserve all remaining semantic/statement hashes, and run the full Lean
build/guard plus actual shared-pass acceptance and mutation tests.

Future Atlas work should extend Merlin's shared semantic/effect/resource
contracts where needed. Atlas instruction selection, encoding, machine timing
and target-specific schedules stay in its OOT compiler. Upstream MLIR or LLVM
verification is not a prerequisite for that work.
