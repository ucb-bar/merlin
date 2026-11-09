---
title: Derived tensor compiler inputs
kind: reference
status: current
owner: compiler
last_verified: 2026-11-09
related: [architecture]
code_refs:
  - src/merlin/frontends/compile_inputs.py
  - src/merlin/frontends/derived_compile_inputs.py
  - src/merlin/frontends/kernel_compile_inputs.py
  - src/merlin/frontends/kernel_shape_abi.py
  - src/merlin/xdsl_dialects/standard_tensor.py
  - src/merlin/xdsl_dialects/standard_bufferization.py
  - src/merlin/xdsl_dialects/standard_memref.py
  - merlin/tests/ir/test_derived_compile_inputs.py
  - merlin/tests/ir/test_kernel_compile_inputs.py
  - merlin/tests/ir/test_standard_bufferization.py
  - merlin/tests/ir/test_standard_memref.py
---

# Derived tensor compiler inputs

`stage_compile_inputs` accepts the existing `m2m.capture-receipt.v1` path and an
explicit `t2m.bound-model-receipt.v1` derived-model path. It publishes a
`merlin.compile-inputs.v1` directory with `source_profile: derived-tensor-model`.
The latter retains a static typed `forward` program, weights and argument
manifest, signature, original parent source/FX trace, TTIR implementations/specs,
and `merlin.derived-source-correspondence.v1` derivation evidence.
Public tensor inputs carry `bufferization.writable = false`: they are borrowed
functional values, including parameters, and must survive repeated invocations.
This prevents function-boundary bufferization's default writable-argument policy
from silently turning destination reuse into caller-visible mutation.

The original FX trace remains under `source/parent/`; it is not relabeled as
correspondence for the transformed model. A compiler-only projection of the
parent receipt binds selected source members. Evaluator arrays, goldens,
execution reports and their digests are absent from the compiler view. All
selected files are checked against the producer receipt and their parent/kernel
relations before an atomic publication. Unknown dependencies, unresolved calls,
unregistered operations and malformed lineage refuse without publishing a stage.

Standard parallel tensor operations have explicit xDSL bindings, including
multi-axis/dynamic-bound syntax. Standard buffer materialization retains compatible
shapes/element types independently of memref layout and checks its result and
writable/restrict attributes against native MLIR's rules. These bindings grant
no aliasing, ownership, numerical or target-execution proof.

The current derived
staging profile requires static tensor entry arguments/results. Runtime scalar/
dynamic signature admission and explicit effectful dependencies need separate
profiles. Staging establishes byte custody and declared lineage; it does not
establish complete source closure, semantic equivalence or compiled correctness.

Standalone `t2m.capture-receipt.v2` packages use the separate
`standalone-tensor-kernel` source profile. This admits a static functional tensor
ABI with read, fresh-write and readwrite buffers. Readwrite buffers retain their
incoming value and return an updated tensor; this profile does not mutate caller
storage. Argument/result types and ordering must agree with the declared spec,
receipt counts and input manifest. The parameter container must be empty.

The stager changes the single defined entry symbol to `forward` for the current
native ABI and records both names in `merlin.kernel-source-correspondence.v1`.
It retains exact original MLIR/TTIR/spec bytes and projects compiler provenance,
including distinct original and normalized operation inventories. Numeric
evaluation reports and reference data/digests are excluded from that projection.
No FX trace or Torch capture identity is manufactured. The standalone and derived
paths share compiler-only custody and borrowed-immutable input semantics; both
still require separate consumer lowering and execution evidence.

The standalone ABI v3 extension additionally admits explicitly typed by-value
runtime scalars at declared original TTIR positions. Their public arguments
follow the logical tensor inputs. The signature labels them `kind: scalar`,
with empty storage shape and an exact scalar dtype; the MLIR input remains a
plain scalar. These inputs receive no tensor bufferization attributes.
Invocation values are forbidden in scalar declarations and remain outside
compiler staging. Actual host execution uses the existing `ScalarArg` ABI;
neither staging nor this profile supplies dynamic shape/stride/grid admission.

ABI v4 additionally retains declared inclusive integer scalar bounds in each
signature entry. The interval must fit the signed scalar type; invocation values
remain excluded. This permits a consumer to validate values separately from the
frontend's executable entry assertions. Bound declarations do not grant a memory
ownership/initialization proof or certify the guard/payload semantics by staging
alone. Actual tensor dimensions remain static in this profile.

The explicit `shared-functional-buffer` payload profile retains the same
functional tensor/scalar boundary with standard memref/bufferization operations
inside it. Staging checks the additional memory witness, continuation input,
output and report against provenance and copies them into `source/` (or each
derived kernel's source directory). The projected receipt retains the payload
profile; evaluator values remain excluded. This custody check does not validate
the producer's memory proof or broaden source admission.

Standard `bufferization.to_tensor` accepts compatible strided memrefs with the
same shape/element checks as native MLIR. Standard `memref.reinterpret_cast`
accepts ranked or unranked sources while checking element type, memory space,
size/stride/offset and dynamic operand segments. These bindings preserve the
upstream raw-buffer representation without changing loop-carried SSA types.

Standalone ABI v5 additionally admits named rank-one runtime dimensions and
logical ceil-divided grid expressions over bounded nonnegative integer scalars.
Its `merlin.compile-signature.v2` retains symbolic shapes and named scalar
bindings for input checks and result allocation. Actual input/result dimensions
are dynamic; maximum bounds are declarations for proof, not captured sizes.
Legacy signatures remain v1. The compiler view receives no evaluator arrays or
example shape values. This declared-contract check does not establish that the
program enforces its guards, covers its outputs or has qualified numerics;
consumer lowering and repeated independent-shape execution remain separate.
