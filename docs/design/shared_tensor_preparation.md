---
title: Shared tensor preparation
kind: design
status: current
owner: core
last_verified: 2026-10-18
related: [architecture, session_entry_projection]
code_refs: [src/merlin/xdsl_dialects/lowering/shared_preparation.py, src/merlin/llvmlower/device_shim.py, merlin/tests/ir/test_shared_preparation.py]
---

# Shared tensor preparation

Independent consumers often prepare the same immutable input separately.
`share_tensor_preparations` introduces one explicit prepared byte tensor and
rewrites compatible consumers to read it. Consumers keep their original order;
the pass creates no hidden cache and assumes no target workspace residency.

The provider supplies a `TensorPreparation` for each supported destination-only
call: input index, deterministic preparation entry, static byte-tensor type,
equivalent consumer entry, and retained original arguments. This is an explicit
semantic contract. The provider must qualify numerical equivalence, format,
layout and synchronous completion, including padding and exceptional inputs.
The generic pass does not infer those facts from symbols or provenance.

Grouping requires equivalent SSA inputs, identical consumer input types and
the same preparation entry and representation. Static unencoded expand/collapse
views preserve flat contents. Distinct closed elementwise floating extensions
also compare equal when they widen the same source SSA through identical
static types and identity indexing. Their entire scalar body must be exactly
`arith.extf` followed by its yield, without attributes, initializer reads or
captures. The key retains both source and destination types; widening never
makes a narrow tensor equivalent to a wide tensor. Transposes, slices, arbitrary
casts, dynamic extents and encoded views do not establish equivalence.
Unknown effects split groups.
Crossed external calls require read-only inputs and a private, freshly allocated
write-only destination. Live consumer results and fanout remain visible.

Each group allocates a destination and calls the preparation immediately before
its first consumer. All group consumers read the resulting tensor. Ordinary
bufferization owns its lifetime independently of any device scratch workspace.
A positive caller-supplied byte budget bounds overlapping new buffers. A group
that cannot fit retains its original execution. The pass clones and verifies
the module and refuses symbol/signature conflicts without modifying its input.

Selection establishes legality and bounded storage, not profitability. The
caller must measure preparation cost, extra allocation and dispatch overhead,
consumer work and retained transfers under an unchanged timing boundary. An
experimental selection must remain explicit until representative executions
justify broader use.

Providers can use `emit_logical_translation_unit` to bridge the resulting mixed
rank and dtype descriptors to target pointer entries. Its guards check dense
strides, static shapes, address arithmetic and output disjointness. Descriptor
zero initialization uses a C aggregate so small freestanding entries do not
require a library `memset` call merely to initialize a return descriptor.
