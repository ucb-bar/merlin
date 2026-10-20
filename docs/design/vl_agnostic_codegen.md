---
title: Scalable vector code generation
kind: design
status: current
owner: compiler
last_verified: 2026-10-12
related: [normal_search_feedback, compiler_plane, optional_passes]
code_refs: [src/merlin/kernels/microkernel.py, src/merlin/mining/from_strategy.py, src/merlin/llvmlower/impr_features.py, src/merlin/kernels/decode/rvv.py]
---

# Scalable vector code generation

A fixed vector type and a scalable vector type make different deployment
assumptions. Fixed types use the selected target's declared minimum vector
length; scalable types size work from the runtime vector length. Neither choice
can assume an undeclared physical width.

`MicrokernelSpec.vl_strategy` records the choice. The selected strategy producer
must lower it into actual loop/vector IR; a changed strategy field alone does
not establish changed code or broader portability.

## Preserve tails and resource legality

A scalable tile requires a legal main loop and complete remainder handling.
Mask removal or `assume_dynamic_dims_match_vec_sizes` requires evidence that
**every** asserted tile extent matches, including any fixed row tile. Peeling
one axis cannot justify another axis's divisibility. Keep unsupported shapes
explicitly refused until their remainder has a legal implementation.

Selected recipes can use a main scalable tile with a separate remainder. Other
recipes may require masked transfers. Support depends on the pinned native
lowering pipeline; an unresolved scalable operation must refuse terminal
lowering rather than disappear or use an unrelated instruction implementation.
Inspect the emitted vector-length setup, register pressure and spills under the
actual compiler and deployment ISA.

## Scope validation

Verify complete outputs for divisible and partial dimensions, and compile
larger shapes under the same resource bounds. Record legal combinations with
other transformations; unsupported compositions remain refused. A successful
finite case does not establish all vector lengths, precisions or layouts.

Measure comparable complete costs before preferring scalable over fixed code.
Hold source, numerical policy and timer boundary constant, preserve unchanged
executable controls, and report the selected hardware/configuration. Historical
board winners are not strategy defaults for a new compiler.
