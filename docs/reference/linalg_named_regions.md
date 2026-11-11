---
title: Native implicit named Linalg regions
kind: reference
status: current
owner: frontends
last_verified: 2026-11-11
related: [lowering_pipeline, architecture]
code_refs: [src/merlin/frontends/linalg_named_regions.py, src/merlin/frontends/linalg_mlir.py, src/merlin/xdsl_dialects/standard_linalg.py]
---

# Native implicit named Linalg regions

The standard Linalg reader retains the named operation and its complete typed
scalar region. MLIR custom syntax omits that implicit region. xDSL 0.68's
`MatmulOp.get_hidden_region` rebuilds mixed operands without the casts to the
accumulator type required by native `linalg.matmul`: BF16 operands with an F32
accumulator produce an invalid BF16/F32 addition. This previously stopped
ordinary whole-model preparation after native printing, although the original
generic source and its native execution were valid.

`restore_named_regions` checks parsed named regions. When their Python
representation fails verification, the selected native MLIR interpreter parses
the exact original source and prints generic syntax with the real implicit
regions. The complete re-read module must verify. This reuses
`llvmlower.generic_form.to_generic_form` with a thirty-second bound and temporary
scratch under the project build directory. Missing tools and malformed original
regions fail; no invalid body is replaced by guessed arithmetic.

The operation remains `linalg.matmul`; indexing maps, operand/output types,
attributes and provenance remain present. No generalization, layout choice,
arithmetic precision change, lifetime permission or target capability follows.
Valid explicit generic regions and ordinary same-type custom operations use the
existing direct reader without native reprinting.

The standard `bufferization.to_buffer` custom binding prints its attribute
dictionary before the type colon, matching BufferizationOps.td. Its previous
after-colon position produced text that only the partial Python reader accepted
when provenance attributes were present. Native round-trip tests retain both
readonly/layout fields and those attributes. Regenerate such historical malformed
prepared text from the original generic source.

`standard_linalg` adds the native optional `cast` property and
`#linalg.type_fn<cast_signed|cast_unsigned>` enum missing from the Python schema.
The extended schema retains the original operand, result, region and map
constraints. Native implicit casts remain authoritative; the property binding
does not implement a separate numeric conversion. Unknown properties, invalid
cast types and malformed regions still refuse.

Tests cover BF16/F16 widening, F32/F64 mixing and narrowing, signed/unsigned
integer extension and integer/floating conversion. They check the actual native
scalar operations, retained named identity/provenance and generic round trips.
An explicitly ill-typed generic region must fail native validation. Target
execution and numerical qualifications remain separate from these reader checks.
