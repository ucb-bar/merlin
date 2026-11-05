---
title: Original reduction source geometry
kind: design
status: current
owner: targetgen
last_verified: 2026-10-10
related: [architecture, conditional_source_value_cones]
code_refs:
  - src/merlin/targetgen/original_reduction_sources.py
  - packages/merlin-experiments/src/merlin_experiments/phase0/original_call_sources.py
  - packages/merlin-experiments/src/merlin_experiments/phase0/declared_run.py
---

# Original reduction source geometry

The shared factory constructs original `mean.dim`, `softmax.int` and
`layer_norm.default` sources from complete public/native schema bindings. It
preserves the ordered tensor/result roster, scalar kinds and values, omitted
defaults, storage, ranks and positive contiguous layout premises. This bounded
vocabulary supports unchanged f32 storage. Promoted or unsupported formats,
empty shapes, malformed axes and unproved layouts stay unavailable.

Original extents witness shape relations. Fresh mean and softmax dimensions
come from explicit extent and rank. Mean retains the original axis list order,
negative axes, `keepdim` and `dtype=None`; the planner normalizes axes only for
shape calculations. `None` and empty dimension lists select all axes. Softmax
preserves its signed dimension and the complete same-shape result. No dtype
conversion is synthesized to fit a numerical policy.

Layer normalization keeps the original positive `normalized_shape` literal,
ordered optional weight/bias inputs, finite FloatLiteral epsilon and Boolean
flag. The fresh input suffix and affine shapes must equal that literal. Only
leading dimensions vary. When the literal fixes every input dimension, the
source is literal-shape evidence and makes no larger-capacity claim. The
result keeps the full input shape and original storage.

Rank is checked against complete original shape/stride metadata before axis
expansion. Signed64 dimensions/products, every logical input, optional affine
input and complete output are bounded before fresh shape or loader allocation.
The tensor-element budget charges only logical tensor elements, including one
element for a scalar result. The ordinary observer
also enforces existing per-source and complete-roster limits before writing
loaders. These budgets govern source construction; they do not select numerical
reduction work, accumulation order, exp/sqrt behavior or a reference tolerance.

Original call-source v11, automatic v18 and declared run v11 explicitly select
this factory. Converter v6 and declared converter selector v5 retain the same
scalar conversion sub-roster beside the complete new source vocabulary. Stable
factory v5 and ledger v7 keep every original call/cohort identity, including
fulfilled source prerequisites. Earlier versions preserve their meanings.

Construction grants no reference comparison, numerical stress, reviewed
operation ownership, alias/effect domain, command/resource mapping, runtime or
phase admission. Independently selected references and packing remain optional
source inputs with their existing mandatory gaps. Neither factory availability
nor finite synthetic controls reissue the historical original coverage ledger.
