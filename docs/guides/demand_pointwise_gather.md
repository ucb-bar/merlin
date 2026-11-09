---
title: Demand-driven pointwise tensor gathers
kind: guide
status: current
owner: compiler
last_verified: 2026-11-09
related:
  - target_resolution
code_refs:
  - src/merlin/llvmlower/demand_pointwise_gather.py
  - src/merlin/llvmlower/prepared_model_transform.py
---

`demand_pointwise_gather.transform_file(source, output, effects=None)` is an
explicit source transform. With no effects contract it returns immediately,
without reading files or discovering tools. It changes no normal compiler
selection and does not quantize weights or select a numeric recipe.

A caller may supply `DemandGatherEffects` only when inputs remain immutable,
floating arithmetic is nontrapping and floating exception flags are unobserved.
These obligations matter because the transform omits evaluations at elements
that the source gather never reads, and repeated indices may repeat arithmetic.
The transform changes neither source arithmetic order nor its rounding/type
properties. Flagged arithmetic and unknown contracts refuse.

The selected path verifies dense static tensor producers with complete identity
output maps and parallel iterators. Input maps must be exact projections or
broadcasts from a proved singleton dimension. Output initialization must be
unused. Scalar bodies contain only the admitted closed standard casts,
constant/add/subtract/multiply operations and yield. It retains their original
operations, properties and typed provenance, and rejects unknown effects,
external captures, partial writes and ambiguous broadcasts. Every producer
result user must belong to the closed producer DAG or the selected extract;
other users cause refusal and remain untouched. No guard or unknown interposed
operation may be crossed.

At the existing `tensor.extract`, the transform projects the **original**
coordinates through those proved maps, loads the original tensor inputs and
clones the original scalar arithmetic. It retains source provenance dictionaries,
indexing properties and scalar/yield attributes. The compiler checks an explicit
expansion budget before constructing scalar operations. It does not use token
values, model names, FQNs or provenance IDs to choose arithmetic.

The original tensor-extract in-bounds precondition remains unchanged. This
transform does not add runtime bounds checks, accept out-of-range indices,
renumber token IDs, specialize a table to observed inputs, or remove persistent
weight rows. Public source/API bounds behavior and target checks must be
qualified separately. The existing analysis-only `gather_specialization` is
unchanged.

The file API first obtains generic-form MLIR from the selected upstream tool,
preserving custom-form attributes that another parser could otherwise discard.
It records source/output/tool hashes and accepted/refused observations. A caller
can use the existing prepared-model transform seam:

```python
from merlin.llvmlower.demand_pointwise_gather import (
    DemandGatherEffects,
    transform_file,
)

effects = DemandGatherEffects(
    immutable_inputs=True,
    nontrapping_floating_point=True,
    floating_flags_unobserved=True,
)

def selected_transform(source, work):
    return transform_file(source, work / "demand-gather", effects)
```

Tensor semantics grant no physical buffer alias or lifetime permission. The
caller must retain readonly input storage and valid disjoint output storage for
the final compiled ABI. Inspect the actual lowered allocations, object and
complete outputs independently: removed logical producers do not by themselves
prove a whole-model memory fit, allocator peak, bandwidth or performance gain.
