---
title: Prepared private endpoint DAG
kind: reference
status: current
owner: compiler
last_verified: 2026-11-09
related: []
code_refs:
  - src/merlin/llvmlower/prepared_endpoint_dag.py
  - src/merlin/llvmlower/endpoint_narrowing.py
---

# Prepared private endpoint DAG

`prepared_endpoint_dag.c_header` is an explicit, default-off producer/consumer
mechanism. It changes no numerical permission. It operates on finite ordered
intervals supplied by the existing policy, and makes no claim that a heuristic
interval is an exact enclosure of the original computation.

The producer copies complete `[part,row,column]` spans in ordinal order. In the
same pass it validates finite endpoints, candidate membership and derives a
maximum absolute endpoint word for each row. Input/output spans, metadata and
caller scratch are disjoint private owners. The producer context binds their
shape and a lexical epoch; partial failure never publishes a public result.

Before evaluating a row, admission follows the same nonnegative scale/add
sequence on its magnitude bound. RNE monotonicity and sign symmetry prove that
this recurrence bounds every signed intermediate and candidate. Every bound
prefix must be finite. The reciprocal interval is checked, and the final bound
must not exceed the largest finite BF16 value. Only then may the same ordered
source operations run without repeated interval constructors. The source
midpoint reciprocal, sign-selected products, BF16 conversion and final clamp
are unchanged. Unsupported range, epoch or environment retains checked code.

Any refinement invalidates that row **before** writing a partial or denominator.
The prepared path then refuses it for the remainder of the epoch; it does not
reuse a certificate across mutation. The original checked row path recomputes
its outputs. Source fallback and original final accuracy gates remain active.
No previously certified row may be mutated. No external alias may change a
producer span or metadata, and no asynchronous use may outlive the epoch.

Preparation/maxima, metadata initialization and stack lifetime are real costs.
The mechanism does not imply a hardware speedup. Quantizer validation remains
unchanged in the first source-bound experiment; it is not bypassed using an
untyped assertion about arbitrary arrays.

## Checked narrowing and dependent observations

`endpoint_narrowing.c_header` composes the unchanged original header with a
separate, explicitly contracted owner. The original prepared owner still
invalidates a row before refinement. The new owner inherits its initial finite
magnitude fact and proves that authorized updates preserve that fact. It does
not change the supplied interval policy or establish source membership: the
caller must retain its original source computation and membership check before
requesting a refinement. An approximate input policy remains approximate.

The additional `EndpointNarrowingContract` requires a private source epoch,
exclusive mutation through the new API, complete original part production,
immutable owner descriptors and original magnitude metadata, disjoint owned
storage, pure scalar returned-value observations without observed interposition
or errno, preserved source operation order and numerical policy,
closed row/column dependencies, stable RNE and gradual underflow, unobserved
nontrapping effects, retained fallback, and publication only after complete
success. These are caller proofs; the emitter does not infer them from a symbol,
shape, model identity or previously measured values.

Initialization checks aligned nonoverlapping spans and storage capacities,
snapshots factor and denominator words, and records source span identity and
dimensions. A partial update must be a finite ordered subset of its current
interval. Its unchanged center must stay in a nonpoint interval; a point uses
the new lower endpoint in the original evaluator. Signed-zero membership is
checked by words at zero boundaries. The original magnitude cap therefore
continues to bound every endpoint and every used center. A denominator update
similarly requires a finite nonnegative subset. Current reciprocal, magnitude
recurrence and finite BF16 output range remain checked before evaluation.

A partial update invalidates one dependent column. A denominator update
invalidates the complete row. Per-column generations and row pending counts
allow unchanged observations to reuse their exact previous words. The evaluator
retains the original scale/add order, midpoint reciprocal, sign-selected final
products, BF16 conversion and clamp. Cross-row or cross-head consumer
observations are not cached by this capability and must be reevaluated.

Replacement, widening, a displaced nonpoint center, nonfinite values, changed
factor/denominator context, a foreign source epoch, unknown writes, unsupported
rounding or version exhaustion refuses the new path. Unknown writes must first
call its explicit invalidation API; no source or metadata alias may bypass the
exclusive mutation contract. A changed global source descriptor invalidates
the owner. The original checked computation remains the fallback. Zero BF16
boundaries or candidates also refuse, because the contract grants no stronger
signed-zero result theorem for C min/max in a different compiled context.

Private cache words may be partially written before numerical refusal and must
then be discarded. They are never public results. Metadata, snapshots, initial
evaluation, subset tests, replay, fallback and output publication are real costs;
dependency reuse alone proves no whole-model or device-cycle improvement.

Both Python modules emit portable private C helpers.
`prepared_endpoint_dag.c_header` requires `PreparedEndpointContract`; the
additional `endpoint_narrowing.c_header` requires both that contract and
`EndpointNarrowingContract`. This is an explicit provider assembly seam, not an
automatic pass that discovers or grants those physical effect proofs.
