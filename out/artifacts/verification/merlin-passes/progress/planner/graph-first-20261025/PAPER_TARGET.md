# Current paper target: resolve the two verification FIXMEs

Latest user steering on 2026-10-25 supplies the paper's verification paragraph.
The two requested deliverables are:

1. State checked transformations, modeled semantics, assumptions and the mapping
   from complete compiler IR into Lean with evidence for each part.
2. Supply an actual worked before/after transformation with retained checker
   output connecting this argument to the separate encoding/execution checks.

All three current graph assignments continue. Numerical work stays deferred.
This is a paper-evidence requirement on the current full batches, not permission
to stop implementation after writing a claim or to weaken their proof obligations.

## Worker delivery required

Each worker supplies `paper-evidence/README.md` in its current artifact root.
It must contain an exact proposed paper sentence, the actual pass/variant and
source revision, the named existing/new Lean roots, the complete relation and
domain, explicit semantic/primitive/effect/alias/completion/allocator hypotheses,
an IR-to-Lean field map, actual source/middle/result paths and digests, retained
checker output and original/fresh replay evidence, valid negative mutation,
review state and concrete remaining gaps. Identify what is only trusted Python.

The worked example must be an actual changed production result. Annotate where
each meaningful IR edit appears in the checked representation, not just a trace
count or identity hash. Bind unchanged functions, observers and error behavior
where the theorem claims them. Keep a compact explanatory IR excerpt alongside
the unabridged machine-checked buffers, so the paper example remains readable.
One complete existing qualified example may be reused unchanged while a new
stronger composition remains explicitly pending. No new native run is needed
solely to regenerate an old printed result.

The coordinator selects the clearest verified example and assembles the final
claim table. Graham can independently review its IR projection and assumptions;
he does not duplicate current implementation work or grant numerical approval.
The [candidate manuscript text and evidence map](PAPER_VERIFICATION_DRAFT.md)
uses the newly sealed scalar/loop example and keeps review and integration gaps
explicit. It is a review input, not an approved final claim.

## Claim wording boundaries

A defensible target is: for the explicitly enumerated Merlin transformations
and supported domains, validator acceptance entails the stated complete
observable preservation relation under the recorded operation/effect contracts.
This wording becomes publishable only after the rows, reviews and evidence
actually support it. Do not predeclare full graph coverage or paper acceptance.

Enumerate missing graph passes and variants. The eight registered common-pipeline
transform kinds are a concrete denominator, not all graph/scheduling mechanisms
in the repository. Broader fusion, configuration, transfers, placement and storage
families must be classified separately; unexplored mechanisms remain open.

The actual panel-pipelining/gSIM figure supplied by the user is execution evidence.
No existing scalar, fusion or buffer theorem automatically proves that exact
serial→pipelined case, asynchronous completion, encoding or measured cycles.
Do not attach Lean status to the figure unless its complete actual boundary is
separately bound to an applicable soundness theorem. No performance remeasurement
or figure revision is assigned to proof workers.

Distinguish the existing host Lean service from new independent graph leaf
validators. A standalone qualified validator does not establish that ordinary
compilation gates on its receipt. The paper must identify the actual selected
checking path; broker integration and fail-closed tests remain separate whenever
not supplied. Numerical, LLVM, backend, runtime/ABI and hardware fidelity remain
outside the structural claim unless separately established.

Historical failed SDK bindings, native crashes, counterexamples and review holds
remain visible. New evidence cannot promote an earlier failed qualification.
