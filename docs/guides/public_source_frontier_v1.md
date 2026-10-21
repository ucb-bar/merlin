# Public source frontier for fast diagnostics

`public_source_frontier.publish(..., protocol="merlin.public_source_frontier.v1",
operator_exclusion=...)` explicitly publishes a complete source-only diagnostic.
It uses the original staged source writer, sealed execution attestation, exact
input/output references, independent integer source arithmetic, private exclusion
owner and all ordinary non-target finishing checks. Its binding must select
`target="source_only"`. The ordinary target-qualified writer remains unchanged.

The capsule carries `public_source_frontier`, with its selected source arithmetic
proof and `target_effects="PENDING"`. It carries no `model_qualification` or target
partial-sum qualification. Reopening the public capsule verifies every original
source/reference byte, the original sealed attestation and independently
recomputes its typed source arithmetic. A pending record cannot be relabelled as
qualified target effects.

The existing frozen public diagnostic corpus can copy these publications and
request stronger execution tiers. The fast numerical/counts selector must still
check each newly compiled candidate's complete outputs, source/emission binding,
ISA and role audit. Full source-effect auditing remains mandatory before
acceptance. Publication grants no formal cohort, timing or promotion authority.
