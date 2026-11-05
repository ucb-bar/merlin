# Complete public source feedback budget

The optional `feedback_budget_protocol="merlin.public_full_source_feedback_budget.v1"`
selects a maximum of 2,400 seconds for a complete selected compile/Spike feedback
pair. Pass it to `CapsuleSourceDevelopment` or directly to
`prepare_public_source_diagnostic`. The default maximum remains 900 seconds.
The live development scope and diagnostic policy bind the selected protocol and
limit. Changing either after issuance refuses revalidation.

A queued host may independently select
`SearchSettings(queued_job_timeout_seconds=2400, ...)`. The broker acknowledgment
budget remains separate. The queue must admit the whole job before its absolute
completion boundary and final reserve. This protocol increases no CPU, memory,
file, artifact or trace limit. The existing V3 trace observation still has its
600-second leaf ceiling, and every source, output, effect and timing-domain check
retains its original obligation.

The longer budget accommodates complete full-size source observations; it grants
no formal cohort admission, hardware qualification, timing certainty or promotion.
Observed control costs are evidence for planning, not a completion guarantee.
