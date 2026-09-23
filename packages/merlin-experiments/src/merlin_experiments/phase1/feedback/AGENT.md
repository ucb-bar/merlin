# `packages/merlin-experiments/src/merlin_experiments/phase1/feedback`

Host-only grading, redaction, certificate promotion and simulator dispatch live here.
Inputs are explicit invocation context and corpus/contract paths, never native imports.
Keep actual adapter construction, observation order, missing-oracle behavior and
enqueue-time certificate attribution unchanged. Public clients remain separate.
Dispatch owns only the closed async simulator policy shared by broker and promotion.
Source moves invalidate new implementation identities, never rewrite old evidence.

`rtlchecks.py` owns strict authored treatment selection, redaction and advisory
round/checkpoint RTL feedback. Its factory captures the explicit invocation context;
callbacks receive the actual graded public roots, including frozen promoted roots at
checkpoint. Never rediscover a live canonical corpus or select Chipyard at import.
Structural failures/errors do not change numerical pass/fail. The feedback subtree
already carries grader-only access identity; core RTL policy owners are explicit
members of the existing implementation source inventory.

`loop_grading.py` owns the authoring snapshot, language gate, numerical-grade call,
stage ledger, shape gate, promotion and verdict publication in their existing order.
GradingInputs keeps invocation roots separate; only the native diagnostic edge may
defer public-root materialization until after the submission gate. Certification
policy similarly resolves public roots only after finding additional cert tiers.
Keep archive-before-shape and publish-after-promotion ordering unchanged.

`brief.py` constructs and publishes the round brief from already-redacted round
verdicts, the candidate's notes and explicit operator errata. Preserve prompt bytes,
notes-staleness stamps, recognized resume prefixes and pre-launch refresh timing.
Refresh must not advance the notes stamp. This owner does not grade, inspect hidden
corpora or independently sanitize its trusted redacted inputs. Only its generated
brief is candidate-visible; the implementation remains in the private feedback tree.

`lifecycle.py` owns client staging, broker child lifetimes, the background-grading
cadence/single-flight handoff and durable channel-health accounting. BrokerConfig
supplies invocation context, resolved treatment tools, explicit timing storage and
distinct public/policy/schema roots; GradeCadence supplies only scheduler settings.
Grade callbacks are mandatory and contain the caller's existing numerical policy.
Background grading surrounds the agent launch; brokers live inside that launch.
Stop brokers first, then join the grader without a timeout before the authoritative
grade. Partial startup unwinds already-created children and closes parent log
handles. Shutdown attempts all owned processes despite signal/wait/kill failures;
after the 15-second graceful wait, forced stops use a bounded reap. Cleanup errors
are aggregated with unreaped PIDs. A propagating provider or startup exception
remains primary, with cleanup notes; otherwise cleanup failure raises. This owns
direct broker processes, not arbitrary detached descendants, and does not qualify
the full installed engine or unrelated orchestration cancellation.

`formal.py` owns public grade, iteration evidence, freeze/re-hash, hidden grade and
formal completion. `freeze.py` owns the existing freeze.json record using the shared
tree-hash and explicit repository identity. Preserve mandatory L3 simulator selection
even for no-oracle diagnostics. Only synthetic external execution is substituted in
installed lifecycle tests; those tier records are not hardware qualification.
