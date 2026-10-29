# Scoped partial configuration hoist — completed and workers stopped

Source commit dbaad5090, primary integration 0415d140b on feat/pass-verification.
All three existing workers completed handoffs and stopped under the user's
explicit finish-hoist-and-stop instruction. No next pass is queued.

The actual e0484f488 `hoist_loop_configuration` producer is validated for one
positive static loop, one invariant pure constant setter and a nonempty flat
ordered body that preserves the setup fields. Kernel headers/descriptors/args,
all retained calls and complete loop metadata are bound to actual source/final
IR. The initial subset rejects completion-token fields, asynchronous/staging/unit
annotations, nested/zero-trip/multi-prefix/IV-dependent forms and all overlapping
body writes, including identical writes supported by other producer variants.

`ActualConfigurationHoist.complete_kernel` proves equal complete modeled
success/error/terminal states, with partial surrounding-context and observer
corollaries. It requires one common indexed call interpretation, exact total pure
setup semantics and declared successful body frames, with a nonaliasing field
mapping and all relevant caller/return/memory/observation data represented.
Fixedness and frame implications are proved, rather than assuming source success
or whole-run equality. Python/parser/import/transport, ISA/native fidelity,
independent human policy approval, actual-model coverage and adjacent-pass
composition/enforcement remain explicit limits. Full-pass/all-graph verification
is not claimed.

## Qualification

- Seven actual changed pairs passed fresh gates and exact-byte consumers; nine
  unsupported boundaries were refused.
- Fifteen compiler-structurally valid mutations were refused by the actual
  `accepted` Lean obligation; one valid completion-token mutation was refused by
  host scope admission. These are different kinds of refusal.
- Sixteen forged-request and twelve live-artifact controls passed, followed by
  a separate-Python fresh replay with a new empty live-authority ledger.
- Each supported query covered fifteen exact roots and 315 transitive theorems,
  standard three axioms only, zero violations. Audit totals overlap.
- Independent F-only witness rebuilding covered 124 roots and 345 transitive
  theorems. Source success is constructed directly, yielding observer61 and
  trace1234567; body error73 at index1/call3 yields terminal37/trace1234; prefix
  error91 occurs before setup. No borrowed W proof objects were used. The final
  read-only comparison preserves 117 exact printed types and seven consistent
  compiler-generated bound-name differences, with raw differences retained.
- Primary integration passed 242 focused SDK-free tests. A fresh primary
  baseline check delivered exact final bytes and a separate-Python replay rebuilt
  all proofs. These repeat one existing case, not seven new cases or a new-host
  qualification. Structure/lint evidence is recorded in the final review.

The proof/auditor/host roots now come from the installed package, closing an
environment-redirection gap found before qualification. Root's pinned source
manifest covers 2,009 e048 Python Git blobs; used origins/helpers and full typed
tree roundtrips are checked. Old provisional cuts, test-snapshot preflight
failures, root-selection finding, token-verifier gap and proof counterexamples
remain preserved. No failed or provisional cut was promoted.

## Handoffs

- [Q — actual IR, source binding, gate and replay](Q/HANDOFF.md)
- [W — proof closure and portable reproduction](W/laptop-handoff/README.md)
- [F — independent qualification, controls and evidence](F/handoff/README.md)
- [Final F status](F/FINAL_STATUS.json) and [publication index](PUBLICATION.json)

Exact source/text archives and member manifests accompany these pointers. SDKs,
compiled objects, symlink/binary attack fixtures and giant build trees are
excluded; original native reports retain their absolute paths and hashes.
Regenerate owned objects for replay. The W source-only launcher is not evidence
of a newly qualified laptop environment. Historical helper scripts labeled as
using Git must not be run under a no-Git worker policy.

The earlier loop/buffer integration is also on Graham's
feat/pass-verification-dce-cse at e0484f488. Configuration elision and this new
hoist are on primary; they have not been ported to Graham's branch. That detached
producer checkout must remain read-only at e048 for the recorded qualification.
