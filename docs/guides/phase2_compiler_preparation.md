# Preparing compiler passes for ordinary Phase 2

Selecting a live reviewed capability does not install it into a compiler. The
ordinary authoring entry now accepts a separate original host choice,
`compiler_pass_preparation`, before the compiler inventory and edit authority
are frozen. The original Phase 1 compiler remains a separate readonly snapshot.

The host supplies `CompilerPassPreparation(mechanism, baseline_policy=...,
candidate_policy=...)` from `merlin_experiments.phase2.compiler_pass_preparation`.
Its versioned JSON format is:

```json
{
  "schema": "merlin.compiler_pass_preparation.v1",
  "mechanism": "original_reviewed_composite",
  "baseline_policy": {"enabled": false},
  "candidate_policy": {"enabled": true}
}
```

`read_preparation(path, expected_sha256)` reopens the exact ordinary file. These
choices grant no executable capability. The host must separately supply the
live `CompilerPassSelection` and exact reviewed compiler-library release. The
registry resolves exact hardware, software effects, numerical policy, runtime
and library identities; it does not discover a backend by target or model name.
Missing qualification remains disabled.

The installed authoring and paired-measurement CLIs accept these additional
explicit inputs alongside the existing selected-library arguments:

```text
--compiler-pass-selection /absolute/operator-selection.json
--compiler-pass-selection-sha256 <SHA256-of-exact-selection-file>
--compiler-pass-admission-root /absolute/fresh-original-admission
--compiler-pass-preparation /absolute/original-policies.json
--compiler-pass-preparation-sha256 <SHA256-of-exact-policy-file>
```

The selection file pins the original factory, which is rerun before
consumption; it is not a saved successful verdict. The paired command also
requires `--compiler-preparation-verification-root /absolute/fresh-replay`
for independent original policy application. Replay and admission directories
must be fresh and disjoint from the campaign, compiler snapshots and fixed
library. Omitting every additional input preserves the legacy route. Partial
selections refuse before preparation or candidate consumption. These arguments
retain the paired command's original gSIM requirement; they do not launch a
simulator when inspecting or validating arguments.

`prepare_pair` applies one selected composite owner independently to each arm
from the same immutable original compiler. It never chains one treatment into
the other. Owned writable copies permit additions without changing the original
readonly permissions or source bytes. Both actual prepared packages must pass
the original bound library/source checker, then become readonly comparison and
candidate-seed snapshots. The candidate's first writable authoring workspace is
copied from that seed. Its inventory and explicit edit contract therefore see
the actual policy and helper files before freezing. Preparation does not widen
the edit contract; a missing policy surface or new-component grant refuses.

The original parent, prepared baseline and candidate seed have separate hashes.
Ordinary feedback and paired execution use the actual prepared baseline hash.
Original Phase 1 fork checks still use the original snapshot and digest. The
prompt and stage context publish these distinct identities without private
checker paths. Host preparation changes are not evidence of a learned policy:
authoring must make an execution-relevant change beyond its initial seed.

Prepared candidate records use schema 5 and normalized handoff version 3. Legacy
candidate schemas 3/4 and handoffs 1/2 retain their original interpretation. A
prepared handoff consumer supplies its own original live pass and library owners,
the independently selected `CompilerPassPreparation`, and a fresh host-selected
`compiler_preparation_verification_root`. It reopens the original and readonly
prepared bytes, rejoins preparation receipts in their own hash domain, freshly
reapplies the original policies, and compares both resulting treatment hashes.
Self-consistent candidate JSON cannot choose a new baseline policy or owner.
The replay destination must be disjoint from the original, both prepared arms,
final candidate and fixed library sources.

Both treatments still need the original complete functional grades under the
same frozen release. The final original owner reopens actual graded packages,
source/ELF/runtime artifacts and complete output evidence and performs its final
emitted-program check. Paired workers carry that same live selection and host
policy; the measurement plan and raw products retain the preparation identity.
`paired_inputs.original_baseline(inputs)` supplies the preserved Phase 1 path;
`inputs.baseline` and `inputs.baseline_sha256` describe the actual comparison arm.

The paired evidence reader accepts plan versions 4 and 5 only when its caller
explicitly supplies the original selected-library binding and, for version 5,
the original preparation record. The raw Phase 1 digest remains
`functional_submission_sha256`; the prepared campaign separately records
`comparison_baseline_sha256`. The reader reopens every original raw execution
product, checks its digest, rejoins the actual scheduled compiler arm, and
requires the same library and preparation identities. Rehashing changed
metadata cannot replace those expected bindings. Adoption and parent-bound
reporting obtain them from the verified handoff's original live owners.
Legacy plan version 3 keeps its original interpretation and cannot inherit a
selected library or prepared treatment.

Production preparation requires the ordinary queued route and its explicit edit
authority. It preserves unwaived Phase 1 qualification, corpus admission, original
numerical and certification checks, and timing-domain requirements. Source-only
diagnostics and an available library do not satisfy those gates. This connection
does not qualify asynchronous overlap, buffering, attention, readouts or any
target transformation whose original capability owner is missing.
