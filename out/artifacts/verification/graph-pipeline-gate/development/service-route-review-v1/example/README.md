# Actual service → graph gate → compiler eligibility, SOURCE-ONLY / MOCKED

Source cut: source-checkpoint-v9. Complete actual CSE input/before/after and query
retained here, with exact full source closure in PREPARED.json. GRAPH-PROFILE.json
is descriptive; MOCKED-SERVICE-OUTPUT.json is NOT re-loadable authority. RESULT.json
lists all mocks and false native/qualification/public-activation flags.

## New host-selected graph profile path

Trusted host, outside untrusted requests, first obtains a separately authorized
exact frozen Context from the reviewed provider/context_arguments recipe. Then:

```python
binding = bind_service_context(
    context, compiler_profile_sha256=admitted_profile["sha256"], timeout_s=120)
service = CommonTransformService(
    workspace=workspace, bundle_manifest=original_manifest, repo=repo,
    graph_binding=binding)
result = service.apply(actual_request, work_root=context.workspace)
allowed = service.compiler_eligible(result, source_ir=actual_request["source_ir"])
```

This code is a review recipe, NOT current permission to construct/run Context.
The real native backend is hard-held. Binding/source/context setup does not issue
a proof or waive original immutable bundle/profile verification. No request JSON
may choose the binding/provider/profile, pass a plan_only flag, supply saved proofs
or enable unsupported passes. graph_binding is an explicit trusted-host constructor
parameter; saved descriptions/dataclass copies fail the process-local ledger.

Binding records exact compiler profile digest, current source closure and context
identity/workspace/budget; validates selected Fv5 class/metadata/finite descriptor,
minimal7 source map and11 dependency roots. No Context factory, legacy _toolchain,
installed/private SDK guard or fallback is invoked by graph selection or service
routing. apply requires shape/integer/storage defaults explicitly disabled, and
only CSE/LICM selectors. A graph-bound authored-check call refuses legacy fallback.

Actual apply runs parse_ir and apply_common_pipeline(plan_only=False). Each actual
change must receive fresh whole-program gate authority and same-context admission.
After observations, sources/context and receipts are rechecked, exact original,
adjacent and final IR buffers are bound and deadline honored before a live delivery
is registered. compiler_eligible requires the SAME unmodified dict/source/binding
and freshly revalidated original receipts. A saved/copied JSON object, output edit,
changed profile/context/source, or a refused second stage cannot grant eligibility.
Unchanged output is descriptive and does not mint changed-boundary authority.

## Evidence and limits

46 SDK-free tests pass v28 and final4 service tests pass v2. Native guards and old
_toolchain are set to fail if reached. The actual service constructor/apply/_verify,
gate/pipeline/eligibility code runs; original bundle/profile custody functions and
selected Context native methods are MOCKED. No mocked result is Lean acceptance.
All source/API evidence is distinguishable from unresolved runtime validation.

The requested profile does not add whole-program equality as a premise. Per-boundary
model correspondence keeps returned/error/world outcomes. Cross-boundary registry
composition is not silently inferred: compiler_pass_chain remains not_established.
Hardware, numerical, poison, total execution and review approval remain separate.
External verifier/custody callback side effects are not claimed transactional.

Coordinator must select/publish the corrected host sources and graph leaves in a
NEW release/bundle/profile and preserve existing original_compiler_imports/custody
requirements. This patch does NOT edit shared profile schemas/import dispatchers,
refresh old source pins, or assert a genuine ordinary package launch. Default
service apply without explicit graph binding now fails closed. Separate authored
legacy checking with no graph binding remains outside this graph claim.

Public activation still needs a separately reviewed new source/context cut; its
two-entry proposed dispatch change remains unapplied. Global runtime hold suspends
all previous native permissions. No new native request or retry is authorized here.
