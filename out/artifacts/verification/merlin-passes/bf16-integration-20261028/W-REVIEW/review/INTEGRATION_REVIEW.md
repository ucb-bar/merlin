# MACC gate and qualification review

Latest scope: MACC_ONLY.md. No environment/loader investigation or prerequisite is proposed. Exact reviewed source bytes are listed in `INTEGRATION_SOURCE_CUT.json`; later edits need their own review. This source review does not approve numerical policy.

## Q: actual hook, policy checker and publication

The actual `zephyr_model._prepare_model_mlir` change wraps the existing stage at its original position with `pass_validation.observe("lower_bf16_matmul_f32acc", ...)`. The algorithm is unchanged. Selection maps to `bf16_f32acc_policy`, not ordinary tensor refinement. Identity boundaries are skipped as no change, not counted as MACC verification. The worker refuses a normal-return draft dictionary unless the policy handler returns its admitted schema/status; the handler performs the substantive checks.

The handler requires fresh actual before/after snapshots, authoritative round trips, exact receipt/export/report byte bindings, current checker/spec identities and actual `pass_boundary` Exact/Sites flags. It recomputes source function/operation/result roster and exact integer counts; all local flags must be actual Booleans. Nested half contractions and unsupported/no-site scopes refuse. The count scanner intentionally overapproximates Lean's candidate scope: an additional unrelated half generic reduction can cause conservative refusal, not silently become coverage. Source-site positions are not inferred from report counts alone.

W independently executed17 controls against the extracted actual report validator using the retained SmolVLA export/report as data, then mutated roots, flags, positions, counts, omitted/duplicated/reordered sites and review fields. All passed (`handler-controls-v1/RESULT.json`). These are **host JSON validation controls**, not fresh Lean or MLIR semantic negative tests. `VALIDATOR_SOURCE_COMPARISON.json` confirms the controlled function ASTs are identical after Q's formatting/API work; no duplicate suite run was needed.

Current human approval is absent. Q requires a fixed host-installed independently attributable decision, matching numerical domain, exact reviewed source/statement bytes and current policy pins. Caller flags or saved receipts cannot provide approval. The trusted coordinator must actually obtain the human decision: neither a JSON field nor source hashes independently authenticate a person. Pending refusal precedes handler native access; the normal environment factory also checks pending policy before constructing tools. Explicit external construction of CheckerTools is its own normal preflight.

The publisher change preserves its existing pin/replay checks, additionally requires the installed attributable review/admission, compares overlapping live theorem statements, binds actual receipt checker/exporter to that admission, retains approval references, and rechecks them before publication. It does not manufacture an approval. Q's7 reported publisher unit tests use mocked governance; they are not an approved-policy positive. Source inspection found no remaining pins-alone route in `reviewed()` at this cut.

The gate admits a separate reviewed **policy** claim. It does not convert the raw driver's abstracted verdict into a tensor-equivalence proof, require whole-capture definedness, or claim numerical/backend accuracy. Candidate checks remain usable independently of policy approval, which is essential to collecting reviewable evidence first.

## F: independent case/evidence plan

The reviewed source fixture matrix covers both half formats, named/generic contractions, tiny/empty K, empty results, transpose maps, allowed flags, wrong accumulator width/source operand/zero/sign/init, missing site, reordered arithmetic, forbidden flags, unrelated suffix, duplicate accumulation, F14 and named-map production defects. Initial31 pairs are source fixtures, not Lean qualification. The actual preparation tests compare instrumented/uninstrumented emitted bytes and use pending-policy refusal to stop consumer return. These are useful hook/host checks; no accepted numerical proof follows from the mocks.

W raised two actionable findings:

1. `fixtures-v1/wrong_final_rounding/after.mlir` was xDSL-invalid. An invalid-input/export failure must not count as a semantic checker refusal. F retains it and is checking authoritative MLIR validity or an independently valid replacement; `fixture matrix` explicitly labels authoritative validity false until observed. Other negative cases must preserve the same validity/outcome distinction.
2. F's synthetic admission fixture still carried the removed `tools` field after Q simplified the protocol. This fails before the intended governance mutations, so it cannot validate their guards. F was notified to adapt the fixture to the current protocol and retain old results separately.

F's native/build/check results are **not imported as an oracle** for this review. W independently read the contract, theorem conclusions, driver dispatch, serializer field mapping and actual retained data. At this review checkpoint ordinary build/check qualification is still F's active work; complete/native/replay/semantic-negative success is not asserted here. New results should identify the exact source cut, raw authoritative validity, actual full check output, site roster and ordinary fresh replay.

## Concrete claim limits and remaining work

- Independent human numerical-policy approval remains absent. `APPROVAL_REQUEST.md` is the concrete minimal packet.
- Fresh exact/local checker positives and meaningful parser-valid negative results are F's engineering evidence; failures/skips are not successful checks.
- F14's dropped nonzero initializer and F18's fused/unfused discrepancy remain excluded/counterexample evidence, not repaired numerics in this task.
- The checker establishes reference-program equality/refinement. An optional combined theorem directly naming `Contract.result` is proposed in `MACC_PROOF_NOTES.md`, not implemented or assumed.
- No whole-model source-definedness, full shared-pass correctness, arbitrary fusion, bit-exact backend or hardware claim follows.

No implementation/proof changes were made by W, no native jobs were launched by W, and no new policy/lock/approval was installed. The earlier pin proposal and loader notes are retained history superseded by MACC_ONLY; they are not active work or blockers.
