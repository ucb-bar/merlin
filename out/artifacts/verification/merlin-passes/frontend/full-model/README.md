# Actual-program runtime checkpoint

Additive, inactive candidate. Read [remaining obligations](remaining-obligations.md) before
interpreting the results as full-model verification. The existing f32 policy/IEEE approval is
retained; runtime semantics and admission still await human review.

## Reproduce

Use the separate `pv/frontend` checkout. Build the production checker/auditor and candidate:

```sh
cd merlin/lean
lake build MerlinVerify merlin-lean merlin-lean-audit
lake -f FrontendRuntime.lakefile.toml build merlin-lean-frontend-runtime
```

Set `PYTHONPATH` to this checkout's `src`, `MERLIN_COMPILER_PYTHON` to the authoritative
model2MLIR environment, `MERLIN_TORCH_PYTHON` to **PyTorch 2.10.0+cpu**, and the native tool
variables documented in the preceding runtime-candidate reproduction guide. The new qualifier
regenerates ignored before/after MLIR from the capsule and production pass; it does not rewrite
the old qualification records. From the repository root:

```sh
python out/artifacts/verification/merlin-passes/frontend/full-model/qualify.py
python -m pytest merlin/tests/ir/test_lean_frontend_runtime_checkpoint.py
```

The API `merlin.verify.lean.frontend_runtime.verify_candidate` snapshots actual MLIR,
re-exports with the authoritative compiler, checks round trips, audits theorem dependencies,
and invokes the compiled Lean candidate. Its receipts bind source bodies, compiled objects,
checker executable, runtime budget/index width, A2 and its scoped approval record. Changed
artifacts, source/spec/configuration, altered claims, mixed frozen receipts and stage order
are refused by `validate_receipt`, which reruns authoritative round trips, dependency audits
and the actual Lean checker. Even a result forged with recomputed hashes is rejected. These
checks are evidence-integrity glue (T2), not signatures
or additional formal proofs. Reproduction creates fresh receipts; historical results remain
historical.

## Main theorems

In plain terms, acceptance of the actual before/after programs establishes:

```text
For index64, budget 1024, and inputs of the declared types with initialized elements:
  the actual prefix plus selection executes successfully and returns typed initialized data;
  the original initial segment interpreted by A2's f32 repair policy returns the same data.

For the complete module, including poison inputs and unsuccessful outcomes:
  source-only A2 policy execution(before) = candidate execution(after).

For every supplied suffix accepted by the same candidate checker:
  candidate execution(final) = source-only A2 policy execution(before).
```

The f32 UNE law is proved for the approved direct IEEE instance, including every NaN encoding;
poison lifting is proved separately. There is no remaining `UneCorrect` assumption in these
specialized claims. Other float operations remain parametric. The logical dependencies are
only `propext`, `Classical.choice`, and `Quot.sound`, alongside the existing kernel/compiler,
export/round-trip and evaluator trust boundaries. Runtime interpretation approval is a review
decision, not a new axiom. Full-module policy matching includes unsuccessful outcomes; the
separate successful-execution claim stops at the proved initial segment.

- `FrontendRuntimeVerification.check_sound`: actual source/repaired initial-segment execution
  and typed initialized results, on the declared initialized input domain.
- `FrontendRuntimeVerification.full_policy`: exact complete-program outcome equality to A2's
  source-only policy under the same candidate interpretation.
- `FrontendRuntimeChain.compose`: checked ordered alpha suffixes preserve that policy claim.
- `FrontendRuntimeObligations.*`: local token-index and float-conversion obstructions.

All names above are under `MerlinVerify.Rules`. The candidate has a separate build configuration
because the existing production lakefile is protected. No candidate runtime interpretation or
claim admission is silently activated by this configuration.

The [production dispatch inventory](../format-inventory.json) remains explicit: signed and
unsigned f32-to-i1 repairs are proved; the other `AnyFloat` classes are unsupported by this
proof. Construction/printing probes for those formats are not formal or native qualification.

Implementation commits: `7769ac530` (prefix initialization), `bfc4d0012` (actual execution
composition), and `e9d85e0a8` (candidate API, replay and qualification tests), based on
`58f6a4bc0`. [integration.patch](integration.patch) contains these additive changes; applying
it was checked against that base with an isolated Git index. It changes no protected definition,
production lakefile, SPEC_LOCK, compiler pass or W8A8 file. Review and cherry-pick the coherent
commits, or review the patch, before admission; this task does not merge into main.

`inventory.json` lists every operation family, float-to-integer cast, tensor read, and dynamic
result in the full after module. An inventory is not a safety/definedness proof. `receipt.json`
and `qualification.json` bind actual checking and native/oracle execution. Test XML, build/audit,
guard and structure logs record the qualification run. Execution samples remain distinct from
universal Lean proofs.

Final qualification: **74 tests passed, zero failures/skips; 9,483 theorems audited, zero
violations; 20 native selection samples and 16 pinned PyTorch selection samples**. Library,
checker and auditor build passed (1,472 jobs); the separate candidate executable build passed
(1,380 jobs). The full guard still reports the same three prior unadmitted specification/schema
entries. The structure gate still reports the unrelated `zephyr_model.py` size ceiling failure.
Neither gate is claimed green. `validation.json` retains the exact results and file hashes.
Raw logs, captures and native result files remain local ignored outputs; reproduction creates
fresh evidence rather than requiring the historical machine's absolute paths.

| Status | Scope |
|---|---|
| Proved | Actual prefix initialization, actual selection safety/termination, composition through SSA, complete f32 repair-policy matching, checked alpha suffix composition |
| Execution-tested | Actual and independent native selections, empty selections and both branches, pinned PyTorch 2.10.0+cpu, f32 NaNs/infinities/signed zeros/fractions, valid mutations and refusal cases |
| Awaiting review | Runtime interpretation/admission; proposed token domain and future conversion contract |
| Unsupported/unproved | Full-model return initialization, non-alpha transformed suffixes under this candidate, other frontend float formats, the unused host-conversion bridge |
