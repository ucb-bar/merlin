# PAUSED BY NEW USER STOP — allowance 5%

Stop checkpoint HEAD: `1483336b07fcc9732b017a1ba2ce93c870e64c05`, branch `feat/pass-verification`. No owned live jobs. No automatic continuation. Native hold persists.

# Portable W worker handoff — 2026-10-26

Current checkout: `/home/jeremy/Projects/merlin`, `feat/pass-verification`, observed HEAD `9d7fe0846c43574ca6f83a5cccc5dc147b06ca6e`. Coordinator may advance HEAD. All paths below are repository-relative (this directory is `out/artifacts/verification/graph-shape-normalization/graph-infra-binding-v1`). No owned live jobs, no implementation edits in the new task, no stage/commit/push. Prepared under quota checkpoint direction, not a claim of completed infrastructure fixes.

## Current assignment and exact state

Own only `src/merlin/verify/refine.py`, `src/merlin/verify/receipts.py`, and new narrowly named SDK-free tests. Fix conservative dtype/width binding including shared leaves, imported `frontends.linalg_patterns` provenance, and audit output ordering against actual ABI before changing it. Baseline source snapshots are under `initial-source/`; hashes in `INFRA-BASELINE.json`. `infra-current.patch` is intentionally empty: no fix was implemented. Preserve those originals and counterexample when starting a successor.

`PURE-HELPER-COUNTEREXAMPLE.json`: the actual extracted `_bind_leaves` AST accepts shape[2,3] width8 shared tensor for dtype i8, i16, f32 and bogus. Only i8 is the matching supported control. This SDK-free observation is a helper admission failure, not solver SAT/refutation or complete encoder execution. `cb_semantics.CommandBufferEncoder.declare_leaves` also installs shared values before its dtype parser; fix the assigned refine boundary without editing that foreign file. `linalg_semantics.py:156` imports `merlin.frontends.linalg_patterns`, omitted from receipts' current `_SEMANTIC_FILES`/bridge digest. No digest-drift reproduction yet. `_bind_outputs` sorts commit names lexically; interface semantics emits commit{n}. Actual lowering/runtime ABI audit unfinished (attempted lowering/runtime.py path does not exist); do not change order merely from name conventions.

## Next source-only steps

1. Read local AGENT/CLAUDE/architecture instructions and `refine.py`, `receipts.py`, the width parser and shared-input path in `cb_semantics.py`, and actual output lowering. Use `rg --files src/merlin/xdsl_dialects/lowering` to locate real runtime emitter.
2. Add narrow helper/mocked admission regressions for correct/mismatched/malformed dtype, widths and shared leaves. Use existing dtype interpretation; no new numerical semantics. Avoid real module import paths that optionally load z3; install process/dlopen/file guards or stub before importing. Use `.venv/bin/python` for pytest (system Python lacks pytest).
3. Add imported frontend source to receipt identity and source-only drift regression. Preserve v1 historical receipt readability; changed verifier identity must not silently qualify historical receipts. Distinguish disk identity from already-loaded Python code identity. Coordinate shared closure changes only through root/Q.
4. Audit commit output ordering versus real ABI, including >9 outputs; fix only an established mapping or explicitly refuse ambiguous mapping. Produce new manifest/patch and exact narrow test logs. No broad archive/old suite rerun.

Do NOT run real `_toolchain`, `verify_transformation`, `qualify_receipt` or existing receipt tests without safe mocks: they invoke/read translator/solver. ALL native/SDK/runtime/archive-image/object access, guards, repair, selection, Lean/SMT execution remain held. F alone investigates. No new agents, numerics, frozen/shared/production/Core/Sem/Spec/lock edits. Root alone publishes. A new root stop wins.

## Finished source packets, not new correctness claims

`EARLIER-SOURCE-PUBLICATION.json` lists 58 earlier ShapeEpoch/ShapeAfter Lean/Python/test sources with live hashes and individual historical manifest comparisons (all exact). These are finished source snapshots for publication/review, NOT qualification. It names small historical evidence entrypoints. No compiled objects included. Preserve prior 61-file packet at `../after-binding-deadline-v1/final-memory-after-v1/publication-context-v1/MANIFEST.json` and all source-context-v1/entries/whole seals. The 61-file packet carries project-source dependency closure, 14 full raw/query preparation cases and coherent review patch. Earlier shape dependencies are in `../epoch-source-only-v2/SOURCE_INVENTORY.json` and `../after-binding-deadline-v1/source-candidate-v1/{LEAN-PROJECT-SOURCES,PYTHON-PROJECT-SOURCES}.json`.

Caller-context candidates derive ONE direct transformed kernel call followed by call-free scalar/index/public load/store continuation, using public heap/user relation rather than full private world equality. Actual raw wrapper is a test extension, NOT the original4499 capture. Nonzero candidate witness: kernel308/index2, public183; bias7 wrapper observes/returns/stores498. Existing abstract Nat interpretation only, no new numerical/native fidelity law. 14 retained SDK-free preparation tests passed historically; Lean candidates UNCOMPILED, query outcomes UNOBSERVED, UNQUALIFIED. Repeated/transitive calls, prefix allocations, private escape and external callers remain refused. Possible-success semantics is not categorized error/divergence/ABI preservation; allocator total/private/unobservable, primitive interpretation/projection/review assumptions remain.

## Preserved failures and portability

Source producer4499eebb959c0f92cefd92cc6d80646ba66474ed remains read-only. Runtime SDK incident cause UNKNOWN; old Wv1 duplicate Init, Wv2 unknown Lean, Wv3 protocol optional-symbol/no-executable-code failures remain under `../after-binding-deadline-v1/consumer-development-v{2,3}/`. No retry or refreshed expectations. Historical 8→16B allocation countermodel, elision crash and numerical counterexamples remain. Prior absolute Linux receipts and contexts are historical evidence, not portable laptop qualification. Do not restore snapshots over live sources or silently install/reselect Lean/SDK on laptop. Review source/closure first; obtain explicit new runtime authorization separately.

Own durable record: `../active/README.md`. Prior complete graph checkpoint copied to `PRIOR-ACTIVE.md`. Publication and independent review are coordinator/Graham responsibilities.
