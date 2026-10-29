# Independent proof, correspondence and gate review

ACTIVE assignment from Jeremy's coordinator. The user explicitly opened two
more workers and asked for assignments. You are a REVIEW worker, not another
coordinator. Main coordinator: `01a1650a-b70b-7cb3-adb6-6f2f50bff524`.
Work this complete batch after compaction until completion or a NEW user stop.
Do not stop after one finding or an acknowledgement. No extra agents.

## Goal and ownership

Find errors that would invalidate the strongest scoped graph-correctness paper
claim. Independently inspect the actual code and theorem types, rather than
accepting author summaries. No numerical proof work or new pass features.

Own ONLY reports, source/data probes and snapshots under
`out/artifacts/verification/merlin-passes/progress/planner/paper-deadline-20261026/independent-proof-review/`.
Write an ACTIVE.md there with current task and next step. Implementation files
and other workers' reports are read-only. Send bugs with minimal counterexamples
to the coordinator for the owner to fix; do not edit the owner's implementation.
Coordinator alone stages/commits/integrates/pushes to `feat/pass-verification`.

Read repository AGENT.md/CLAUDE.md, this deadline's README and HUMAN_REVIEW_GUIDE,
then Graham's deliverables1–3 via `git show` at review commit
`5a7f872db33d378d6b0c4a294e4099e80efd4956`. Published primary now337ff1afd;
historical qualified source baseline58dbc7c00; new dirty code has NEW identities.
Original producer `/home/jeremy/Projects/merlin-graph-proof-intake` is READONLY
4499eebb9. New producer fixes are in
`out/build/graph-paper-deadline/producer-corrections`.

## Full work batch, in priority order

1. **Common execution-context review first.** F owns newly drafted
   `src/merlin/verify/lean_graph_original_image_context*.py`. Inspect source and
   SDK-free tests; identify concrete blockers promptly so Q/W can progress.
   Check original-pinned metadata vs new observations, private-only paths,
   source/import/object closure, exact query/root/type/audit binding, live-result
   provenance/revalidation, closed environment/loader, pre/post checks and
   permanent stop on unknown runtime failures. The approved design may use
   compiled-only STANDARD modules: actual object/sidecar refs must already be
   original-selected; hypothetical --src-deps paths are names, not source-byte
   evidence. Project source/object closure must remain exact. No new pins or
   hidden fallback. Send a source-review verdict, assumptions and precise
   missing tests to root; YOU DO NOT authorize or launch native execution.
2. **Compiler gate authority.** Review Q's `lean_graph_pipeline_gate*.py`, the
   corrective producer lean_checker.py/pipeline.py and its exact checkpoint
   packets under `out/artifacts/verification/graph-pipeline-gate/`. Trace actual
   plan_only=False use. Attempt source/data/mock counterexamples for local/id
   receipts, forged/stale/missing/wrong-domain/context evidence, skipped checker,
   invalid root/query association and complete entry/observer omissions. Check
   the final generated `merlin_transform` theorem actually mentions both actual
   programs and covers unchanged definitions, not a local equation. Preserve
   mock-vs-native and unavailable-vs-semantic-refusal distinctions.
3. **IR mapping adversaries.** Independently review F's input-admission guard,
   capture hooks and checkpoint under graph-regions-declared/input-admission-review-v1.
   Check resource loss before parse/print, nested attributes/types/regions,
   indirect/function-value calls, supplied-buffer/public bypasses, and shape's
   narrow proved direct-call exception. Inspect W's source-candidate-v1 under
   graph-shape-normalization/after-binding-deadline-v1: separate actual-after term,
   scoped SSA/call renaming and source/target semantic bridge. Mutate VALID IR;
   prove candidates reach a changed projection/query. Query sensitivity alone
   is not a successful Lean refusal. Note the final raw memory-SCF target bridge
   remains open; do not generalize the new shape result to it.
4. **Theorem/premise and composition audit.** Start with one retained full
   scalar/SCF chain and exact generated final types, then integer, shape/reuse,
   and declared-call rows that the paper intends to name. Check domains,
   quantifiers, canonical inputs, initialized/reached typing, complete errors/
   terminal world vs possible successful outcomes vs selected observers,
   common interpretations and exact-middle transport. Look for circular whole-
   program equality assumptions or vacuous domains. Retain allocator 8→16,
   numerical and other existing counterexamples. Classify local contracts,
   control, private allocation, frame/ABI and parser/host fidelity explicitly.
5. **Paper claim response.** Produce a row per reviewed family with source cut,
   final theorem/type, admitted/excluded variants, proved facts, premises,
   trusted mapping, actual artifact coverage, evidence status and exact permitted
   wording. Respond to Graham D1-G09/D2-G01/G02/G03/D3-G10 as closed-source-only,
   qualified, open, or narrowed claim, with evidence. Propose precise manuscript
   edits for both FIXMEs. Agent review is not human/specification approval.

## Evidence and execution boundary

Read historical files/snapshots/archives as DATA. No Lean/MLIR/native compiler
launches, old dependency guards, installed SDK reads/rehashes, SDK repairs or
failed-profile retries. Root's private original-image Init probe succeeded, but
does NOT authorize graph contexts or your native use. SDK-free Python tests are
allowed only after checking their call paths; retain full fixtures/raw outcomes.
No changes to Core/Sem/Spec/locks or any frozen capsule. No new axioms, sorry,
admit, native_decide, bv_decide, reflection, unsafe, extern or substituted proof.

## Deliverables

Keep FINDINGS.md with severity, exact file/line+source digest, reproduction,
claim impact, proposed resolution and owner. Keep exact command/result records,
immutable probe candidates, reviewed-source inventory, a common-context source
verdict, CLAIM_ROWS.md and a concise human handoff. State what you did NOT run.
Send immediate blockers; otherwise report one substantial batch milestone.
After one section, continue the next; avoid repeated status messages or test
expansion that only mirrors implementation. Native availability is not needed
to complete the source/correspondence/claim audit. Re-review owner fixes against
new explicit source cuts when available; never retroactively approve old ones.
