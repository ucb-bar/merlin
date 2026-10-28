> CURRENT SCOPE: MACC_ONLY.md supersedes environment/loader/pinning-system work. The prior pinning proposal and environment-review notes below are retained history, not active requirements or blockers. Current deliverables are APPROVAL_REQUEST.md, IR_MAPPING.md, MACC_PROOF_NOTES.md and INTEGRATION_REVIEW.md. F uses ordinary repository build/check mechanisms.

# Independent source/contract review — W, 2026-10-28

**Scope:** `lower_bf16_matmul_f32acc`, base `65e44afbf4dbb6c7270e8936c4ef76b579a78492`, branch `feat/bf16-f32acc-verification`. This is an agent source review, not human approval, a new proof audit, or fresh native qualification. The implementation, Lean sources, locks and old evidence were not modified by W. Q's integration and F's engineering evidence are reviewed separately below as they arrive.

## Independent reconstruction and comparison

The intended policy is a higher-precision accumulator for a **zero-initialized**, two-input contraction of same-format bf16 or f16 tensors. Read the operands using their declared indexing maps; extend each scalar to f32; update a separate f32 accumulator for each output coordinate in the declared ordered iteration; convert each finished accumulator once to the original output format. This is a deliberate numerical policy, not equivalence to a loop that rounds every partial sum to bf16/f16. Greater intermediate precision is not a universal accuracy theorem.

This reconstruction is consistent with MLIR's specification that named matmul promotes operands to its accumulator type and accepts explicit transposition/broadcast maps, and that `extf`/`truncf` change scalar format. MLIR does **not** establish Merlin's choice to replace a half accumulator, drop nonzero initial state, impose lexicographic order across all backends, or select a particular fusion strategy. These require the explicit policy and fidelity premises. Sources: [MLIR Linalg matmul](https://mlir.llvm.org/docs/Dialects/Linalg/#linalgmatmul-linalgmatmulop), [MLIR Arith](https://mlir.llvm.org/docs/Dialects/ArithOps/). Accessed 2026-10-28; these web docs are reference context, not the pinned installed compiler's specification identity.

`Rules/Bf16F32AccContract.lean` states six clauses: attribute-free widening; source indexing; lexicographic ordered per-cell fold; f32 positive-zero bits; attribute-free final narrowing; exactly `none`/`contract` flags on multiply/add independently. `step` orders multiply operands as left then right, and addition as accumulator then product. `Bf16F32AccBody` fixes the five scalar accumulate operations and two scalar final-conversion operations. `Bf16F32AccReference` builds the six-operation tensor replacement and rejects any in-scope candidate it cannot recognize.

Important precision of the claim: the reference is a **source-derived family**, with the output supplying only checked choices of multiply/add flags and semantics-free annotations. It is not one unique reference independent of those choices. Those choices cannot select maps, operands, initialization, arithmetic order, types or arbitrary replacement operations.

| Issue | Actual checked boundary | Review disposition |
|---|---|---|
| Formats | Static, unencoded tensor shapes; operands and result share one bf16 or f16 format | No mixed bf16/f16 operands, dynamic shape or arbitrary tensor encoding claim |
| Source kind | Direct operations in single-block top-level defined functions: named matmul or two-input generic with reduction and multiply/add | Named batch_matmul and nested transformations are not covered; Q additionally refuses a conservative superset of omitted half reductions |
| Source scalar dataflow | Exactly multiply, add, yield, with source mul/add pair swaps allowed and absent/none flags | Source commutation is policy recognition, **not** a proved law of arbitrary FloatModel; policy canonicalizes the operand order |
| Maps and bounds | Projected maps, no symbols, output dimensions parallel and distinct; local `structuredRanges?` derives extents and all accesses | A syntactic `contractionMaps` result alone is insufficient; strict local qualification supplies geometry/support obligations |
| Initialization | Literal bit-zero of the same format, through source splat or fill | `-0.0`, nonzero, runtime initializer and alternate producer chains are not this domain |
| Initializer producers | `Sites.initialization?` copies actual literal/splat, or literal/empty/fill | A fill must trace to operand-free `tensor.empty`; exact matching alone has a broader recognizer and is insufficient |
| Accumulation order | `allPoints` and per-cell ordered fold; source maps retained | Ledger A5 is a lowering premise, not a universal Linalg reordering theorem |
| Truncation | One attribute-free `arith.truncf` per output cell | No explicit target rounding mode proved; current MLIR docs defer unspecified runtime rounding to the backend |
| Flags | Four combinations of exactly none/contract; no reassoc/nnan/ninf/nsz/fast | Permission is not a proof that actual backend fusion is uniform or bitwise identical |
| Local domain | Typed, initialized operands at the receipt's index width; new gate limits width to64 | Not all whole-model intermediate tensors are proved typed/initialized |
| Empty dimensions | `operand_domain_nonempty` constructs zero tensors including empty shapes | Nonempty mathematical input domain, not a whole captured-model execution witness |

The original contract author read the production pass. This review has also read it. Neither meets the design's barrier-authored condition; an independent human must approve the exact numerical recipe and explicitly accept the reviewed-candidate process before any approved policy claim.

## What each theorem actually establishes

1. `Bf16F32AccExact.check_sound`: if its Boolean check succeeds, the output and the reconstructed reference have equal `denote` for every context, function name and argument list, for an operation interpretation satisfying `ParametricAlpha.Laws`. `ParametricModule` checks complete module skeleton, function roster/attributes/declarations and alpha-related bodies; calls use a shared fuel induction. This includes equality of *modeled* stuck/UB/success results. `denote` uses function-count+1 fuel: it is not native recursion/divergence/exception/ABI correctness.
2. `Bf16F32AccSites.check_sound`: exact whole-output check **and** strict policy qualification for every collected actual source/target fragment. `collectOps_coverage` preserves positions, original source operations and order for the traversed function bodies. It does not recurse into all arbitrary nested regions. No local success excuses an unrelated module change.
3. `Bf16F32Acc.qualify_sound`: signatures agree, the local original source is defined on typed initialized arguments, and actual output strictly refines the policy reference in frozen `Sem.Ops.opSem`. Strict refinement excludes stuck-reference vacuity but should not be restated as a generic successful-output theorem without the numerical/definedness bridges. The old **whole-capture** qualification is a different invocation and remains false.
4. `Bf16F32AccTensor.{accumulate_sound,truncate_sound,zero_fill_sound}` and `Bf16F32AccSite.{accumulating_op_sound,final_op_sound}` explain the emitted replacement under checked geometry, matching bodies and frozen semantics. `operand_domain_nonempty` constructs initialized inputs; it does not assume source execution.
5. `ParametricAlpha.Laws.operation` requires interpretation invariance under the exact compared operation head, operand arity and region-presence facts; `call_local` restricts call dependence to the recorded callee. Unknown surrounding operations are preserved under these explicit laws, not semantically implemented by the checker.

All theorem sources were read; W did not elaborate/audit them. `THEOREM_ROOTS.json` names17 existing claim roots. `PROJECT_SOURCE_CLOSURE.json` records291 project Lean source modules imported by Main and those roots and unresolved external module names. This is a static source closure, not an audited loaded-object closure. Standard-three-only must be checked freshly by F's authorized environment.

## Actual driver and serialization

`Main.lean` dispatches `check-bf16-f32acc` to `Driver.Bf16F32Acc.run`. `report` computes the older conditional check, whole-capture domain diagnostics, and **also** `siteReport`. The latter actually calls `Exact.check`, `Sites.collect`, every `pairCheck`, and `Sites.accept`; it is not merely a printed theorem name. The correct pass-local admission fields are `pass_boundary.checked`, `semantic_match`, `site_qualification_passed`, all actual site obligations, and exact coverage counts. Top-level `conditional_match` or `qualification_passed` alone does not express this claim. Driver returns1 and `abstracted`, `policy_reviewed=false` even on an accepted candidate; only a separate approved admission may promote a fresh result.

The authoritative serializer is `mlir_export.py`, run by the selected compiler Python via `export_file(backend="mlir")`. Its native parser is trusted. Types, maps, scalar bit patterns, all attributes and SSA uses/definitions are encoded. Inherent/discardable attributes share the attrs dictionary on this route; props is empty. Opaque attributes remain visible, not erased. Multi-block regions/successors refuse. `export.py` has a separate xDSL route; do not silently substitute it for authoritative native round-trip evidence. See `IR_MAPPING.md`.

W executed six extracted-helper controls for positive/negative zero across bf16/f16/f32 in the two actual exporters. Their distinct sign bits are preserved. These tests did not import a parser, invoke Lean, or establish general floating literal/NaN fidelity. Native round trip, exact raw/export/report binding, host trust and clean replay remain necessary.

## Retained24 sites and worked example

`RETAINED_IR_INVENTORY.json` independently scans historical exported JSON and checks the bound raw MLIR, exports and report bytes. It finds24 direct generic contraction sites in SmolVLA; the segment reuses exactly the same source export and24 sites; TinyLlama has0. Independently computed function/index/result rosters agree with the old reports. **One distinct positive boundary and24 sites, not48.** These data checks do not replay the checker or make old receipts fresh.

`ACTUAL_SITE_0.json` retains one actual source initializer/contraction and its six actual target operations, types/maps and exact raw/export bindings. The first site is function `forward`, source operation7624, result17210, bf16 result shape[15,113,64]. It uses its own four-dimensional batched contraction maps. This supplies a concrete paper example to pair with F's fresh retained checker output; it is not a newly generated capture or new qualification.

The old receipt lock hash differs from the current lock (`HISTORICAL_COMPARISON.json`). Q/DQ semantics changed in the base. Historical59 fast tests,3 replays and audits in FINDINGS are historical evidence only. We did not read/hash old runtime images or compiled objects.

## Preserved counterexamples and isolated fix branch

- **F14:** nonzero source destination is discarded by the base production pass. Source `init=1.0` produces a zero-initialized replacement; the policy checker rejects the source. Native evidence is retained in old tests/reports, not rerun. No gate may call this a successful zero-init transformation or weaken initialization to admit it.
- **F18:** bf16 products `-2^127`, then `2^128` can give positive infinity with separately rounded f32 multiply/add, versus `2^127` with fused steps (bf16 result bits7f80 versus7f00). This refutes unconditional bit-identical fusion/NaN/Inf preservation and the production comment's universal accuracy implication.
- The explicit named-matmul transpose bug, `a*a` generic rewritten as `a*b`, and invalid f32→f32 extension on mixed-width inputs remain outside accepted scope and must retain refusals.
- Isolated `fix/bf16-f32acc-contractions` at `5cf6196ce0247c0f277908ad4f9cfb25e8a6a333` preserves maps/dataflow and extends nonzero initialization via `extf(C0)`. That changes the policy's initial accumulator and emitted template. It needs a distinct reviewed contract and proofs, and is excluded here. Its whole-file branch diff also contains older Q/DQ changes; wholesale merging is inappropriate.
- `fused_step` encodes a uniformly fused model using an injective widening on one format and no unrelated uses of its licensed keys. It is not arbitrary per-site fusion, NaN payload injectivity, real backend instruction selection or independent accuracy evidence. The raw driver reports some premises but does not prove target uniformity.

## Pinning and approval

`PROPOSED_AFTER_HUMAN_APPROVAL.patch` is an **unapplied** current-base proposal. It preserves current semantics/locks, moves unchanged candidate contract bytes behind a compatibility import, and adds statement/definition pins. The candidate warning is deliberately retained; comments do not constitute approval. The proposal includes extra definition files behind exact/domain claims absent from the historical patch. `PROPOSAL.json` records exact hashes and future actions. The remaining source closure must be bound by the admitted build, and current elaborated statement hashes must come from F rather than the old candidate. `git apply --check` succeeded; no patch or lock was applied.

Existing publisher `reviewed()` recognizes matching Spec/lock/statement hashes without independent reviewer attribution. That is not sufficient for the new gate; Q's protocol adds a fixed host-installed, exact-byte-bound attributable review record. Such a record is governance, not cryptographic proof that a human read it. Root must obtain/retain the real review. No agent writes or installs it.

## Integration review in progress

Q's initial protocol correctly separates raw candidate results from approved gate acceptance, checks actual source roster with a conservative nested/unsupported refusal, and refuses pending review before handler SDK access. Two review obligations were sent: include the actual imported parser/host source in source bindings; bind the audit's loaded objects/dynamic runtime and authoritative compiler environment, not just executable hashes/package versions. The gate's initial PIN_FILES also adds Site; W will reconcile its unapplied proposal. A final reviewed-source digest and actionable findings will be recorded in `INTEGRATION_REVIEW.md` after the delivered source/test cut is available.

F owns fresh actual positive/mutation/round-trip/checker/replay and exact environment qualification. W must not infer a successful semantic refusal from infrastructure failure. Final source review is not approved gate acceptance, whole shared interval verification, original-half equivalence, whole-model definedness, or backend/hardware correctness.
