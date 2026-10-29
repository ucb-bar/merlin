# Ordered attention numerical checks — active implementation

Continue the [concrete-context assignment](../../progress/planner/assignments/w8a8-concrete-context/README.md)
and [eight-day goal](../../progress/planner/eight-day-paper/README.md). The user wants continued
owned work until an explicit stop. The separate planner is active; an old request for another
planner briefing is obsolete. This is ongoing implementation, not a qualified attention release.

The preceding [joint numerical context](../joint-model/README.md) is qualified: 65 tests,
10,168 theorem audits with zero violations, eight curated separate-process replays, retained
TinyLlama163-bound replay, unchanged historical/current identities. Its complete89-file
delivery and frozen build must remain byte-for-byte unchanged. All broad flags remainfalse.

## Current source and build

NEW private source/build profile:
`/home/jeremy/Projects/merlin/out/build/lean-w8a8-attention/repo`.
It is a reflink copy, not hardlinked to a sealed profile. New modules are copied from the
owned main source paths into that profile before building. Packages/toolchain remain the
same pinned read-only dependencies. Use project-disk temporary storage:
`/home/jeremy/Projects/merlin/out/build/lean-w8a8-attention/process-tmp-v1`.
Do not rebuild the old joint-model, modular, rotary, numerical or historical capsules.

The following NEW additive modules compile, with only the standard three axioms in roots:

| Module | Proved component |
| --- | --- |
| Rules/W8A8OrderedMagnitude | Exact unpacked absolute magnitude in integer units of2^-149; one-sided pack bound; conversion between integer magnitudes and exponent caps |
| Rules/W8A8OrderedRounding | Both actual rounding shifts, their magnitude error and finite packed-round bound; no assumed rounding-error law |
| Rules/W8A8OrderedSum | Actual signed addition, its conservative count invariant, actual ordered fold and derived exponent cap; no floating reassociation |
| Rules/W8A8OrderedDot | Actual rounded products followed by ordered additions; every prefix; recomputed product/count guard |
| Rules/W8A8OrderedBody | Exact actual scalar-region semantics for both accumulator operand positions and poison; model specialization |
| Check/PolicyW8A8OrderedBody | SSA/type/fastmath-sensitive alpha checker and soundness for the identical explicit W arithmetic family |
| Rules/W8A8TypedTensorTrace | Ordered mapped-write simulation with an invariant on states actually accessed |
| Rules/W8A8OrderedCells | Typed cell updates, exact selected cell trace, numerical cell bound and reduction count derived from projection geometry |
| Rules/W8A8OrderedTensor | Exact body-at-point and complete generic execution using actual regions, reads, mapped writes and runtime initial state |
| Rules/W8A8OrderedLayout | Independently checked f32 shape/map geometry, output projection and reduction size; integer layout data is reused without integer typing assumptions |
| Check/PolicyW8A8OrderedRefined | Complete generic runtime execution with actual mapped initialized inputs, zero SSA initializer and initialized bounded output facts |
| Check/PolicyW8A8OrderedOp | Actual matmul/batch_matmul/generic decoder and semantic lift; malformed explicit maps refuse instead of falling back |
| Check/PolicyW8A8OrderedRule | Contextual rule soundness recomputing operand caps, layout, actual body, initializer, static result type and count guard |
| Check/DefinedContextPrefix | Reusable contextual prefix walker with actual SSA shadowing and execution soundness |
| Check/PolicyW8A8OrderedPrefix | New book and entry-bound theorem for the explicit model; old book is restricted through the proved ContextOk implication |

Latest successful logs: `ordered-build-v8.log`, `ordered-dot-build-v5.log`,
`ordered-body-check-build-v3.log`, `typed-trace-build-v1.log`,
`ordered-cells-build-v2.log`, `ordered-tensor-build-v3.log` (including complete tensor
execution plus per-cell numerical cap).
Further successful logs: `ordered-layout-build-v3.log`, `ordered-refined-build-v5.log`,
`ordered-op-build-v2.log`, `ordered-rule-build-v4.log`, `prefix-engine-build-v1.log`,
`ordered-prefix-build-v1.log`, `ordered-driver-build-v2.log`. All new proof roots report
only the standard three axioms. Additive Driver/W8A8OrderedContracts and W8A8Ordered plus
W8A8OrderedMain compile. Source and reference demands are separate fields in the new contract
schema; existing contract/driver routes remain unchanged. New owned Python ordered_policy.py
and ordered_evidence.py implement fresh runtime audits, source/export/contract binding and
strict JSON replay; tests are currently in DEVELOPMENT. Ruff passes these additions.
These are development builds; no full new guard/freeze/native/receipt qualification exists yet.

Kernel fixture `merlin/tests/fixtures/lean_w8a8_ordered.lean` passes (`ordered-kernel-v5.log`).
It specializes the theorem to every2048-term sequence with operand cap127 (result cap142),
checks count/product guards, distinguishes two different floating addition orders, retains
subnormal/signed-zero controls, and proves a rounding-to-infinity counterexample showing why
raising the magnitude limit from252 to253 is unsound. Both scalar body templates have
kernel-checked positive acceptance, using equations for the mutual alpha checker.
Failed development elaborations are preserved separately and are not audited/qualified
results; no sorry/admit/new axiom/native_decide appears in source.

`existing_context_allows_nan_trig` is a kernel countermodel: the already proved ContextOk
candidate has finite input0 yet returns NaN for both sin(0) and cos(0), from its explicitly
unqualified fallback. Thus the existing seven-plus-three laws and nine Agreement fields do
not imply a trig range property. This is not a claim about native sin/cos behavior. Actual
SSA evidence in `trig-dependencies-v1.json` finds cos carrier1180 and sin carrier1193 upstream
of quantizer1533 through the rotary/logit/softmax/final-V path. The graph is conservative
syntactic dataflow, not a semantic execution or necessary-dependence theorem. Positions
include0, so intermediate angle0 cannot be excluded merely by nonzero input contracts.

## Invariant and runtime representation

If each rounded product has magnitude at most2^limit units, the ordered accumulator afteri
steps has magnitude at most2*i*2^limit. The proof includes the actual exponent alignment,
signed sum, preliminary left shift and both rounding shifts. The guard requires limit≤252
and reduction count≤2^22. A separate logarithmic conversion derives the final exponent cap
from the actual reduction count. Empty reductions retain zero initialization.

The tensor trace uses actual runtime Val contents and constrains only in-bounds cells.
It does not replace arbitrary out-of-bounds data with fabricated floating values. The typed
state invariant is preserved at every update; checked scalar regions consume the actual
mapped accumulator. Pure per-cell filtering preserves the original order and isolates each
output cell. Projection geometry counts its reductions without enumerating the full grid.

Observed actual retained source: results1283/1308/1333 are **linalg.matmul**, each with its
own actual region and K=2048. The body computes product+accumulator; the final V generic
computes accumulator+product. Both templates are checked. Commutativity is proved only for
the pinned logical addition, including its canonical NaN behavior, not arbitrary FloatModels
or native NaN payloads. All accumulation order stays unchanged. Existing facts give cap127
to inputs1271/1273/1276/1277; the proposed first QKV dot bound is142.

## Current retained development probe and resume

`retained-development-v1.json` preserves the original failed request for cap142 on all three
Q/K/V results and on the reference. It is not a qualification. Actual source operand caps are
127/127 forQ,127/128 forK,127/123 forV, withK=2048; resulting caps are142/143/138. The expanded
reference does not yet infer these three reconstructed-result caps. Do not relabel that gap
as equal source/reference intermediate bounds under an approximate policy.

`retained-development-v2.json` runs the new scoped contract
`retained-exponent-contracts-v2.json`: all166SOURCE demands pass (the previous163plus Q/K/V
142/143/138), and all163REFERENCE demands pass. Counts remain1060/1087; both next refusals
remain1533; structural_match=true, complete_boundary_checked=false, whole_pass_verified=false.
It directly uses the retained actual source/target exports listed in retained-input-paths-v1.json.
This proves development checker integration, not fresh source-capture admission, final receipt
qualification, new reference-QKV bounds or full-pass coverage.

Immediate current command: four-case new API actual-pass pytest development batch, log
development-v1.log and basetemp out/build/lean-w8a8-attention/test-development-v1. Poll its
recorded session in worker-status; do not edit candidate Lean/Python or rebuild while it runs.
After it completes, correct any failures cleanly, run the full new suite including valid
mutations/strict replay, add meaningful tensor/native comparison and hostile-schema controls,
then a fresh transitive audit, freeze, separate-process retained/curated replays and identity
rechecks. Keep all earlier capsules and source deliveries untouched. Continue trig/softmax
after qualification rather than treating this numerical milestone as a stop.

The older forward plan below describes the now-implemented operation/prefix pieces and the
still-pending qualification/attention pieces; use this current section to resume.

## Further work

Implement the actual-operation/refined-fact rule: recompute named/generic maps, iterator kinds,
region, static operand/result types, zero initializer, product bounds and per-output reduction
count. Use the compiled tensor/cell proofs to discharge the rule. Bind named matmul and
batch_matmul through frozen LinalgNamed semantics, which uses the actual region. Check generic
operand segments too. Unknown metadata, wrong SSA/fastmath, unsafe product/count or nonzero
initializer must refuse. Do not modify a frozen or production file to add support.

Create an explicit-family book/prefix specialization using existing rotary fact/domain
semantics; older books require only ContextOk, so restrict them using the proved implication
from the explicit family. The new reduction rule requires the explicit family, not an
invented Agreement assumption. Reuse or add a generic parameterized prefix engine instead
of silently changing the sealed hardcoded old book. Integrate a NEW driver/API/receipt route,
actual QKV/source-reference demands, compiler-valid mutations, native comparisons and fresh
audits/freeze/replays. The last three named matrix results are not numerically accepted by
the current sealed checker yet; compiled component lemmas alone do not discharge that gap.

After those bounds, continue finite-angle sin/cos, attained row maxima with real negative
infinity masks, exp(-inf)=0, exp(0)=1, positive finite softmax denominators and final V reduction.
The current arithmetic family leaves transcendental fallback arbitrary; no universal trig
law follows from it. Preserve the quantizer1533 refusal and all earlier counterexamples until
the exact obligations are proved. Independent review, selected capture/domain binding,
native fidelity, semantic composition and full production enforcement remain separate.

All staging, commits, integration, pushes, shared Main/CLI/ledger, frozen/spec/lock,
production-pass and foreign-file edits remain held. Frontend owns the model adapter and
quant-ext owns its reciprocal/source-semantics bridge. No new agents or duplicate assignments.
