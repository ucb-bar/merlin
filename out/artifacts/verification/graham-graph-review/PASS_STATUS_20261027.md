# Merlin Lean validator status — independent audit, 2026-10-27

Merlin has working Lean validators for restricted pass domains, historical
qualification for several bounded compositions, and fresh compilation evidence
for selected scheduling and empty-inline queries. General graph-pass compilation
gating is not ready. The new graph native gate is explicitly held in source;
the older opt-in tensor/preprocessing gate is a separate, bounded route.

On 2026-10-27, both checkouts were updated with `git pull --ff-only` and initially
reported already up to date. The graph reproduction/source baseline remains
**c62f81a8a11347ff53b60101fa75280e69d002a6**. Upstream `feat/pass-verification`
now contains the published dtype-admission fix at
**68f3646082d87ca2a4909117b1d84867d34fb53c**, whose remote ref was verified after
push. Its only changes are the two SMT binding modules and their focused tests;
no Lean source or graph proof profile changed. The audit branch's starting tip
was **10ccb46e26608f1708a0277cf2d424d28ab4cdae**.
`feat/pass-verification-verf` remains a separate auditing branch with review
document changes only. Frozen evidence, runtime contexts and historical
qualifications remain unchanged, and no workers were restarted.

This is a dated synthesis of source and retained evidence, not another Lean or
native execution. Earlier numerical/frontend records are attributed to their
own dates and contexts; their old active-work instructions are not current
worker status. The original graph reviews concern verification source
**58dbc7c00** and producer **4499eebb9**. Their findings must be read with the
[2026-10-26 coverage supplement](deliverable-1/UPDATE-20261026.md) and
[fresh reproduction](execution-a26-20261026-v1/README.md).

Here, **historically qualified** means the packet owner's bounded engineering
suite and replay were recorded as passing, with original source/tool/domain
identities. It does not mean every original packet is locally replayable,
independent human approval is complete, native semantics are proved, or the
ordinary compiler enforces that validator. **Fresh query pass** means the exact
selected query compiled and its requested theorem types/axioms were audited.
That alone does not establish semantic-mutation qualification or non-vacuity.

The eight registered common graph kinds have the following status. Variants and
mechanism inventory rows are not additional verified passes.

| Pass family | Working evidence and actual claim | Outstanding work / limits |
| --- | --- | --- |
| Scalar redundancy elimination: CSE/DCE | Existing scalar checker preserves complete modeled returns/world or error/terminal world. A retained CSE/DCE→LICM example and the static four-pass composition have historical qualification. | Internal calls are outside the scalar projection. Arbitrary producer variants and ordinary consumer enforcement are not established. The current-source closure includes new admission code; historical qualification is not transferred to it automatically. |
| Scalar loop-invariant motion: LICM | Same scalar outcome framework; historical qualification includes its bounded composition with CSE/DCE and exact-bulk unroll→segment. | The composed domain is static and no-tail. Runtime-stride producer forms are refused by the inspected projection. Complete primitive/effect/control interpretations and IR correspondence remain explicit. |
| Exact integer simplification | Sealed v2 checks complete before/after boundaries for wrapping add/sub/mul folding and supported neutral identities, including errors/terminal worlds, on canonical word inputs. An integer prefix composes with the four-pass suffix using an exact middle. | Successful opaque results must satisfy local canonical typing; native/poison fidelity is separate. Most historical case originals are unavailable here. The old producer's local BitVec receipts do not provide this whole-program claim. |
| Shape/view normalization | Standalone-v1 is historically qualified for identical memref-cast deletion and static dimensions in static ranked integer memrefs with identity layout/default space. It relates possible complete successful returns and whole worlds. | Failure categories and divergence are not equated. Descriptor fidelity and the Python match between actual IR and one blueprint's two modes remain trusted. Later shape→memory/caller-context extensions are unqualified and blocked in fresh proof reproduction. |
| Ordered loop segmentation | Existing trace/context proofs; historical bounded four-pass and reuse→two-initializer compositions. The four-pass bridge explicitly supplies its stronger error/world relation. | The leaf trace relation alone does not prove first-error/divergence equality. Wider nested/multiple boundaries and arbitrary composition remain outside the established subset. |
| Ordered loop unrolling | Exact bulk, with static bounds and no remainder, participates in the historically qualified four-pass chain. The new empty-inline case query passes freshly. | Four nonempty inline case queries and all five inline witnesses fail. Tail's support proof fails before its four queries. Nested/wider variants do not inherit exact-bulk qualification. A separate same-statement tail lemma repair has not repaired or qualified the full profile. |
| Buffer reuse | Existing public-outcome checker and historically qualified reuse→two-initializer segmentation, in a narrow private-memory domain with initialization/geometry checks. | Allocation is assumed total/private/unobservable. The retained 8-byte source / 16-byte candidate peak counterexample prevents an unconditional finite-memory claim. Wider widths/layouts/spaces and the new caller/entry extensions remain restricted or unqualified. |
| Sequential dependency scheduling | The fresh seven-module schedule profile and all three selected case queries pass exact audits: two changed cases and one identity. Its final relation covers modeled entry behavior, including errors and terminal state. Older scalar-return-vector and total-state scheduling results also exist, with different assumptions. | New profile semantic-mutation/replay qualification and real compiler enforcement are pending. These selected query successes do not qualify every scheduling strategy or transfer the older results across models. |

The [original coverage review](deliverable-1/coverage-review.md),
[variant matrix](deliverable-1/coverage.csv),
[assumption review](deliverable-3/assumption-review.md), and
[current source-scope table](execution-a26-20261026-v1/agent-1/SUPPORTED_SCOPE.csv)
give the precise domains and evidence provenance for those rows.

Fresh execution on a26 remains the following; there has been no subsequent
proof repair or rerun in this status update.

| Candidate profile | Exact observed result | Next concrete step |
| --- | --- | --- |
| Q schedule | 7/7 modules; 3/3 case queries compile and audit. | Qualify meaningful changed positives, compiler-valid semantic mutations and replay; bind the actual compiler consumer. |
| Q inline | 23/23 modules; empty case query passes; four nonempty case queries fail at `Query.lean:51`. All five separate witness files fail, including the empty witness's concrete evaluations. | Repair Nat/Int continuation rewriting and witness evaluations in separately identified source successors, then recheck all cases. |
| Q tail | 24 dependencies compile; `GraphStatePipelineTail.lean:33` fails in `loop_append`. Four case queries are unexecuted. | Incorporate the separately checked same-statement list-case repair into a new profile, then compile the entire module and four queries. |
| W caller context | 40 dependencies compile; `GraphLoopsShapeAfterRename.lean:115` and later renaming obligations fail. Fourteen queries are unexecuted. | Repair argument/environment/binding renaming obligations in a source successor and continue through the remaining dependency closure and queries. |
| W entries | Independently reaches the same support-module failure after 40 dependencies; twelve queries are unexecuted. | Recheck its independent closure, complete entry roster and dependent queries after the shared repair. |

Across 38 case queries, four compile and pass exact root audits, four fail and
30 are unexecuted. All five witness files fail. The four accepted queries have
76 requested root-audit occurrences using only `propext`, `Classical.choice`
and `Quot.sound`; these are not 76 distinct verified passes. An unexecuted
expected-refusal case is not a successful refusal. Exact failures, statements
and commands are retained in the
[case roster](execution-a26-20261026-v1/agent-3/CASE_ROSTER.csv) and
[reproduction summary](execution-a26-20261026-v1/agent-3/REPRODUCTION_SUMMARY.json).

Other Merlin mechanisms need separate status; the eight-kind registry does not
cover all compiler changes. The following rows summarize retained records,
without independently rerunning their historical suites in this refresh.

| Area | Current evidence | Remaining concern / WIP |
| --- | --- | --- |
| General tensor/MLIR checker and preprocessing | Registered proofs cover bounded integer folding/neutral identities, supported cast chains, comparison/select, reshape aliases, selected named-op generalization, bounded motion/fusion and dead producers. An opt-in strict preprocessing handoff and optional native-prefix gate exist. | Whole-model builders are ungated by default. Index32 programs are refused by the bounded gate because of a retained native/model mismatch. General layouts/motion, model fidelity, bufferization and the remaining native suffix are outside the full proof claim. |
| Declared-call / fan-out fusion | Declared-call v1 has historical qualification for two actual typed boundaries; observer preservation is derived from per-instruction, packing, ABI, Action and frame contracts. | Native implementation truth remains a premise. This is observer preservation under total/no-fault contracts, not general error/world preservation. The stronger wrapper remains partial/unqualified after an unexplained pre-oracle host-gate failure; source-operational extensions do not inherit v1 qualification. |
| Configuration hoisting and elimination | Hoisting has a narrow flat-loop overwrite-based theorem/report. Elision has proof/checker source and counterexamples. | Hoisting's raw historical bundle is missing. Elision remains unqualified after Lean SIGSEGV, with zero qualified originals/replays. Setter/barrier deletion needs its own physical-effect contracts. |
| Wider scheduling, copies, prefetch, storage and source rewrites | Local geometry/effect/trace facts and behavioral tests exist. The audit inventories dense/conv schedules, copy forwarding/coalescing/pipelines, spills, physical fusion, recomputation, row rewrites, ABI changes and final segmentation. | Complete actual-boundary Lean qualification is not established for those inspected producer variants. A legal plan, capacity bound or callback interface does not prove the realized transformation or asynchronous behavior. |
| Connected-region planning and resource geometry | Narrow plan/partition and fit/alignment/lifetime/capacity results exist. | They do not prove native fused instructions, complete byte-dataflow preservation, finite allocation, or timing/overlap correctness. Broader realization and raw historical replay evidence remain incomplete. |
| Qinner lifting | Concrete byte-decoder proofs establish supported typing, initialization and exact bits, and specialize complete-program correspondence to archive-bound source intent. | Tagged source empties mean archive tensors in this contract; this is not literal ordinary `tensor.empty` equivalence. Archive authenticity, host/runtime agreement and actual delivery remain assumptions. Retained model examples with zero lifts do not establish changed model coverage. |
| Boolean→f32 repair | A complete-program intent theorem proves false/+0, true/+1 and preserved poison, with nested/call/suffix support. | Independent admission remains pending; other float formats refuse. The claim is repair intent, not equality to the original signed-cast behavior. |
| TorchAO affine materialization | Exact ordered scalar-expression theorem and complete correspondence to source-driven materialization. | Captured opaque-call execution refinement is not proved; tensor-reference/layout/poison/parameter and concrete reduction, zero-point and clamp/round/conversion bridges remain open. Receipts explicitly retain `source_call_refinement_proved=false`. |
| Quant-ext lowering | Ordered per-tensor/channel contracts and retained same-host independent receipt replays support bounded materialization/source-adapter claims. | Draft policy/native/capture premises remain. The common-model SmolVLA complete-output-comparison failure is unexplained; successful replay of other subjects does not qualify that release. i64 F12 remains refused; retained TinyLlama has zero channel sites. |
| Scalar-square, over-rank matmul, bf16 accumulation | Complete correspondence to selected multiplication/repair/f32-accumulation references and modular policy proofs exist. | Native pow-versus-multiply counterexamples refute unconditional exact-bit scalar-square equivalence. Over-rank matmul repairs invalid source IR. bf16 changes the accumulation policy; numerical/intent review and native/initialization assumptions remain. |
| W8A8 | Extensive conditional helper, contraction, domain and receipt proofs/tests are retained. | Full pass verification, policy review and primitive-model-law verification remain false. Numerical definedness and concrete law admission are unfinished; conditional bounded receipts return `abstracted`. Historical real-model receipt failure is not resolved by graph-query successes. |
| Separate linalg/interface SMT lowering | A retained example and source review exist. The dtype-binding defect is fixed and published upstream in implementation commit `68f364608`; 48 focused regressions pass. | Imported `frontends.linalg_patterns` is still absent from the receipt semantic-source digest. Output ABI ordering review is unfinished, and no matching solver receipt was located for the retained lowering pair. These are source/helper findings, not demonstrated complete-solver or Lean false acceptance. |

The tensor route's exact assumptions and gate scope are in the upstream
[assumption ledger](https://github.com/ucb-bar/merlin/blob/c62f81a8a11347ff53b60101fa75280e69d002a6/merlin/lean/LEDGER.md).
Wider graph rows come from the audit's
[coverage matrix](deliverable-1/coverage.csv).
Frontend/numerical rows are bounded by the upstream
[planner records](https://github.com/ucb-bar/merlin/blob/c62f81a8a11347ff53b60101fa75280e69d002a6/out/artifacts/verification/merlin-passes/progress/planner/README.md),
[common quant-ext replay](https://github.com/ucb-bar/merlin/blob/c62f81a8a11347ff53b60101fa75280e69d002a6/out/artifacts/verification/merlin-passes/progress/planner/eight-day-paper/integration/quant-ext-common-independent-replay-v2/README.md)
and [channel replay](https://github.com/ucb-bar/merlin/blob/c62f81a8a11347ff53b60101fa75280e69d002a6/out/artifacts/verification/merlin-passes/progress/planner/eight-day-paper/integration/quant-ext-channel-independent-replay-v1/README.md).
The separate SMT work's exact source-only observations and empty fix patch are
in [W's infrastructure handoff](https://github.com/ucb-bar/merlin/blob/c62f81a8a11347ff53b60101fa75280e69d002a6/out/artifacts/verification/graph-shape-normalization/graph-infra-binding-v1/HANDOFF.md).

The dtype issue was an input-model binding defect at original upstream
`c62f81a8a`. In
[`refine._bind_leaves`](https://github.com/ucb-bar/merlin/blob/c62f81a8a11347ff53b60101fa75280e69d002a6/src/merlin/verify/refine.py#L299),
the implementation checked input count and shape, then reused the interface's
symbolic tensor without comparing dtype or width. In
[`CommandBufferEncoder.declare_leaves`](https://github.com/ucb-bar/merlin/blob/c62f81a8a11347ff53b60101fa75280e69d002a6/src/merlin/verify/cb_semantics.py#L139),
a supplied shared tensor was installed before the normal dtype parser ran.
The retained helper example uses shape 2×3 and an 8-bit shared tensor: `i8`,
`i16`, `f32` and `bogus` declarations all receive that same tensor. Only `i8`
is the valid control. A solver could therefore receive a model with different
input semantics from the declared artifact unless a later check refuses; a
complete solver false acceptance or miscompilation has not been demonstrated.
Published implementation commit
[68f3646082d87ca2a4909117b1d84867d34fb53c](https://github.com/ucb-bar/merlin/commit/68f3646082d87ca2a4909117b1d84867d34fb53c)
on `feat/pass-verification`
now parses supported integer dtypes and checks width in `_bind_leaves` before
sharing symbols. The encoder validates declared rank/dtype first and checks
shared shape/width before reuse. Floats of the same bit width remain refused.
The audit branch retains only this result; it does not merge that source commit.
The remote implementation branch was independently verified at that exact commit.

The new `merlin/tests/ir/test_refinement_input_binding.py` exercises the actual
binding/encoder helpers and public compilation admission. On original
`c62f81a8a`, it reproduces **30 failures and 18 passing controls**. With the fix,
**all 48 tests pass, zero fail and zero skip**. Cases cover supported integer
widths, input/weight controls, narrower/wider declarations, same-width floats,
invalid/missing-type values, shared shape mismatches and refusal before target
encoding or solver invocation. Tests use a source-encoder test double for the
public admission cases; no solver, translator or Lean process is invoked.
The fresh Python environment uses Python 3.12, xDSL 0.68.0 and pytest 9.1.1.
Its missing optional pytest-timeout plugin causes two configuration warnings;
each whole test command is separately bounded to 120 seconds.

The regression command is `python -m pytest
merlin/tests/ir/test_refinement_input_binding.py -q`, with a project-local temp
root and plugin autoload disabled. Raw red/green logs, JUnit results, commands,
exit codes and source hashes remain under the primary checkout's
`out/build/verification-dtype-admission-20261027-v1/`; they are not a portable
solver-qualification packet. The committed source/test bytes match the green
run's hashes. Documentation, regex and target-name checks pass. The full
structure gate retains the pre-existing unchanged `zephyr_model.py`
3557-versus-3537-line failure; its other 18 subchecks pass. The dtype fix does
not address the separate imported-source digest or output-ABI review gaps.

The compiler-gate concerns are concrete:

1. **Public graph checking is held.** Both `check` and `revalidate` in the
   upstream [native gate](https://github.com/ucb-bar/merlin/blob/c62f81a8a11347ff53b60101fa75280e69d002a6/src/merlin/verify/lean_graph_pipeline_gate_native.py)
   unconditionally raise `GateUnavailable`. Fresh independent SDK diagnostics
   and query compilation do not change that code or resolve the old runtime
   incident. A newly reviewed context/integration is still required.
2. **Legacy receipts cannot certify a whole transformation.** The historical
   producer's shape identity and integer local equations do not name the whole
   changed program. Corrective producer patches are retained, but canonical
   producer destinations are absent from the audited primary tree. The
   [integration readiness report](execution-a26-20261026-v1/INTEGRATION_SOURCE_READINESS.json)
   finds 31 of 66 selected prerequisite paths absent and two different. Byte
   recovery of 29 proposed payloads from a patch is not applied integration.
3. **Actual IR and output binding remain trusted work.** Admission improved:
   32 independent controls pass, including indirect-call/resource cases. The
   original 81-test producer replay is unavailable. Parser/projection fidelity,
   full-tree matching, descriptor/control registries, ordered complete entry
   rosters and actual compiler buffers need review and end-to-end checks.
4. **Relations and resource assumptions must match the consumer.** Error/world
   equality, possible successful public outcomes, observer-only preservation
   and numerical repair policies cannot be composed without explicit bridges.
   Keep the finite-allocation and numerical counterexamples in the admission
   contract; hashes alone cannot provide such bridges.
5. **New profiles are not qualified by partial success.** Repair the exact
   failed queries/witnesses and support proofs in successors. Then run genuine
   changed positives, compiler-valid negative mutations and exact replay, with
   source/tool/axiom/actual-buffer associations. Most historical raw case
   packets are not present here, so broad independent replay remains limited.
6. **The actual compiler must enforce the checked boundary.** The current
   [service selection](https://github.com/ucb-bar/merlin/blob/c62f81a8a11347ff53b60101fa75280e69d002a6/src/merlin/verify/lean_graph_pipeline_gate_service.py)
   admits only explicitly selected CSE/DCE and LICM and requires same-live
   completed receipts, exact adjacent buffer joins and the final delivered
   output. Saved JSON, identity/local proofs, mock completion and plan-only
   results provide no whole-program authority. Missing proofs, infrastructure
   failures and unsupported input must never be labeled verified.

The recommended integration sequence is bounded shadow checking for CSE/DCE
and LICM, followed by qualification of their actual ordinary compiler path and
then a restricted hard gate after the runtime/context hold is properly
resolved. Supported-input admission, complete changed-boundary coverage and
output consumption need explicit tests. For unsupported optimization inputs,
choose and document retaining the original program or refusing compilation.
Wider scheduling, shape/memory composition and numerical policies need their
own subsequent domains and qualification; this audit does not authorize gate
activation or claim full-model/compiler correctness.

For the next audit refresh, pull upstream separately, record its exact commit,
review changed source and handoffs, and append newly attributed execution
evidence. Preserve historical manifests and failed contexts. The
[review index](README.md) and original deliverables remain the evidence trail.
