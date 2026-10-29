---
title: Bounded compiler pass verification profiles
kind: guide
status: current
owner: verification
last_verified: 2026-10-29
related: [lean_compiler_stack, lean_pass_integration_completion, qdq_macc_numerical_contract_review]
code_refs:
  - src/merlin/verify/compiler_profile.py
  - src/merlin/verify/compiler_coverage.py
  - src/merlin/verify/compiler_ir_source_profile.py
  - src/merlin/verify/lean/source_profile_export.py
  - src/merlin/verify/compiler_loop_source_profile.py
  - src/merlin/verify/compiler_buffer_source_profile.py
  - src/merlin/verify/compiler_configuration_source_profile.py
  - merlin/lean/MerlinVerify/Driver/SourceProfile.lean
  - src/merlin/verify/pass_validation.py
---

# Bounded compiler pass verification profiles

Selecting verification applies a bounded model to each actual pass boundary.
The original source and selected options determine eligibility. A known source
exclusion runs the ordinary compiler transformation and is listed as **unverified
at the end**. It receives no proof credit. A supported source must pass its actual
input/output check; a bad target, unsupported target, missing tool, timeout or
custody failure blocks compilation. Failed target checking never changes source
eligibility.

These profiles bound the current integration milestone. Extending a profile is
future engineering, not a requirement to finish the bounded gate. Keeping normal
compiler behavior outside a model does not approve or prove that behavior.
Ordinary compiler legality errors still propagate.

| Pass | Bounded source model | Principal exclusions and remaining premises |
| --- | --- | --- |
| Tensor DCE | Declared index64 function signatures and operations accepted by the existing Lean source-support rulebook; complete actual module boundary | Unsupported dialect/type/region rules; concrete MLIR primitive fidelity remains trusted. The separate memory route admits ordinary memory-context dead index constants whose intrinsic source constants and materialized dimensions fit signed index64, with complete successful return/world preservation rather than categorized error equality. A large view type without such a dimension read is not excluded merely for its size. |
| CSE | Same typed tensor rulebook at the actual native leading CSE | Changed explicit-memory CSE runs unverified. An actual unchanged capture needs no proof job and earns zero proof credit. Canonicalization and subsequent native passes have no proof from this gate. |
| Ordered unrolling | Existing scalar-word/index model, including qualified inline, bulk, tails, surrounding calls and supported nested depths | Literal nonnegative loop lowers, static fitting selected controls, positive strides, 2048 weighted syntax records and a source-derived expansion budget. Arbitrary source index arithmetic needs another model. |
| Generic-value unrolling | Separate arbitrary-Value model: exactly one `scf.for` in the entire module, at top level, zero carries, factor two, three immediate literal controls, odd trip count at least three, one opaque body event with IV/data capture, then an immediate captured return; other entries and prefix remain exact | More body events, carried data, effectful suffixes and any additional `scf.for` (including opaque regions/other entries) are excluded. Payloads are never encoded as integers. Whole-entry outcomes are relative to the explicit occurrence-aware projection and common partial action interpretation. |
| Ordered tiling | Existing scalar-word/index model with fitting static selected controls and supported nested depths; separately, the actual complete private-buffer chain. Generic singleton tiling uses exactly one top-level `scf.for` in the module, zero carries, three immediate literal controls, one opaque body event and an immediate captured return, positive stride, nonempty trips, fitting index64 controls and tile sizes 2–65536. | Generic unroll→tile composition, additional `scf.for` loops, carried data, effectful suffixes and broader generic loop bodies remain excluded. Qualified word unroll→tile uses one original-source admission and one actual whole-chain job, including its actual intermediate IR. |
| Buffer reuse | Static positive rank-one/rank-two private i8/i16/i32/i64 views, captured uniform initialization, explicit nonescaping lifetimes, at most 256KiB source and 5000 operations; optional tile sizes 2–65536 | Computed initializer prefixes, rank three, dynamic/floating storage, escaping private views and completion control are excluded. The relation concerns possible successful public observations under total private unobservable allocation, not allocation failure equivalence or a native allocator theorem. Rank-two qualification covers all four widths; retained rank-one compiler/native fixtures use i32. |
| Q/DQ lowering | Existing proposed source model and typed rulebook: signed i8/f32 per-tensor Q/DQ with scalar i32/i64 zero points; supported per-channel DQ with i32 zero points | Per-channel i64 zero points, other quantization formats/ranges and unsupported source rules run ordinary lowering with an unverified label. Source-model review, general PyTorch/IEEE equivalence and unconditional quantizer conversion success remain open. Optional pinned source qualification stays separate. |
| bf16/f16 MACC | Source-derived uniform half-format static contractions with the exact multiply/add body, supported projected maps and actual positive-zero initialization; f32 accumulation policy | Mixed formats, nonzero initialization and unrecognized contractions are excluded from this profile. Normal producer rejection remains rejection. Numerical admission remains draft; component theorems do not widen compiler acceptance or prove IEEE fidelity. |
| Configuration-write elision | Original selected ISA/effect contracts; straight-line integer instruction entries or Kernel calls; at most 4096 calls/256 fields, within original selection bounds | Instruction control/nested regions, dynamic decoded ISA operands and floating instruction metadata are excluded. Unknown Kernel calls remain exact barriers. Full modeled failure/state preservation assumes the declared common ISA/effect interpretation. |
| Configuration hoisting | Positive flat, sibling or recursive counted Kernel trees; 4096 Kernel statements (calls and loops)/256 fields/depth64; ordinary extents at most one million or signed64-positive extents with the original C64 consumer | Dynamic configuration dependencies, completion/staging and unmodeled expressions are excluded. All-index/all-environment instruction Models and exact source ownership remain explicit premises; finite examples do not discharge them. |
| Host dispatch fusion | One public entry, static unencoded byte-addressable integer tensors, pure identity-indexed pointwise stages, initialized inputs, unread destinations and supported direct private calls | One actual fused target region of 2–128 stages. Floating arithmetic, reductions, broader indexing/effects and multiple fused target regions remain outside acceptance. Public results are preserved under common total pure typed actions and unobservable successful logical allocation. |

Tensor/Q/DQ/MACC admission uses the installed source recognizers through
`merlin-lean source-profile tensor|macc source.json 64`. This command reads **one
source**, grants no boundary proof or policy approval, and reports its model
decision. The authoritative export must bind that exact source. A source-only
adapter classifies the native parser's `MLIRError` on the original assembly as a
known grammar exclusion, binding the exact input and tool bytes before returning
that decision. This lets ordinary xDSL preprocessing handle source forms outside
the native reader's grammar. It does not classify any proposed target. Unexpected
context/import/resource failures and failures during actual export or checking
remain blocking. The source adapter bounds complete IR to 4MiB and 5000
operations; parser dense-storage resource failures are not silently converted
into model exclusions.

The retained LLM-context proofs use actual recaptured floating/integer model
prefixes with an added selected SCF probe. The original captures had no selected
SCF loops. This qualifies preservation of those unchanged contexts around the
bounded loop rewrite; it does not demonstrate optimization of original model
loops or a performance improvement. More general LLM loop bodies and captures
outside the declared syntax/resource budgets remain unverified compiler cases.

Generic loop intake additionally bounds raw IR/query/dense storage to 4MiB,
weighted syntax and proof work to 65,536, block records to 4,096 and region depth
to 32. It excludes scalar NaN literals, floating-complex dense literals and
normalization-sensitive raw assembly. Runtime opaque float/complex payloads
remain values; supported literal forms retain infinities, signed zero and exact
ordinary dense NaN bytes. Word/private-memory literal intake likewise refuses
scalar NaN identity loss rather than merging different literals. Resource/tool
failures remain blocking; only a known original-source profile exclusion can
produce unverified coverage.

Loop, buffer and configuration predicates return a fixed profile identifier or
raise `SourceOutsideProfile(profile_id, code, reason)`. They use original inputs
only and call no producer or proposed-target matcher. Source admission itself
is not a semantic preservation proof. Standalone proof/replay APIs retain their
strict unsupported/refusal behavior.

## Interface and final coverage

A pass reaches its bounded stopping state when its source predicate, actual
input/output worker, consumer enforcement, source binding, replay where promised
and focused positive/negative qualification agree on the same declared domain.
Its exclusions remain visible below rather than becoming hidden proof premises.
Completed gates can be integrated independently; another pass's broader model
extension does not hold them back. Current qualification and publication status
is recorded in the [handoff](../../out/artifacts/handoff/pass-verif-integration-20261028/README.md).

Transformation selection remains independent of `--validate-passes` or
`PassValidation`. For example, selecting unroll validation does not enable
unrolling; `Options(unroll_factor=2)` selects the transformation. The supported
actual pair goes to the canonical worker. A source outside that loop profile
still receives ordinary unrolling and an unverified coverage entry.

After the outer gate joins all workers, `pass-validation/coverage.json` records
each observed boundary's profile, original/final SHA-256, status and proof credit.
Verified rows must bind the exact request and endpoint bytes of a job owned by
that invocation, with one matching accepted verdict. The inputs are reopened
after checking. Zero jobs plus an excluded boundary means **zero verified
boundaries**, never an accepted proof. Unchanged bytes also receive zero proof
credit. Failure to finalize a required coverage record blocks successful return.

The end log lists unverified boundaries and reasons. A successful compilation can
therefore contain both verified and unverified boundaries. An earlier checked
stage does not establish preservation through a later unverified stage. A buffer
whole-chain row does not assert separate intermediate-stage theorems. Coverage
records grant no live-completion or saved-receipt authority.

The [compiler stack guide](lean_compiler_stack.md) maps each interface to its
actual consumer and example. The [integration handoff](../../out/artifacts/handoff/pass-verif-integration-20261028/README.md)
distinguishes source-predicate tests, runtime protocol tests, kernel proofs,
fresh replay and finite native evidence at their recorded source cuts. LLVM IR
rewrites, backend translation, linking and executable fidelity remain outside
the formal boundary.

## Review points and future extensions

These are bounded proofs of the stated relations. A source exclusion identifies
missing formal coverage, not a failed proof of an admitted pair. Extending the
cases in the table requires new engineering and qualification; it does not
require weakening an existing theorem or admitting the case as an assumption.

| Review point | What remains explicit |
| --- | --- |
| Meaning of MLIR operations | The tensor rulebook's primitive semantics, purity and supported effects must match the intended operations. Native MLIR/xDSL parsing and Python projection are trusted bridges. |
| Generic loops | Payloads remain arbitrary values. The occurrence-aware control-role projection, its fitting-control conditions and common partial action interpretation are premises; this is not arbitrary raw external-call equivalence or concrete floating arithmetic. |
| Configuration | The selected ISA's instruction/effect/frame contracts and all-environment models need domain review. Small native scenarios exercise the implementation but do not prove those contracts universally. |
| Private storage and host fusion | The logical allocator is successful and unobservable. Buffer reuse's possible-success relation does not establish allocation-failure equivalence; neither gate proves a native allocator. Host fusion additionally assumes common total pure typed actions. |
| Numerical lowering | Q/DQ stays proposed-model, with general PyTorch/IEEE fidelity and unconditional conversion success open. MACC stays draft under the selected f32 accumulation policy pending independent review. Component proofs do not approve either contract or expand compiler admission. |
| Delivery policy | Review the admitted/excluded case lists and end-of-invocation coverage. A compilation with unverified boundaries is usable compiler output with partial formal coverage; it is not a whole-pipeline proof. |

General generic-loop bodies/carries/nesting, generic unroll→tile composition,
computed buffer initialization, rank-three/dynamic/floating private storage,
changed memory CSE and broader numerical/host graph domains are future model
extensions. LLVM IR optimization and executable/hardware correspondence remain
outside this effort's agreed compiler-pass boundary.
