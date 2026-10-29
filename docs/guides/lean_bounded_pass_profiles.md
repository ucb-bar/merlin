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
| Tensor DCE | Declared index64 function signatures and operations accepted by the existing Lean source-support rulebook; complete actual module boundary | Unsupported dialect/type/region rules; concrete MLIR primitive fidelity remains trusted. The separate memory route admits ordinary memory-context dead index constants, with complete successful return/world preservation rather than categorized error equality. |
| CSE | Same typed tensor rulebook at the actual native leading CSE | Changed explicit-memory CSE runs unverified. An actual unchanged capture needs no proof job and earns zero proof credit. Canonicalization and subsequent native passes have no proof from this gate. |
| Ordered unrolling | Existing scalar-word/index model, including qualified inline, bulk, tails, surrounding calls and supported nested depths | Literal nonnegative loop lowers, static fitting selected controls, positive strides, 2048 weighted syntax records and a source-derived expansion budget. Arbitrary source index arithmetic needs another model. |
| Generic-value unrolling | Separate arbitrary-Value model: one selected top-level zero-carry loop, factor two, static nonnegative controls, odd trip count at least three, one opaque body event with IV/data capture; retained prefix/suffix and other entries | More body events, carried data and broader selected nesting remain outside this profile. Payloads are never encoded as integers. Whole-entry outcomes are relative to the explicit occurrence-aware projection and common partial action interpretation. |
| Ordered tiling | Existing scalar-word/index model with fitting static selected controls and supported nested depths; separately, the actual complete private-buffer chain | Generic-value tiling and generic unroll→tile composition are not enabled by a private smoke proof. Qualified word unroll→tile uses one original-source admission and one actual whole-chain job, including its actual intermediate IR. |
| Buffer reuse | Static positive rank-one/rank-two private integer views, captured uniform initialization, explicit nonescaping lifetimes, at most 256KiB source and 5000 operations; optional tile sizes 2–65536 | Computed initializer prefixes, rank three, dynamic/floating storage, escaping private views and completion control are excluded. The relation concerns possible successful public observations under total private unobservable allocation, not allocation failure equivalence or a native allocator theorem. |
| Q/DQ lowering | Existing proposed source model and typed rulebook: signed i8/f32 per-tensor Q/DQ with scalar i32/i64 zero points; supported per-channel DQ with i32 zero points | Per-channel i64 zero points, other quantization formats/ranges and unsupported source rules run ordinary lowering with an unverified label. Source-model review, general PyTorch/IEEE equivalence and unconditional quantizer conversion success remain open. Optional pinned source qualification stays separate. |
| bf16/f16 MACC | Source-derived uniform half-format static contractions with the exact multiply/add body, supported projected maps and actual positive-zero initialization; f32 accumulation policy | Mixed formats, nonzero initialization and unrecognized contractions are excluded from this profile. Normal producer rejection remains rejection. Numerical admission remains draft; component theorems do not widen compiler acceptance or prove IEEE fidelity. |
| Configuration-write elision | Original selected ISA/effect contracts; straight-line integer instruction entries or Kernel calls; at most 4096 calls/256 fields, within original selection bounds | Instruction control/nested regions, dynamic decoded ISA operands and floating instruction metadata are excluded. Unknown Kernel calls remain exact barriers. Full modeled failure/state preservation assumes the declared common ISA/effect interpretation. |
| Configuration hoisting | Positive flat, sibling or recursive counted Kernel trees; 4096 calls/256 fields/depth64; ordinary extents at most one million or signed64-positive extents with the original C64 consumer | Dynamic configuration dependencies, completion/staging and unmodeled expressions are excluded. All-index/all-environment instruction Models and exact source ownership remain explicit premises; finite examples do not discharge them. |
| Host dispatch fusion | One public entry, static unencoded byte-addressable integer tensors, pure identity-indexed pointwise stages, initialized inputs, unread destinations and supported direct private calls | One actual fused target region of 2–128 stages. Floating arithmetic, reductions, broader indexing/effects and multiple fused target regions remain outside acceptance. Public results are preserved under common total pure typed actions and unobservable successful logical allocation. |

Tensor/Q/DQ/MACC admission uses the installed source recognizers through
`merlin-lean source-profile tensor|macc source.json 64`. This command reads **one
source**, grants no boundary proof or policy approval, and reports its model
decision. The authoritative export must bind that exact source. Unexpected
exporter/checker failures remain blocking. The source adapter bounds complete
IR to 4MiB and 5000 operations; parser dense-storage resource failures are not
silently converted into model exclusions.

The retained LLM-context proofs use actual recaptured floating/integer model
prefixes with an added selected SCF probe. The original captures had no selected
SCF loops. This qualifies preservation of those unchanged contexts around the
bounded loop rewrite; it does not demonstrate optimization of original model
loops or a performance improvement. More general LLM loop bodies and captures
outside the declared syntax/resource budgets remain unverified compiler cases.

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
