---
title: Q/DQ and MACC numerical contract review
kind: design
status: draft
owner: verification
last_verified: 2026-10-29
code_refs:
  - merlin/lean/MerlinVerify/Rules/QuantizeDefinedness.lean
  - merlin/lean/MerlinVerify/Rules/Bf16F32AccInitialized.lean
  - src/merlin/llvmlower/passes_xdsl.py
  - src/merlin/verify/bf16_policy_gate.py
---

# Numerical contract strengthening and separate review proposals

The selected MACC behavior remains f32 accumulation. Equality to the original
half-precision recurrence is not the intended contract. Independent source-model
and numerical-policy approval remains pending. No frozen semantics, specification,
statement pin, lock or approval record is changed by this work.

## Implemented additive results

| Result | Exact domain and conclusion | Compiler status |
| --- | --- | --- |
| `QuantizeDefinedness.scalar_defined_iff` | For actual scalar values, nonpoison quantization is equivalent to success of the exact final signed conversion. The reciprocal, rounding, zero-point addition and clamp order are unchanged. | A theorem about the existing proposed signed-i8 contract. |
| `QuantizeDefinedness.result_defined`, `run_defined` | Actual scalar parameters exist; every observed activation exists; the final conversion succeeds for each observed element. Runtime types/shapes and an i32/i64 zero point give successful source execution with nonpoison output. | Conditional input-domain theorem; no new compile-time definedness verdict. |
| `QuantizeDefinedness.target_result_defined` | The same domain plus the existing complete site checks gives successful execution and a defined result at the actual six-operation lowering boundary. | Reuses `QuantizeSite.target_sound`; this does not establish a whole-model domain. |
| `QuantizeDefinedness.initialized_counterexample` | A permitted FloatModel returns `none` for every integer conversion. Typed, initialized rank-zero inputs, including scale +1, still produce poison. | Preserved negative evidence against unconditional definedness. |
| `Bf16F32AccInitialized.result_defined` | Initialized A/B and checked mapped access bounds give a nonpoison result for the existing ordered zero-initialized f32 contract, for every FloatModel. | Additive contract theorem; current gate approval remains draft. |
| `Bf16F32AccInitialized.accumulateFrom_sound` | The actual accumulating generic computes the ordered initialized contract, for independent operand formats, arbitrary per-cell f32 initialization and checked geometry. Poison is retained. | A reusable component theorem; broader source recognition and producer acceptance are not enabled. |
| `zero_specialization`, `accumulateFrom_cell`, `empty_trace`, `resultFrom_defined` | The old zero contract is exactly preserved; each cell sees its ordered writes; empty traces retain the initializer before final conversion; initialized inputs with checked bounds give defined output. | No reassociation, floating identity or assumption about extending half zero. |

The MACC producer and host denominator now distinguish plain sums/products from
multiply-add reductions. The former are left unchanged. Named half matmuls and
generic half reductions containing both multiply and add remain candidates,
including malformed operand segments or bodies. Exact eligibility still refuses
unsupported candidates before any mutation. This is a conservative superset of
the existing Lean candidate predicate, not a new numerical policy. Complete output
matching still rejects changes to an unrelated reduction.

## Assumptions and exclusions shared by these results

- Lean's kernel and the three standard logical axioms remain the proof base.
  Exporter/parser fidelity, the installed checker/auditor execution and source/
  tool binding remain trusted implementation boundaries; logs are not live
  completion authority.
- Float operations are keyed, law-free functions. Floating exceptions, flags,
  rounding environments, NaN payloads and a concrete IEEE implementation are not
  provided by these theorems. `toInt` may fail; the Q domain explicitly checks it.
- Tensor values use the existing canonical, in-bounds observation semantics.
  Structured geometry/access bounds must be checked. Reduction order is the
  existing lexicographic model (ledger A5), not a proof that arbitrary upstream
  transformations or framework kernels choose that order.
- Q/DQ value purity remains ledger A9. Successful modeled execution does not
  model every Python exception. Current capture qualification binds the exact
  producer checkout `905d2c918f839596c9d1198cbdeb42d4c12a325b` and PyTorch
  `2.10.0+cpu`; the selected wheel SHA-256 is
  `ee40b8a4b4b2cf0670c6fd4f35a7ef23871af956fecb238fbf5da15a72650b1d`.
  Dependency/interpreter and original graph/parameter byte bindings are checked
  separately from source-specification approval.
- The installed compiler gates retain their exact static tensor, format, map,
  attribute and source-body restrictions. Mathematical component parameters do
  not admit new compiler cases. Local initialized-input qualification does not
  prove a whole captured model succeeds or a site is reached at runtime.
- W8A8 policy correctness, upstream MLIR/LLVM transformations, physical aliasing,
  native allocation, object/ISA execution and hardware correctness are outside
  this Q/DQ/MACC result. Bounded native references supply evidence, not those
  universal fidelity theorems. See the full [assumption ledger](../../merlin/lean/LEDGER.md).

## Proposal Q1: discharging quantizer definedness for a concrete IEEE instance

The law-free model cannot prove conversion success from scale positivity alone.
`Domain` exposes the exact final `fptosi` obligation instead of assuming that an
operation named clamp is sufficient. The generic receipt must continue reporting
unrestricted quantizer definedness as unestablished.

A separate concrete-instance proof could discharge this domain for finite f32
activations, finite positive scales whose computed f32 reciprocal is finite,
initialized i32/i64 zero points, the declared clamp bounds and an agreed rounding/
exception environment. Positive finite scale alone is insufficient: a tiny scale
can overflow the reciprocal, and multiplying zero by infinity can produce NaN.
These are proposed sufficient conditions, not an IEEE theorem supplied here.

Review must bind the source overload, scalar extraction/conversion precision,
rounding, NaNs, infinities, subnormals, zero scales, conversion failures, exceptions,
flags and target execution. In particular, PyTorch's original scalar scale may be
binary64 while capture materializes f32. The retained 50-versus-49 counterexample
must remain refused unless that distinction is explicitly represented.

The source reference is [PyTorch v2.10.0 `_decomposed.py`](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/ao/quantization/fx/_decomposed.py).
Its per-tensor recipe and tensor-overload extraction must be reviewed separately
from the [MLIR conversion semantics](https://mlir.llvm.org/docs/Dialects/ArithOps/#arithfptosi-arithfptosiop).
Current tool/capture pins and bounded observations remain recorded in the
[qualification handoff](../../out/artifacts/handoff/quantization-lean-qualification-20261029/README.md).
No end-to-end PyTorch/IEEE equivalence follows from the abstract lowering theorem.

## Proposal Q2: more Q/DQ formats and granularities

The installed quantizer specification fixes signed i8, f32 arithmetic, per-tensor
parameters and full range [-128,127]. A u8/i16/f16/bf16/per-channel/group extension
requires a separately reviewed source specification and new checked lowering
rules; changing a type allowlist cannot establish it. Each extension needs its
own source overload/version, parameter shape/indexing, conversion sign/width,
range/axis validation, poison/exception domain, native references, mutations and
actual before/after receipts on both routes. Existing supported per-channel
dequantization is not evidence for per-channel quantization.

No such frozen-interface patch is installed. Keep the current supported domain
until a concrete extension and its statement/pinning changes are independently
reviewed; do not regenerate `SPEC_LOCK` merely to make a build pass.

## Proposal M1: initialized and mixed-format compiler profile

The new component contract is ready for a separately named profile. To enable it:

1. Specify source-body templates before implementation. A mixed f16/bf16 MLIR
   multiply needs explicit conversions to a common arithmetic format. Admit only
   the reviewed conversion/multiply/add/yield topology, with no discarded source
   operation, hidden cast or extra numerical flag.
2. Capture the actual destination C0 and extend each initial cell to f32 once.
   State this conversion explicitly. Do not replace the existing literal-f32-zero
   recipe with `ext(+0_half)`: these differ in a law-free FloatModel. Preserve the
   existing profile and its exact `zero_specialization` theorem.
3. Build a source-derived whole-module reference for the new profile. Bind A, B
   and C0 with alias-aware SSA packing, including shared operands/initializers and
   dependent sites. Extend actual-fragment qualification, not just report fields.
4. Reuse `accumulateFrom_sound`, existing scalar-body and final-truncation proofs.
   Derive maps, extents, access bounds, iteration order and output coverage from
   actual IR. Incompatible/dynamic/encoded maps continue refusing until proved.
5. Require complete output matching, actual public compiler delivery, fresh replay,
   nonzero/negative-zero/empty reductions, mixed-format native references and
   genuine operand/initializer/map/order/rounding mutations before enabling it.

This work does not add that compiler profile. Its remaining source/reference/SSA
integration is engineering work, not a pending approval flag. Native fusion also
remains separate: [`contract`](https://llvm.org/docs/LangRef.html#fast-math-flags)
permits changed rounding. The existing uniform-fusion/injective-extension lemma
does not cover arbitrary per-site fusion, NaN payloads or every target FP mode.

## Paper wording available now

The central compiler-validation claim can cover complete entries and actual
enforcement, while retaining each check's supported domain and model contracts:

> Merlin validates actual compiler boundaries using checkers whose soundness is
> proved in Lean. For supported loop and configuration transformations, the proofs
> cover complete entries and preserve modeled outcomes and state, including
> failures and surrounding effects. Validation is enforced at the compiler
> boundary, and products are released only after all requested checks succeed.

The numerical claims remain more specific:

> For the supported Q/DQ domain, translation validation checks actual lowering
> boundaries against a version-qualified proposed source model. Quantizer
> definedness is proved conditionally on success of the exact final conversion,
> and this condition transfers to the checked lowering. For supported bf16/f16
> contractions, validation checks conformance to an explicit ordered
> f32-accumulation reference. Additive component theorems cover initialized f32
> accumulators and independent operand formats; the current compiler profile
> retains its narrower source eligibility. Independent numerical-policy review,
> concrete IEEE/framework fidelity and downstream MLIR/LLVM/hardware behavior
> remain separate obligations.

Do not claim approved source semantics, universal quantizer definedness, original
half-precision equivalence, broader compiler coverage from component lemmas, or
full imported-project auditing from a report that audits only consumed roots.

Buffer reuse separately preserves successful public observations under the
explicit allocator model; it does not establish allocation-failure equivalence
or native allocator correctness. Conditional quantizer definedness does not
establish general PyTorch/IEEE equivalence. Initialized and mixed-format MACC
component proofs do not enable those source cases in the compiler gate.
