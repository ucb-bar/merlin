# Quantization Lean proof handoff — 2026-10-28

## Current integration checkpoint

Continue from `feat/pass-verif-integration`, which includes the inherited
quantization histories and the configuration updates from
`feat/pass-verification-dce-cse` at `018c3795d3`. See the
[current integration handoff](../pass-verif-integration-20261028/README.md).
This session added graph/compiler-gate integration, not new quantization proofs.

The current compiler selector has five checks: tensor DCE, CSE, ordered
unrolling, ordered segmentation and private buffer reuse. `all` selects those
five; it does not select Q/DQ expansion, W8A8 or bf16/f16 policy proofs.
Configuration elision and hoisting retain their separate explicit APIs.

Source Q/DQ contracts and lowering proofs are inherited, with their original
profile exclusions and historical evidence below. The W8A8 source-domain,
calibration and accumulation side proofs remain review candidates; they do not
approve the numerical policy or establish whole-model numerical execution.
The bf16/f16 f32-accumulation contract also remains a review candidate with
independent admission outstanding. No numerical admission or frozen semantic
assumption was changed by this checkpoint.

For the next quantization owner: start with
[the assumption ledger](../../../../merlin/lean/LEDGER.md),
[the design's quantization and policy sections](../../../../docs/design/merlin_lean_verification.md),
and the remaining review tasks below. Requalify the intended numerical path on
this exact branch before enabling a new compiler gate. Graph proof success and
a clean theorem audit are not human source-contract approval.

The branch/source descriptions below retain their original checkpoint scope.

Prepared while completing `integrate/lean-dce-cse-gate`. Quantization work is
parked for a separate agent. No quantization contract, arithmetic, admitted
datatype, source semantics or theorem statement is changed by this deliverable.
The shared alpha-checker optimization only adds imports of a proved equal
implementation to existing callers, including the dequantizer body checker.

## Source branches and boundaries

* `feat/pass-verification`, commit `65e44afbf4dbb6c7270e8936c4ef76b579a78492`,
  contains the completed source Q/DQ expansion proofs, compiler corrections,
  quantization gates and regression tests. This is the published reference.
* `feat/lean-prune-gate` and `feat/lean-quant-contract` currently point to
  `a21388cd3`, the earlier combined compiler integration worktree state.
* `feat/pass-verification-dce-cse` (converged with
  `feat/megaphase-feedback-integration`) now retains the complete original histories of
  `feat/pass-verification` through `c3ad8e10f` and the independent review branch
  `feat/pass-verification-verf` through `4b1d71a20`. It inherits the source Q/DQ
  proofs and numerical compiler corrections. Quantization and policy compiler
  gate registrations are disabled; standalone proof/review tooling is retained.
  Frozen tensor semantics and SPEC_LOCK still match the published reference.
  The only production compiler gate selection is
  `prune_dead_pure_tensor_ops,cse`; `all` selects exactly those two. No WIP policy
  admission or quantization proof is required by that selector.

The ancestry repair restores existing upstream work; it does not advance
quantization proofs or broaden their profile. Requalify standalone numerical proofs against
the integrated checkout before proposing a separate numerical gate. Earlier instructions to port
isolated numerical fixes applied to the former main-based import and are
superseded by the restored upstream ancestry.

## Implemented proof scope

`quant_ext.quantize_per_tensor`: static f32 activation tensors, scalar f32
scales, scalar i32/i64 zero points and full-range signed-i8 output. The contract
retains binary64 reciprocal formation, f32 conversion and multiplication,
round-even, zero-point conversion/addition, lower-then-upper clamp and signed
conversion. It preserves poison and assumes no floating algebra. Registered
C1 sites check all six target operations, geometry, captures and temporary SSA
values. Exact typed collapse also handles supported nested generic bodies.

`quant_ext.dequantize_per_tensor`: signed i8 to f32, scalar i32/i64 zero point,
with convert-before-subtract ordering. The captured [-128,127] and [-127,127]
ranges are metadata, not clamps. `dequantize_per_channel` has the separate
i8/i32/f32 promote-subtract-convert contract. Per-channel i64 formal coverage,
f64 scales, unsigned storage and general other granularities remain excluded.

Primary files: `merlin/lean/MerlinVerify/Spec/{Quantize,Dequant}.lean`,
`Sem/Ops/{Quantize,Dequant}.lean`, `Rules/Quantize{Scalar,Tensor,Site,Rule}.lean`,
`Check/QuantizeCollapse.lean`, `Check/RulesQuantize.lean`, and `SPEC_LOCK`.

## Published evidence and important commits

* `b9dd8bc1e`: i64 per-tensor dequantization support/proofs.
* `64a4cca34`, `e0337d2b8`: ordered quantizer components and complete lowering.
* `7e4a401c0`: compiler reciprocal-precision correction, promoted per-channel
  i32/i64 subtraction and authoritative named-body preservation. The i64
  compiler correction does not register a per-channel i64 Lean contract.
* `560d33309`: composition of exact CSE with modeled source quantizers.
* `65e44afbf`: combined gate integration and reproducible export driver.

The published tests check 230 retained quantizers (35 forms) and 627 retained
per-tensor dequantizers (83 forms), with geometry/width, nested, composition and
mutation cases. The prior combined publication qualification records 118 gate,
lowering, fuzz and supervision tests plus 3 Spike tests. Those counts belong to
that source version; they are not fresh qualification of the ancestry repair.

Full prior qualification is retained relative to the original pass-verification
checkout (local evidence, not guaranteed to exist in a new clone):

`out/artifacts/verification/v1/verification_v1_20261028T135131Z_65e44af/qualification/summary.json`

It indexes 12 model/route exports and 25 accepted checks, with identical gated
and ungated LLVM output. Six exports concern source-quantizer capsules:
`SY_micro_model`, `M2_microvit_gemmini`, and `GX0_interop_rvv_lane`, each through
both xDSL and textual routes. Raw exports and receipts are local and byte-bound;
rebuild/replay for a new checkout rather than treating saved JSON as authority.

## Remaining review and next-agent work

1. Independently review the specifications against pinned PyTorch v2.10.0 and
   bind the intended meanings to the actual capture producer/version. Verify
   overloads, scalar extraction, arithmetic precision/order, range handling,
   poison behavior and exclusions. References in the headers are PyTorch source
   line numbers, not Lean line numbers.
2. Review the deliberate SPEC_LOCK updates and ledger A9: the three modeled
   Q/DQ operations only compute values, permitting DCE and CSE reuse. Lock
   consistency and passing tests are not human approval.
3. Requalify the inherited numerical compiler corrections and standalone proofs
   on the integrated branch; validate actual before/after buffers on both routes.
   The restored ancestry already contains those corrections; no new port is needed.
   Carry forward the current gate's reporting correction: universal conditional
   boundary acceptance does not widen strict refinement beyond the declared
   input types on which source support was proved. Older human-readable report
   labels can overstate this; use the actual typed chain theorem/domain.
4. Regenerate receipts from that exact final compiler/checker. Preserve existing
   failed evidence, exclusions and negative controls. The compiler corrections
   and Q/DQ gate tests are inherited from the published reference; their old
   qualification counts do not qualify this newer integration cut.

This is a tensor-stage refinement result under the ledger assumptions. It does
not prove complete IEEE fidelity, whole-model poison-free execution, W8A8 policy
accuracy, bufferization, LLVM or hardware behavior. Review approval does not
remove those trust boundaries.

## Baseline route observation during CSE qualification

Direct `lower_model(..., textual=True)` on the raw `SY_micro_model` capsule
fails in the ungated baseline: the native parser encounters an unlowered
`quant_ext` operation. This is a quantization/preparation boundary, not a CSE
checker refusal. Its original attempt is retained under
`out/artifacts/verification/v1/verification_v1_20261028T223434Z_d73cbcf/` in the
CSE integration worktree. Compatible prepared/consistent captures qualify the
textual CSE route separately. Preserve this negative evidence when porting the
published textual quantization corrections; do not count this attempt as a
successful textual export.
