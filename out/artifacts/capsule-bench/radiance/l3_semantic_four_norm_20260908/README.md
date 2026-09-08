# Radiance four-norm candidate rejection (2026-09-08)

## Outcome

No third family was promoted. The compiler now recognizes a structurally chained pair of RMSNorm
commands as `decoder_four_norm`, including its fp32 dtype and 16x16 shape, but the existing native
two-stage emitter failed the independent RP13 golden at L3 GSIM (256 values checked, 139.497 seconds).
The production registration therefore remains fail-closed: selection records `kernels/gemma_4norm`,
then emission refuses because that family has no qualified registered emitter.

A perturbed-golden arm was intentionally not run. Once the unchanged positive arm fails, a negative
control cannot establish correctness and would consume another bounded ~140 seconds without changing
the decision.

## Preferred-family blockers

- `kernels/rmsnorm_qkv_fused` requires MXFP8 M=64, K=256, N=64. The real native compiler capsule is
  lowered to fp32 M=16, K=16, N=48. Routing it through the MX fallback would use golden-only operand
  codes and reference emission, which is forbidden.
- `kernels/rope_qkv_fused` requires MXFP4 M=64, K=2048, N=64 and `mesh_to_shared_memory`; the native
  capsule is fp32 M=16, K=16, N=32, and the hardware contract does not declare that transfer feature.
- `kernels/gemma_4norm` has matching semantic/shape/dtype selection facts, but its fp32 native lowering
  does not reproduce the bf16 frontend graph within the capsule tolerance. Its exact arithmetic
  discrepancy needs diagnosis before registration.

The frozen `cases/positive` files are a rejected-candidate diagnostic, not qualification evidence and
not a performance measurement. No PR reference code was copied, linked, called, or dispatched.
