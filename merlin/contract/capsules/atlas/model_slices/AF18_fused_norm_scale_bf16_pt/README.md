# AF18_fused_norm_scale_bf16_pt

AF18_fused_norm_scale_bf16_pt ports npu_model's `smolvla_fused_norm_scale`: out[i, j] = matrix[i, j] * rsqrt(variance[i, j]), a pattern that appears 64 times in the model and whose full shape is variance[241] x matrix[241, 960]. Like npu_model's own Program this capsule is the one-tile 32x32 form with the variance already broadcast to the matrix shape on the host. Execution reference: atlas-npu `baremetal/assembly/smolvla_fused_norm_scale.S` -- there is no VRSQRT in this ISA, so rsqrt is `VSQRT.BF16` followed by `VRECIP.BF16`, then `VMUL.BF16`; no VTRPOSE, because nothing reaches the MXU. DTYPE -- option (a): bf16 end to end, which is what npu_model states and what the VPU computes.

kind=model_slice label=public op=fused_norm_scale modes={}
