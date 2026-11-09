# AF15_mul_bf16_pt

AF15_mul_bf16_pt ports npu_model's `smolvla_elementwise_mul`, the 32x32 canonical form of a pattern that appears 664 times across SmolVLA (gate/up fusion in the MLPs, norm x scale, attention score x mask) in 19 shape variants. Execution reference: atlas-npu `baremetal/assembly/smolvla_elementwise_mul.S` (`VMUL.BF16` pair-op); no VTRPOSE, because nothing reaches the MXU. DTYPE -- option (a): bf16 IS the hardware dtype here. npu_model builds both operands as `torch.bfloat16` and the VPU computes in bf16, so the declared dtype, the MLIR and the datapath agree.

kind=model_slice label=public op=mul modes={}
