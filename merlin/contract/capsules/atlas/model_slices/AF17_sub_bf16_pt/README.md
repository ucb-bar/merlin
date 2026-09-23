# AF17_sub_bf16_pt

AF17_sub_bf16_pt ports npu_model's `smolvla_elementwise_sub`, the 32x32 canonical form of a pattern with 114 instances and 6 shape variants across SmolVLA (softmax's `x - rowmax` step, residual-difference paths). Execution reference: atlas-npu `baremetal/assembly/smolvla_elementwise_sub.S` (`VSUB.BF16` pair-op); no VTRPOSE, because nothing reaches the MXU. DTYPE -- option (a): bf16 is the hardware dtype, so the declared dtype, the MLIR and the datapath agree.

kind=model_slice label=public op=sub modes={}
