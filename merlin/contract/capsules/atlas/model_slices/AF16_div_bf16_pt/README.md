# AF16_div_bf16_pt

AF16_div_bf16_pt ports npu_model's `smolvla_elementwise_div`, the 32x32 canonical form of a pattern with 217 instances and 6 shape variants in SmolVLA. Execution reference: atlas-npu `baremetal/assembly/smolvla_elementwise_div.S` -- the ISA has no divide, so the kernel is `VRECIP.BF16` followed by `VMUL.BF16`; no VTRPOSE, because nothing reaches the MXU. DTYPE -- option (a): bf16 is the hardware dtype, and the declared dtype, the MLIR and the datapath agree. The divisor is held away from zero (|b| < 0.25 replaced by 0.5), the same guard npu_model applies, so the golden is not dominated by a handful of unbounded elements. One honest gap: this capsule's reference is an exact bf16 divide, while the kernel's reciprocal-then-multiply differs from it by about a bf16 ULP -- inside the declared tolerance, but it is a tolerance the capsule relies on.

kind=model_slice label=public op=div modes={}
