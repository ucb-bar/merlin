# RP2_rmsnorm_fp16_pt

RP2_rmsnorm_fp16_pt: rmsnorm over X[16, 16]:fp16, G[1, 16]:fp16, authored from PyTorch row RMSNorm fp16: x * rsqrt(mean(x^2)+eps) * gamma.

kind=model_slice label=public op=rmsnorm modes={'rmsnorm': True}
