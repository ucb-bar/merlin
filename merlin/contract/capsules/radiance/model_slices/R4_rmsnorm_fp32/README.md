# R4_rmsnorm_fp32

R4_rmsnorm_fp32: rmsnorm over X[16, 16]:f32, G[1, 16]:f32, authored from row RMSNorm (PR#1 SIMT op): x * rsqrt(mean(x^2)+eps) * gamma.

kind=model_slice label=public op=rmsnorm modes={'rmsnorm': True}
