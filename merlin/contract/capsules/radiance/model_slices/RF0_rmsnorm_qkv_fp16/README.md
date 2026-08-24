# RF0_rmsnorm_qkv_fp16

RF0_rmsnorm_qkv_fp16: rmsnorm_qkv over X[16, 16]:fp16, G[1, 16]:fp16, Wqkv[16, 48]:fp16, authored from fused pre-norm QKV: Y = rmsnorm(X)@Wqkv fp16 (radiance-kernels rmsnorm_qkv_fused).

kind=model_slice label=public op=rmsnorm_qkv modes={'rmsnorm': True}
