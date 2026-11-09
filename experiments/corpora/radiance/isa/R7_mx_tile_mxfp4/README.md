# R7_mx_tile_mxfp4

R7_mx_tile_mxfp4: matmul over W[32, 32]:mxfp4, A0[32, 32]:mxfp4, A0_scale[1, 32]:e8m0, W_scale[1, 32]:e8m0, authored from contained MX PE tile (mxfp4 e2m1, E8M0 block scale, bf16 accumulate) via simt_cluster.mx_pe (gemm_mxgemmini fp4 ladder).

kind=isa label=public op=matmul modes={}
