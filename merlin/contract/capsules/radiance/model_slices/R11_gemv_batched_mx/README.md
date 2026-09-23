# R11_gemv_batched_mx

R11_gemv_batched_mx: gemv_batched over W[2, 32, 16]:mxfp8, A0[2, 16, 32]:mxfp8, A0_scale[2, 1, 16]:e8m0, W_scale[2, 1, 16]:e8m0, authored from batched MX matmul (mxfp8, E8M0), 2x [16x32]@[32x16] — decode-time gemv_batched.

kind=model_slice label=public op=gemv_batched modes={}
