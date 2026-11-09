# GB0_batched_matmul_i8

GB0_batched_matmul_i8: gemv_batched over W[2, 32, 16]:i8, A0[2, 16, 32]:i8, authored from batched int8 contraction: two independent (M,H)x(H,N) slices, the rank-3 shape the target contract admits and no other capsule declares.

kind=layer label=public op=gemv_batched modes={}
