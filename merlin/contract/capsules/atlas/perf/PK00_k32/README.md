# PK00_k32

PK00_k32: matmul over W[32, 32]:bf16, A0[32, 32]:bf16, authored from reduction-depth sweep at a fixed single-tile parallel extent: 1, 2, 4 and 8 tile passes, which is the smallest set that can fit a rate and an intercept and still leave a residual to look at (tile=32; K=32, M=32, N=32).

kind=model_slice label=dev op=matmul modes={'relu': False, 'acc_scale': False}
