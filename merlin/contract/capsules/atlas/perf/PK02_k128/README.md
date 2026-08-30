# PK02_k128

PK02_k128: matmul over W[128, 32]:bf16, A0[32, 128]:bf16, authored from reduction-depth sweep at a fixed single-tile parallel extent: 1, 2, 4 and 8 tile passes, which is the smallest set that can fit a rate and an intercept and still leave a residual to look at (tile=32; K=128, M=32, N=32).

kind=model_slice label=dev op=matmul modes={'relu': False, 'acc_scale': False}
