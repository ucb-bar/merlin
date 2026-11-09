# PK02_k128

PK02_k128: matmul over W[128, 32]:fp8_e4m3, A0[32, 128]:fp8_e4m3, authored from shared reduction-depth sweep at fixed single-tile parallel extents (tile=32; K=128, M=32, N=32).

kind=model_slice label=dev op=matmul modes={'relu': False, 'acc_scale': False}
