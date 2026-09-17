# PK00_k32

PK00_k32: matmul over W[32, 32]:fp8_e4m3, A0[32, 32]:fp8_e4m3, authored from shared reduction-depth sweep at fixed single-tile parallel extents (tile=32; K=32, M=32, N=32).

kind=model_slice label=dev op=matmul modes={'relu': False, 'acc_scale': False}
