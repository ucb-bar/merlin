# PK03_k256

PK03_k256: matmul over W[256, 32]:fp8_e4m3, A0[32, 256]:fp8_e4m3, authored from shared reduction-depth sweep at fixed single-tile parallel extents (tile=32; K=256, M=32, N=32).

kind=model_slice label=dev op=matmul modes={'relu': False, 'acc_scale': False}
