# PK03_k128

PK03_k128: matmul over W[128, 16]:i8, A0[16, 128]:i8, authored from shared reduction-depth sweep at fixed single-tile parallel extents (tile=16; K=128, M=16, N=16).

kind=model_slice label=dev op=matmul modes={'relu': False, 'acc_scale': False, 'i8': False}
