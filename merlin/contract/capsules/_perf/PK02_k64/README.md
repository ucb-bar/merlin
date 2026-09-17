# PK02_k64

PK02_k64: matmul over W[64, 16]:i8, A0[16, 64]:i8, authored from shared reduction-depth sweep at fixed single-tile parallel extents (tile=16; K=64, M=16, N=16).

kind=model_slice label=dev op=matmul modes={'relu': False, 'acc_scale': False, 'i8': False}
