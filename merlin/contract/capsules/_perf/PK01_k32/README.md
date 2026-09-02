# PK01_k32

PK01_k32: matmul over W[32, 16]:i8, A0[16, 32]:i8, authored from shared reduction-depth sweep at fixed single-tile parallel extents (tile=16; K=32, M=16, N=16).

kind=model_slice label=dev op=matmul modes={'relu': False, 'acc_scale': False, 'i8': False}
