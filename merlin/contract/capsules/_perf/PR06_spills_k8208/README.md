# PR06_spills_k8208

PR06_spills_k8208: matmul over W[8208, 16]:i8, A0[16, 8208]:i8, authored from shared operand-residency ladder: one deep-K contraction per derived residency band of the target's own operand store, three depths spread across each band, at fixed single-tile parallel extents (tile=16; K=8208, M=16, N=16; K=8208 is spills).

kind=model_slice label=dev op=matmul modes={'relu': False, 'acc_scale': False, 'i8': False}
