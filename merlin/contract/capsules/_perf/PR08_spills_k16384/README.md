# PR08_spills_k16384

PR08_spills_k16384: matmul over W[16384, 16]:i8, A0[16, 16384]:i8, authored from shared operand-residency ladder: one deep-K contraction per derived residency band of the target's own operand store, three depths spread across each band, at fixed single-tile parallel extents (tile=16; K=16384, M=16, N=16; K=16384 is spills).

kind=model_slice label=dev op=matmul modes={'relu': False, 'acc_scale': False, 'i8': False}
