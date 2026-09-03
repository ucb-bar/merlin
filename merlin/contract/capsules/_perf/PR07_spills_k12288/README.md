# PR07_spills_k12288

PR07_spills_k12288: matmul over W[12288, 16]:i8, A0[16, 12288]:i8, authored from shared operand-residency ladder: one deep-K contraction per derived residency band of the target's own operand store, three depths spread across each band, at fixed single-tile parallel extents (tile=16; K=12288, M=16, N=16; K=12288 is spills).

kind=model_slice label=dev op=matmul modes={'relu': False, 'acc_scale': False, 'i8': False}
