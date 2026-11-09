# PR01_fits_double_k2048

PR01_fits_double_k2048: matmul over W[2048, 16]:i8, A0[16, 2048]:i8, authored from shared operand-residency ladder: one deep-K contraction per derived residency band of the target's own operand store, three depths spread across each band, at fixed single-tile parallel extents (tile=16; K=2048, M=16, N=16; K=2048 is fits_double).

kind=model_slice label=dev op=matmul modes={'relu': False, 'acc_scale': False, 'i8': False}
