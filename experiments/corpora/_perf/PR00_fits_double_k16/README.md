# PR00_fits_double_k16

PR00_fits_double_k16: matmul over W[16, 16]:i8, A0[16, 16]:i8, authored from shared operand-residency ladder: one deep-K contraction per derived residency band of the target's own operand store, three depths spread across each band, at fixed single-tile parallel extents (tile=16; K=16, M=16, N=16; K=16 is fits_double).

kind=model_slice label=dev op=matmul modes={'relu': False, 'acc_scale': False, 'i8': False}
