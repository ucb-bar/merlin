# PR05_fits_single_k8192

PR05_fits_single_k8192: matmul over W[8192, 16]:i8, A0[16, 8192]:i8, authored from shared operand-residency ladder: one deep-K contraction per derived residency band of the target's own operand store, three depths spread across each band, at fixed single-tile parallel extents (tile=16; K=8192, M=16, N=16; K=8192 is fits_single).

kind=model_slice label=dev op=matmul modes={'relu': False, 'acc_scale': False, 'i8': False}
