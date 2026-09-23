# PR00_fits_double_k32

PR00_fits_double_k32: matmul over W[32, 32]:fp8_e4m3, A0[32, 32]:fp8_e4m3, authored from shared operand-residency ladder: one deep-K contraction per derived residency band of the target's own operand store, three depths spread across each band, at fixed single-tile parallel extents (tile=32; K=32, M=32, N=32; K=32 is fits_double).

kind=model_slice label=dev op=matmul modes={'relu': False, 'acc_scale': False}
