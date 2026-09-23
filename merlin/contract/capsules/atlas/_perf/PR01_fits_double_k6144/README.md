# PR01_fits_double_k6144

PR01_fits_double_k6144: matmul over W[6144, 32]:fp8_e4m3, A0[32, 6144]:fp8_e4m3, authored from shared operand-residency ladder: one deep-K contraction per derived residency band of the target's own operand store, three depths spread across each band, at fixed single-tile parallel extents (tile=32; K=6144, M=32, N=32; K=6144 is fits_double).

kind=model_slice label=dev op=matmul modes={'relu': False, 'acc_scale': False}
