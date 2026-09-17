# PR02_fits_double_k12288

PR02_fits_double_k12288: matmul over W[12288, 32]:fp8_e4m3, A0[32, 12288]:fp8_e4m3, authored from shared operand-residency ladder: one deep-K contraction per derived residency band of the target's own operand store, three depths spread across each band, at fixed single-tile parallel extents (tile=32; K=12288, M=32, N=32; K=12288 is fits_double).

kind=model_slice label=dev op=matmul modes={'relu': False, 'acc_scale': False}
