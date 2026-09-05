# PR05_fits_single_k24576

PR05_fits_single_k24576: matmul over W[24576, 32]:fp8_e4m3, A0[32, 24576]:fp8_e4m3, authored from shared operand-residency ladder: one deep-K contraction per derived residency band of the target's own operand store, three depths spread across each band, at fixed single-tile parallel extents (tile=32; K=24576, M=32, N=32; K=24576 is fits_single).

kind=model_slice label=dev op=matmul modes={'relu': False, 'acc_scale': False}
