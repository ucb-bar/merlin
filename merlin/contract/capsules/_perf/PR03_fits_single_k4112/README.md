# PR03_fits_single_k4112

PR03_fits_single_k4112: matmul over W[4112, 16]:i8, A0[16, 4112]:i8, authored from shared operand-residency ladder: one deep-K contraction per derived residency band of the target's own operand store, three depths spread across each band, at fixed single-tile parallel extents (tile=16; K=4112, M=16, N=16; K=4112 is fits_single).

kind=model_slice label=dev op=matmul modes={'relu': False, 'acc_scale': False, 'i8': False}
