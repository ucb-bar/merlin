# PR06_spills_k24608

PR06_spills_k24608: matmul over W[24608, 32]:fp8_e4m3, A0[32, 24608]:fp8_e4m3, authored from shared operand-residency ladder: one deep-K contraction per derived residency band of the target's own operand store, three depths spread across each band, at fixed single-tile parallel extents (tile=32; K=24608, M=32, N=32; K=24608 is spills).

kind=model_slice label=dev op=matmul modes={'relu': False, 'acc_scale': False}
