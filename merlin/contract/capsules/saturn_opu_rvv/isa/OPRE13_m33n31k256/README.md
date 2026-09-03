# OPRE13_m33n31k256

OPRE13_m33n31k256: matmul over W[256, 31]:i8, A0[33, 256]:i8, authored from tile-edge bracket over both parallel extents at two reduction depths: below, at and one over the logical tile, which is where a tiling loop's off-by-one lands and nowhere else (tile=32; K=256, M=33, N=31).

kind=isa label=public op=matmul modes={'relu': False, 'acc_scale': False, 'i8': False}
