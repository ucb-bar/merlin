# OPRE02_m31n32k64

OPRE02_m31n32k64: matmul over W[64, 32]:i8, A0[31, 64]:i8, authored from tile-edge bracket over both parallel extents at two reduction depths: below, at and one over the logical tile, which is where a tiling loop's off-by-one lands and nowhere else (tile=32; K=64, M=31, N=32).

kind=isa label=public op=matmul modes={'relu': False, 'acc_scale': False, 'i8': False}
