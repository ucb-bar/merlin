# OPRN00_m1n1k64

OPRN00_m1n1k64: matmul over W[64, 1]:i8, A0[1, 64]:i8, authored from narrow-vs-full in both parallel extents, so an operand swap in the accumulate fails in both directions instead of cancelling (tile=32; K=64, M=1, N=1).

kind=isa label=public op=matmul modes={'relu': False, 'acc_scale': False, 'i8': False}
