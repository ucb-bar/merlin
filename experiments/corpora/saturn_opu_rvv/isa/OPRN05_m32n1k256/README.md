# OPRN05_m32n1k256

OPRN05_m32n1k256: matmul over W[256, 1]:i8, A0[32, 256]:i8, authored from narrow-vs-full in both parallel extents, so an operand swap in the accumulate fails in both directions instead of cancelling (tile=32; K=256, M=32, N=1).

kind=isa label=public op=matmul modes={'relu': False, 'acc_scale': False, 'i8': False}
