# OPRN06_m32n32k64

OPRN06_m32n32k64: matmul over W[64, 32]:i8, A0[32, 64]:i8, authored from narrow-vs-full in both parallel extents, so an operand swap in the accumulate fails in both directions instead of cancelling (tile=32; K=64, M=32, N=32).

kind=isa label=public op=matmul modes={'relu': False, 'acc_scale': False, 'i8': False}
