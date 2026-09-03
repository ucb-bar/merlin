# OPRK02_k64

OPRK02_k64: matmul over W[64, 32]:i8, A0[32, 64]:i8, authored from reduction depth swept alone at fixed parallel extents: an odd tiny depth, one over a 64-boundary, two logical tiles, and the deepest depth the workload actually runs (tile=32; K=64, M=32, N=32).

kind=isa label=public op=matmul modes={'relu': False, 'acc_scale': False, 'i8': False}
