# OP9_k_long

OP9_k_long: matmul over W[1024, 8]:i8, A0[8, 1024]:i8, authored from a long reduction: per-tile-pair overhead amortized away, so the rate shows.

kind=isa label=public op=matmul modes={'relu': False, 'acc_scale': False, 'i8': False}
