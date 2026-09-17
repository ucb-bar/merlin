# OP14_requant_narrow_m

OP14_requant_narrow_m: matmul over W[32, 8]:i8, A0[1, 32]:i8, authored from epilogue with a single output row: the readout writes one row.

kind=isa label=public op=matmul modes={'relu': False, 'acc_scale': True, 'i8': False}
