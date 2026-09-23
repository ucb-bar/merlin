# OP12_full_range_int8

OP12_full_range_int8: matmul over W[255, 8]:i8, A0[8, 255]:i8, authored from full-range int8 operands over a long reduction: the largest accumulator the datapath can be handed, which distinguishes a magnitude bug from a plain arithmetic one.

kind=isa label=public op=matmul modes={'relu': False, 'acc_scale': False, 'i8': False}
