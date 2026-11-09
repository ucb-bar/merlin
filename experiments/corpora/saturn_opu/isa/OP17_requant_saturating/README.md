# OP17_requant_saturating

OP17_requant_saturating: matmul over W[255, 8]:i8, A0[8, 255]:i8, authored from full-range operands over a long reduction, so the narrowing saturates.

kind=isa label=public op=matmul modes={'relu': False, 'acc_scale': True, 'i8': False}
