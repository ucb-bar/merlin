# OP16_requant_tail

OP16_requant_tail: matmul over W[24, 33]:i8, A0[17, 24]:i8, authored from epilogue across a short row tile AND a short column tile at once.

kind=isa label=public op=matmul modes={'relu': False, 'acc_scale': True, 'i8': False}
