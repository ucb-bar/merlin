# OP18_relu_epilogue

OP18_relu_epilogue: matmul over W[32, 8]:i8, A0[8, 32]:i8, authored from matmul + relu epilogue: a clamp the accumulator alone cannot check.

kind=isa label=public op=matmul modes={'relu': True, 'acc_scale': False, 'i8': False}
