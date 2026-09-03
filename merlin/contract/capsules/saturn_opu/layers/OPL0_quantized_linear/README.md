# OPL0_quantized_linear

OPL0_quantized_linear: linear over W[32, 16]:i8, X[16, 32]:i8, authored from nn.Linear int8 x int8 -> i32 across two reduction tiles.

kind=layer label=public op=linear modes={'relu': False, 'acc_scale': False, 'i8': False}
