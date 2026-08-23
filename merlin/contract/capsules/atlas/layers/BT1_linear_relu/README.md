# BT1_linear_relu

BT1_linear_relu: linear over W[32, 32]:fp8_e4m3, X[32, 32]:fp8_e4m3, authored from nn.Linear + ReLU, fp8 -> bf16.

kind=layer label=public op=linear modes={'relu': True, 'k_accumulate': True}
