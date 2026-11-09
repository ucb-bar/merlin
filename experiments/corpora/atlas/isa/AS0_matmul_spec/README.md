# AS0_matmul_spec

AS0_matmul_spec: matmul over W[32, 32]:fp8_e4m3, A0[32, 32]:fp8_e4m3, authored from specir spec-sourced fp8->bf16 MXU matmul (program + refmodel golden + coverage).

kind=isa label=public op=matmul modes={'relu': False, 'acc_scale': False}
