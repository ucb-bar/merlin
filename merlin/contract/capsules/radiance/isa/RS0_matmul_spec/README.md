# RS0_matmul_spec

RS0_matmul_spec: matmul over W[16, 16]:f32, A0[16, 16]:f32, authored from specir spec-sourced SIMT warp matmul (warp schedule + bit-exact IEEE golden + coverage).

kind=isa label=public op=matmul modes={'relu': False, 'acc_scale': False}
