# SY_epilogue_bias_add

SY_epilogue_bias_add: matmul over W[64, 32]:fp8_e4m3, A0[32, 64]:fp8_e4m3, B[32]:bf16, authored from synthesized for the epilogue axis: this target can fuse a 'bias_add' stage onto a contraction (evidenced by ['isa_instruction_class']), and a (family, dtype, alignment) cell cannot demand a particular stage -- so without this the capability is reported covered by whichever single stage the cell axis happened to pick.

kind=layer label=public op=matmul modes={}
