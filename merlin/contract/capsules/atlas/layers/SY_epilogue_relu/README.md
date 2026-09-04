# SY_epilogue_relu

SY_epilogue_relu: matmul over W[64, 32]:fp8_e4m3, A0[32, 64]:fp8_e4m3, authored from synthesized for the epilogue axis: this target can fuse a 'relu' stage onto a contraction (evidenced by ['isa_instruction_class']), and a (family, dtype, alignment) cell cannot demand a particular stage -- so without this the capability is reported covered by whichever single stage the cell axis happened to pick.

kind=layer label=public op=matmul modes={}
