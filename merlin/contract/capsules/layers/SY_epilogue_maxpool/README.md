# SY_epilogue_maxpool

SY_epilogue_maxpool: matmul over W[32, 16]:i8, A0[16, 32]:i8, authored from synthesized for the epilogue axis: this target can fuse a 'maxpool' stage onto a contraction (evidenced by ['manifest_composed_with']), and a (family, dtype, alignment) cell cannot demand a particular stage -- so without this the capability is reported covered by whichever single stage the cell axis happened to pick.

kind=layer label=public op=matmul modes={}
