# SY_regime_spills

SY_regime_spills: matmul over W[32, 32]:i8, A0[8192, 32]:i8, authored from synthesized for memory regime 'spills': the declared inputs occupy 16448 of 16384 operand-store rows (1.003906 of capacity), which is what puts this capsule in that regime. Extents resolved by memory_regime.extents_for_regime with the same sizing the coverage gate measures with.

kind=isa label=public op=matmul modes={}
