# SY_regime_fits_single

SY_regime_fits_single: matmul over W[16, 16]:bf16, A0[2048, 16]:bf16, authored from synthesized for memory regime 'fits_single': the declared inputs occupy 2064 of 4096 operand-store rows (0.503906 of capacity), which is what puts this capsule in that regime. Extents resolved by memory_regime.extents_for_regime with the same sizing the coverage gate measures with. 32768 operand elements exceeds the 862 a 300.0s certification budget affords on this target, so it is graded at the loop tier and rests on 'SY_contraction_bf16_aligned'.

kind=isa label=public op=matmul modes={}
