# FE01_fused_epilogue_tail_15x32x17

FE01_fused_epilogue_tail_15x32x17: matmul over W[32, 17]:i8, A0[15, 32]:i8, authored from derived tail-path coverage for the fused epilogue: this target declares elementwise_map reachable ONLY composed with a contraction, so a standalone elementwise capsule would be the wrong capsule -- the eligibility oracle refuses one as a false fallback. The epilogue is therefore the only way the cell can be evidenced, and it had never been evidenced at a partial tile (tile=16; K=32, M=15, N=17).

kind=isa label=public op=matmul modes={'i8': True, 'relu': True, 'acc_scale': True}
