# B5_matmul_bias_seed_i8

B5_matmul_bias_seed_i8: matmul over W[32, 16]:i8, A0[32, 32]:i8, B[16]:i32 with a fused `bias_add` readout, i8 output. An int32 bias seeded into the accumulator before a two-K-tile contraction that accumulates onto it, over two output row panels. The target's contract scopes MVIN3 -- its full-width int32 accumulator-seed load (load state 2 at shrink = 0) -- to a contraction whose readout adds a bias (encoding.stage_scoped_classes), so this capsule demands that load, or a hardware loop that issues it itself (encoding.class_subsumes).

Written with `merlin.targetgen.corpus_spec.build` from the gemmini corpus binding; its required instruction classes are the contract's own derivation, not hand-listed.

kind=layer label=public op=matmul
