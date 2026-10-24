# B5_matmul_bias_seed_i8

B5_matmul_bias_seed_i8: matmul over W[32, 16]:i8, A0[32, 32]:i8, B[16]:i32 with a fused `bias_add` readout, i8 output. An int32 bias seeded into the accumulator before a two-K-tile contraction that accumulates onto it, over two output row panels. The target's contract scopes MVIN3 -- its full-width int32 accumulator-seed load (load state 2 at shrink = 0) -- to a contraction whose readout adds a bias (encoding.stage_scoped_classes), and declares a plain MVIN into an accumulator address as an alternative form of it (encoding.class_alternatives). This capsule keeps MVIN3 requirable by demanding that form itself (expected.strict_classes); a hardware loop that issues it is accepted (encoding.class_subsumes).

Written with `merlin.targetgen.corpus_spec.build` from the gemmini corpus binding; its required instruction classes are the contract's own derivation, not hand-listed; `strict_classes` is the one hand-authored demand.

Labelled `dev`, not `public`: it is a conformance demand on the contract, kept out of the public source pool so it does not move the frozen source cardinality of a sealed experiment (`grading.expected_cohort`). The declared-class reachability checks read every tracked capsule regardless of label.

kind=layer label=dev op=matmul
