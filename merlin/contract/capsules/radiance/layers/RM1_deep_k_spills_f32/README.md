# RM1_deep_k_spills_f32

derived memory-mapping coverage: a contraction whose operands occupy 33280 of the shared_memory's 32768 rows (101.6%) -- the SMALLEST tile-aligned contraction whose operands exceed the operand store, so the compiler must tile the contraction rather than resident-pack it whole

Every extent is DERIVED, not chosen: K=1040 is the smallest 16-aligned contraction depth whose two operands occupy 33280 rows of the 32768-row `shared_memory`, computed with `merlin.targetgen.address_space.working_set_rows` -- the same function the grader classifies regimes with, so this capsule cannot sit in a regime other than the one it is named for.

Store provenance: `facts.memories['shared_memory'].row_bytes = 4 (RTL SRAM word width of the representative bank)`.

`label: dev`: the contract's three labels are public / dev / hidden, and there is no "authored but not yet graded" state, so inventing a fourth is not the answer. `dev` is correct and does not overstate anything -- coverage measures what the corpus ASKS about, while the score measures whether the target ANSWERS. Until an oracle has graded this capsule the regime is asked about, not satisfied, and that distinction lives in the score rather than in the label.
