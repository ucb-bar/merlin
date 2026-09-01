# RM0_deep_k_fits_single_f32

derived memory-mapping coverage: a contraction whose operands occupy 24576 of the shared_memory's 32768 rows (75.0%) -- it fits ONCE and cannot be double-buffered, so serialising movement against compute is correct here and failing to stage is not a defect

Every extent is DERIVED, not chosen: K=768 is the smallest 16-aligned contraction depth whose two operands occupy 24576 rows of the 32768-row `shared_memory`, computed with `merlin.targetgen.address_space.working_set_rows` -- the same function the grader classifies regimes with, so this capsule cannot sit in a regime other than the one it is named for.

Store provenance: `facts.memories['shared_memory'].row_bytes = 4 (RTL SRAM word width of the representative bank)`.

`label: dev`: the contract's three labels are public / dev / hidden, and there is no "authored but not yet graded" state, so inventing a fourth is not the answer. `dev` is correct and does not overstate anything -- coverage measures what the corpus ASKS about, while the score measures whether the target ANSWERS. Until an oracle has graded this capsule the regime is asked about, not satisfied, and that distinction lives in the score rather than in the label.
