# AGENT.md — src/merlin/verify/lean/Merlin

`Foundation.lean` supplies a typed Lean-MLIR wrapping-integer dialect and generic
ordered-segment, tail-index and lifetime lemmas. It contains no target facts,
instruction strategies, model names or schedules. Lemmas quantify over sizes and
state rather than expanding tensor elements. Unsupported instruction effects and
machine behavior need separate semantics and independent checks; do not hide
those assumptions behind an equality theorem.

`StateEffects.lean` derives commutation from independent read/write frames and
preserves arbitrary surrounding sequential contexts and composed exchanges.
`ResourceGeometry.lean` checks capacity/alignment, reservation/live intervals,
declared last-use overwrite and address/read/completion relations. Host
projections bind these obligations to actual submitted IR. The truth of
physical resource/effect declarations remains an independent premise.
