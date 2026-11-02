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

The compiler gates build on these in four module families. Each gate compiles
the exact source closure it binds with the selected `lean -o`; there is no Lake
project here.

- `GraphLoops*`: counted-call expansion under explicit total sequential
  semantics, used by ordered loop unrolling/segmentation/tiling and, through
  `GraphLoopsMemorySCF*`, by private buffer reuse and memory DCE.
- `GraphState*`: checked composition of projected adjacent exchanges, used by
  configuration write elision and loop hoisting (`GraphStateConfiguration*`,
  `GraphStateElision*`). `GraphStateAudit.lean` is the standalone transitive
  axiom audit; gates bind it separately and do not compile it as a query.
- `GraphRegions*`: connected-region correspondence over `StateEffects` actions,
  including declared-region capture and binding.
- `HostGraphFusion.lean`: logical host dispatch fusion under the `GraphRegions`
  model, used by the `host_graph_fusion` gate.

The Python side that projects actual IR into these modules is
`src/merlin/verify/{lean_graph_*,compiler_*,buffer_reuse_*,configuration_*,host_graph_fusion_*}.py`.
