# Paper evidence — qualified Merlin reuse → nested segmentation

Proposed conditional paper sentence: **For the admitted static rank1/rank2
private-buffer domain, acceptance of our translation validator entails equality
of all possible complete public outcomes of Merlin buffer_reuse followed by
ordered_loop_segmentation, under one partial operation interpretation and a total,
private, unobservable allocator. Parsing and complete IR projection are trusted.**
Independent projection/contract review remains pending; do not claim all graph
passes, ordinary pipeline gating, categorized failures/divergence or native code.

This resolves the two requested verification FIXMEs for THIS enumerated row:

1. [The complete packet](../memory-scf-mode-v1/README.md) states the checked
   transformations/revision, exact supported domain/relation, old/new named roots,
   primitive/effect/allocator/alias/completion premises and mapping/trust limits.
2. [The actual worked example](WORKED_EXAMPLE.md) explains each meaningful IR
   edit alongside the unabridged original/exact-middle/final bindings and
   [retained checker output](CHECKER_OUTPUT.json), including a valid negative.

| Actual IR field/edit | Checked representation |
| --- | --- |
| static memref shape/type/layout/alignment | old Objects geometry/compatibility, allocation-relative byte bounds |
| removed/retained allocations/frees | logical epoch→slot map, lifetime/live/ready/reserved protocol |
| pure prefix ops and actual operands/results | new PureOp keys, explicit SSA reads/definitions, scoped lexical captures |
| nested IVs and store value/indices | new Initializer/Leaf/Emit, no-shadow and all-cell coverage, sequential heap stores |
| stride, add, min, generated loops and both yields | Segment.Model numeric/structural facts, checked signed control corollaries |
| both changed initializers and unchanged instructions | explicit full registry + whole instruction-list congruence and exact middle |
| calls/caller observations/returns/other functions | complete typed tree/SSA matching and arbitrary common Public interpretation |

Named new roots: GraphLoopsMemorySCF.heap_run_iff_log, Run.covers/materializes,
initialization_induced, related_initialize_logs, checked_complete_observations,
checked_reuse_then_segment, Model.checked_chain/checked_complete/checked_context,
checked_outer_control/checked_inner_control/checked_initializer_byte_offset.
The selected old roots are GraphLoopsReuse step/execution/checked_observations,
GraphLoopsSegment.complete_trace_iff, StateEffects and ResourceGeometry.
Foundation.disjointLifetime is a conceptual earlier anchor; full Lean-MLIR and
the frozen MerlinVerify SCF evaluator are NOT selected or extended.

Evidence: **71 qualification + ONE fresh71-test replay**, zero skips;61 main exact
transitive std3 roots,63 for the actual nonzero inhabited whole-case witness;
3original provider tests pass. New sources are24 files, all72 with prior owned
sources byte-checked through patches. See ../memory-scf-mode-v1/SEAL.json,
AUDIT.json, SOURCE_FREEZE.json, QUALIFICATION.json and DELIVERY.json. Old48 evidence,
old refused join and first failures remain unchanged. The old refusal is correct
for its frozen decoder; this NEW stronger admitted route closes this exact join.

Open boundaries: Python parser/projection and declared effect fidelity; scalar/
index primitive and native allocator/ABI conformance; total/private/unobservable
allocation premise (preserved actual8→16-byte capacity counterexample); reviewed
specifications; selected model captures; ordinary compiler-service integration;
broader graph roster; numerics, LLVM/backend and hardware. Possible COMPLETE
outcome equivalence is not tagged failure/divergence equivalence or totality.
Graham's independent review is separate, not completed by this engineering packet.
Foreign SDK incident remains UNKNOWN and foreign elision remains UNQUALIFIED;
this closure was verified unaffected without rehash/repair/promotion. Linux
absolute-path receipts are not laptop qualification. The gSIM panel figure is
separate execution evidence with no automatic Lean graph/async/encoding claim.
