# Connected regions and fan-out fusion: scoped graph correspondence

Implementation is in `/home/jeremy/Projects/merlin` on `feat/pass-verification`.
All new sources are owned additive leaves. Production, frozen semantics, locks,
shared proof/checker/CLI sources and other workers' files remain unchanged by this
track. Nothing is staged, committed, pushed or merged. The production target is
the read-only intake at `4499eebb959c0f92cefd92cc6d80646ba66474ed`.

This checkpoint extends **`Merlin.StateEffects.Action/execute`** to actual
`plan_connected_region` and multi-output fusion boundaries. Its status is **scoped
graph proof; full pass/pipeline/native verification remains partial**. The corrected
candidate completed **55 tests, zero failures/errors/skips**, including all four
genuine/refused fresh-process plan/fusion replays. See
[qualification](QUALIFICATION-v4.json),
[source manifest](source-delivery-v2/MANIFEST.json) and
[owned patch](source-delivery-v2/owned.patch).

| Paper row | Region planning and one supported fan-out realization |
| --- | --- |
| Actual transformation | P feeds A and B; the planner must also retain an externally returned P. A two-stage selection must retain P for the unselected second consumer. Physical multi-output fusion refuses an escaped P; its supported case returns both A and B and has an unchanged subsequent observer of B. Actual production discovery, planner, realization, before/after Kernel JSON, conformance and encoded image are captured. |
| Existing → new roots | `StateEffects.Action/execute` → `GraphRegions.checkBoundary_sound`, `replace_region_preserves_observers` → `GraphRegionsFusion.checkRealization_sound`. `checkRealization_native` states the separate local native implementation contract. |
| Relation and domain | Every accepted complete plan decodes to exactly the ordered source nodes. Every accepted logical fused region preserves the complete function observation list after arbitrary unchanged prefix/suffix execution. Node meanings are deterministic total pure Actions with complete SSA read/write frames, read locality and explicit type preservation, for a well-typed initial state. No arithmetic order changes. |
| Inhabited domain | `inhabited_domain` constructs tagged-value actions and an initialized tagged state for every slot-type map. This proves nonvacuity of the abstract domain; it is **not** a concrete integer-tensor arithmetic/native interpretation or an upstream definedness theorem. |
| Checked correspondence | Lean independently decodes all positional operand/result indices, stage ordinal and semantic key/type lists; checks canonical SSA value order, complete ordered live-in/out lists, all lifetimes, source SSA availability/freshness, selected contiguous scope and the entire retained source graph/context. Full source context is losslessly encoded in 64-character chunks, including initializers and untouched observers. |
| Trusted projection | Primary-owned xDSL parses both actual programs, enumerates operation/result/block-argument identities and every use position, excludes unread destination initializers from logical data dependencies only after a closed-body check, and projects the native product's guarded operation/input/output identities. These Python steps are **not** proved Lean-MLIR operational correspondence. Hashes are opaque operation identities, not arithmetic laws. |
| Source subset | One single-block function, positive static byte-addressable integer tensors, identity-indexed parallel one-result generics, closed unflagged integer bodies. Initialization reads, partial division, overflow poison flags, effects, captures, dynamic/empty shapes, calls, control flow, reductions, floating point and unknown contracts are unsupported in this leaf. No actual-model coverage claim. |
| Realization contract | The native compiler's complete effects, synchronous private composition, no-fault suppressed transfers and unobserved suppressed transfers are captured declarations. Their truth, physical alias/ownership/completion assumptions and primitive implementation correctness remain independently unproved. Fresh production replay and native Python structural checks do not supply a Lean proof of these premises. |
| Composition | The new theorem proves complete surrounding-program observer composition within the existing Action semantics. Direct substitution of a native implementation additionally requires the stated **local** live-output and frame/read-locality/totality contract. Bridges from legacy `MerlinVerify`, other owners' loop/storage/state projections, actual upstream arithmetic execution and target runtime remain pending. |
| Tool/audit | Existing pinned Lean 4.35.0 executable, fresh private project objects, installed Init import closure and shared-runtime identities; transitive theorem axiom audit restricted to `propext`, `Classical.choice`, `Quot.sound`. No reflection, native decision, new axiom, `sorry`/`admit`, downloads or proof-limit increase. Matching Lean-MLIR manifest unavailable. |
| Review and final artifact | Composition declarations and concrete operation interpretation remain provisional; no self-approval. No shared pipeline receipt/claim authority is issued. No LLVM, hardware, native numerical or whole-model correctness follows. |

The integrated theorem's conclusion is, in plain terms: executing the checked
logical region as one unit publishes every value that the rest of the program
can observe, so the unchanged remainder returns the same values. It applies to
the complete projected actual program, not just a manually chosen scalar lemma.
The separate native theorem reaches the same conclusion **if** the native region
implements that local contract. We have not proved that implementation premise.

`qualification-v1` is preserved as **unqualified**: the fixture failed before
tests ran because elaborating an implicitly instantiated actual theorem crashed
Lean with SIGSEGV. Explicit theorem parameters passed a subsequent targeted build.
`qualification-v2` completed 45 tests and failed the remaining positive fusion
replay because a fresh compilation of `GraphRegions.lean` received SIGSEGV. The
cause of either crash has not been established; neither failed profile is promoted.
The next profile records explicit `-j1` worker selection, stronger saved proof/audit
content binding, and further forgery tests. No automatic retry or proof-limit
increase is used. All failed capsules remain at their original paths.

Old numerical evidence is separate and unchanged: frontend trig-context commit
`9ce96f1d9deaa5109490925aed6ec6ceea1c9650` has compiled mathematical results but
an unqualified audit/test attempt (both SIGTERM, zero completed qualifications).
Older HOST provenance, NaN/runtime, unsigned-policy/native/capture, W8A8 and
TorchAO obligations remain visible in their preserved workstreams. This graph
checkpoint neither discharges nor removes any numerical gate.

The v3 profile is **unqualified**: 52 tests passed, two fresh fusion replays
rejected saved native comparison metadata. The bounded diagnostic is
[native-context-diagnostic-v1/DIAGNOSTIC.json](native-context-diagnostic-v1/DIAGNOSTIC.json).
All 15 raw artifact buffers matched exactly. The first differing field was
`realization.source_passes[0].after.context.execution`; all six before/after
execution-context hashes differed. Both were independently recomputed with the
pinned production `term(law, scalar_binding, term(selected.record()))` calculation.
The law and scalar binding matched; only the selected facts/software **paths**
differed. Those paths are intentional inputs to the native identity. The correction
therefore retains the original selected-input locations during fresh production
reissue, records their strict file references and selection record, and keeps exact
native context comparison. No execution-context field is ignored or normalized
away. The v1/v2 SIGSEGV causes remain unknown.

The correction passed bounded diagnostics before the single final profile ran:
[exact context comparison](native-origin-targeted-v1/CONTEXT-COMPARISON.json),
[bounded fresh-process replays](native-origin-targeted-v1/BOUNDED-REPLAY.json) and
[genuine relocated control plus five exact-reason forgery refusals](native-relocation-control-v1/RESULT.json).
All **16** current native artifacts and all normalized metadata match; normalization
validates and removes only strict file-reference locations. Selected input paths
inside the actual native execution identity stay fixed and are compared exactly.

| Completed evidence | Exact scope |
| --- | --- |
| Formal | `checkBoundary_sound`, `checkRealization_sound`, and the conditional `checkRealization_native`; actual accepted program terms instantiate the integrated theorem. Complete plan decoding, SSA freshness/availability, ordered interface, lifetime and untouched-context checks; arbitrary unchanged sequential context and complete returned-observer equality. |
| Kernel audit | 45 successful kernel-proof processes, 541 transitive theorem prints including actual verdicts/program instantiations. Observed axioms are only `propext` and `Quot.sound`, within the standard-three policy. Repeated audit instances are **not** independent transformation cases. |
| Actual producer cases | Four full programs: 3×5 all-i16 observed P/two consumers; production mixed i16/i8 observed-P physical refusal; independent 2×11 two-stage plan retaining P for its outside consumer; supported 3×5 mixed-type physical realization plus unchanged downstream observer. All are synthetic actual production-API inputs, not actual-model coverage. |
| Compiler validity | 13 valid programs and one invalid-SSA refusal. This includes eight baseline before/retained-after validations, two changed untouched-observer programs, and three compiler-valid but explicitly unsupported semantic variants. |
| Kernel mutations | Thirteen proposed-plan mutations, two compiler-valid changed-observer programs, and three integrated logical-fusion mutations. Ordered inputs/results, same-type SSA identity, missing P/second output/consumer, type/body/ordinal mismatch, wrong lifetime, duplicate identities and out-of-range indices are refused. Proposed-plan/body JSON mutations submit unchanged compiler-valid MLIR; they are distinct from the two actual MLIR mutations. |
| Host binding | Missing/substituted originals, Boolean reference sizes, stale semantic/tool/proof identities, rehashed proofs/audits, consistent source substitution, changed worker profile, promoted claim flags, rehashed native image/specification, swapped associations and forged native context. These are **trusted Python protocol refusals**, not Lean semantic counterexamples. |
| Native production | Original native structural checks, fresh original-source/spec/kernel/image reissue and byte comparison. Three commands, one producer and both epilogues, 15 elements per result, actual 87-byte image. No native numerical execution or hardware qualification is inferred. |
| Source delivery | Ten additive owned sources, reverse-checked patch, unchanged shared `StateEffects` dependency, 653 installed Init source modules/3,300 bound core/toolchain files, 84 pinned production-reference source identities, host and independent compiler identities. Zero compiled objects copied. |

The realized case retains identical before/after MLIR bytes (2,853 bytes,
`8226a326935485479884f149c39fffbfe581f499a8413f56362ea9fb39e9e15b`): the actual
transformation changes the region representation and lowered Kernel, while keeping
upstream MLIR intact. The before Kernel SHA is
`eb49b2f7486ecc7c5725904f17f4feb4398ae87c7f96bf1e5ea238471cbea8ef`; the after
Kernel SHA is `4a3a3a777e172785064087367a079b3dfb0d91dd8500e380ddcfd66c6034c62c`.
The 87-byte image SHA is
`8ec21e0aeb20e44792de6f0f69197cc853918f71888a557c8cae76de6d3f6cd4`.
All original/snapshot/content associations and selected native input origins are
explicit in the receipts. `QUALIFICATION-v4.json` indexes every audit and receipt.

The selected tools are Lean 4.35.0 (executable SHA
`bf8d54e4714cc4b03d3f6bb34c83b7202b87e49c8bfcbff6895c085bb90ceb38`, explicit
`-j1`), Python 3.14.7, xDSL **0.73.0**, pytest 9.1.1 and PyYAML 6.0.3. Independent
compiler verification uses installed torch-mlir **20261001** on Python **3.12.14**,
with actual native-module hashes recorded in
[compiler identity](source-delivery-v2/compiler-oracle.json). This compiler is
not a selected matching Lean-MLIR dependency. No downloads or skips occurred.

Remaining obligations are explicit:

1. **Unproved operational correspondence:** instantiate the abstract node meanings
   with actual upstream integer-tensor execution, including the unread-allocation
   abstraction; no bridge to legacy numerical `MerlinVerify` is assumed.
2. **Unproved native local contract:** concrete primitives, complete physical effects,
   alias/ownership/completion and no-fault/unobserved suppressed transfers. Passing
   structural checks and declarative `true` fields do not establish their truth.
3. **Adjacent native/pipeline bridge:** the same Action model already proves arbitrary
   surrounding-program composition; other owners' loop/storage/state projections
   still require a common instantiated interpretation and checked boundary binding.
   Shared pipeline authorization/registry integration has not been performed.
4. **Awaiting review:** native composition/fixture declarations and the proposed concrete
   operation interpretation. No declaration has been self-approved here.
5. **Unsupported/unmeasured:** effects, dynamic/empty tensors, control flow/calls,
   reductions, floats/poison-sensitive bodies, general fusion routes, actual models,
   runtime/hardware behavior and the cause of the historical v1/v2 Lean crashes.

To reproduce this candidate in primary with the existing pinned dependencies:

```bash
PYTHONPATH=src MERLIN_GRAPH_REGIONS_QUALIFICATION="$PWD/out/artifacts/verification/graph-regions/qualification-new" \
  .venv/bin/python -B -m pytest merlin/tests/ir/test_lean_graph_regions.py -q
```

Use a fresh output path; preserved capsules are immutable. The replay functions are
`lean_graph_regions.replay_plan` and `lean_graph_regions_fusion.replay_fusion`, each
consuming its actual `EVIDENCE.json` and a new output path. The source package is a
reviewable owned delta for `feat/pass-verification`, with the exact unchanged shared
dependency supplied by the coordinator's intake. It contains no broad experimental
merge or shared Main/CLI/registry edits. No staging, commit or push was performed.
