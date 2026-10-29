# Scoped unroll → ordered 1D segment: proof-worker evidence

This packet checks Q's **actual three-buffer example** against the existing graph semantics. It is development proof evidence, not F's independent engineering qualification, human approval, a general tiling theorem, or ordinary compiler-wide enforcement. No existing Lean source needed a fix.

## Exact result

`Merlin.GraphStatePipelineInstance.complete` proves equal `evalEntry` observations for **every defined entry**, every argument vector and every initial world in the projected original/final modules. Return vectors, modeled errors and entire terminal worlds agree. Missing entry indices return `none` on both sides; wrong entry arity returns the same arity fault and unchanged world. Named entry/declaration identity is supplied by the trusted complete-module projection.

The parameters are one common total, deterministic, state-free `Prim → List Int → Int` and one common possibly failing stateful `External World Error registry`. Its sole success premise is the declared local result arity. No source-success, body-equivalence, whole-run equality, or opaque-loop equivalence premise is assumed. Mathematical index constants/add/min have the explicit frozen interpretation; generated signed-index bounds and literal fits are checked. This does not prove runtime numeric fidelity, poison freedom, termination of real external calls, native exception behavior or hardware correctness.

The actual example has source lower=3, upper=27, stride=2, unroll factor=2, and segment tile=3, with one carried i32 and two function inputs. The actual body adds the second input, calls the declared partial `step`, and returns its carried result; the post-loop `watch` observer remains in the complete boundary. Source trips are `[3,5,...,25]`. Unrolling yields six bulk iterations; ordered segmentation groups them into two groups of three. This example has no short final segment; F's independent cases must supply that separate coverage.

## Raw IR → theorem mapping

| Actual field | Checked term / obligation |
|---|---|
| Three complete raw buffers | `actual-example/stage{0,1,2}.mlir`; hashes in `PREPARED.json` and `DELIVERY.json` |
| Source/unrolled/segmented bodies | `F2.S0`, `F2.S3`, `F2.S4`; internal S1/S2 duplicate source for emitter compatibility |
| All module functions/declarations/metadata | Complete projection; `originalEntries` / `finalEntries` contain defined entries; declaration descriptors share one registry |
| SSA operands, captures, carries, yields, results | Actual `Program` terms; `typed0..4`, `body_group`, `innerAccepted`, `inner_same` |
| Constants, stride, bounds, factor, tile, inner add/min | `controlAccepted`, `literalsFit`, `unrollBoundsChecked`, `segmentBoundsChecked`, `outer_form`, `outer_same` |
| Post-loop continuation and observer | `suffixAccepted23/34`, `suffix_same23/34`; no discarded caller observation |
| Adjacent transformations | `unroll_actual`, `segment_actual` |
| Whole defined-entry original→final | `actual`, then `complete` using existing `entries_cons` |

Two identity-prefix checks preserve the frozen five-stage emitter interface. They claim **no actual CSE/LICM optimization**. Producer reexecution attributes the supplied stages to pinned4499; it is not the semantic proof. Python parsing, exact descriptor projection, source/transport binding and live gate execution remain trusted.

## Fresh observed checks

- `BUILD.json`: all 24 existing selected project modules plus the actual query compiled with ordinary pinned Lean4.35. Private object tree: `out/build/loop-scope-20261028/proof`. No broad library build.
- `AUDIT-v2.json`: 882 project theorems; only `propext`, `Classical.choice`, `Quot.sound`; zero violations.
- `TYPES-v2.json`: all 33 actual generated root statements, including `complete`.
- `IMPORT_DAG.json`: exact source hashes, imports and topological order. `source-cut/` copies those dependencies, the canonical auditor, query and new witness source. It is a source-only snapshot, not replacement semantics.
- `Witness-v2.process.json`: source-bound witness module compiles. `WITNESS_AUDIT.json` and `WITNESS_TYPES.json` retain the corresponding fresh transitive audit and printed types.

The first audit attempt could not launch child `lean` because its bin directory was absent from PATH. `AUDIT.json`/stderr/process preserve that unavailable attempt. The corrected ordinary command explicitly prepends the already-selected executable's bin directory; no SDK changes or environment investigation occurred. Q received that direct integration fix.

## Inhabited success and first fault

`Witness.lean` imports the exact checked Query, instantiates total wrapping word operations, and defines an explicit arity-correct external provider. The provider returns 17, records every successful retained call by increasing the complete world counter, and can fail at IV9 after three successful body calls. Inputs `[3,5]` inhabit i32, are nonzero and have the exact required arity. Initial world is 7.

| Provider mode | Source and final result | Interpretation |
|---|---|---|
| Success | `.ok ([17],20)` | 12 body calls plus the retained post-loop observer |
| Fault at IV9 | `.error (.external 17,15)` | Three successful body calls then fault effect +5; no remaining body calls or final observer |

The source results are proved from actual body/control syntax; final results follow from the checked actual preservation root. Operand initialization is derived independently from checked typing. No execution premise supplies these witnesses, and no claim is made that this illustrative provider is a delivered runtime.

The first witness elaboration failed solely in the new finite input-domain membership proof. Exact first source/logs remain in `witness-first-failure/`; the successor replaces its incomplete list simplification. No frozen theorem or model changed.

## Scope and remaining work

One changed top-level loop; static nonnegative bounds, positive constant stride, index64, no-tail bulk count>factor divisible by factor, factors2..4, tiles2..16, one to four word carries, two inputs to the changed entry, region-free original body, supported complete declaration/function roster. General multidimensional/reordered/parallel tiling, unroll tails/inline/empty variants, internal calls, unsupported nested source control and unchanged loop entries are refused. Ordered 1D segmentation is the only tiling-like claim here.

Q owns source/consumer binding and fresh gate delivery. F owns independent actual positives, compiler-valid mutations, complete qualification and fresh replay, including short final segments and independent carry/stride cases. Human projection/assumption review remains separate. No semantic negative outcome or complete qualification is asserted by this W packet. Coordinator alone publishes; main checkout and old capsules are unchanged.

To reproduce the query with a fresh build, compile the ordered sources in `IMPORT_DAG.json` using the commands in `BUILD.json`, replacing only output paths with a new private tree and setting `LEAN_PATH` to it. Compile retained `actual-example/Query.lean`, then `Witness.lean` (import `Query`). Run canonical `GraphStateAudit.lean --run` with module `Query` or `Witness`, using the selected Lean bin on PATH; exact root lists are `ROOTS.json` / `WITNESS_ROOTS.json`. Do not overwrite this evidence or use absolute Linux receipt paths as laptop qualification.
