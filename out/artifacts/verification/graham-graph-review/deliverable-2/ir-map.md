# Deliverable 2: IR-to-Lean correspondence audit

Reviewer: Graham independent review (agent). Date: 2026-10-26.
Verification repository `feat/pass-verification` at `58dbc7c00acdb6770d440b773cc2954328c78472` (equals `origin`; `src/` clean). Producer checkout `out/build/upstreams/merlin-graph-proof-intake` at `4499eebb959c0f92cefd92cc6d80646ba66474ed`, clean, used read-only (the same checkout D1 used, `source-index.json`). Handoff read at working-tree sha256 `14446420…a491` (c399e441e plus 116 added lines).

Gap IDs D2-G01…D2-G08 are in [omissions-bugs.md](omissions-bugs.md). Status vocabulary:

- **Verified by me**: checked by a reviewer script or by reading the cited source lines.
- **Trusted/unchecked**: a host step that no Lean check constrains; named and left open.
- **Evidence absent locally**: needed bytes are not on this machine (D1-G03).

## Method and independence

| Artifact | Role | Imports implementation? |
|---|---|---|
| `audit/mlir_subset.py` | Character-level MLIR reader: the generic form plus the custom forms that occur in the reviewed buffers. Refuses unknown syntax and keeps `{-# … #-}` and `loc(...)` as raw text. | No (no xDSL, no Merlin) |
| `audit/audit_scalar.py` → `audit_scalar.out.json` | Builds its own de Bruijn term with its own whitelist and descriptor keys, parses the Lean query, checks structural equality under one descriptor bijection shared by all stages, checks registry signatures, checks projection.json records against the Lean terms (serializer check), and runs a concrete differential test (MLIR interpreted directly vs. Lean terms) | No |
| `audit/audit_scalar_sensitivity.py` → `probes/audit-sensitivity/SENSITIVITY.json` | Applies 5 mutations (operand order, descriptor swap, registry width, middle operand, projection record) to show the audit is not vacuous. All 5 are detected. | No |
| `audit/audit_memory_scf.py` → `audit_memory_scf.out.json` | Reviewer's own specification of reuse and of tile-2 segmentation, applied to the previous stage and compared under canonical renaming. Direct interpretation of all three stages. | No |
| `audit/audit_declared.py` → `audit_declared.out.json` | Reads the `actualBefore` Module literal against the MLIR. Decodes the final Kernel from the provider's storage declarations into a dataflow. | No |
| `audit/regen_scalar_variants.py`, `audit/regen_memory_scf.py` | Re-execute the implementation (the subject) to rebuild absent evidence, then compare bytes with the seals | Subject only |
| `audit/probe_scalar_boundary.py`, `audit/probe_memory_scf.py`, `audit/probe_pipeline_indirect.py`, `audit/probe_parser_keys.py` | Crafted inputs fed to the implementation's projection. Raw outcomes retained under `probes/`. | Subject only |

Python: `.venv/bin/python -I` for the independent scripts. Subject runs use the implementation's own command (`PYTHONPATH=<producer>/src`). Lean: `~/.elan/toolchains/leanprover--lean4---v4.35.0/bin/lean` (sha256 `bf8d54e4…eb38`, the hash pinned in `lean_graph_state.py:22`). It ran without `-o` against the pre-existing development build `out/build/verification-setup/20261025/graph-smoke-MJ63mO/lib/lean`, whose Merlin sources byte-match the current GraphStateScalar*.lean. No memory-SCF, shape or declared Lean objects exist locally, so no Lean run was possible for those rows.

**The local parser is not the qualified parser.** This machine has xDSL 0.68.0 on Python 3.12. The worked example's recorded runtime (`actual/projection.json:external_runtime`) is Python 3.14 plus 67 xDSL files. None of the 14 xDSL installs on this machine matches those files: the best is 47/67 (0.71.0) and the repo-locked 0.68.0 matches 32/67. Every local re-execution is therefore attributed to the local environment (D2-G05).

## 1. Scalar CSE/DCE → LICM worked chain (supplied, sealed copy)

Inputs (sha256): `original/source.mlir` = `checked/before.mlir` = `6640d030…6137`. `actual/stage-000.mlir` = `actual/middle.mlir` = `checked/middle.mlir` = `e9e448db…0b77`. `actual/stage-001.mlir` = `actual/after.mlir` = `checked/after.mlir` = `acc07b31…6137`. Query `checked/kernel/GraphStateScalarInstance.lean` = `544b1dfd…babc`. The `checked/source-closure` copies byte-match the current sources (25/25).

### Descriptor/type/control registry (one registry for all three stages)

| Lean id | Signature in query | Reader key (independent) | projection.json descriptor |
|---|---|---|---|
| 0 | `[(.bits 8),.index] → []` | `opaque.observe`, no metadata, (i8, index) → () | UnregisteredOp, `op_name__ = opaque.observe` |
| 1 | `[] → [.index]` | `arith.constant 0 : index` | ConstantOp, `index_constant = 0` |
| 2 | `[] → [.index]` | `arith.constant 2 : index` | ConstantOp, `index_constant = 2` |
| 3 | `[] → [.index]` | `arith.constant 7 : index` | ConstantOp, `index_constant = 7` |

Verified by me: the mapping from reader key to id is a bijection, and it is the same in all six Lean terms (`before`, `after`, `componentBefore`, `middleLeft`, `middleRight`, `componentAfter`). Each signature matches the reader's types. No registry entry is unused. The registry is justified by construction: every stage is projected over the same universe {before, middle, after} (`lean_graph_state_scalar_capture.py:137-160`, `:365-372`). The equality checks at `:369-372` and `lean_graph_state_scalar_chain.py:18-21` therefore cannot fail (D2-G04). Lean receives only signatures. The constant values 0/7/2 and the observer's identity never reach Lean; descriptor injectivity is a host property (`metadata()` at `:67-75`).

### Operation-by-operation map (environment index 0 = most recent binding)

| Stage | MLIR operation | Lean constructor | Operand slots → values |
|---|---|---|---|
| all | entry `(%a, %b)` | `inputs = [(.bits 8),(.bits 8)]`, env `[a,b]` | function-type and block-arg equality are host checks (`capture:279-282`); Lean checks arity and types |
| all | `%lower/%upper/%step = arith.constant 0/7/2 : index` | `.opaque 1`, `.opaque 3`, `.opaque 2`, each `[] [.index]` | none (index constants are opaque; not index laws) |
| before/middle | `scf.for %i = %lower to %upper step %step iter_args(%acc=%a) -> i8` | `.loop 2 1 0 [3] [(.bits 8)]` | lower=2, upper=1, step=0, init `[3]`=%a; body env `[i, acc, step, upper, lower, a, b]` |
| before | `%first = addi %a,%b` | `.pure (.add 8) [5,6]` | a, b |
| before | `%dup = addi %a,%b` | `.pure (.add 8) [6,7]` | a, b |
| before | `%dead = subi %a,%b` | `.pure (.sub 8) [7,8]` | a, b |
| before | `%product = muli %dup,%b` | `.pure (.mul 8) [1,9]` | dup, b |
| before | `"opaque.observe"(%first,%i)` | `.opaque 0 [3,4] []` | first, i |
| before | `%next = subi %acc,%product` | `.pure (.sub 8) [5,0]` | acc, product |
| all | `scf.yield %next` / `func.return %result` | `.ret [0]` / `.ret [0]` | next / loop result |
| middle | `%first`; `%product = muli %first,%b`; observe; `%next` | `.pure add [5,6]`; `.pure mul [0,7]`; `.opaque 0 [1,2]`; `.pure sub [3,0]` | CSE redirected the dup use; DCE removed `%dead` |
| after | `%first`, `%product` hoisted before the loop | `.pure add [3,4]`; `.pure mul [0,5]` | a, b; first, b |
| after | `scf.for … iter_args(%acc=%a)` | `.loop 4 3 2 [5] [(.bits 8)]` | env `[product, first, step, upper, lower, a, b]` |
| after | observe; `%next` | `.opaque 0 [3,0]`; `.pure sub [1,2]` | first, i; acc, product (captured) |

All rows were verified by me: 64 constructors (24 opaque, 6 loop, 22 pure, 12 ret) match exactly, and the projection.json records serialize to exactly these terms. The differential test ran 81 input vectors under one concrete model (wrapping i8, signed index `scf.for`, observer log as the world). MLIR before, middle and after, interpreted directly, and the four Lean terms all produce identical returned values and identical ordered observer logs. Example: inputs (2,3) give 198 with log `observe(5,0)…observe(5,6)`. This is a differential test of this instance under one model, not a proof.

### Category checklist

| Category | How it is bound | Status |
|---|---|---|
| Operations (every op in every function body) | `block_code` classifies each op as terminator, whitelisted pure op, `scf.for`, or opaque. Any other region, successor, or direct defined-callee call is refused (`capture:200-260`). No silent op drop was found in the projection. | Verified by me (reading + probes) |
| Integer widths / properties | Pure: unflagged signless i1..i256 add/sub/mul/ext/trunc/constant (`capture:78-116`). Anything flagged or attributed becomes opaque with its metadata in the key. | Verified. Primitive totality is a premise. |
| Ordered operands/results | de Bruijn indices; Lean `checkTypes` and `normalize` resolve each one | Verified (Lean + audit) |
| Entry signatures | Host checks function_type and block args equal across stages (`capture:265-282`). Lean checks inputs/outputs typing. | Host part trusted; Lean part verified |
| Declarations / function and module attributes | `metadata()` equality only (`capture:124-125, 265-266`). Declarations get no Lean term (`lean_graph_state_scalar.py:173-174`). | Trusted/unchecked |
| Captures, scoped SSA | Body env = `[iv, carries…, captured…]` (`capture:189, 243`). Lean `lift` and typing check availability. | Verified |
| IVs, carries, yields | Host checks slot types (`capture:231-236`). Lean typing covers carries/yield, and loop results are bound in `next`. | Verified |
| Loop control | Host requires a positive constant step (`capture:224-230`). Lean `model.iterations` is an arbitrary function of the three bound values. Nothing ties it to signed `scf.for` semantics. | Premise (D2-G06) |
| Return values | `.ret` with Lean typing against `outputs` | Verified |
| Persistent effects / observers | Opaque steps are kept in order, with exact descriptors and operands. They may fail with a terminal world. | Verified structurally. Effect contracts are premises. |
| Untouched functions | Each defined function gets its own `accepted`/`actual` theorem. No inter-function semantics, because direct local calls are refused. Indirect ones are admitted (D2-G01). | Partially verified |
| Middle is the same actual buffer | Supplied `checked/middle.mlir` sha equals producer `stage-000.mlir`. LICM ran on the in-memory module (`capture:316-331`), so the binding also relies on print/parse round-trip. `middle_join` is equal by construction. | Hash binding verified; round-trip trusted (D2-G04) |

### Trusted host steps vs Lean checks

| Host step (file:line) | Corresponding Lean check |
|---|---|
| Parse and verify each buffer with pinned xDSL; contexts are per call, but the dialect resource state is process-global (`producer:certificates.py:29-36`) | none |
| Module header, roster, and function headers/declarations equal (`capture:124-136, 265-268`) | none |
| Op classification, whitelist, descriptor key (`capture:67-116, 137-148, 247-259`) | registry signature typing only |
| SSA → index (`capture:195-198, 260`) | `checkTypes` (availability, types); `normalize` |
| Positive constant step (`capture:224-230`) | none |
| Serializer `program()`/`emit_query` (`lean_graph_state_scalar.py:81-206`) | none. Independently confirmed by `audit_scalar.py` on all 4 cases. |
| Chain: registry and middle equal across stages (`capture:365-377`; `chain.py:18-35`) | `middle_join`, `source_join`, `final_join` (tautological) |

### Retained zero-trip and nested chain variants

Evidence absent locally: the sealed `qualification-v1/originals/{zero-chain,nested-chain}` directories (D1-G03). `regen_scalar_variants.py` rebuilt each source from the frozen test generator (`merlin/tests/ir/test_lean_graph_state_scalar.py:31-153`) and ran the implementation's capture and chain emitter. Results (`probes/scalar-variants/REGENERATION.json`):

| Case | source.mlir vs seal | Generated query vs seal | Lean (local, no `-o`) | middle/after vs seal |
|---|---|---|---|---|
| chain (control) | match | match (`544b1dfd…`) | rc 0 | differ only in block labels `^bb1` vs `^bb0` |
| zero-chain (upper 0) | match (`e53ef429…`) | match (`cf774e0d…`) | rc 0 | regenerated; sealed bytes absent |
| nested-chain | match (`ba7db6ce…`) | match (`97470a5c…`) | rc 0 | regenerated; sealed bytes absent |

`audit_scalar.py` matches all three regenerated cases (zero-chain: 81 vectors, empty loop, returns `a`, empty log; nested-chain: 9 ordered observations per call). The byte-identical queries bind the projection result to the seal. The regenerated middle/after are not the sealed bytes.

### Boundary probes (implementation as subject; `probes/scalar-boundary/PROBES.json`)

| Probe | Raw outcome | Sound behaviour? |
|---|---|---|
| control CSE | lean_accepted | yes |
| `func.constant` + `func.call_indirect` to a local function | **lean_accepted** | **no**: local call admitted (D2-G01) |
| direct `func.call` to a local function | projection_refused | yes |
| `scf.if` region / unregistered region op / module-level non-func op / multi-block CFG / extra function | projection_refused | yes, whole boundary |
| module attribute changed / function visibility changed | projection_refused | yes |
| opaque attribute changed / observer dropped / `nsw` introduced / index constant changed / dense literal changed | lean_rejected | yes |
| resource payload changed / dropped (supplied after) | lean_rejected, but only because a xDSL global handle rename gave distinct descriptors | **accidental** (D2-G02) |
| producer CSE on a module with a resource | producer after.mlir **has no payload**; lean_accepted | **no** (D2-G02) |
| `scf.for` with `unsignedCmp` (generic) | capture_unavailable (xDSL VerifyException uncaught) | fail-closed, but reported as unavailable rather than refused |
| location-only difference | lean_accepted | yes (not semantic) |

## 2. Memory-SCF reuse → nested segmentation (two initializers)

Buffers (local, hashes equal `EXAMPLE.json`): before `bb5d985a…0110`, middle `9e2a5e83…9ff5`, after `a98bd757…4773`. Evidence absent locally: the sealed `CHECK.json`, `model.json` and `Obligation.lean` (20/1061 local). `regen_memory_scf.py` re-ran the subject on the three local buffers. **model.json `0f1f9ff5…` and Obligation.lean `afd557e3…` are byte-identical to the sealed EVIDENCE_INDEX entries.** Re-running the producer (`plan_storage`/`reuse_storage`, then `segment_loops(tile 2, 64-bit)`) in one process reproduces the retained middle and after, apart from block labels. So the retained middle is the actual reuse output, and segmentation consumed it in memory.

**Independent correspondence (verified by me; `audit_memory_scf.out.json`).**

- Reuse spec applied to before ⇒ equals retained middle. The spec deletes the first epoch's `memref.dealloc` and the second `memref.alloc`, and renames `%buffer1` to `%buffer0`. Alloc metadata, types and non-overlap are checked.
- Segmentation spec applied to middle ⇒ equals retained after. Each owner-level initializer nest becomes: `arith.constant 2 : index`; outer `scf.for lower to upper step stride { addi(iv, stride); minui(sum, upper); inner scf.for iv to end step 1 {original body}; yield }`.
- `@touch` and `@untouched` are equal in all stages.
- Direct interpretation of the three stages uses fresh allocations with poisoned cells, freed-buffer traps, and real calls into `@touch`. Over 18 runs it gives identical returns, caller memory and ordered `touch` traces.

**What Lean sees (D1-G02 checked in depth; D2-G03).** The query has no final program. `outcomes … false false instructions source` and `… true true instructions middle` evaluate ONE source-derived instruction list in two modes (`GraphLoopsMemorySCFModes.lean:130-164`). Each `retained` instruction is an opaque string key built from the left-hand (source) tree (`lean_graph_loops_reuse_projection.py:321-331`). Every Lean string equality compares values that the emitter or matcher duplicated:

- `sourceBody = targetBody` and `sourceContext = targetContext`. Both trees come from `first` (`lean_graph_loops_memory_scf_projection.py:149-150`), and the holes are identical (`:252-253`). Checked at `GraphLoopsSegmentModel.lean:50`.
- `sourceInitializer = config.key`, `middleInitializer = segmentationSourceInitializer`, `reuseMiddle = segmentationMiddle`. Each pair is emitted from one Python value (`lean_graph_loops_memory_scf_model.py:119-128`). Checked at `GraphLoopsMemorySCFModel.lean:30-31, 49`.

**Correction to D1-G02:** for memory-SCF, Lean checks equality of duplicated strings, not nonemptiness. The binding is still entirely the Python matcher. For shape, D1's wording is exact: `p.sourceBinding ≠ "" ∧ p.targetBinding ≠ ""` (`GraphLoopsShapeModel.lean:26`).

The only Lean-visible data taken from the final program is the segmentation plan: target lower/upper/stride/inner step (`Numeric`), and the generated add/min/inner operand and IV identities (`Structural`). Mutation layer probe results (`probes/memory-scf-mutations/MUTATIONS.json`):

| Mutation of final buffer | Reviewer semantics vs original | Python matcher | Obligation.lean with the matcher check (`projection.py:137`) disabled in-process |
|---|---|---|---|
| `@touch` body addi→subi | differs | refused | **byte-identical to positive** |
| second initializer stores `%seed0` | differs | refused | **byte-identical** |
| second observer reads `%value0` | differs | refused | **byte-identical** |
| `@untouched` constant 1→2 | `@untouched` result differs; not reachable from `@kernel` | refused | **byte-identical** |
| first `minui` upper operand `%one` | row 1 of epoch 0 never initialized; unobservable here because only cell [0,0] is read | accepted | differs; reviewer evaluation of `Structural.minOperands` gives false, so Lean would refuse |
| final free removed | leak | refused | refused elsewhere |

Conclusion: every wrong final program except the generated-control wiring is excluded only by the trusted Python full-tree matcher. The Lean theorem is a statement about the source-derived blueprint.

## 3. Declared-call (qualification-v1 fresh-replay 0 = i8, 1 = i16shared)

Local: 209/209 sealed files; the two original receipts are absent (D1-G03). The changed artifact is the schedule Kernel (offsets 1024 → 0 on `out`/`ao`, three fields). The source MLIR is unchanged (`retained-source.mlir` byte-identical). Verified by me (`audit_declared.out.json`):

- The `actualBefore` Module literal matches the MLIR in all 31 pre-order events (names, result/operand/block-arg numbering). Attribute keys differ in one way: the Lean literal carries `overflowFlags = none` on `arith.trunci`, which the text lacks. It is an xDSL default, materialized by the exporter. The literal transcribes the xDSL in-memory form, not the text.
- `actualRetained` equals `actualBefore`.
- The final Kernel was decoded via `selected-storage-declarations.json` choices and the windows' backing table (private0/private1 share backing in i16shared). It gives produce(`%input`,`%w0`,`%state`)→`%p` @private0+0, add(`%p`,`%w1`)→`%c0` @output0, multiply(`%p`,`%w1`)→`%c1` @output1. This equals the three generic bodies, and every access is within capacity.

Unlike memory-SCF, the final Kernel itself is a Lean literal and is decoded in Lean (`checkKernel actualKernel …`, `checkSequence`). `originalKernel` contributes only its name and argument roster. Trusted/unchecked:

- the positional ABI association (kernel read args ↔ `original-connected-plan.inputs`, write args ↔ `outputs`);
- the schema tables (capacity 512, pointer→window), `addresses`, and `initialKnown` literals;
- the per-instruction, packing, `publication` and `native` Action premises of `actual_complete_observers` (package 3).

See D2-G08.

## 4. Other sealed rows (time-limited)

- **Four-pass pipeline (sealed-v1) and integer v2.** Both reuse the scalar `project()` (`lean_graph_state_pipeline_capture.py:20-23, 67`; `lean_graph_state_integer_capture.py:16-20, 51-54`). Their extra call check only inspects `func.CallOp` (`pipeline_capture.py:56-69`, `integer_capture.py:32`). Verified: `func.call_indirect` to a local function is admitted by `project_modules` (`probes/pipeline-indirect-call/RESULT.json`), so D2-G01 and D2-G02 apply to both. Their Lean queries contain complete per-stage `Program`s, like the scalar row. Not checked: the pipeline/integer queries, theorem types and concrete control model. Evidence absent locally (63/4001 and 173/2250 files), and no compiled objects exist.
- **Shape (standalone-mode-v1).** Same blueprint mechanism as memory-SCF. Lean gets one source-derived module with candidate constant values substituted at the changed dimension sites (`lean_graph_loops_shape_projection.py:140-162`), and `complete … false ↔ … true`. Target binding is the nonempty-string check above. Unregistered ops such as `func.call_indirect` become `action` commands (`:203-212`); only `func.call` targets are recorded. Not checked further: no local objects (28/1001 files).

## 5. Not checked, and why

- Original Lean objects, kernel/audit replay for any row: compiled objects are not portable, and only the scalar modules have a local development build.
- The qualified parser: the xDSL bytes are absent (D2-G05).
- Sealed zero/nested middle/after bytes; memory-SCF sealed CHECK.json; pipeline, integer and shape queries; the declared original receipts: evidence absent locally (D2-G07, D1-G03).
- The semantics of the Lean modes, effect contracts and allocation premises, beyond what binds IR to Lean: package 3.

## Reproduce

```
cd out/artifacts/verification/graham-graph-review/deliverable-2
PY=/home/grammy/blast/merlin-pass-verification/.venv/bin/python
$PY -I -B audit/regen_scalar_variants.py      # writes probes/scalar-variants (must not exist)
$PY -I -B audit/audit_scalar.py
$PY -I -B audit/audit_scalar_sensitivity.py    # writes probes/audit-sensitivity
$PY -I -B audit/probe_scalar_boundary.py       # writes probes/scalar-boundary
$PY -I -B audit/regen_memory_scf.py            # writes probes/memory-scf
$PY -I -B audit/audit_memory_scf.py
$PY -I -B audit/probe_memory_scf.py            # writes probes/memory-scf-mutations
$PY -I -B audit/audit_declared.py
PYTHONPATH=<repo>/out/build/upstreams/merlin-graph-proof-intake/src $PY -B audit/probe_pipeline_indirect.py
PYTHONPATH=<repo>/out/build/upstreams/merlin-graph-proof-intake/src $PY -B audit/probe_parser_keys.py
```
