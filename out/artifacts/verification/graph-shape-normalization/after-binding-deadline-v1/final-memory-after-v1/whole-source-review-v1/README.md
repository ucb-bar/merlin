# W final-memory source batch — 2026-10-26

**SOURCE CANDIDATES ONLY. All 31 new MemorySCFAfter Lean modules are UNCOMPILED.**
The shared private runtime hold remains in force. No new compilation, root audit,
Lean positive/negative result, qualification, replay or pipeline receipt exists.
This report does not upgrade historical evidence or turn preparation tests into proofs.

Primary: `/home/jeremy/Projects/merlin`, `feat/pass-verification`;
observed HEAD `e501e823fdce2ffd804674eb8673730fff40da22`. Root alone publishes.
No owned process remains live. All native/SDK/archive reads and runtime checks remain held.

## What changed

The previous 17-source expansion checkpoint had actual-final materialization and
local initializer/access candidates but lacked source-plan→expanded execution.
The new draft fills that **source-code obligation** through:

1. Scoped register correspondence, allowing extra target scratch and preseeded
   logical allocation ghosts without equating complete environments.
2. Actual slot allocation, live-owner aliases, release, partial loads/stores,
   scalar actions, and concrete observer calls selected from the expanded module.
3. Actual tile constants, add/min controls, dynamic inner bounds, scoped loops,
   ordered initialization, raw capture names and complete private/caller heaps.
4. Exhaustive finite `Expansion` cases, ordered command induction, actual entry,
   all finite call depths, returns and public observations.
5. Independently decoded actual-final materialization and a conservative-extension
   proof giving the original source and final program **one interpretation**.

The final candidate root is
`Merlin.GraphLoopsMemorySCF.After.checked_actual_chain`; the concrete emitted root
is `GraphMemoryAfterWhole.complete_actual_chain`. Neither has been checked by Lean.
The final statements do not take a whole-run equality or `LocalSimulations` as a
premise: the source draft derives the latter from finite plan, scope and body facts.
Elaboration and proof-script corrections may still be needed once checking is authorized.

## Evidence and exact delivery

- `../source-whole-v1/`: integrity-only immutable snapshot and source-only patch;
  41 owned files (31 Lean, 5 Python, 5 tests), 106 project Lean sources and 42
  project Python sources. These counts overlap: owned sources are included in closures.
- `OWNED-SOURCES.json`, `PROJECT-LEAN-SOURCES.json`, `PROJECT-PYTHON-SOURCES.json`,
  `IMPORTS.json`, `DECLARATIONS.json` and `QUERIES.json` inside that snapshot bind
  exact source/term/root bytes. External SDK modules were listed by source import
  names only; their runtime files were not opened. Python closure is conservative
  static import discovery, not a claim about every dynamic import.
- `../tests-boundary-v1/`: **8 SDK-free source/data tests passed**. Each mutation
  passes Python/xDSL verification and changes the independently decoded final term.
- `../tests-whole-v1/`: **10 SDK-free source/data tests passed**: actual positive,
  independently changed source return, and eight changed finals. Complete raw
  before/after files, declaration-only Lean queries and preparation reports remain.
- Negative cases cover wrong buffer, in-bounds wrong address after an update,
  wrong initializer seed/address, empty inner control, return, premature free and
  observer arithmetic operands. Their Lean outcomes are **UNOBSERVED**, not refusals.
- Test audit hooks blocked optional Z3 `dlopen` attempts; no external library was
  loaded by those attempts. Two expected pytest timeout-plugin warnings remain.
- All 17 earlier expansion source files remain byte-identical. Earlier source
  snapshots, sealed graph packets, counterexamples and failed contexts are preserved.
- `LEXICAL-POLICY.json` reports no prohibited code tokens in the 31 candidates.
  This is a source-text check, **not a standard-three-axiom/kernel audit**.

`EVIDENCE.json` names the exact positive/negative query files and historical first
native-failure logs. Linux absolute paths inside old receipts are historical.

## Domain and actual IR mapping

The retained produced example is `epoch-cases-v1/two_epochs`: two static `2×3xi32`
private epochs, one reused physical allocation, two tiled initializers of tile 2,
ordered updates/loads, two calls to `touch`, one retained free, and scalar/index
returns. All three functions (`kernel`, `touch`, `untouched`) are independently
serialized, including unused surrounding code. The new chain is for the admitted
**kernel entry**; serialization of other functions alone is not a theorem for all
compiler entry points or all pass inputs.

| Actual field | Lean representation / check |
|---|---|
| Raw source and final functions, parameters, result kinds | Independently decoded `Module`/`Function`; source-term equality and materialization |
| Every SSA definition/use/capture | `Binding`, scoped `Context`, `RegistersRelated`, checked finite permutation |
| Static descriptor shape/stride/offset/element/address space | `ViewType`, `Epoch.view`, mapped allocation only |
| `alloc`/`dealloc`, eliminated allocation/free | Actual action or scoped alias/empty fragment; physical counter and slot facts |
| All initializer stores and nested yields | Actual `Code`, typed nested SCF, empty yield checks, ordered row/cell semantics |
| Outer/inner bounds, `addi`, `minui` | Actual operand IDs; `IndexAction`; fitting signed-index obligations |
| Actual internal call target and observer body | `functionAt` on expanded module plus frozen `Touch.call` execution |
| Return operands and complete caller heap | `NoViewBindings`, frozen return instruction, `Epoch.Observation` |
| Non-semantic module/function metadata | Complete frame equality; Python/parser fidelity remains explicit |

The Python generator proposes an expansion using the old source plan; it never
copies the actual-final body into that proposal. The actual source is decoded
again from raw text. Lean declarations check both source-plan identity and complete
computed-expanded→actual-final syntax. Python diagnostic comparisons are not the
proof of those equalities.

## Premises and claim limits

- Outcomes are **possible finite successful executions**. Equal errors, divergence,
  resource usage and native allocator success are not claimed.
- Allocation uses the existing total, private, unobservable naming model. Caller
  views are outside the private partition; caller/caller aliasing remains allowed.
  Private descriptors do not escape through the supported scalar/index returns.
- Typed inputs and caller descriptors are explicit input-domain requirements.
  Heap reads remain partial; the proof never invents initialized data.
- Scalar arithmetic remains one arbitrary ordered partial relation. No numerical
  identity, quantization law, reassociation or numerical-gate removal was added.
- Generated indices are nonnegative and fit the declared signed-index bounds;
  the current actual query selects 64 bits. Native MLIR index fidelity is separate.
- Python parsing/projection, metadata interpretation, descriptor/primitive/allocator
  fidelity and authenticity of the pinned producer artifacts remain review obligations.
  The retained producer is 4499; these tests did not rerun a compiler pass.
- Graham's independent projection/assumption review, the original runtime incident,
  all failed contexts and the historical allocation/capacity counterexamples remain.
  No hardware/gSIM evidence is imported as a graph theorem.

## Remaining work and next authorized steps

1. Continue source review against the exact candidate snapshot. Ordinary corrections
   must retain this cut and be delivered as a new cut, not refresh old evidence.
2. Only after root lifts the runtime hold for a new concrete context: compile the
   exact closure, fix elaboration diagnostics, inspect exact root types and transitive
   axioms. Do not reuse or retry the failed W v1/v2/v3 contexts.
3. Run the concrete positive and valid changed-boundary obligations, retain the
   first failure, and separately distinguish semantic rejection from infrastructure.
4. Freeze a new supported qualification and one fresh replay after successful checks.
   New candidates are not covered by any historical qualification.
5. Finish independent assumption/projection review and paper wording. Until those
   steps succeed the defensible statement is **implemented source candidates and
   SDK-free obligation preparation**, not a newly verified pass.

The source batch has reached a complete candidate chain. Native proof acceptance
is still a real external blocker to claiming that this new chain is proved.
