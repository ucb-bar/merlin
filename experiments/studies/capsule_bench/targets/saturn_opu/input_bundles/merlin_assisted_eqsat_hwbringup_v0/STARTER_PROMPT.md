# Task: generate a saturn_opu_mxv256d128 MLIR out-of-tree target backend (capsule_bench — REALISTIC)

You are an autonomous agent. Produce a **non-exempt out-of-tree MLIR target backend package** for the
saturn_opu_mxv256d128 accelerator under `submission/`. Your package is graded — through its CLI entrypoints only,
never imported — by compiling workload **capsules** (interface MLIR) and matching the target's reference
behavior. This is a **compiler/backend** task: your COMPILER generates the target artifact by lowering
the interface — you never author a compute kernel.

## Scope
Make **every** public/dev capsule under the declared corpus pass. Families are discovered, not restated:
- `merlin/contract/capsules/saturn_opu/isa/`
- `merlin/contract/capsules/saturn_opu/layers/`
- `merlin/contract/capsules/saturn_opu/model_slices/`
Read each capsule's `capsule.yaml` + `capsule.interface.mlir` for its op/shapes/dtypes/epilogue, and the
target-agnostic contracts (`command_buffer_abi.yaml`, `interface_grammar.md`, the command-buffer schema).
Derive everything (rounding, tiling, dtypes, im2col, padding) from the contract + the target's own docs
below — nothing is restated here. The numeric reference golden is withheld; iterate against the QA gate.
Build ONE general backend for every family — do not special-case individual capsules.

## Deliverable (write into `submission/`)
```
submission/
  manifest.yaml   # artifact_type: mlir_oot_target_backend; target: saturn_opu_mxv256d128; language: cpp|python;
                  # integrity_exempt: false; (cpp) a build block; the 4 command argv templates
  mlir_oot/       # your OOT sources: input dialect + saturn_opu_mxv256d128 target dialect + passes + saturn_opu_mxv256d128-opt
  REPORT.md       # what you built + honest scope/limitations + a final status line (see end)
  docs/           # PLAN.md (first-round design plan) + public_facts_used.md (facts used + source) + iteration_notes.md
```

## The 4 CLI entrypoints (your package is invoked ONLY via these)
- `parse`: `{tool} --verify-diagnostics {input_mlir}` — parse + verify the `merlin_iface` interface MLIR
- `lower_interface_to_target`: `{tool} --convert-iface-to-saturn_opu_mxv256d128 {input_mlir}` — emit saturn_opu_mxv256d128-dialect MLIR
- `emit_command_buffer`: `{tool} --emit-command-buffer={output_json} {input_mlir}` — schema-valid `command_buffer.json`
- `emit_target_artifact`: `{tool} --convert-iface-to-saturn_opu_mxv256d128 --emit-target-artifact {input_mlir}` — emit the target's schema-valid command buffer directly — the artifact the target's runtime consumes; no `.insn` assembly (the target has no command ISA, e.g. a spatial tensor tile driven by one-hot op ports): a command stream driving the discovered 16x16 outer-product accumulator tile over the int8, float8 datapaths. The unit's op ports are {macc, mvin, shift} -- that is the HARDWARE's command set, which the backend drives; it is not what you write in the buffer. The buffer's own `opcode` values are the ABI's: {RES_PACK, MATMUL_RESIDENT, MATMUL, COMMIT, EVICT, VECTOR_MAP, VREDUCE, RMSNORM, ATTENTION_QK, BATCHED_MATMUL, LAYERNORM, SOFTMAX, GELU, SOFTCAP, GEGLU, ATTENTION_FULL, ROPE, CONV, MATMUL_BATCHED} (closed enum, validated)

Declare these four commands in `manifest.yaml` exactly as the runner expects — see the OOT backend
contract (`mlir_oot_backend_contract.yaml`) and the manifest schema (`schemas/manifest.schema.json`).

## Plan before you build (FIRST round only)
If `qa/verdict.json` does not exist yet, this is the first round: **before writing any code, write
`docs/PLAN.md`** surveying the whole task, then build to that plan. Do NOT re-plan from scratch on later
rounds — follow and refine PLAN.md. Keep each item to a line or two:
- **Corpus**: the families/capsules you must pass and the distinct op/shape/dtype/epilogue cases in them.
- **Input ingestion**: how your `parse` entrypoint consumes the interface MLIR — parse it **structurally**
  (a real IR / grammar parser), do NOT hand-roll a lexer or text-parser; a bespoke input parser is the most
  common self-inflicted first-round failure.
- **Dialect + lowering**: the target-dialect ops you define and the interface->target rewrite passes.
- **Encoding**: how each instruction class is packed from the derived ISA facts (opcodes/fields), and how
  you check that encoding before grading.
- **Addressing + termination**: where operand addresses come from and how the program signals completion.
- **Verification loop**: the cheapest self-check per change, escalating to the full set only to converge.
It is your design contract with yourself — short and honest; update it only when your strategy changes.

## Cross-round memory (each round is a FRESH session)
You have NO memory of prior rounds except what is on disk. Between rounds the harness writes
`qa/round_brief.md` — your progress log across all graded rounds (per-round pass count, failure planes,
lowest mismatch) plus your own notes and a nudge if you stopped journaling. **At the START of every round,
read `qa/round_brief.md` and `docs/iteration_notes.md` before touching code**: build on what you already
worked out, and do NOT undo a change that improved an earlier round. **After every change, append to
`docs/iteration_notes.md`** what you changed, what the verdict showed, and your next hypothesis — that file
and the brief are your only durable memory across rounds.

## Grading + your QA signal
Per capsule the runner certifies exact-integer `golden == reference(cb) == simulate(cb) == oracle` (no tolerance) across the sim tier ladder — derived from the manifest, not restated:
- `(the target's declared sim tiers)`
and checks the required instruction coverage per capsule (it decodes your emitted artifact into an
instruction trace). You cannot run the oracle; after each round a QA gate writes a redacted
`qa/verdict.json` per capsule — `status`, `failure_plane`, `trace_violations`, `numeric_status`,
`mismatch_count`, `tiers` (L0–L3), and `all_pass` — with NO golden/expected values. Read it at each
round start and fix by `failure_plane` + `trace_violations`. Iterate until `all_pass: true`.

Useful self-checks you CAN run locally (no oracle needed): build your tool, run the 4 entrypoints on
each `capsule.interface.mlir`, and confirm the emitted `command_buffer.json` validates against the
command-buffer schema and your lowered artifact looks right.

**Iterate FAST — smallest scope, cheapest checks first.** When you `self_check`, check ONLY the capsule you
just changed (it accepts a single capsule or subset and returns in seconds) — do NOT re-grade all capsules
on every edit; run the full set once before you declare done. The slow cycle-accurate RTL check runs only
AFTER you converge on the fast functional tier, so tight, narrow loops cost you nothing.

## Hard rules (integrity)
- `integrity_exempt: false`; no `import merlin`, no `merlin.runtime.reference` calls, no baked-in reference outputs.
- **Compute must be compiler-GENERATED, never an authored/library kernel.** No hand C compute kernels, no
  copying/calling the target's high-level device libraries as the answer — your passes generate the code.
- Never hardcode/embed outputs (hidden capsules run after you freeze). One general backend.
- Do not read withheld goldens, hidden capsules, prior backends, or Merlin internals.

## Target ISA facts (derived — build your lowering on these)
# Spatial tensor-tile facts: saturn_opu_mxv256d128 (kind=spatial)
_Derived by static OPU state-manifest + HW-dialect discovery (no model run, no arc probe). 7/7 fields grounded; ungrounded = unavailable, not guessed._

- **Tile**: 16x16 (256 cells) = 4x4 clusters of 4x4 cells (source: OPU state-manifest cell-count geometry (NOT discover_mesh_dim))
- **MRF depth**: 16 banks/cell
- **Element widths**: operand i8 -> accumulator i32
- **Dtypes**: int8, float8
- **FMA latency**: int8 MAC 0 cyc; fp8 FMA 1 cyc
- **Op categories**: ['macc', 'mvin', 'shift']
- **Accumulator**: {'int8': 'i32', 'fp8': 'f32'}

## Menu of OOT modification points (merlin_assisted — the machine-checkable lever set)
The granted CCA spine is not just files to read: two answer-free calls ENUMERATE the full,
target-specific set of compiler seams you may modify for `saturn_opu_mxv256d128`, so you build the right lever set
instead of guessing from the file tree (neither imports the oracle or the grader). Both are runnable
CLIs exactly like `isa_tools.py` — run them from the workspace root:
- `python cca_contract.py check-bijection saturn_opu_mxv256d128` — the *what-to-build* checklist: which lever axes
  this target's ISA/RTL admits vs. which the compiler already routes (`orphan_fields` = leverable axes
  still to wire; `orphan_routes` = routes with no backing lever). Build every leverable axis; add no
  phantom. (API form, if you prefer: `from cca_contract import check_bijection; check_bijection("saturn_opu_mxv256d128")`.)
- `python action_catalog.py escalation-ladder <axis> saturn_opu_mxv256d128` — for one axis, the full
  FLAG→KNOB→HEURISTIC→PASS→CODEGEN ladder weakest→strongest, each row naming the concrete OOT-relative
  seam file to edit and whether it is forkable today (the "which section, and the next stronger lever"
  answer). The seams point at YOUR generated OOT package, not our in-tree reference. (API form:
  `from action_catalog import escalation_ladder; escalation_ladder("<axis>", "saturn_opu_mxv256d128")`.)

## MANDATORY development workflow (do ALL of these BEFORE the final status line — not optional)
1. Your compiler backend lives under `submission/`; compute is COMPILER-GENERATED (never a hand kernel).
2. Base every ISA / mesh / datapath / encoding decision on the **Target ISA facts** above + the
   capability contract under `merlin/contract/` — never guess or hardcode; derive any fact not given.
3. After EVERY build, run `python3 agent_selfcheck.py --submission submission --capsules all` and
   iterate until all required capsules pass — a submission you did not self-check is not acceptable.
4. GRADEABLE-FLOOR FIRST (do this in your FIRST minutes, before deep encoder / ISA / parse work):
   write `submission/manifest.yaml` declaring your entrypoints + a minimal CLI that ANSWERS all of
   them (even trivially / with empty output) so `agent_selfcheck` can invoke your package and the
   grader reaches the capsules. A round that ends WITHOUT a valid manifest scores 0 no matter how
   much compiler you built — make the package structurally gradeable EARLY, THEN iterate on real
   codegen. If you run low on time, a graded-but-imperfect package beats an ungradeable one.
5. Author the backend as an **xDSL pass pipeline** (`xdsl_dialects/`, `targetgen/synthesize/`, `targetgen/generate/`) — structured IR passes, NOT ad-hoc string assembly, and with **NO regular expressions** (`import re` / regex text-matching is prohibited; parse the IR structurally). This is checked on your submission.
6. Enumerate your lever set: run `python cca_contract.py check-bijection saturn_opu_mxv256d128` + `python action_catalog.py escalation-ladder <axis> saturn_opu_mxv256d128` (runnable CLIs, like `isa_tools.py`) and build every leverable axis they list.
7. Produce your instruction words with `python isa_tools.py asm ops.txt` rather than hand-packed shifts: it takes every field POSITION from this target's own derived ISA model, whereas a hand-built `(a << 32) | (b << 16) | ...` re-derives those positions by hand and is where a field quietly ends up in the wrong one. How you emit stays your call.
8. Before every self_check, run `python isa_tools.py lint` and `disasm` on your emitted submission/*.mlir and confirm every instruction decodes (nothing UNKNOWN or ambiguous) and the kernel halts. A clean `lint` is NOT enough — a correctly-NAMED instruction carrying a WRONG FIELD decodes cleanly and still diverges on the hardware plane. So also read the `disasm` operand fields back against what your command buffer declares for that same command — DMA/DRAM address and row pitch, readout dtype / element width, accumulate and dataflow bits, config scale — and reconcile every one that disagrees.

## Final status line (end of `submission/REPORT.md`) — write exactly one of:
1. "Backend passes all required public/dev capsules and is ready for hidden grading."
2. "Backend does not yet pass all required public/dev capsules; remaining failures listed by capsule + plane."
3. "Backend is not comparable because it violates the compiler/runtime/integrity boundary."
