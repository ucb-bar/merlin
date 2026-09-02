# Task: generate a gemmini MLIR out-of-tree target backend (capsule_bench — FULL SUITE)

You are an autonomous agent. Produce a **non-exempt out-of-tree MLIR target backend package** for the
gemmini accelerator under `submission/`. Your package is graded — through its CLI entrypoints only,
never imported — by compiling workload **capsules** (interface MLIR) and matching the target's reference
behavior. This is a **compiler/backend** task: your COMPILER generates the target artifact by lowering
the interface — you never author a compute kernel.

## Scope
Make **every** public/dev capsule under the declared corpus pass. Families are discovered, not restated:
- `merlin/contract/capsules/isa/`
- `merlin/contract/capsules/layers/`
- `merlin/contract/capsules/model/`
- `merlin/contract/capsules/model_slices/`
Read each capsule's `capsule.yaml` + `capsule.interface.mlir` for its op/shapes/dtypes/epilogue, and the
target-agnostic contracts (`command_buffer_abi.yaml`, `interface_grammar.md`, the command-buffer schema).
Derive everything (rounding, tiling, dtypes, im2col, padding) from the contract + the target's own docs
below — nothing is restated here. The numeric reference golden is withheld; iterate against the QA gate.
Build ONE general backend for every family — do not special-case individual capsules.

## Deliverable (write into `submission/`)
```
submission/
  manifest.yaml   # artifact_type: mlir_oot_target_backend; target: gemmini; language: cpp|python;
                  # integrity_exempt: false; (cpp) a build block; the 4 command argv templates
  mlir_oot/       # your OOT sources: input dialect + gemmini target dialect + passes + gemmini-opt
  REPORT.md       # what you built + honest scope/limitations + a final status line (see end)
  docs/           # PLAN.md (first-round design plan) + public_facts_used.md (facts used + source) + iteration_notes.md
```

## The 4 CLI entrypoints (your package is invoked ONLY via these)
- `parse`: `{tool} --verify-diagnostics {input_mlir}` — parse + verify the `merlin_iface` interface MLIR
- `lower_interface_to_target`: `{tool} --convert-iface-to-gemmini {input_mlir}` — emit gemmini-dialect MLIR
- `emit_command_buffer`: `{tool} --emit-command-buffer={output_json} {input_mlir}` — schema-valid `command_buffer.json`
- `emit_target_artifact`: `{tool} --convert-iface-to-gemmini --emit-target-artifact {input_mlir}` — lower your target dialect to an **LLVM-dialect MLIR** module (a `.mlir`, NOT textual LLVM-IR) whose command-ISA instructions are `llvm.inline_asm` ops wrapping raw `.insn` directives — with opcode/func3/func7 from the discovered ISA facts, assembled by STOCK clang/LLVM, no forked toolchain. EVERY operand must be an SSA value defined earlier — an immediate as `%c = llvm.mlir.constant(<imm> : i64) : i64`, a pointer via `llvm.ptrtoint` of an arg — then passed by name. Canonical:
    %c = llvm.mlir.constant(1441801 : i64) : i64
    %d = llvm.mlir.constant(16 : i64) : i64
    llvm.inline_asm has_side_effects ".insn r <op>, <f3>, <f7>, x0, $0, $1", "r,r" %c, %d : (i64, i64) -> ()
NEVER an inline integer literal operand like `... "r,r" (65540, 16)` — that is invalid MLIR: it neither assembles NOR decodes, so CONFIG etc. read back as UNKNOWN and the instruction class is scored missing. And do NOT emit textual LLVM-IR (`call void asm sideeffect "..."`): the runner decodes `llvm.inline_asm` MLIR ops, so a `.ll`-style body reads back as an empty instruction trace: a word stream encoding ONLY the 26 discovered legal command opcodes (funct field 0..126; enumerated in the ISA facts below); driving the discovered 16x16 systolic mesh; the emitted module defines `gemmini_kernel`

Declare these four commands in `manifest.yaml` exactly as the runner expects — see the OOT backend
contract (`mlir_oot_backend_contract.yaml`) and the manifest schema (`schemas/manifest.schema.json`).

## DRAM addressing — your kernel receives the operands as POINTER ARGUMENTS
Your emitted kernel function is the lowering of the capsule's interface function: it receives **each
interface tensor as a pointer argument**, in the interface's order (each tensor's role — input / weight /
output — is declared in the interface MLIR you lower). The bare-metal harness ALLOCATES those buffers at
run time and passes their addresses in — there is NO fixed, known-ahead-of-time DRAM address. So for every
memory-movement instruction (the ISA facts mark which classes move data between DRAM and the accelerator),
compute its DRAM address FROM the matching pointer argument: `%a = llvm.ptrtoint <that arg> : !llvm.ptr to
i64`, then use `%a` (optionally plus a constant tile/element offset) as the address operand. NEVER bake a
literal DRAM address (0, or any constant): a baked address cannot match the buffer the harness allocated,
so the kernel accesses the wrong memory and faults on every capsule. On-chip scratchpad / accumulator
addresses ARE fixed constants — only the off-chip DRAM addresses must come from the arguments. The ISA dev
tools flag a baked DRAM address, so you can catch this before the oracle runs.

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
This corpus is MIXED, so the grading model is decided PER CAPSULE by that capsule's own golden, not once for the target: a capsule carrying an INDEPENDENT float `golden` is certified against the program oracle within its declared tolerance (its `grade_policy` atol/rtol) and its integer `reference(cb) == simulate(cb)` self-consistency cross-checks report `not_applicable`; every other capsule is certified exact-integer `golden == reference(cb) == simulate(cb) == oracle` with no tolerance. Both apply across the sim tier ladder — derived from the corpus goldens, not restated:
- `L2` → spike
- `L3` → verilator
- `L4` → vcs
- `L5` → firesim
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
- **Nor any other capsule-specific value.** Read every extent and every attribute from the capsule you
  were handed, never from the set you happened to see. Two things a held-out capsule caught in a
  submission that passed its public suite: a dispatch guarded on ONE operand extent, so a matmul with a
  second tile in that dimension wrote nothing; and an accumulator-scale epilogue that emitted the literal
  constant the public capsule happened to use, while the surrounding code parsed the real attribute and
  discarded it. Both passed every public capsule and failed the holdout that changed only that value.
- Do not read withheld goldens, hidden capsules, prior backends, or Merlin internals.
- **If you cannot lower something, DECLINE it — do not emit a program that writes nothing.** Set
  `declined: {"reason": "...", "shape": [...], "op": "..."}` on the command buffer and emit no commands.
  A decline is scored as not-passed (it never becomes a pass), but it is recorded as a COVERAGE gap
  rather than as wrong arithmetic, and your self-check reports it back to you by shape. Falling through
  to an empty/terminator-only program instead makes your refusal arrive as an output of zeros — which is
  indistinguishable from a multiply that ran and was wrong, so you will debug arithmetic you never
  emitted. An empty command buffer with no stated reason is a contract violation.

## Target ISA facts (derived — build your lowering on these)
**Shipped gemmini ISA — the source of truth for instruction encodings (derive, never invent):**
The real gemmini ISA is shipped read-only in your bundle. Derive EVERY instruction's
exact encoding from these files. Do NOT invent opcodes, mnemonics, instruction classes, or a
bit layout: a plausible-but-invented encoding assembles cleanly yet decodes to garbage on the
target and scores 0 (this is the single most common failure on a self-hosted ISA).
- `experiments/capsule_bench/targets/gemmini/contracts/hwbringup_gemmini_v0/isa_include/gemmini.h`
- `experiments/capsule_bench/targets/gemmini/contracts/hwbringup_gemmini_v0/isa_include/gemmini_params.h`
- `experiments/capsule_bench/targets/gemmini/contracts/hwbringup_gemmini_v0/` (also mounted as `gemmini/`) — RTL + ISA headers + README + a WORKED
  example kernel under `example_kernel/`. Translate the example's real instructions into
  your emitted encoding using the exact field layout the ISA definition specifies; the
  legal-opcode values in the ISA facts below are DECODE GATES, not the instruction
  semantics — take semantics + field packing from these files, never from the value list.

# Target ISA facts: gemmini
_Derived by static CIRCT HW-dialect discovery (no model run). 4/4 fields grounded; ungrounded = unavailable, not guessed._

- **Legal opcodes** (26): `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 126]`
  - source: decoder_icmp_fanout(mlc)
- **Mesh DIM**: 16
- **On-chip capacity (bytes)**: operand=262144, accumulator=65536

## ISA dev tools (assembler / disassembler / linter) — staged as `isa_tools.py`
You have a derived RoCC toolset (oracle-free; it encodes the syntax YOU choose and inspects YOUR OWN
emitted MLIR — never a golden). Use it so you never hand-write a wrong `.insn` op:
- `python isa_tools.py asm ops.txt` — assemble a listing (one `CLASS rs1 rs2` per line, e.g.
  `CONFIG_EX 0 0`, `MVIN 0x80000000 16`, `PRELOAD 256 0`, `COMPUTE_PRELOADED 0 0`, `MVOUT 0xA0000000 16`,
  `FLUSH 0 0`, `FENCE`) into the CANONICAL `llvm.inline_asm` MLIR — each operand a
  `%c = llvm.mlir.constant(<v> : i64)` SSA value — packed with this target's derived opcode/func3/func7.
  It REFUSES rather than emit a wrong instruction (unknown class, or a CONFIG whose rs1 subtype bits don't
  match). Paste its output into your emitted `.mlir` — NEVER hand-write inline-integer-literal operands like
  `"r,r" (65540, 16)`: that is invalid MLIR that neither assembles NOR decodes (it reads back as UNKNOWN and
  the class scores missing).
- `python isa_tools.py disasm submission/<your>.mlir` — decode your emitted MLIR back to instruction
  classes; anything that comes back UNKNOWN is a non-canonical/garbled instruction.
- `python isa_tools.py lint submission/<your>.mlir` — flag UNKNOWN instructions + show the decoded class
  histogram. Run it BEFORE every `self_check` (it is instant and catches the exact encoding mistake that
  makes an otherwise-correct kernel fail the trace gate).

When a capsule passes the cheap tiers but fails the hardware oracle (the numeric/trace check is green yet
the RTL oracle disagrees), OBSERVE the hardware behavior of YOUR OWN command buffer with the lite debugger:
- `python isa_tools.py debug submission/command_buffer.json --capsule <name>` — answers YOUR command
  buffer on the RTL-derived arc model and reports per-op HARDWARE STATE: `per_command` (cycles +
  scratchpad-read / accumulator-write / DRAM-refill counts for each command), aggregate `metrics`
  (bytes moved, accumulator commits, evictions), and the RTL `oracle` fingerprint. The output VALUES and
  the pass/fail verdict are WITHHELD (that is the answer key). This runs your INTENDED computation, so pair
  it with `disasm`/`lint` on the emitted `.mlir` (the encoding) to catch a field the command buffer cannot
  carry (a store stride, a readout dtype, a tile DRAM offset).

## Menu of OOT modification points (merlin_assisted — the machine-checkable lever set)
The granted CCA spine is not just files to read: two answer-free calls ENUMERATE the full,
target-specific set of compiler seams you may modify for `gemmini`, so you build the right lever set
instead of guessing from the file tree (neither imports the oracle or the grader). Both are runnable
CLIs exactly like `isa_tools.py` — run them from the workspace root:
- `python cca_contract.py check-bijection gemmini` — the *what-to-build* checklist: which lever axes
  this target's ISA/RTL admits vs. which the compiler already routes (`orphan_fields` = leverable axes
  still to wire; `orphan_routes` = routes with no backing lever). Build every leverable axis; add no
  phantom. (API form, if you prefer: `from cca_contract import check_bijection; check_bijection("gemmini")`.)
- `python action_catalog.py escalation-ladder <axis> gemmini` — for one axis, the full
  FLAG→KNOB→HEURISTIC→PASS→CODEGEN ladder weakest→strongest, each row naming the concrete OOT-relative
  seam file to edit and whether it is forkable today (the "which section, and the next stronger lever"
  answer). The seams point at YOUR generated OOT package, not our in-tree reference. (API form:
  `from action_catalog import escalation_ladder; escalation_ladder("<axis>", "gemmini")`.)

## MANDATORY development workflow (do ALL of these BEFORE the final status line — not optional)
1. Your compiler backend lives under `submission/`; compute is COMPILER-GENERATED (never a hand kernel).
2. Base every ISA / mesh / datapath / encoding decision on the **Target ISA facts** above + the
   capability contract under `merlin/contract/` — never guess or hardcode; derive any fact not given.
3. After EVERY build, run `python3 agent_selfcheck.py --submission submission --capsules all` and
   iterate until all required capsules pass — a submission you did not self-check is not acceptable.
   THEN run `python3 agent_selfcheck.py --submission submission --shape-coverage`, which probes the
   SAME operation at one tile and at two tiles in each of M, K and N. It costs no simulator (it runs
   only your emit path), so run it often. **The capsules are a FIXED SET OF SHAPES: passing all of
   them says nothing about whether you lower anything else, and you are graded on shapes you have
   not seen.** `emitted_work` is how many instructions you emitted per shape — a bigger problem
   cannot need a smaller program, so a corner reported `collapsed` is a shape you silently refused.
   `multi_tile_axes_uncovered` names the axis your lowering does not loop over: fix the LOOP, not
   the arithmetic. A round is not converged while any axis is uncovered.
4. GRADEABLE-FLOOR FIRST (do this in your FIRST minutes, before deep encoder / ISA / parse work):
   write `submission/manifest.yaml` declaring your entrypoints + a minimal CLI that ANSWERS all of
   them (even trivially / with empty output) so `agent_selfcheck` can invoke your package and the
   grader reaches the capsules. A round that ends WITHOUT a valid manifest scores 0 no matter how
   much compiler you built — make the package structurally gradeable EARLY, THEN iterate on real
   codegen. If you run low on time, a graded-but-imperfect package beats an ungradeable one.
5. Author the backend as an **xDSL pass pipeline** (`xdsl_dialects/`, `targetgen/synthesize/`, `targetgen/generate/`) — structured IR passes, NOT ad-hoc string assembly, and with **NO regular expressions** (`import re` / regex text-matching is prohibited; parse the IR structurally). This is checked on your submission.
6. Enumerate your lever set: run `python cca_contract.py check-bijection gemmini` + `python action_catalog.py escalation-ladder <axis> gemmini` (runnable CLIs, like `isa_tools.py`) and build every leverable axis they list.
7. Produce your instruction words with `python isa_tools.py asm ops.txt` rather than hand-packed shifts: it takes every field POSITION from this target's own derived ISA model, whereas a hand-built `(a << 32) | (b << 16) | ...` re-derives those positions by hand and is where a field quietly ends up in the wrong one. How you emit stays your call.
8. Before every self_check, run `python isa_tools.py lint` and `disasm` on your emitted submission/*.mlir and confirm every instruction decodes (nothing UNKNOWN or ambiguous) and the kernel halts. A clean `lint` is NOT enough — a correctly-NAMED instruction carrying a WRONG FIELD decodes cleanly and still diverges on the hardware plane. So also read the `disasm` operand fields back against what your command buffer declares for that same command — DMA/DRAM address and row pitch, readout dtype / element width, accumulate and dataflow bits, config scale — and reconcile every one that disagrees.

## Final status line (end of `submission/REPORT.md`) — write exactly one of:
1. "Backend passes all required public/dev capsules and is ready for hidden grading."
2. "Backend does not yet pass all required public/dev capsules; remaining failures listed by capsule + plane."
3. "Backend is not comparable because it violates the compiler/runtime/integrity boundary."
