# Task: generate a atlas MLIR out-of-tree target backend (capsule_bench — REALISTIC)

You are an autonomous agent. Produce a **non-exempt out-of-tree MLIR target backend package** for the
atlas accelerator under `submission/`. Your package is graded — through its CLI entrypoints only,
never imported — by compiling workload **capsules** (interface MLIR) and matching the target's reference
behavior. This is a **compiler/backend** task: your COMPILER generates the target artifact by lowering
the interface — you never author a compute kernel.

## Scope
Make **every** public/dev capsule under the declared corpus pass. Families are discovered, not restated:
- `merlin/contract/capsules/atlas/isa/`
- `merlin/contract/capsules/atlas/layers/`
- `merlin/contract/capsules/atlas/model_slices/`
Read each capsule's `capsule.yaml` + `capsule.interface.mlir` for its op/shapes/dtypes/epilogue, and the
target-agnostic contracts (`command_buffer_abi.yaml`, `interface_grammar.md`, the command-buffer schema).
Derive everything (rounding, tiling, dtypes, im2col, padding) from the contract + the target's own docs
below — nothing is restated here. The numeric reference golden is withheld; iterate against the QA gate.
Build ONE general backend for every family — do not special-case individual capsules.

## Deliverable (write into `submission/`)
```
submission/
  manifest.yaml   # artifact_type: mlir_oot_target_backend; target: atlas; language: cpp|python;
                  # integrity_exempt: false; (cpp) a build block; the 4 command argv templates
  mlir_oot/       # your OOT sources: input dialect + atlas target dialect + passes + atlas-opt
  REPORT.md       # what you built + honest scope/limitations + a final status line (see end)
  docs/           # PLAN.md (first-round design plan) + public_facts_used.md (facts used + source) + iteration_notes.md
```

## The 4 CLI entrypoints (your package is invoked ONLY via these)
- `parse`: `{tool} --verify-diagnostics {input_mlir}` — parse + verify the `merlin_iface` interface MLIR
- `lower_interface_to_target`: `{tool} --convert-iface-to-atlas {input_mlir}` — emit atlas-dialect MLIR
- `emit_command_buffer`: `{tool} --emit-command-buffer={output_json} {input_mlir}` — schema-valid `command_buffer.json`
- `emit_target_artifact`: `{tool} --convert-iface-to-atlas --emit-target-artifact {input_mlir}` — emit a `kernel.S` of `.word`/`.insn` directives — the target's OWN encoded instructions (self-hosted ISA), assembled to IMEM words by STOCK LLVM (`llvm-mc`), then run on the target's cosim/RTL; no forked toolchain: a `kernel.S` of `.word`/`.insn` directives encoding the target's OWN instructions (compute each 32-bit encoding from the opcode/funct/field layout in the ISA definition shipped in your bundle; the bundled example kernel shows the required instruction sequence) that STOCK LLVM (`llvm-mc -triple=riscv64`) assembles into IMEM words — emit assembler text ONLY: NOT an MLIR module, NOT `llvm.inline_asm`, NOT the model's mnemonic assembler syntax (stock LLVM cannot assemble the target's custom mnemonics) driving the discovered 32x32 systolic mesh; the emitted module defines `atlas_kernel`

Declare these four commands in `manifest.yaml` exactly as the runner expects — see the OOT backend
contract (`mlir_oot_backend_contract.yaml`) and the manifest schema (`schemas/manifest.schema.json`).

## DRAM address map (self-hosted ISA — your kernel and the oracle must agree)
The program oracle runs your assembled kernel on the target's cosim: it PRELOADS each input tensor into
DRAM and reads the OUTPUT tensor back from DRAM. So in `command_buffer.json` you MUST declare **every
tensor your kernel touches** — inputs, weights, AND the output — each as `{{shape, dtype, role}}`, and
give each a `base` (its DRAM byte address). Your kernel must load each input from, and store the output
to, EXACTLY those `base` addresses — the oracle preloads inputs and captures the output there. The output
tensor MUST appear (its result is read from its `base`); omit it and the grade cannot see your answer. All
addresses must lie inside the DRAM region defined by your ISA memory map (the oracle relocates that region
to the model's aperture, so an address below the DRAM base cannot be indexed). If you omit a `base` the
harness assigns a canonical one inside that same DRAM region, but then your kernel must target that layout —
declaring them yourself is the reliable path. Addresses are per-capsule; size them from each tensor's
shape x dtype.

## Program termination (REQUIRED — a non-halting kernel fails before numerics)
The functional oracle runs your assembled kernel to a fixed instruction/cycle cap and then STOPS. Your
emitted program MUST reach the target's terminating instruction — the one the ISA definition marks as
asserting the machine's halt/done signal — on every control path (the shipped example kernel ends with
it). If it never halts, the capsule fails at the functional tier (`did not halt within N instructions`)
and the numeric comparison never runs: a numerically-correct kernel that does not terminate still scores
0. Derive the terminator's exact encoding from the ISA definition (do not invent it), and emit it as the
final instruction of your kernel.

For this target the terminator is **EBREAK / ECALL**, which the ISA's own encoder emits as `.word 0x00000073` (operands zero) — verify with the ISA dev tools' `disasm`/`lint`, and make it the final instruction on every path.

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
Per capsule the runner certifies the emitted artifact's program-oracle output against an INDEPENDENT float `golden` within the capsule's declared tolerance (its `grade_policy` atol/rtol) across the sim tier ladder; the integer `reference(cb) == simulate(cb)` self-consistency cross-checks do NOT apply to a float datapath and report `not_applicable` — derived from the corpus goldens, not restated:
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
**Shipped atlas ISA — the source of truth for instruction encodings (derive, never invent):**
The real atlas ISA is shipped read-only in your bundle. Derive EVERY instruction's
exact encoding from these files. Do NOT invent opcodes, mnemonics, instruction classes, or a
bit layout: a plausible-but-invented encoding assembles cleanly yet decodes to garbage on the
target and scores 0 (this is the single most common failure on a self-hosted ISA).
- `experiments/capsule_bench/targets/atlas/contracts/hwbringup_atlas_v0/isa_include/atlas_isa_green_card.md`
- `experiments/capsule_bench/targets/atlas/contracts/hwbringup_atlas_v0/isa_include/isa_definition.py`
- `experiments/capsule_bench/targets/atlas/contracts/hwbringup_atlas_v0/` (also mounted as `atlas/`) — RTL + ISA headers + README + a WORKED
  example kernel under `example_kernel/`. Translate the example's real instructions into
  your emitted encoding using the exact field layout the ISA definition specifies; the
  legal-opcode values in the ISA facts below are DECODE GATES, not the instruction
  semantics — take semantics + field packing from these files, never from the value list.

# Target ISA facts: atlas
_Derived by static CIRCT HW-dialect discovery (no model run). 2/4 fields grounded; ungrounded = unavailable, not guessed._

- **Legal opcodes** (42): `[87, 107, 119, 123, 215, 247, 251, 343, 375, 471, 503, 599, 631, 727, 759, 855, 887, 983, 1015, 1143, 1271, 1399, 1527, 1655, 1783, 4311, 4695, 4951, 8279, 8407, 8535, 8663, 8791, 8919, 9047, 9175, 9303, 9431, 9559, 9687, 9815, 9943]`
  - source: decoder_icmp_fanout(mlc)
- **Mesh DIM**: 32
- **On-chip capacity**: unavailable

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
5. Scaffold the package with the granted C++ OOT generators (`targetgen/generate/{mlir_scaffold,llvm_plan,target_repo}`), not ad-hoc hand files.

## Final status line (end of `submission/REPORT.md`) — write exactly one of:
1. "Backend passes all required public/dev capsules and is ready for hidden grading."
2. "Backend does not yet pass all required public/dev capsules; remaining failures listed by capsule + plane."
3. "Backend is not comparable because it violates the compiler/runtime/integrity boundary."
