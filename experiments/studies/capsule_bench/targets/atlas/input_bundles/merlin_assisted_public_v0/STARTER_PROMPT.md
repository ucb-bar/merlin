# Task: generate a atlas MLIR out-of-tree target backend (capsule_bench — FULL SUITE)

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
  docs/           # public_facts_used.md (every target fact you used + its source) + iteration_notes.md
```

## The 4 CLI entrypoints (your package is invoked ONLY via these)
- `parse`: `{tool} --verify-diagnostics {input_mlir}` — parse + verify the `merlin_iface` interface MLIR
- `lower_interface_to_target`: `{tool} --convert-iface-to-atlas {input_mlir}` — emit atlas-dialect MLIR
- `emit_command_buffer`: `{tool} --emit-command-buffer={output_json} {input_mlir}` — schema-valid `command_buffer.json`
- `emit_target_artifact`: `{tool} --convert-iface-to-atlas --emit-target-artifact {input_mlir}` — emit a `kernel.S` of `.word`/`.insn` directives — the target's OWN encoded instructions (self-hosted ISA), assembled to IMEM words by STOCK LLVM (`llvm-mc`), then run on the target's cosim/RTL; no forked toolchain: a `kernel.S` of `.word`/`.insn` directives encoding the target's OWN instructions (compute each 32-bit encoding from the opcode/funct/field layout in the ISA definition shipped in your bundle; the bundled example kernel shows the required instruction sequence) that STOCK LLVM (`llvm-mc -triple=riscv64`) assembles into IMEM words — emit assembler text ONLY: NOT an MLIR module, NOT `llvm.inline_asm`, NOT the model's mnemonic assembler syntax (stock LLVM cannot assemble the target's custom mnemonics); the emitted module defines `atlas_kernel`

Declare these four commands in `manifest.yaml` exactly as the runner expects — see the OOT backend
contract (`mlir_oot_backend_contract.yaml`) and the manifest schema (`schemas/manifest.schema.json`).

## DRAM address map (self-hosted ISA — your kernel and the oracle must agree)
The program oracle runs your assembled kernel on the target's cosim: it PRELOADS each input tensor into
DRAM and reads the OUTPUT tensor back from DRAM. So in `command_buffer.json` you MUST declare **every
tensor your kernel touches** — inputs, weights, AND the output — each as `{{shape, dtype, role}}`, and
give each a `base` (its DRAM byte address). Your kernel must load each input from, and store the output
to, EXACTLY those `base` addresses — the oracle preloads inputs and captures the output there. The output
tensor MUST appear (its result is read from its `base`); omit it and the grade cannot see your answer. If
you omit a `base` the harness assigns a canonical one, but then your kernel must target that same layout —
declaring them yourself is the reliable path. Addresses are per-capsule; size them from each tensor's
shape x dtype.

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
_Derived by static CIRCT HW-dialect discovery (no model run). 0/4 fields grounded; ungrounded = unavailable, not guessed._

- **Legal opcodes**: unavailable (no decoder signal / no HW dialect for this target)
- **Mesh DIM**: unavailable
- **On-chip capacity**: unavailable

## Menu of OOT modification points (merlin_assisted — the machine-checkable lever set)
The granted CCA spine is not just files to read: two answer-free calls ENUMERATE the full,
target-specific set of compiler seams you may modify for `atlas`, so you build the right lever set
instead of guessing from the file tree (neither imports the oracle or the grader):
- `cca_contract.check_bijection("atlas")` — the *what-to-build* checklist: which lever axes this
  target's ISA/RTL admits vs. which the compiler already routes (`orphan_fields` = leverable axes still
  to wire; `orphan_routes` = routes with no backing lever). Build every leverable axis; add no phantom.
- `action_catalog.escalation_ladder(axis, "atlas")` — for one axis, the full
  FLAG→KNOB→HEURISTIC→PASS→CODEGEN ladder weakest→strongest, each row naming the concrete OOT-relative
  seam file to edit and whether it is forkable today (the "which section, and the next stronger lever"
  answer). The seams point at YOUR generated OOT package, not our in-tree reference.

## Final status line (end of `submission/REPORT.md`) — write exactly one of:
1. "Backend passes all required public/dev capsules and is ready for hidden grading."
2. "Backend does not yet pass all required public/dev capsules; remaining failures listed by capsule + plane."
3. "Backend is not comparable because it violates the compiler/runtime/integrity boundary."
