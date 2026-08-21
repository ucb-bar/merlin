# Task: generate a vortex MLIR out-of-tree target backend (capsule_bench — REALISTIC)

You are an autonomous agent. Produce a **non-exempt out-of-tree MLIR target backend package** for the
vortex accelerator under `submission/`. Your package is graded — through its CLI entrypoints only,
never imported — by compiling workload **capsules** (interface MLIR) and matching the target's reference
behavior. This is a **compiler/backend** task: your COMPILER generates the target artifact by lowering
the interface — you never author a compute kernel.

## Scope
Make **every** public/dev capsule under the declared corpus pass. Families are discovered, not restated:
- `merlin/contract/capsules/vortex/isa/`
- `merlin/contract/capsules/vortex/layers/`
- `merlin/contract/capsules/vortex/model_slices/`
Read each capsule's `capsule.yaml` + `capsule.interface.mlir` for its op/shapes/dtypes/epilogue, and the
target-agnostic contracts (`command_buffer_abi.yaml`, `interface_grammar.md`, the command-buffer schema).
Derive everything (rounding, tiling, dtypes, im2col, padding) from the contract + the target's own docs
below — nothing is restated here. The numeric reference golden is withheld; iterate against the QA gate.
Build ONE general backend for every family — do not special-case individual capsules.

## Deliverable (write into `submission/`)
```
submission/
  manifest.yaml   # artifact_type: mlir_oot_target_backend; target: vortex; language: cpp|python;
                  # integrity_exempt: false; (cpp) a build block; the 4 command argv templates
  mlir_oot/       # your OOT sources: input dialect + vortex target dialect + passes + vortex-opt
  REPORT.md       # what you built + honest scope/limitations + a final status line (see end)
  docs/           # PLAN.md (first-round design plan) + public_facts_used.md (facts used + source) + iteration_notes.md
```

## The 4 CLI entrypoints (your package is invoked ONLY via these)
- `parse`: `{tool} --verify-diagnostics {input_mlir}` — parse + verify the `merlin_iface` interface MLIR
- `lower_interface_to_target`: `{tool} --convert-iface-to-vortex {input_mlir}` — emit vortex-dialect MLIR
- `emit_command_buffer`: `{tool} --emit-command-buffer={output_json} {input_mlir}` — schema-valid `command_buffer.json`
- `emit_target_artifact`: `{tool} --convert-iface-to-vortex --emit-target-artifact {input_mlir}` — emit an LLVM-dialect MLIR kernel lowering (compiled fork-free): an LLVM-dialect MLIR module (`builtin.module` with `llvm.func @<kernel>`) — a COMPILER LOWERING your xDSL passes produce — which the runner compiles FORK-FREE (mlir-translate → STOCK clang, no vendor/forked toolchain) and runs on the cosim; NOT C/C++ source, NOT `.word`/`.insn` assembler, NOT a self-hosted kernel; the emitted module defines `merlin_kernel_body`

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
Per capsule the runner certifies the emitted artifact's program-oracle output against an INDEPENDENT float `golden` within the capsule's declared tolerance (its `grade_policy` atol/rtol) across the sim tier ladder; the integer `reference(cb) == simulate(cb)` self-consistency cross-checks do NOT apply to a float datapath and report `not_applicable` — derived from the corpus goldens, not restated:
- `L2` → simx
- `L3` → rtlsim
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
**Your 4th artifact is an LLVM-dialect MLIR module (`submission/lowered.llvm.mlir`) defining `llvm.func @merlin_kernel_body` — a COMPILER LOWERING, not a hand kernel.** The runner compiles it FORK-FREE via the SHARED `llvmlower` front (MLIR → LLVM IR → STOCK clang object) and runs it on the cosim. Unlike a BSP-managed SIMT core, **your compiler OWNS the SIMT execution** — that mapping is the work under test (`compiler_obligations`: must_map_to_cta_grid, must_insert_reconvergence, must_respect_l2_coherence). Contract:
1. Emit ONE `builtin.module` containing `llvm.func @merlin_kernel_body(...)` — the declared kernel ABI is `void merlin_kernel_body(const merlin_vx_kernel_arg_t* arg)`. Its argument is one args-struct pointer `%arg: !llvm.ptr`. The device address of the i-th operand is `arg->args[i]` (the granted ABI header gives the exact struct layout); operands are in `merlin.arg_table` order (**the capsule's `inputs[]` order, as flat device addresses in `arg->args[]`**). Read each address, then use ordinary loads/stores at it.
2. The kernel runs ONCE PER (block, thread) coordinate the hardware launches. READ this coordinate's identity from the identity CSRs (a `csrr` inline-asm — the spec's CSR map gives the numbers); identity is NOT passed in. MAP this coordinate's slice of the iteration space onto the frozen machine geometry and compute it — do NOT assume one coordinate per output element.
3. Divergence is NOT automatic: a branch whose predicate differs across threads in a warp is silently WRONG unless bracketed by the ISA's split/join (see the reconvergence contract in the ISA spec). Uniform control flow and fully-predicated branch-free code need neither.
4. Emit NO prints and NO halt instruction — the kernel is a CALLEE that `llvm.return`s (the harness calls it per coordinate and owns boot/print/teardown). Respect the memory model for host-visible stores (see the spec's memory section). Also declare the `merlin.arg_table` (operand order) and `merlin.grid = <N> : i64` (block count) module attributes — a module with no `merlin.grid` is rejected.
5. Derive EVERY custom instruction + CSR encoding from the shipped ISA definition + spec sheet (do NOT invent one — verify with `isa_tools`). BUILD the module with your xDSL pass pipeline (typed IR, `verify()`-checked) — NEVER by string assembly and NEVER with regex; this is checked on your submission.

The module skeleton (structure is mandatory):
```mlir
module attributes {
    // declare BOTH: the operand order the harness binds to, and the block count your mapping assumes
    merlin.arg_table = ...,  merlin.grid = 64 : i64
} {
  llvm.func @merlin_kernel_body(%arg: !llvm.ptr) {
    // 1) read THIS coordinate's identity from the CTA CSRs (thread_id/block_id/... — see the spec's
    //    CSR map), via `csrr` in llvm.inline_asm; 2) load each operand's device address from
    //    arg->args[i]; 3) map THIS coordinate's slice of the iteration space and compute; 4) bracket
    //    any per-thread-divergent control in split/join (see the reconvergence contract). Then:
    llvm.return
  }
}
```

# Target ISA facts: vortex
_Derived by Vortex CIRCT HW-dialect import (hw/syn/circt -> Vortex-sv2v.mlir) walked with mlc.discover irgraph+decode; equality fan-out over eq AND four-state ceq. 1/4 fields grounded; ungrounded = unavailable, not guessed._

- **Legal opcodes**: unavailable (no decoder signal / no HW dialect for this target)
- **Mesh DIM**: unavailable
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

## Final status line (end of `submission/REPORT.md`) — write exactly one of:
1. "Backend passes all required public/dev capsules and is ready for hidden grading."
2. "Backend does not yet pass all required public/dev capsules; remaining failures listed by capsule + plane."
3. "Backend is not comparable because it violates the compiler/runtime/integrity boundary."
