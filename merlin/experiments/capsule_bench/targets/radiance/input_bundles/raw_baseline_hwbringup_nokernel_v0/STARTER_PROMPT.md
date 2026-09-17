# Task: generate a radiance MLIR out-of-tree target backend (capsule_bench — REALISTIC)

You are an autonomous agent. Produce a **non-exempt out-of-tree MLIR target backend package** for the
radiance accelerator under `submission/`. Your package is graded — through its CLI entrypoints only,
never imported — by compiling workload **capsules** (interface MLIR) and matching the target's reference
behavior. This is a **compiler/backend** task: your COMPILER generates the target artifact by lowering
the interface — you never author a compute kernel.

## Scope
Make **every** public/dev capsule under the declared corpus pass. Families are discovered, not restated:
- `merlin/contract/capsules/radiance/isa/`
- `merlin/contract/capsules/radiance/layers/`
- `merlin/contract/capsules/radiance/model/`
- `merlin/contract/capsules/radiance/model_slices/`
Read each capsule's `capsule.yaml` + `capsule.interface.mlir` for its op/shapes/dtypes/epilogue, and the
target-agnostic contracts (`command_buffer_abi.yaml`, `interface_grammar.md`, the command-buffer schema).
Derive everything (rounding, tiling, dtypes, im2col, padding) from the contract + the target's own docs
below — nothing is restated here. The numeric reference golden is withheld; iterate against the QA gate.
Build ONE general backend for every family — do not special-case individual capsules.

## Deliverable (write into `submission/`)
```
submission/
  manifest.yaml   # artifact_type: mlir_oot_target_backend; target: radiance; language: cpp|python;
                  # integrity_exempt: false; (cpp) a build block; 4 required command argv templates
  mlir_oot/       # your OOT sources: input dialect + radiance target dialect + passes + radiance-opt
  REPORT.md       # what you built + honest scope/limitations + a final status line (see end)
  docs/           # PLAN.md (first-round design plan) + public_facts_used.md (facts used + source) + iteration_notes.md
```

## The 4 CLI entrypoints (your package is invoked ONLY via these)
- `parse`: `{tool} --verify-diagnostics {input_mlir}` — parse + verify the `merlin_iface` interface MLIR
- `lower_interface_to_target`: `{tool} --convert-iface-to-radiance {input_mlir}` — emit radiance-dialect MLIR
- `emit_command_buffer`: `{tool} --emit-command-buffer={output_json} {input_mlir}` — schema-valid `command_buffer.json`
- `emit_target_artifact`: `{tool} --convert-iface-to-radiance --emit-target-artifact {input_mlir}` — emit an LLVM-dialect MLIR kernel lowering (compiled fork-free): an LLVM-dialect MLIR module (`builtin.module` with `llvm.func @<kernel>`) — a COMPILER LOWERING your xDSL passes produce — which the runner compiles FORK-FREE (stock LLVM rv32 + the target's own RTL-derived instruction re-encode, no vendor fork) and runs on the cosim; NOT C/C++ source, NOT `.word`/`.insn` assembler, NOT a self-hosted kernel; the emitted module defines `radiance_kernel`

Declare these four commands in `manifest.yaml` exactly as the runner expects — see the OOT backend
contract (`mlir_oot_backend_contract.yaml`) and the manifest schema (`schemas/manifest.schema.json`).

For fast whole-model analysis, also declare the optional `emit_analysis_bundle` command when your
driver can run its lowering pipeline once while handling both output flags. It writes the command
buffer to `{output_json}` and the target artifact to stdout; the host feature-detects it and falls
back to the two required emission commands when it is absent. This is an execution optimization,
not a replacement for either independently required artifact or its validation.

### Also declare `components:` — it is what keeps your passing capsules passing
Add a top-level `components:` block mapping **each command you declared** to the submission-relative
files that implement it. A capsule certified on the expensive RTL tier keeps that certificate across any
edit that touches no component it rides on; without the block the harness has to work the mapping out
from your imports, and anything it cannot place invalidates every certificate you have earned.
```yaml
components:
  parse: [mlir_oot/frontend/]                  # only the files this command actually uses
  lower_interface_to_target: [mlir_oot/lowering/]
  emit_command_buffer: [mlir_oot/cmdbuf.py]
  emit_target_artifact: [mlir_oot/codegen/, mlir_oot/tables/]
  emit_analysis_bundle: [mlir_oot/]              # only when you declare the optional command
```
Rules: keys must be command names you declared (anything else is rejected and reported); paths are
submission-relative files or directories; the longest matching path wins, so a nested entry is not
swallowed by its parent. Keep it accurate rather than narrow — an under-declared component is how a
certificate outlives the code it was earned on.

### Declare the compiler's semantic optimization surfaces
Phase 2 automatically inventories the source symbols reachable from `components:` and joins measured
whole-model bottlenecks to this manifest map. Add top-level `optimization_surfaces:` entries for the
real places an optimization agent may change. This is not a list of hoped-for features: every `path`
must be inside a declared component, every `symbol` must be the exact Python AST name (`Class.method`
or function), and each entry must say what emitted artifact change would prove the mechanism fired.
Use only the shared effects accepted by the schema: placement, layout, dtype, encoding, quantization,
movement, residency, fusion, synchronization, issue, tiling, latency_hiding.
```yaml
optimization_surfaces:
  - id: schedule-selection
    scope: heuristic                    # flag | knob | heuristic | pass | codegen
    path: mlir_oot/lowering/schedule.py
    symbol: Scheduler.choose
    effects: [movement, residency, latency_hiding]
    cca_axes: [dispatch.dma_overlap, communication.resident_across_calls]
    mechanism: choose a legal whole-region schedule from derived target capabilities
    emitted_delta: fewer declared transfers or dependencies with identical required work
    validation: warm reduced witness, then complete-model analytical re-plan
    abandonment: no emitted delta, legality failure, or warm compute cycles do not improve
```
Do not invent a surface to satisfy the form. If a lever does not exist yet, implement the general
compiler mechanism first and then declare its real symbol. Phase 2 treats an absent or invalid mapping
as UNKNOWN rather than guessing from a filename.

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

## Durable run memory
The certified schedule keeps one continuing session, but the process can still resume after a timeout,
quota boundary, or operator restart. Do not rely on conversational memory alone. The harness writes
`qa/round_brief.md` with graded progress, and your own durable notes live in `docs/iteration_notes.md`.
Read both before changing an existing submission. After every substantive change, append what changed,
what the verdict showed, and the next hypothesis; do not undo a change that improved an earlier grade.

## Grading + your QA signal
This corpus is MIXED, so the grading model is decided PER CAPSULE by that capsule's own golden, not once for the target: a capsule carrying an INDEPENDENT float `golden` is certified against the program oracle within its declared tolerance (its `grade_policy` atol/rtol) and its integer `reference(cb) == simulate(cb)` self-consistency cross-checks report `not_applicable`; every other capsule is certified exact-integer `golden == reference(cb) == simulate(cb) == oracle` with no tolerance. Both apply across the sim tier ladder — derived from the corpus goldens, not restated:
- `L2` → cyclotron
- `L3` → verilator
and checks the required instruction coverage per capsule (it decodes your emitted artifact into an
instruction trace). You cannot run the oracle; a QA gate writes a redacted `qa/verdict.json` per
capsule — `status`, `failure_plane`, `trace_violations`, `numeric_status`, `mismatch_count`,
`tiers` (L0–L3), and `all_pass` — with NO golden/expected values.

**`qa/verdict.json` is refreshed WHILE YOU WORK, and it does not exist when you start.** Grading runs
on its own schedule in the background, so a single check at the beginning tells you nothing: the file
appears only after the first grade lands. Re-read it periodically — after each substantive change, and
whenever you are choosing what to work on next — and fix by `failure_plane` + `trace_violations`.
Its content is the only feedback you get; an early "not found" is a timing artifact, not an answer.
Iterate until `all_pass: true`.

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
**Your 4th artifact is an LLVM-dialect MLIR module (`submission/lowered.llvm.mlir`) defining `llvm.func @radiance_kernel` — a COMPILER LOWERING, not a hand kernel.** The runner compiles it FORK-FREE via the SHARED `llvmlower` front (MLIR → LLVM IR → STOCK clang rv32 object), then re-encodes it to the target's own ISA and runs it on the cosim. Contract:
1. Emit ONE `builtin.module` containing `llvm.func @radiance_kernel(...)`. Every argument is `!llvm.ptr`, in the ABI order **[weight] ++ [lhs in command order] ++ [outputs in command order]** (the generic kernel_abi); pointees are row-major f32.
2. The function COMPUTES the op (loads → multiply-accumulate → stores into the output pointers) and `llvm.return`s. It is plain scalar compute over the pointer operands — the SIMT warps / barriers / scheduling are the RUNTIME's (the fork-free BSP spawns warps around your kernel), so you do NOT write `mu_schedule`, barriers, or thread-id logic.
3. Emit NO prints, NO `.insn`/`.word`, NO DRAM base map, NO halt — the runner owns the harness (it embeds the operands, calls your kernel, and prints the `OUT`/`DONE` protocol) and the BSP owns boot/halt. A kernel that writes its output pointers and returns is complete.
4. BUILD the module with your xDSL pass pipeline (typed IR, `verify()`-checked) — NEVER by string assembly and NEVER with regex; this is checked on your submission.

The module skeleton (structure is mandatory; the reference backend emits this shape):
```mlir
module {
  llvm.func @radiance_kernel(%W: !llvm.ptr, %L: !llvm.ptr, %O: !llvm.ptr) {
    // O = L @ W  (row-major; M, K, N come from the interface). Emit the loads
    // (llvm.getelementptr + llvm.load), the multiply-accumulate (llvm.fmul / llvm.fadd), and the
    // stores (llvm.store) — or lower scf/arith loops to this. Your xDSL passes BUILD this IR.
    llvm.return
  }
}
```

# Target ISA facts: radiance
_Derived by RadianceMuonConfig elaborated FIRRTL type widths + module-instance hierarchy + SRAM macro list + device tree (CIRCT/firtool output only); cyclotron perf-model config used as a CROSS-CHECK, never as a fact source + static CIRCT HW-dialect discovery (no model run). 5/9 fields grounded; ungrounded = unavailable, not guessed._

- **Execution geometry**: lanes_per_warp=16, warps_per_core=8, cores=1, threads_per_core=128
  - source: merlin.targets.muon.backend.muon_introspect — lanes_per_warp: MuonCore io trace bits tmask : UInt<16> — one mask bit per lane; warps_per_core: MuonCore io perf perWarp : {...}[8] — one counter set per warp; cores: elaborated instance tree: 1 x MuonCore (1 x MuonTile, 1 x RadianceCluster)
- **Register budget**: arch_max=256, compiler_limit=128
  - source: merlin.targets.muon.backend.muon_introspect — MuonCore io trace regs address : UInt<8> -> 256 architectural registers
- **Shared memory**: bytes_per_cluster=131072
  - source: merlin.targets.muon.backend.muon_introspect — RTL SRAM macros under RadianceSharedMem: 64x radiance_smem_bank_ext (depth 512 x width 32b = 2048 B) = 131072 B/cluster
- **FP datapath**: dtype=f32, flop_per_fma=2, peak_flops_per_cycle=32, clock_hz=500000000, peak_gflops=16.0
  - source: merlin.targets.muon.backend.muon_introspect — 1 cores x 16 lanes x 2 flop/FMA = 32 flop/cycle @ 500 MHz = 16 GFLOP/s
- **Instruction encoding**: encoding_bits=64, max_src_operands=3, max_dst_operands=1, predicated_execution=True, address_spaces={'global': 0, 'shared': 1}
  - **instruction classes** (24): `['AUIPC', 'BRANCH', 'CUSTOM0', 'CUSTOM1', 'CUSTOM2', 'CUSTOM3', 'JAL', 'JALR', 'LOAD', 'LUI', 'MADD', 'MISC_MEM', 'MSUB', 'NM_ADD', 'NM_SUB', 'NU_COMPLETE', 'NU_INVOKE', 'NU_INVOKE_IMM', 'NU_PAYLOAD', 'OP', 'OP_FP', 'OP_IMM', 'STORE', 'SYSTEM']`
  - source: merlin.targets.muon.backend.muon_introspect — DERIVED from mlc isa_encoding fact (muon_isa.json): RTL-decoder field layout + opcode table + address-space macros; intrinsic names from lib/include/*intrinsics.h

## MANDATORY development workflow (do ALL of these BEFORE the final status line — not optional)
1. Your compiler backend lives under `submission/`; compute is COMPILER-GENERATED (never a hand kernel).
2. Base every ISA / mesh / datapath / encoding decision on the **Target ISA facts** above + the
   capability contract under `merlin/contract/` — never guess or hardcode; derive any fact not given.
3. After each substantive build, run `python3 agent_selfcheck.py --submission submission
   --capsules <changed-capsule-or-subset>` for the smallest affected scope. Run `--capsules all`
   at convergence milestones and once before declaring done — not after every edit. A submission
   you did not self-check is not acceptable. Also run `python3 agent_selfcheck.py --submission
   submission --shape-coverage`, which probes the
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

## Final status line (end of `submission/REPORT.md`) — write exactly one of:
1. "Backend passes all required public/dev capsules and is ready for hidden grading."
2. "Backend does not yet pass all required public/dev capsules; remaining failures listed by capsule + plane."
3. "Backend is not comparable because it violates the compiler/runtime/integrity boundary."
