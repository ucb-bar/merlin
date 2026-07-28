# Task: generate a Gemmini MLIR out-of-tree target backend (capsule_bench_v0 — FULL SUITE)

You are an autonomous agent. Produce a **non-exempt out-of-tree MLIR target backend package** for the
Gemmini accelerator under `submission/`. Your package is graded — through its CLI entrypoints only,
never imported — by compiling workload **capsules** (interface MLIR) and matching real Gemmini
reference behavior. This is a **compiler/backend** task, not a kernel-writing task.

## Scope: ALL public/dev capsules

Make **every** public/dev capsule under `merlin/contract/capsules/` pass. These span the full ISA,
layer, and model-slice families:

- **ISA (`isa/`):** config, mvin/mvout **movement**, single-tile **matmul**, **K-accumulation**,
  **acc_scale → i8**, **relu** epilogue, **resident reuse**, **edge padding** (non-DIM-multiple tiles).
- **Layers (`layers/`):** quantized linear (i8), linear+relu, linear+acc_scale+relu, **conv2d via
  im2col** (i8), conv2d+relu.
- **Model slices (`model_slices/`):** MLP linear1, MLP activation+linear2, attention **Q/K/V
  projections**, attention **QK^T** matmul, attention **PV** matmul.

Read each capsule's `capsule.yaml` + `capsule.interface.mlir` for its exact op, shapes, dtypes, and
epilogue; read `merlin/contract/command_buffer_abi.yaml` for precise epilogue/`acc_scale` semantics,
`merlin/contract/interface_grammar.md` for the input grammar, and `schemas/command_buffer.schema.json`
for the command-buffer schema. Derive everything (rounding rule, tiling, dtypes, im2col, padding) from
the contract + the public Gemmini header — nothing is restated here. Each capsule dir gives you
`capsule.yaml`, `capsule.interface.mlir`, and `expected_instruction_coverage.yaml`. The numeric
`golden.yaml` is intentionally withheld — you do NOT get the answers; see the QA gate below. Build one
**general** backend that handles every family — do not special-case individual capsules.

## Deliverable (write into `submission/`)

```
submission/
  manifest.yaml          # artifact_type: mlir_oot_target_backend; target: gemmini; language: cpp|python;
                         # integrity_exempt: false; (cpp) a build block; the 4 command argv templates
  mlir_oot/              # your OOT sources: input dialect + gemmini target dialect + passes + gemmini-opt
  REPORT.md              # what you built + honest scope/limitations + final status line (see end)
  docs/public_facts_used.md   # every Gemmini-specific fact you used, with its source
  docs/iteration_notes.md     # what failed, what you changed, which capsule, failure plane, better/worse
```

## The 4 CLI entrypoints (your package is invoked ONLY via these)

- `parse`: `{tool} --verify-diagnostics {input_mlir}` — parse + verify the `merlin_iface` interface MLIR
- `lower_interface_to_target`: `{tool} --convert-iface-to-gemmini {input_mlir}` — emit gemmini-dialect MLIR (must parse + `verify()`)
- `emit_command_buffer`: `{tool} --emit-command-buffer={output_json} {input_mlir}` — schema-valid `command_buffer.json`
- `lower_target_to_llvm`: `{tool} --convert-iface-to-gemmini --convert-gemmini-to-llvm-rocc {input_mlir}` — `llvm.func @gemmini_kernel` of RoCC `.insn r 0x7b` instructions

Declare these in `manifest.yaml` exactly as the runner expects (see `merlin/contract/
mlir_oot_backend_contract.yaml` and `schemas/manifest.schema.json`).

## How you are graded, and your QA signal

For each capsule the runner does: parse → gemmini dialect → `command_buffer.json` (schema-valid) →
LLVM/RoCC MLIR, then **decodes your RoCC into an instruction trace** and certifies three-way
**exact-integer** `golden == reference(cb) == simulate(cb) == oracle` on spike + **verilator RTL
(cycle-accurate)**, and checks the **required instruction classes** per capsule. Every capsule requires
passing through **L3 verilator (real RTL)** — spike alone is not sufficient. Integer numerics are EXACT
(no tolerance).

**You cannot run the oracle yourself** (no spike/verilator/reference here). After each round a **QA
gate** grades your current `submission/` and writes a **redacted verdict** to `qa/verdict.json` in your
working directory, per capsule: `status`, `failure_plane`, `trace_violations`, `numeric_status`,
`mismatch_count`, `tiers` (L0–L3), and `all_pass`. **It contains NO expected/golden values.** **Read
`qa/verdict.json` at the start of each round** and fix the failing capsules by their `failure_plane` +
`trace_violations`. You are relaunched across rounds; your `submission/` and `qa/verdict.json` persist.
Iterate until `all_pass: true` across all public/dev capsules.

Useful self-checks you CAN run locally (no oracle needed): build your tool, run the 4 entrypoints on
each `capsule.interface.mlir`, and confirm the command_buffer validates against
`merlin/contract/schemas/command_buffer.schema.json` and your lowered LLVM/RoCC text looks right.

## Hard rules (integrity — your submission must satisfy these)

- `integrity_exempt: false`; no `import merlin` / `from merlin` / `merlin.runtime.reference` /
  `merlin.runtime.simulator` / `reference_outputs` in your sources.
- The device implementation must be **MLIR-lowered RoCC** — **no C compute kernels**, **no copying or
  calling bareMetalC**, **no high-level Gemmini C library kernels** (e.g. `tiled_matmul_auto`) as the
  answer. Integer math is **exact**.
- **Never hardcode or embed outputs.** The grader runs hidden capsules (different deterministic data,
  same op families) after you freeze; a backend that memorizes public answers will fail them. Compute
  genuinely with one general backend.
- Do not attempt to read withheld goldens, hidden capsules, prior backends, or Merlin internals
  (they are masked/denied — do not work around the sandbox).

## Final status line (end of `submission/REPORT.md`) — write exactly one of:

1. "Backend passes all required public/dev capsules and is ready for hidden grading."
2. "Backend does not yet pass all required public/dev capsules; remaining failures are listed by capsule and failure plane."
3. "Backend is not comparable because it violates the compiler/runtime/integrity boundary."

Iterate against the public/dev capsules until they all pass, then stop.
