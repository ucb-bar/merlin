# Task: generate a Gemmini MLIR out-of-tree target backend (graded by capsule_bench_v0)

You are an autonomous agent. Produce a **non-exempt out-of-tree MLIR target backend package** for the
Gemmini accelerator. Your package will be graded — through its CLI entrypoints only — by compiling a
suite of public/dev workload **capsules** (ISA / layer / model-slice) and matching real Gemmini
reference behavior. You will be measured from start to finish; you cannot ask questions.

## Deliverable (write into `submission/`)

```
submission/
  manifest.yaml          # artifact_type: mlir_oot_target_backend; target: gemmini; language: cpp;
                         # integrity_exempt: false; a build block; and the 4 command argv templates
  mlir_oot/              # your OOT MLIR sources (input dialect + gemmini target dialect + 3 passes
                         # + a `gemmini-opt` tool); builds to mlir_oot/build/bin/gemmini-opt
  REPORT.md              # what you built + honest scope/limitations
```

## The 4 CLI entrypoints (your package is invoked ONLY via these — never imported)

- `parse`: `{tool} --verify-diagnostics {input_mlir}` — parse + verify a `merlin_iface` interface MLIR
- `lower_interface_to_target`: `{tool} --convert-iface-to-gemmini {input_mlir}` — emit gemmini-dialect MLIR (must parse + `verify()`)
- `emit_command_buffer`: `{tool} --emit-command-buffer={output_json} {input_mlir}` — emit a schema-valid `command_buffer.json`
- `lower_target_to_llvm`: `{tool} --convert-iface-to-gemmini --convert-gemmini-to-llvm-rocc {input_mlir}` — emit `llvm.func @gemmini_kernel` of RoCC `.insn r 0x7b` instructions

## What you may read
The frozen contract `merlin/contract/` (schemas, `interface_grammar.md`, `command_buffer_abi.yaml`,
`integrity_policy.md`), the **public/dev capsules** under `merlin/contract/capsules/{isa,layers,
model_slices}/` (each has `capsule.yaml` + `capsule.interface.mlir` + `expected_instruction_coverage.yaml`),
the public Gemmini ISA header (`include/gemmini.h`, `gemmini_params.h`), and the LLVM/MLIR toolchain.

## How you are graded (capsule_bench_v0)
For each capsule: parse → gemmini dialect (parses+verifies) → `command_buffer.json` (schema-valid) →
LLVM/RoCC MLIR. The runner then certifies three-way **exact-integer** bit-exact
`golden == reference(cb) == simulate(cb) == oracle` on **spike (L2)** and **verilator RTL (L3)**, and
decodes your emitted RoCC to check the required instruction classes per capsule. Cycles are recorded
but are **diagnostic only** (never gate pass/fail). You pass when **all required public/dev capsules
pass**. Hidden capsules are run only after your submission is frozen.

## Hard rules (integrity — your submission must satisfy these)
- `integrity_exempt: false`; no `import merlin` / `from merlin` / `merlin.runtime.reference` /
  `merlin.runtime.simulator` / `reference_outputs` in your sources.
- The device implementation must be **MLIR-lowered RoCC** — **no C compute kernels**, **no copying or
  calling bareMetalC**, **no high-level Gemmini C library kernels** (e.g. `tiled_matmul_auto`) as the
  answer. Integer math is **exact** (no tolerance).
- Do not read hidden capsules/goldens or any prior backend.

Iterate against the public/dev capsules until they all pass, then stop.
