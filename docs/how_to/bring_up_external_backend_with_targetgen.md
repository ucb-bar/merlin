# Bring Up An External Backend With TargetGen

This guide walks through using `./merlin targetgen` to bring up a new
hardware target — an external MLIR dialect, a Chipyard generator, an RTL
block, an ISA-doc-only target, or a full GPU backend — into the Merlin
compiler/runtime stack.

The flow is deterministic and reviewable. The planner reads source trees
and emits patch surfaces; an LLM (Claude Code via the MCP server) executes
the edits inside the surfaces TargetGen identifies.

## TL;DR

```bash
# 1. Inspect the target source.
./merlin targetgen ingest \
    --target-name my_target \
    --source /path/to/target/repo \
    --source /path/to/target/docs

# 2. Classify and emit a starting capability draft.
./merlin targetgen classify \
    --target-name my_target \
    --from-dir build/generated/targetgen

# 3. Promote the draft to target_specs/examples/my_target/capability.yaml,
#    fill in vendor/maturity/ISA fields, and validate.
./merlin targetgen validate target_specs/examples/my_target/capability.yaml

# 4. Get the per-stage modification map.
./merlin targetgen modification-map \
    target_specs/examples/my_target/capability.yaml

# 5. (Optional) launch the MCP server for Claude Code.
./merlin targetgen mcp
```

## Step 1 — Ingest the source

`targetgen ingest` walks one or more source paths and emits a
`SourceInventory` describing what it found. Each scanner is deterministic
(no LLM):

| Scanner | Detects |
| --- | --- |
| `mlir` | `*Ops.td`, `*Dialect.td`, `*Passes.td`, MLIR registration calls |
| `cmake` | `add_mlir_dialect`, `iree_register_external_hal_driver`, `iree_register_compiler_plugin`, FetchContent/ExternalProject |
| `llvm` | `IntrinsicsRISCV.td`, `RISCVFeatures.td`, files under `llvm/lib/Target/`, inline asm |
| `hal` | `driver.c`, `device.c`, `iree_hal_*` symbols |
| `chipyard` | `build.sbt`, `generators/`, FireSim collateral |
| `chisel` | RoCC / MMIO / TileLink / AXI4 / BlackBox attachment kinds |
| `rtl` | Verilog/SystemVerilog modules, AXI/TileLink ports, DPI-C |
| `systemc` | `SC_MODULE`, `tlm::`, `b_transport` |
| `docs` | ISA / Memory / Synchronization / Runtime / Driver / Simulator / Build headings in Markdown/RST |

Outputs:

- `build/generated/targetgen/<target>/source_inventory.json`
- `build/generated/targetgen/<target>/evidence_graph.json`

## Step 2 — Classify

`targetgen classify` consumes the inventory and emits two layers of
integration styles:

- **Source-facing styles** describe what the target is:
  `external_mlir_bridge`, `external_toolchain_bridge`, `chipyard_generator`,
  `rocc_accelerator`, `mmio_accelerator`, `rtl_or_systemc_model`,
  `llvm_backend_extension`, `gpu_codegen_stack`.
- **TargetGen styles** are the existing four Merlin styles already understood
  by the planner: `runtime_hal`, `structured_text_isa`,
  `post_global_plugin`, `llvm_ukernel`.

Mapping:

| Source style | TargetGen styles |
| --- | --- |
| `external_mlir_bridge` | `post_global_plugin` |
| `external_toolchain_bridge` | `post_global_plugin`, `runtime_hal` |
| `chipyard_generator` | `runtime_hal` |
| `rocc_accelerator` | `post_global_plugin`, `llvm_ukernel` |
| `mmio_accelerator` | `runtime_hal` |
| `rtl_or_systemc_model` | `runtime_hal` |
| `llvm_backend_extension` | `llvm_ukernel` |
| `gpu_codegen_stack` | `post_global_plugin`, `structured_text_isa`, `runtime_hal` |

Outputs:

- `build/generated/targetgen/<target>/classification.json`
- `build/generated/targetgen/<target>/capability.draft.yaml` — promote and
  enrich, then place at `target_specs/examples/<target>/capability.yaml`.

## Step 3 — Modification map

`targetgen modification-map` reads a capability spec and emits a
nine-stage map. Stages match the Merlin compilation pipeline:

1. `ml_framework_import` — `models/<target>.yaml`, `target_specs/`.
2. `linalg_arith_dialect` — `compiler/src/merlin/Dialect/<Target>/{IR,Transforms}/`.
3. `global_optimization` — `compiler/plugins/target/<Target>/`,
   `iree_compiler_plugin.cmake` (matches
   [`add_compiler_dialect_plugin.md`](add_compiler_dialect_plugin.md)).
4. `dispatch_generation` — Merlin dialect Transforms.
5. `data_tiling` — branched per integration style; can include IREE Codegen
   and ukernel paths.
6. `dispatch_scheduling` — runtime HAL driver scheduling code.
7. `executable_sources_llvm_intrinsics` — LLVM RISCV intrinsics, custom
   exporters, embedded binaries.
8. `vm_hw_synchronization` — HAL device sync primitives.
9. `hal_driver` — `runtime/src/iree/hal/drivers/<target>/`,
   `iree_runtime_plugin.cmake`, `./merlin build` (matches
   [`add_runtime_hal_driver.md`](add_runtime_hal_driver.md)).

Each stage reports `applies`, a `reason`, `read_paths`, `write_paths`,
`validation_commands` (always `./merlin …`), and any `blocking_questions`
the source did not answer.

Outputs:

- `build/generated/targetgen/<target>/modification_map.json`
- `build/generated/targetgen/<target>/modification_map.md`

## Step 4 — Stage mutation, then implement

The existing `./merlin targetgen stage-mutation` flow remains the canonical
non-live mutation path. It produces a reviewable `proposed_tree/` you can
diff before adopting any change.

For LLM-driven implementation, launch the MCP server (next section) and use
the [`/merlin-targetgen`](../../.claude/commands/merlin-targetgen.md)
Claude Code command, which enforces the procedure: ingest → classify →
plan → modification-map → allowed patch surfaces → small edit → validate.

## Examples

| Target shape | Run |
| --- | --- |
| External MLIR dialect (cuda-tile-style) | `./merlin targetgen ingest --target-name cuda_tile --source /path/to/cuda-tile` → `external_mlir_bridge` → `post_global_plugin` |
| Chipyard RoCC accelerator (Gemmini-style) | `./merlin targetgen ingest --target-name new_rocc --source /path/to/chipyard/generators/new_rocc` → `rocc_accelerator + chipyard_generator` → `post_global_plugin + llvm_ukernel + runtime_hal` |
| GPU with ISA + HAL (Radiance-style) | `./merlin targetgen ingest --target-name new_gpu --source /path/to/gpu/runtime --source /path/to/gpu/docs` → `gpu_codegen_stack` → `post_global_plugin + structured_text_isa + runtime_hal` |
| MMIO FFT generator | `./merlin targetgen ingest --target-name fft --source /path/to/fft_generator` → `mmio_accelerator + chipyard_generator` → `runtime_hal` |
| RISC-V CPU extension (SpacemiT/Saturn-style) | Skip ingest; write capability.yaml directly with `isa.exposure.kind = llvm_intrinsics` → `llvm_ukernel` |
