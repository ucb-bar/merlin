---
title: LLVM integration
kind: guide
status: current
owner: runtime
last_verified: 2026-07-14
related: [lowering_pipeline]
code_refs: [merlin/python/merlin/llvmlower]
---

# LLVM integration

LLVM-project modifications are **optional and late-stage**. The MVP requires no LLVM backend
patches. TargetGen emits an `llvm_extension_plan.yaml` first; real changes come only when a
target needs genuine instruction/register/codegen support.

## Three modes

```
Mode 0: no LLVM changes
  target dialect -> command buffer / simulator / Zephyr driver / C runtime calls
Mode 1: out-of-tree LLVM/MLIR extension
  external TableGen fragments, MLIR dialects, intrinsic headers, runtime calls, patch series
Mode 2: LLVM fork
  new backend / target registration / register classes / instruction selection /
  assembler-disassembler / MC encoding / RISC-V custom extension integration
```

`third_party/llvm-project` is a pinned upstream clone (a git submodule). It is **not modified
by default**; a fork is created only when `fork_triggers` in the plan are hit.

## What lives out-of-tree

Inside a generated target repo's `llvm/`:

```
llvm/
├── llvm_extension_plan.yaml   # requires_llvm_fork, initial_strategy, fork_triggers
├── td/        # reviewable .td fragments (empty during MVP)
├── patches/   # reviewable patch series against pinned LLVM (none during MVP)
└── tests/     # lit/codegen/asm tests expecting a patched LLVM (empty during MVP)
```

## Per-target default posture

| Target   | requires_llvm_fork | initial strategy |
| -------- | ------------------ | ---------------- |
| toy_npu  | false              | runtime calls / command buffer |
| gemmini  | false              | C/RoCC wrapper calls; patch only if custom-instruction emission needed |
| saturn   | maybe              | RVV intrinsics / existing RISC-V vector path (custom extensions may later need TableGen/backend) |
| radiance | false              | command-processor packets / external SIMT toolchain |

## Hard rule

Do not write LLVM backend patches before target-dialect + simulator validation passes. A
fork is justified only when real machine-instruction emission is required and the simulator/
runtime path already shows end-to-end benefit.
