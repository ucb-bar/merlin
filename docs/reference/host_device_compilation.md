---
title: Host and device compilation
kind: reference
status: current
owner: runtime
last_verified: 2026-10-04
related: [runtime, zephyr, adding_a_target, experiment_abi]
code_refs: [src/merlin/targetgen/contract/build_recipe.py, src/merlin/targetgen/contract/build_service.py, src/merlin/targetgen/contract/compile.py, src/merlin/runtime/backends/base.py, merlin/runtime/c]
---

# Host and device compilation

Merlin has two different kinds of host work. **Host computation** is a
qualified CPU implementation of a source operation. **Host control** allocates
buffers, moves data, submits accelerator work, waits, and manages model state.
An accelerator launch may need host control even in strict-native mode; it
does not make an unsupported tensor computation a supported host route.

The compiler should produce one execution plan with separate, explicit assets:

```text
captured graph and declared constants
  → checked region partition and typed edge ABI
  ├─ host compute regions → LLVM MLIR → host objects
  └─ accelerator regions  → target OOT compiler → target program assets
  → host control and target adapter → linked host image + device assets
  → selected bare-metal, Zephyr, Linux, or simulator executor
```

The partition, numerical policy, buffer representation, output order, state,
and transfer obligations are compiler decisions. The execution environment
must not quietly move a failed required-native region to host computation.
Runtime input tensors are passed at invocation; expected outputs and grading
data belong only to the evaluator.

This diagram is the intended whole-model boundary, not a claim that all
adapters are connected today. `MerlinProgram` carries a target-independent
dispatch and memory plan, but its C replay consumer is absent and the builder
has no production caller outside tests. Atlas's current `execution_plan.json`
describes one diagnostic kernel, not a mixed host/device model invocation.
The existing whole-model Zephyr route compiles RVV host work; it does not yet
load and schedule Atlas device programs. A combined artifact builder and
compiled mixed-route replay still need implementation and qualification.

## Shared build boundary

[`HarnessBuildRecipe`](../../src/merlin/targetgen/contract/build_recipe.py)
already supplies the target-independent host compile/link command shape: a
selected compiler, flags and ISA, include roots, support sources, linker
script, link flags, and static stack policy. The
[`BuildOnlyService`](../../src/merlin/targetgen/contract/build_service.py)
pins a trusted renderer and its source bytes without importing an execution
backend or reference model. The generic
[`compile_lowered_to_elf`](../../src/merlin/targetgen/contract/compile.py)
path translates LLVM MLIR, builds an object with the target recipe's ISA, then
links it to a runner-owned harness. A target adapter supplies the recipe; core
code must not select compiler flags or target instructions by target name.

The existing model C substrate in [`merlin/runtime/c/`](../../merlin/runtime/c)
builds MLIR memref descriptors and invokes a compiled whole-model function.
Its `merlin_host_main.c` is a host verification driver. The tracked
`merlin_hal.h` describes a broader replay seam, but its documented
`merlin_program.c` consumer does not exist, so that header is not yet a
working universal accelerator dispatcher.

Target packages own device-specific code. A RoCC target can link its custom
instructions into the host ELF: the host core issues the instructions, and
the accelerator executes the commands. A self-hosted accelerator can instead
have a separate instruction binary that its host driver loads and starts.
Gemmini's bounded RoCC path uses the former shape; Atlas's selected diagnostic
uses the latter. The shared build recipe accepts either target's support
sources without pretending their launch and memory rules are identical.

## Runtime environment

The runtime ABI describes discovery, buffers, command submission, completion,
handles, metrics, and traces in [Runtime](runtime.md). Platform implementations
can be bare metal, Zephyr, Linux, or a simulator harness. Zephyr is an execution
environment and driver framework; it is not required for LLVM to compile the
host object or for a bare-metal bring-up ELF to run. A model may still need a
particular selected environment for its threading, memory, or I/O contract.

There is a working, separately scoped whole-model RVV Zephyr backend in
[`zephyr_model.py`](../../src/merlin/runtime/backends/zephyr_model.py). The
general target-module generator in
[`zephyr_module.py`](../../src/merlin/targetgen/generate/zephyr_module.py)
currently emits a non-building driver scaffold. Generating that scaffold does
not establish an Atlas or Gemmini Zephyr driver. A target's Zephyr adapter
must implement the actual register or RoCC protocol, memory visibility,
completion mechanism, errors, and metrics against a selected board.

The target-independent interface should keep the same invocation identity
across environments: program and constants identity, target revision, engine,
host and device ABI, input/output/state bindings, launch policy, and explicit
host/accelerator route accounting. An environment switch may change the host
image and driver, but cannot silently change the accelerator program,
numerical policy, or required-native coverage.

## What to qualify per target and platform

| Boundary | Shared responsibility | Target/platform evidence |
| --- | --- | --- |
| Host compile | LLVM translation and `HarnessBuildRecipe` invocation | ISA flags, linker, support sources, stack and ELF ABI |
| Device compile | Semantic region and artifact identity | Instruction selection, allocation, emission and target binary checks |
| Host/device edge | Typed buffers, state and transfer accounting | Address spaces, packing, visibility and ownership |
| Launch and completion | Explicit submit/wait result | MMIO/RoCC sequence, DMA lifetime, interrupt or polling rule |
| Execution | Separate host and accelerator routes | Actual emitted instruction execution and source-to-output checks |

For a new target, first provide a bare-metal or simulator adapter that proves
its launch and memory contract on bounded programs. Add a Zephyr binding when
the selected deployment requires Zephyr services. Both use the same compiler
assets and route accounting; neither substitutes for qualification of the
other platform. Kernel execution, full-model invocation, and physical SoC
execution remain distinct results.
