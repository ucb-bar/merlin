---
title: 7. Build an image for the selected platform
kind: guide
status: current
owner: compiler
last_verified: 2026-11-09
related: [06-lowering, 08-running]
code_refs: [src/merlin/compiler/build/image.py, src/merlin/compiler/build/regions.py, src/merlin/compiler/build/completion.py, src/merlin/compiler/build/objects.py, src/merlin/compiler/build/process.py, src/merlin/compiler/build/zephyr.py, src/merlin/compiler/system.py, src/merlin/compiler/records.py, src/merlin/runtime/backends/zephyr/image_build.py, src/merlin/runtime/boards/catalog.py, src/merlin/runtime/boards/image_audit.py, examples/compilation/quickstart/run.py]
---

# 7. Build an image for the selected platform

[Previous](06-lowering.md) | [Guide index](../README.md) | [Next](08-running.md)


A region artifact is not necessarily a runnable program. Building combines the
selected region products with control code, startup, runtime libraries and the
linker configuration for one deployment.

## Know which product you have

| Product | What it contains | Next step |
| --- | --- | --- |
| Compiler IR | An admitted intermediate representation | Continue the selected lowering pipeline |
| Region object | A compiled function and ABI requirements | Link it into an image |
| Device binary or commands | Backend-specific executable work | Supply its admitted wrapper/loader and runtime |
| Deployment-bound ELF | A complete linked image for a selected platform | Execute using a compatible runner |

Use `merlin inspect <compilation-directory>` to check the recorded artifact
hashes. It does not execute the program or establish numerical correctness.

## Shared image construction

[`compiler/build/objects.py`](../../src/merlin/compiler/build/objects.py)
exposes `compile_llvm_object(source, destination, *, compiler, flags, run, timeout_s=None)`.
The caller supplies the selected compiler, target flags and observed process
runner. `build/process.py` owns bounded execution and invocation receipts for
the shared build route.

[`build/regions.py`](../../src/merlin/compiler/build/regions.py) emits shared
C dispatch from the original selected partition and admitted callable wrappers.
`build/completion.py` retains the original caller's platform-completion admission.
[`build/image.py`](../../src/merlin/compiler/build/image.py) links an explicitly
selected image using `CImageLink`.

The caller supplies the compiler, flags, startup compilation, link inputs and
`Deployment`. `CImageLink` requires `compiler`, `flags`, a saved startup
`Compilation`, its ordered `link_inputs`, and `deployment`; `context_entry` is
optional. It rechecks the compiler/startup identities and corroborates supported
ELF header fields. Startup and link order must match the selected ABI and platform.
The image builder does not guess a board from the compiler backend's name.
An explicitly selected image is optional; a valid compilation can stop at region
artifacts when no complete image was requested.

For a small host example, follow
[`examples/compilation/quickstart/run.py`](../../examples/compilation/quickstart/run.py). It checks the
emitted C interface, invokes a host C compiler and runs the resulting Linux ELF.
That step needs Linux x86-64 and a compatible compiler; reading the generated
source and notebook outputs does not.

## Bare metal and Zephyr

Bare-metal and Zephyr images share compiler mechanisms but have different
startup, memory and runtime contracts. The selected SoC affects memory regions,
device visibility, ISA features and the available execution resources.

Zephyr configuration is consolidated in
[`compiler/build/zephyr.py`](../../src/merlin/compiler/build/zephyr.py):

| API | Responsibility |
| --- | --- |
| `TreeConfiguration` | Read symbols and unconditional literal defaults from the selected tree. |
| `project_config`, `image_cpus`, `cpu_disable_overlay` | Derive image configuration from the selected board, tree and hart choices. |
| `memory_plan`, `require_external_arena` | Check the declared DRAM layout and the linked image's remaining low-memory arena. |

These helpers choose no compiler and execute no simulator. The saved-model
Zephyr build route is `runtime/backends/zephyr/image_build.py:build_app`, using
the application sources in `runtime/backends/zephyr/application.py` and an
explicit board from the target-owned catalog.

Do not copy a fixed `prj.conf` between SoCs and assume it describes the new
system. Record the actual board/tree/toolchain selection and derived configuration.
`runtime/boards/image_audit.py` inspects the produced ELF against selected board
properties. Header or memory-fit checks do not establish startup behavior,
numerical correctness or actual device execution. Hart count is part of the
deployment contract when the image binds it; a runner
cannot silently override a built image's assumptions.

## Keep build outputs together

Write build products under `out/build/`. Retain the compilation manifest,
generated sources, object/binary artifacts and relevant tool/process records.
Use fresh output directories so a previous successful artifact cannot be
mistaken for the result of a failed build.

The build records establish what was produced and how. The next two chapters
cover what executed and whether its behavior was correct.
