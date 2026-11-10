---
title: 7. Build an image for the selected platform
kind: guide
status: current
owner: compiler
last_verified: 2026-11-09
related: [06-lowering, 08-running]
code_refs: [src/merlin/compiler/build/image.py, src/merlin/compiler/build/control.py, src/merlin/compiler/build/objects.py, src/merlin/compiler/build/zephyr, src/merlin/compiler/artifact.py]
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

`compiler/build/objects.py` constructs common object-compilation commands.
`build/control.py` emits shared C dispatch for supported wrappers.
`build/image.py` links an explicitly selected image using `CImageLink`.

The caller supplies the compiler, flags, startup compilation, link inputs and
`Deployment`. Startup and link order must match the selected ABI and platform.
The image builder does not guess a board from the compiler backend's name.
An explicitly selected image is optional; a valid compilation can stop at region
artifacts when no complete image was requested.

For a small host example, follow
[`examples/quickstart/run.py`](../../examples/quickstart/run.py). It checks the
emitted C interface, invokes a host C compiler and runs the resulting Linux ELF.
That step needs Linux x86-64 and a compatible compiler; reading the generated
source and notebook outputs does not.

## Bare metal and Zephyr

Bare-metal and Zephyr images share compiler mechanisms but have different
startup, memory and runtime contracts. The selected SoC affects memory regions,
device visibility, ISA features and the available execution resources.

Zephyr configuration is owned by `compiler/build/zephyr/`:

* `config.py` derives the selected image configuration.
* `tree.py` reads the selected Zephyr tree's available symbols and defaults.
* `memory.py` builds the image memory layout.

Do not copy a fixed `prj.conf` between SoCs and assume it describes the new
system. Record the actual board/tree/toolchain selection and derived configuration.
Hart count is part of the deployment contract when the image binds it; a runner
cannot silently override a built image's assumptions.

## Keep build outputs together

Write build products under `out/build/`. Retain the compilation manifest,
generated sources, object/binary artifacts and relevant tool/process records.
Use fresh output directories so a previous successful artifact cannot be
mistaken for the result of a failed build.

The build records establish what was produced and how. The next two chapters
cover what executed and whether its behavior was correct.
