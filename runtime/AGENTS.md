# `runtime/` — agent guide

## Mental model

Runtime-side C/C++ code that gets linked into the merlin-extended IREE
runtime. Mirrors the structure of upstream IREE's `runtime/src/iree/`
so the merlin extensions slot into the standard plugin discovery path.

| Path | Role |
|---|---|
| `runtime/src/iree/hal/drivers/<backend>/` | New HAL drivers (Radiance, QNN, …) that IREE's runtime discovers via `iree_runtime_plugin.cmake` at repo root. |
| `runtime/src/iree/builtins/ukernel/` | Architecture-specific ukernels (RVV custom kernels for SpacemiT X60, OPU custom instructions, …). |
| `runtime/src/iree/hal/local/elf/arch/` | Bare-metal ELF loader hooks for FireSim / Zephyr profiles. |

## Pitfalls

- **Plugin discovery is via `iree_runtime_plugin.cmake`.** If a new HAL
  driver doesn't show up at runtime, check the plugin manifest. Add the
  driver's CMake target name to the list there.
- **HAL driver init must register BEFORE first device-create.** Per-driver
  registration calls are emitted by the macro stack in
  `iree_runtime_plugin.cmake`. Don't try to hand-call them from a
  sample's `main()` — that bypasses the macro and breaks downstream.
- **Bare-metal vs hosted variants of the same driver are NOT
  interchangeable.** Spike / FireSim / Zephyr builds compile against a
  reduced libc; calling `dlopen` or hitting filesystem in the driver's
  hot path will work on the QRB5165 board and crash bare-metal.
- **API drift between merlin-pinned IREE and upstream.** [[iree_api_runner_port]]
  documents the `iree_hal_device_create_params_t` refactor — older
  samples in `samples/common/xpu-rt/` still use the pre-refactor API.

## Cross-references

- Built by: `./merlin build` with the matching profile. The default
  `--profile vanilla` includes core IREE only; `--profile full-plugin`
  links every backend.
- Consumed by: every binary in `samples/` (see `samples/AGENTS.md`).
- Companion compiler side: `compiler/src/merlin/Dialect/<backend>/`
  generates IR that the matching `runtime/src/iree/hal/drivers/<backend>/`
  executes.
- Docs: `docs/how_to/add_runtime_hal_driver.md` for the new-driver flow.

## Update triggers

Re-read this file and update it in the same turn if you:

- Add a new HAL driver under `runtime/src/iree/hal/drivers/<X>/` —
  refresh layout table; touch `iree_runtime_plugin.cmake` registration.
- Modify ukernel surface under `runtime/src/iree/builtins/ukernel/` —
  check `compiler/src/merlin/Target/` for the matching compile-side hook.
- IREE API drift (function signature change in vendored `iree_bar`) —
  update the Pitfalls warning and the [[iree_api_runner_port]] memory.
