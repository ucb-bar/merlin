---
name: merlin-plugin-dev
description: Add or modify compiler dialect plugins and runtime HAL integrations using Merlin’s plugin format. Use script-first workflows, preserve APIs when possible, and follow Gemmini/NPU/Radiance patterns.
---

# Merlin Plugin Dev

## Overview

Use this skill for:

- adding a new compiler target plugin + dialect stack
- wiring pass pipelines into IREE plugin hooks
- adding or modifying runtime HAL drivers in `runtime/src/iree/hal/drivers`
- connecting compiler/runtime plugin toggles in CMake + `tools/build.py`

Primary references:

- `docs/how_to/add_compiler_dialect_plugin.md`
- `docs/how_to/add_runtime_hal_driver.md`
- `compiler/plugins/target/Gemmini`
- `compiler/plugins/target/NPU`
- `runtime/src/iree/hal/drivers/radiance`
- `iree_compiler_plugin.cmake`
- `iree_runtime_plugin.cmake`
- `tools/build.py`

## Workflow

1. Start from an existing in-tree pattern (Gemmini/NPU for compiler, Radiance for runtime).
2. Keep changes localized and staged:
   - dialect IR and passes in `compiler/src/merlin/Dialect/<Target>/`
   - plugin glue in `compiler/plugins/target/<Target>/`
   - build wiring in `iree_compiler_plugin.cmake` / `iree_runtime_plugin.cmake`
3. Use `tools/build.py` profiles/scopes for validation; avoid direct CMake unless necessary.
4. Add pass/driver smoke tests before broad feature expansion.
5. Preserve CLI/API compatibility whenever possible.

## Validation Commands

- Compiler plugin visibility:
```bash
build/host-merlin-release/install/bin/iree-compile --iree-list-plugins
```

- NPU-like compiler build:
```bash
conda run -n merlin-dev uv run tools/merlin.py build --profile npu --config release
```

- Radiance runtime smoke build:
```bash
conda run -n merlin-dev uv run tools/merlin.py build --profile radiance --cmake-target iree_hal_drivers_radiance_testing_transport_smoke_test
```

## Guardrails

- Always use `merlin-dev`.
- Run Python scripts via `uv run`.
- No shortcut workflows that bypass repository scripts for normal operations.
- If modifying scripts, make minimal scalable changes and avoid breaking user-facing APIs.
