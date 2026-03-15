---
name: merlin-compile-target
description: Use `tools/compile.py` and `models/*.yaml` to compile models, add/modify compile targets, select `--hw` profiles, and locate generated artifacts under `build/compiled_models`. Always run in `merlin-dev` with `uv run`.
---

# Merlin Compile Target

## Overview

Use this skill when the task involves:

- running model compilation with `tools/compile.py`
- adding or updating target YAML files in `models/`
- debugging compile flag selection (`generic`, `targets`, quantized, model overrides)
- locating generated `.mlir`/`.vmfb` outputs

Primary references:

- `docs/how_to/add_compile_target.md`
- `docs/reference/cli.md`
- `tools/compile.py`
- `models/`

## Workflow

1. Ensure execution in `merlin-dev` (`conda run -n merlin-dev ...`).
2. Run Python scripts with `uv run`.
3. Prefer changing YAML configuration over hardcoding flags in ad hoc commands.
4. Keep `tools/compile.py` API stable when editing script behavior.
5. Report final artifact path(s) in `build/compiled_models/...`.

## Standard Commands

- Compile one model:
```bash
conda run -n merlin-dev uv run tools/compile.py <model_or_file> --target <target_name>
```

- Compile with hardware profile:
```bash
conda run -n merlin-dev uv run tools/compile.py <model_or_file> --target <target_name> --hw <hw_profile>
```

- Use specific build tools dir:
```bash
conda run -n merlin-dev uv run tools/compile.py <model_or_file> --target <target_name> --build-dir <build_dir_name>
```

## Guardrails

- No shortcuts: use `tools/compile.py`, not direct `iree-compile` for user-facing flows unless explicitly requested.
- Minimize script changes and preserve existing flags/CLI semantics.
- Prefer scalable YAML-driven configuration in `models/`.
