---
name: merlin-samples
description: Add, build, and locate Merlin sample applications (including async-style deployments) using repository CMake structure and tools/build.py. Always run with `merlin-dev` and `uv run`.
---

# Merlin Samples

## Overview

Use this skill for:

- adding a new sample app under `samples/`
- wiring sample CMake targets correctly
- building specific sample executables via `tools/build.py --cmake-target`
- locating built binaries in `build/<target>-<variant>-<config>/...`

Primary references:

- `docs/how_to/add_sample_application.md`
- `samples/CMakeLists.txt`
- `samples/SpacemiTX60/baseline_dual_model_async`
- `iree_runtime_plugin.cmake`
- `tools/build.py`

## Workflow

1. Place sample in the correct subtree:
   - `samples/common/<app>`
   - `samples/<Platform>/<app>`
2. Add/adjust `CMakeLists.txt` with stable target/output names.
3. Ensure parent sample CMake files include the new subdir.
4. Build with `tools/build.py` and explicit `--cmake-target`.
5. Report output binary path(s) found in build tree.

## Standard Commands

- Build SpacemiTX60 async sample:
```bash
conda run -n merlin-dev uv run tools/merlin.py build --profile spacemit --config perf --cmake-target merlin_baseline_dual_model_async_run
```

- Locate produced binary:
```bash
find build -type f -name '*baseline-dual-model-async-run*'
```

## Guardrails

- Use repository build scripts; avoid ad hoc CMake calls unless requested.
- Preserve existing sample target names/API behavior when possible.
- Prefer small, scalable CMake/source changes.
