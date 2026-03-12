---
name: merlin-build
description: Build and set up Merlin using repository scripts only (`tools/merlin.py`, `tools/setup.py`, `tools/build.py`) with `merlin-dev` and `uv run`. Use for host/spacemit/firesim builds, profile selection, plugin toggles, clean rebuilds, and output-path discovery.
---

# Merlin Build

## Overview

Use this skill for reproducible builds using Merlin-owned scripts. Do not use ad
hoc CMake commands unless explicitly requested.

Primary references (always prefer these over duplicated logic):

- `docs/how_to/use_build_py.md`
- `docs/reference/cli.md`
- `tools/merlin.py`
- `tools/build.py`
- `tools/setup.py`

## Build Workflow

1. Confirm the repository root is active.
2. Ensure commands execute in `merlin-dev`:
   - use `conda run -n merlin-dev ...` when shell activation is uncertain.
3. Run Python tooling with `uv run`.
4. Prefer `tools/merlin.py build` (wrapper) for normal usage.
5. Use `tools/build.py` directly only when exact low-level flagging is needed.
6. Report build and install directories from command output.

## Standard Commands

- Sync env:
```bash
conda run -n merlin-dev uv run tools/merlin.py setup env
```

- Host baseline:
```bash
conda run -n merlin-dev uv run tools/merlin.py build --profile vanilla
```

- Host plugin build:
```bash
conda run -n merlin-dev uv run tools/merlin.py build --profile full-plugin
```

- Targeted build target:
```bash
conda run -n merlin-dev uv run tools/merlin.py build --profile spacemit --cmake-target <target_name>
```

## Guardrails

- Do not bypass `tools/build.py` with direct CMake unless requested.
- Prefer profile-driven workflows over manual flag bundles.
- Keep API compatibility when touching scripts.
- If script changes are needed:
  - make minimal, scalable changes,
  - preserve existing flags and behavior unless explicitly requested.
