---
name: merlin-build
description: Build and set up the Merlin repository using the project-standard workflow (`conda`, `tools/setup.py`, and `tools/build.py`). Use when asked to configure environment dependencies, compile for `host`, `spacemit`, or `firesim`, switch build configs (`debug`, `release`, `asan`, `trace`, `perf`), enable the Merlin plugin, clean/rebuild, or locate build/install outputs.
---

# Merlin Build

## Overview

Use this skill to run reproducible Merlin setup/build commands across targets and configs. Prefer `tools/setup.py` and `tools/build.py` over ad hoc CMake unless explicitly requested.

## Build Workflow

1. Confirm the repository root is active.
2. Prepare the environment:
- First-time machine/bootstrap: `conda env create -f env_linux.yml` then `conda activate merlin-dev`.
- Ongoing sync in the environment: `python tools/setup.py env`.
3. Select target/config/plugin flags and run `python tools/build.py ...`.
4. If required for cross targets, ensure toolchain prerequisites are met before building.
5. Report build and install directories from command output.

## Standard Commands

- Host debug baseline:
```bash
python tools/build.py --target host --config debug
```

- Host debug with plugin:
```bash
python tools/build.py --target host --config debug --with-plugin
```

- Clean rebuild:
```bash
python tools/build.py --target host --config debug --clean
```

- Build a specific CMake target:
```bash
python tools/build.py --target host --config debug --cmake-target <target_name>
```

## Target Matrix

Load [references/targets.md](references/targets.md) for:
- target-specific requirements (`host`, `spacemit`, `firesim`)
- config behavior (`debug`, `release`, `asan`, `trace`, `perf`)
- plugin flags and output path conventions
- troubleshooting and common fixes
