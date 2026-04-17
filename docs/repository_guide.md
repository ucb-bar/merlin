# Repository Guide

Merlin is organized to separate model frontends, compiler internals, and hardware-targeted runtimes.

This page is contributor-facing. If you are seeing the repository for the first
time and just want to use Merlin, read [user_paths.md](user_paths.md) first.

## First-User View

Most new users only need to care about four places:

- `docs/`: how to use Merlin and where the workflows are documented
- `tools/`: the `tools/merlin.py` entrypoint and helper CLIs (use `./merlin`)
- `models/`: model inputs and current compile-target views
- `build/`: generated outputs and compiled artifacts

The rest of the repo matters when you start bringing up new hardware or editing
compiler/runtime internals.

## Contributor Layers

- User layer: `docs/`, `tools/`, `models/`, `build/`
- Target bring-up layer: `target_specs/`, `build_tools/hardware/`, `models/*.yaml`
- Implementation layer: `compiler/`, `runtime/`, `third_party/iree_bar`
- Research and sidecar tooling: `benchmarks/`, `samples/research/`, `projects/`

## Core Directories

- `compiler/`: C++ and MLIR compiler code (dialects, passes, plugins).
- `tools/`: Python developer entrypoints (`build.py`, `compile.py`, `setup.py`, `ci.py`, etc.).
- `models/`: Model definitions, exports, and quantization helpers.
- `target_specs/`: Canonical hardware capability specs and deployment overlays for TargetGen.
- `samples/`: C/C++ runtime examples and hardware-facing sample flows.
- `benchmarks/`: Benchmark scripts and board-specific profiling helpers.
- `docs/`: Documentation source consumed by MkDocs.

## What New Users Commonly Mistake

- `third_party/` is not the first place to start. It is only relevant once a
  change needs to reach the IREE or LLVM forks.
- `benchmarks/` and `samples/` are useful, but they are not the primary entry
  point for model compilation.
- `build_tools/` is not one concept. It contains packaging, recipes,
  toolchains, and patch helpers.
- `models/*.yaml` are compile-target views, while `target_specs/` is the newer
  canonical capability-spec surface used by TargetGen.

## Placement Conventions (Where New Code Should Go)

- New compiler dialects/passes/transforms: `compiler/src/merlin/`.
- New plugin/target registration glue: `compiler/plugins/`.
- New model exports or conversion flows: `models/<model_name>/`.
- New target flag bundles for `tools/compile.py`: `models/<target>.yaml`.
- New board/runtime sample executables: `samples/<platform>/`.
- New benchmark flows and parsers: `benchmarks/<target>/`.
- New end-user docs and guides: `docs/`.

## Browse the Tracked Tree

For a current snapshot — more reliable than any committed copy that can rot —
generate one on demand:

```bash
git ls-files | tree --fromfile -L 3
```

If you do not have `tree` installed, `git ls-files` alone gives a flat list,
and `git ls-tree -d --name-only HEAD` lists tracked top-level directories.
