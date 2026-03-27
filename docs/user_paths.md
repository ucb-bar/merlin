# User Paths

Merlin has several layers in one repository. That is useful for target bring-up,
but it can be confusing if you approach the tree as a first-time user.

This page is the user-first map: what to look at first, what you can ignore, and
which folders matter for which kind of work.

## Start Here

If you are new to Merlin, begin with only these paths:

- `docs/`
- `tools/`
- `models/`
- `build/` once you start producing outputs

For many users, that is enough.

## Three Practical Paths

### 1. I want to compile or run models

Read or use:

- `README.md`
- `docs/getting_started.md`
- `docs/reference/cli.md`
- `tools/merlin.py`
- `models/`
- `build/compiled_models/`

Ignore for now:

- `compiler/`
- `runtime/`
- `third_party/`
- `projects/`
- most of `benchmarks/`

### 2. I want to bring up a new hardware target

Read or use:

- `docs/architecture/target_generator.md`
- `docs/architecture/ray_control_plane.md`
- `docs/how_to/add_compile_target.md`
- `docs/hardware_backends/`
- `target_specs/`
- `models/*.yaml`
- `build_tools/hardware/`
- `tools/merlin.py targetgen ...`

You will likely also touch:

- `compiler/plugins/`
- `compiler/src/merlin/`
- `runtime/src/iree/hal/drivers/`
- `third_party/iree_bar` and possibly LLVM surfaces

Ignore at first:

- large research samples under `samples/research/`
- unrelated benchmarks
- unrelated target overlays and historical dev logs

### 3. I want to change compiler or runtime internals

Read or use:

- `docs/repository_guide.md`
- `docs/how_to/add_compiler_dialect_plugin.md`
- `docs/how_to/add_runtime_hal_driver.md`
- `docs/architecture/plugin_and_patch_model.md`
- `compiler/`
- `runtime/`
- `third_party/iree_bar`

You will likely care about:

- `build_tools/patches/`
- `docs/dev_blog/`
- `tests/`

## What Each Top-Level Folder Means

- `tools/`: the main user and developer CLI surface. If you are unsure what to run, start here.
- `docs/`: the human-facing source of truth for setup, architecture, and workflows.
- `models/`: example or real model inputs plus current compile-target views.
- `build/`: generated outputs. This is where compiled artifacts, local build trees, and generated TargetGen bundles land.
- `target_specs/`: the newer canonical hardware capability specs and deployment overlays used by TargetGen.
- `compiler/`: MLIR and compiler-side implementation work.
- `runtime/`: runtime, HAL, and device/backend integration work.
- `build_tools/`: packaging, toolchains, docker helpers, hardware recipes, and patch-management helpers.
- `samples/`: runtime applications and target-facing example flows.
- `benchmarks/`: benchmark and profiling workflows. Useful later, not usually the first stop.
- `third_party/`: submodules and forks. Important for advanced work, noisy for first-time users.
- `projects/`: sidecar tools and research integrations, not the default entrypoint.

## What Is Currently Confusing

From a first-user perspective, the main sources of confusion are:

- the repo mixes product use, compiler development, hardware bring-up, and research work in one tree
- `models/` currently contains both user-facing model assets and target YAML views
- `build_tools/` mixes several concerns: packaging, toolchains, recipes, and patch helpers
- `third_party/` is necessary for advanced work but visually dominates the repository
- historical workstreams in `docs/dev_blog/` are valuable for contributors but too deep for first contact

The current documentation cleanup tries to compensate by routing users by task
instead of by folder name.

## A Simpler Mental Model

If you want a compact way to think about the repo, use this:

- `tools/ + docs/ + models/` is the user layer
- `target_specs/ + build_tools/hardware/` is the target bring-up layer
- `compiler/ + runtime/ + third_party/iree_bar` is the implementation layer
- `build/` is where everything generated should end up

That is the structure TargetGen is moving toward as well: canonical specs and
generated staging artifacts on top, with compiler/runtime/fork changes promoted
only after review.
