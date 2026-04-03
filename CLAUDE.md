# Merlin — Claude Code Instructions

## Golden Rules

1. **Always use `tools/`** — Never invoke `cmake`, `ninja`, `make`, or raw build commands directly. All build, compile, benchmark, CI, and setup operations go through `tools/merlin.py` subcommands. See `docs/reference/cli.md` for the full CLI reference.
2. **Always consult `docs/`** — Before proposing changes or answering questions about architecture, workflows, hardware targets, or conventions, read the relevant documentation under `docs/`. This includes dev-blogs (`docs/dev_blog/`), architecture notes (`docs/architecture/`), how-to guides (`docs/how_to/`), and reference pages (`docs/reference/`).
3. **Never skip the environment** — All commands must run inside the correct environment (see below).
4. **Never commit or push** — Do not run `git commit`, `git push`, or create branches unless the user explicitly asks you to. Stage files (`git add`) only when instructed. The user manages all version control operations.
5. **Out-of-tree first** — Always prefer out-of-tree Merlin plugin/core changes. Never patch IREE core (`third_party/iree_bar`) unless absolutely unavoidable. See "Plugin and Patch Model" below.

## Environment

Merlin uses a two-layer environment: **conda** for system/native toolchain deps and **uv** for Python packages.

### Running tools (build, compile, benchmark, etc.)

```bash
conda run -n merlin-dev uv run tools/merlin.py <subcommand> [args...]
```

Or, if the conda environment is already activated in the shell:

```bash
uv run tools/merlin.py <subcommand> [args...]
```

### Running any Python script

Always use `uv run` so the correct virtualenv and dependencies are resolved:

```bash
conda run -n merlin-dev uv run python <script.py> [args...]
# or if conda is already active:
uv run python <script.py> [args...]
```

Never use bare `python3` or `pip install` — the project uses `uv` (managed via conda) with `pyproject.toml` and `uv.lock`.

## tools/ — The Developer CLI

`tools/merlin.py` is the single unified entrypoint. Available subcommands:

| Subcommand   | Module             | Purpose                                          |
| ------------ | ------------------ | ------------------------------------------------ |
| `build`      | `tools/build.py`   | Configure and build Merlin and target runtimes   |
| `compile`    | `tools/compile.py`  | Compile MLIR/ONNX models to target artifacts     |
| `setup`      | `tools/setup.py`    | Bootstrap developer environment and toolchains   |
| `ci`         | `tools/ci.py`       | Run repository CI/lint/patch workflows           |
| `patches`    | `tools/patches.py`  | Verify submodule state and manage upstream patches|
| `benchmark`  | `tools/benchmark.py`| Run benchmark helper scripts                     |
| `chipyard`   | `tools/chipyard.py` | Manage Chipyard hardware backend interactions    |

When you need to understand what flags or options a subcommand accepts, read the corresponding `tools/<module>.py` file or run `uv run tools/merlin.py <subcommand> --help`.

### Common build examples

```bash
# Host-only vanilla build (compiler tools)
uv run tools/merlin.py build --profile vanilla

# SpacemiT cross-compile with plugin
uv run tools/merlin.py build --profile spacemit

# Build a specific cmake target
uv run tools/merlin.py build --profile spacemit --cmake-target <target_name>

# Compile a model
uv run tools/merlin.py compile models/dronet/dronet.mlir --target spacemit_x60
```

## docs/ — Always Read Before Acting

The `docs/` directory is the authoritative source for how this project works. Key locations:

- **`docs/getting_started.md`** — Quickstart for first-time usage.
- **`docs/repository_guide.md`** — Repo layout and placement conventions for new code.
- **`docs/architecture/`** — Design decisions: plugin/patch model, cmake presets, etc.
- **`docs/how_to/`** — Step-by-step guides for adding dialects, HAL drivers, samples, compile targets.
- **`docs/reference/`** — CLI reference, cmake targets, C++ API, MLIR ops, Python API.
- **`docs/dev_blog/`** — Active engineering logs with debugging context, decisions, and progress. These are invaluable for understanding *why* things are the way they are and what is currently being worked on.

When working on a task, check whether there is a relevant dev-blog entry or how-to guide before starting.

## Repository Layout

```
merlin/
├── compiler/        — C++/MLIR compiler code (dialects, passes, plugins)
│   ├── plugins/target/<Target>/  — IREE plugin registration glue per target
│   └── src/merlin/               — Merlin-owned dialects, transforms, heuristics
├── runtime/         — HAL drivers and runtime plugin code
│   └── src/iree/hal/drivers/<driver>/  — HAL driver implementations
├── tools/           — Python developer entrypoints (build, compile, setup, ci, etc.)
├── models/          — Model definitions, exports, quantization helpers, target YAMLs
├── samples/         — C/C++ runtime examples and hardware sample flows
├── benchmarks/      — Benchmark scripts and board-specific profiling
├── docs/            — Documentation source (MkDocs)
├── config/          — Target configuration files (targets.json, etc.)
├── build_tools/     — Toolchains, Docker, FireSim, hardware manifests, patches
├── third_party/     — Submodules (iree_bar, cuda-tile, torch-mlir, etc.)
├── iree_compiler_plugin.cmake  — Top-level compiler plugin entry (IREE discovers this)
├── iree_runtime_plugin.cmake   — Top-level runtime plugin entry (IREE discovers this)
├── env_linux.yml    — Conda environment definition
├── pyproject.toml   — Python/uv project definition
└── uv.lock          — Locked Python dependencies
```

---

## Plugin and Patch Model

This is the most important architectural concept in Merlin. Read `docs/architecture/plugin_and_patch_model.md` for the full specification.

### Core Principle: Out-of-Tree First

Merlin keeps its own code **out-of-tree** relative to IREE. IREE is a git submodule at `third_party/iree_bar`, and Merlin hooks into it via `-DIREE_CMAKE_PLUGIN_PATHS`. This means:

- **Compiler plugins** live in `compiler/plugins/target/<Target>/` and register via `iree_compiler_register_plugin()`.
- **HAL drivers** live in `runtime/src/iree/hal/drivers/<driver>/` and register via `iree_register_external_hal_driver()`.
- **Dialects and passes** live in `compiler/src/merlin/Dialect/<Target>/`.
- **None of this touches IREE source code.**

### When In-Tree IREE Changes Are Unavoidable

If you must modify IREE itself:

1. Make the edit in `third_party/iree_bar` on the `ucb-bar/main` branch.
2. Commit with a `[Merlin]` prefix: `[Merlin] Brief description of change`.
3. Keep commits atomic — one concern per commit.
4. Push to the fork, then update the submodule pointer in Merlin.
5. Update `IREE_UPSTREAM_BASE` in `build_tools/patches/manifest.env` after rebasing.

### Adding a New Compiler Plugin (Dialect + Passes)

Follow `docs/how_to/add_compiler_dialect_plugin.md`. The pattern:

1. **Dialect + passes** in `compiler/src/merlin/Dialect/<Target>/`:
   - `IR/` — dialect, ops, attrs (TableGen + C++)
   - `Transforms/` — lowering/rewrite passes
   - `Translation/` — target-specific translation (optional)
   - `Register<Target>.*` — dialect/pass registration entry points

2. **Plugin glue** in `compiler/plugins/target/<Target>/`:
   - `CMakeLists.txt` — uses `iree_cc_library` + `iree_compiler_register_plugin`
   - `<Target>Options.h/.cpp` — plugin options struct with `bindOptions(OptionsBinder&)`
   - `PluginRegistration.cpp` — `PluginSession` subclass implementing:
     - `registerPasses()` — static, registers all target passes
     - `onRegisterDialects()` — inserts target dialects into registry
     - `extendPostGlobalOptimizationPassPipeline()` — hooks IREE pipeline
   - Extern C registration: `iree_register_compiler_plugin_<name>`

3. **CMake wiring** in `iree_compiler_plugin.cmake`:
   - Add `MERLIN_ENABLE_TARGET_<TARGET>` / `MERLIN_BUILD_<TARGET>` toggle
   - Guard with `MERLIN_ENABLE_CORE` dependency
   - `add_subdirectory(compiler/plugins/target/<Target>)`

4. **Build profile** in `tools/build.py`:
   - Add to `--compiler-scope` choices
   - Map to cmake flag `-DMERLIN_ENABLE_TARGET_<TARGET>=ON`

Reference implementations: **Gemmini** and **NPU** plugins.

### Adding a New HAL Driver

Follow `docs/how_to/add_runtime_hal_driver.md`. The pattern:

1. **Driver sources** in `runtime/src/iree/hal/drivers/<driver>/`:
   - `api.h` — public options/types
   - `driver.c` — HAL driver
   - `device.c` — HAL device + queue behavior
   - `registration/driver_module.c` — factory registration
   - `testing/` — smoke tests

2. **CMake wiring** in `iree_runtime_plugin.cmake`:
   - `iree_register_external_hal_driver(NAME <driver> ...)`
   - Add toggle: `MERLIN_RUNTIME_ENABLE_HAL_<DRIVER>`

3. **Compiler/HAL separation**: compiler decides executable format + target flags; HAL runtime decides device/queue/transport at execution time.

Reference implementation: **Radiance** HAL driver.

### Adding a Compile Target

Follow `docs/how_to/add_compile_target.md`:

1. Create `models/<target>.yaml` with flag bundles.
2. Add profile in `tools/build.py` if special compiler/runtime config needed.

### Plugin Naming Conventions

| Entity | Pattern | Example |
|--------|---------|---------|
| Plugin ID | lowercase, no prefix | `gemmini`, `npu` |
| CMake toggle | `MERLIN_ENABLE_TARGET_<UPPER>` | `MERLIN_ENABLE_TARGET_GEMMINI` |
| CMake package prefix | `iree::<lower>::compiler::plugin` | `iree::gemmini::compiler::plugin` |
| C registration fn | `iree_register_compiler_plugin_<name>` | `iree_register_compiler_plugin_gemmini` |
| PluginSession class | `<Target>Session` | `GemminiSession` |
| Options struct | `<Target>Options` | `GemminiOptions` |
| Dialect namespace | `<Target>::` | `Gemmini::`, `NPU::` |
| Pass prefix | `create<PassName>Pass` | `Gemmini::createConvertToGemminiPass` |

### Pipeline Integration Pattern

Recommended (used by all existing plugins):

1. Keep generic MLIR/IREE semantics as long as possible.
2. Hook at `extendPostGlobalOptimizationPassPipeline` for target recovery/lowering.
3. Keep target semantics explicit in target dialect ops.
4. Only lower to backend-specific forms when needed.
5. Add `canonicalize` + `CSE` between major lowering steps.
6. Keep plugin options for strict/fallback behavior.

### Submodule and Patch Verification

```bash
# Verify submodule is a clean rebase of pinned upstream
uv run tools/merlin.py patches verify

# Show Merlin-specific commits on ucb-bar/main
uv run tools/merlin.py patches log

# Check drift from upstream
uv run tools/merlin.py patches drift
```

### Upstream Bump Workflow

```bash
cd third_party/iree_bar
git fetch https://github.com/iree-org/iree main
git rebase FETCH_HEAD
# Resolve per-commit conflicts, then: git rebase --continue
git push origin ucb-bar/main --force-with-lease
cd ../..
git add third_party/iree_bar
# Update IREE_UPSTREAM_BASE in build_tools/patches/manifest.env
```

---

## Code Style — Pre-commit Must Pass

All generated or modified code **must** pass the pre-commit hooks defined in `.pre-commit-config.yaml`. Treat these as hard requirements, not suggestions.

### Python (ruff)

- Format with **ruff-format** and lint with **ruff** (`--fix`). This means:
  - Use double quotes for strings (ruff default).
  - No unused imports, no unused variables.
  - Trailing commas on multi-line collections.
  - Sort imports (isort-compatible via ruff).
  - Keep lines within the configured length limit.

### C / C++ / CUDA (clang-format v17)

- All C, C++, and CUDA files are formatted with **clang-format v17**. The repo's `.clang-format` file defines the style.
- Before writing C/C++ code, read `.clang-format` if you are unsure of the style (brace placement, indent width, etc.).

### CMake (cmake-format)

- All `CMakeLists.txt` and `.cmake` files are formatted with **cmake-format**. Follow the existing style in nearby CMake files.

### Shell (shellcheck)

- All shell scripts must pass **shellcheck** (with `SC2054` and `SC2029` excluded). Use proper quoting, avoid common bash pitfalls, and prefer `"$var"` over `$var`.

### General (pre-commit-hooks)

- Files must end with a newline.
- No trailing whitespace on any line.
- YAML must be valid (checked with `check-yaml --unsafe`).
- Do not add large files to the repo (`check-added-large-files`).

### Workflow

If you are unsure whether your code passes, suggest running `pre-commit run --files <changed-files>` to verify before committing.

## Key Conventions

- **Build outputs** land in `build/<target>-<variant>-<config>/` (e.g., `build/host-vanilla-release/`, `build/host-merlin-release/`).
- **Compiled model artifacts** land in `build/compiled_models/<model>/<target>/`.
- **Never commit** `build/`, `.venv/`, or toolchain binaries.
- **Submodules** (e.g., `third_party/iree_bar`) are managed via `tools/patches.py`. Do not manually `git submodule update` without checking patch state first.
- **Pre-commit hooks** are configured — run `pre-commit install` after setup.
- **Tests early** — At minimum: unit lit tests for each pass, one post-global-opt hook test proving plugin integration.
- **Dev blog entries** — For non-trivial work (new backends, HAL drivers, workstream investigations), add a dated entry under `docs/dev_blog/` using `docs/dev_blog/TEMPLATE.md`.

## IREE Plugin Discovery

IREE discovers Merlin's plugins via the cmake flag:

```
-DIREE_CMAKE_PLUGIN_PATHS=<path-to-merlin-root>
```

This causes IREE to include:
- `iree_compiler_plugin.cmake` — compiler plugins (dialects, passes, target backends)
- `iree_runtime_plugin.cmake` — runtime plugins (HAL drivers, samples, benchmarks)

Both files live at the Merlin repo root. They are the entry points for all out-of-tree integration.

## Third-Party Dependencies as Submodules

External dependencies go in `third_party/`. Current submodules include `iree_bar`, `cuda-tile`, `torch-mlir`, `gemmini-mx`, `saturn-vectors`, and others. When adding a new submodule:

1. Use `git submodule add <url> third_party/<name>`.
2. Pin to a specific commit (not a branch tip).
3. Document the LLVM commit alignment if the submodule contains LLVM/MLIR code.
4. Add any build integration to `iree_compiler_plugin.cmake` or `iree_runtime_plugin.cmake`.
