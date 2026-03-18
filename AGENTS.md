# Merlin — Agent Instructions

## Golden Rules

1. **Always use `tools/`** — Never invoke `cmake`, `ninja`, `make`, or raw build commands directly. All build, compile, benchmark, CI, and setup operations go through `tools/merlin.py` subcommands. See `docs/reference/cli.md` for the full CLI reference.
2. **Always consult `docs/`** — Before proposing changes or answering questions about architecture, workflows, hardware targets, or conventions, read the relevant documentation under `docs/`. This includes dev-blogs (`docs/dev_blog/`), architecture notes (`docs/architecture/`), how-to guides (`docs/how_to/`), and reference pages (`docs/reference/`).
3. **Never skip the environment** — All commands must run inside the correct environment (see below).
4. **Never commit or push** — Do not run `git commit`, `git push`, or create branches unless the user explicitly asks you to. Stage files (`git add`) only when instructed. The user manages all version control operations.

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

| Subcommand   | Module               | Purpose                                           |
| ------------ | -------------------- | ------------------------------------------------- |
| `build`      | `tools/build.py`     | Configure and build Merlin and target runtimes    |
| `compile`    | `tools/compile.py`   | Compile MLIR/ONNX models to target artifacts      |
| `setup`      | `tools/setup.py`     | Bootstrap developer environment and toolchains    |
| `ci`         | `tools/ci.py`        | Run repository CI/lint/patch workflows            |
| `patches`    | `tools/patches.py`   | Verify submodule state and manage upstream patches |
| `benchmark`  | `tools/benchmark.py` | Run benchmark helper scripts                      |
| `chipyard`   | `tools/chipyard.py`  | Manage Chipyard hardware backend interactions     |

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

```text
merlin/
├── compiler/        — C++/MLIR compiler code (dialects, passes, plugins)
├── tools/           — Python developer entrypoints (build, compile, setup, ci, etc.)
├── models/          — Model definitions, exports, quantization helpers
├── samples/         — C/C++ runtime examples and hardware sample flows
├── benchmarks/      — Benchmark scripts and board-specific profiling
├── docs/            — Documentation source (MkDocs)
├── config/          — Target configuration files (targets.json, etc.)
├── build_tools/     — Toolchains, Docker, FireSim, hardware manifests
├── third_party/     — Submodules (iree_bar, etc.)
├── env_linux.yml    — Conda environment definition
├── pyproject.toml   — Python/uv project definition
└── uv.lock          — Locked Python dependencies
```

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

- **Build outputs** land in `build/<target>-<variant>-<config>/` (e.g., `build/host-vanilla-release/`, `build/spacemit-merlin-release/`).
- **Compiled model artifacts** land in `build/compiled_models/<model>/<target>/`.
- **Never commit** `build/`, `.venv/`, or toolchain binaries.
- **Submodules** (e.g., `third_party/iree_bar`) are managed via `tools/patches.py`. Do not manually `git submodule update` without checking patch state first.
- **Pre-commit hooks** are configured — run `pre-commit install` after setup.
