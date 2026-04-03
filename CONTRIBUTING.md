# Contributing

This project is maintained by a small team. Keep changes small, automated, and
repeatable.

## Development Principles

1. **Out-of-tree first.** All Merlin logic goes in `compiler/`, `runtime/`,
   `samples/`, `benchmarks/`, and scripts in this repository.
2. **Treat submodules as pinned dependencies.** `third_party/iree_bar` and its
   nested `third_party/llvm-project` are not yours to edit casually.
3. **If in-tree IREE changes are unavoidable**, commit on `ucb-bar/main` in the
   fork with a `[Merlin]` prefix. Keep commits atomic (one concern each).
   Update `build_tools/patches/manifest.env` after rebasing.
4. **Keep commands scriptable.** Avoid one-off manual steps without docs.
5. **Use the CLI.** All builds go through `tools/merlin.py build`, all compiles
   through `tools/merlin.py compile`. Never invoke cmake/ninja directly.

## Adding New Backends

Adding a new hardware backend touches three layers. Each has a how-to guide:

| Layer | Guide | What you create |
|-------|-------|-----------------|
| Compiler plugin | `docs/how_to/add_compiler_dialect_plugin.md` | Dialect, passes, `PluginRegistration.cpp` |
| HAL driver | `docs/how_to/add_runtime_hal_driver.md` | Driver, device, registration module |
| Compile target | `docs/how_to/add_compile_target.md` | Target YAML, build profile |

Follow the existing Gemmini/NPU (compiler) and Radiance (HAL) implementations as
reference. The key files for wiring are:

- `iree_compiler_plugin.cmake` — compiler-side plugin discovery
- `iree_runtime_plugin.cmake` — runtime-side driver discovery
- `tools/build.py` — build profile and `--compiler-scope` mapping

## Plugin Structure

Every compiler plugin follows this layout:

```
compiler/
├── src/merlin/Dialect/<Target>/
│   ├── IR/          — Dialect, ops, attrs (TableGen + C++)
│   ├── Transforms/  — Lowering and rewrite passes
│   ├── Translation/ — Target-specific translation (optional)
│   └── Register<Target>.* — Registration entry points
└── plugins/target/<Target>/
    ├── CMakeLists.txt         — iree_cc_library + iree_compiler_register_plugin
    ├── <Target>Options.h/cpp  — Plugin options (OptionsBinder pattern)
    └── PluginRegistration.cpp — PluginSession subclass
```

## Commit Scope

1. Keep PRs focused (one subsystem per PR when possible).
2. Include docs updates for workflow/process changes.
3. If behavior changes, include at least one runnable command in PR description.
4. For non-trivial work, add a dated dev-blog entry under `docs/dev_blog/`.

## Commit Messages for Fork Edits

When committing to `third_party/iree_bar` on `ucb-bar/main`:

- Prefix with `[Merlin]`: `[Merlin] Add RISC-V mmt4d ukernel architecture`
- One concern per commit (bare-metal fixes separate from ukernel changes).
- Order commits to minimize context-line dependencies during rebase.

## CI Expectations

PRs should pass:

1. Script/python lint gates (`pre-commit run`).
2. Patch-stack verification and drift checks (`merlin ci patch-gate`).
3. CLI documentation drift check (`merlin ci cli-docs-drift`).
4. Any workflow-specific checks touched by your change.

## Code Style

- **Python**: ruff-format + ruff lint (double quotes, sorted imports, no unused vars)
- **C/C++/CUDA**: clang-format v17 (see `.clang-format`)
- **CMake**: cmake-format (follow existing style)
- **Shell**: shellcheck (SC2054/SC2029 excluded)
- **General**: files end with newline, no trailing whitespace, valid YAML

## Testing

At minimum for any new pass or dialect:

- Unit lit tests for each pass.
- One post-global-opt hook test proving plugin integration works end-to-end.
- See `compiler/src/merlin/Dialect/*/Transforms/tests/` for examples.

## Submodule Management

- Use `tools/merlin.py patches verify` before touching submodule state.
- Never `git submodule update` without checking patch state first.
- New third-party deps go in `third_party/` as submodules pinned to specific commits.
