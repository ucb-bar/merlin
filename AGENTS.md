# Merlin — Agent Instructions

Read at session start. Companion files: `tools/<subcmd>/AGENTS.md` for
per-package depth, `.claude/skills/*/SKILL.md` for multi-step workflows
(auto-matched against user intent), `.mcp.json` for structured tool servers.

## Golden Rules

1. **Always use `./merlin`** — never invoke `cmake` / `ninja` / `make` / raw
   build commands. All build, compile, benchmark, CI, setup operations go
   through `./merlin`. If `build_check_freshness` (MCP) says a rebuild is
   needed, the ONLY acceptable response is `./merlin build --profile <X>`
   — never `cmake --build` or `ninja` in the build dir directly.
2. **Default to `release` config** unless the user explicitly asks for
   `debug` or `trace`. Never silently switch to debug to "see more output."
3. **Always consult `docs/`** before proposing changes — architecture notes
   in `docs/architecture/`, how-tos in `docs/how_to/`, dev-blogs in
   `docs/dev_blog/` (the "why" trail).
4. **Never skip the environment** — every command runs inside the
   `merlin-dev` conda env with `uv run`.
5. **Never commit or push** — no `git commit`, `git push`, no branch
   creation unless the user explicitly asks. `git add` only when told.
6. **Before invoking `./merlin compile` / `run` / `verify-output`**, call
   the `build_check_freshness` MCP tool for the relevant profile. If it
   says `needs_rebuild`, rebuild first — stale binaries lie about what
   the current source code does.

## Environment

```bash
./merlin <subcmd> [args...]                                # the wrapper
conda run -n merlin-dev uv run tools/merlin.py <subcmd>    # the wrapper's actual call
```

Never `python3` or `pip install` directly — the project uses `uv` with
`pyproject.toml`/`uv.lock`.

## Registered subcommands

`tools/merlin.py:COMMANDS` is the authoritative list. If a tool isn't there,
it isn't a subcommand. For flags run `./merlin <subcmd> --help`.

| Subcommand        | Entry point              | Package layout |
| ----------------- | ------------------------ | --- |
| `build`           | `tools/build/cli.py`     | `cli` + `presets` + `cmake` + `packaging` + `radiance` (5 modules) |
| `compile`         | `tools/compile/cli.py`   | `cli` + `iree_tools` + `postprocess` + `radiance` + `feedback_overlay` + 6 kernel helpers (12+ modules) |
| `quantize`        | `tools/quantize/cli.py`  | `cli` + `int8` + `analyze` |
| `verify-output`   | `tools/verify/cli.py`    | `cli` + `het_e2e` + `int8` |
| `perf-decompose`  | `tools/perf/cli.py`      | `cli` + `decompose` + 4 profilers + 2 plot + `trace_to_profile` + 2 `.sh` |
| `coverage-check`  | `tools/coverage/cli.py`  | `cli` + `check` + `audit` |
| `setup`           | `tools/setup.py`         | single file (env, submodules, toolchain, prebuilt all merged) |
| `ci`              | `tools/ci.py`            | single file |
| `patches`         | `tools/patches.py`       | single file |
| `benchmark`       | `tools/benchmark.py`     | single file |
| `chipyard`        | `tools/chipyard/cli.py`  | `cli` + `config` + `recipe` + `git_ops` + `bare_metal` + `firesim` + `zephyr` + `radiance` + `status` (9 modules) |
| `ray`             | `tools/ray/cli.py`       | `cli` + `model` + `service` |
| `targetgen`       | `tools/targetgen/cli.py` | `cli` + `spec` + `paths` + `io` + `render` + framework (intake/, prompt_library/, planner, executor, generator, ...) |
| `spike`           | `tools/spike.py`         | single file (companion: `build_tools/spike-hetero/` build artifact) |
| `sim`             | `tools/sim.py`           | single file |
| `run`             | `tools/run/cli.py`       | `cli` + 6 modes + 3 `sched_*` helpers |
| `mcp`             | `tools/mcp_servers/cli.py`       | `cli` + `scaffold` + 6 per-domain registries (build, compile, run, perf, verify, targetgen) |

The daily-driver subcommands (`build`, `compile`, `run`, `perf-decompose`,
`verify-output`, `targetgen`) are exposed as structured MCP tools via
`.mcp.json` — prefer MCP over Bash shelling when an MCP tool covers the
intent. In particular:

- `merlin-build` exposes `build_list_profiles`, `build_status`,
  `build_check_freshness` (the killer tool — deterministically decides if
  a rebuild is needed), and `build_profile`.
- `merlin-compile` exposes `compile_list_targets`, `compile_list_models`,
  `compile_model` (returns parsed `{passed, vmfb, phase_dumps,
  per_dispatch_benchmarks, target_devices}`).
- `merlin-run` exposes `run_list_modes`, `run_help`, `execute_run`
  (returns parsed `{makespan_ms, per_instance_ms, hashes}`).
- `merlin-perf` exposes `perf_decompose` (returns parsed
  `{total_cycles, top_k_hot, by_kind}`).
- `merlin-verify` exposes `verify_output` (returns parsed
  `{passed, hashes, hash_match, max_diff}`).

## Tool Extension Protocol

**The default is always "extend an existing tool."** The historical failure
mode was Claude facing a novel task, not realizing an existing subcommand
covered ~80% of it, and dropping a fresh one-shot script under `tools/`.
That caused the 70-file `tools/` explosion this repo is built to prevent.

Before writing new code under `tools/`:

```
Does an existing ./merlin subcommand already cover ~80% of what's needed?
   yes → extend it (add a flag, add a module under tools/<subcmd>/).
   no  → is this a one-time debug helper that runs once and never again?
          yes → put it in tmp/.
          no  → new reusable subcommand:
                  1. tools/<name>/cli.py shim (≤200 LOC, argparse + dispatch)
                  2. optionally tools/<name>/ for helpers (mirror existing packages)
                  3. register in tools/merlin.py:COMMANDS
                  4. add a row to this file's subcommand table
                  5. consider adding an MCP wrapper if it's daily-driver
```

### The no-overfit rule

> **When extending a `tools/<x>/` package to handle a new case, the
> extension must accept the varying dimension as a parameter, never
> hardcode it.** If you find yourself writing `if model == "dronet"`,
> `target == "qrb5165"`, or a literal absolute path inside `tools/<x>/`,
> stop: thread the value through as a CLI flag on `tools/<x>/cli.py` and
> a function argument in the package. Mechanical test: could a colleague
> reuse your code for a different model/target/board by passing different
> flags? If no, you have overfit. Per-case configuration belongs in
> `models/<target>.yaml`, `target_specs/<target>/`, or
> `build_tools/hardware/<board>.yaml`. Toolchain paths use
> `tools/utils.py:find_toolchain_binary(...)` with `$MERLIN_*` env vars,
> not absolute paths.

## Maintenance Protocol — Keep MCP / Skills / Docs / sub-AGENTS.md In Sync

**Load-bearing rule:** if your edit touches a folder that owns an
`AGENTS.md` (root or sub), re-read that `AGENTS.md` and update it in the
**same** turn if the change invalidates its contents (module map, pitfalls,
cross-references, file list, schema). Sub-AGENTS.md files declare an
`## Update triggers` section that names the specific file-edit patterns
which require an AGENTS.md review — consult it when you edit files in
that folder. A PostToolUse hook (`.claude/hooks/agents_md_reminder.sh`)
fires a reminder after each Edit/Write to catch missed updates, but the
discipline is yours; the hook is a safety net.

When you edit a tool, you almost always need to update its **dependents**:
the MCP wrapper that calls it, the skill that mentions it, the docs that
cite its CLI surface, and the sub-AGENTS.md that maps the package. If you
don't, the next session will pick up stale guidance and produce wrong work.

**The discipline**: every MCP `tools.py`, every `SKILL.md`, and every
sub-`AGENTS.md` starts with a `Depends on:` header line listing the files
it wraps or references. When you edit a source file, grep for it in these
dependent locations and update.

The grep recipe (run after editing any `tools/<X>/cli.py`, `tools/<X>/<y>.py`,
or registered subcommand's argparse):

```bash
# Find every dependent that needs review.
grep -rln "<X>" \
    .claude/skills/ \
    tools/mcp_servers/*.py \
    AGENTS.md \
    tools/*/AGENTS.md \
    docs/architecture/
```

Updates required:
- **MCP tool `input_schema`** — if you added/removed/renamed a CLI flag on
  the underlying subcommand, the MCP schema must match.
- **MCP tool description** — if the tool's purpose shifted, the
  description (which is auto-matched against user intent) must reflect it.
- **Skill procedure steps** — if you changed an invocation pattern
  (`./merlin compile X --target Y` → `./merlin compile X --target Y --new-flag Z`),
  every skill that runs that command must be updated.
- **`AGENTS.md` subcommand table** — if you added/removed a subcommand or
  changed its purpose, the root table is the source of truth.
- **Sub-`AGENTS.md`** — if a package's extension points changed, update
  the relevant `tools/<x>/AGENTS.md`.
- **`__init__.py` docstring** — same as sub-AGENTS.md but for the
  per-module map.

If you're unsure whether you've found every dependent, run the grep
above plus a `git grep <symbol>` to catch anything outside the standard
locations.

## Where Other Kinds of Content Go

| Content | Goes in |
|---|---|
| One-time debug helper / investigation note | `tmp/` (active) or `tmp/archive/investigations/<name>/` (frozen) |
| IREE compile-flag bundle for a target | `models/<target>.yaml` (`generic`/`targets`/`quantized`/`models` keys) |
| TargetGen hardware capability spec | `target_specs/<target>/capability.yaml` (different schema: `execution_model`/`isa`/`operations`) |
| Chipyard hardware recipe | `build_tools/hardware/<board>.yaml` |
| Per-target build artifacts (toolchain config, linker scripts) | `build_tools/<target>/` |
| Board/runtime sample executable | `samples/<Board>/` (PascalCase) |
| Benchmark flow / board profiling driver | `benchmarks/<target>/` |
| Model definition / quantization helper | `models/<model_name>/` |
| Captured result paired with a build | `build/artifacts/<name>/` |
| Multi-step shell driver composing `./merlin` calls | `scripts/` (see `scripts/README.md`) |

**Never delete results during cleanup.** Plots, images, CSV/JSON, logs,
investigation writeups always move (to `build/artifacts/` or `tmp/archive/`),
never to `/dev/null`. The reproducibility script that produced a result
lives next to the result.

**Never invent a new top-level directory.** Every kind of content has a
home above. If you genuinely think a new top-level is needed, ask first.

## Repository Layout

```
merlin/
├── compiler/        C++/MLIR compiler code
├── kernels/         Kernel-embedding framework (core/ + qnn/ + spike/)
├── tools/           Python developer entrypoints (one package per ./merlin subcommand)
├── models/          Model assets + per-target compile-flag yamls
├── samples/         C/C++ runtime examples
├── benchmarks/      Benchmark scripts and board profiling helpers
├── docs/            Documentation (MkDocs)
├── target_specs/    TargetGen hardware capability specs
├── build_tools/     Toolchains, FireSim, Chipyard hardware recipes
├── third_party/     Submodules (iree_bar, etc.)
├── scripts/         Multi-step shell drivers (see scripts/README.md)
├── tmp/             Active scratchpad + tmp/archive/ frozen investigations
└── build/           Build outputs + build/artifacts/ + build/compiled_models/
```

## Code Style — Pre-commit Must Pass

All generated code must pass `.pre-commit-config.yaml` hooks:

- **Python** — `ruff-format` + `ruff --fix` (double quotes, trailing
  commas, sorted imports, no unused).
- **C/C++/CUDA** — `clang-format v17` per repo `.clang-format`.
- **CMake** — `cmake-format`.
- **Shell** — `shellcheck` (excluding `SC2054`, `SC2029`).
- **General** — file ends with newline; no trailing whitespace; valid YAML.

Run `pre-commit run --files <changed-files>` before declaring done.

## Key Conventions

- Build outputs in `build/<target>-<variant>-<config>/`.
- Compiled VMFBs in `build/compiled_models/<model>/<target>/`.
- Never commit `build/`, `.venv/`, toolchain binaries.
- Submodules managed via `./merlin patches`. Don't `git submodule update`
  manually without checking patch state first.

## Where to Go Next

- Per-package guidance: `tools/<subcmd>/AGENTS.md`.
- Architecture pattern: `docs/architecture/tools_architecture.md`.
- CLI reference: `docs/reference/cli.md`.
- Active engineering context: `docs/dev_blog/`.
