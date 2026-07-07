# AGENT.md — (root)

## Purpose

Repository root for **merlin**, a compiler-centered framework for exploring which hardware/software abstractions are worth exposing to the compiler. The root follows an XLA-style layout: it stays small, and almost all project code lives under the internal `merlin/` directory.

## What belongs here

- Top-level config: `README.md`, `pyproject.toml`, `CMakeLists.txt`, `.gitignore`.
- `build_tools/`, `docs/`, `third_party/`, and the internal `merlin/` tree.
- Generated `runs/`, `artifacts/`, `build/` (gitignored; see CLAUDE.md "Generated-output convention") and local `tmp/` scratch.

## What does not belong here

- New top-level directories without strong justification — prefer adding under `merlin/`.
- Source code, schemas, or experiments at the root.
- Vendored external repositories.

## Interfaces

The CLI surface is the console-scripts declared in `pyproject.toml [project.scripts]` (each a thin entrypoint into `merlin/python/merlin/`), documented in the generated `docs/reference/cli.md`. `merlin/schemas/` is the cross-workstream contract. Start at the docs hub `docs/README.md`; see `docs/reference/repo_structure.md` and `docs/design/parallel_workstreams.md`.

## Invariants

- Keep the root clean. New top-level directories require justification in the PR.
- Default to placing new work under the internal `merlin/` tree.
- `build/`, `output/`, and `tmp/` contents stay gitignored.

## Testing expectations

Run `python build_tools/scripts/check_structure.py` after structural changes.

## Notes for future agents

This is an active codebase with working end-to-end pipelines (kernel-mining → compiler, DSE,
targetgen, capsule/perf benchmarks, whole-model board bring-up, baselines). Land real changes, keep
the root clean, and default new work under the internal `merlin/` tree. Read `CLAUDE.md` (shared-tree
+ generated-output conventions), the directory's own `AGENT.md`, and `docs/` before working. See the
three workstreams in `docs/parallel_workstreams.md`.
