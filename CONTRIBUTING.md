# Contributing to merlin

Thanks for your interest in contributing! merlin is an active research codebase; this guide covers
the conventions the tooling enforces so your PRs land smoothly.

## Setup

merlin uses [uv](https://docs.astral.sh/uv/):

```bash
uv sync --all-extras                 # .venv with merlin (editable) + xDSL + dev deps
python build_tools/scripts/install_git_hooks.py   # enable the pre-commit gate (per clone)
```

External dependencies (chipyard, model2MLIR, boards, sibling repos) are **not vendored** — point at
them via environment variables (copy `.env.example` → `.env` and edit). The repo locates itself via
`merlin.common.paths.repo_root()`; never hard-code absolute paths.

## Where things go

- **Code**: under the internal `merlin/` tree (XLA-style). Keep the repo root clean — new top-level
  directories need justification.
- **Tests**: `merlin/tests/<bucket>/test_<area>.py`, one of the subsystem buckets
  (`kernels/ rvv/ dse/ gemmini/ targetgen/ ir/ runtime/ infra/`). Resolve paths via
  `merlin.common.paths`, never `Path(__file__).parents[N]`. Run: `.venv/bin/python -m pytest merlin/tests`.
- **Generated output**: only under `out/{runs,artifacts,build}` — via
  `merlin.common.artifacts` (`start_run`/`new_product`/`cache_dir`), never hand-built paths. A
  PreToolUse/pre-commit gate blocks writes outside `out/`. See `.claude/skills/artifact-layout`.
- **Docs**: durable docs live in `docs/` under `reference/` (code-derived), `guides/` (how-to),
  `design/` (rationale), each with YAML front-matter. Point-in-time reports go under
  `out/artifacts/`, not `docs/`. Generated docs (CLI/module/schema indexes, the hub) are regenerated,
  not hand-edited. See `.claude/skills/docs-layout`.

## Before you open a PR

- Keep PRs focused and reasonably small; write a clear description of the change and its rationale.
- Ensure the gates pass (the pre-commit hook runs these; you can run them directly):
  ```bash
  python build_tools/scripts/check_structure.py        # repo/test structure
  python build_tools/scripts/check_artifact_layout.py  # out/ layout
  python build_tools/scripts/check_docs.py             # docs freshness / front-matter
  .venv/bin/python -m pytest merlin/tests              # tests
  ```
- Match the surrounding code style. C/C++ follows `.clang-format`/`.clang-tidy`; Python follows the
  `[tool.ruff]` config in `pyproject.toml`.
- Commit messages follow `type(scope): imperative summary` (e.g. `fix(runtime): ...`).

## Code of conduct

Be respectful and constructive. This is a research project — questions and design discussion in the
issue tracker are welcome.
