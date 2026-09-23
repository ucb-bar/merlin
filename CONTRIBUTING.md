# Contributing to merlin

Thanks for your interest in contributing! merlin is an active research codebase; this guide covers
the conventions the tooling enforces so your PRs land smoothly.

## Setup

merlin uses [uv](https://docs.astral.sh/uv/):

```bash
uv venv
uv pip install -e '.[dev,xdsl,targetgen]' -e packages/merlin-experiments -e packages/merlin-dse -e packages/merlin-mining -e packages/merlin-analysis
.venv/bin/python build_tools/scripts/install_git_hooks.py   # enable the pre-commit gate (per clone)
```

This resolves each distribution's declared dependencies, including the pinned AET revision;
do not use `--no-deps` for a fresh setup. Run `.venv/bin/python` or `uv run --no-sync`
afterwards: a root-only `uv sync` can remove separately installed extensions. For a core-only
checkout, omit the four `packages/` arguments. See [getting started](docs/guides/getting_started.md)
for workflow-specific installation and external prerequisites.

External dependencies (chipyard, model2MLIR, boards, sibling repos) are **not vendored** — point at
them via environment variables (copy `.env.example` → `.env` and edit). The repo locates itself via
`merlin.common.paths.repo_root()`; never hard-code absolute paths.

## Where things go

- **Code**: shared compiler machinery in `src/merlin`; optional research in `packages/`.
  Preserve the shared namespace initializers owned by core. Experiment definitions live in
  `experiments/catalog.yaml`, not in ad-hoc launch scripts. See the
  [repository structure](docs/reference/repo_structure.md).
- **Tests**: `merlin/tests/<bucket>/test_<area>.py`, one of the subsystem buckets
  (`kernels/ rvv/ dse/ gemmini/ targetgen/ ir/ runtime/ infra/`). Distribution-specific installation
  and orchestration tests also live in `packages/*/tests/`. Resolve paths via
  `merlin.common.paths`, never `Path(__file__).parents[N]`.
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
  .venv/bin/python build_tools/scripts/check_structure.py        # repo/test structure
  .venv/bin/python build_tools/scripts/check_artifact_layout.py  # out/ layout
  .venv/bin/python build_tools/scripts/check_docs.py             # docs freshness / front-matter
  .venv/bin/python -m pytest merlin/tests packages/merlin-experiments/tests packages/merlin-dse/tests packages/merlin-mining/tests
  ```
- Distinguish code regressions from unavailable inputs: the full suite includes private corpora,
  retained experiment records, compiler builds and hardware. Report the actual passing subset and
  missing prerequisites; do not manufacture fixtures or weaken grading to make a fresh clone green.
- Match the surrounding code style. C/C++ follows `.clang-format`/`.clang-tidy`; Python follows the
  `[tool.ruff]` config in `pyproject.toml`, formatted with the pinned ruff
  (`uvx ruff@0.16.8 check --select I --fix <files> && uvx ruff@0.16.8 format <files>`). The pre-commit
  hook refuses changed Python that is not formatted. If a formatter run moves a `# target-ok:`-style
  marker off the line it excuses, pin that statement with `  # fmt: skip`.
- Restyle-only commits are listed in `.git-blame-ignore-revs`; run
  `git config blame.ignoreRevsFile .git-blame-ignore-revs` once so blame skips them.
- Commit messages follow `type(scope): imperative summary` (e.g. `fix(runtime): ...`).

## Code of conduct

Be respectful and constructive. This is a research project — questions and design discussion in the
issue tracker are welcome.
