# Contributing to merlin

Thanks for your interest in contributing! merlin is an active research codebase; this guide covers
the conventions the tooling enforces so your PRs land smoothly.

## Setup

merlin uses [uv](https://docs.astral.sh/uv/):

```bash
uv venv
uv pip install -e '.[dev,verify,semantic-search,targetgen]'
.venv/bin/python build_tools/scripts/install_git_hooks.py   # enable the pre-commit gate (per clone)
```

One Merlin distribution contains compiler and research code. Select additional
dependency extras such as `experiments` or `analysis` on that same package when
needed; do not install the retired code distributions from `packages/`. The timing
observer is an independent package. See [getting started](docs/guides/getting_started.md)
for workflow-specific extras and external prerequisites.

External dependencies (chipyard, model2MLIR, boards, sibling repos) are **not vendored** — point at
them via environment variables (copy `.env.example` → `.env` and edit). The repo locates itself via
`merlin.common.paths.repo_root()`; never hard-code absolute paths.

## Where things go

- **Code**: shared compiler machinery in `src/merlin`; research workflows in
  `src/merlin_experiments` and the logical research owners under `src/merlin`.
  Preserve the shared namespace initializers owned by core. Experiment definitions live in
  `experiments/catalog.yaml`, not in ad-hoc launch scripts. See the
  [repository structure](docs/reference/repo_structure.md).
- **Tests**: the compact suite in `merlin/tests/<bucket>/test_<area>.py` has a
  repository-wide maximum of 150 collected cases, including parameterization.
  Another package or example directory cannot bypass that budget. Resolve paths via
  `merlin.common.paths`, never `Path(__file__).parents[N]`.
- **Generated output**: only under `out/{runs,artifacts,build}` — via
  `merlin.common.artifacts` (`start_run`/`new_product`/`cache_dir`), never hand-built paths. A
  PreToolUse/pre-commit gate blocks writes outside `out/`. See `.claude/skills/artifact-layout`.
- **Docs**: durable docs live in `docs/` under `reference/` (code-derived), `guides/` (how-to),
  `design/` (rationale), each with YAML front-matter. Point-in-time reports go under
  `out/artifacts/`, not `docs/`. Generated docs (CLI/module/schema indexes, the hub) are regenerated,
  not hand-edited. See `.claude/skills/docs-layout`.

## Module ownership and readability

A module should explain one coherent concept: its inputs, governing invariants,
algorithm and result. Keep a record and its validation together when they express
the same invariant. A reader should not need a chain of forwarding files to
understand one transformation. The [repository map](docs/reference/repo_structure.md)
identifies the current owner to open first.

Aim for roughly 100–350 handwritten lines when the responsibility fits naturally.
Review decomposition around 400 lines; ordinary new or substantially rewritten
modules should generally stay below 600. These are review thresholds, not targets
to fill. Generated code, declarative data and mathematical sources need separate
judgment. A longer coherent algorithm can be justified, but a file just below a
larger CI ceiling is not automatically maintainable.

Split when responsibilities change independently, have different consumers, or
cross a real contract: planning versus materialization, target lowering versus
independent verification, or process execution versus artifact admission. Keep a
short orchestration function that shows the full flow. Do not split merely to
move lines elsewhere, make a module for every function, or introduce catch-all
`utils`, forwarding wrappers and parameter bundles used only to reconnect the
pieces. Private helpers that serve one algorithm usually stay beside it.

Public entry points should document their accepted inputs, result and important
refusal conditions. Comments explain reasons and constraints: numerical ordering,
ownership, aliasing, units, synchronization or an external ABI requirement. For
example, “Ownership must be derived after the final allocation move” explains a
pipeline ordering constraint. Comments that merely narrate an assignment add
maintenance work. Avoid change diaries, duplicated code descriptions and manually
maintained module inventories in implementation files.

New callers use canonical owners. Compatibility modules preserve existing imports
and frozen research identities; they must name their actual owner and contain no
second implementation. Do not treat every older directory as a compatibility
alias: `llvmlower` and `targetgen` still contain active implementations.

## Before you open a PR

- Keep PRs focused and reasonably small; write a clear description of the change and its rationale.
- Ensure the gates pass (the pre-commit hook runs these; you can run them directly):
  ```bash
  .venv/bin/python build_tools/scripts/check_structure.py        # repo/test structure
  .venv/bin/python build_tools/scripts/check_artifact_layout.py  # out/ layout
  .venv/bin/python build_tools/scripts/check_docs.py             # docs freshness / front-matter
  .venv/bin/python build_tools/scripts/check_test_budget.py --output out/build/python/ci/test-budget.json
  .venv/bin/python -m pytest merlin/tests -q -ra
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
