# Contributing to merlin

Thanks for your interest in contributing! merlin is an active research codebase; this guide covers
the conventions the tooling enforces so your PRs land smoothly.

## Setup

Use Python 3.12 or newer and [uv](https://docs.astral.sh/uv/). To try the
compiler before contributing, start with the [small compilation example](examples/compilation/quickstart/README.md).
The contributor environment adds the compact suite's dependencies:

```bash
uv venv --python 3.12
uv pip install -e '.[dev,verify,semantic-search,targetgen]'
.venv/bin/python build_tools/hooks/install_git_hooks.py   # enable the pre-commit gate (per clone)
```

One Merlin distribution contains compiler and research code. Select additional
dependency extras such as `experiments` or `analysis` on that same package when
needed. The console and independent timing observer live under `tools/`. See [getting started](docs/guides/01-inputs.md)
for workflow-specific extras and external prerequisites.

External dependencies (chipyard, model2MLIR, boards, sibling repos) are **not vendored** — point at
them via environment variables (copy `.env.example` → `.env` and edit). The repo locates itself via
`merlin.common.paths.repo_root()`; never hard-code absolute paths.

## Where things go

- **Code**: shared compiler machinery in `src/merlin`; test and backend generation engines in
  `src/merlin/development`, workflow selection/tracking in `src/merlin/experiments`,
  external-framework comparisons in `src/merlin/evaluation/frameworks`, and analysis
  in `src/merlin/analysis`. There is one installed
  `merlin` namespace; do not create another root package or project-named façade.
  Experiment definitions live in
  `experiments/catalog.yaml`; the root experiments tree contains only nine authored
  catalog/definition/template/instruction files. Generated corpora, input bundles and
  historical run payloads stay under ignored `out/` and are selected explicitly. See the
  [repository structure](docs/reference/repo_structure.md).
- **Tests**: the compact suite in `tests/<bucket>/test_<area>.py` has a
  repository-wide maximum of 150 collected cases, including parameterization.
  Another package or example directory cannot bypass that budget. Resolve paths via
  `merlin.common.paths`, never `Path(__file__).parents[N]`.
- **Generated output**: only under `out/{runs,artifacts,build}` — via
  `merlin.common.artifacts` (`start_run`/`new_product`/`cache_dir`), never hand-built paths. A
  PreToolUse/pre-commit gate blocks writes outside `out/`. See `.claude/skills/artifact-layout`.
- **Docs**: follow the authored reading order in `docs/README.md`. Keep the numbered
  chapters and compact references source-backed. Generated references and run reports
  belong under ignored `out/`; private audit notes belong in `tmp/`. Add local
  instructions only when a directory has distinct constraints.

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
  .venv/bin/python build_tools/checks/check_structure.py        # repo/test structure
  .venv/bin/python build_tools/checks/check_artifact_layout.py  # out/ layout
  .venv/bin/python build_tools/checks/check_docs.py             # docs freshness / front-matter
  .venv/bin/python build_tools/checks/check_test_budget.py --output out/build/python/ci/test-budget.json
  .venv/bin/python -m pytest tests -q -ra
  ```
- Distinguish code regressions from unavailable inputs: selected workflows include private corpora,
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
