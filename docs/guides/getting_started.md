---
title: Getting started
kind: guide
status: current
owner: core
last_verified: 2026-07-07
related: [architecture, repo_structure, dse, kernel_mining, targetgen]
code_refs: [pyproject.toml, build_tools/scripts, merlin/python/merlin/common/artifacts.py]
---

# Getting started

A newcomer's path from clone to a first run. For *what* the repo is, read
[Architecture](../reference/architecture.md); for *where things live*, read
[Repository structure](../reference/repo_structure.md).

## 1. Environment

The project uses [uv](https://docs.astral.sh/uv/) (Python 3.13).

```bash
uv sync --all-extras                                    # .venv + merlin (editable) + xdsl + dev deps
uv run python build_tools/scripts/check_structure.py    # verify the tree/docs invariants hold
.venv/bin/python -m pytest merlin/tests                 # run the suite (plain `python` is not on PATH)
```

External tool repos (XNNPACK, Autocomp, Exo, …) are **adapters**, never vendored — pass them by env
var (e.g. `export MERLIN_XNNPACK_REPO=/path/to/XNNPACK`). See [Integrations](integrations.md).

## 2. The CLI surface

Every workflow is a console-script (`pip install -e` installs them); each is a thin entrypoint over
a `merlin.*` module. The full table is the generated [CLI reference](../reference/cli.md). Run any
with `--help`. The main entry points, by workstream:

| Workstream | CLI | Guide |
|---|---|---|
| Kernel mining | `merlin-rvv-mine`, `kernel-index`, `kernel-bench` | [Kernel mining](kernel_mining.md) |
| Design-pressure & DSE | `merlin-design-pressure`, `merlin-dse`, `merlin-dse-guidance` | [DSE](dse.md), [DSE guidance](dse_guidance.md) |
| Target generation | `merlin-targetgen`, `merlin-compare` | [Target generation](targetgen.md) |

## 3. Where output goes

**Never** hand-build an output path. Generated output lives in exactly three roots — `runs/`,
`artifacts/`, `build/` — created via `merlin.common.artifacts` (`start_run`, `new_product`,
`cache_dir`, `recaptures_dir`). A PreToolUse hook blocks writes elsewhere. See `CLAUDE.md`
"Generated-output convention" and the `artifact-layout` skill. Point-in-time reports live under
`artifacts/`, **not** in `docs/`.

## 4. Conventions before you commit

- **Shared working tree** — do not switch branches; commit on the current branch, small verified
  batches (`CLAUDE.md`).
- **Tests** go in `merlin/tests/<bucket>/test_*.py` (`test-layout` skill).
- **Docs** carry front-matter and are indexed by the hub; run
  `build_tools/scripts/gen_docs_index.py` after adding one. See `docs/AGENT.md`.
- Re-run `check_structure.py` after any structural change.
