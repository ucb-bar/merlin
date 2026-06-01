# merlin

A compiler-centered framework for studying **which hardware/software abstractions are worth
exposing to the compiler**.

> Status: **scaffold**. This repository currently provides structure, schemas, docs, and
> placeholder modules — not working compiler functionality. See `docs/` before adding code.

## Three workstreams

merlin is organized so three Claude Code sessions can work in parallel, coordinating through
**shared schemas** (`merlin/schemas/`), not prose:

1. **TargetGen / dialect generation** — ISA/docs/RTL → target contract → dialect plan → dialect
   scaffold. Owns `merlin/python/merlin/targetgen/`, `merlin/targets/`.
2. **Kernel abstraction mining** — external kernels → kernel records → abstraction candidates →
   policy rules. Owns `merlin/python/merlin/kernels/`, `merlin/integrations/`.
3. **Design-pressure & DSE** — workloads → design-pressure reports → candidate contracts →
   variant comparison → exploitability. Owns `merlin/python/merlin/design_pressure/`,
   `merlin/python/merlin/dse/`.

See `docs/parallel_workstreams.md` for ownership and the shared-artifact flow.

## Two compiler planes

- **xDSL (Python) — default prototyping plane.** Fast iteration on dialects, contracts, interface
  IR, and DSE experiments (`merlin/python/merlin/xdsl_dialects/`).
- **MLIR/C++ — eventual stable plane.** Durable dialects, lowering passes, target plugins
  (`merlin/compiler/`). Scaffold only for now.

## Core dialects

`merlin.contract`, `merlin.schedule`, `merlin.interface`, `merlin.runtime`. DSE search spaces and
kernel-derived policies stay as **schemas/YAML/JSON** — there is no `merlin.dse` or `merlin.kernel`
dialect. See `docs/dialects.md`.

## Layout

```
build_tools/  docs/  third_party/  tools/   merlin/   (build/ output/ tmp/ are gitignored)
```

Almost all code lives under the internal `merlin/` tree. See `docs/repo_structure.md`.

## Getting started

The project uses [uv](https://docs.astral.sh/uv/). The environment includes the dev tools and the
xDSL prototyping plane:

```bash
uv sync --all-extras                       # create .venv, install merlin (editable) + xdsl + dev deps
uv run python build_tools/scripts/check_structure.py   # verify the scaffold is intact
uv run pytest                              # run the smoke tests
```

(Equivalent without uv: `pip install -e '.[dev,xdsl]'`.)

External projects (XNNPACK, Autocomp, Exo, Triton, IREE, ...) are **adapters** under
`merlin/integrations/`, never vendored. Pass their repos by env var, e.g.:

```bash
export MERLIN_XNNPACK_REPO=/path/to/XNNPACK
export MERLIN_AUTOCOMP_REPO=/path/to/autocomp
```

Every directory has an `AGENT.md` describing its purpose and constraints — read it before
working in that directory.
