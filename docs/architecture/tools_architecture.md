# `tools/` Architecture — the Subcommand-Package Pattern

This document describes the structural pattern that governs everything under
`tools/`. It is the single rule for placing new code in the developer CLI.

## The pattern in one sentence

**Every `./merlin` subcommand is a Python package with a `cli.py` entry.**

```
tools/
├── merlin.py                # CLI dispatcher (imports each subcommand's cli)
├── utils.py                 # shared helpers (REPO_ROOT, run, eprint, …)
├── <subcmd>/                # package — one per registered subcommand
│   ├── __init__.py          # extension-points map / mental model
│   ├── cli.py               # argparse + main dispatch (the entry point)
│   └── <module>.py          # topic-specific helpers (one per concern)
```

Small subcommands that need no helpers (≤ ~400 LOC, no co-resident concerns)
may live as a single top-level file (e.g. `tools/ci.py`, `tools/sim.py`).
**Everything else is a package.**

The earlier `*_cmd.py` shim convention has been removed; the package's
`cli.py` IS the shim. `tools/merlin.py:COMMANDS` imports via
`from <subcmd> import cli as <subcmd>_cmd` (alias only).

## Why this pattern

| Property | How the pattern delivers it |
|---|---|
| **Clear ownership** | Every helper has exactly one subcommand it belongs to. Adding a helper means choosing the right `tools/<subcmd>/`. |
| **Bounded `cli.py` size** | Argparse + main + dispatch is short by construction; heavy logic lives in topic-specific modules next to it. |
| **No top-level proliferation** | New code always lands inside a package directory. `tools/` top level only ever contains `merlin.py`, `utils.py`, and a small set of single-file subcommands. |
| **Navigability** | A reader landing in `tools/` sees only single-file commands + subdirs. Inside each subdir, `__init__.py` lists the extension points; `cli.py` is the entry. |
| **Test parallelism** | `tests/<subcmd>/` mirrors `tools/<subcmd>/`. |

## Reference implementations

- `tools/ray/{cli.py, model.py, service.py}` — three-file package.
- `tools/perf/` — `cli` + `decompose` + 4 profilers + 2 plotters + 2 `.sh`
  drivers + `trace_to_profile`. The `cli.py` defers heavy imports until
  inside `main()` so `./merlin --help` stays fast.
- `tools/compile/` — `cli` + 6 internal helpers (`iree_tools`,
  `postprocess`, `radiance`, `feedback_overlay`, …) + 6 kernel-related
  helpers (`breakdown_vmfb`, `chunk_extractor`, …). A bigger package; same
  shape.
- `tools/chipyard/` — `cli` (157 LOC) + 8 topic modules (`config`,
  `recipe`, `git_ops`, `bare_metal`, `firesim`, `zephyr`, `radiance`,
  `status`). Demonstrates a clean per-subaction split.
- `tools/mcp_servers/` — `cli` + `scaffold` + 6 per-domain MCP tool registries.
  Same shape, different role: the package serves all MCP servers, with
  `./merlin mcp <name>` dispatching by registry-module name.

## How to add a new subcommand

```python
# tools/foo/__init__.py
"""Implementation package for the `foo` subcommand (`./merlin foo`).

Extension points:
- core.py — main logic
"""
```

```python
# tools/foo/cli.py
"""`./merlin foo` — one-line description."""

from __future__ import annotations

import argparse


def setup_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("input", help="…")
    parser.add_argument("--flag", action="store_true", help="…")


def main(args: argparse.Namespace) -> int:
    # Lazy-import heavy deps so `./merlin --help` stays fast.
    from foo import core
    return core.run(args.input, flag=args.flag)
```

```python
# tools/foo/core.py
def run(input_path: str, *, flag: bool) -> int:
    ...
    return 0
```

Register in `tools/merlin.py:COMMANDS`:

```python
from foo import cli as foo_cmd
...
COMMANDS = (
    ...,
    ("foo", foo_cmd, "One-line description (shown by ./merlin --help)", False),
)
```

`False` means the subcommand does not get a built-in `--dry-run` flag; use
`True` for subcommands that emit side effects (build, compile, etc.).

## How to grow an existing subcommand

Add a new module under the existing package:

```python
# tools/perf/new_analyzer.py
def analyze(...):
    ...
```

Import it from `cli.py` or another module in the same package. No top-level
file is ever added to `tools/` for an existing subcommand.

## What does NOT belong in `tools/`

| Kind | Goes in |
|---|---|
| One-shot debug script (used once, never imported) | `tmp/` or `tmp/archive/investigations/<name>/scripts/` |
| Multi-step shell driver chaining `./merlin` calls | `scripts/` (see `scripts/README.md`) |
| Per-target sample executable | `samples/<Board>/` |
| Captured run output (CSV, PNG, log) | `build/artifacts/<name>/` (live with build trees) |
| Frozen investigation evidence | `tmp/archive/investigations/<name>/` |
| Vendor-SDK build artifact (e.g. `libsaturn_opu.so`) | `build_tools/spike-hetero/` or equivalent topical subdir |

## Path computation inside packages

When code inside `tools/<subcmd>/` needs `REPO_ROOT`, use:

```python
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
```

(Two parents up: `<file>.py` → `<subcmd>/` → `tools/` → repo root.)

Prefer importing `utils.REPO_ROOT` instead — it's the one source of truth:

```python
import utils
REPO_ROOT = utils.REPO_ROOT
```

## Anti-patterns

- **Empty or one-file folder.** If a package has only `__init__.py` (or
  `__init__.py` + one helper), inline its contents into a top-level
  `tools/<subcmd>.py`. Folders exist to group ≥ 2 meaningful modules.
- **Top-level standalone script that isn't a registered subcommand.** Goes
  in `tools/archive/<topic>/` if frozen for reproducibility, otherwise
  inside the relevant subcommand's package.
- **Helper duplicated across packages.** Extract into `tools/utils.py` if
  general-purpose, or move to the package whose subcommand owns it.
- **Heavy logic in `cli.py`.** Argparse + main + a few dispatch lines is
  the entry surface. Anything beyond ~300 LOC of logic moves into a sibling
  module.
- **Cross-package imports of internal modules.** If `tools/perf/decompose.py`
  needs something from `tools/compile/qnn.py`, the dependency is misplaced.
  Re-think the topic boundary or extract the shared piece into
  `tools/utils.py`.
