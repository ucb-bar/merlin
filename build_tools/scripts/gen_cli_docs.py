#!/usr/bin/env python3
"""Generate docs/cli.md from pyproject.toml [project.scripts] (single source of truth).

The repo's CLI surface is the console-scripts declared in pyproject; there is no separate tools/
mirror to drift. This emits one table row per console-script -> backing merlin.* module.

Usage:
  python build_tools/scripts/gen_cli_docs.py           # (re)write docs/cli.md
  python build_tools/scripts/gen_cli_docs.py --check    # exit 1 if docs/cli.md is stale vs pyproject
"""
from __future__ import annotations

import sys
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PYPROJECT = REPO / "pyproject.toml"
OUT = REPO / "docs" / "cli.md"

HEADER = (
    "# CLI reference\n\n"
    "_Generated from `pyproject.toml [project.scripts]` by "
    "`build_tools/scripts/gen_cli_docs.py` — do not edit by hand; run the generator._\n\n"
    "These console-scripts are installed by `pip install -e merlin/python`. Each is a thin "
    "entrypoint over a module in the `merlin` package (no separate `tools/` layer). Run any with `--help`.\n\n"
    "| Command | Backing module |\n|---|---|\n"
)


def render() -> str:
    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    scripts = data.get("project", {}).get("scripts", {})
    rows = "".join(f"| `{name}` | `{target}` |\n" for name, target in sorted(scripts.items()))
    return HEADER + rows


def main(argv: list[str]) -> int:
    new = render()
    if "--check" in argv:
        cur = OUT.read_text(encoding="utf-8") if OUT.exists() else ""
        if cur != new:
            sys.stderr.write("docs/cli.md is stale vs pyproject [project.scripts]; "
                             "run: python build_tools/scripts/gen_cli_docs.py\n")
            return 1
        print("docs/cli.md: up to date")
        return 0
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(new, encoding="utf-8")
    print(f"wrote {OUT.relative_to(REPO)} ({len(render().splitlines())} lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
