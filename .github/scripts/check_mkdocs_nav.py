#!/usr/bin/env python3
"""Verify that every file referenced in mkdocs.yml's nav exists on disk.

Run from the repo root. Exits 1 (with a list of missing paths) if any nav
entry points at a file that does not exist; 0 otherwise.

Used by .github/workflows/docs-pages.yml because zensical's `--strict` flag
is currently unsupported.
"""

from __future__ import annotations

import pathlib
import sys

import yaml


def _ignore_unknown(_loader, _tag_suffix, _node):
    return None


def main() -> int:
    yaml.SafeLoader.add_multi_constructor("!", _ignore_unknown)
    docs_root = pathlib.Path("docs")
    nav = yaml.safe_load(open("mkdocs.yml")).get("nav", [])

    missing: list[str] = []

    def walk(node):
        if isinstance(node, dict):
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)
        elif isinstance(node, str):
            if not (docs_root / node).exists():
                missing.append(node)

    walk(nav)

    if missing:
        print("Missing nav targets:")
        for path in missing:
            print(f"  - docs/{path}")
        return 1

    print("All mkdocs nav targets exist.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
