#!/usr/bin/env python3
"""Regenerate a data-only support provider's contracts from the example's reviewed contracts.

A support provider must carry its contract inside its own root (``providers.contained_resource``),
and a data-only provider's contract is exactly the reviewed ``examples/<example>/target/contracts``
file plus the provider-owned ``plugin`` block (``merlin/tests/gemmini/test_support_contract_agrees.py``
holds that equality). This rewrites each provider copy as the reviewed bytes with the provider's
CURRENT ``plugin`` block inserted after the reviewed ``family:`` line, so a reviewed-contract edit is
propagated by running one command instead of by hand-copying YAML::

    python build_tools/scripts/sync_support_contract.py <example>          # rewrite
    python build_tools/scripts/sync_support_contract.py <example> --check  # exit 1 when stale
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
HEADER = (
    "# GENERATED from ../../target/contracts/{name} plus this provider's plugin block by\n"
    "# build_tools/scripts/sync_support_contract.py. Edit the reviewed contract, then re-run it.\n"
)


def _plugin_block(plugin: dict) -> str:
    return yaml.safe_dump({"plugin": plugin}, sort_keys=False, default_flow_style=False)


def render(reviewed_text: str, plugin: dict | None, name: str) -> str:
    """The provider copy: the reviewed text, with ``plugin`` (if any) after the top-level ``family:``."""
    lines = reviewed_text.splitlines(keepends=True)
    out = [HEADER.format(name=name)]
    inserted = plugin is None
    for line in lines:
        out.append(line)
        if not inserted and line.startswith("family:"):
            out.append(_plugin_block(plugin))
            inserted = True
    if not inserted:
        out.append(_plugin_block(plugin))
    return "".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("example", help="the examples/<example> directory name")
    parser.add_argument("--check", action="store_true", help="report staleness instead of rewriting")
    args = parser.parse_args(argv)
    reviewed_dir = ROOT / "examples" / args.example / "target" / "contracts"
    provided_dir = ROOT / "examples" / args.example / "support" / "contracts"
    if not reviewed_dir.is_dir() or not provided_dir.is_dir():
        print(f"no reviewed or provider contracts for example {args.example!r}", file=sys.stderr)
        return 2
    stale = []
    for provided in sorted(provided_dir.glob("*.yaml")):
        reviewed = reviewed_dir / provided.name
        if not reviewed.is_file():
            stale.append(f"{provided} has no reviewed counterpart {reviewed}")
            continue
        current = yaml.safe_load(provided.read_text(encoding="utf-8")) or {}
        wanted = render(reviewed.read_text(encoding="utf-8"), current.get("plugin"), provided.name)
        if provided.read_text(encoding="utf-8") == wanted:
            continue
        if args.check:
            stale.append(f"{provided} differs from {reviewed} + its plugin block")
        else:
            provided.write_text(wanted, encoding="utf-8")
            print(f"rewrote {provided.relative_to(ROOT)}")
    for line in stale:
        print(line, file=sys.stderr)
    return 1 if stale else 0


if __name__ == "__main__":
    raise SystemExit(main())
