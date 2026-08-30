#!/usr/bin/env python3
"""Create ``third_party/ext/<name>`` for every external dependency declared in ``.env``.

Tracked content must never spell a machine-local path. Content that genuinely lives outside the repo (a
vendor RTL checkout, a toolchain) is reached through a link here whose target comes from
``MERLIN_EXT_<NAME>``, so the committed bytes stay repo-relative and portable while the one
machine-specific fact stays in the one gitignored place that is already the convention for it.

Run once per clone (and again after editing ``.env``). Anything already correct is left alone.

Usage: link_externals.py [--check]
  --check  report what is missing or misaimed and exit non-zero; create nothing.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "merlin" / "python"))

from merlin.common.paths import _dotenv, ext_link, ext_path, ext_root, repo_root  # noqa: E402


def declared() -> list[str]:
    """Short keys of every external declared in ``.env`` (``MERLIN_EXT_<NAME>`` -> ``<name>``)."""
    return sorted(k[len("MERLIN_EXT_"):].lower() for k in _dotenv() if k.startswith("MERLIN_EXT_"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="report only; create nothing")
    a = ap.parse_args()

    names = declared()
    if not names:
        print("no MERLIN_EXT_* entries in .env (copy .env.example -> .env)", file=sys.stderr)
        return 1

    rc = 0
    for name in names:
        target = ext_path(name)
        link = ext_root() / name
        aimed = Path(os.readlink(link)) if link.is_symlink() else None
        if aimed == target and target.exists():
            print(f"  ok       {link.relative_to(repo_root())} -> {target}")
            continue
        if not target.exists():
            # Report it; do NOT refuse to make the link. A missing external is a setup fact about this
            # machine, and a link that aims at it is still the right committed shape -- the useful
            # failure is "chipyard is not where .env says", not a silently absent link.
            print(f"  ABSENT   {name}: MERLIN_EXT_{name.upper()} names {target}, which does not exist",
                  file=sys.stderr)
            rc = 1
        if a.check:
            if aimed != target:
                print(f"  MISSING  {link.relative_to(repo_root())} -> {target}", file=sys.stderr)
                rc = 1
            continue
        ext_link(name)
        print(f"  linked   {link.relative_to(repo_root())} -> {target}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
