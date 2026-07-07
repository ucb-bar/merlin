#!/usr/bin/env python3
"""Docs anti-drift: validate front-matter, and detect docs that fell behind their code.

Two independent signals:

  1. SCHEMA (hard) — every docs/ file that HAS front-matter must have the required keys with
     valid enum values. Files with no front-matter are reported soft (they show up as
     "Uncategorized" in the hub) so a work-in-progress doc never breaks the build.
  2. DRIFT (soft) — for each doc, compare its `last_verified` date to the newest git commit
     date touching any of its `code_refs`. A doc whose code moved on after it was last verified
     is a drift candidate: the deterministic worklist the docs-doctor agent consumes.

Usage:
  check_docs_freshness.py            # human report (schema + drift + uncategorized)
  check_docs_freshness.py --check    # exit 1 on SCHEMA errors only (fast; wired into check_structure)
  check_docs_freshness.py --json     # machine worklist of drift candidates (for the docs-doctor loop)
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"

REQUIRED_KEYS = ("title", "kind", "status", "owner", "last_verified")
KINDS = {"reference", "guide", "design"}
STATUSES = {"current", "draft", "superseded"}
SKIP_NAMES = {"README.md", "AGENT.md"}
# Generated references legitimately carry no hand-authored front-matter.
GENERATED = {"reference/cli.md", "reference/module_index.md", "reference/schemas.md"}


def parse_front_matter(text: str) -> dict | None:
    if not text.startswith("---\n"):
        return None
    end = text.find("\n---", 4)
    if end == -1:
        return None
    fm: dict = {}
    for line in text[4:end].splitlines():
        if not line.strip() or ":" not in line:
            continue
        key, _, val = line.partition(":")
        key, val = key.strip(), val.strip()
        if val.startswith("[") and val.endswith("]"):
            fm[key] = [x.strip() for x in val[1:-1].split(",") if x.strip()]
        else:
            fm[key] = val
    return fm


def _docs() -> list[Path]:
    return sorted(p for p in DOCS.rglob("*.md") if p.name not in SKIP_NAMES)


def schema_errors() -> list[str]:
    errs: list[str] = []
    for p in _docs():
        rel = p.relative_to(DOCS).as_posix()
        fm = parse_front_matter(p.read_text(encoding="utf-8"))
        if fm is None:
            continue  # soft: no front-matter -> Uncategorized, not an error
        for k in REQUIRED_KEYS:
            if k not in fm or not fm[k]:
                errs.append(f"{rel}: missing front-matter key '{k}'")
        if fm.get("kind") and fm["kind"] not in KINDS:
            errs.append(f"{rel}: invalid kind {fm['kind']!r} (expected {sorted(KINDS)})")
        if fm.get("status") and fm["status"] not in STATUSES:
            errs.append(f"{rel}: invalid status {fm['status']!r} (expected {sorted(STATUSES)})")
        lv = fm.get("last_verified", "")
        if lv and not (len(lv) == 10 and lv[4] == "-" and lv[7] == "-"):
            errs.append(f"{rel}: last_verified {lv!r} not YYYY-MM-DD")
    return errs


def uncategorized() -> list[str]:
    out = []
    for p in _docs():
        rel = p.relative_to(DOCS).as_posix()
        if rel in GENERATED:
            continue
        if parse_front_matter(p.read_text(encoding="utf-8")) is None:
            out.append(rel)
    return out


def _last_commit_date(path: str) -> str | None:
    """Newest committer date (YYYY-MM-DD) touching path, or None if untracked/unknown."""
    r = subprocess.run(["git", "-C", str(ROOT), "log", "-1", "--format=%cs", "--", path],
                       capture_output=True, text=True)
    d = (r.stdout or "").strip()
    return d or None


def drift() -> list[dict]:
    """Docs whose last_verified predates the newest commit to a code_ref."""
    out: list[dict] = []
    for p in _docs():
        fm = parse_front_matter(p.read_text(encoding="utf-8"))
        if not fm:
            continue
        lv = fm.get("last_verified", "")
        refs = fm.get("code_refs", []) or []
        stale = []
        for ref in refs:
            if not (ROOT / ref).exists():
                stale.append({"path": ref, "last_commit": "MISSING"})
                continue
            d = _last_commit_date(ref)
            if d and lv and d > lv:
                stale.append({"path": ref, "last_commit": d})
        if stale:
            out.append({"doc": p.relative_to(DOCS).as_posix(), "last_verified": lv,
                        "stale_code_refs": stale})
    return out


def main(argv: list[str]) -> int:
    if "--check" in argv:
        errs = schema_errors()
        if errs:
            sys.stderr.write("docs front-matter schema FAILED:\n")
            for e in errs:
                sys.stderr.write(f"  - {e}\n")
            return 1
        print("docs front-matter: OK")
        return 0
    if "--json" in argv:
        print(json.dumps({"drift": drift(), "uncategorized": uncategorized()}, indent=2))
        return 0
    # human report
    errs, drft, uncat = schema_errors(), drift(), uncategorized()
    print(f"schema: {'OK' if not errs else str(len(errs)) + ' error(s)'}")
    for e in errs:
        print(f"  - {e}")
    print(f"drift candidates: {len(drft)}")
    for d in drft:
        refs = ", ".join(f"{s['path']}@{s['last_commit']}" for s in d["stale_code_refs"])
        print(f"  - {d['doc']} (verified {d['last_verified']}) < {refs}")
    print(f"uncategorized (no front-matter): {len(uncat)}")
    for u in uncat:
        print(f"  - {u}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
