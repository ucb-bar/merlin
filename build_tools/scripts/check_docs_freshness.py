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
  check_docs_freshness.py --ratchet  # exit 1 if a doc drifted that is NOT in docs_freshness_ratchet.txt

The drift signal was advisory everywhere, so it only grew: measured 2026-09-13, 54 of 85 docs had a
code_ref commit newer than their last_verified. --ratchet freezes that list as known debt (it may only
shrink; check_ratchets_shrink.py holds it) and fails on any NEW drift. A doc leaves the list when the
docs-doctor loop reconciles it and bumps last_verified.

Needs full history: in a shallow clone `git log -1 -- <path>` returns the one grafted commit for every
path, so every doc would read as drifted.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _front_matter  # noqa: E402  (sibling module, stdlib only)

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"

REQUIRED_KEYS = ("title", "kind", "status", "owner", "last_verified")
KINDS = {"reference", "guide", "design"}
STATUSES = {"current", "draft", "superseded"}
SKIP_NAMES = {"README.md", "AGENT.md"}
# Generated references legitimately carry no hand-authored front-matter.
GENERATED = {"reference/cli.md", "reference/module_index.md", "reference/schemas.md",
             "reference/repo_map.md"}
RATCHET = ROOT / "build_tools" / "scripts" / "docs_freshness_ratchet.txt"


parse_front_matter = _front_matter.parse


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
        # A code_ref that does not resolve makes the drift signal unmeasurable for that doc: the
        # ref reads as infinitely old and the doc is drifted forever, or -- worse -- the reader
        # silently drops it. Four such refs were pointing at files that had moved under
        # merlin/targets/<target>/backend/ when this rule was added (2026-09-16).
        for ref in fm.get("code_refs") or []:
            if not (ROOT / ref).exists():
                errs.append(f"{rel}: code_ref {ref!r} does not exist")
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


def _ratchet_entries() -> set[str]:
    if not RATCHET.is_file():
        return set()
    out = set()
    for line in RATCHET.read_text(encoding="utf-8").splitlines():
        entry = line.split("#", 1)[0].strip()
        if entry:
            out.add(entry)
    return out


def ratchet() -> int:
    allowed = _ratchet_entries()
    drifted = {d["doc"]: d for d in drift()}
    new = sorted(set(drifted) - allowed)
    healed = sorted(allowed - set(drifted))
    if healed:
        print(f"[note] {len(healed)} ratcheted doc(s) no longer drift; delete their lines from "
              f"{RATCHET.name}: {', '.join(healed)}")
    if new:
        sys.stderr.write(f"docs freshness FAILED -- {len(new)} doc(s) drifted behind their code_refs and "
                         f"are not in {RATCHET.name}:\n")
        for doc in new:
            refs = ", ".join(f"{s['path']}@{s['last_commit']}" for s in drifted[doc]["stale_code_refs"])
            sys.stderr.write(f"  - {doc} (verified {drifted[doc]['last_verified']}) < {refs}\n")
        sys.stderr.write("Reconcile the doc with its code and bump last_verified (docs-doctor skill); "
                         "do not add it to the ratchet.\n")
        return 1
    print(f"docs freshness: OK ({len(drifted)} drifted, all ratcheted; the list may only shrink)")
    return 0


def main(argv: list[str]) -> int:
    if "--ratchet" in argv:
        return ratchet()
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
