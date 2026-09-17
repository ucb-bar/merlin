#!/usr/bin/env python3
"""Gate: a ratchet or allowlist may only SHRINK.

Every ``build_tools/scripts/*_ratchet.txt`` and ``*_allowlist.txt`` is written as a may-only-shrink
ledger of known debt, but nothing held them to it. Measured 2026-09-13: three ledgers had gained
entries (mesh_assertion 26->76, provenance 28->68, conformance 4->6) and no check could notice,
because each gate reads its own ledger as the definition of "allowed". This gate compares each
ledger's ENTRY count -- non-blank lines that are not ``#`` comments -- against a base revision and
fails when any grew.

Accepted baseline, recorded so the growth above is not re-litigated: mesh_assertion's jump came from
1cb13ecb, which made that gate fail closed after it had been scanning NOTHING (debt discovered, not
added); provenance's from ae2fe4e3 (K1 cross-framework baselines ratcheted as pre-dating the
convention); conformance's from c327c25e. From here on every ledger may only fall.

Two deliberate allowances, both visible in the diff a reviewer reads:
  * a same-count SWAP (one entry retired, another added) passes -- entries legitimately change
    spelling when a file moves -- but every new entry is printed;
  * growth passes only when the SAME change adds a ``# growth-accepted: <reason>`` line to that
    ledger (e.g. a gate widened its scope and discovered existing debt). The reason is printed.

Usage:
  check_ratchets_shrink.py                 # working tree vs HEAD
  check_ratchets_shrink.py --staged        # index vs HEAD (pre-commit)
  check_ratchets_shrink.py --base <rev>    # working tree vs <rev> (CI: the PR base / previous push)

CI needs full history (``fetch-depth: 0``): an unresolvable base FAILS -- "could not compare" is not
"nothing grew". The one exception is git's all-zero sentinel (a push that creates a branch has no
previous revision), which is reported as UNMEASURED, never as OK.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

LEDGER_DIR = "build_tools/scripts"
SUFFIXES = ("_ratchet.txt", "_allowlist.txt")
ACCEPT_MARKER = "# growth-accepted:"
ZERO_SHA = "0" * 40


def _git(root: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=check)


def _entries(text: str | None) -> list[str]:
    """Entry keys in file order: the part of each non-comment, non-blank line before any ``#``."""
    if text is None:
        return []
    out = []
    for line in text.splitlines():
        s = line.strip()
        if s and not s.startswith("#"):
            out.append(s.split("#", 1)[0].strip())
    return out


def _accept_reasons(text: str | None) -> set[str]:
    if text is None:
        return set()
    return {
        ln.strip()[len(ACCEPT_MARKER) :].strip() for ln in text.splitlines() if ln.strip().startswith(ACCEPT_MARKER)
    }


def _show(root: Path, spec: str) -> str | None:
    got = _git(root, "show", spec, check=False)
    return got.stdout if got.returncode == 0 else None


def _is_ledger(name: str) -> bool:
    return name.endswith(SUFFIXES)


def compare(root: Path, base: str, staged: bool) -> tuple[list[str], list[str]]:
    """Return (failures, notes)."""
    names = {p.name for p in (root / LEDGER_DIR).glob("*") if _is_ledger(p.name)}
    listed = _git(root, "ls-tree", "--name-only", f"{base}:{LEDGER_DIR}", check=False)
    if listed.returncode == 0:
        names |= {n for n in listed.stdout.splitlines() if _is_ledger(n)}
    if staged:
        idx = _git(root, "ls-files", "--cached", "--", LEDGER_DIR).stdout.splitlines()
        names |= {Path(n).name for n in idx if _is_ledger(n)}
    failures: list[str] = []
    notes: list[str] = []
    for name in sorted(names):
        rel = f"{LEDGER_DIR}/{name}"
        before = _show(root, f"{base}:{rel}")
        if staged:
            now = _show(root, f":{rel}")
        else:
            p = root / rel
            now = p.read_text(encoding="utf-8") if p.is_file() else None
        old, new = _entries(before), _entries(now)
        if before is None:
            if now is not None:
                notes.append(f"{rel}: new ledger ({len(new)} entries) -- its baseline starts here")
            continue
        added = [e for e in new if e not in set(old)]
        if len(new) > len(old):
            reasons = _accept_reasons(now) - _accept_reasons(before)
            if reasons:
                notes.append(f"{rel}: grew {len(old)} -> {len(new)}, ACCEPTED ({'; '.join(sorted(reasons))})")
                continue
            listing = "".join(f"\n      + {e}" for e in added) or "\n      (duplicate of an existing entry)"
            failures.append(f"{rel}: grew {len(old)} -> {len(new)} entries{listing}")
        elif added:
            notes.append(f"{rel}: {len(old)} -> {len(new)} entries, new: {', '.join(added)}")
        elif len(new) < len(old):
            notes.append(f"{rel}: shrank {len(old)} -> {len(new)}")
    return failures, notes


def main(argv: list[str]) -> int:
    staged = "--staged" in argv
    base = "HEAD"
    if "--base" in argv:
        i = argv.index("--base")
        if i + 1 >= len(argv) or not argv[i + 1].strip():
            sys.stderr.write("[FAIL] --base needs a revision\n")
            return 2
        base = argv[i + 1].strip()
    try:
        top = _git(Path.cwd(), "rev-parse", "--show-toplevel").stdout.strip()
        if not top:
            raise OSError("`git rev-parse --show-toplevel` produced no path")
        root = Path(top)
        if base == ZERO_SHA:
            print("ratchets: UNMEASURED -- no base revision (a push that created the branch); nothing was compared")
            return 0
        if _git(root, "rev-parse", "--verify", "--quiet", f"{base}^{{commit}}", check=False).returncode:
            sys.stderr.write(
                f"[FAIL] ratchets: base revision {base!r} does not resolve (shallow clone?); "
                "NOTHING was compared, which is not the same as nothing grew.\n"
            )
            return 1
        failures, notes = compare(root, base, staged)
    except (OSError, subprocess.CalledProcessError) as exc:
        sys.stderr.write(f"[FAIL] ratchets: could not list the ledgers ({exc}); nothing was compared.\n")
        return 1
    for n in notes:
        print(f"[note] {n}")
    if failures:
        sys.stderr.write("ratchets FAILED -- a may-only-shrink ledger grew:\n")
        for f in failures:
            sys.stderr.write(f"  - {f}\n")
        sys.stderr.write(
            "Fix the new debt instead. If a gate WIDENED its scope and found existing debt, "
            f"add a '{ACCEPT_MARKER} <reason>' line to that ledger in the same change.\n"
        )
        return 1
    print(f"ratchets: OK (vs {base}{', staged' if staged else ''}; every ledger held or shrank)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
