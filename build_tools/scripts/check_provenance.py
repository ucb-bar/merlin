#!/usr/bin/env python3
"""Gate: a result that claims a hardware verdict must record which hardware revision it came from.

WHY. A session certified a microkernel against the only saturn revision containing the outer-product unit
while believing it was the revision named for the tapeout, which does not contain that unit at all.
Nothing in the artifact recorded which revision the numbers belonged to, so the mistake was invisible
until someone happened to name a commit in conversation. A result attributed to the wrong device is worse
than no result.

WHAT IS CHECKED, in increasing severity:

1. The pin registry loads and every pin declares a full sha (always enforced -- a malformed registry means
   nothing downstream can be verified).
2. Tracked reports that CLAIM a verdict (``certified: true``) carry a ``provenance`` block naming the
   revision. Enforced for reports not in the ratchet list below.
3. Where the checkout is reachable, pins are verified and material drift is reported.

ADVISORY BY DEFAULT, RATCHETED. Existing reports predate this convention and are listed in
``provenance_ratchet.txt``; the list may only shrink. New claims are enforced immediately, which is the
direction that matters -- the point is to stop producing unattributable results, not to retro-fit old ones.

Usage:
  check_provenance.py                 # tracked reports + registry
  check_provenance.py --staged        # only what is staged (pre-commit)
  check_provenance.py --stop-hook     # session gate; same checks, hook-shaped output
  check_provenance.py --verify-pins   # additionally verify pins against live checkouts
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
_ROOT = _HERE.parents[2]
sys.path.insert(0, str(_ROOT / "merlin" / "python"))

RATCHET = _HERE.parent / "provenance_ratchet.txt"

#: Reports live under the generated-output root; only these are candidates.
_REPORT_SUFFIXES = (".json",)

#: Purgeable/huge subtrees that never hold verdicts. Skipped by name so the scan stays cheap; the
#: recaptures tree alone is ~130 GB and the cache is regenerable by definition.
_SKIP_DIRS = frozenset({"cache", "recaptures", "build", ".git", "__pycache__"})

#: Bound on files examined. Reported when hit rather than silently truncating -- a gate that quietly
#: stopped looking would read as "nothing to report".
_SCAN_CAP = 20000


def _tracked(staged: bool) -> list[Path]:
    args = (["git", "diff", "--cached", "--name-only", "--diff-filter=ACM"] if staged
            else ["git", "ls-files"])
    got = subprocess.run(args, capture_output=True, text=True, cwd=_ROOT)
    if got.returncode != 0:
        return []
    return [_ROOT / line for line in got.stdout.splitlines() if line.strip()]


def _scan_out() -> list[Path]:
    """Verdict-bearing report candidates under the generated-output root.

    The reports this gate exists for are UNTRACKED -- `out/` is gitignored -- so scanning only tracked
    files checked nothing at all, which is how the first version of this gate passed while an
    unattributed certification sat on disk.
    """
    try:
        from merlin.common.paths import artifacts_dir
        root = Path(artifacts_dir())
    except Exception:                                     # noqa: BLE001 — no output root yet
        return []
    if not root.is_dir():
        return []
    found: list[Path] = []
    stack = [root]
    seen = 0
    while stack:
        d = stack.pop()
        try:
            for entry in d.iterdir():
                seen += 1
                if seen > _SCAN_CAP:
                    print(f"  NOTE: scan capped at {_SCAN_CAP} entries; some reports were not examined")
                    return found
                if entry.is_symlink():
                    continue
                if entry.is_dir():
                    if entry.name not in _SKIP_DIRS:
                        stack.append(entry)
                elif entry.suffix in _REPORT_SUFFIXES:
                    found.append(entry)
        except OSError:
            continue
    return found


def _ratcheted() -> set[str]:
    if not RATCHET.is_file():
        return set()
    return {l.strip() for l in RATCHET.read_text(encoding="utf-8").splitlines()
            if l.strip() and not l.startswith("#")}


def _claims_a_verdict(payload: object) -> bool:
    """True when a report asserts a hardware result, i.e. when attribution matters."""
    if not isinstance(payload, dict):
        return False
    for key in ("certified", "correct", "passed"):
        if payload.get(key) is True:
            return True
    return False


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    staged = "--staged" in argv
    stop_hook = "--stop-hook" in argv
    verify_pins = "--verify-pins" in argv
    problems: list[str] = []
    notes: list[str] = []

    # 1. The registry itself.
    try:
        from merlin.common import provenance as P
        pins = P.load_pins()
    except Exception as exc:                              # noqa: BLE001 — any failure is fatal here
        print(f"provenance: FAILED — pin registry unusable: {exc}")
        return 1
    notes.append(f"{len(pins)} pin(s) declared: {', '.join(sorted(pins))}")

    # 2. Tracked reports that claim a verdict.
    allow = _ratcheted()
    checked = 0
    candidates = _tracked(staged) if staged else (_tracked(False) + _scan_out())
    for path in candidates:
        if path.suffix not in _REPORT_SUFFIXES or not path.is_file():
            continue
        try:
            rel = path.relative_to(_ROOT).as_posix()
        except ValueError:
            rel = path.as_posix()
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not _claims_a_verdict(payload):
            continue
        checked += 1
        prov = payload.get("provenance") if isinstance(payload, dict) else None
        if prov:
            continue
        if rel in allow:
            notes.append(f"ratcheted (pre-dates the convention): {rel}")
            continue
        problems.append(f"{rel}: claims a verdict but records no `provenance` block, so the result "
                        f"cannot be attributed to a hardware revision. Record one via "
                        f"merlin.common.provenance.record(), or add the path to "
                        f"{RATCHET.name} with a reason.")

    # 3. Optional live verification.
    if verify_pins:
        for name in sorted(pins):
            got = P.verify(name)
            if got.ok:
                notes.append(f"pin {name}: ok at {got.observed.commit[:12]}")
                continue
            detail = "; ".join([*got.drift,
                                *([f"missing {list(got.missing_paths)}"] if got.missing_paths else []),
                                *([f"forbidden present {list(got.forbidden_present)}"]
                                  if got.forbidden_present else [])])
            notes.append(f"pin {name}: DRIFT — {detail}")

    for n in notes:
        print(f"  {n}")
    if problems:
        print("provenance: FAILED")
        for p in problems:
            print(f"  - {p}")
        return 1
    print(f"provenance: OK ({checked} verdict-claiming report(s) checked)"
          + (" [stop-hook]" if stop_hook else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
