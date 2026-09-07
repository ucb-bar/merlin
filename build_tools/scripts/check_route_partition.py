#!/usr/bin/env python3
"""Gate: does every produced command buffer identify exactly one admission route?

The obligation is :mod:`merlin.runtime.route_partition` — a compiler must land in ``A`` (target work
emitted), ``H`` (an explicit host/composed route) or ``D`` (an explicit decline with no target effect),
the routes must be disjoint, and a decline must SAY so. An empty command list plus a zero exit code
identifies nothing.

Why this is a gate and not a validator: ``validate_command_buffer`` is on the grading path
(``preflight``, ``oot_runner``, ``capsule_common``), so making it reject these would retro-fail live and
archived submissions and change the instrument mid-experiment. This reports instead, against a ratchet
that MAY ONLY SHRINK, which is how the other gates in this directory handle pre-existing debt.

Modes, mirroring the sibling gates:

  --root PATH          tree to walk (repeatable); default: every run root under ``out/runs``
  --target NAME        restrict to buffers whose path names this target (repeatable)
  --json               machine-readable
  --ratchet PATH       pre-existing debt that may only shrink; unlisted new violations fail
  --fail-on-violation  exit non-zero when a non-ratcheted violation is present (default: report only)

TARGET-AGNOSTIC: targets are DISCOVERED, never listed. The discovery source is the set of conformance
specs under ``merlin/contract/capsules/conformance/``, which is the same source
``check_conformance_coverage.py`` uses, and a path is attributed to a target only when one of those
names appears as a path COMPONENT. A buffer under no known target is reported under ``?`` rather than
dropped, because a silently-unattributed violation is the failure mode this whole gate exists to catch.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "merlin" / "python"))

from merlin.common.paths import repo_root, runs_dir            # noqa: E402
from merlin.runtime.route_partition import (                   # noqa: E402
    ROUTE_ACCEPT, ROUTE_DECLINE, ROUTE_VIOLATION, VIOLATION_KINDS, route_of)

BUFFER_NAME = "command_buffer.json"

#: Where the tracked ratchet lives, beside its siblings.
DEFAULT_RATCHET = repo_root() / "build_tools" / "scripts" / "route_partition_ratchet.txt"

#: Where target names come from. One file per target, stem == the target name.
CONFORMANCE_DIR = repo_root() / "merlin" / "contract" / "capsules" / "conformance"

#: Reported for a buffer whose path names no known target. Never dropped.
UNATTRIBUTED = "?"


def known_targets() -> list[str]:
    """Every target with a conformance spec. Discovered, so adding a target needs no edit here."""
    if not CONFORMANCE_DIR.is_dir():
        return []
    return sorted(p.stem for p in CONFORMANCE_DIR.glob("*.yaml"))


def target_of(path: Path, targets: list[str]) -> str:
    """The target a buffer belongs to, by path component. Longest name first, so a qualified name
    (``saturn_opu_rvv``) is not shadowed by the shorter one it contains (``saturn_opu``)."""
    parts = set(path.parts)
    for name in sorted(targets, key=len, reverse=True):
        if name in parts:
            return name
    return UNATTRIBUTED


def _debt(target: str, kind: str, item: str) -> str:
    """One ratchet key. Same ``<target> <axis>:<item>`` shape the conformance ratchet uses."""
    return f"{target} {kind}:{item}"


#: Directory names that hold a buffer but do not NAME it. Keying the ratchet on one of these collapses
#: every violation on a target into a single entry, which cannot shrink case by case and so cannot show
#: a regression -- the first cut of this gate did exactly that and produced three useless keys.
_ANONYMOUS_DIRS = frozenset({"generated", "commands", "out", "output", "artifacts"})


def subject_of(path: Path) -> str:
    """The name to hold a violation against: the nearest enclosing directory that names something.

    A buffer normally sits at ``<capsule>/generated/command_buffer.json``, so the capsule is one level
    up from the anonymous holder. Walks up until a meaningful name is found rather than assuming a
    fixed depth, because the depth differs between run layouts.
    """
    for parent in path.parents:
        if parent.name and parent.name not in _ANONYMOUS_DIRS:
            return parent.name
    return path.parent.name or "?"


def load_ratchet(path: Path | None) -> set[str]:
    if not path or not path.is_file():
        return set()
    out = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            out.add(line)
    return out


def audit(roots: list[Path], *, targets: list[str], only: set[str] | None = None) -> dict:
    """Walk every command buffer under ``roots`` and classify it."""
    known = known_targets()
    per_target: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    violations: list[dict] = []
    n_buffers = n_unreadable = 0

    for root in roots:
        if not root.exists():
            continue
        for p in sorted(root.rglob(BUFFER_NAME)):
            tgt = target_of(p, known)
            if only and tgt not in only:
                continue
            n_buffers += 1
            try:
                cb = json.loads(p.read_text(encoding="utf-8"))
            except Exception as exc:
                # An unreadable buffer is its own fact, never folded into a route.
                n_unreadable += 1
                per_target[tgt]["unreadable"] += 1
                violations.append({"target": tgt, "path": str(p), "kind": "unreadable",
                                   "detail": f"{type(exc).__name__}: {exc}"[:200]})
                continue
            if not isinstance(cb, dict):
                n_unreadable += 1
                per_target[tgt]["unreadable"] += 1
                continue
            v = route_of(cb)
            per_target[tgt][v.route] += 1
            for kind, detail in v.violations:
                per_target[tgt][kind] += 1
                violations.append({"target": tgt, "path": str(p), "kind": kind, "detail": detail})

    return {"roots": [str(r) for r in roots], "targets_known": known,
            "n_buffers": n_buffers, "n_unreadable": n_unreadable,
            "per_target": {t: dict(c) for t, c in sorted(per_target.items())},
            "violations": violations}


def report(rec: dict, ratchet: set[str]) -> tuple[list[str], list[str]]:
    """Print the human report. Returns ``(ratcheted, unratcheted)`` debt keys."""
    print(f"command buffers walked: {rec['n_buffers']:,}"
          f"{f'  ({rec['n_unreadable']} unreadable)' if rec['n_unreadable'] else ''}")
    print(f"targets discovered:     {', '.join(rec['targets_known']) or '(none)'}")

    by_key: dict[str, int] = collections.Counter()
    for v in rec["violations"]:
        by_key[_debt(v["target"], v["kind"], subject_of(Path(v["path"])))] += 1

    ratcheted = sorted(k for k in by_key if k in ratchet)
    unratcheted = sorted(k for k in by_key if k not in ratchet)

    print()
    hdr = f"{'target':<16}{'A':>8}{'D':>8}{'!':>8}   violation kinds"
    print(hdr)
    print("-" * len(hdr))
    for tgt, counts in rec["per_target"].items():
        kinds = {k: counts[k] for k in VIOLATION_KINDS if counts.get(k)}
        if counts.get("unreadable"):
            kinds["unreadable"] = counts["unreadable"]
        print(f"{tgt:<16}{counts.get(ROUTE_ACCEPT, 0):>8}{counts.get(ROUTE_DECLINE, 0):>8}"
              f"{counts.get(ROUTE_VIOLATION, 0):>8}   {kinds or '-'}")

    if rec["violations"]:
        print("\nviolations, one example per kind per target:")
        seen: set[tuple[str, str]] = set()
        for v in rec["violations"]:
            key = (v["target"], v["kind"])
            if key in seen:
                continue
            seen.add(key)
            rel = v["path"]
            root = str(repo_root())
            rel = rel[len(root) + 1:] if rel.startswith(root) else rel
            print(f"\n  [{v['target']}] {v['kind']}")
            print(f"    {rel}")
            print(f"    {v['detail']}")

    if unratcheted:
        print(f"\n{len(unratcheted)} un-ratcheted violation group(s):")
        for k in unratcheted[:25]:
            print(f"  * {k}  x{by_key[k]}")
        if len(unratcheted) > 25:
            print(f"  ... and {len(unratcheted) - 25} more")
    if ratcheted:
        print(f"\n{len(ratcheted)} ratcheted (pre-existing debt, may only shrink)")
    stale = sorted(ratchet - set(by_key))
    if stale:
        print(f"\n{len(stale)} ratchet entr(ies) no longer needed -- REMOVE them so the ratchet "
              f"cannot hide a regression:")
        for k in stale[:15]:
            print(f"  - {k}")
    return ratcheted, unratcheted


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", action="append", default=None,
                    help="tree to walk (repeatable); default: out/runs")
    ap.add_argument("--target", action="append", default=None,
                    help="restrict to this target (repeatable)")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--ratchet", default=str(DEFAULT_RATCHET))
    ap.add_argument("--write-ratchet", action="store_true",
                    help="regenerate the ratchet from what is present now (records debt; use once)")
    ap.add_argument("--fail-on-violation", action="store_true")
    a = ap.parse_args(argv)

    roots = [Path(r) for r in a.root] if a.root else [runs_dir()]
    only = set(a.target) if a.target else None
    rec = audit(roots, targets=known_targets(), only=only)
    ratchet_path = Path(a.ratchet) if a.ratchet else None

    if a.write_ratchet:
        keys = sorted({_debt(v["target"], v["kind"], subject_of(Path(v["path"])))
                       for v in rec["violations"]})
        body = ("# Pre-existing admission-partition debt. MAY ONLY SHRINK -- never add a line.\n"
                "# Key: <target> <violation-kind>:<containing-directory-name>\n"
                "# See merlin/python/merlin/runtime/route_partition.py for what each kind means.\n"
                + "".join(k + "\n" for k in keys))
        ratchet_path.write_text(body, encoding="utf-8")
        print(f"wrote {len(keys)} ratchet entr(ies) to {ratchet_path}")
        return 0

    if a.json:
        print(json.dumps(rec, indent=2))
        return 0

    _, unratcheted = report(rec, load_ratchet(ratchet_path))
    if unratcheted and a.fail_on_violation:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
