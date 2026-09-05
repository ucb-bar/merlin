#!/usr/bin/env python3
"""No TRACKED artifact may name a held-out capsule, or embed a local absolute path.

A held-out capsule's NAME is an answer key. Knowing which shapes are graded privately is most of the
advantage the holdout exists to deny -- an agent that knows the hidden set contains a strided padded
convolution can prepare for exactly that and call it generalization. Masking the golden VALUES from the
running agent (which the sandbox does) is a different and weaker guarantee than not publishing the
names.

WHY THIS GATE EXISTS SEPARATELY FROM ITS THREE SIBLINGS. ``check_no_answer_keys`` asks whether golden
surfaces are tracked; ``check_holdout_disjointness`` asks whether the hidden set overlaps the public one;
``verify_no_cheat`` asks whether the GRANTED tree leaks names into a running experiment. None of them
asks the question here: does any tracked file, anywhere, contain a holdout name? Two separate leaks were
found on one day that all three missed, in two different subsystems:

* ``CostFit.to_dict()`` serialized the run file each cost sample came from, and the conformance spec
  embeds that dict twice -- publishing 10 holdout names, plus 60 more through the per-class fits, as
  absolute ``/scratch`` paths, into a file every arm reads.
* a performance rate table harvested from the same certification runs published 8 holdout names and 497
  absolute paths, independently.

THE GENERALIZATION IS THE POINT: **anything harvested from certification runs republishes their
provenance**, because those runs include the grading passes over the hidden set. That is a class of bug,
not two incidents, and it will recur in the next artifact that learns to cite its sources. A gate that
names the class catches the third one before it ships.

THE FIX IS WITHHOLDING, NOT SANITIZING. Both repairs kept the auditable COUNTS (how many samples, over
what range, at what fit quality) and the refusal REASONS, and dropped only the identity- and
path-bearing fields, behind an explicit opt-in for a local diagnostic. A sanitized path is still a path
somebody will one day un-sanitize; a withheld one cannot leak.

WHAT IS SCANNED: files git reports as TRACKED. Untracked build output is not published and is covered by
``check_bundled_data`` instead. Binary files are skipped by content sniff, never by extension.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "merlin" / "python"))

from merlin.common.paths import merlin_dir  # noqa: E402

#: Path roots whose OWN business is the holdout, so naming one there is the point rather than a leak.
#: Each entry needs a reason: an allowlist without one is how a gate becomes decorative.
_ALLOWED: dict[str, str] = {
    "merlin/contract/capsules/hidden": "the hidden corpus itself; its own capsules name themselves",
    # RATCHETS RECORD DEBT BY NAME, DELIBERATELY. A ratchet whose entries were anonymised could not be
    # checked against the thing it ratchets, which is the whole mechanism; `holdout_disjointness_ratchet`
    # has always worked this way. They are tracked, reviewed, and may only shrink.
    "build_tools/scripts/holdout_disjointness_ratchet.txt":
        "the disjointness ratchet records the pairs it is ratcheting, by name, on purpose",
    "build_tools/scripts/cert_affordability_ratchet.txt":
        "an affordability ratchet names the capsules whose cost is accepted debt",
    "build_tools/scripts/mesh_assertion_ratchet.txt":
        "a mesh-assertion ratchet names the capsules whose assertion gap is accepted debt",
    "build_tools/generalization_debt.txt":
        "the generalization debt list names what it is holding open",
    # A TEST ABOUT THE LEAK HAS TO BE ABLE TO SPELL ONE. These construct a holdout name as a fixture or
    # assert that some artifact does NOT contain it; forbidding the string here would forbid testing the
    # property. Tests are not published artifacts.
    "merlin/tests":
        "a test whose subject is the holdout must be able to name one as a fixture",
    "build_tools/scripts/check_no_holdout_names.py":
        "this gate's own docstring describes the leak it was written for",
}


def _holdout_stores() -> dict[str, str]:
    """Every hidden corpus on disk, DISCOVERED rather than listed.

    A target that owns its own subtree keeps its holdouts at ``<target>/hidden``, and naming a holdout
    there is the point rather than a leak. Spelling those targets out here would hardcode a target name
    in a shared scan root, which this repo forbids and gates -- and worse, it would mean a target added
    tomorrow either trips this gate on its own hidden corpus or gets quietly forgotten. Reading the
    directories is the same answer with neither failure mode.
    """
    root = merlin_dir() / "contract" / "capsules"
    found = dict(_ALLOWED)
    for hidden in sorted(root.glob("*/hidden")):
        if not hidden.is_dir():
            continue
        rel = hidden.relative_to(merlin_dir().parent).as_posix()
        found.setdefault(rel, "a target's own hidden corpus; its capsules name themselves")
    return found


def _tracked_files() -> list[Path]:
    out = subprocess.run(["git", "ls-files", "-z"], cwd=REPO, capture_output=True, text=True,
                         check=False)
    if out.returncode != 0:
        raise SystemExit(f"[FAIL] no-holdout-names: `git ls-files` failed: {out.stderr[-300:]}")
    return [REPO / p for p in out.stdout.split("\0") if p]


def holdout_names() -> set[str]:
    """Every capsule whose label is not public/dev, read from the corpus itself.

    Derived, never listed: a gate carrying its own copy of the hidden set would be the leak it exists to
    prevent, and would go stale the first time somebody adds a capsule.
    """
    import yaml

    names: set[str] = set()
    root = merlin_dir() / "contract" / "capsules"
    for cy in root.rglob("capsule.yaml"):
        try:
            doc = yaml.safe_load(cy.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError:
            continue
        if isinstance(doc, dict) and str(doc.get("label") or "") not in ("public", "dev"):
            names.add(str(doc.get("name") or cy.parent.name))
    return {n for n in names if n}


def _ratchet() -> dict[str, str]:
    """Pre-existing offenders, accepted as debt. MAY ONLY SHRINK.

    Distinct from ``_ALLOWED`` and the distinction is the point: an allowlist entry says "naming a
    holdout here is CORRECT" (a ratchet file records what it ratchets; a test needs a fixture); a
    ratchet entry says "this is a leak we have not fixed yet". Collapsing them would let real debt hide
    behind a word that means the opposite.
    """
    path = REPO / "build_tools" / "scripts" / "holdout_name_ratchet.txt"
    out: dict[str, str] = {}
    if not path.is_file():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        rel, _, why = line.partition("#")
        out[rel.strip()] = why.strip()
    return out


def _allowed(rel: str) -> str | None:
    for prefix, why in _ALLOWED.items():
        if rel == prefix or rel.startswith(prefix + "/"):
            return why
    return None


def _is_text(path: Path) -> bool:
    """Content sniff, not an extension list: an extension list is a way to miss the next format."""
    try:
        chunk = path.open("rb").read(4096)
    except OSError:
        return False
    return b"\0" not in chunk


def _local_absolute(text: str) -> bool:
    """A path under a home or scratch root. Parsed structurally -- the repo forbids regex."""
    for token in text.replace('"', " ").replace("'", " ").split():
        if not token.startswith("/"):
            continue
        parts = token.split("/")
        if len(parts) > 3 and parts[1] in ("home", "scratch", "scratch2", "Users"):
            return True
    return False


def scan(*, limit: int = 20) -> dict:
    names = holdout_names()
    if not names:
        # UNKNOWN, never a pass. A worktree has no hidden corpus, so an empty set means the question
        # could not be asked -- and a gate that cannot fail must say so rather than print [ ok].
        return {"status": "undeterminable", "n_holdouts": 0, "leaks": [], "abs_paths": [],
                "detail": ("no held-out capsule is readable in this checkout (a worktree carries no "
                           "hidden corpus), so this gate could not be evaluated")}

    ratchet = _ratchet()
    leaks, ratcheted, abs_paths, skipped = [], [], [], 0
    for path in _tracked_files():
        rel = path.relative_to(REPO).as_posix()
        if _allowed(rel) or not path.is_file():
            continue
        if not _is_text(path):
            skipped += 1
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="strict")
        except (OSError, UnicodeDecodeError):
            skipped += 1
            continue
        hits = sorted(n for n in names if n in text)
        if hits:
            row = {"file": rel, "names": hits[:6], "n_names": len(hits)}
            (ratcheted if rel in ratchet else leaks).append(row)
        if _local_absolute(text):
            abs_paths.append(rel)
    return {"status": "ok", "n_holdouts": len(names), "n_tracked_scanned": len(_tracked_files()),
            "n_binary_skipped": skipped, "leaks": leaks[:limit], "n_leaks": len(leaks),
            "ratcheted": [r["file"] for r in ratcheted], "n_ratcheted": len(ratcheted),
            "ratchet_declared": len(ratchet),
            "abs_paths": abs_paths[:limit], "n_abs_paths": len(abs_paths)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--strict-paths", action="store_true",
                    help="also FAIL on a tracked file embedding a local absolute path. Off by default: "
                         "that is a real but far broader pre-existing problem (default paths in "
                         "scripts, pins, docs) and folding it in would make this gate fail for a "
                         "reason other than the one it is named for")
    a = ap.parse_args(argv)

    rep = scan(limit=a.limit)
    if a.json:
        print(json.dumps(rep, indent=1))

    if rep.get("n_ratcheted") is not None and not a.json:
        stale = rep["ratchet_declared"] - rep["n_ratcheted"]
        print(f"   accepted as pre-existing debt: {rep['n_ratcheted']} "
              f"(ratchet declares {rep['ratchet_declared']}"
              + (f"; {stale} entr(y/ies) no longer leak and MUST be removed" if stale > 0 else "")
              + ")")

    if rep["status"] == "undeterminable":
        print(f"[note] no-holdout-names: UNDETERMINABLE — {rep['detail']}")
        return 0

    if not a.json:
        print(f"no-holdout-names: {rep['n_holdouts']} held-out capsule name(s), "
              f"{rep['n_tracked_scanned']} tracked file(s) scanned "
              f"({rep['n_binary_skipped']} binary skipped)")

    rc = 0
    if rep["leaks"]:
        print(f"\n[FAIL] no-holdout-names: {rep['n_leaks']} tracked file(s) NAME a held-out capsule. "
              f"A holdout's name is an answer key -- knowing which shapes are graded privately is most "
              f"of the advantage the holdout exists to deny:")
        for row in rep["leaks"]:
            print(f"  - {row['file']}: {row['names']}"
                  + (f" (+{row['n_names'] - len(row['names'])} more)"
                     if row["n_names"] > len(row["names"]) else ""))
        print("  Fix by WITHHOLDING, not sanitizing: keep the auditable counts and the refusal reasons, "
              "drop the identity- and path-bearing fields behind an explicit opt-in.")
        rc = 1

    if rep["abs_paths"]:
        head = "[FAIL]" if a.strict_paths else "[note]"
        print(f"\n{head} no-holdout-names: {rep['n_abs_paths']} tracked file(s) embed a LOCAL ABSOLUTE "
              f"path. These files are published; a path under someone's home or scratch root is neither "
              f"portable nor reviewable, and it is the carrier the holdout names travelled on:")
        for rel in rep["abs_paths"]:
            print(f"  - {rel}")
        if a.strict_paths:
            rc = 1

    if rc == 0:
        print("\n[  ok] no-holdout-names: no tracked artifact names a held-out capsule.")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
