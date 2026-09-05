#!/usr/bin/env python3
"""Recover the BEST-MEASURED compiler from a tuning trial, and package it for someone else to use.

WHY THIS EXISTS. The stage seals the agent's LAST tree, not its best one. That is deliberate -- the
stage grades and promotes nothing -- and it is documented only as a line of prompt text asking the
agent to keep the best itself, which nothing enforces. So a trial whose final edit regressed ships a
compiler slower than one it had already measured, and the sealed record says nothing about it.

Everything needed to fix that offline is already recorded. Each host-owned measurement writes a
redacted feedback document naming the `candidate_sha256` it measured and the per-member cycles it
got, and each measurement now runs against a SNAPSHOT of the candidate taken at the start of that
call. So the exact bytes behind every measured number are still on disk, and the best one can be
identified and extracted by digest rather than reconstructed by guesswork.

WHAT "BEST" MEANS HERE, and what it deliberately refuses to do:

* Only cells the CERTIFIED timing engine measured and marked comparable are totalled. A cell the
  sweep did not pay for contributes nothing, and is not read as a zero.
* Totals are only ever compared across measurements that covered the SAME member set. The sweep may
  stop early once a candidate is already losing, so a short sweep's total is smaller for a reason
  that has nothing to do with being faster. Comparing those two numbers would systematically crown
  the worst candidates. Documents outside the largest common member set are reported, never silently
  dropped and never quietly compared.
* A tie is reported as a tie. Two candidates at the same total are not ordered by recency.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "merlin" / "python"))

from merlin.benchharness import hash_tree                              # noqa: E402

#: Where a measurement's own copy of the candidate is kept, relative to its call directory.
SNAPSHOT_DIR = "_measured_candidate"
FEEDBACK_GLOB = "control/round_*/feedback/sha256/*.json"
SEALED = "sealed_candidate/submission"


def _measured_cells(document: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Cells this document actually paid for AND timed on the certified engine.

    A document written before the sweep could stop early carries no ``measured`` key at all, because
    every cell in it was paid for -- the field arrived with the early stop. So an ABSENT key means
    measured, while a present-and-false one means skipped. Reading absence as false silently drops
    every historical trial, which is how this returned "no measurement" on a trial holding ten of
    them; reading false as measured would count a cell nobody ran. Both are checked for explicitly.
    """
    out = []
    for c in (document.get("cells") or []):
        if not isinstance(c, Mapping):
            continue
        if "measured" in c and not c["measured"]:
            continue
        if (c.get("comparable") and isinstance(c.get("candidate_gsim_cycles"), int)
                and isinstance(c.get("baseline_gsim_cycles"), int)):
            out.append(c)
    return out


def _member_key(cells: "Sequence[Mapping[str, Any]]") -> tuple[str, ...]:
    return tuple(sorted(f"{c.get('family')}/{c.get('capsule')}" for c in cells))


def read_measurements(stage_dir: Path) -> list[dict[str, Any]]:
    """Every measurement in a trial, with its digest, its totals and the bytes it ran."""
    out: list[dict[str, Any]] = []
    for path in sorted(stage_dir.glob(FEEDBACK_GLOB)):
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        cells = _measured_cells(document)
        if not cells:
            continue
        call = document.get("invocation")
        round_index = document.get("round")
        snapshot = None
        if isinstance(call, int) and isinstance(round_index, int):
            candidate = (stage_dir / "_development_feedback" / f"round_{round_index:02d}"
                         / f"call_{call:03d}" / SNAPSHOT_DIR)
            snapshot = candidate if candidate.is_dir() else None
        out.append({
            "document": str(path),
            "round": round_index, "call": call,
            "candidate_sha256": document.get("candidate_sha256"),
            "members": _member_key(cells),
            "n_members": len(cells),
            "baseline_total_cycles": sum(int(c["baseline_gsim_cycles"]) for c in cells),
            "candidate_total_cycles": sum(int(c["candidate_gsim_cycles"]) for c in cells),
            "snapshot": str(snapshot) if snapshot else None,
        })
    return out


def choose_best(measurements: "Sequence[Mapping[str, Any]]") -> dict[str, Any]:
    """The lowest total among measurements that covered the SAME members, with the rest reported."""
    if not measurements:
        return {"status": "no_measurement",
                "reason": "the trial recorded no measurement with a comparable certified cell"}
    by_members: dict[tuple[str, ...], list[Mapping[str, Any]]] = {}
    for row in measurements:
        by_members.setdefault(row["members"], []).append(row)
    # The widest sweep that was run more than trivially is the comparison set; ties on width are
    # broken by how many measurements share it, because that is the set the search actually used.
    cohort_key = max(by_members, key=lambda k: (len(k), len(by_members[k])))
    cohort = by_members[cohort_key]
    excluded = [
        {"call": r["call"], "n_members": r["n_members"],
         "candidate_total_cycles": r["candidate_total_cycles"],
         "why": ("covered a different member set, so its total is not comparable with the cohort's")}
        for k, rows in by_members.items() if k != cohort_key for r in rows]

    lowest = min(r["candidate_total_cycles"] for r in cohort)
    winners = [r for r in cohort if r["candidate_total_cycles"] == lowest]
    baseline = cohort[0]["baseline_total_cycles"]
    return {
        "status": "tie" if len(winners) > 1 else "decided",
        "cohort_members": list(cohort_key),
        "cohort_size": len(cohort),
        "baseline_total_cycles": baseline,
        "best_total_cycles": lowest,
        "speedup_vs_baseline": (baseline / lowest) if lowest else None,
        "winners": [dict(r, members=list(r["members"])) for r in winners],
        "excluded_from_comparison": excluded,
        "all": [{"call": r["call"], "candidate_total_cycles": r["candidate_total_cycles"],
                 "n_members": r["n_members"], "candidate_sha256": r["candidate_sha256"]}
                for r in cohort],
    }


def main(argv: "Sequence[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage-dir", required=True, type=Path,
                    help="a trial stage directory (…/agent_stages/<experiment>__trial_NN)")
    ap.add_argument("--out", type=Path, default=None,
                    help="directory to copy the winning tree into; omit to only report")
    ap.add_argument("--json", type=Path, default=None, help="write the full report here")
    args = ap.parse_args(argv)

    stage_dir = args.stage_dir.resolve()
    if not stage_dir.is_dir():
        print(f"no such stage directory: {stage_dir}")
        return 2
    measurements = read_measurements(stage_dir)
    verdict = choose_best(measurements)
    sealed = stage_dir / SEALED
    verdict["sealed_candidate"] = str(sealed) if sealed.is_dir() else None
    verdict["sealed_sha256"] = str(hash_tree(sealed)["sha256"]) if sealed.is_dir() else None

    print(f"measurements with certified comparable cells : {len(measurements)}")
    if verdict["status"] == "no_measurement":
        print(f"  {verdict['reason']}")
        return 1
    print(f"comparison cohort                            : {verdict['cohort_size']} measurement(s) "
          f"over {len(verdict['cohort_members'])} member(s)")
    for row in verdict["all"]:
        mark = "  <- best" if row["candidate_total_cycles"] == verdict["best_total_cycles"] else ""
        print(f"  call {row['call']:>3}  {row['candidate_total_cycles']:>9} cycles{mark}")
    if verdict["excluded_from_comparison"]:
        print(f"  ({len(verdict['excluded_from_comparison'])} measurement(s) covered a different "
              f"member set and are not comparable)")
    print(f"baseline total                               : {verdict['baseline_total_cycles']}")
    print(f"best total                                   : {verdict['best_total_cycles']} "
          f"({verdict['speedup_vs_baseline']:.4f}x)" if verdict["speedup_vs_baseline"] else "")

    winner = verdict["winners"][0]
    if verdict["status"] == "tie":
        print(f"NOTE: {len(verdict['winners'])} candidates tie at the best total; reporting the "
              f"first and NOT ordering them by recency")

    # DOES THE SEALED TREE ACTUALLY CARRY THE BEST RESULT? This is the question this script exists
    # for, and it is answered by comparing digests rather than by trusting the prompt's request that
    # the agent keep its best.
    if verdict["sealed_sha256"]:
        sealed_rows = [r for r in measurements
                       if r.get("candidate_sha256") == verdict["sealed_sha256"]]
        verdict["sealed_was_measured"] = bool(sealed_rows)
        verdict["sealed_total_cycles"] = (
            min(r["candidate_total_cycles"] for r in sealed_rows) if sealed_rows else None)
        # ANY winner, not the first one. With a tie the sealed tree may be a different winner than
        # the one reported above, and comparing only against that one says "NO" beside a 0.00% gap.
        same = verdict["sealed_sha256"] in {w.get("candidate_sha256") for w in verdict["winners"]}
        verdict["sealed_is_best"] = same
        print(f"sealed tree was measured at all               : "
              f"{'YES' if sealed_rows else 'NO -- its speed is unknown'}")
        print(f"sealed tree is the best-measured one          : {'YES' if same else 'NO'}")
        if sealed_rows and not same:
            gap = verdict["sealed_total_cycles"] - verdict["best_total_cycles"]
            verdict["sealed_cycles_behind_best"] = gap
            print(f"  the sealed tree measured {verdict['sealed_total_cycles']} cycles, "
                  f"{gap} more than the best ({100.0 * gap / verdict['best_total_cycles']:.2f}% "
                  f"slower); shipping it discards that much of the search")
        elif not sealed_rows:
            print("  the sealed tree is an edit made AFTER the last measurement, so nothing "
                  "measured it; ship the verified snapshot instead")
    if winner.get("snapshot"):
        actual = str(hash_tree(Path(winner["snapshot"]))["sha256"])
        verdict["winner_snapshot_sha256"] = actual
        verdict["winner_snapshot_verified"] = actual == winner.get("candidate_sha256")
        print(f"winning snapshot                             : {winner['snapshot']}")
        print(f"  digest matches the measured candidate       : "
              f"{'YES' if verdict['winner_snapshot_verified'] else 'NO'}")
    else:
        verdict["winner_snapshot_sha256"] = None
        verdict["winner_snapshot_verified"] = False
        print("winning snapshot                             : NOT ON DISK -- this trial predates "
              "per-measurement snapshots, so the winning bytes cannot be recovered exactly")

    if args.out is not None:
        source = winner.get("snapshot")
        if not source or not verdict.get("winner_snapshot_verified"):
            print("refusing to export: the winning bytes are not on disk and verified")
            return 1
        destination = args.out.resolve()
        if destination.exists():
            print(f"refusing to export: {destination} already exists")
            return 1
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, destination)
        (destination.parent / f"{destination.name}.provenance.json").write_text(
            json.dumps({"extracted_from": str(stage_dir), **verdict}, indent=1), encoding="utf-8")
        print(f"exported the best-measured compiler to        : {destination}")

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(verdict, indent=1), encoding="utf-8")
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
