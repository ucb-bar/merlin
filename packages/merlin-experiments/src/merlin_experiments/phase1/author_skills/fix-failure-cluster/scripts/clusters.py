#!/usr/bin/env python3
"""Print the grader's failure clusters compactly, with reproduce commands and a helper brief.

Reads ``failure_clusters`` from ``qa/verdict.json`` when that verdict is current, else from
``qa/latest_completed_grade.json`` (the newest completed result per test case). Read-only; stdlib only.

  python clusters.py                 # every listed cluster, one block each
  python clusters.py --cluster 2     # one cluster in full: detail, members, per-member rows, commands
  python clusters.py --brief 2       # a self-contained helper brief for cluster 2
  python clusters.py --json          # the selected section as JSON
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

VERDICT = Path("qa") / "verdict.json"
COMPLETED = Path("qa") / "latest_completed_grade.json"
KEY = "failure_clusters"
MEMBERS_SHOWN = 8


def _load(path: Path):
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return document if isinstance(document, dict) else None


def select(workspace: Path, choice: str) -> tuple[Path | None, dict | None]:
    """The feedback file to read and its document; ``auto`` prefers a current live verdict."""
    candidates = {"verdict": [VERDICT], "completed": [COMPLETED], "auto": [VERDICT, COMPLETED]}[choice]
    for relative in candidates:
        document = _load(workspace / relative)
        if document is None:
            continue
        if choice == "auto" and relative == VERDICT and not (document.get("snapshot_current") and KEY in document):
            continue
        return relative, document
    return None, None


def _clip(value, limit: int) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _rows(document: dict) -> dict:
    return {row["capsule"]: row for row in document.get("per_capsule") or ()
            if isinstance(row, dict) and isinstance(row.get("capsule"), str)}


def _current_source(document: dict):
    counts = document.get("completed_rows") if isinstance(document.get("completed_rows"), dict) else {}
    return counts.get("current_source_sha256") or document.get("current_source_sha256")


def _graded_source(row: dict, document: dict):
    graded = row.get("graded") if isinstance(row.get("graded"), dict) else {}
    return graded.get("graded_source_sha256") or document.get("graded_source_sha256")


def members_command(cluster: dict) -> str:
    """The cluster's reproduce command with every member named instead of the one example."""
    words = str(cluster.get("reproduce") or "").split()
    members = ",".join(str(name) for name in cluster.get("test_cases") or ())
    if "--capsules" not in words or not members:
        return " ".join(words)
    at = words.index("--capsules")
    return " ".join([*words[: at + 1], members, *words[at + 2:]])


def _tiers(row: dict) -> str:
    tiers = row.get("tiers") if isinstance(row.get("tiers"), dict) else {}
    return ",".join(f"{name}={value.get('status') if isinstance(value, dict) else value}"
                    for name, value in sorted(tiers.items())) or "-"


def _header(relative: Path, document: dict, section: dict) -> list[str]:
    pointer = document.get("latest_completed_grade") if isinstance(document.get("latest_completed_grade"), dict) else {}
    return [f"source: {relative} graded_at={pointer.get('graded_at') or document.get('graded_at')} "
            f"graded_source={str(document.get('graded_source_sha256'))[:12]} "
            f"current_source={str(_current_source(document))[:12]} "
            f"snapshot_current={document.get('snapshot_current')}",
            f"{section.get('failing_test_cases')} failing test case(s) in {section.get('clusters_total')} "
            f"cluster(s){'; ' + str(section['clusters_not_listed']) + ' smaller not listed' if section.get('clusters_not_listed') else ''}"]


def _summary(index: int, cluster: dict, rows: dict, document: dict, detail: int) -> list[str]:
    names = [str(name) for name in cluster.get("test_cases") or ()]
    current = _current_source(document)
    on_current = sum(1 for name in names if current and _graded_source(rows.get(name) or {}, document) == current)
    shown = ", ".join(names[:MEMBERS_SHOWN]) + (f" (+{len(names) - MEMBERS_SHOWN} more)" if len(names) > MEMBERS_SHOWN else "")
    lines = [f"{index}. {cluster.get('count')} case(s) {cluster.get('failure_plane')}/{cluster.get('failure_category')}"
             f" [{on_current}/{len(names)} graded on the current source]",
             f"   detail: {_clip(cluster.get('example_failure_detail'), detail)}",
             f"   reproduce: {cluster.get('reproduce')}"]
    if cluster.get("inspect"):
        lines.append(f"   inspect: {cluster['inspect']}")
    lines.append(f"   members: {shown}")
    return lines


def _full(index: int, cluster: dict, rows: dict, document: dict) -> list[str]:
    lines = _summary(index, cluster, rows, document, 100_000)
    lines[-1] = "   members: " + ", ".join(str(name) for name in cluster.get("test_cases") or ())
    lines.append(f"   all members: {members_command(cluster)}")
    lines.append("   rows: name status tiers mismatch_count graded_at")
    for name in cluster.get("test_cases") or ():
        row = rows.get(name) or {}
        graded = row.get("graded") if isinstance(row.get("graded"), dict) else {}
        lines.append(f"     {name} {row.get('status')} {_tiers(row)} {row.get('mismatch_count', '-')} "
                     f"{graded.get('graded_at', '-')}")
    return lines


def brief(index: int, cluster: dict, relative: Path) -> str:
    """A self-contained brief a helper can start from with no other context."""
    names = ", ".join(str(name) for name in cluster.get("test_cases") or ())
    return "\n".join([
        f"GOAL: make the {cluster.get('count')} test case(s) of failure cluster {index} pass "
        f"({cluster.get('failure_plane')}/{cluster.get('failure_category')}) without breaking passing cases.",
        f"FAILURE (example {', '.join(cluster.get('examples') or [])}): "
        f"{_clip(cluster.get('example_failure_detail'), 1200)}",
        f"CASES: {names}",
        f"REPRODUCE: {cluster.get('reproduce')}",
        f"ALL CASES: {members_command(cluster)}",
        *([f"INSPECT: {cluster['inspect']}"] if cluster.get("inspect") else []),
        "OWNED FILES: (fill in: the compiler files this helper may edit; no other agent edits them)",
        "WORK IN: (fill in: submission/ or a variants/NAME/ copy, and pass --submission accordingly)",
        "SKILLS TO OPEN: fix-failure-cluster (steps 3-6); (add others the fix needs)",
        f"CONTEXT: clusters come from {relative}; re-run this script with --cluster {index} for every member's row.",
        "REPORT: root cause in one or two sentences, files changed, self-check request ids with pass counts "
        "for these cases and for any regression cases, and what is still failing.",
    ])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workspace", default=".", help="workspace root (default: the current directory)")
    parser.add_argument("--file", choices=("auto", "verdict", "completed"), default="auto",
                        help="auto (default): the live verdict when current, else the latest completed grade")
    parser.add_argument("--cluster", type=int, help="print one cluster (1-based) in full")
    parser.add_argument("--brief", type=int, metavar="N", help="print a helper brief for cluster N")
    parser.add_argument("--detail", type=int, default=240, help="characters of detail per cluster in the list")
    parser.add_argument("--json", action="store_true", help="print the selected cluster section as JSON")
    args = parser.parse_args(argv)

    relative, document = select(Path(args.workspace), args.file)
    if document is None:
        print("no readable verdict or completed grade yet (qa/verdict.json, qa/latest_completed_grade.json)")
        return 1
    section = document.get(KEY) if isinstance(document.get(KEY), dict) else None
    if section is None:
        print(f"{relative}: no {KEY} -- no test case with a failed result in that grade")
        return 0
    clusters = section.get("clusters") or []
    if args.json:
        print(json.dumps(section, indent=2))
        return 0
    chosen = args.cluster or args.brief
    if chosen is not None and not 1 <= chosen <= len(clusters):
        print(f"cluster {chosen} is not listed; {len(clusters)} cluster(s) are", file=sys.stderr)
        return 2
    if args.brief is not None:
        print(brief(args.brief, clusters[args.brief - 1], relative))
        return 0
    rows = _rows(document)
    lines = _header(relative, document, section)
    if args.cluster is not None:
        lines += _full(args.cluster, clusters[args.cluster - 1], rows, document)
    else:
        for index, cluster in enumerate(clusters, 1):
            lines += _summary(index, cluster, rows, document, args.detail)
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
