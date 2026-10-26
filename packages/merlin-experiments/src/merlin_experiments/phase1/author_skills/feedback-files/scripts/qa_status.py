#!/usr/bin/env python3
"""How fresh each feedback file is: age, size and the freshness fields it carries about itself.

Lists every file in ``qa/`` (and the newest self-check request) with its age and size. For a JSON file
small enough to parse quickly it prints the top-level fields that say which grade and which source
bytes it describes. Read-only; stdlib only.

  python qa_status.py                  # the workspace in the current directory
  python qa_status.py --max-parse-mb 16
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

#: Top-level fields that say what a feedback document describes and when it was written.
FIELDS = ("schema", "state", "stage", "graded_at", "updated_at", "snapshot_current", "graded_source_sha256",
          "current_source_sha256", "n_passed", "n_capsules", "n_completed", "all_pass", "certified_complete",
          "tiers_graded", "tiers_not_run")
#: Nested summaries worth one line each: key -> fields.
NESTED = {"latest_completed_grade": ("token", "graded_at", "snapshot_current"),
          "completed_rows": ("rows", "executed", "passed", "from_newest_grade", "from_current_source"),
          "failure_clusters": ("failing_test_cases", "clusters_total")}


def _age(seconds: float) -> str:
    seconds = max(0, int(seconds))
    if seconds < 120:
        return f"{seconds}s"
    if seconds < 7200:
        return f"{seconds // 60}m"
    return f"{seconds // 3600}h{(seconds % 3600) // 60:02d}m"


def _short(value) -> str:
    text = json.dumps(value) if not isinstance(value, str) else value
    return text[:16] if len(text) == 64 and all(ch in "0123456789abcdef" for ch in text) else text[:80]


def describe(path: Path, shown: str, now: float, max_parse: int) -> list[str]:
    stat = path.stat()
    lines = [f"{shown}  age={_age(now - stat.st_mtime)}  size={stat.st_size:,}B"]
    if path.suffix != ".json" or stat.st_size > max_parse:
        return lines
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return [*lines, "  (not readable as JSON right now; it may be mid-write)"]
    if not isinstance(document, dict):
        return lines
    fields = [f"{key}={_short(document[key])}" for key in FIELDS if key in document]
    if fields:
        lines.append("  " + " ".join(fields))
    for key, names in NESTED.items():
        nested = document.get(key)
        if isinstance(nested, dict):
            lines.append(f"  {key}: " + " ".join(f"{name}={_short(nested.get(name))}" for name in names
                                                 if name in nested))
    return lines


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workspace", default=".", help="workspace root (default: the current directory)")
    parser.add_argument("--max-parse-mb", type=float, default=8.0,
                        help="parse JSON files up to this size for their freshness fields (default 8)")
    args = parser.parse_args(argv)
    root = Path(args.workspace)
    qa = root / "qa"
    if not qa.is_dir():
        print("no qa/ directory yet: no grade has been published")
        return 1
    now, limit = time.time(), int(args.max_parse_mb * (1 << 20))
    lines = []
    for path in sorted(qa.iterdir()):
        if path.is_file() and not path.name.startswith("."):
            lines += describe(path, f"qa/{path.name}", now, limit)
    requests = root / "selfcheck_out" / "requests"
    if requests.is_dir():
        entries = [entry for entry in requests.iterdir() if entry.is_dir()]
        if entries:
            newest = max(entries, key=lambda entry: entry.stat().st_mtime)
            lines.append(f"selfcheck_out/requests: {len(entries)} request(s); newest {newest.name} "
                         f"age={_age(now - newest.stat().st_mtime)}")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
