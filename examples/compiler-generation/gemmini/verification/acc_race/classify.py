#!/usr/bin/env python3
"""Read the accumulator load-order reproducers' consoles and say which ordering produced each error.

Both reproducers (``acc_race_repro.c``, ``vend_repro.c``) print one line per test and repetition:

    RACE cand07 rep=0 bad=32 lhs_only=32 rhs_only=0 other=0 first=52016 bins512=0,...,16
    VEND cold rows=98 bad=32 lhs_only=32 rhs_only=0 other=0 first=2560 bins=0,...,32

Every wrong element was already classified on the device as ``lhs_only`` (the first, overwriting
operand survived alone: the race's signature), ``rhs_only`` or ``other``. This script parses those
lines structurally and rolls them up per test, so a run answers two questions at once: did the test
fail, and was every failure the race's signature rather than some other wrong answer.

For an elementwise two-operand capsule, whose console prints only the result tensor, the same
classification is done here on the host from the operands and the device result
(``classify_elements``).

    classify.py <console.log> [--json]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path

#: The line kinds the reproducers print, and the fields every one of them must carry.
KINDS = ("RACE", "VEND")
COUNTS = ("bad", "lhs_only", "rhs_only", "other")


def parse_line(line: str) -> dict | None:
    """One reproducer line as a record, or None for any other console line.

    A line that starts with a reproducer kind but lacks a count is refused rather than skipped: a
    truncated console must not read as a clean repetition.
    """
    fields = line.split()
    if len(fields) < 2 or fields[0] not in KINDS:
        return None
    record: dict = {"kind": fields[0], "test": fields[1]}
    for field in fields[2:]:
        key, sep, value = field.partition("=")
        if not sep:
            raise ValueError(f"unparseable field {field!r} in {line!r}")
        if key in ("bins512", "bins"):
            record["bins"] = [int(v) for v in value.split(",") if v]
        else:
            record[key] = int(value)
    missing = [name for name in COUNTS if name not in record]
    if missing:
        raise ValueError(f"reproducer line lacks {missing}: {line!r}")
    if record["bad"] != record["lhs_only"] + record["rhs_only"] + record["other"]:
        raise ValueError(f"reproducer line counts do not add up: {line!r}")
    return record


def summarize(lines: Iterable[str]) -> dict:
    """Per test: repetitions, repetitions with any wrong element, and the classes of those elements."""
    tests: dict[str, dict] = {}
    for line in lines:
        record = parse_line(line.strip())
        if record is None:
            continue
        name = f"{record['kind']} {record['test']}"
        row = tests.setdefault(name, {"reps": 0, "failing_reps": 0, **{k: 0 for k in COUNTS}})
        row["reps"] += 1
        row["failing_reps"] += record["bad"] > 0
        for key in COUNTS:
            row[key] += record[key]
    for row in tests.values():
        # The race's signature: something failed and every wrong element is the first operand alone.
        row["race_signature"] = row["bad"] > 0 and row["bad"] == row["lhs_only"]
    return tests


def classify_elements(lhs: Sequence[int], rhs: Sequence[int], device: Sequence[int]) -> dict[str, int]:
    """Classify each element of a unit-scale two-operand sum by which operand survived."""
    if not len(lhs) == len(rhs) == len(device):
        raise ValueError(f"operand and result lengths differ: {len(lhs)}, {len(rhs)}, {len(device)}")
    counts = {"ok": 0, "lhs_only": 0, "rhs_only": 0, "other": 0}
    for a, b, d in zip(lhs, rhs, device):
        if d == a + b:
            counts["ok"] += 1
        elif d == a:
            counts["lhs_only"] += 1
        elif d == b:
            counts["rhs_only"] += 1
        else:
            counts["other"] += 1
    return counts


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("console", type=Path)
    parser.add_argument("--json", action="store_true", help="print the summary as JSON")
    args = parser.parse_args(argv)
    summary = summarize(args.console.read_text(errors="replace").splitlines())
    if not summary:
        print(f"no reproducer lines in {args.console}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(summary, indent=1, sort_keys=True))
    else:
        for name, row in sorted(summary.items()):
            verdict = "race signature" if row["race_signature"] else ("clean" if not row["bad"] else "OTHER ERROR")
            print(
                f"{name:16s} {row['failing_reps']}/{row['reps']} reps failing  bad={row['bad']} "
                f"lhs_only={row['lhs_only']} rhs_only={row['rhs_only']} other={row['other']}  {verdict}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
