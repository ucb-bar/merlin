#!/usr/bin/env python3
"""Grade a Merlin FireSim UART independently of the queue lifecycle result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re


def grade(text: str, *, expected_logits: int, expected_top1: int) -> dict[str, object]:
    result_matches = re.findall(
        r"MERLIN_RESULT logits_checked=(\d+) bad=(\d+) nonfinite=(\d+) "
        r"top1=(\d+) expected_top1=(\d+)", text)
    profile_matches = re.findall(r"MERLIN_PROFILE measured end rc=(\d+)", text)
    cycle_matches = re.findall(r"MERLIN_METRIC cycles=(\d+)", text)
    reasons: list[str] = []
    if len(result_matches) != 1:
        reasons.append(f"expected one result record, found {len(result_matches)}")
        result = None
    else:
        checked, bad, nonfinite, top1, declared_top1 = map(int, result_matches[0])
        result = {
            "logits_checked": checked,
            "bad": bad,
            "nonfinite": nonfinite,
            "top1": top1,
            "declared_expected_top1": declared_top1,
        }
        if checked != expected_logits:
            reasons.append(f"checked {checked} logits, expected {expected_logits}")
        if bad != 0 or nonfinite != 0:
            reasons.append(f"numeric mismatch: bad={bad} nonfinite={nonfinite}")
        if top1 != expected_top1 or declared_top1 != expected_top1:
            reasons.append(
                f"top1 mismatch: got={top1} declared_expected={declared_top1} expected={expected_top1}")
    if profile_matches != ["0"]:
        reasons.append(f"measured profile did not end once with rc=0: {profile_matches}")
    if len(cycle_matches) != 1 or int(cycle_matches[0]) <= 0:
        reasons.append(f"expected one positive cycle metric, found {cycle_matches}")
    if "*** FAILED ***" in text or "FAIL:" in text:
        reasons.append("UART contains an explicit payload failure marker")
    return {
        "schema": "merlin_firesim_uart_grade_v1",
        "ok": not reasons,
        "cycles": int(cycle_matches[0]) if len(cycle_matches) == 1 else None,
        "result": result,
        "reasons": reasons,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("uartlog", type=Path)
    parser.add_argument("--expected-logits", type=int, default=1000)
    parser.add_argument("--expected-top1", type=int, default=258)
    args = parser.parse_args()
    report = grade(
        args.uartlog.read_text(errors="replace"),
        expected_logits=args.expected_logits,
        expected_top1=args.expected_top1,
    )
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
