#!/usr/bin/env python3
"""Gate: the exchange-rate window plan still isolates the term each of its arms claims to measure.

An exchange rate is a slope, and a slope is only a rate if exactly one thing moved between the
points. That property is easy to state, easy to believe, and invisible in a plan written as prose:
the sketch this plan replaced read as three isolated arms, and one of them was not isolated, one was
aimed at a model that prices the intended change at exactly zero, and the third varied two terms at
once. None of that shows until the numbers come back wrong, by which point a campaign has been spent.

:func:`merlin.perf.exchange_rate_windows.check_plan` recomputes each window's model terms from its
declared tiling and asserts, per arm, that the arm's own term moves and every term it claims to pin
is constant. That check is only worth having if something RUNS it before the plan is measured with,
so this gate runs it over the declared plan -- an edit that breaks an arm's isolation fails here
rather than silently measuring two things at once.

It is pure data: no target, no simulator, no network. Nothing is written.

Usage::

    python build_tools/scripts/check_exchange_rate_windows.py           # exit 1 on a broken plan
    python build_tools/scripts/check_exchange_rate_windows.py --report  # print the computed terms
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "merlin" / "python"))

from merlin.perf.exchange_rate_windows import WindowPlanError, check_plan  # noqa: E402


def main(argv: "list[str] | None" = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        action="store_true",
        help="print the per-arm term table the verdict was reached on, as JSON",
    )
    args = parser.parse_args(argv)
    try:
        report = check_plan()
    except WindowPlanError as broken:
        print(f"[FAIL] exchange-rate windows: {broken}")
        return 1
    except FileNotFoundError:
        # A repo that ships no plan has nothing to check. That is not a pass claimed about a plan;
        # it is the absence of one, and it is said rather than counted as green.
        print("[ ok ] exchange-rate windows: no plan is declared in this checkout")
        return 0
    if args.report:
        print(json.dumps(report, indent=2, default=str))
    arms = report.get("arms") or []
    print(
        f"[  ok] exchange-rate windows: {len(arms)} arm(s) isolate what they claim "
        f"(design {report.get('design')!r}, fit order {report.get('fit_order')})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
