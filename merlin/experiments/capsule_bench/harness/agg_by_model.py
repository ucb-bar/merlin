#!/usr/bin/env python3
"""Compatibility CLI for the analysis-owned Phase 1 by model report."""

from __future__ import annotations

import sys


def main(argv=None) -> int:
    try:
        from merlin.agentreport.phase1 import by_model as report
        from merlin.agentreport.phase1.runs import ReportInputs
    except ModuleNotFoundError as exc:
        if exc.name and exc.name.startswith("merlin.agentreport"):
            raise SystemExit("Install merlin-analysis to generate Phase 1 reports.") from exc
        raise
    args = list(sys.argv[1:] if argv is None else argv)
    # Explicit context belongs to the canonical parser, including missing-value
    # errors. Only legacy invocations initialize the ambient native descriptor.
    if "--help" in args or "-h" in args or any(arg.partition("=")[0] in {"--target", "--descriptor"} for arg in args):
        return report.main(args)
    import _common as C

    return report.main(args, defaults=ReportInputs(C.TARGET, C.RUNS, C.REPORTS))


if __name__ == "__main__":
    raise SystemExit(main())
