#!/usr/bin/env python3
"""Legacy layout selection for the installed managed measured-claims envelope."""

from __future__ import annotations

import sys
from pathlib import Path

from merlin_experiments.phase2 import chia_envelope as ENVELOPE
from merlin_experiments.phase2.chia_envelope import (  # noqa: F401 — retained inspection API
    _canonical,
    _content_addressed_receipt,
    run_coordinator,
    validate_assigned_resources,
)

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]


def _target(arguments: list[str]) -> str:
    try:
        descriptor = arguments[arguments.index("--descriptor") + 1]
    except (ValueError, IndexError):
        return "gemmini"
    import yaml

    document = yaml.safe_load(Path(descriptor).read_text(encoding="utf-8")) or {}
    return str(document.get("target") or "gemmini")


def _context(arguments: list[str]) -> ENVELOPE.EnvelopeContext:
    from merlin.benchharness.chia_bridge import driver_python

    return ENVELOPE.EnvelopeContext(
        python=Path(driver_python()),
        cwd=REPO,
        wrapper_source=Path(__file__).resolve(),
        coordinator_prefix=(str(HERE / "run_agentic_perf_experiment.py"),),
        suite="gemmini-perf-bench",
        target=_target(arguments),
    )


def plan_command(coordinator_args: list[str], *, stub_seconds: float = 0.0) -> list[str]:
    return ENVELOPE.plan_command(coordinator_args, context=_context([]), stub_seconds=stub_seconds)


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    envelope_arguments = arguments[: arguments.index("--")] if "--" in arguments else arguments
    context_arguments = [] if "--dry-run" in envelope_arguments else arguments
    return ENVELOPE.main(arguments, context=_context(context_arguments))


if __name__ == "__main__":
    raise SystemExit(main())
