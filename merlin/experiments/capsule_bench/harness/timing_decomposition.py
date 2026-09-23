#!/usr/bin/env python3
"""Legacy CLI defaults for the canonical phase1.telemetry report command."""

from merlin_experiments.phase1.telemetry.report import main


def _context():
    import _common

    return _common.CONTEXT


if __name__ == "__main__":
    raise SystemExit(main(context=_context))
