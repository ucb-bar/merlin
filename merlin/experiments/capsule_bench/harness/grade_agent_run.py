#!/usr/bin/env python3
"""Legacy CLI edge; implementation lives in phase1.feedback.formal."""

from merlin_experiments.phase1.feedback.formal import main


def _context():
    import _common

    return _common.CONTEXT


if __name__ == "__main__":
    raise SystemExit(main(context=_context))
