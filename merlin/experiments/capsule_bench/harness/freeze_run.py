#!/usr/bin/env python3
"""Legacy CLI edge; implementation lives in phase1.feedback.freeze."""

from merlin_experiments.phase1.feedback.freeze import main


def _repo():
    import _common

    return _common.REPO


if __name__ == "__main__":
    raise SystemExit(main(repo=_repo))
