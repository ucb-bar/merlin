"""Explicit installed launch edge for the measured-claims managed envelope.

Envelope options precede ``--``; coordinator options follow it unchanged. Supply
the installed ``chia_envelope.py`` path as the coordinator's ``--chia-wrapper``:
it is the exact receipt authority, not this argument-parsing CLI module. Discover
that path with ``merlin.common.paths.module_source_path`` before launching.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from merlin.common.paths import module_source_path
from merlin_experiments.phase2 import chia_envelope as ENVELOPE


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--driver-python", type=Path, required=True)
    parser.add_argument("--cwd", type=Path, required=True)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--target", required=True)
    args, remaining = parser.parse_known_args(argv)
    try:
        context = ENVELOPE.EnvelopeContext(
            python=args.driver_python,
            cwd=args.cwd,
            wrapper_source=module_source_path(ENVELOPE.__name__),
            coordinator_prefix=("-m", "merlin_experiments.phase2.checkpoint_cli"),
            suite=args.suite,
            target=args.target,
        )
    except ValueError as exc:
        parser.error(str(exc))
    return ENVELOPE.main(remaining, context=context)


if __name__ == "__main__":
    raise SystemExit(main())
