"""Capsule evaluation: identity."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

CONTRACT_VERSION = "0.1"


_MODEL_NUMERIC_UNSET = object()


_SUITE_MODEL_PIN = object()


def concurrency_stamp(workers: int | None = None) -> dict:
    """What else the host was doing at the moment of measurement.

    Taken at measurement time, not at suite start: load average moves, and a stamp that describes the
    machine an hour ago describes nothing. ``workers`` is the fan-out this suite was told to use;
    ``None`` means the caller did not state one (a direct :func:`run_capsule` call), and it stays
    ``None`` rather than being guessed at 1 -- other processes share this host.

    Why this exists at all: cycle counts are concurrency-invariant and wall times are not, by a
    measured factor of 6.3x. Every wall number in a ``timing`` block is uninterpretable without it,
    and 22,845 blocks already on disk have to be read as "concurrency unrecorded" because it did not
    exist when they were written. They are NOT retro-labelled: the concurrency they ran at is gone.
    """
    try:
        load = round(os.getloadavg()[0], 2)
    except (OSError, AttributeError):  # no load average on this platform
        load = None
    return {
        "workers": None if workers is None else int(workers),
        "nproc": os.cpu_count(),
        "load_avg": load,
        "serial": None if workers is None else bool(int(workers) <= 1),
        "sampled": "at the adapter's return, so the load average covers the measurement itself",
    }


def _graded_program(res: Any, package_dir: str | Path) -> str | None:
    """WHICH PROGRAM produced this tier's cycle count, or ``None`` when it cannot be established.

    Only an adapter that BUILT AND RAN AN ELF from the package under grade may claim the submission,
    and ``res["elf"]`` is that evidence. A tier graded on a harness fixture -- a block-scaled capsule
    runs the harness's own reference kernel, not the submission -- returns ``None`` and stays
    UNATTRIBUTED. Defaulting the other way is what let a fixture's pass travel as a submission's, in
    the recorded run that scored 40/40 with nine of the forty measuring the fixture.

    The path is RESOLVED to match :func:`submission_identity`, which records the resolved package on
    every tier record. Comparing an unresolved string against a resolved one makes
    ``comparand.attribute`` return OTHER_PROGRAM -- a FALSE claim about a different program, strictly
    worse than the UNATTRIBUTED it replaces.
    """
    if not isinstance(res, dict) or not res.get("elf"):
        return None
    pkg = Path(package_dir)
    return str(pkg.resolve()) if pkg.exists() else str(pkg)


def submission_identity(package_dir: str | Path, *, run_id: str | None = None) -> dict:
    """WHICH SUBMISSION a measurement belongs to, stated explicitly.

    The harvest layer has had to infer this from where a file sits (the parent of the nearest
    enclosing ``runs`` directory), and a path heuristic mis-keys as soon as a layout changes -- which
    pools incomparable programs into one series. The package directory is the submission and is known
    here for certain, so it is recorded; the campaign labels (run id, arm, round) are supplied by
    whichever harness is driving, through the environment, and are ``None`` when nothing supplied
    them rather than being reconstructed from a guess.
    """
    pkg = Path(package_dir)
    env = os.environ
    ident = {
        "package": str(pkg.resolve()) if pkg.exists() else str(pkg),
        "run_id": run_id or env.get("MERLIN_SUBMISSION_RUN_ID") or None,
        "arm": env.get("MERLIN_SUBMISSION_ARM") or None,
        "round": env.get("MERLIN_SUBMISSION_ROUND") or None,
    }
    ident["stated"] = sorted(k for k, v in ident.items() if k != "package" and v)
    return ident


def _pin_model_capsule(capsule: dict, destination: Path) -> dict:
    """Keep the original runner patchpoint; implementation belongs to source custody."""
    from merlin.evaluation.capsule.source import _pin_model_capsule as pin

    return pin(capsule, destination)
