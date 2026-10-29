"""Sandbox assembly runs in a worker only after the worker produced the identical argv."""

import os

from merlin_experiments.phase1.providers import execution as EX

from merlin.targetgen import grade_offload

SEEN = []


def _argv(n):
    return ["--ro-bind", str(n), str(os.getpid())]  # the pid tells where it ran


def _stable(n):
    return ["--ro-bind", str(n)]


def test_a_step_moves_to_a_worker_only_once_proven_identical(monkeypatch):
    monkeypatch.setattr(EX, "_ASSEMBLY_VERIFIED", {})
    try:
        assert EX._assemble("stable", _stable, 1) == _stable(1)
        assert EX._ASSEMBLY_VERIFIED == {"stable": True}
        # A step whose worker result differs (here: the pid) is kept in process for the run.
        assert EX._assemble("differs", _argv, 1) == _argv(1)
        assert EX._ASSEMBLY_VERIFIED["differs"] is False
        assert EX._assemble("differs", _argv, 2) == _argv(2)
        assert EX._assemble("stable", _stable, 3) == _stable(3)
    finally:
        grade_offload.shutdown()
