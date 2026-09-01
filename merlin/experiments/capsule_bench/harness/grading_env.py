"""Which interpreter grades a submission.

Kept in one file because both brokers spawn the grader and two expressions of one policy drift. The
answer is NOT ``sys.executable``: that is whatever launched the broker, and the broker is launched by
whatever launched the run. A batch driven through an orchestrator venv (Ray, a batch driver) therefore
handed the grading child an interpreter that could not import the submission's own dependencies, and
every graded capsule died in its parse entrypoint with ``ModuleNotFoundError`` -- surfaced as
``plane: parse, category: tool_crash``, which reads as the agent shipping something broken.

Measured on radiance: 4561 promoted certs across three repeats, all of them this one cause, zero
certifications recorded -- while the same submission bytes graded clean under the repo's own venv. The
run's own log said "could not certify (tooling, not the submission)", which was true and still left the
cause invisible.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


def grading_python() -> str:
    """Resolve the grading interpreter: explicit override, then the repo's venv, then this interpreter.

    Never assumed and never hard-coded to one path. ``MERLIN_GRADE_PYTHON`` wins so an operator can point
    the grade at a specific environment; otherwise the repo's own venv is preferred because it is the
    environment the submission's dependencies are installed into; otherwise we fall back to the current
    interpreter, which is the old behaviour and correct when the run was launched from that venv.
    """
    env = os.environ.get("MERLIN_GRADE_PYTHON")
    if env and Path(env).is_file():
        return env
    try:
        from merlin.common.paths import repo_root
        cand = Path(repo_root()) / ".venv" / "bin" / "python"
        if cand.is_file():
            return str(cand)
    except Exception:  # noqa: BLE001 -- an unresolvable repo root leaves this interpreter
        pass
    return sys.executable


def announce(who: str) -> str:
    """Return the grading interpreter, printing it when it differs from the one asking.

    Printed rather than silent: a grade running under an unexpected interpreter is invisible until it
    shows up as thousands of parse-plane crashes charged to the agent.
    """
    py = grading_python()
    if py != sys.executable:
        print(f"[{who}] grading interpreter: {py} (this process runs on {sys.executable})", flush=True)
    return py
