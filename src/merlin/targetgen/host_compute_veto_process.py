"""Evaluate the host-compute veto in a short-lived child process that imports the code on disk.

A campaign's grader is one long-lived process: it imports the audit once at launch, so a corrected
check never reached a background or final grade of a run already going, only fresh self-check
processes. Each grade therefore runs the veto in a new interpreter (``python -m`` this module),
resolving imports through the caller's ``sys.path`` and working directory, the same way grade
offload workers start. The child applies exactly the in-process veto, writes the same result files,
and hands back the updated rows, which the caller copies into its own row objects.

The child cannot change what the veto may do: it can fail a pass, never manufacture one. If it
cannot be started, dies, or its reply is unreadable, the veto runs in process as before.
``MERLIN_HOST_COMPUTE_VETO_PROCESS=0`` keeps it in process.
"""

from __future__ import annotations

import os
import pickle
import subprocess
import sys
from pathlib import Path
from typing import Any

ENV = "MERLIN_HOST_COMPUTE_VETO_PROCESS"
#: Set in the child so a veto there never starts another child.
_CHILD_ENV = "MERLIN_HOST_COMPUTE_VETO_CHILD"


def enabled() -> bool:
    return os.environ.get(ENV, "1").strip() != "0" and _CHILD_ENV not in os.environ


def run(results: list[dict[str, Any]], suite_runs: Path, *, capsules, target) -> list[dict[str, Any]] | None:
    """The vetoed rows, with ``results`` updated in place; ``None`` when the child gave no answer."""
    try:
        request = pickle.dumps((results, str(suite_runs), capsules, target), protocol=pickle.HIGHEST_PROTOCOL)
    except Exception:  # noqa: BLE001 -- unpicklable rows stay in process
        return None
    env = dict(os.environ)
    env[_CHILD_ENV] = "1"
    # Resolve imports exactly as the caller does, including paths it added at run time.
    env["PYTHONPATH"] = os.pathsep.join(path for path in sys.path if path)
    try:
        stderr = sys.stderr.fileno()
    except (AttributeError, OSError, ValueError):
        stderr = None
    try:
        done = subprocess.run(
            [sys.executable, "-m", __name__],
            input=request,
            stdout=subprocess.PIPE,
            stderr=stderr,
            env=env,
            close_fds=True,
            check=False,
        )
    except (OSError, ValueError):
        return None
    if done.returncode != 0:
        return None
    try:
        updated, vetoed = pickle.loads(done.stdout)
    except Exception:  # noqa: BLE001 -- an unreadable reply: the in-process veto decides
        return None
    if not isinstance(updated, list) or len(updated) != len(results) or not isinstance(vetoed, list):
        return None
    for row, new in zip(results, updated, strict=True):
        if row != new:
            row.clear()
            row.update(new)
    return vetoed


def _serve() -> int:
    """Child: read one request on stdin, apply the veto, reply on the original stdout."""
    reply = os.fdopen(os.dup(sys.stdout.fileno()), "wb")
    # Anything the audit prints goes to stderr, so it cannot corrupt the reply.
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    results, suite_runs, capsules, target = pickle.loads(sys.stdin.buffer.read())
    from merlin.targetgen.offload_census import _apply_host_compute_veto_here

    vetoed = _apply_host_compute_veto_here(results, Path(suite_runs), capsules=capsules, target=target)
    reply.write(pickle.dumps((results, vetoed), protocol=pickle.HIGHEST_PROTOCOL))
    reply.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(_serve())
