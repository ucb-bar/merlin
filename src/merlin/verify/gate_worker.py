"""Run one speculative gate check in this process and write its verdict.

``python -m merlin.verify.gate_worker <job-dir>`` reads ``<job-dir>/request.json``, resolves the
named check (``module:function``, or a built-in name), calls it with the request, and writes
``<job-dir>/verdict.json``. Only a check that returns normally is ``accepted``; every exception
is classified, and an unclassified one is ``unavailable``, never a pass.
"""

from __future__ import annotations

import importlib
import json
import os
import signal
import sys
import traceback
from pathlib import Path

BUILTIN_CHECKS = ("tensor_refinement", "bf16_f32acc_policy")


def resolve(name: str):
    if name in BUILTIN_CHECKS:
        from . import gate_checks

        return getattr(gate_checks, name)
    module, _, function = name.partition(":")
    if not module or not function:
        raise LookupError(f"check {name!r} is neither a built-in name nor module:function")
    return getattr(importlib.import_module(module), function)


def classify(error: BaseException) -> str:
    """Map a check's exception onto the four verdict kinds; anything unknown is unavailable."""
    from .gate_errors import GateRejected, GateUnavailable, GateUnsupported

    if isinstance(error, GateUnsupported):
        return "unsupported"
    if isinstance(error, GateRejected):
        return "not_established"
    if isinstance(error, GateUnavailable):
        return "unavailable"
    return "unavailable"


def write_verdict(directory: Path, verdict: dict) -> None:
    temporary = directory / "verdict.json.tmp"
    temporary.write_text(json.dumps(verdict, indent=2, sort_keys=True, default=str))
    os.replace(temporary, directory / "verdict.json")


def _stop_group(signum, frame) -> None:
    """A stop request (from the gate, or the parent-death signal when the compile dies) ends the
    whole check: the worker leads its own group, and the checker processes it runs are in it."""
    os.killpg(os.getpgid(0), signal.SIGKILL)


def main(argv: list[str]) -> int:
    if os.getpgid(0) == os.getpid():  # only as the leader of its own group (how the gate starts it)
        signal.signal(signal.SIGTERM, _stop_group)
        parent = os.environ.get("MERLIN_GATE_PARENT_PID")
        if parent is not None and str(os.getppid()) != parent:
            # Covers death before setpriv armed the kernel signal. No checker has
            # been started yet; never run one for an already vanished compile.
            _stop_group(signal.SIGTERM, None)
    directory = Path(argv[1])
    request = json.loads((directory / "request.json").read_text())
    try:
        detail = resolve(request["check"])(request) or {}
        if request["check"] == "bf16_f32acc_policy":
            # This draft check historically returns abstracted reports. Never
            # interpret a normally returning report as policy admission.
            from .gate_errors import GateRejected

            if (
                not isinstance(detail, dict)
                or detail.get("schema") != "merlin.bf16.policy.gate.v1"
                or detail.get("status") != "admitted"
            ):
                raise GateRejected("bf16 handler did not return admitted policy evidence")
        verdict = {"verdict": "accepted", "reason": "", "detail": detail}
    except BaseException as error:  # noqa: BLE001 -- every outcome must be recorded as a verdict
        verdict = {
            "verdict": classify(error),
            "reason": str(error) or type(error).__name__,
            "detail": {"error_type": type(error).__name__, "traceback": traceback.format_exc()[-4000:]},
        }
    write_verdict(directory, verdict)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
