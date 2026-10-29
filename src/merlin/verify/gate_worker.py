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

BUILTIN_CHECKS = (
    "tensor_refinement",
    "compiler_loop_refinement",
    "compiler_buffer_refinement",
    "compiler_macc_contract",
)


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


def _require_builtin_result(name, detail):
    if name != "compiler_macc_contract":
        return
    from .compiler_macc import THEOREM
    from .gate_errors import GateRejected

    # The canonical handler must complete its live proof/delivery path. An
    # accidental pending/empty return cannot stand in for that claim. This
    # shape check is not independent proof or numerical approval authority.
    if (
        type(detail) is not dict
        or detail.get("status") != "checked_draft_contract"
        or detail.get("claim_type") != "draft_policy_conformance"
        or detail.get("theorem") != THEOREM
        or detail.get("policy_reviewed") is not False
        or detail.get("numerical_policy_verdict") != "abstracted"
        or detail.get("original_source_equivalence") is not False
        or type(detail.get("sites")) is not list
        or not detail["sites"]
    ):
        raise GateRejected("draft MACC handler did not establish its complete conditional claim")
    seen = set()
    for site in detail["sites"]:
        if (
            type(site) is not dict
            or set(site) != {"function", "source_operation", "source_results", "qualification_passed"}
            or type(site["function"]) is not str
            or not site["function"]
            or type(site["source_operation"]) is not int
            or site["source_operation"] < 0
            or site["qualification_passed"] is not True
            or type(site["source_results"]) is not list
            or not site["source_results"]
            or any(type(value) is not int or value < 0 for value in site["source_results"])
            or len(set(site["source_results"])) != len(site["source_results"])
        ):
            raise GateRejected("draft MACC handler returned a malformed or unqualified site")
        key = (site["function"], site["source_operation"])
        if key in seen:
            raise GateRejected("draft MACC handler returned duplicate site coverage")
        seen.add(key)


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
        detail = resolve(request["check"])(request)
        _require_builtin_result(request["check"], detail)
        detail = detail or {}
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
