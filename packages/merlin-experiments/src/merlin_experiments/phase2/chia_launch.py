"""Exact Chia launch admission shared by native and installed coordinators.

Version 2 pins this policy alongside the actual script or module source. Historical
v1 records are not rewritten and cannot authorize new execution through this owner.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from merlin.common.paths import module_source_path

from .contracts import PerformanceExperimentError as ExperimentError

PYTHON_SOURCE_ENVIRONMENT_KEYS = (
    "PYTHONPATH",
    "PYTHONSAFEPATH",
    "PYTHONHOME",
    "PYTHONNOUSERSITE",
    "PYTHONUSERBASE",
)


def _canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def _sha_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _is_sha(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def command_artifacts(command: Sequence[str]) -> list[dict[str, Any]]:
    """Pin the interpreter and real entrypoint; flags never masquerade as files."""
    if isinstance(command, (str, bytes)) or len(command) < 2 or any(not isinstance(v, str) or not v for v in command):
        raise ExperimentError("CHIA command must declare an executable and entrypoint")
    executable = Path(command[0]).resolve(strict=True)
    if command[1] == "-c":
        if len(command) < 3 or not executable.is_file():
            raise ExperimentError("CHIA inline Python smoke command is malformed")
        return [
            {"index": 0, "path": str(executable), "sha256": _sha_file(executable)},
            {"index": 2, "kind": "inline_python", "sha256": hashlib.sha256(command[2].encode()).hexdigest()},
        ]
    if command[1] == "-m":
        if len(command) < 3 or any(not part.isidentifier() for part in command[2].split(".")):
            raise ExperimentError("CHIA Python module entrypoint is malformed")
        entry = module_source_path(command[2])
        # Python -m executes a package's __main__, not its initializer.
        if entry.name == "__init__.py":
            entry = module_source_path(command[2] + ".__main__")
        index = 2
    else:
        if command[1].startswith("-"):
            raise ExperimentError("CHIA command supports a script or explicit Python -m module only")
        entry = Path(command[1])
        index = 1
    paths = ((0, executable), (index, entry.resolve(strict=True)))
    if any(not path.is_file() for _, path in paths):
        raise ExperimentError("CHIA command artifact is not a regular file")
    return [{"index": i, "path": str(path), "sha256": _sha_file(path)} for i, path in paths]


def policy_identity() -> dict[str, str]:
    path = module_source_path(__name__).resolve(strict=True)
    return {"path": str(path), "sha256": _sha_file(path)}


def verify_launch_receipt(*, command: Sequence[str], wrapper: Path, environment: Mapping[str, str]) -> dict[str, Any]:
    """Verify immutable assignment evidence before campaign mutation or paid work."""
    plan_sha256 = environment.get("MERLIN_CHIA_ENVELOPE_PLAN_SHA256")
    receipt_sha256 = environment.get("MERLIN_CHIA_LAUNCH_RECEIPT_SHA256")
    receipt_value = environment.get("MERLIN_CHIA_LAUNCH_RECEIPT")
    if not (_is_sha(plan_sha256) and _is_sha(receipt_sha256) and receipt_value):
        raise ExperimentError("actual campaign requires a content-addressed CHIA launch receipt, not only a plan id")
    receipt_path = Path(receipt_value)
    if receipt_path.is_symlink() or not receipt_path.is_file() or receipt_path.stat().st_mode & 0o222:
        raise ExperimentError("CHIA launch receipt is absent, linked, writable, or has changed")
    payload = receipt_path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != receipt_sha256:
        raise ExperimentError("CHIA launch receipt is absent, linked, writable, or has changed")
    try:
        receipt = json.loads(payload)
    except (UnicodeError, ValueError) as exc:
        raise ExperimentError("CHIA launch receipt is malformed") from exc
    if not isinstance(receipt, Mapping) or any(
        not isinstance(receipt.get(key), Mapping)
        for key in ("plan", "required_resources", "assigned_resources", "wrapper", "chia_trace", "launch_policy")
    ):
        raise ExperimentError("CHIA launch receipt is malformed")
    plan = receipt["plan"]
    required, assigned = receipt["required_resources"], receipt["assigned_resources"]
    recorded = receipt.get("command")
    command = list(command)
    expected_artifacts = command_artifacts(command)
    matches = isinstance(recorded, list) and len(recorded) == len(command) and all(isinstance(x, str) for x in recorded)
    if matches:
        matches = Path(recorded[0]).resolve() == Path(command[0]).resolve() and recorded[1:] == command[1:]
        if command[1] not in ("-m", "-c"):
            matches = (
                Path(recorded[0]).resolve() == Path(command[0]).resolve()
                and Path(recorded[1]).resolve() == Path(command[1]).resolve()
                and recorded[2:] == command[2:]
            )
    wrapper = Path(wrapper).resolve(strict=True)
    wrapper_record, chia_record = receipt["wrapper"], receipt["chia_trace"]
    chia_path = Path(str(chia_record.get("path") or ""))
    artifacts = receipt.get("command_artifacts")
    if (
        receipt.get("schema") != "merlin.chia-agentic-perf-launch.v2"
        or receipt.get("status") != "assigned_before_coordinator"
        or receipt.get("plan_sha256") != plan_sha256
        or plan.get("sha256") != plan_sha256
        or hashlib.sha256(_canonical({k: v for k, v in plan.items() if k != "sha256"})).hexdigest() != plan_sha256
        or plan.get("command") != recorded
        or not matches
        or required != {"codex_slots": 1, "gsim_slots": 1}
        or any(
            not isinstance(assigned.get(n), (int, float))
            or isinstance(assigned[n], bool)
            or (isinstance(assigned[n], float) and not math.isfinite(assigned[n]))
            or assigned[n] < 1
            for n in required
        )
        or artifacts != expected_artifacts
        or plan.get("command_artifacts") != artifacts
        or wrapper_record != {"path": str(wrapper), "sha256": _sha_file(wrapper)}
        or plan.get("wrapper") != wrapper_record
        or chia_path.is_symlink()
        or not chia_path.is_file()
        or chia_record.get("sha256") != _sha_file(chia_path)
        or plan.get("chia_trace") != chia_record
        or receipt["launch_policy"] != policy_identity()
        or plan.get("launch_policy") != receipt["launch_policy"]
    ):
        raise ExperimentError("CHIA launch receipt does not attest this exact assigned invocation")
    return {
        "path": str(receipt_path.resolve()),
        "sha256": receipt_sha256,
        "plan_sha256": plan_sha256,
        "required_resources": dict(required),
        "assigned_resources": dict(assigned),
        "wrapper": dict(wrapper_record),
        "chia_trace": dict(chia_record),
        "launch_policy": dict(receipt["launch_policy"]),
        "command": list(recorded),
        "command_artifacts": [dict(row) for row in artifacts],
    }
