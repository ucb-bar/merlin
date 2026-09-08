"""Verify and seal one completed, queue-owned FireSim measurement.

This module is deliberately post-run only.  It never imports a FireSim runner and never invokes a
queue command.  Instead it consumes the immutable inputs a submitter expected plus the three logs
produced by one completed job.  A receipt is emitted only when those independent evidence streams
agree on queue ownership, lifecycle order, warm/measured protocol, cycles, workload, and a
workload-supplied correctness policy.

The parser is target-neutral.  In particular, it knows no output tensor, class label, tolerance, or
accelerator-specific fence.  A target/workload adapter supplies exact successful-validation UART
lines in a small JSON policy; the generic verifier only checks their placement and uniqueness.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .execution_policy import (
    FIRESIM_QUEUE_OPERATION,
    FIRESIM_QUEUE_PHASES,
    FireSimQueuePreflight,
    QueueLogEvidence,
    QueuedFireSimReceipt,
    WarmComputeReceipt,
    WarmProfileContract,
)


RECEIPT_SCHEMA = "merlin_queued_firesim_receipt_v1"
VALIDATION_POLICY_SCHEMA = "merlin_firesim_uart_validation_policy_v1"
_CLIENT_PREFIX = "[firesim-queue]"
_PROFILE_LINES = (
    "MERLIN_PROFILE warmup begin",
    "MERLIN_PROFILE warmup end rc=0",
    "MERLIN_PROFILE measured begin",
    "MERLIN_PROFILE measured end rc=0",
)
_INVOCATION_LINE = "MERLIN_INVOCATIONS warmup=1 measured=1"
_EXPECTED_COMMANDS = (
    ("INFRASETUP", "kill"),
    ("INFRASETUP", "infrasetup"),
    ("RUNNING", "runworkload"),
    ("TEARDOWN", "kill"),
)


class FireSimReceiptError(ValueError):
    """Post-run evidence cannot prove the requested FireSim receipt."""


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class PlainFileEvidence:
    """Content identity for a non-log input used to establish a receipt."""

    role: str
    path: str
    sha256: str
    must_be_executable: bool = False

    def __post_init__(self) -> None:
        if not self.role.strip():
            raise FireSimReceiptError("file evidence must have a role")
        self.verified_bytes()

    def verified_bytes(self) -> bytes:
        artifact = Path(self.path)
        if not artifact.is_absolute() or artifact.is_symlink() or not artifact.is_file():
            raise FireSimReceiptError(
                f"{self.role} must name an absolute plain file, observed {artifact}")
        mode = artifact.stat().st_mode
        if self.must_be_executable and not mode & (stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH):
            raise FireSimReceiptError(f"{self.role} is not executable: {artifact}")
        payload = artifact.read_bytes()
        observed = hashlib.sha256(payload).hexdigest()
        if observed != self.sha256:
            raise FireSimReceiptError(f"content hash changed for {self.role}")
        return payload

    def to_dict(self) -> dict[str, object]:
        self.verified_bytes()
        return {"path": self.path, "sha256": self.sha256}


def _file_evidence(path: str | Path, role: str, *, executable: bool = False) \
        -> tuple[PlainFileEvidence, bytes]:
    artifact = Path(path)
    if not artifact.is_absolute() or artifact.is_symlink() or not artifact.is_file():
        raise FireSimReceiptError(
            f"{role} must name an absolute plain file, observed {artifact}")
    payload = artifact.read_bytes()
    evidence = PlainFileEvidence(
        role=role,
        path=str(artifact),
        sha256=hashlib.sha256(payload).hexdigest(),
        must_be_executable=executable,
    )
    return evidence, payload


def _log_evidence(path: str | Path, role: str) -> tuple[QueueLogEvidence, bytes]:
    artifact = Path(path)
    if not artifact.is_absolute() or artifact.is_symlink() or not artifact.is_file():
        raise FireSimReceiptError(
            f"{role} must name an absolute plain file, observed {artifact}")
    payload = artifact.read_bytes()
    evidence = QueueLogEvidence(
        role=role, path=str(artifact), sha256=hashlib.sha256(payload).hexdigest())
    return evidence, payload


def _decode_utf8(payload: bytes, role: str) -> str:
    try:
        return payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise FireSimReceiptError(f"{role} is not valid UTF-8: {exc}") from exc


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise FireSimReceiptError(f"JSON input repeats object key {key!r}")
        result[key] = value
    return result


def _load_json(payload: bytes, role: str) -> Any:
    try:
        return json.loads(
            _decode_utf8(payload, role), object_pairs_hook=_unique_json_object)
    except json.JSONDecodeError as exc:
        raise FireSimReceiptError(f"{role} is not valid JSON: {exc}") from exc


@dataclass(frozen=True)
class UartValidationPolicy:
    """Workload-owned exact UART lines that establish successful validation."""

    policy_id: str
    workload: str
    success_markers: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.policy_id.strip() or not self.workload.strip():
            raise FireSimReceiptError("validation policy must name its policy and workload")
        if not self.success_markers:
            raise FireSimReceiptError("validation policy needs at least one success marker")
        if len(set(self.success_markers)) != len(self.success_markers):
            raise FireSimReceiptError("validation policy success markers must be unique")
        reserved = ("MERLIN_PROFILE", "MERLIN_INVOCATIONS", "METRIC")
        for marker in self.success_markers:
            if (not isinstance(marker, str) or not marker or marker != marker.strip()
                    or "\n" in marker or "\r" in marker or "\0" in marker):
                raise FireSimReceiptError(
                    "validation policy markers must be nonempty exact UART lines")
            if marker.startswith(reserved):
                raise FireSimReceiptError(
                    "correctness markers cannot reuse the profile or metric protocol")

    @classmethod
    def from_json(cls, value: object) -> UartValidationPolicy:
        if not isinstance(value, Mapping):
            raise FireSimReceiptError("validation policy must be a JSON object")
        expected_keys = {"schema", "policy_id", "workload", "success_markers"}
        if set(value) != expected_keys:
            raise FireSimReceiptError(
                "validation policy keys must be exactly " + repr(sorted(expected_keys)))
        if value.get("schema") != VALIDATION_POLICY_SCHEMA:
            raise FireSimReceiptError(
                f"validation policy schema must be {VALIDATION_POLICY_SCHEMA!r}")
        markers = value.get("success_markers")
        if (not isinstance(markers, Sequence)
                or isinstance(markers, (str, bytes))
                or any(not isinstance(marker, str) for marker in markers)):
            raise FireSimReceiptError("validation policy success_markers must be a JSON array of strings")
        policy_id = value.get("policy_id")
        workload = value.get("workload")
        if not isinstance(policy_id, str) or not isinstance(workload, str):
            raise FireSimReceiptError("validation policy_id and workload must be strings")
        return cls(policy_id, workload, tuple(markers))

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": VALIDATION_POLICY_SCHEMA,
            "policy_id": self.policy_id,
            "workload": self.workload,
            "success_markers": list(self.success_markers),
        }


@dataclass(frozen=True)
class VerifiedQueuedFireSimReceipt:
    """A queue receipt plus the evidence identities and strict parser observations."""

    queue_receipt: QueuedFireSimReceipt
    expected_workload: str
    queue_executable_evidence: PlainFileEvidence
    submission_evidence: PlainFileEvidence
    submission_sha256: str
    validation_policy_evidence: PlainFileEvidence
    validation_policy: UartValidationPolicy
    client_submission_line: int
    client_workload_line: int
    client_terminal_line: int
    lifecycle_observations: tuple[tuple[str, str, int], ...]
    invocation_line: int
    profile_lines: tuple[int, ...]
    metric_line: int
    correctness_lines: tuple[int, ...]

    def to_dict(self) -> dict[str, object]:
        queue = self.queue_receipt.to_dict()
        executable = self.queue_executable_evidence.to_dict()
        submission = self.submission_evidence.to_dict()
        policy_source = self.validation_policy_evidence.to_dict()
        policy_document = self.validation_policy.to_dict()
        return {
            "schema": RECEIPT_SCHEMA,
            "status": "passed",
            "expected_workload": self.expected_workload,
            "queue_receipt": queue,
            "evidence": {
                "queue_executable": executable,
                "expected_submission": {
                    **submission,
                    "argv_sha256": self.submission_sha256,
                },
                "validation_policy": {
                    **policy_source,
                    "document_sha256": _canonical_sha256(policy_document),
                },
            },
            "verification": {
                "queue_client": {
                    "submission_line": self.client_submission_line,
                    "workload_line": self.client_workload_line,
                    "terminal_line": self.client_terminal_line,
                    "terminal_state": "DONE",
                },
                "daemon_lifecycle": [
                    {"phase": phase, "command": ["firesim", command], "line": line}
                    for phase, command, line in self.lifecycle_observations
                ],
                "uart": {
                    "invocation_line": self.invocation_line,
                    "profile_lines": [
                        {"marker": marker, "line": line}
                        for marker, line in zip(_PROFILE_LINES, self.profile_lines, strict=True)
                    ],
                    "metric": {
                        "name": "cycles",
                        "value": self.queue_receipt.warm_profile.total_compute_cycles,
                        "line": self.metric_line,
                    },
                    "correctness_markers": [
                        {"marker": marker, "line": line}
                        for marker, line in zip(
                            self.validation_policy.success_markers,
                            self.correctness_lines,
                            strict=True,
                        )
                    ],
                },
            },
        }


def _key_values(tokens: Sequence[str], *, role: str, line_number: int) -> dict[str, str]:
    values: dict[str, str] = {}
    for token in tokens:
        key, separator, value = token.partition("=")
        if not separator:
            continue
        if not key or not value or key in values:
            raise FireSimReceiptError(
                f"{role} line {line_number} has malformed or duplicate key/value tokens")
        values[key] = value
    return values


def _verify_submission(argv: tuple[str, ...], expected_executable: str,
                       expected_workload: str) -> FireSimQueuePreflight:
    preflight = FireSimQueuePreflight(expected_executable, argv)
    workloads: list[str] = []
    index = 0
    while index < len(argv):
        token = argv[index]
        if token == "--workload":
            if index + 1 >= len(argv):
                raise FireSimReceiptError("queue submission ends after --workload")
            workloads.append(argv[index + 1])
            index += 2
            continue
        prefix = "--workload="
        if token.startswith(prefix):
            workloads.append(token[len(prefix):])
        index += 1
    if workloads != [expected_workload]:
        raise FireSimReceiptError(
            "queue submission must contain exactly one --workload matching the expected workload")
    return preflight


def _verify_client(text: str, job_id: int, workload: str) -> tuple[int, int, int]:
    job_lines: list[tuple[int, dict[str, str], tuple[str, ...]]] = []
    workload_lines: list[tuple[int, dict[str, str]]] = []
    terminal_lines: list[tuple[int, dict[str, str], tuple[str, ...]]] = []
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        tokens = tuple(raw_line.strip().split())
        if not tokens or tokens[0] != _CLIENT_PREFIX:
            continue
        values = _key_values(tokens[1:], role="queue client", line_number=line_number)
        if "job_id" in values:
            try:
                observed_job = int(values["job_id"])
            except ValueError as exc:
                raise FireSimReceiptError(
                    f"queue client line {line_number} has a non-integer job id") from exc
            if observed_job != job_id:
                raise FireSimReceiptError(
                    f"queue client log contains job id {observed_job}, expected only {job_id}")
            job_lines.append((line_number, values, tokens))
        if "workload" in values:
            workload_lines.append((line_number, values))
        if "terminal" in tokens:
            terminal_lines.append((line_number, values, tokens))

    submissions = [
        (line, values) for line, values, _tokens in job_lines
        if values.get("kind") == FIRESIM_QUEUE_OPERATION and values.get("state") == "QUEUED"
    ]
    if len(submissions) != 1:
        raise FireSimReceiptError(
            "queue client log must contain exactly one matching runworkload-full submission line")
    matching_workloads = [
        (line, values) for line, values in workload_lines
        if values.get("workload") == workload
    ]
    if len(workload_lines) != 1 or len(matching_workloads) != 1:
        raise FireSimReceiptError(
            "queue client log must contain exactly one matching workload line")
    if len(terminal_lines) != 1:
        raise FireSimReceiptError(
            "queue client log must contain exactly one terminal record")
    terminal_line, terminal, _tokens = terminal_lines[0]
    if (terminal.get("job_id") != str(job_id)
            or terminal.get("state") != "DONE" or terminal.get("exit_code") != "0"):
        raise FireSimReceiptError(
            "queue client terminal record must prove the expected job reached "
            "state=DONE and exit_code=0")
    return submissions[0][0], matching_workloads[0][0], terminal_line


def _daemon_phase(raw_line: str, line_number: int, expected_job_id: int) -> str | None:
    tokens = raw_line.strip().split()
    if len(tokens) < 2 or tokens[0] != "===" or tokens[1] != _CLIENT_PREFIX:
        return None
    if len(tokens) != 5 or tokens[-1] != "===":
        raise FireSimReceiptError(f"queue daemon line {line_number} has a malformed phase banner")
    values = _key_values(tokens[2:-1], role="queue daemon", line_number=line_number)
    if set(values) != {"phase", "job_id"}:
        raise FireSimReceiptError(f"queue daemon line {line_number} has a malformed phase banner")
    try:
        observed_job = int(values["job_id"])
    except ValueError as exc:
        raise FireSimReceiptError(
            f"queue daemon line {line_number} has a non-integer job id") from exc
    if observed_job != expected_job_id:
        raise FireSimReceiptError(
            f"queue daemon phase belongs to job {observed_job}, expected {expected_job_id}")
    return values["phase"]


def _verify_daemon(text: str, job_id: int) -> tuple[tuple[str, str, int], ...]:
    phases: list[str] = []
    commands: list[tuple[str, str, int]] = []
    current_phase: str | None = None
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        phase = _daemon_phase(raw_line, line_number, job_id)
        if phase is not None:
            phases.append(phase)
            current_phase = phase
            continue
        stripped = raw_line.strip()
        if not stripped.startswith("Running:"):
            continue
        tokens = stripped.split()
        if len(tokens) != 2 or tokens[0] != "Running:" or current_phase is None:
            raise FireSimReceiptError(
                f"queue daemon line {line_number} has an unscoped or malformed command marker")
        commands.append((current_phase, tokens[1], line_number))

    collapsed_phases: list[str] = []
    for phase in phases:
        if not collapsed_phases or collapsed_phases[-1] != phase:
            collapsed_phases.append(phase)
    if tuple(collapsed_phases) != FIRESIM_QUEUE_PHASES:
        raise FireSimReceiptError(
            f"queue daemon phases must be exactly {FIRESIM_QUEUE_PHASES}, "
            f"observed {tuple(collapsed_phases)}")
    observed = tuple((phase, command) for phase, command, _line in commands)
    if observed != _EXPECTED_COMMANDS:
        raise FireSimReceiptError(
            "queue daemon must prove exactly kill -> infrasetup -> runworkload -> kill "
            f"under their required phases, observed {observed}")
    return tuple(commands)


def _verify_uart(text: str, policy: UartValidationPolicy) \
        -> tuple[int, tuple[int, ...], int, int, tuple[int, ...]]:
    lines = text.splitlines()

    invocation_lines = [
        number for number, line in enumerate(lines, start=1)
        if line.startswith("MERLIN_INVOCATIONS")
    ]
    if len(invocation_lines) != 1 or lines[invocation_lines[0] - 1] != _INVOCATION_LINE:
        raise FireSimReceiptError(
            f"UART must contain exactly one {_INVOCATION_LINE!r} line")

    observed_profiles = [
        (number, line) for number, line in enumerate(lines, start=1)
        if line.startswith("MERLIN_PROFILE")
    ]
    if tuple(line for _number, line in observed_profiles) != _PROFILE_LINES:
        raise FireSimReceiptError(
            "UART must prove exactly one successful warm window before one successful measured window")
    profile_numbers = tuple(number for number, _line in observed_profiles)

    metric_lines = [
        (number, line) for number, line in enumerate(lines, start=1)
        if line.startswith("METRIC")
    ]
    if len(metric_lines) != 1:
        raise FireSimReceiptError("UART must contain exactly one METRIC line")
    metric_line, metric_text = metric_lines[0]
    metric = metric_text.split()
    if len(metric) != 3 or metric[:2] != ["METRIC", "cycles"]:
        raise FireSimReceiptError("the sole UART metric must be METRIC cycles N")
    try:
        cycles = int(metric[2])
    except ValueError as exc:
        raise FireSimReceiptError("UART cycle metric must be an integer") from exc
    if cycles <= 0:
        raise FireSimReceiptError("UART cycle metric must be positive")

    warm_begin, warm_end, measured_begin, measured_end = profile_numbers
    if not (invocation_lines[0] < warm_begin < warm_end < measured_begin
            < metric_line < measured_end):
        raise FireSimReceiptError(
            "UART invocation, warm, measured, and metric markers are not in the required order")

    correctness_lines: list[int] = []
    for marker in policy.success_markers:
        matches = [
            number for number, line in enumerate(lines, start=1) if line == marker
        ]
        if len(matches) != 1:
            raise FireSimReceiptError(
                f"UART must contain correctness marker {marker!r} exactly once")
        correctness_lines.append(matches[0])
    if correctness_lines != sorted(correctness_lines):
        raise FireSimReceiptError("UART correctness markers are not in validation-policy order")
    if any(not measured_begin < line < metric_line for line in correctness_lines):
        raise FireSimReceiptError(
            "UART correctness markers must follow measured begin and precede metric publication")
    return invocation_lines[0], profile_numbers, metric_line, cycles, tuple(correctness_lines)


def parse_queued_firesim_receipt(
        *, queue_client_log: str | Path, queue_daemon_log: str | Path,
        uart_log: str | Path, expected_queue_executable: str | Path,
        expected_submission_json: str | Path, expected_job_id: int,
        expected_workload: str, validation_policy_json: str | Path,
        ) -> VerifiedQueuedFireSimReceipt:
    """Parse all post-run evidence and return a receipt only when every proof agrees."""
    if isinstance(expected_job_id, bool) or not isinstance(expected_job_id, int) or expected_job_id < 1:
        raise FireSimReceiptError("expected queue job id must be a positive integer")
    if not isinstance(expected_workload, str) or not expected_workload.strip():
        raise FireSimReceiptError("expected workload must be a nonempty string")

    executable_evidence, _executable_payload = _file_evidence(
        expected_queue_executable, "queue_executable", executable=True)
    submission_evidence, submission_payload = _file_evidence(
        expected_submission_json, "expected_submission")
    policy_evidence, policy_payload = _file_evidence(
        validation_policy_json, "validation_policy")
    client_evidence, client_payload = _log_evidence(queue_client_log, "queue_client")
    daemon_evidence, daemon_payload = _log_evidence(queue_daemon_log, "queue_daemon")
    uart_evidence, uart_payload = _log_evidence(uart_log, "uart")

    submission_value = _load_json(submission_payload, "expected submission")
    if (not isinstance(submission_value, Sequence)
            or isinstance(submission_value, (str, bytes))
            or any(not isinstance(token, str) or not token for token in submission_value)):
        raise FireSimReceiptError(
            "expected submission JSON must be a raw array of nonempty argv strings")
    submission = tuple(submission_value)
    preflight = _verify_submission(
        submission, str(expected_queue_executable), expected_workload)

    policy = UartValidationPolicy.from_json(
        _load_json(policy_payload, "validation policy"))
    if policy.workload != expected_workload:
        raise FireSimReceiptError(
            "validation policy workload does not match the expected queue workload")

    client_submission_line, client_workload_line, client_terminal_line = _verify_client(
        _decode_utf8(client_payload, "queue client log"), expected_job_id,
        expected_workload,
    )
    lifecycle = _verify_daemon(
        _decode_utf8(daemon_payload, "queue daemon log"), expected_job_id)
    invocation_line, profile_lines, metric_line, cycles, correctness_lines = _verify_uart(
        _decode_utf8(uart_payload, "UART log"), policy)

    commands = tuple(("firesim", command) for _phase, command, _line in lifecycle)
    warm = WarmComputeReceipt(
        workload=expected_workload,
        total_compute_cycles=cycles,
        contract=WarmProfileContract(),
        provenance="one queue-owned post-warm FireSim compute-cycle window",
    )
    queued = QueuedFireSimReceipt(
        queue_job_id=expected_job_id,
        queue_owned=True,
        preflight=preflight,
        queue_phases=FIRESIM_QUEUE_PHASES,
        commands=commands,
        logs=(client_evidence, daemon_evidence, uart_evidence),
        warm_profile=warm,
    )
    return VerifiedQueuedFireSimReceipt(
        queue_receipt=queued,
        expected_workload=expected_workload,
        queue_executable_evidence=executable_evidence,
        submission_evidence=submission_evidence,
        submission_sha256=_canonical_sha256(list(submission)),
        validation_policy_evidence=policy_evidence,
        validation_policy=policy,
        client_submission_line=client_submission_line,
        client_workload_line=client_workload_line,
        client_terminal_line=client_terminal_line,
        lifecycle_observations=lifecycle,
        invocation_line=invocation_line,
        profile_lines=profile_lines,
        metric_line=metric_line,
        correctness_lines=correctness_lines,
    )


def write_queued_firesim_receipt(
        receipt: VerifiedQueuedFireSimReceipt, output: str | Path) -> Path:
    """Atomically write the deterministic JSON representation of ``receipt``."""
    destination = Path(output)
    if not destination.is_absolute():
        raise FireSimReceiptError("receipt output must be an absolute path")
    if destination.exists() and (destination.is_symlink() or not destination.is_file()):
        raise FireSimReceiptError("receipt output cannot replace a symlink or non-file")
    if not destination.parent.is_dir():
        raise FireSimReceiptError("receipt output parent directory must already exist")
    payload = json.dumps(receipt.to_dict(), indent=2, sort_keys=True) + "\n"
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=destination.parent,
                prefix=f".{destination.name}.", suffix=".tmp", delete=False) as temporary:
            temporary_name = temporary.name
            temporary.write(payload)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_name, destination)
        temporary_name = None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)
    return destination


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify and atomically seal one completed queue-owned FireSim measurement.")
    parser.add_argument("--queue-client-log", required=True)
    parser.add_argument("--queue-daemon-log", required=True)
    parser.add_argument("--uart-log", required=True)
    parser.add_argument("--queue-executable", required=True)
    parser.add_argument(
        "--submission-json", required=True,
        help="absolute path to a JSON array containing the exact queue client argv")
    parser.add_argument("--job-id", required=True, type=int)
    parser.add_argument("--workload", required=True)
    parser.add_argument(
        "--validation-policy", required=True,
        help="absolute path to a workload-owned UART validation-policy JSON file")
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    arguments = parser.parse_args(argv)
    try:
        receipt = parse_queued_firesim_receipt(
            queue_client_log=arguments.queue_client_log,
            queue_daemon_log=arguments.queue_daemon_log,
            uart_log=arguments.uart_log,
            expected_queue_executable=arguments.queue_executable,
            expected_submission_json=arguments.submission_json,
            expected_job_id=arguments.job_id,
            expected_workload=arguments.workload,
            validation_policy_json=arguments.validation_policy,
        )
        output = write_queued_firesim_receipt(receipt, arguments.output)
    except (FireSimReceiptError, ValueError, OSError) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
