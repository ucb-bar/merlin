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
from typing import TYPE_CHECKING, Any

from merlin.common import jsonio as _mjson

from .execution_policy import (
    FIRESIM_QUEUE_OPERATION,
    FIRESIM_QUEUE_PHASES,
    FIRESIM_QUEUE_SCOPED_COMMANDS,
    FireSimQueuePreflight,
    QueuedFireSimReceipt,
    QueueLogEvidence,
    WarmComputeReceipt,
    WarmProfileContract,
)

if TYPE_CHECKING:  # `firesim_batch` imports FireSimReceiptError from here; see _cycle_claim_policy.
    from .firesim_batch import BatchValidationPolicy


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
# Each lifecycle command under the phase the daemon actually scopes it to, derived from the one
# observed trace in execution_policy so this cannot drift from the phase sequence.
_EXPECTED_COMMANDS = FIRESIM_QUEUE_SCOPED_COMMANDS


class FireSimReceiptError(ValueError):
    """Post-run evidence cannot prove the requested FireSim receipt."""


def _canonical_sha256(value: object) -> str:
    return _mjson.canonical_sha256(value, allow_nan=True)


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
            raise FireSimReceiptError(f"{self.role} must name an absolute plain file, observed {artifact}")
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


def _file_evidence(path: str | Path, role: str, *, executable: bool = False) -> tuple[PlainFileEvidence, bytes]:
    artifact = Path(path)
    if not artifact.is_absolute() or artifact.is_symlink() or not artifact.is_file():
        raise FireSimReceiptError(f"{role} must name an absolute plain file, observed {artifact}")
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
        raise FireSimReceiptError(f"{role} must name an absolute plain file, observed {artifact}")
    payload = artifact.read_bytes()
    evidence = QueueLogEvidence(role=role, path=str(artifact), sha256=hashlib.sha256(payload).hexdigest())
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
        return json.loads(_decode_utf8(payload, role), object_pairs_hook=_unique_json_object)
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
            if (
                not isinstance(marker, str)
                or not marker
                or marker != marker.strip()
                or "\n" in marker
                or "\r" in marker
                or "\0" in marker
            ):
                raise FireSimReceiptError("validation policy markers must be nonempty exact UART lines")
            if marker.startswith(reserved):
                raise FireSimReceiptError("correctness markers cannot reuse the profile or metric protocol")

    @classmethod
    def from_json(cls, value: object) -> UartValidationPolicy:
        if not isinstance(value, Mapping):
            raise FireSimReceiptError("validation policy must be a JSON object")
        expected_keys = {"schema", "policy_id", "workload", "success_markers"}
        if set(value) != expected_keys:
            raise FireSimReceiptError("validation policy keys must be exactly " + repr(sorted(expected_keys)))
        if value.get("schema") != VALIDATION_POLICY_SCHEMA:
            raise FireSimReceiptError(f"validation policy schema must be {VALIDATION_POLICY_SCHEMA!r}")
        markers = value.get("success_markers")
        if (
            not isinstance(markers, Sequence)
            or isinstance(markers, (str, bytes))
            or any(not isinstance(marker, str) for marker in markers)
        ):
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
class CycleClaim:
    """What a v2 policy proved about the one measured window, recorded as part of the receipt.

    A receipt without this block is a LEGACY receipt: it proves the run printed the exact lines its
    v1 policy named, and nothing about the values the run computed.  That is the shape FireSim job
    730 walked through, so a cycle number is citable only from a receipt that carries this.
    """

    window: str
    group_checksums: tuple[tuple[str, int], ...]
    #: How the workload SPELLS a checksum line, as its policy declared it -- carried as pairs so
    #: this stays a frozen value like every other piece of receipt evidence.
    checksum_line: tuple[tuple[str, object], ...]
    #: WHAT SPAN THE CYCLE NUMBER COUNTED, as the window's policy declared it: a contiguous
    #: ``t0 ... work ... t1``, or a sum of per-call deltas that prices nothing between the calls.
    #: Recorded here because the receipt is what a later reader has, and two receipts whose numbers
    #: mean different things are not comparable however similar they look.  ``None`` seals as
    #: ``UNKNOWN`` -- never as a kind -- and :func:`~merlin.perf.firesim_batch.cycle_ratio` refuses
    #: to divide with it.
    window_kind: str | None = None

    def to_dict(self) -> dict[str, object]:
        from .firesim_batch import WINDOW_KIND_UNKNOWN

        return {
            "window": self.window,
            "window_kind": self.window_kind or WINDOW_KIND_UNKNOWN,
            "checksum_line": dict(self.checksum_line),
            "group_checksums_verified": len(self.group_checksums),
            "group_checksums": dict(self.group_checksums),
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
    validation_policy: UartValidationPolicy | BatchValidationPolicy
    client_submission_line: int
    client_workload_line: int
    client_terminal_line: int
    lifecycle_observations: tuple[tuple[str, str, int], ...]
    invocation_line: int
    profile_lines: tuple[int, ...]
    metric_line: int
    correctness_lines: tuple[int, ...]
    #: The exact marker lines the policy required, in policy order.  Held here rather than read back
    #: off ``validation_policy`` because a v1 and a v2 policy spell "the markers" differently and a
    #: receipt that guessed which it held would describe the wrong evidence.
    correctness_markers: tuple[str, ...] = ()
    #: Present only for a cycle claim.  A legacy receipt omits the key entirely, so every receipt
    #: already sealed serialises byte-identically.
    cycle_claim: CycleClaim | None = None

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
                            self.correctness_markers,
                            self.correctness_lines,
                            strict=True,
                        )
                    ],
                },
                **({"cycle_claim": self.cycle_claim.to_dict()} if self.cycle_claim is not None else {}),
            },
        }


def _key_values(tokens: Sequence[str], *, role: str, line_number: int) -> dict[str, str]:
    values: dict[str, str] = {}
    for token in tokens:
        key, separator, value = token.partition("=")
        if not separator:
            continue
        if not key or not value or key in values:
            raise FireSimReceiptError(f"{role} line {line_number} has malformed or duplicate key/value tokens")
        values[key] = value
    return values


def _verify_submission(
    argv: tuple[str, ...], expected_executable: str, expected_workload: str
) -> FireSimQueuePreflight:
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
            workloads.append(token[len(prefix) :])
        index += 1
    if workloads != [expected_workload]:
        raise FireSimReceiptError("queue submission must contain exactly one --workload matching the expected workload")
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
                raise FireSimReceiptError(f"queue client line {line_number} has a non-integer job id") from exc
            if observed_job != job_id:
                raise FireSimReceiptError(f"queue client log contains job id {observed_job}, expected only {job_id}")
            job_lines.append((line_number, values, tokens))
        if "workload" in values:
            workload_lines.append((line_number, values))
        if "terminal" in tokens:
            terminal_lines.append((line_number, values, tokens))

    submissions = [
        (line, values)
        for line, values, _tokens in job_lines
        if values.get("kind") == FIRESIM_QUEUE_OPERATION and values.get("state") == "QUEUED"
    ]
    if len(submissions) != 1:
        raise FireSimReceiptError("queue client log must contain exactly one matching runworkload-full submission line")
    matching_workloads = [(line, values) for line, values in workload_lines if values.get("workload") == workload]
    if len(workload_lines) != 1 or len(matching_workloads) != 1:
        raise FireSimReceiptError("queue client log must contain exactly one matching workload line")
    if len(terminal_lines) != 1:
        raise FireSimReceiptError("queue client log must contain exactly one terminal record")
    terminal_line, terminal, _tokens = terminal_lines[0]
    if terminal.get("job_id") != str(job_id) or terminal.get("state") != "DONE" or terminal.get("exit_code") != "0":
        raise FireSimReceiptError(
            "queue client terminal record must prove the expected job reached state=DONE and exit_code=0"
        )
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
        raise FireSimReceiptError(f"queue daemon line {line_number} has a non-integer job id") from exc
    if observed_job != expected_job_id:
        raise FireSimReceiptError(f"queue daemon phase belongs to job {observed_job}, expected {expected_job_id}")
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
            raise FireSimReceiptError(f"queue daemon line {line_number} has an unscoped or malformed command marker")
        commands.append((current_phase, tokens[1], line_number))

    collapsed_phases: list[str] = []
    for phase in phases:
        if not collapsed_phases or collapsed_phases[-1] != phase:
            collapsed_phases.append(phase)
    if tuple(collapsed_phases) != FIRESIM_QUEUE_PHASES:
        raise FireSimReceiptError(
            f"queue daemon phases must be exactly {FIRESIM_QUEUE_PHASES}, observed {tuple(collapsed_phases)}"
        )
    observed = tuple((phase, command) for phase, command, _line in commands)
    if observed != _EXPECTED_COMMANDS:
        raise FireSimReceiptError(
            "queue daemon must prove exactly kill -> infrasetup -> runworkload -> kill "
            f"under their required phases, observed {observed}"
        )
    return tuple(commands)


def _verify_uart(text: str, success_markers: Sequence[str]) -> tuple[int, tuple[int, ...], int, int, tuple[int, ...]]:
    """The UART's SHAPE: one warm window, one measured window, one metric, the markers inside it.

    Takes the marker lines rather than a policy object so the same shape check serves a v1 policy's
    ``success_markers`` and a v2 window's ``markers``.  What a marker MEANS is the policy's business;
    what this function owns is placement and uniqueness, and it owns it for both.
    """
    lines = text.splitlines()

    invocation_lines = [number for number, line in enumerate(lines, start=1) if line.startswith("MERLIN_INVOCATIONS")]
    if len(invocation_lines) != 1 or lines[invocation_lines[0] - 1] != _INVOCATION_LINE:
        raise FireSimReceiptError(f"UART must contain exactly one {_INVOCATION_LINE!r} line")

    observed_profiles = [
        (number, line) for number, line in enumerate(lines, start=1) if line.startswith("MERLIN_PROFILE")
    ]
    if tuple(line for _number, line in observed_profiles) != _PROFILE_LINES:
        raise FireSimReceiptError(
            "UART must prove exactly one successful warm window before one successful measured window"
        )
    profile_numbers = tuple(number for number, _line in observed_profiles)

    metric_lines = [(number, line) for number, line in enumerate(lines, start=1) if line.startswith("METRIC")]
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
    if not (invocation_lines[0] < warm_begin < warm_end < measured_begin < metric_line < measured_end):
        raise FireSimReceiptError("UART invocation, warm, measured, and metric markers are not in the required order")

    correctness_lines: list[int] = []
    for marker in success_markers:
        matches = [number for number, line in enumerate(lines, start=1) if line == marker]
        if len(matches) != 1:
            raise FireSimReceiptError(f"UART must contain correctness marker {marker!r} exactly once")
        correctness_lines.append(matches[0])
    if correctness_lines != sorted(correctness_lines):
        raise FireSimReceiptError("UART correctness markers are not in validation-policy order")
    if any(not measured_begin < line < metric_line for line in correctness_lines):
        raise FireSimReceiptError("UART correctness markers must follow measured begin and precede metric publication")
    return invocation_lines[0], profile_numbers, metric_line, cycles, tuple(correctness_lines)


def _cycle_claim_policy(document: object, window_label: str):
    """The v2 policy and the one window a cycle claim names, or a refusal saying why not.

    Imported HERE rather than at module scope because :mod:`merlin.perf.firesim_batch` imports
    :class:`FireSimReceiptError` from this module; a top-level import would close the cycle.  That
    module owns what a v2 policy IS -- including the refusal of v1 BY NAME, and the refusal of a v2
    window that declares no checksum -- and this function reuses it rather than restating it, so
    the two cannot drift into disagreeing about what a cycle claim requires.
    """
    from .firesim_batch import BatchValidationPolicy

    policy = BatchValidationPolicy.from_json(document)
    if not isinstance(window_label, str) or not window_label.strip():
        raise FireSimReceiptError(
            "a cycle claim must name the measured window it is a claim about; an unattributable "
            "window is not a measurement"
        )
    return policy, policy.window(window_label)


def _verify_cycle_claim(uart_text: str, policy, window) -> tuple[CycleClaim, int | None]:
    """Admit the named window on EVERY declared marker AND group checksum, or refuse the claim.

    THE JOB-730 RULE, on the solo seal path.  That run published 23,787,829 cycles -- under both
    declared cycle thresholds -- with an argmax marker byte-identical to a correct run's, while
    group 1's output summed to 5,652,929 against an oracle 5,663,048.  An argmax is a one-bit check
    on a thousand-class vector; this compares what every group computed, because the window it is
    admitted against declares all of them.

    RUN BEFORE :func:`_verify_uart`, deliberately.  That function would refuse a wrong run too, but
    on the first marker line it could not find -- describing an arithmetic failure as a missing
    string.  Asking :func:`~merlin.perf.firesim_batch.admit_window` first means the refusal names
    the groups that disagreed and says the window RAN and was wrong, which is the distinction that
    module exists to keep: a wrong result reported as an absence hides a defect.

    Returns the claim and the cycle count published inside the frame, for the caller to cross-check
    against the one :func:`_verify_uart` reads; ``None`` when the frame published none, which that
    function then refuses on its own terms.
    """
    from .firesim_batch import PASS, admit_window

    admission = admit_window(uart_text, window, policy.checksum_line)
    if admission.status != PASS:
        detail = admission.reason
        if admission.checksum_mismatches:
            detail += "; " + ", ".join(
                f"group {group} summed {'ABSENT (no checksum line)' if observed is None else observed} "
                f"against an oracle {expected}"
                for group, observed, expected in admission.checksum_mismatches
            )
        raise FireSimReceiptError(
            f"the cycle claim for window {window.label!r} is not supported by its UART ({admission.status}): {detail}"
        )
    claim = CycleClaim(
        window=window.label,
        group_checksums=tuple(window.checksums),
        checksum_line=tuple(sorted(policy.checksum_line.to_dict().items())),
        window_kind=window.window_kind,
    )
    return claim, admission.cycles


def parse_queued_firesim_receipt(
    *,
    queue_client_log: str | Path,
    queue_daemon_log: str | Path,
    uart_log: str | Path,
    expected_queue_executable: str | Path,
    expected_submission_json: str | Path,
    expected_job_id: int,
    expected_workload: str,
    validation_policy_json: str | Path,
    cycle_claim_window: str | None = None,
) -> VerifiedQueuedFireSimReceipt:
    """Parse all post-run evidence and return a receipt only when every proof agrees.

    ``cycle_claim_window`` names the measured window when the receipt is to support a CYCLE CLAIM.
    Giving it changes what counts as proof: the validation policy must then be a v2 document
    declaring, for that window, what every compute group's output sums to, and the run must have
    printed all of them.  Omitting it keeps the legacy proof exactly as it was -- a v1 policy's
    exact marker lines, and nothing about the values computed.

    WHY V1 IS NOT REFUSED GLOBALLY.  The markers-only shape is what FireSim job 730 defeated, so it
    cannot support a cycle number.  It is still sound for what it actually proves, and receipts
    sealed under it (jobs 535 and 610) remain valid; refusing it here would retroactively invalidate
    them to fix a claim they never made.  The refusal is scoped to the claim, not to the schema.
    """
    if isinstance(expected_job_id, bool) or not isinstance(expected_job_id, int) or expected_job_id < 1:
        raise FireSimReceiptError("expected queue job id must be a positive integer")
    if not isinstance(expected_workload, str) or not expected_workload.strip():
        raise FireSimReceiptError("expected workload must be a nonempty string")

    executable_evidence, _executable_payload = _file_evidence(
        expected_queue_executable, "queue_executable", executable=True
    )
    submission_evidence, submission_payload = _file_evidence(expected_submission_json, "expected_submission")
    policy_evidence, policy_payload = _file_evidence(validation_policy_json, "validation_policy")
    client_evidence, client_payload = _log_evidence(queue_client_log, "queue_client")
    daemon_evidence, daemon_payload = _log_evidence(queue_daemon_log, "queue_daemon")
    uart_evidence, uart_payload = _log_evidence(uart_log, "uart")

    submission_value = _load_json(submission_payload, "expected submission")
    if (
        not isinstance(submission_value, Sequence)
        or isinstance(submission_value, (str, bytes))
        or any(not isinstance(token, str) or not token for token in submission_value)
    ):
        raise FireSimReceiptError("expected submission JSON must be a raw array of nonempty argv strings")
    submission = tuple(submission_value)
    preflight = _verify_submission(submission, str(expected_queue_executable), expected_workload)

    # The POLICY DOCUMENT parsed is the one whose bytes were just hashed into the evidence, not the
    # file read a second time; a policy edited between the two reads would otherwise seal against
    # one document and record the digest of another.
    policy_document = _load_json(policy_payload, "validation policy")
    window = None
    if cycle_claim_window is None:
        policy = UartValidationPolicy.from_json(policy_document)
        markers = policy.success_markers
    else:
        policy, window = _cycle_claim_policy(policy_document, cycle_claim_window)
        markers = window.markers
    if policy.workload != expected_workload:
        raise FireSimReceiptError("validation policy workload does not match the expected queue workload")

    client_submission_line, client_workload_line, client_terminal_line = _verify_client(
        _decode_utf8(client_payload, "queue client log"),
        expected_job_id,
        expected_workload,
    )
    lifecycle = _verify_daemon(_decode_utf8(daemon_payload, "queue daemon log"), expected_job_id)
    uart_text = _decode_utf8(uart_payload, "UART log")
    # THE CLAIM IS CHECKED FIRST so a wrong run is described by its arithmetic rather than by the
    # first marker line the shape check could not find; _verify_uart then holds the same UART to
    # the measurement protocol, and the two readings of the metric are compared.
    claim, framed_cycles = (None, None) if window is None else _verify_cycle_claim(uart_text, policy, window)
    invocation_line, profile_lines, metric_line, cycles, correctness_lines = _verify_uart(uart_text, markers)
    if claim is not None and framed_cycles != cycles:
        # Two readings of the same METRIC line disagreeing means the frame does not contain the
        # metric the receipt is about, so the number and the correctness evidence describe
        # different windows.
        raise FireSimReceiptError(
            f"window {claim.window!r} published {framed_cycles} cycles inside its frame and the "
            f"receipt reads {cycles}; the claim and its correctness evidence are not the same window"
        )

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
        correctness_markers=tuple(markers),
        cycle_claim=claim,
    )


BATCH_RECEIPT_SCHEMA = "merlin_queued_firesim_batch_receipt_v1"


@dataclass(frozen=True)
class SealedWindow:
    """One admitted window of a batched receipt: its cycles and the checksums that admitted it."""

    label: str
    cycles: int
    group_checksums: tuple[tuple[str, int], ...]
    markers: tuple[str, ...]
    is_order_control: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "label": self.label,
            "cycles": self.cycles,
            "group_checksums_verified": len(self.group_checksums),
            "group_checksums": dict(self.group_checksums),
            "markers": list(self.markers),
            "is_order_control": self.is_order_control,
        }


@dataclass(frozen=True)
class VerifiedBatchedFireSimReceipt:
    """N measured windows from ONE queue-owned job, each admitted on its own declared checksums.

    The queue-ownership half of this receipt is the solo receipt's, unchanged: the same submission
    argv, the same client terminal record, the same ``kill -> infrasetup -> runworkload -> kill``
    lifecycle.  What differs is the UART half.  A solo receipt asks
    :func:`_verify_uart` of the WHOLE log, which admits exactly one ``METRIC cycles`` line and so
    cannot read a batch at all.  This asks the same function of EACH WINDOW'S FRAME, so a batched
    window is held to exactly the protocol a solo window is -- one warm window, one measured window,
    its markers after ``measured begin`` and before the metric -- rather than to a second, laxer
    rule written for batches.
    """

    batch_id: str
    expected_workload: str
    queue_job_id: int
    preflight: FireSimQueuePreflight
    queue_executable_evidence: PlainFileEvidence
    submission_evidence: PlainFileEvidence
    submission_sha256: str
    validation_policy_evidence: PlainFileEvidence
    validation_policy: BatchValidationPolicy
    logs: tuple[QueueLogEvidence, ...]
    client_submission_line: int
    client_workload_line: int
    client_terminal_line: int
    lifecycle_observations: tuple[tuple[str, str, int], ...]
    invocation_line: int
    windows: tuple[SealedWindow, ...]
    order_effect_ppm: int
    order_effect_bound_ppm: int
    checksum_line: tuple[tuple[str, object], ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": BATCH_RECEIPT_SCHEMA,
            "status": "passed",
            "batch_id": self.batch_id,
            "expected_workload": self.expected_workload,
            "queue_receipt": {
                "queue_job_id": self.queue_job_id,
                "queue_owned": True,
                "queue_phases": list(FIRESIM_QUEUE_PHASES),
                "commands": [["firesim", command] for _phase, command, _line in self.lifecycle_observations],
                "logs": {log.role: {"path": log.path, "sha256": log.sha256} for log in self.logs},
            },
            "evidence": {
                "queue_executable": self.queue_executable_evidence.to_dict(),
                "expected_submission": {
                    **self.submission_evidence.to_dict(),
                    "argv_sha256": self.submission_sha256,
                },
                "validation_policy": {
                    **self.validation_policy_evidence.to_dict(),
                    "document_sha256": _canonical_sha256(self.validation_policy.to_dict()),
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
                    "checksum_line": dict(self.checksum_line),
                    "windows": [window.to_dict() for window in self.windows],
                },
                "order_effect": {
                    "observed_ppm": self.order_effect_ppm,
                    "declared_bound_ppm": self.order_effect_bound_ppm,
                    "why": (
                        "the batch's last window repeats window 0; without that control nothing "
                        "distinguishes a faster candidate from one that ran later, with the array "
                        "and DRAM in a state the first did not see"
                    ),
                },
            },
        }


def parse_batched_firesim_receipt(
    *,
    queue_client_log: str | Path,
    queue_daemon_log: str | Path,
    uart_log: str | Path,
    expected_queue_executable: str | Path,
    expected_submission_json: str | Path,
    expected_job_id: int,
    expected_workload: str,
    validation_policy_json: str | Path,
    batch,
) -> VerifiedBatchedFireSimReceipt:
    """Seal ONE queue job carrying N measured windows, or refuse and say which window failed.

    ``batch`` is the :class:`~merlin.perf.firesim_batch.LinkedBatch` the submission was planned
    from.  It is required, not inferred from the log: the window list, the order-effect control and
    its bound are the PLAN, and a receipt that read them back out of the run it is verifying would
    be grading the run against itself.
    """
    from .firesim_batch import PASS, BatchValidationPolicy, _frame, admit_batch

    if isinstance(expected_job_id, bool) or not isinstance(expected_job_id, int) or expected_job_id < 1:
        raise FireSimReceiptError("expected queue job id must be a positive integer")
    if not isinstance(expected_workload, str) or not expected_workload.strip():
        raise FireSimReceiptError("expected workload must be a nonempty string")

    executable_evidence, _payload = _file_evidence(expected_queue_executable, "queue_executable", executable=True)
    submission_evidence, submission_payload = _file_evidence(expected_submission_json, "expected_submission")
    policy_evidence, policy_payload = _file_evidence(validation_policy_json, "validation_policy")
    client_evidence, client_payload = _log_evidence(queue_client_log, "queue_client")
    daemon_evidence, daemon_payload = _log_evidence(queue_daemon_log, "queue_daemon")
    uart_evidence, uart_payload = _log_evidence(uart_log, "uart")

    submission_value = _load_json(submission_payload, "expected submission")
    if (
        not isinstance(submission_value, Sequence)
        or isinstance(submission_value, (str, bytes))
        or any(not isinstance(token, str) or not token for token in submission_value)
    ):
        raise FireSimReceiptError("expected submission JSON must be a raw array of nonempty argv strings")
    submission = tuple(submission_value)
    preflight = _verify_submission(submission, str(expected_queue_executable), expected_workload)

    policy = BatchValidationPolicy.from_json(_load_json(policy_payload, "validation policy"))
    if policy.workload != expected_workload:
        raise FireSimReceiptError("validation policy workload does not match the expected queue workload")

    client_submission_line, client_workload_line, client_terminal_line = _verify_client(
        _decode_utf8(client_payload, "queue client log"), expected_job_id, expected_workload
    )
    lifecycle = _verify_daemon(_decode_utf8(daemon_payload, "queue daemon log"), expected_job_id)
    uart_text = _decode_utf8(uart_payload, "UART log")

    # THE ARITHMETIC FIRST, for the reason _verify_cycle_claim gives on the solo path: a window that
    # RAN and was WRONG must be described by its checksums, not by the first marker a shape check
    # could not find.
    admission = admit_batch(uart_text, batch, policy)
    if admission.status != PASS:
        detail = [admission.reason]
        for window in admission.windows:
            if window.status != PASS:
                detail.append(f"{window.label}: {window.status} -- {window.reason}")
        raise FireSimReceiptError(f"batch {batch.batch_id!r} is not admitted: " + "; ".join(detail))

    lines = uart_text.splitlines()
    invocation_lines = [number for number, line in enumerate(lines, start=1) if line.startswith("MERLIN_INVOCATIONS")]
    if len(invocation_lines) != 1 or lines[invocation_lines[0] - 1] != _INVOCATION_LINE:
        raise FireSimReceiptError(f"UART must contain exactly one {_INVOCATION_LINE!r} line")

    windows: list[SealedWindow] = []
    for label in batch.labels:
        declared = policy.window(label)
        frame = _frame(uart_text, label)
        if frame is None:  # pragma: no cover -- admit_batch already refused this
            raise FireSimReceiptError(f"window {label!r} published no frame")
        # THE SAME SHAPE RULE AS A SOLO RECEIPT, applied to this window's frame.  Reusing
        # _verify_uart rather than restating it is the point: a batched window and a solo window
        # cannot come to be held to two different definitions of "one warm run, then one measured
        # run, with the correctness evidence inside the measured window".
        _invocation, _profiles, _metric_line, cycles, _markers = _verify_uart(
            "\n".join([_INVOCATION_LINE, *frame]), declared.markers
        )
        observed = next(window for window in admission.windows if window.label == label)
        if observed.cycles != cycles:
            raise FireSimReceiptError(
                f"window {label!r} reads {observed.cycles} cycles under the batch admission and "
                f"{cycles} under the measurement protocol; the two describe different windows"
            )
        windows.append(
            SealedWindow(
                label=label,
                cycles=cycles,
                group_checksums=tuple(declared.checksums),
                markers=tuple(declared.markers),
                is_order_control=label == batch.repeat_label,
            )
        )

    return VerifiedBatchedFireSimReceipt(
        batch_id=batch.batch_id,
        expected_workload=expected_workload,
        queue_job_id=expected_job_id,
        preflight=preflight,
        queue_executable_evidence=executable_evidence,
        submission_evidence=submission_evidence,
        submission_sha256=_canonical_sha256(list(submission)),
        validation_policy_evidence=policy_evidence,
        validation_policy=policy,
        logs=(client_evidence, daemon_evidence, uart_evidence),
        client_submission_line=client_submission_line,
        client_workload_line=client_workload_line,
        client_terminal_line=client_terminal_line,
        lifecycle_observations=lifecycle,
        invocation_line=invocation_lines[0],
        windows=tuple(windows),
        order_effect_ppm=int(admission.order_effect_ppm or 0),
        order_effect_bound_ppm=batch.order_effect_bound_ppm,
        checksum_line=tuple(sorted(policy.checksum_line.to_dict().items())),
    )


def write_queued_firesim_receipt(
    receipt: VerifiedQueuedFireSimReceipt | VerifiedBatchedFireSimReceipt, output: str | Path
) -> Path:
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
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
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
        description="Verify and atomically seal one completed queue-owned FireSim measurement."
    )
    parser.add_argument("--queue-client-log", required=True)
    parser.add_argument("--queue-daemon-log", required=True)
    parser.add_argument("--uart-log", required=True)
    parser.add_argument("--queue-executable", required=True)
    parser.add_argument(
        "--submission-json", required=True, help="absolute path to a JSON array containing the exact queue client argv"
    )
    parser.add_argument("--job-id", required=True, type=int)
    parser.add_argument("--workload", required=True)
    parser.add_argument(
        "--validation-policy", required=True, help="absolute path to a workload-owned UART validation-policy JSON file"
    )
    parser.add_argument(
        "--cycle-claim-window",
        default=None,
        help="seal this as a CYCLE CLAIM about the named measured window: the validation policy "
        "must then be a v2 document declaring every compute group's expected output checksum, and "
        "the run must have printed all of them. Without it the receipt proves only that the run "
        "printed the exact lines a v1 policy named, which is the shape FireSim job 730 defeated",
    )
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
            cycle_claim_window=arguments.cycle_claim_window,
        )
        output = write_queued_firesim_receipt(receipt, arguments.output)
    except (FireSimReceiptError, ValueError, OSError) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
