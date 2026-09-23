"""Host-owned Phase 2 tool broker: admission, execution and receipt closure.

Native scientific analyses enter through explicit services, never controller imports.
HTTP bodies have an absolute read deadline; headers have an idle socket timeout.
Receipt joining alone does not establish a whole-server shutdown deadline.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import secrets
import subprocess
import threading
import time
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from merlin.common.digest import sha256_bytes as _sha256
from merlin.targetgen.target_experiment import TargetExperiment

from .contracts import StageGateError
from .contracts import canonical_json as _canonical_json
from .contracts import write_json as _write_json

if TYPE_CHECKING:
    from .broker_policy import WorkflowPolicy


class SandboxPolicy(Protocol):
    argv: tuple[str, ...]
    network: str
    clear_environment: bool
    env_prefix: str | None
    process_cwd: Path | None

    def verify_execution(self) -> None: ...


AGENT_CORPUS_MOUNT = Path("/perf-corpus")


BROKER_NAME = "/perf-control/perf_tool.py"
BROKER_RECEIPT_MOUNT = Path("/perf-control/receipts.jsonl")


@dataclass(frozen=True)
class BrokerAction:
    name: str
    argv_template: tuple[str, ...]
    placeholders: tuple[str, ...]
    purpose: str
    required: bool
    #: WHY this action cannot be answered in this run, or None when it can. An action whose host
    #: provider was not installed is advertised WITH its refusal rather than as an ordinary action:
    #: see :data:`ACTION_PROVIDER_REQUIREMENTS` for the run in which an advertised-then-refused
    #: probe cost the agent every measured signal it had.
    unavailable_reason: str | None = None

    @property
    def available(self) -> bool:
        return self.unavailable_reason is None

    def as_dict(self) -> dict[str, Any]:
        """The SEALED identity: what may be executed, and with which bindings.

        Deliberately free of availability. This row is what `action_registry_contract` seals and
        transcript admission replays, so it is about execution authority; whether a provider is
        installed in one run is guidance, not authority, and belongs in :meth:`advertised`.
        """
        return {
            "name": self.name,
            "argv_template": list(self.argv_template),
            "placeholders": list(self.placeholders),
            "purpose": self.purpose,
            "required": self.required,
        }

    def advertised(self) -> dict[str, Any]:
        """The identity PLUS whether this run can answer it -- the row the agent is shown.

        The stage context used to list an action with no provider exactly like one that could be
        answered, and report the provider's absence in a separate field nothing joined to it.
        """
        return {**self.as_dict(), "available": self.available, "unavailable_reason": self.unavailable_reason}


def _placeholder_tokens(value: str) -> Iterator[tuple[int, int, str]]:
    """Find brace-delimited ASCII identifiers, retaining literal malformed text.

    Nested opening braces may start a valid token, e.g. ``{{name}}`` contains
    ``{name}``. Non-ASCII letters/digits and leading underscores are not names.
    """
    position = 0
    while position < len(value):
        start = value.find("{", position)
        if start < 0:
            return
        cursor = start + 1
        if cursor < len(value) and ("A" <= value[cursor] <= "Z" or "a" <= value[cursor] <= "z"):
            cursor += 1
            while cursor < len(value) and (
                "A" <= value[cursor] <= "Z"
                or "a" <= value[cursor] <= "z"
                or "0" <= value[cursor] <= "9"
                or value[cursor] == "_"
            ):
                cursor += 1
            if cursor < len(value) and value[cursor] == "}":
                yield start, cursor + 1, value[start + 1 : cursor]
                position = cursor + 1
                continue
        position = cursor


def placeholder_names(value: str) -> tuple[str, ...]:
    """Return command binding names in occurrence order, including repetitions."""
    return tuple(name for _, _, name in _placeholder_tokens(value))


def _render_placeholders(value: str, bindings: Mapping[str, str]) -> str:
    """Replace discovered tokens once; absent bindings preserve the KeyError contract."""
    parts: list[str] = []
    position = 0
    for start, end, name in _placeholder_tokens(value):
        parts.extend((value[position:start], bindings[name]))
        position = end
    parts.append(value[position:])
    return "".join(parts)


def inner_command(
    policy: SandboxPolicy,
    target_experiment: TargetExperiment,
    candidate: Path,
    argv: Sequence[str],
    timeout_s: int,
) -> list[str]:
    """Construct one shell-free payload for the inner broker."""
    if policy.network != "available_not_an_isolation_claim" or not policy.clear_environment:
        raise StageGateError("inner command requires the explicit clear-environment policy")
    if (
        not argv
        or any(not isinstance(value, str) or not value or "\0" in value for value in argv)
        or len(argv) > 256
        or sum(len(value) for value in argv) > 131_072
    ):
        raise StageGateError("inner tool argv is empty, malformed, or too large")
    if isinstance(timeout_s, bool) or not isinstance(timeout_s, int) or timeout_s <= 0:
        raise StageGateError("inner tool timeout must be a positive integer")
    verifier = getattr(policy, "verify_execution", None)
    if not callable(verifier):
        raise StageGateError("inner command requires a captured execution policy")
    verifier()
    environment = policy.env_prefix
    return [*policy.argv, "--chdir", str(candidate), "bash", "-c", environment + 'exec "$@"', "perf-tool", *argv]


class Broker:
    """A bounded localhost bridge from Codex to the credential-free inner bwrap."""

    def __init__(
        self,
        policy: SandboxPolicy,
        target_experiment: TargetExperiment,
        candidate: Path,
        actions: Sequence[BrokerAction],
        receipt_path: Path,
        *,
        deadline: float,
        workflow: WorkflowPolicy,
        max_calls: int,
        max_tool_seconds: int,
        mandatory_analysis_reserve_seconds: float = 0.0,
    ):
        self.workflow = workflow
        if (
            workflow.candidate != candidate
            or workflow.receipt_path != receipt_path
            or workflow.target_experiment != target_experiment
        ):
            raise StageGateError("broker workflow is bound to another invocation")
        self.policy = policy
        self.target_experiment = target_experiment
        self.candidate = candidate
        self.deadline = deadline
        if (
            isinstance(mandatory_analysis_reserve_seconds, bool)
            or not isinstance(mandatory_analysis_reserve_seconds, (int, float))
            or not math.isfinite(mandatory_analysis_reserve_seconds)
            or mandatory_analysis_reserve_seconds < 0
        ):
            raise StageGateError("mandatory analysis reserve must be finite and nonnegative")
        self.mandatory_analysis_reserve_seconds = float(mandatory_analysis_reserve_seconds)
        self.non_analysis_deadline = deadline - self.mandatory_analysis_reserve_seconds
        self.max_calls = max_calls
        self.max_tool_seconds = max_tool_seconds
        self.actions = {action.name: action for action in actions}
        workflow.validate_actions(self.actions)
        if not self.actions or len(self.actions) != len(actions):
            raise StageGateError("broker requires a non-empty unique action registry")
        self.receipt_path = receipt_path
        self.receipt_path.parent.mkdir(parents=True, exist_ok=True)
        if self.receipt_path.exists() or self.receipt_path.is_symlink():
            raise StageGateError(f"broker receipt path is not fresh: {self.receipt_path}")
        self.receipt_path.touch(mode=0o600)
        self.token = secrets.token_urlsafe(32)
        self.calls: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    def _record_refusal_locked(self, action_name: str, bindings: Mapping[str, Any], reason: str) -> StageGateError:
        """Record a refusal while ``self._lock`` is held and return its exception."""
        if len(self.calls) >= self.max_calls:
            return StageGateError("inner tool-call budget is exhausted")
        call_index = len(self.calls)
        recorded = {key: str(value) for key, value in sorted(bindings.items())}
        entry: dict[str, Any] = {
            "index": call_index,
            "action": action_name,
            "workflow_id": self.workflow.workflow_id,
            "bindings": recorded,
            "argv_sha256": _sha256(_canonical_json(list(self.actions[action_name].argv_template))),
            "timeout_s": 0,
            "state": "rejected",
            "returncode": 126,
            "stdout_sha256": _sha256(b""),
            "stderr_sha256": _sha256(b""),
            "rejection_reason": reason,
        }
        self.calls.append(entry)
        receipt = dict(entry)
        receipt["receipt_schema_version"] = 1
        receipt["bindings_command_sha256"] = _sha256(
            _canonical_json([f"{key}={value}" for key, value in sorted(recorded.items())])
        )
        payload = _canonical_json(receipt)
        with self.receipt_path.open("ab", buffering=0) as stream:
            stream.write(payload)
            os.fsync(stream.fileno())
        return StageGateError(reason)

    def _refuse(self, action_name: str, bindings: Mapping[str, Any], reason: str) -> StageGateError:
        """Record a REFUSED invocation in the ledger, then hand back the error to raise.

        A refusal the agent sees but the ledger never records is a hole in the receipts-to-transcript
        join: the transcript shows an invocation with no receipt, and `verify_broker_receipts` cannot
        tell "the broker refused this" from "a receipt went missing", so it refuses the entire run.
        Measured 2026-09-03 on perf_stage_20260903T151801Z -- the agent aimed one `output_json=` at
        /workspace (outside the mounts), was refused here before any ledger entry existed, corrected
        itself on the very next call, and a complete round was thrown away over 25 receipts against
        26 invocations.

        This ADDS evidence rather than relaxing anything: a refused row can never satisfy a required
        action, because only `state == "complete"` with returncode 0 counts, and an escape ATTEMPT is
        now visible in the ledger instead of vanishing from it.
        """
        with self._lock:
            return self._record_refusal_locked(action_name, bindings, reason)

    def execute(self, request: Mapping[str, Any]) -> dict[str, Any]:
        action_name, bindings = request.get("action"), request.get("bindings", {})
        if not isinstance(action_name, str) or action_name not in self.actions:
            raise StageGateError("broker request names an undeclared action")
        if not isinstance(bindings, Mapping):
            raise StageGateError("broker request bindings must be a mapping")
        action = self.actions[action_name]
        if set(bindings) != set(action.placeholders):
            raise self._refuse(
                action_name, bindings, f"broker action {action_name!r} requires exact bindings {action.placeholders}"
            )
        rendered: dict[str, str] = {}
        for name, value in bindings.items():
            if not isinstance(value, str) or not value or "\0" in value or len(value) > 8192:
                raise self._refuse(action_name, bindings, f"broker binding {name!r} is malformed")
            path = Path(value)
            if path.is_absolute():
                allowed_roots = (self.candidate, AGENT_CORPUS_MOUNT)
                if not any(path == root or root in path.parents for root in allowed_roots):
                    raise self._refuse(action_name, bindings, f"broker binding {name!r} escapes declared inputs")
            elif ".." in path.parts:
                raise self._refuse(action_name, bindings, f"broker binding {name!r} escapes the candidate")
            if (
                name.startswith("output")
                and path.is_absolute()
                and not (path == self.candidate or self.candidate in path.parents)
            ):
                raise self._refuse(action_name, bindings, f"broker output binding {name!r} is not candidate-scoped")
            rendered[name] = value
        raw_argv = [_render_placeholders(value, rendered) for value in action.argv_template]
        budget_error: StageGateError | None = None
        with self._lock:
            if len(self.calls) >= self.max_calls:
                raise StageGateError("inner tool-call budget is exhausted")
            admission = self.workflow.admission(action_name)
            action_limit, reserved = admission.limit, admission.reserved
            action_uses = sum(call.get("action") == action_name for call in self.calls)
            if action_limit is not None and action_uses >= action_limit:
                budget_error = self._record_refusal_locked(
                    action_name,
                    rendered,
                    admission.limit_refusal,
                )
                call_index, timeout_s = -1, 0
            else:
                action_deadline = self.deadline if reserved else self.non_analysis_deadline
                remaining = int(action_deadline - time.monotonic())
                requested = request.get("timeout_s", self.max_tool_seconds)
                if isinstance(requested, bool) or not isinstance(requested, int):
                    raise StageGateError("broker timeout must be an integer")
                timeout_s = min(requested, self.max_tool_seconds, remaining)
                if timeout_s <= 0:
                    reason = (
                        admission.reserve_refusal
                        if not reserved
                        and self.mandatory_analysis_reserve_seconds > 0
                        and self.deadline - time.monotonic() > 0
                        else "performance stage wall-clock budget is exhausted"
                    )
                    budget_error = self._record_refusal_locked(action_name, rendered, reason)
                    call_index, timeout_s = -1, 0
                else:
                    call_index = len(self.calls)
                    self.calls.append(
                        {
                            "index": call_index,
                            "action": action_name,
                            "workflow_id": self.workflow.workflow_id,
                            "bindings": dict(sorted(rendered.items())),
                            "argv_sha256": _sha256(_canonical_json(raw_argv)),
                            "timeout_s": timeout_s,
                            "state": "running",
                        }
                    )
        if budget_error is not None:
            raise budget_error
        started = time.monotonic()
        try:
            return self._execute_allocated(request, action_name, rendered, raw_argv, call_index, timeout_s, started)
        except StageGateError:
            # EVERY ALLOCATED INDEX GETS A RECEIPT. The index is taken before the inner command is
            # built, and building it can refuse (clear-environment policy, malformed argv, a
            # non-positive timeout). A refusal there used to leave the index allocated with nothing
            # written, so the receipt stream skipped a number and `verify_broker_receipts` rejected
            # the NEXT row with "violates the action schema" -- which killed a trial on 2026-09-03
            # (perf_agentic_20260903T212924Z__trial_01, receipt 6) after 42 clean invocations. The
            # ledger has to be gapless for the join to mean anything.
            with self._lock:
                # The allocated entry is already marked "running"; only a row that never
                # reached "complete" needs the refusal receipt written for it.
                if len(self.calls) > call_index and self.calls[call_index].get("state") == "running":
                    self.calls[call_index] = {
                        **self.calls[call_index],
                        "state": "rejected",
                        "returncode": 126,
                        "stdout_sha256": _sha256(b""),
                        "stderr_sha256": _sha256(b""),
                        "rejection_reason": "inner command was refused before it could run",
                    }
                    receipt = dict(self.calls[call_index])
                    receipt["receipt_schema_version"] = 1
                    receipt["bindings_command_sha256"] = _sha256(
                        _canonical_json([f"{k}={v}" for k, v in sorted(rendered.items())])
                    )
                    payload = _canonical_json(receipt)
                    with self.receipt_path.open("ab", buffering=0) as stream:
                        stream.write(payload)
                        os.fsync(stream.fileno())
            raise

    def _execute_allocated(
        self,
        request: Mapping[str, Any],
        action_name: str,
        rendered: dict[str, str],
        raw_argv: list[str],
        call_index: int,
        timeout_s: int,
        started: float,
    ) -> dict[str, Any]:
        outcome = self.workflow.execute(request, action_name, rendered, call_index, timeout_s, started)
        feedback_document = None
        if outcome is not None:
            result, feedback_document = outcome
        else:
            command = inner_command(self.policy, self.target_experiment, self.candidate, raw_argv, timeout_s)
            try:
                proc = subprocess.run(
                    command, cwd=str(self.policy.process_cwd), capture_output=True, text=True, timeout=timeout_s
                )
                result = {
                    "returncode": proc.returncode,
                    "stdout": (proc.stdout or "")[-1_000_000:],
                    "stderr": (proc.stderr or "")[-1_000_000:],
                    "elapsed_s": round(time.monotonic() - started, 3),
                }
            except subprocess.TimeoutExpired as exc:
                result = {
                    "returncode": 124,
                    "stdout": str(exc.stdout or "")[-1_000_000:],
                    "stderr": str(exc.stderr or "")[-1_000_000:],
                    "timed_out": True,
                    "elapsed_s": round(time.monotonic() - started, 3),
                }
        with self._lock:
            self.calls[call_index].update(
                {key: value for key, value in result.items() if key not in ("stdout", "stderr")}
            )
            self.calls[call_index]["stdout_sha256"] = _sha256(result["stdout"].encode("utf-8"))
            self.calls[call_index]["stderr_sha256"] = _sha256(result["stderr"].encode("utf-8"))
            if feedback_document is not None and result["returncode"] == 0:
                feedback_payload = _canonical_json(feedback_document)
                feedback_sha = _sha256(feedback_payload)
                feedback_dir = self.receipt_path.parent / "feedback" / "sha256"
                feedback_dir.mkdir(parents=True, exist_ok=True)
                feedback_path = feedback_dir / f"{feedback_sha}.json"
                if feedback_path.exists() or feedback_path.is_symlink():
                    if feedback_path.is_symlink() or feedback_path.read_bytes() != feedback_payload:
                        raise StageGateError("development feedback receipt digest collision")
                else:
                    with feedback_path.open("xb") as stream:
                        stream.write(feedback_payload)
                        stream.flush()
                        os.fsync(stream.fileno())
                    feedback_path.chmod(0o444)
                self.calls[call_index]["feedback_receipt_sha256"] = feedback_sha
                self.calls[call_index]["feedback_receipt_path"] = str(feedback_path.resolve())
            self.calls[call_index]["state"] = "complete"
            receipt = dict(self.calls[call_index])
            receipt["receipt_schema_version"] = 1
            receipt["bindings_command_sha256"] = _sha256(
                _canonical_json([f"{key}={value}" for key, value in sorted(rendered.items())])
            )
            payload = _canonical_json(receipt)
            with self.receipt_path.open("ab", buffering=0) as stream:
                stream.write(payload)
                os.fsync(stream.fileno())
        return result

    @contextlib.contextmanager
    def serving(self) -> Iterator[tuple[str, int]]:
        owner = self

        class ReceiptJoiningHTTPServer(ThreadingHTTPServer):
            # A return from this context is the host's receipt-ledger sealing boundary.  The
            # stdlib default is daemon_threads=True, so ThreadingMixIn.server_close() otherwise
            # returns with allocated execute requests still running.  The outer round then makes
            # receipts.jsonl read-only and the late handler loses its receipt.  Every action is
            # bounded after admission. Transport deadlines below also bound body reads.
            # Header reads have an idle timeout, not an absolute slow-client deadline;
            # joining preserves receipts but is not a whole-server shutdown guarantee.
            daemon_threads = False
            block_on_close = True

        class Handler(BaseHTTPRequestHandler):
            # StreamRequestHandler.setup installs this before parsing HTTP headers.
            timeout = 5.0

            def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
                try:
                    if self.path != "/execute" or self.headers.get("X-Perf-Token") != owner.token:
                        self.send_error(403)
                        return
                    length = int(self.headers.get("Content-Length") or 0)
                    if length <= 0 or length > 1_000_000:
                        self.send_error(400)
                        return
                    deadline = min(owner.deadline, time.monotonic() + self.timeout)
                    body = bytearray()
                    while len(body) < length:
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            raise TimeoutError("broker request-body deadline expired")
                        self.connection.settimeout(remaining)
                        # read1 returns after one underlying read, so slow progress
                        # cannot keep resetting an unbounded full-body read.
                        chunk = self.rfile.read1(min(length - len(body), 65536))
                        if not chunk:
                            raise ValueError("truncated broker request body")
                        body.extend(chunk)
                    if time.monotonic() >= deadline:
                        raise TimeoutError("broker request-body deadline expired")
                    self.connection.settimeout(self.timeout)
                    request = json.loads(body)
                    if not isinstance(request, dict):
                        raise StageGateError("broker request must be a JSON mapping")
                    response = owner.execute(request)
                    payload = _canonical_json(response)
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                except TimeoutError:
                    self.close_connection = True
                    # A stalled/disconnected peer may no longer accept an error.
                    with contextlib.suppress(OSError):
                        self.send_error(408)
                except (StageGateError, ValueError, json.JSONDecodeError) as exc:
                    payload = _canonical_json({"error": f"{type(exc).__name__}: {exc}"})
                    self.send_response(400)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)

            def log_message(self, _format: str, *args: object) -> None:
                return

        self._server = ReceiptJoiningHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        try:
            yield "127.0.0.1", int(self._server.server_address[1])
        finally:
            self._server.shutdown()
            self._server.server_close()
            self._thread.join(timeout=5)


_BROKER_SHIM = """#!/usr/bin/env python3
import json, pathlib, sys, urllib.error, urllib.request
cfg = json.loads((pathlib.Path(__file__).parent / ".perf_broker.json").read_text())
argv = sys.argv[1:]
if not argv or argv[0] not in cfg["actions"]:
    raise SystemExit("usage: python3 /perf-control/perf_tool.py ACTION [NAME=VALUE ...]")
action, bindings = argv[0], {}
for item in argv[1:]:
    if "=" not in item:
        raise SystemExit("broker action arguments must be exact NAME=VALUE bindings")
    name, value = item.split("=", 1)
    if not name or name in bindings:
        raise SystemExit("broker action binding is empty or repeated")
    bindings[name] = value
request = urllib.request.Request(
    cfg["url"], data=json.dumps({"action": action, "bindings": bindings,
                                 "timeout_s": cfg["tool_timeout_s"]}).encode(),
    headers={"Content-Type": "application/json", "X-Perf-Token": cfg["token"]}, method="POST")
try:
    with urllib.request.urlopen(request, timeout=cfg["tool_timeout_s"] + 10) as response:
        result = json.load(response)
except urllib.error.HTTPError as exc:
    sys.stderr.write(exc.read().decode(errors="replace"))
    raise SystemExit(125)
sys.stdout.write(result.get("stdout") or "")
sys.stderr.write(result.get("stderr") or "")
raise SystemExit(int(result.get("returncode", 125)))
"""


def stage_broker_shim(
    control_dir: Path, *, host: str, port: int, token: str, tool_timeout_s: int, actions: Sequence[BrokerAction]
) -> Path:
    if control_dir.is_symlink() or (control_dir.exists() and not control_dir.is_dir()):
        raise StageGateError(f"broker control directory is unsafe: {control_dir}")
    control_dir.mkdir(parents=True, exist_ok=True)
    unexpected = [path for path in control_dir.iterdir() if path.name != "receipts.jsonl"]
    if unexpected:
        raise StageGateError(f"broker control directory is not fresh: {control_dir}")
    shim = control_dir / Path(BROKER_NAME).name
    shim.write_text(_BROKER_SHIM, encoding="utf-8")
    shim.chmod(0o555)
    config = control_dir / ".perf_broker.json"
    _write_json(
        config,
        {
            "url": f"http://{host}:{port}/execute",
            "token": token,
            "tool_timeout_s": tool_timeout_s,
            "actions": sorted(action.name for action in actions),
        },
    )
    config.chmod(0o444)
    return shim


def action_registry_contract(actions: Sequence[BrokerAction], candidate: Path) -> list[dict[str, Any]]:
    """Normalize only the per-round candidate root; every manifest argument remains pinned."""
    root = str(candidate)
    return [
        {
            **action.as_dict(),
            "argv_template": [
                value.replace(root, "{candidate}", 1) if value == root or value.startswith(root + os.sep) else value
                for value in action.argv_template
            ],
        }
        for action in actions
    ]


def actions_from_registry_contract(rows: object, candidate: Path) -> tuple[BrokerAction, ...]:
    """Rebuild only the sealed action identities needed to replay transcript admission."""
    if not isinstance(rows, list) or not rows:
        raise StageGateError("sealed broker action registry is absent")
    actions: list[BrokerAction] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise StageGateError(f"sealed broker action {index} is malformed")
        name, argv = row.get("name"), row.get("argv_template")
        placeholders, purpose, required = (row.get("placeholders"), row.get("purpose"), row.get("required"))
        if (
            not isinstance(name, str)
            or not name
            or Path(name).name != name
            or not isinstance(argv, list)
            or not argv
            or any(not isinstance(value, str) or not value for value in argv)
            or not isinstance(placeholders, list)
            or any(not isinstance(value, str) or not value for value in placeholders)
            or len(placeholders) != len(set(placeholders))
            or not isinstance(purpose, str)
            or not purpose
            or not isinstance(required, bool)
        ):
            raise StageGateError(f"sealed broker action {index} violates the registry schema")
        found_placeholders = {name for value in argv for _, _, name in _placeholder_tokens(value)}
        if found_placeholders - {"candidate"} != set(placeholders):
            raise StageGateError(f"sealed broker action {index} changes its binding contract")
        expanded = tuple(
            str(candidate) + value.removeprefix("{candidate}")
            if value == "{candidate}" or value.startswith("{candidate}/")
            else value
            for value in argv
        )
        actions.append(BrokerAction(name, expanded, tuple(placeholders), purpose, required))
    if len({action.name for action in actions}) != len(actions):
        raise StageGateError("sealed broker action registry repeats an action")
    return tuple(actions)
