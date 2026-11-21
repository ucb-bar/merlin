"""Opt-in, bounded file transport for the original authenticated Broker.

Requests are untrusted data, never a command or an evaluator selection. Only
Broker.execute dispatches them and writes the original authoritative receipts.
Replies contain exactly the already public HTTP response, never private files.
The selected limits are 256 KiB/request, 512 KiB/reply and one dispatcher.
Original action calls retain their budget, with 16 extra envelope refusals.
Queue workers retain their own concurrency. Arbitrarily concurrent client
transcript ordering is not qualified; the original receipt join stays strict.
"""

from __future__ import annotations

import hmac
import json
import math
import os
import signal
import socket
import stat
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

from merlin.common.digest import hash_tree
from merlin.common.process import ownership as owned_children
from merlin.common.process.ownership import become_child_subreaper, reap_owned_children
from merlin.compiler.partition.global_plan import CycleInterval
from merlin.development.backend.brokers import compiler_call_channel as IO
from merlin.development.backend.optimization.campaign.portfolio_launch import acquire_host_resource_lease
from merlin.development.backend.optimization.contracts import StageGateError, mapping_file, sha256_file, write_json
from merlin.experiments.execution import _protocol as protocol

# ---- broker_file_channel -----------------------------------------------------------------------

PROTOCOL = "merlin.phase2_broker_file_channel.v1"

MAX_REQUEST_BYTES = IO.MAX_REQUEST_BYTES

MAX_REPLY_BYTES = 2 * IO.MAX_REQUEST_BYTES

MAX_REQUESTS = IO.MAX_PENDING_ENTRIES

BAD_REQUEST_ALLOWANCE = 16

MAX_CONCURRENT = 1  # Preserve the original completion-index receipt order.

REPLIES = "broker-replies"


def validate_call_budget(max_calls):
    """Reject incompatible stage selections before expensive owner setup."""
    if type(max_calls) is not int or not 1 <= max_calls <= MAX_REQUESTS - BAD_REQUEST_ALLOWANCE:
        raise StageGateError("file transport requires a bounded original action-call budget")


def _directory(path):
    path = Path(path)
    if not path.is_absolute() or path.resolve(strict=True) != path or not path.is_dir():
        raise StageGateError("file broker requires canonical original directories")
    if any(parent.is_symlink() for parent in (path, *path.parents)):
        raise StageGateError("file broker directory is linked")
    return path


def _request_id(name):
    if not name.startswith("req_") or not name.endswith(".json"):
        return None
    value = name[4:-5]
    return value if len(value) == 32 and all(c in "0123456789abcdef" for c in value) else None


def _refusal(reason):
    return {"returncode": 125, "stdout": "", "stderr": "file broker transport refused: " + reason}


def _dispatch(broker, requests, replies, name, request_id):
    unexpected = None
    try:
        before = os.stat(name, dir_fd=requests, follow_symlinks=False)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise ValueError("request is not a single-link regular file")
        document = IO._read(requests, name, MAX_REQUEST_BYTES)
        after = os.stat(name, dir_fd=requests, follow_symlinks=False)

        def fields(s):
            return (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns, s.st_nlink)

        if fields(before) != fields(after):
            raise ValueError("request path changed during read")
        if (
            type(document) is not dict
            or set(document) != {"schema", "request_id", "token", "request"}
            or document["schema"] != PROTOCOL
            or document["request_id"] != request_id
            or type(document["token"]) is not str
            or not document["token"].isascii()
            or not hmac.compare_digest(document["token"], broker.token)
            or type(document["request"]) is not dict
        ):
            raise ValueError("request authentication or closed envelope failed")
        if time.monotonic() >= broker.deadline:
            raise TimeoutError("original broker deadline expired")
        response = broker.execute(document["request"])
    except (OSError, ValueError, StageGateError, TimeoutError):
        response = _refusal("invalid, moved, unauthenticated or expired request")
    except Exception as error:
        # A refused host scope must reach the waiting client immediately. Keep
        # the original failure on the future so final channel cleanup refuses;
        # this transport reply is never an original successful action receipt.
        unexpected = error
        response = _refusal("original dispatch failed: " + type(error).__name__)
    if len(json.dumps(response, sort_keys=True).encode()) > MAX_REPLY_BYTES:
        response = _refusal("original response exceeded the selected reply byte ceiling; original receipt retained")
    try:
        IO._write(replies, "resp_" + request_id + ".json", response)
    except Exception as error:
        if unexpected is not None:
            unexpected.add_note("file broker failure reply also failed: " + type(error).__name__)
            raise unexpected from error
        raise
    if unexpected is not None:
        raise unexpected


@contextmanager
def serving(broker, *, workspace, control_dir):
    from merlin.development.backend.optimization.author_interface.broker_service import Broker

    if type(broker) is not Broker:
        raise StageGateError("file transport requires the original broker")
    validate_call_budget(broker.max_calls)
    request_limit = broker.max_calls + BAD_REQUEST_ALLOWANCE
    workspace, control = _directory(workspace), _directory(control_dir)
    if not broker.candidate.is_relative_to(workspace) or control != (
        broker.public_control_dir or broker.receipt_path.parent
    ):
        raise StageGateError("file channel differs from the original broker owners")
    inbox, outbox = workspace / ".perf-broker-requests", control / REPLIES
    for path in (inbox, outbox):
        if path.exists() or path.is_symlink():
            raise StageGateError("file broker channel must be fresh")
    inbox.mkdir(mode=0o700)
    outbox.mkdir(mode=0o700)
    request_fd = os.open(inbox, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    reply_fd = os.open(outbox, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    identities = [(path, os.fstat(fd)) for path, fd in ((inbox, request_fd), (outbox, reply_fd))]

    def verify_paths():
        for path, issued in identities:
            _directory(path)
            actual = path.stat()
            if (actual.st_dev, actual.st_ino) != (issued.st_dev, issued.st_ino):
                raise StageGateError("file broker directory was replaced")

    stop, errors = threading.Event(), []
    processed = set()
    pool = ThreadPoolExecutor(max_workers=MAX_CONCURRENT, thread_name_prefix="original-file-broker")
    futures = []

    def poll():
        try:
            while not stop.wait(0.02) and time.monotonic() < broker.deadline:
                verify_paths()
                with os.scandir(request_fd) as entries:
                    rows = []
                    for index, row in enumerate(entries):
                        if index >= IO.MAX_PENDING_ENTRIES:
                            raise StageGateError("file broker exceeded its pending-entry ceiling")
                        request_id = _request_id(row.name)
                        if request_id is not None and row.name not in processed:
                            rows.append((row.stat(follow_symlinks=False).st_mtime_ns, row.name, request_id))
                for _arrived, name, request_id in sorted(rows):
                    if len(processed) >= request_limit:
                        raise StageGateError("file broker exhausted its selected request ceiling")
                    processed.add(name)
                    futures.append(pool.submit(_dispatch, broker, request_fd, reply_fd, name, request_id))
        except BaseException as error:
            errors.append(error)
            stop.set()

    thread = threading.Thread(target=poll, name="original-file-broker-poll", daemon=True)
    thread.start()
    primary = None
    try:
        yield {
            "schema": PROTOCOL,
            "request_dir": str(inbox),
            "reply_dir": "/perf-control/" + REPLIES,
            "max_request_bytes": MAX_REQUEST_BYTES,
            "max_reply_bytes": MAX_REPLY_BYTES,
            "max_requests": request_limit,
            "max_concurrent": MAX_CONCURRENT,
        }
    except BaseException as error:
        primary = error
        raise
    finally:
        stop.set()
        thread.join()
        pool.shutdown(wait=True)
        try:
            verify_paths()
        except BaseException as error:
            errors.append(error)
        for future in futures:
            try:
                future.result()
            except BaseException as error:
                errors.append(error)
        os.close(request_fd)
        os.close(reply_fd)
        if errors:
            if primary is not None:
                primary.add_note("file broker final cleanup also failed: " + type(errors[0]).__name__)
            else:
                raise StageGateError("file broker transport failed: " + type(errors[0]).__name__) from errors[0]


# ---- feedback_protocol -------------------------------------------------------------------------
# Closed bounded feedback values; no pickle or caller-selected reconstruction.
#
# Only evaluator-owned scalar/collection data and the two existing cost value types
# cross this transport. Limits and deadline checks refuse late or excessive results;
# bounded local decoding is not an interruptible OS or physical-runtime deadline.

SCHEMA = "merlin.feedback_value.v1"


@dataclass(frozen=True)
class FeedbackValueLimits:
    max_bytes: int = 8 * 1024 * 1024
    max_nodes: int = 65536
    max_depth: int = 64
    max_string_bytes: int = 128 * 1024

    def verify(self):
        ceilings = (8 * 1024 * 1024, 65536, 64, 128 * 1024)
        values = (self.max_bytes, self.max_nodes, self.max_depth, self.max_string_bytes)
        if any(
            type(value) is not int or not 0 < value <= ceiling for value, ceiling in zip(values, ceilings, strict=True)
        ):
            raise StageGateError("component feedback requires bounded byte/node/depth/string limits")
        return self


def check_deadline(deadline):
    if deadline is not None and time.monotonic() >= deadline:
        raise StageGateError("component feedback exceeded result deadline")


class _Budget:
    def __init__(self, limits, deadline=None):
        if type(limits) is not FeedbackValueLimits:
            raise StageGateError("component feedback requires typed result limits")
        self.limits, self.deadline, self.nodes = limits.verify(), deadline, 0

    def visit(self, depth):
        check_deadline(self.deadline)
        self.nodes += 1
        if depth > self.limits.max_depth or self.nodes > self.limits.max_nodes:
            raise StageGateError("component feedback exceeded result depth or node limit")

    def scalar(self, value):
        if type(value) is str:
            if len(value.encode("utf-8")) > self.limits.max_string_bytes:
                raise StageGateError("component feedback exceeded scalar string limit")
        elif type(value) is int:
            if value.bit_length() > 128:
                raise StageGateError("component feedback exceeded scalar integer limit")
        elif type(value) is float and not math.isfinite(value):
            raise StageGateError("component feedback contains nonfinite scalar")
        elif type(value) not in (type(None), bool, float):
            raise StageGateError("component feedback contains unsupported scalar")
        return value


def _encode(value, budget, depth=0):
    budget.visit(depth)
    kind = type(value)
    if kind in (type(None), bool, int, float, str):
        return ["scalar", budget.scalar(value)]
    if kind in (tuple, list):
        if len(value) > budget.limits.max_nodes:
            raise StageGateError("component feedback exceeded collection node limit")
        return ["tuple" if kind is tuple else "list", [_encode(item, budget, depth + 1) for item in value]]
    if kind is dict:
        if len(value) > budget.limits.max_nodes:
            raise StageGateError("component feedback exceeded mapping node limit")
        return [
            "dict",
            [[_encode(key, budget, depth + 1), _encode(item, budget, depth + 1)] for key, item in value.items()],
        ]
    if kind is CycleInterval:
        return ["interval", _encode((value.lo, value.hi, value.provenance, value.missing), budget, depth + 1)]
    # This exact existing evaluator value type has no arbitrary reconstruction hook.
    from merlin.development.backend.optimization.components.analytical_provider import (
        ComponentAnalyticalResults,
    )

    if kind is ComponentAnalyticalResults:
        return ["analytical", _encode((dict(value), value.reports, value.screening), budget, depth + 1)]
    raise StageGateError("component feedback contains unsupported result type")


def _decode(value, budget, depth=0):
    budget.visit(depth)
    if type(value) is not list or len(value) != 2 or type(value[0]) is not str:
        raise StageGateError("component feedback value frame is malformed")
    tag, body = value
    if tag == "scalar":
        return budget.scalar(body)
    if tag in ("tuple", "list", "dict"):
        if type(body) is not list or len(body) > budget.limits.max_nodes:
            raise StageGateError("component feedback collection frame exceeds its limit")
        if tag != "dict":
            values = [_decode(item, budget, depth + 1) for item in body]
            return tuple(values) if tag == "tuple" else values
        result = {}
        for pair in body:
            if type(pair) is not list or len(pair) != 2:
                raise StageGateError("component feedback mapping entry is malformed")
            key, item = (_decode(part, budget, depth + 1) for part in pair)
            try:
                if key in result:
                    raise StageGateError("component feedback mapping has duplicate keys")
                result[key] = item
            except TypeError as error:
                raise StageGateError("component feedback mapping key is not plain hashable data") from error
        return result
    if tag == "interval":
        fields = _decode(body, budget, depth + 1)
        if (
            type(fields) is not tuple
            or len(fields) != 4
            or any(type(item) not in (type(None), float, int) for item in fields[:2])
            or any(type(item) is not tuple or any(type(s) is not str for s in item) for item in fields[2:])
        ):
            raise StageGateError("component feedback interval frame is malformed")
        return CycleInterval(*fields)
    if tag == "analytical":
        from merlin.development.backend.optimization.components.analytical_provider import (
            ComponentAnalyticalResults,
        )

        fields = _decode(body, budget, depth + 1)
        if type(fields) is not tuple or len(fields) != 3 or any(type(item) is not dict for item in fields[:2]):
            raise StageGateError("component feedback analytical frame is malformed")
        return ComponentAnalyticalResults(*fields)
    raise StageGateError("component feedback result tag is outside the closed protocol")


def encode_feedback_value(value, *, limits):
    frame = {"schema": SCHEMA, "value": _encode(value, _Budget(limits))}
    data = json.dumps(frame, ensure_ascii=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    if len(data) > limits.max_bytes:
        raise StageGateError("component feedback exceeded received byte limit")
    return data


def _syntax_depth(data, limits, deadline):
    """Bound nesting before JSON builds objects, including braces inside strings."""
    depth, units, quoted, escaped = 0, 0, False, False
    for offset, byte in enumerate(data):
        if offset % 4096 == 0:
            check_deadline(deadline)
        if quoted:
            if escaped:
                escaped = False
            elif byte == 92:
                escaped = True
            elif byte == 34:
                quoted = False
        elif byte == 34:
            quoted = True
            units += 1
        elif byte in (91, 123):
            depth += 1
            units += 1
            if depth > 4 * limits.max_depth + 8:
                raise StageGateError("component feedback exceeded received syntax depth limit")
        elif byte in (44, 58):
            units += 1
        elif byte in (93, 125):
            depth -= 1
            if depth < 0:
                raise StageGateError("component feedback result syntax is malformed")
        if units > 6 * limits.max_nodes + 16:
            raise StageGateError("component feedback exceeded received syntax node limit")


def decode_feedback_value(data, *, limits, deadline=None):
    budget = _Budget(limits, deadline)
    check_deadline(deadline)
    if type(data) is not bytes or len(data) > limits.max_bytes:
        raise StageGateError("component feedback exceeded received byte limit")
    _syntax_depth(data, limits, deadline)
    try:
        frame = json.loads(data, object_pairs_hook=_unique_object)
        check_deadline(deadline)
        if type(frame) is not dict or set(frame) != {"schema", "value"} or frame["schema"] != SCHEMA:
            raise StageGateError("component feedback result schema differs")
        value = _decode(frame["value"], budget)
        check_deadline(deadline)
        return value
    except (ValueError, TypeError, RecursionError, OverflowError) as error:
        raise StageGateError("component feedback result decoding refused") from error


def receive_feedback_value(path: Path, *, limits, deadline):
    """Read at most the admitted bytes from an owned plain file, then decode data."""
    if type(limits) is not FeedbackValueLimits:
        raise StageGateError("component feedback requires typed result limits")
    limits.verify()
    check_deadline(deadline)
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        info = os.fstat(descriptor)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_uid != os.getuid()
            or info.st_mode & 0o077
            or info.st_size > limits.max_bytes
        ):
            raise StageGateError("component feedback result is not an owned bounded plain file")
        data = bytearray()
        while True:
            check_deadline(deadline)
            block = os.read(descriptor, min(65536, limits.max_bytes + 1 - len(data)))
            if not block:
                break
            data.extend(block)
            if len(data) > limits.max_bytes:
                raise StageGateError("component feedback exceeded received byte limit")
        if len(data) != info.st_size:
            raise StageGateError("component feedback result changed during reception")
        result = decode_feedback_value(bytes(data), limits=limits, deadline=deadline)
        return result, len(data)
    finally:
        os.close(descriptor)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise StageGateError("component feedback result has duplicate object keys")
        result[key] = value
    return result


# ---- feedback_guardian -------------------------------------------------------------------------
# Independent per-call callback child owner; no daemon or serialized authority.
#
# Fork preserves the live issued evaluator context. Only this new guardian becomes
# a subreaper. Its private socket and owner pidfd never enter the callback worker.
# The guardian keeps a custodial lease descriptor until actual ECHILD cleanup, so
# a lost coordinator cannot release the lease while descendant work is still live.

_WORKER_LEASE = None


def has_parent_feedback_lease(path):
    """Recognize only a live exact fork delegation in this callback process.

    This is a host scheduling observation, never a target measurement capability.
    A delegated child's own fork has a different PID and cannot inherit this role.
    """
    return _WORKER_LEASE == (os.getpid(), path)


def _callback_worker(callback, kwargs, destination, limits, control, owner_fd, lease_fd, lease_path):
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    control.close()
    os.close(owner_fd)
    if lease_fd is not None:
        os.close(lease_fd)
        global _WORKER_LEASE
        _WORKER_LEASE = (os.getpid(), lease_path)

    _worker(callback, kwargs, destination, limits)


def run_feedback_guardian(
    callback, kwargs, destination, limits, deadline, control, other_control, owner_fd, lease_fd, lease_path
):
    """Own all direct/adopted descendants until reaped, even after owner loss."""
    other_control.close()
    process, child_fd, error, reason = None, None, None, "setup_failure"
    stop = False

    def stopping(*_args):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, stopping)
    try:
        os.setsid()
        become_child_subreaper()
        protocol.send(control, {"state": "ready", "guardian_pid": os.getpid()})
        while not protocol.readable([control, owner_fd], 0.02):
            if stop or time.monotonic() >= deadline:
                raise TimeoutError("callback admission expired")
        if protocol.exited(owner_fd):
            raise EOFError("callback owner exited before admission")
        command, fds = protocol.receive(control)
        protocol.close_fds(fds)
        if command != {"operation": "start"} or fds:
            raise ValueError("callback guardian requires exact private start admission")
        pid = os.fork()
        if pid == 0:
            try:
                _callback_worker(callback, kwargs, destination, limits, control, owner_fd, lease_fd, lease_path)
            finally:
                os._exit(1)
        process = SimpleNamespace(pid=pid, returncode=None)
        child_fd = protocol.pidfd_open(pid)
        protocol.send(control, {"state": "started", "worker_pid": pid})
        announced = False
        while True:
            if stop or protocol.exited(owner_fd):
                reason = "owner_lost" if protocol.exited(owner_fd) else "cancelled"
                break
            if time.monotonic() >= deadline:
                reason = "worker_deadline"
                break
            if protocol.exited(child_fd):
                reason = "worker_exit"
                break
            if destination.exists() and not announced:
                protocol.send(control, {"state": "result_ready"})
                announced = True
            if not protocol.readable([control], 0.02):
                continue
            command, fds = protocol.receive(control)
            protocol.close_fds(fds)
            if command != {"operation": "cancel"} or fds:
                raise ValueError("callback guardian requires exact private cancellation")
            reason = "cancelled"
            break
    except BaseException as exc:
        error = type(exc).__name__
    finally:
        if child_fd is not None:
            os.close(child_fd)
    attempt = 0
    while True:
        cleanup = reap_owned_children(process, 4)
        receipt = {
            "schema": "merlin.feedback_guardian_cleanup.v1",
            "guardian_pid": os.getpid(),
            "worker_pid": None if process is None else process.pid,
            "worker_exitcode": None if process is None else process.returncode,
            "reason": reason,
            "error": error,
            "cleanup": cleanup,
            "custodial_lease": lease_fd is not None,
            "scope": "invocation descendants only; delegated processes and OS hard deadline UNKNOWN",
        }
        path = destination.parent / ("guardian_cleanup_" + str(attempt) + ".json")
        write_json(path, receipt)
        path.chmod(0o400)
        try:
            protocol.send(control, {"state": "finished", "receipt": receipt})
            # Do not close a socket with an unread cancellation and discard its
            # final packet. The coordinator acknowledges bounded cleanup data.
            acknowledged_until = time.monotonic() + 1
            while time.monotonic() < acknowledged_until and not protocol.exited(owner_fd):
                if not protocol.readable([control], 0.02):
                    continue
                command, fds = protocol.receive(control)
                protocol.close_fds(fds)
                if command == {"operation": "cleanup_received"} and not fds:
                    break
                if command != {"operation": "cancel"} or fds:
                    break
        except (OSError, EOFError):
            pass
        if cleanup["complete"]:
            break
        # Retain the child owner and custodial lock when termination is unresolved.
        # A later actual ECHILD may close the tree; no snapshot grants release.
        attempt += 1
        time.sleep(0.05)
    if lease_fd is not None:
        os.close(lease_fd)
    os.close(owner_fd)
    control.close()
    os._exit(0 if error is None else 1)


# ---- supervised_feedback -----------------------------------------------------------------------
# Closed bounded callback transport with an independent invocation child owner.
#
# Only the guardian owns/reaps callback descendants. Parent reception rejects late
# results; local OS calls and decoding are not interruptible hard OS deadlines.
# Leases stay held on unresolved cleanup. Delegated non-descendant work is UNKNOWN.

_UNRESOLVED = []


def _worker(callback, kwargs, destination, limits):
    os.setsid()
    try:
        data = encode_feedback_value({"result": callback(**kwargs)}, limits=limits)
    except BaseException as exc:
        data = encode_feedback_value({"failure": [type(exc).__name__, str(exc)[:512]]}, limits=FeedbackValueLimits())
    temporary = destination.with_suffix(".partial")
    with temporary.open("xb") as stream:
        os.fchmod(stream.fileno(), 0o600)
        stream.write(data)
    temporary.replace(destination)
    # The guardian owns and reaps this process, including adopted double forks.
    while True:
        signal.pause()


def _packet(control, guardian_fd, deadline):
    def receive():
        try:
            value, fds = protocol.receive(control)
        except (OSError, EOFError) as error:
            raise StageGateError("component callback guardian control lost; cleanup UNKNOWN") from error
        protocol.close_fds(fds)
        if fds:
            raise StageGateError("component callback control has foreign descriptors")
        return value

    while True:
        check_deadline(deadline)
        ready = protocol.readable([control, guardian_fd], min(0.02, deadline - time.monotonic()))
        if control in ready:
            return receive()
        if guardian_fd in ready:
            # Exit readiness can appear after select sampled the socket. Drain a
            # final atomic packet before treating the owner as evidence-less.
            if protocol.readable([control], 0):
                return receive()
            raise StageGateError("component callback guardian exited without cleanup evidence")


def _close_guardian(control, guardian_pid, guardian_fd, receipt=None):
    """Request cancellation; require actual guardian receipt and parent reaping."""
    if receipt is None and not protocol.exited(guardian_fd):
        protocol.send_signal(guardian_fd, signal.SIGCONT)
        try:
            protocol.send(control, {"operation": "cancel"})
        except OSError:
            pass
    deadline = time.monotonic() + 5
    try:
        while receipt is None:
            value = _packet(control, guardian_fd, deadline)
            if value.get("state") == "finished":
                receipt = value.get("receipt")
        protocol.send(control, {"operation": "cleanup_received"})
    except (StageGateError, OSError, EOFError):
        pass
    while time.monotonic() < deadline:
        # The PID cannot recycle before this sole parent reaps its child. A
        # multiprocessing pipe sentinel can be inherited by escaped children;
        # actual pidfd exit and waitpid, rather than pipe EOF, close this owner.
        if protocol.exited(guardian_fd):
            waited, status = os.waitpid(guardian_pid, os.WNOHANG)
            if waited:
                return receipt, True, os.waitstatus_to_exitcode(status)
        time.sleep(0.01)
    return receipt, False, None


def _complete_cleanup(receipt, process_pid, root):
    if not isinstance(receipt, dict) or receipt.get("guardian_pid") != process_pid:
        return False
    retained = root / "guardian_cleanup_0.json"
    try:
        if not retained.is_file() or retained.is_symlink() or mapping_file(retained) != receipt:
            return False
    except (OSError, ValueError, StageGateError):
        return False
    return receipt.get("cleanup", {}).get("complete") is True and receipt["cleanup"].get("error") is None


def bounded_feedback(callback, *, timeout_s, output, kwargs, limits=FeedbackValueLimits(), lease_path=None):
    """Refuse late data; release a parent lease only after actual descendant closure."""
    if not isinstance(timeout_s, (int, float)) or isinstance(timeout_s, bool) or not 0 < timeout_s <= 2400:
        raise StageGateError("component feedback needs a bounded positive wall budget")
    started, primary, result, received_bytes = time.monotonic(), None, None, None
    deadline = started + timeout_s
    if type(limits) is not FeedbackValueLimits:
        raise StageGateError("component feedback requires typed result limits")
    limits.verify()
    output = Path(output)
    if not output.is_absolute() or output.resolve() != output or output.is_symlink():
        raise StageGateError("component feedback requires canonical private output")
    output.mkdir(parents=True, mode=0o700, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="component_worker_", dir=output))
    destination = root / "result.json"
    lease = None
    if lease_path is not None:
        lease_path = Path(lease_path)
        if not lease_path.is_absolute() or lease_path.resolve() != lease_path or lease_path.is_symlink():
            raise StageGateError("component feedback requires an exact parent-owned lease path")
        lease = acquire_host_resource_lease(root, lease_path=lease_path)
        if lease is None:
            raise StageGateError("component feedback parent resource lease is busy")
    control = child_control = owner_fd = None
    try:
        # Probe the actual kernel before any guardian or callback starts.
        owner_fd = protocol.self_pidfd()
        control, child_control = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    except BaseException:
        if lease is not None:
            lease.close()
        if control is not None:
            control.close()
        if child_control is not None:
            child_control.close()
        if owner_fd is not None:
            os.close(owner_fd)
        raise
    guardian_fd, cleanup, reaped, exitcode, worker_pid, guardian_pid = None, None, False, None, None, None
    try:
        guardian_pid = os.fork()
        if guardian_pid == 0:
            try:
                run_feedback_guardian(
                    callback,
                    kwargs,
                    destination,
                    limits,
                    deadline,
                    child_control,
                    control,
                    owner_fd,
                    None if lease is None else lease.fileno(),
                    lease_path,
                )
            finally:
                os._exit(1)
        guardian_fd = protocol.pidfd_open(guardian_pid)
        child_control.close()
        os.close(owner_fd)
        owner_fd = None
        ready = _packet(control, guardian_fd, deadline)
        if ready != {"state": "ready", "guardian_pid": guardian_pid}:
            raise StageGateError("component callback guardian failed independent child-owner setup")
        protocol.send(control, {"operation": "start"})
        launched = _packet(control, guardian_fd, deadline)
        if set(launched) != {"state", "worker_pid"} or launched["state"] != "started":
            raise StageGateError("component callback guardian failed worker admission")
        worker_pid = launched["worker_pid"]
        while not destination.exists():
            if destination.is_symlink():
                raise StageGateError("component feedback worker result is not a plain file")
            if time.monotonic() >= deadline:
                raise StageGateError("component feedback exceeded worker deadline")
            if protocol.readable([control, guardian_fd], 0.02):
                packet = _packet(control, guardian_fd, deadline)
                if packet.get("state") == "finished":
                    cleanup = packet.get("receipt")
                    detail = (
                        "exceeded worker deadline"
                        if cleanup.get("reason") == "worker_deadline"
                        else "ended before a complete result"
                    )
                    raise StageGateError("component feedback worker " + detail)
        check_deadline(deadline)
        result, received_bytes = receive_feedback_value(destination, limits=limits, deadline=deadline)
        if type(result) is not dict or set(result) not in ({"result"}, {"failure"}):
            raise StageGateError("component feedback worker omitted its closed outcome envelope")
        if "failure" in result:
            failure = result["failure"]
            if type(failure) is not list or len(failure) != 2 or any(type(s) is not str for s in failure):
                raise StageGateError("component feedback worker failure frame is malformed")
            raise StageGateError("component feedback worker refused: " + ": ".join(failure))
    except BaseException as error:
        primary = error
    finally:
        if guardian_fd is not None:
            observed, reaped, exitcode = _close_guardian(control, guardian_pid, guardian_fd, cleanup)
            cleanup = observed if observed is not None else cleanup
        closed = reaped and _complete_cleanup(cleanup, guardian_pid, root)
        if closed and worker_pid is None:
            # A callback can pause its coordinator before STARTED is received.
            # The actual closed guardian product still names the reaped worker.
            worker_pid = cleanup.get("worker_pid")
        if lease is not None and (closed or guardian_pid is None):
            lease.close()
        if guardian_pid is not None and not closed:
            _UNRESOLVED.append((guardian_pid, guardian_fd, lease, root))
        elif guardian_fd is not None:
            os.close(guardian_fd)
        if owner_fd is not None:
            os.close(owner_fd)
        control.close()
        child_control.close()
        if primary is None and not closed:
            primary = StageGateError("component feedback descendant cleanup is unresolved; resource lease retained")
        if primary is None and (exitcode != 0 or cleanup.get("error") is not None):
            primary = StageGateError("component feedback guardian failed its private lifecycle protocol")
        if primary is None:
            try:
                check_deadline(deadline)
            except StageGateError as error:
                primary = error
        sources = (
            Path(__file__).resolve(),
            Path(run_feedback_guardian.__code__.co_filename).resolve(),
            Path(owned_children.__file__).resolve(),
            Path(owned_children.reap_owned_children.__code__.co_filename).resolve(),
            Path(protocol.__file__).resolve(),
            Path(protocol.readable.__code__.co_filename).resolve(),
            Path(receive_feedback_value.__code__.co_filename).resolve(),
            Path(acquire_host_resource_lease.__code__.co_filename).resolve(),
        )
        write_json(
            root / "lifecycle.json",
            {
                "schema": "merlin.feedback_worker_lifecycle.v2",
                "status": "refused" if primary else "returned",
                "limits": asdict(limits),
                "elapsed_seconds": time.monotonic() - started,
                "deadline_exceeded": time.monotonic() >= deadline,
                "received_bytes": received_bytes,
                "worker": {
                    "pid": worker_pid,
                    "reaped": closed and worker_pid is not None,
                    "exitcode": None if cleanup is None else cleanup.get("worker_exitcode"),
                },
                "guardian": {"pid": guardian_pid, "reaped": reaped, "exitcode": exitcode},
                "failure": None if primary is None else {"type": type(primary).__name__, "detail": str(primary)[:512]},
                "cleanup": {
                    "status": "COMPLETE" if closed else "UNKNOWN",
                    "method": "subreaper and waitpid ECHILD",
                    "receipt_sha256": sha256_file(root / "guardian_cleanup_0.json") if closed else None,
                },
                "lease_release": {
                    "status": "NOT_REQUESTED" if lease is None else "RELEASED" if closed else "UNKNOWN",
                    "owner": "parent with guardian custodial descriptor",
                    "path": None if lease_path is None else str(lease_path),
                },
                "sources": [(str(path), sha256_file(path)) for path in sources],
                "scope": (
                    "owned invocation descendants only; "
                    "external delegation/namespace isolation/OS hard deadline UNKNOWN"
                ),
            },
        )
        (root / "lifecycle.json").chmod(0o400)
    if primary is not None:
        raise primary
    return result["result"]


def record_feedback_refusal(*, output, action, call_index, candidate, expected_candidate_sha256, error):
    """Retain actual mutation/refusal evidence privately; no restoration authority."""
    output = Path(output)
    if not output.is_absolute() or output.resolve() != output or output.is_symlink():
        raise StageGateError("component feedback refusal requires canonical private output")
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    root = Path(tempfile.mkdtemp(prefix="component_refusal_", dir=output))
    try:
        observed = str(hash_tree(candidate)["sha256"])
    except Exception:  # noqa: BLE001 - an unreadable mutated candidate remains unknown
        observed = None
    write_json(
        root / "refusal.json",
        {
            "schema": "merlin.component_feedback_refusal.v1",
            "action": action,
            "call_index": call_index,
            "failure": {"type": type(error).__name__, "detail": str(error)[:512]},
            "candidate_before_sha256": expected_candidate_sha256,
            "candidate_after_sha256": observed,
            "candidate_changed": (
                observed != expected_candidate_sha256
                if observed is not None and expected_candidate_sha256 is not None
                else "UNKNOWN"
            ),
            "descendant_cleanup": "UNKNOWN",
            "lease_release": "UNKNOWN",
            "scope": "private mutation/refusal observation; no rollback, cleanup or target authority",
        },
    )
    (root / "refusal.json").chmod(0o400)
