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
import os
import stat
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path

from merlin_experiments.phase1.brokers import compiler_call_channel as IO

from .contracts import StageGateError

PROTOCOL = "merlin.phase2_broker_file_channel.v1"
MAX_REQUEST_BYTES = IO.MAX_REQUEST_BYTES
MAX_REPLY_BYTES = 2 * IO.MAX_REQUEST_BYTES
MAX_REQUESTS = IO.MAX_PENDING_ENTRIES
BAD_REQUEST_ALLOWANCE = 16
MAX_CONCURRENT = 1  # Preserve the original completion-index receipt order.
REPLIES = "broker-replies"


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
    from .broker import Broker

    if type(broker) is not Broker:
        raise StageGateError("file transport requires the original broker")
    if type(broker.max_calls) is not int or not 1 <= broker.max_calls <= MAX_REQUESTS - BAD_REQUEST_ALLOWANCE:
        raise StageGateError("file transport requires a bounded original action-call budget")
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
