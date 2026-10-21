"""Bounded, complete subprocess trace transport without a plaintext disk copy.

The original invocation record continues to bind both exact encoded streams.
A separate receipt records their complete decoded hash and limits. This module
issues no source, arithmetic, simulator, timing or execution qualification.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import signal
import subprocess
import threading
import time
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path

from merlin.common import invocation_record

SCHEMA = "merlin.complete_gzip_trace.v2"
MEMBER = "trace-transport.json"


@dataclass(frozen=True)
class TraceLimits:
    max_decoded_bytes: int = 16 << 30
    max_compressed_bytes: int = 128 << 20
    max_stdout_bytes: int = 16 << 30
    max_line_bytes: int = 64 << 10

    def __post_init__(self):
        if any(type(v) is not int or v < 1 for v in asdict(self).values()):
            raise ValueError("complete trace requires positive finite byte limits")
        if self.max_line_bytes > self.max_decoded_bytes:
            raise ValueError("complete trace line bound exceeds decoded byte bound")


def _pin(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.resolve() != path:
        raise ValueError("complete trace requires an ordinary canonical file")
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=h.hexdigest())


def capture_gzip_streams(argv, *, stdout_path, stderr_path, env, timeout_s, limits,
                        preexec_fn=None):
    """Run exact argv and compress every output byte with bounded backpressure.

    The caller owns its compute/resource admission and original invocation scope.
    Both output leaves must be fresh. Failures retain partial encoded artifacts
    but never return a complete transport receipt.
    """
    if type(limits) is not TraceLimits or not 0 < timeout_s <= 14_400:
        raise ValueError("complete trace needs typed limits and a bounded timeout")
    stdout_path, stderr_path = Path(stdout_path), Path(stderr_path)
    if stdout_path == stderr_path or any(p.exists() or p.is_symlink() for p in (stdout_path, stderr_path)):
        raise ValueError("complete trace needs distinct fresh output leaves")
    for path in (stdout_path, stderr_path):
        if not path.is_absolute() or path.resolve() != path or not path.parent.is_dir():
            raise ValueError("complete trace outputs must be canonical task-owned leaves")
    deadline = time.monotonic() + timeout_s
    failures, statistics = [], {"streams": {}}
    with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
        process = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env,
                                   start_new_session=True, preexec_fn=preexec_fn)

        def stop():
            # Descendants can retain a pipe after their original parent exits.
            # The fresh process group remains ours until both pipes drain.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

        def compress(role, source, destination, decoded_limit):
            decoded = hashlib.sha256()
            ndecoded = nencoded = 0
            compressor = zlib.compressobj(level=1, wbits=31)
            try:
                assert source is not None
                with source:
                    while block := source.read(1 << 16):
                        ndecoded += len(block)
                        if ndecoded > decoded_limit:
                            raise ValueError(f"complete trace {role} exceeds decoded byte limit")
                        decoded.update(block)
                        encoded = compressor.compress(block)
                        if nencoded + len(encoded) > limits.max_compressed_bytes:
                            raise ValueError("complete trace exceeds compressed byte limit")
                        destination.write(encoded)
                        nencoded += len(encoded)
                    encoded = compressor.flush()
                    if nencoded + len(encoded) > limits.max_compressed_bytes:
                        raise ValueError("complete trace exceeds compressed byte limit")
                    destination.write(encoded)
                    nencoded += len(encoded)
                statistics["streams"][role] = dict(decoded_bytes=ndecoded,
                    decoded_sha256=decoded.hexdigest(), compressed_bytes=nencoded)
            except BaseException as error:
                failures.append(error)
                stop()

        reader = threading.Thread(target=compress, args=("stderr", process.stderr, stderr, limits.max_decoded_bytes),
                                  name="complete-trace-stderr", daemon=True)
        output_reader = threading.Thread(target=compress, args=("stdout", process.stdout, stdout, limits.max_stdout_bytes),
                                         name="complete-trace-stdout", daemon=True)
        reader.start()
        output_reader.start()
        try:
            while process.poll() is None:
                if failures:
                    raise failures[0]
                if time.monotonic() >= deadline:
                    raise subprocess.TimeoutExpired(argv, timeout_s)
                time.sleep(min(0.05, max(0.001, deadline - time.monotonic())))
            reader.join(timeout=max(0, deadline - time.monotonic()))
            output_reader.join(timeout=max(0, deadline - time.monotonic()))
            if reader.is_alive() or output_reader.is_alive():
                raise subprocess.TimeoutExpired(argv, timeout_s)
            if failures:
                raise failures[0]
        finally:
            stop()
            process.wait()
            reader.join(timeout=5)
            output_reader.join(timeout=5)
            if reader.is_alive() or output_reader.is_alive():
                raise RuntimeError("complete trace compressor did not drain after process cleanup")
    statistics.update(encoding="gzip", stdout=_pin(stdout_path), stderr=_pin(stderr_path))
    return subprocess.CompletedProcess(list(argv), process.returncode), statistics


# Preserve the first prospective helper name; its V2 receipt explicitly states
# that both streams are gzip encoded. Existing V1/V2 source owners never use it.
capture_gzip_stderr = capture_gzip_streams


def seal_transport(invocation, statistics, limits):
    """Bind a completed encoded stream to its original verified invocation."""
    if type(limits) is not TraceLimits:
        raise ValueError("complete trace transport requires typed prospective limits")
    invocation = Path(invocation)
    record = invocation_record.verify(invocation)
    for role in ("stdout", "stderr"):
        observed = _pin(record[role]["path"])
        if observed != statistics[role] or observed["sha256"] != record[role]["sha256"]:
            raise ValueError("complete trace changed before transport sealing")
    if record.get("returncode") != 0 or record.get("status") != "completed":
        raise ValueError("complete trace transport cannot seal an unsuccessful invocation")
    document = dict(schema=SCHEMA, invocation=_pin(invocation), encoding="gzip",
                    limits=asdict(limits), statistics=statistics)
    path = invocation.parent / MEMBER
    with path.open("x") as stream:
        json.dump(document, stream, sort_keys=True, indent=2)
        stream.write("\n")
    return path


class SourceTraceTextView:
    """Reopen the original encoded invocation for every complete line traversal."""
    def __init__(self, invocation, *, role="stderr"):
        if role not in {"stdout", "stderr"}:
            raise ValueError("complete trace selects stdout or stderr")
        self.role = role
        self.invocation = Path(invocation)
        self.transport_path = self.invocation.parent / MEMBER
        self.transport_pin = _pin(self.transport_path)
        self._document = json.loads(self.transport_path.read_bytes())
        doc = self._document
        if set(doc) != {"schema", "invocation", "encoding", "limits", "statistics"} or doc["schema"] != SCHEMA or doc["encoding"] != "gzip":
            raise ValueError("unsupported complete trace transport")
        self.limits = TraceLimits(**doc["limits"])
        self._revalidate()

    def _revalidate(self):
        if _pin(self.transport_path) != self.transport_pin or _pin(self.invocation) != self._document["invocation"]:
            raise ValueError("complete trace transport or original invocation changed")
        record = invocation_record.verify(self.invocation)
        if record.get("returncode") != 0 or record.get("status") != "completed":
            raise ValueError("complete trace has no successful original invocation")
        for role in ("stdout", "stderr"):
            pin = _pin(record[role]["path"])
            if pin != self._document["statistics"][role] or pin["sha256"] != record[role]["sha256"]:
                raise ValueError("complete trace encoded stream changed")
        return record

    def splitlines(self):
        """Yield every UTF-8 line; exhaustion verifies the complete decoded hash."""
        record = self._revalidate()
        statistics = self._document["statistics"]
        stream_statistics = statistics["streams"][self.role]
        if any(statistics[role]["bytes"] > self.limits.max_compressed_bytes for role in ("stdout", "stderr")):
            raise ValueError("complete trace encoded payload exceeds its prospective limits")
        digest, size = hashlib.sha256(), 0
        decoded_limit = self.limits.max_decoded_bytes if self.role == "stderr" else self.limits.max_stdout_bytes
        with gzip.open(record[self.role]["path"], "rb") as stream:
            while line := stream.readline(self.limits.max_line_bytes + 1):
                if len(line) > self.limits.max_line_bytes:
                    raise ValueError("complete trace exceeds line byte limit")
                size += len(line)
                if size > decoded_limit:
                    raise ValueError("complete trace exceeds decoded byte limit")
                digest.update(line)
                yield line.decode("utf-8", errors="strict").rstrip("\r\n")
        if size != stream_statistics["decoded_bytes"] or digest.hexdigest() != stream_statistics["decoded_sha256"]:
            raise ValueError("complete trace decoded bytes differ from the sealed transport")
        self._revalidate()


def source_trace_text_view(invocation, *, role="stderr"):
    return SourceTraceTextView(invocation, role=role)
