"""Run an RTL emulator with its diagnostic stream BOUNDED on disk and stopped at its first hardware assertion.

    capture = run_capped(argv, stdout=console, stderr_path=out / "emulator.stderr.txt", timeout_s=...)

WHY A BOUND. An emulator's stderr is its own diagnostics, and a design in a bad state can print one
assertion per simulated cycle for as long as the cycle budget lets it. MEASURED 2026-10-01: one failing
cell candidate wrote 8.5 GB of stderr per program -- a ReservationStation "pipeline stall" assertion
every cycle from the stall to a 60M-cycle budget -- four programs at once, and took the shared scratch
filesystem down to 9.6 GB free. The FIRST and LAST bytes are what anyone reads (the load path and the
first failure at the head; the FINISHED line, the dump status and the stats at the tail), so the head
and tail are kept, the middle is counted and dropped, and the file states how many bytes it elided.

WHY STOP AT THE FIRST ASSERTION. A hardware assertion is the design saying it is in a state its
authors declared impossible; nothing after it is a measurement of the program. Treating it as the
verdict (refused, never correct) ends the run at the assertion instead of the cycle budget: the same
candidate above ran 5.5 h per program to reach a refusal its first assertion had already decided.
Surveyed 2026-10-01 over 400 cell program runs: none of the 384 graded runs printed an assertion; all
12 that did were refused. The emulator is stopped by its own Popen handle (SIGTERM, then SIGKILL after
``grace_s``), never by name; the dump-capable harness turns SIGTERM into a non-zero exit with no dump,
so a stopped run cannot be read as finished.

The assertion spelling is the RTL generator's, not a target's: a FIRRTL ``assert`` lowers to a printf
of ``Assertion failed: <message>`` followed by an ``at <file>:<line>`` location, and the emulator's own
code emitter recognises the same prefix. Nothing here names a target.
"""

from __future__ import annotations

import subprocess
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import IO, Any

__all__ = [
    "ASSERTION_PREFIX",
    "DEFAULT_HEAD_BYTES",
    "DEFAULT_TAIL_BYTES",
    "ELISION_PREFIX",
    "Capture",
    "run_capped",
]

#: How a lowered FIRRTL assertion prints (the line may be preceded by terminal colour codes).
ASSERTION_PREFIX = "Assertion failed"
#: The line this module writes where it dropped the middle of the stream.
ELISION_PREFIX = "[merlin-capture] elided"
DEFAULT_HEAD_BYTES = 4 * 1024 * 1024
DEFAULT_TAIL_BYTES = 4 * 1024 * 1024
#: A line longer than this is not scanned for an assertion (bounds the scanner's memory, not the file).
_MAX_SCAN_LINE = 64 * 1024
_CHUNK = 1 << 16


@dataclass
class Capture:
    """What the run did and what of its stderr was kept."""

    returncode: int | None
    timed_out: bool
    total_bytes: int
    kept_bytes: int
    elided_bytes: int
    head_bytes: int
    tail_bytes: int
    #: ``{"message", "location", "at_byte"}`` of the FIRST assertion, or None.
    assertion: dict[str, Any] | None
    #: True when this module ended the run because of that assertion.
    stopped_on_assertion: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _assertion_message(line: str) -> str | None:
    """The message of an assertion line, or None. Leading non-alphanumeric bytes (colour codes) are
    allowed before the prefix; anything else in front means the prefix is quoted, not printed."""
    before, sep, after = line.partition(ASSERTION_PREFIX)
    if not sep or any(ch.isalnum() for ch in before):
        return None
    return after.lstrip(":").strip()


class _Sink:
    """Streams the head to disk as it arrives; keeps a bounded tail; scans lines for the first assertion."""

    def __init__(self, path: Path, head: int, tail: int, on_assertion: Callable[[], None] | None) -> None:
        self.file: IO[bytes] = path.open("wb")
        self.head, self.tail = int(head), int(tail)
        self.written = 0
        self.total = 0
        self.buffer = bytearray()
        self.carry = bytearray()
        self.assertion: dict[str, Any] | None = None
        self.awaiting_location = False
        self.on_assertion = on_assertion

    def feed(self, chunk: bytes) -> None:
        start = self.total
        self.total += len(chunk)
        room = self.head - self.written
        if room > 0:
            self.file.write(chunk[:room])
            self.file.flush()
            self.written += min(room, len(chunk))
            chunk_tail = chunk[room:]
        else:
            chunk_tail = chunk
        if chunk_tail:
            self.buffer += chunk_tail
            if len(self.buffer) > 2 * self.tail:
                del self.buffer[: len(self.buffer) - self.tail]
        if self.assertion is None or self.awaiting_location:
            self._scan(chunk, start)

    def _scan(self, chunk: bytes, start: int) -> None:
        offset = start - len(self.carry)
        data = bytes(self.carry) + chunk
        *lines, rest = data.split(b"\n")
        for raw in lines:
            line = raw.decode("utf-8", errors="replace").strip()
            if self.awaiting_location:
                if line:
                    if line.startswith("at "):
                        self.assertion["location"] = line[3:].strip()
                    self.awaiting_location = False
            elif self.assertion is None:
                message = _assertion_message(line)
                if message is not None:
                    self.assertion = {"message": message, "location": None, "at_byte": offset}
                    self.awaiting_location = True
                    if self.on_assertion is not None:
                        self.on_assertion()
            offset += len(raw) + 1
            if self.assertion is not None and not self.awaiting_location:
                break
        self.carry = bytearray(rest[-_MAX_SCAN_LINE:])

    def close(self) -> tuple[int, int]:
        """Write the kept tail (with an elision line when bytes were dropped); ``(kept, elided)``."""
        tail = bytes(self.buffer[-self.tail :]) if self.tail else b""
        elided = self.total - self.written - len(tail)
        if elided > 0:
            # Start the kept tail on a whole line so a line-oriented reader never sees half of one.
            cut = tail.find(b"\n")
            if 0 <= cut < len(tail) - 1:
                elided += cut + 1
                tail = tail[cut + 1 :]
            self.file.write(
                (
                    f"\n{ELISION_PREFIX} {elided} bytes of emulator stderr (kept the first {self.written} "
                    f"and the last {len(tail)} of {self.total})\n"
                ).encode()
            )
        self.file.write(tail)
        self.file.close()
        return self.written + len(tail), max(elided, 0)


def run_capped(
    argv: Sequence[str],
    *,
    stdout: IO[bytes] | int | None,
    stderr_path: str | Path,
    env: Mapping[str, str] | None = None,
    cwd: str | Path | None = None,
    timeout_s: float | None = None,
    preexec_fn: Callable[[], None] | None = None,
    head_bytes: int = DEFAULT_HEAD_BYTES,
    tail_bytes: int = DEFAULT_TAIL_BYTES,
    stop_on_assertion: bool = True,
    grace_s: float = 30.0,
    on_start: Callable[[int], None] | None = None,
) -> Capture:
    """Run ``argv`` to completion with stderr bounded to ``head_bytes + tail_bytes`` in ``stderr_path``.

    ``stdout`` is passed through unchanged (a program's console is its own and small). ``timeout_s``
    is the wall-clock budget (the process is killed and ``timed_out`` is set). With
    ``stop_on_assertion`` the first hardware assertion terminates the process by its own handle.
    ``on_start`` receives the child's PID (e.g. to record it beside the run)."""
    process = subprocess.Popen(
        list(argv),
        stdout=stdout,
        stderr=subprocess.PIPE,
        stdin=subprocess.DEVNULL,
        env=dict(env) if env is not None else None,
        cwd=str(cwd) if cwd is not None else None,
        preexec_fn=preexec_fn,
    )
    if on_start is not None:
        on_start(process.pid)
    stopped = threading.Event()

    def stop() -> None:
        if not stop_on_assertion or stopped.is_set():
            return
        stopped.set()
        if process.poll() is None:
            process.terminate()
            timer = threading.Timer(grace_s, lambda: process.poll() is None and process.kill())
            timer.daemon = True
            timer.start()

    sink = _Sink(Path(stderr_path), head_bytes, tail_bytes, stop)
    assert process.stderr is not None

    def drain() -> None:
        stream = process.stderr
        while True:
            chunk = stream.read1(_CHUNK)
            if not chunk:
                break
            sink.feed(chunk)

    reader = threading.Thread(target=drain, name="emulator-stderr", daemon=True)
    reader.start()
    timed_out = False
    try:
        returncode: int | None = process.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
        returncode, timed_out = None, True
    reader.join(timeout=grace_s)
    process.stderr.close()
    kept, elided = sink.close()
    return Capture(
        returncode=returncode,
        timed_out=timed_out,
        total_bytes=sink.total,
        kept_bytes=kept,
        elided_bytes=elided,
        head_bytes=int(head_bytes),
        tail_bytes=int(tail_bytes),
        assertion=sink.assertion,
        stopped_on_assertion=stopped.is_set(),
    )
