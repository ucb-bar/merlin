"""Keep a long-lived process alive when its log cannot be written.

The runner and its brokers write their logs to a file on the run's own volume. When that volume
filled, every ``print`` raised ``OSError`` -- including the ones inside the ``except`` blocks that
were there to report a failed grade -- so the grading threads and the self-check broker died while
the agent kept working, and the run went three hours without feedback. A log line is never worth a
grader: :func:`install` wraps ``sys.stdout``/``sys.stderr`` so a failed write or flush drops that
text, counts it, and says so in one line once a write succeeds again.
"""

from __future__ import annotations

import os
import sys
import threading


class ResilientStream:
    """A text stream whose ``write``/``flush`` report an ``OSError`` as dropped text, never raise it."""

    def __init__(self, stream) -> None:
        self._stream = stream
        self._lock = threading.Lock()
        self.dropped = 0
        self.reason = ""

    def _failed(self, error: OSError, *, dropped: int) -> None:
        with self._lock:
            self.dropped += dropped
            self.reason = os.strerror(error.errno) if error.errno else type(error).__name__

    def write(self, text):
        with self._lock:
            dropped, reason = self.dropped, self.reason
        try:
            if dropped:
                self._stream.write(f"[output] {dropped} writes dropped: {reason}\n")
                with self._lock:
                    self.dropped -= dropped
            return self._stream.write(text)
        except OSError as error:
            self._failed(error, dropped=1)
            return len(text)

    def writelines(self, lines) -> None:
        for line in lines:
            self.write(line)

    def flush(self) -> None:
        try:
            self._stream.flush()
        except OSError as error:
            self._failed(error, dropped=0)  # Buffered text is retried by the next flush.

    def __getattr__(self, name):
        return getattr(self._stream, name)


def install() -> None:
    """Wrap this process's ``sys.stdout`` and ``sys.stderr`` once; child processes are unaffected."""
    for name in ("stdout", "stderr"):
        stream = getattr(sys, name)
        if stream is not None and not isinstance(stream, ResilientStream):
            setattr(sys, name, ResilientStream(stream))


__all__ = ["ResilientStream", "install"]
