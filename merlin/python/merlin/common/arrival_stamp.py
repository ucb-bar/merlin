#!/usr/bin/env python3
"""ONE arrival-time convention for every agent driver's transcript.

An activity-share plot ("how much of the wall clock went to thinking vs reading vs
writing code vs shell vs waiting?") needs a REAL time for each transcript event.
Until this module existed only the Codex driver had one: its reader stamps every
line it takes off the stream with ``arrived_at`` as the line arrives. The claude
CLI path redirected the child's stdout STRAIGHT into the transcript file, so no
process in the chain ever observed a line and nothing could stamp it -- which is
why the trajectory plot laid messages out by *weighted* time inside a round
instead of by when they actually happened.

This module is the shared implementation of that one convention, so the readers
(``conformance``, ``experiment_tokens``, the plots) consume every driver the same
way:

* the key is ``arrived_at`` and the value is an ISO-8601 UTC timestamp -- exactly
  what :mod:`codex_agent` already writes;
* it is **appended**, never inserted: every field a transcript already carried
  keeps its value AND its position, so a reader that never heard of arrival times
  is unaffected. Adding a field to the graded artifact is only safe if it cannot
  perturb the artifact's existing content, and appending is what makes that true;
* a line that is not a JSON object is passed through **verbatim** rather than
  dropped. An unparseable line is evidence, not noise;
* a line that already carries ``arrived_at`` is left alone, so stamping is
  idempotent and a driver that stamps its own events (Codex) can stream through
  here unchanged.

:func:`stream_stamped` is the launcher: it runs the child with a stdout PIPE,
stamps line by line, and tees the untouched bytes to an optional ``raw_path``
BEFORE interpreting anything -- so a re-serialisation bug or a kill still leaves
the original stream on disk. Timeout handling matches what the call sites relied
on when they used ``subprocess.communicate(timeout=...)``: the whole process
GROUP is killed (the command is ``bash -c '<bwrap ... agent ...>'``, so killing
the outer bash alone leaves the tree alive) and ``subprocess.TimeoutExpired`` is
re-raised for the caller to map.
"""

from __future__ import annotations

import json
import os
import selectors
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

#: The single field name. Identical to the one :mod:`codex_agent` emits.
ARRIVED_AT = "arrived_at"

#: How long a read waits before re-checking the deadline / child liveness.
_POLL_S = 0.25


def now_iso() -> str:
    """The arrival stamp format: ISO-8601, UTC, timezone-aware."""
    return datetime.now(timezone.utc).isoformat()


def stamp_obj(obj: dict, arrived: str) -> dict:
    """Append ``arrived_at`` to ``obj`` unless it already has one. Mutates and returns."""
    if ARRIVED_AT not in obj:
        obj[ARRIVED_AT] = arrived
    return obj


def stamp_line(line: str, arrived: str) -> str:
    """Return ``line`` (no trailing newline) with an arrival stamp appended.

    A blank line, a line that is not JSON, or a JSON value that is not an object
    comes back UNCHANGED -- there is nowhere to put a field on those, and losing
    them would lose evidence.
    """
    text = line.rstrip("\n")
    if not text.strip():
        return text
    try:
        obj = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return text
    if not isinstance(obj, dict):
        return text
    return json.dumps(stamp_obj(obj, arrived))


def _kill_group(proc: subprocess.Popen) -> None:
    """SIGKILL the child's whole process group (bash -> bwrap -> agent)."""
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            proc.kill()
        except OSError:
            return
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        pass


def stream_stamped(
    cmd: list[str],
    *,
    cwd: str | os.PathLike | None,
    transcript: str | os.PathLike,
    stderr_path: str | os.PathLike,
    timeout: float | None = None,
    env: dict | None = None,
    raw_path: str | os.PathLike | None = None,
    now=now_iso,
) -> int:
    """Run ``cmd``, writing its stdout to ``transcript`` one arrival-stamped line at a time.

    Returns the child's return code. Raises :class:`subprocess.TimeoutExpired`
    after killing the process group when ``timeout`` elapses -- the transcript
    (and the raw tee) keep everything that arrived before the kill.
    """
    tpath = Path(transcript)
    tpath.parent.mkdir(parents=True, exist_ok=True)
    epath = Path(stderr_path)
    epath.parent.mkdir(parents=True, exist_ok=True)
    raw_f = None
    if raw_path is not None:
        rp = Path(raw_path)
        rp.parent.mkdir(parents=True, exist_ok=True)
        raw_f = open(rp, "wb")

    deadline = None if timeout is None else time.monotonic() + float(timeout)
    buf = bytearray()
    timed_out = False
    with open(tpath, "w") as tf, open(epath, "w") as ef:

        def _write(line_bytes: bytes) -> None:
            arrived = now()
            if raw_f is not None:                      # durable FIRST, interpreted second
                raw_f.write(line_bytes)
                raw_f.flush()
            text = line_bytes.decode("utf-8", errors="replace")
            had_nl = text.endswith("\n")
            tf.write(stamp_line(text, arrived) + ("\n" if had_nl else ""))
            tf.flush()

        proc = subprocess.Popen(
            cmd, cwd=None if cwd is None else str(cwd), stdout=subprocess.PIPE, stderr=ef,
            env=env, start_new_session=True,
        )
        try:
            sel = selectors.DefaultSelector()
            sel.register(proc.stdout, selectors.EVENT_READ)
            while True:
                if deadline is not None and time.monotonic() >= deadline:
                    timed_out = True
                    break
                wait = _POLL_S if deadline is None else max(
                    0.0, min(_POLL_S, deadline - time.monotonic()))
                if not sel.select(timeout=wait):
                    if proc.poll() is not None and not sel.select(timeout=0):
                        break
                    continue
                try:
                    chunk = os.read(proc.stdout.fileno(), 65536)
                except OSError:
                    break
                if not chunk:
                    break
                buf += chunk
                while True:
                    nl = buf.find(b"\n")
                    if nl < 0:
                        break
                    _write(bytes(buf[:nl + 1]))
                    del buf[:nl + 1]
            if buf:                                    # a final line with no newline
                _write(bytes(buf))
                buf.clear()
        finally:
            try:
                proc.stdout.close()
            except OSError:
                pass
            if raw_f is not None:
                raw_f.close()
            if timed_out:
                _kill_group(proc)
            else:
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    _kill_group(proc)
    if timed_out:
        raise subprocess.TimeoutExpired(cmd, timeout)
    return proc.returncode
