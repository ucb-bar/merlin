"""Passive host-stage wall times; no numerical or timing-model authority."""

from __future__ import annotations

import json
import logging
import os
import time
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def measure(name, document=None, *, events=None):
    """Record completed host work even on errors, without obscuring its result."""
    started = time.perf_counter()
    try:
        yield
    finally:
        elapsed = round(time.perf_counter() - started, 6)
        if isinstance(document, dict):
            timings = document.setdefault("host_stage_timings_s", {})
            # First completed interval initializes this stage; subsequent intervals accumulate.
            if name in timings:
                elapsed_total = timings[name] + elapsed
            else:
                elapsed_total = elapsed
            timings[name] = round(elapsed_total, 6)
        if events is not None:
            try:
                path = Path(events)
                path.parent.mkdir(parents=True, exist_ok=True)
                event = {"stage": name, "wall_s": elapsed, "at_ns": time.time_ns(), "pid": os.getpid()}
                with path.open("a") as stream:
                    stream.write(json.dumps(event, separators=(",", ":")) + "\n")
            except OSError:
                logging.getLogger(__name__).warning(
                    "Host stage timing event unavailable; original grade/error unchanged", exc_info=True
                )
