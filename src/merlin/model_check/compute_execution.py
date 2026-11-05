"""Share the selected compute CPUs across model compilation and simulation steps."""

from __future__ import annotations

import time
from contextlib import contextmanager

from merlin.common.cpu_pool import shared_compute_pool


@contextmanager
def compute_step(timeout_s: float | None = None):
    """Lease only the current native step; queue time consumes its timeout."""
    started = time.monotonic()
    with shared_compute_pool(timeout_s=timeout_s):
        remaining = None if timeout_s is None else timeout_s - (time.monotonic() - started)
        if remaining is not None and remaining <= 0:
            raise TimeoutError("model-check compute-slot wait exhausted the step deadline")
        yield remaining
