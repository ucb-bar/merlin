"""The self-check broker's lane sizing and worker reaping (:mod:`.selfcheck`).

How many request handlers the dispatcher runs at once and of which kind, the operator's per-request
capsule bound, and how one handler's original process tree is stopped. The broker re-exports every
name, so ``selfcheck._lane_admits`` and the rest stay where callers and tests look them up.
"""

from __future__ import annotations

import json
import os
import signal
import time
from pathlib import Path


def _forwarded_probes():
    from .selfcheck import FORWARDED_PROBES

    return FORWARDED_PROBES


def _dispatch_capacity() -> int:
    from merlin.common.cpu_pool import compute_capacity

    # A shared compute pool arbitrates the handlers' CPU, so it selects the dispatcher by default.
    value = os.environ.get("MERLIN_SELFCHECK_BROKER_JOBS", "2" if compute_capacity() else "1")
    if value not in {"1", "2"}:
        raise ValueError("MERLIN_SELFCHECK_BROKER_JOBS must be 1 or 2")
    return int(value)


#: Named capsules one ordinary request may carry per leased compute CPU under
#: ``MERLIN_SELFCHECK_MAX_CAPSULES=auto``. Each build or simulator step of a capsule leases one CPU
#: from the shared pool, so a request this size keeps the pool busy for several waves without the
#: author splitting one affected operator family across requests (eight CPUs -> 64 capsules).
CAPSULES_PER_COMPUTE_CPU = 8


def _public_corpus_size(capsules_root) -> int | None:
    """Capsules in the public corpus this broker grades, or ``None`` when it cannot be counted."""
    if capsules_root is None:
        return None
    roots = [capsules_root] if isinstance(capsules_root, (str, Path)) else list(capsules_root)
    try:
        return len({path.parent for root in roots for path in Path(root).rglob("capsule.yaml")}) or None
    except OSError:
        return None


def _ordinary_capsule_limit(corpus_size: int | None = None) -> int | None:
    """The operator's per-request capsule bound, or ``None`` when none is selected.

    ``auto`` sizes it from the shared compute pool the request's tiers will lease from
    (``MERLIN_COMPUTE_CPUS``), falling back to this process's CPU affinity when no pool is selected.
    An incomplete pool selection refuses, as it does for every other shared-compute consumer.
    ``auto`` is never below the public corpus (``corpus_size``): sized from the pool alone it was
    8 x 18 = 144 against a 146-capsule suite, which refused the full check the author is told to run.
    An explicit number is the operator's own bound and is used exactly.
    """
    raw = os.environ.get("MERLIN_SELFCHECK_MAX_CAPSULES", "").strip()
    if not raw:
        return None
    if raw == "auto":
        from merlin.common.cpu_pool import grading_cpus

        cpus = len(grading_cpus()) or len(os.sched_getaffinity(0))
        return max(CAPSULES_PER_COMPUTE_CPU * max(1, cpus), corpus_size or 0)
    if not raw.isdecimal() or int(raw) < 1:
        raise ValueError("MERLIN_SELFCHECK_MAX_CAPSULES must be a positive integer or 'auto'")
    return int(raw)


def _feedback_scope_error(request, *, corpus_size: int | None = None):
    """Operator-selected iteration bound; formal qualification uses its own path.

    Under ``auto`` a full-suite request is admitted once the corpus is counted, because the bound
    covers it; an explicit number keeps full-suite feedback disabled, as that operator chose.
    """
    limit = _ordinary_capsule_limit(corpus_size)
    if limit is None:
        return None
    if request.get("action") or request.get("caller_layout") or any(request.get(key) for key in _forwarded_probes()):
        return None
    selected = str(request.get("capsules", "all"))
    names = {name.strip() for name in selected.split(",") if name.strip()}
    full_suite_fits = bool(corpus_size) and os.environ.get("MERLIN_SELFCHECK_MAX_CAPSULES", "").strip() == "auto"
    if (selected.strip() == "all" and not full_suite_fits) or len(names) > limit:
        return (
            f"iteration feedback accepts at most {limit} named capsules per request; select affected capsules. "
            "Full-suite feedback during authoring is disabled by the operator. "
            "The separate final frozen qualification still covers the complete corpus."
        )
    return None


def _ordinary_capacity():
    from merlin.common.cpu_pool import compute_capacity

    # Concurrent numerical handlers; each one's steps lease the shared pool, which is the only limiter.
    pool = compute_capacity()
    raw = os.environ.get("MERLIN_SELFCHECK_ORDINARY_JOBS", str(pool or 1))
    if not raw.isascii() or not raw.isdecimal() or not 1 <= int(raw) <= max(2, pool):
        raise ValueError("MERLIN_SELFCHECK_ORDINARY_JOBS must be 1 or 2, or at most the shared compute pool size")
    return int(raw)


def _process_record(pid):
    try:
        stat = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return int(stat[1]), int(stat[19]), stat[0]
    except (OSError, ValueError, IndexError):
        return None


def _stop_owned_worker(process):
    """Stop only this original process tree, checking birth before each signal."""
    owned = {}
    root = _process_record(process.pid)
    if root is None:
        process.wait(timeout=5)
        return
    owned[process.pid] = root[1]
    records = {}
    for directory in Path("/proc").iterdir():
        if directory.name.isdecimal():
            row = _process_record(int(directory.name))
            if row is not None:
                records[int(directory.name)] = row
    changed = True
    while changed:
        children = {pid: row[1] for pid, row in records.items() if row[0] in owned and pid not in owned}
        owned.update(children)
        changed = bool(children)
    for kind in (signal.SIGTERM, signal.SIGKILL):
        for pid, birth in reversed(tuple(owned.items())):
            current = _process_record(pid)
            if current is not None and current[1] == birth and current[2] != "Z":
                try:
                    os.kill(pid, kind)
                except ProcessLookupError:
                    pass
        end = time.monotonic() + 3
        while time.monotonic() < end:
            if not any(
                (row := _process_record(pid)) is not None and row[1] == birth and row[2] != "Z"
                for pid, birth in owned.items()
            ):
                break
            time.sleep(0.02)
    process.wait(timeout=5)


def _diagnostic_request(request: Path, *, request_bytes=None) -> bool:
    try:
        body = json.loads(request.read_bytes() if request_bytes is None else request_bytes)
        return bool(
            body.get("action") in ("mechanism-question", "reference-analysis")
            or body.get("caller_layout")
            or any(body.get(key) for key in _forwarded_probes())
        )
    except (OSError, ValueError, AttributeError):
        return False  # The unchanged handler reports the malformed request.


def _lane_admits(
    active, *, diagnostic: bool, variant: bool, ordinary_capacity: int, variant_capacity: int, overflow: bool = False
) -> bool:
    """Whether one more handler of this kind may start beside the ``active`` ones.

    Up to ``ordinary_capacity`` numerical requests run at once plus one diagnostic probe. Variant
    requests share the numerical lane, at most ``variant_capacity`` of them, so a queue of variants
    never holds every slot a ``submission/`` check could use. A request skipped here stays queued in
    arrival order without blocking a later request of another kind. ``overflow`` (see
    :func:`_overflow_admits`) lets one numerical request start past the ordinary lanes.
    """
    active = tuple(active)
    if diagnostic:
        return sum(job["diagnostic"] for job in active) < 1
    variants = sum(job["variant"] and not job["diagnostic"] for job in active)
    if variant and variants >= variant_capacity:
        return False
    return overflow or sum(not job["diagnostic"] for job in active) < ordinary_capacity


def _author_cpu_backlog() -> int | None:
    """Author-class leases queued in the shared compute pool now; ``None`` without a pool."""
    from merlin.common.cpu_pool import AUTHOR, queued_waiters

    return queued_waiters(AUTHOR)


def _overflow_admits(active, *, ordinary_capacity: int, author_backlog: int | None, pool: int) -> bool:
    """Whether one author request may start past the ordinary lanes now.

    A handler leases a CPU only for its build and simulator steps and spends the rest of its life
    copying, starting interpreters and waiting on children, so a lane count equal to the CPU count
    keeps only part of the pool busy with author work. Measured on one run: with every lane held and
    96 requests queued at the dispatcher, self-checks used about 8 of 20 reserved CPUs while
    background grading and promotions used 7 and 4 sat idle. A request queued here is invisible to
    the pool's priority order, so those CPUs went to lower classes.

    While no author lease is queued in the pool (``author_backlog == 0``), every admitted author step
    is being served and one more author request would take a free CPU, or the next one a lower class
    releases. Once author leases queue, admission stops and the pool orders the backlog. Work that
    runs without a lease cannot appear as a backlog, so the extra handlers are bounded by the pool's
    own size (``pool``): at most one more handler per reserved CPU.
    """
    if author_backlog != 0 or pool < 1:
        return False
    return sum(not job["diagnostic"] for job in tuple(active)) < ordinary_capacity + pool
