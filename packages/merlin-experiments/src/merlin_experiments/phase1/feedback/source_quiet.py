"""Launch a due background grade only once the author's source has gone quiet.

An author who edits every couple of minutes otherwise gets most grades started mid-edit and
superseded at once. This is the non-blocking form of ``lifecycle.wait_for_stable_source`` for the
owner loop in ``dual_snapshots``: the same source identity, a quiet period
(``MERLIN_GRADE_QUIET_S``, off by default; 30 s is a typical setting) and a cap on the added delay
(``MERLIN_GRADE_QUIET_MAX_DELAY_S``, default 120 s) so a continuously editing author is still graded.

Off by default because a replay of two recorded runs made it a net loss: the author edits in bursts
(median 15-21 s between edits), so at 30/120 a quiet period removed 1-3 empty grades but raised the
time with no case running by 7-10 points and the edit-to-first-result delay by 11-16 s; 15/60 was
neutral on one run and worse on the other. Enable it for an author that edits less often.

Scheduling only: the grade still captures and verifies its own source. The identity is a full
read of the submission under a CPU lease, so it is sampled at most once per check interval while a
grade waits, and never between grades. When a sample shows a new identity, the time of the newest
metadata change (a stat walk, no lease) dates the edit, so a source that changed long before the
grade fell due launches at once instead of waiting a fresh quiet period.
"""
from __future__ import annotations

import math
import os
from pathlib import Path
import time

#: Longest gap between two identity samples while a due grade waits (``wait_for_stable_source``).
MAX_CHECK_S = 5.0
QUIET_DEFAULT_S = 0.0
MAX_DELAY_DEFAULT_S = 120.0


def _seconds(name: str, default: float) -> float:
    """A non-negative finite number of seconds from ``name``; anything else keeps ``default``."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return value if math.isfinite(value) and value >= 0 else default


def quiet_s() -> float:
    return _seconds("MERLIN_GRADE_QUIET_S", QUIET_DEFAULT_S)


def max_delay_s() -> float:
    return _seconds("MERLIN_GRADE_QUIET_MAX_DELAY_S", MAX_DELAY_DEFAULT_S)


def latest_change_epoch(root) -> float | None:
    """Newest mtime/ctime under ``root``, skipping what the source identity skips; None if unreadable.

    Directories count, so a deletion or rename dates itself through its parent. ctime also moves on
    a restored mtime or an atomic replacement.
    """
    from ..brokers.selfcheck import _DIGEST_IGNORES

    try:
        pending = [os.fspath(root)]
        newest = os.stat(pending[0], follow_symlinks=False)
        latest = max(newest.st_mtime_ns, newest.st_ctime_ns)
        while pending:
            with os.scandir(pending.pop()) as entries:
                for entry in entries:
                    if entry.name in _DIGEST_IGNORES:
                        continue
                    status = entry.stat(follow_symlinks=False)
                    latest = max(latest, status.st_mtime_ns, status.st_ctime_ns)
                    if entry.is_dir(follow_symlinks=False):
                        pending.append(entry.path)
    except OSError:
        return None
    return latest / 1e9


class QuietSource:
    """Decide, poll by poll, whether a due grade may launch now.

    ``ready()`` is called only while a grade is otherwise due. It returns ``(launch, source)``,
    ``source`` being the identity sampled on this call (None when none was): the launch uses it, so
    a settled grade costs no second read. Disabled, every call samples and launches, as before.
    """

    def __init__(self, source_identity, *, quiet_s, max_delay_s, poll_s=1.0, last_change=None,
                 clock=time.monotonic, wall=time.time):
        self.source_identity, self.last_change = source_identity, last_change
        self.quiet, self.max_delay = float(quiet_s), float(max_delay_s)
        self.check_s = min(MAX_CHECK_S, max(float(poll_s), min(self.quiet, self.max_delay) / 3))
        self.clock, self.wall = clock, wall
        self.source = None
        self.sampled_at = None
        self.stable_since = None
        self.waiting_since = None
        #: Seconds a due grade has waited, as of the newest sample that did not settle; else None.
        self.waiting_s = None

    @property
    def enabled(self) -> bool:
        return self.quiet > 0 and self.max_delay > 0

    def _changed_at(self, now: float, lower_bound: float | None) -> float:
        """Monotonic time of the newest edit: metadata if readable, else ``now``."""
        epoch = self.last_change() if self.last_change is not None else None
        if epoch is None:
            return now
        estimate = now - max(0.0, self.wall() - epoch)
        return estimate if lower_bound is None else max(lower_bound, estimate)

    def _sample(self) -> float:
        previous_sample = self.sampled_at
        current = self.source_identity()
        # Dated after the read, which may have waited for a CPU lease.
        now = self.sampled_at = self.clock()
        if current is None:
            self.source, self.stable_since = None, None
        elif current != self.source or self.stable_since is None:
            self.source = current
            self.stable_since = self._changed_at(now, previous_sample)
        return now

    def ready(self):
        if not self.enabled:
            return True, self.source_identity()
        now = self.clock()
        if self.waiting_since is None:
            self.waiting_since = now
        if self.sampled_at is not None and now - self.sampled_at < self.check_s:
            return False, None
        now = self._sample()
        if self.source is None:
            self.waiting_s = None
            return False, None
        quiet = now - self.stable_since >= self.quiet
        capped = now - self.waiting_since >= self.max_delay
        self.waiting_s = None if quiet or capped else now - self.waiting_since
        return (quiet or capped), self.source

    def launched(self) -> None:
        """A grade started: the next one's added delay counts from its own due time."""
        self.waiting_since = self.waiting_s = None


def for_workspace(workspace, source_identity, *, poll_s) -> QuietSource:
    root = Path(workspace) / "submission"
    return QuietSource(source_identity, quiet_s=quiet_s(), max_delay_s=max_delay_s(), poll_s=poll_s,
                       last_change=lambda: latest_change_epoch(root))
