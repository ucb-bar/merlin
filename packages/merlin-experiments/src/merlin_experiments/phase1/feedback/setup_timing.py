"""Private wall-time diagnostics; never evidence of custody or numerical success."""
from __future__ import annotations

import contextlib
import contextvars
import json
import os
import threading
import time
import uuid
from pathlib import Path

_ACTIVE = contextvars.ContextVar('compiler_setup_timing', default=None)
_CAPSULE = contextvars.ContextVar('compiler_setup_capsule', default=(None, None))


def _thread_identity():
    return {
        'native_tid': threading.get_native_id(),
        'thread_ident': threading.get_ident(),
        'thread_name': threading.current_thread().name,
    }


@contextlib.contextmanager
def capsule_scope(name, *, tier=None):
    """Exact native scheduling identity, never inferred from requested tiers."""
    token = _CAPSULE.set((name, tier))
    try:
        yield
    finally:
        _CAPSULE.reset(token)


class Timeline:
    def __init__(self, root, source_sha256):
        self.path = Path(root) / 'setup-timing' / (uuid.uuid4().hex + '.json')
        self.rows = []
        self.source = source_sha256
        self.capsule, self.tier = _CAPSULE.get()
        self.thread = _thread_identity()

    def write(self):
        # Diagnostics cannot change a verifier's result, including on full disks.
        try:
            self.path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
            temporary = self.path.with_suffix('.tmp')
            temporary.write_text(json.dumps({
                'schema': 'merlin.compiler_setup_timing.v1',
                'pid': os.getpid(), **self.thread, 'source_sha256': self.source,
                'capsule': self.capsule, 'tier': self.tier,
                'measurement': 'host_wall_seconds_including_waits',
                'authority': 'diagnostic_only', 'stages': self.rows,
            }, sort_keys=True) + '\n')
            temporary.replace(self.path)
        except OSError:
            pass

    @contextlib.contextmanager
    def stage(self, name):
        start = time.monotonic()
        row = {'stage': name, 'status': 'running', 'started_ns': time.time_ns(),
               **_thread_identity()}
        self.rows.append(row)
        self.write()
        try:
            yield
        except BaseException as error:
            row.update(status='raised', exception_type=type(error).__name__)
            raise
        else:
            row['status'] = 'returned'
        finally:
            row.update(elapsed_s=time.monotonic() - start, finished_ns=time.time_ns())
            self.write()


@contextlib.contextmanager
def selected(root, source_sha256):
    timeline = Timeline(root, source_sha256)
    token = _ACTIVE.set(timeline)
    try:
        yield timeline
    finally:
        _ACTIVE.reset(token)


@contextlib.contextmanager
def stage(name):
    timeline = _ACTIVE.get()
    if timeline is None:
        yield
    else:
        with timeline.stage(name):
            yield


@contextlib.contextmanager
def timed_context(name, manager):
    """Time context boundaries while preserving exception/suppression semantics."""
    with contextlib.ExitStack() as stack:
        with stage(name + '.enter'):
            value = stack.enter_context(manager)
        try:
            yield value
        except BaseException:
            import sys
            with stage(name + '.exit'):
                suppressed = stack.__exit__(*sys.exc_info())
            if not suppressed:
                raise
        else:
            with stage(name + '.exit'):
                stack.close()
