"""Private wall-time diagnostics; never evidence of custody or numerical success."""
from __future__ import annotations

import contextlib
import contextvars
import functools
import json
import os
import threading
import time
import uuid
from pathlib import Path

_ACTIVE = contextvars.ContextVar('compiler_setup_timing', default=None)
_CAPSULE = contextvars.ContextVar('compiler_setup_capsule', default=(None, None))
_IDENTITY = contextvars.ContextVar('compiler_host_identity', default={})


def _thread_identity():
    return {
        'native_tid': threading.get_native_id(),
        'thread_ident': threading.get_ident(),
        'thread_name': threading.current_thread().name,
    }


@contextlib.contextmanager
def capsule_scope(name, *, tier=None, request_id=None, snapshot_id=None):
    """Exact native scheduling identity, never inferred from requested tiers."""
    token = _CAPSULE.set((name, tier))
    identity = _IDENTITY.set({**_IDENTITY.get(), **{
        key: value for key, value in {'request_id': request_id, 'snapshot_id': snapshot_id}.items()
        if value is not None}})
    try:
        yield
    finally:
        _CAPSULE.reset(token)
        _IDENTITY.reset(identity)


class Timeline:
    def __init__(self, root, source_sha256, *, operation=None, request_id=None, snapshot_id=None):
        self.path = Path(root) / 'setup-timing' / (uuid.uuid4().hex + '.json')
        self.rows = []
        self.source = source_sha256
        self.capsule, self.tier = _CAPSULE.get()
        self.thread = _thread_identity()
        parent = _ACTIVE.get()
        self.parent = str(parent.path) if parent is not None else None
        self.identity = {**_IDENTITY.get(), **{
            key: value for key, value in {'request_id': request_id, 'snapshot_id': snapshot_id,
                                         'operation': operation}.items() if value is not None}}
        self._stack = []

    def write(self):
        # Diagnostics cannot change a verifier's result, including on full disks.
        try:
            self.path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
            temporary = self.path.with_suffix('.tmp')
            temporary.write_text(json.dumps({
                'schema': 'merlin.compiler_setup_timing.v1',
                'pid': os.getpid(), **self.thread, 'source_sha256': self.source,
                **self.identity, 'timeline_id': self.path.stem, 'parent_timeline': self.parent,
                'capsule': self.capsule, 'tier': self.tier,
                'measurement': 'host_wall_seconds_including_waits',
                'authority': 'diagnostic_only', 'stages': self.rows,
                'current_stage': self._stack[-1]['stage'] if self._stack else None,
                'last_stage': self.rows[-1]['stage'] if self.rows else None,
                'last_failure': next((dict(row) for row in reversed(self.rows)
                                      if row['status'] == 'raised'), None),
                'intervals': 'inclusive; nested stages overlap and must not be summed',
            }, sort_keys=True) + '\n')
            temporary.replace(self.path)
        except OSError:
            pass

    @contextlib.contextmanager
    def stage(self, name):
        start = time.monotonic()
        row = {'stage': name, 'status': 'running', 'started_ns': time.time_ns(),
               'stage_id': len(self.rows),
               'parent_stage_id': self._stack[-1]['stage_id'] if self._stack else None,
               **_thread_identity()}
        self.rows.append(row)
        self._stack.append(row)
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
            self._stack.pop()
            self.write()


@contextlib.contextmanager
def selected(root, source_sha256, *, operation=None, request_id=None, snapshot_id=None):
    timeline = Timeline(root, source_sha256, operation=operation, request_id=request_id,
                        snapshot_id=snapshot_id)
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


def measured(name):
    """Observe an existing host call, preserving arguments, return and errors."""
    def decorate(function):
        @functools.wraps(function)
        def call(*args, **kwargs):
            with stage(name):
                return function(*args, **kwargs)
        return call
    return decorate


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
