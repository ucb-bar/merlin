"""Private author startup events; observation only, never admission authority.

Unlike capsule setup timelines, this append-only stream must retain an unfinished
start across process termination, with the original provider round/turn identity.
Only static stage names and exception types are recorded, never call arguments.
"""
from __future__ import annotations

import contextvars
import functools
import inspect
import json
import os
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

_OWNER = contextvars.ContextVar('author_startup_owner', default=None)
_PARENT = contextvars.ContextVar('author_startup_parent', default=None)


def observed(function):
    """Bind one existing provider invocation without changing its signature."""
    signature = inspect.signature(function)

    @functools.wraps(function)
    def wrapped(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        owner = (Path(bound.arguments['run_dir']), bound.arguments['rnd'], 0)
        token = _OWNER.set(owner)
        parent = _PARENT.set(None)
        try:
            return function(*args, **kwargs)
        finally:
            _PARENT.reset(parent)
            _OWNER.reset(token)

    return wrapped


def turn(index):
    """Set diagnostic identity only; does not admit or start a provider turn."""
    owner = _OWNER.get()
    if owner is not None:
        _OWNER.set((*owner[:2], index))


def _emit(owner, event):
    # Full disks/unavailable telemetry cannot change the supervised operation.
    try:
        path = owner[0] / 'rounds' / 'author_startup.jsonl'
        path.parent.mkdir(parents=True, exist_ok=True)
        row = {'schema': 'merlin.author_startup_stage.v1', 'round': owner[1],
               'turn': owner[2], 'pid': os.getpid(),
               'at': datetime.now(UTC).isoformat(), 'monotonic_s': time.monotonic(),
               'authority': 'diagnostic_only', **event}
        with path.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row, separators=(',', ':')) + '\n')
    except OSError:
        pass


def call(stage, function, *args, **kwargs):
    """Time the original call once, preserving return values and exceptions."""
    owner = _OWNER.get()
    if owner is None:
        return function(*args, **kwargs)
    identity = uuid.uuid4().hex
    event = {'stage': stage, 'stage_id': identity, 'parent_stage_id': _PARENT.get()}
    started = time.monotonic()
    _emit(owner, {**event, 'event': 'start'})
    parent = _PARENT.set(identity)
    try:
        result = function(*args, **kwargs)
    except BaseException as error:
        _emit(owner, {**event, 'event': 'error', 'wall_s': time.monotonic() - started,
                      'exception_type': type(error).__name__})
        raise
    else:
        _emit(owner, {**event, 'event': 'finish', 'wall_s': time.monotonic() - started})
        return result
    finally:
        _PARENT.reset(parent)
