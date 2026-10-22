"""Live feedback belongs to the newest owned snapshot; archived evidence belongs to each."""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
import fcntl
import json
from pathlib import Path

_OWNER = ContextVar('phase1_snapshot_publication', default=None)


@contextmanager
def _lock(workspace):
    directory = Path(workspace) / 'qa'
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / '.snapshot-publication.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield directory / '.snapshot-publication.json'


def claim(workspace, token):
    """Called by the single lifecycle owner before dispatching its newest snapshot."""
    with _lock(workspace) as path:
        temporary = path.with_suffix('.next.json')
        temporary.write_text(json.dumps({'token': str(token)}) + '\n')
        temporary.replace(path)


@contextmanager
def scope(workspace, token):
    selected = _OWNER.set((str(Path(workspace).resolve()), str(token)))
    try:
        yield
    finally:
        _OWNER.reset(selected)


def publish_live(workspace, action, *, default=None):
    """Serialize the ownership check and write; stale workers retain only their archives."""
    owner = _OWNER.get()
    if owner is None:
        return action()
    if owner[0] != str(Path(workspace).resolve()):
        raise ValueError('snapshot publication workspace differs')
    with _lock(workspace) as path:
        if json.loads(path.read_text()).get('token') != owner[1]:
            return default
        return action()


def current_token():
    owner = _OWNER.get()
    return owner[1] if owner is not None else None
