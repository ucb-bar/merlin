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


_SOURCES_KEPT = 16


def _record(path):
    try:
        record = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    return record if isinstance(record, dict) else {}


def claim(workspace, token, source=None):
    """Called by the single lifecycle owner before dispatching its newest snapshot.

    ``source`` (the snapshot's source identity) is remembered per token so a running grade can tell a
    newer snapshot of DIFFERENT source (supersedes it) from a same-source overlap (complements it)."""
    with _lock(workspace) as path:
        sources = dict(_record(path).get('sources') or {})
        if source is not None:
            sources[str(token)] = str(source)
        sources = dict(sorted(sources.items(), key=lambda item: int(item[0]) if item[0].isdecimal() else 0)[-_SOURCES_KEPT:])
        temporary = path.with_suffix('.next.json')
        temporary.write_text(json.dumps({'token': str(token), 'sources': sources}) + '\n')
        temporary.replace(path)


def release(workspace, token, fallback=None):
    """A claimed snapshot that never graded (deferred or failed to start) stops being the newest."""
    with _lock(workspace) as path:
        record = _record(path)
        if record.get('token') != str(token):
            return
        sources = {k: v for k, v in (record.get('sources') or {}).items() if k != str(token)}
        record = {'token': str(fallback) if fallback is not None else '', 'sources': sources}
        temporary = path.with_suffix('.next.json')
        temporary.write_text(json.dumps(record) + '\n')
        temporary.replace(path)


def superseding_snapshot(workspace, token):
    """The newer claimed token whose source differs from ``token``'s, else ``None``.

    Reads the atomically replaced claim file without the lock. Unknown sources never supersede."""
    if token is None:
        return None
    record = _record(Path(workspace) / 'qa' / '.snapshot-publication.json')
    newest, sources = record.get('token'), record.get('sources') or {}
    token = str(token)
    if (not isinstance(newest, str) or newest == token or not newest.isdecimal() or not token.isdecimal()
            or int(newest) <= int(token)):
        return None
    mine, theirs = sources.get(token), sources.get(newest)
    return newest if mine is not None and theirs is not None and mine != theirs else None


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
