"""Explicit content reuse inside one bounded thread and process replay call.

Only caller-selected canonical files participate. Actual process observation
never uses this owner. Complete content reads before and after the call bound
the reuse; every record role and every unselected file remains caller-owned.
This is a verification cost control, not source or stage admission.
"""

from __future__ import annotations

import contextlib
import hashlib
import itertools
import os
import threading
import weakref
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

_ACTIVE = weakref.WeakSet()


def _canonical(path):
    path = Path(path)
    if (
        not path.is_absolute()
        or not path.is_file()
        or path.resolve() != path
        or any(item.is_symlink() for item in (path, *path.parents))
    ):
        raise ValueError("selected replay pins require explicit canonical regular files")
    return path


def _fresh_pin(path):
    path = _canonical(path)
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


@dataclass(frozen=True, eq=False)
class SelectedPinReplay:
    """A live call-local owner, issued only by ``replay_selected_pins``."""

    _pins: MappingProxyType
    _thread: threading.Thread
    _process: int

    def get(self, path):
        if self not in _ACTIVE or threading.current_thread() is not self._thread or os.getpid() != self._process:
            raise ValueError("selected pin replay requires its live same-thread same-process call scope")
        key = str(Path(path))
        if key not in self._pins:
            return None
        _canonical(path)
        return dict(self._pins[key])


@contextlib.contextmanager
def replay_selected_pins(paths, *, max_pins):
    """Share explicit pins only until mandatory complete post-call rereads.

    The cardinality limit precedes file expansion and hashing. No filename,
    mtime, previous scope, nested scope or ambient process supplies a selection.
    Callers still own source membership and file-byte limits. Exceptional exits
    also reread every selected file and revoke the owner.
    """
    if type(max_pins) is not int or max_pins < 1:
        raise ValueError("selected replay needs an explicit positive pin-count limit")
    selected = tuple(itertools.islice(paths, max_pins + 1))
    if not selected or len(selected) > max_pins:
        raise ValueError("selected replay exceeds its complete pin-count limit")
    canonical = tuple(_canonical(path) for path in selected)
    if len(set(canonical)) != len(canonical):
        raise ValueError("selected replay pins must have distinct canonical identities")
    pins = {str(path): _fresh_pin(path) for path in canonical}
    owner = SelectedPinReplay(
        MappingProxyType({path: MappingProxyType(pin) for path, pin in pins.items()}),
        threading.current_thread(),
        os.getpid(),
    )
    _ACTIVE.add(owner)
    try:
        yield owner
    finally:
        _ACTIVE.discard(owner)
        failures = []
        for path in canonical:
            try:
                actual = _fresh_pin(path)
            except (OSError, ValueError) as error:
                failures.append(error)
            else:
                if actual != pins[str(path)]:
                    failures.append(ValueError("selected file bytes differ"))
        if failures:
            raise ValueError("selected replay file content changed during its call scope") from failures[0]


def replayed_pin(owner, path):
    if type(owner) is not SelectedPinReplay:
        raise ValueError("pin replay cannot accept saved or substitute owners")
    return owner.get(path)
