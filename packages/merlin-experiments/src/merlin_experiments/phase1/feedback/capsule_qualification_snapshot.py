"""Host-selected source custody for the fixed historical qualification child.

The current queue keeps its own source owner. Only the original qualification
replay enters this separately sealed snapshot; no result or callback is retained.
"""

from __future__ import annotations

import contextvars
import hashlib
import json
import os
import sys
import weakref
from contextlib import contextmanager
from pathlib import Path

from merlin.common import bundle_snapshot as BW
from merlin.common import frozen_imports
from merlin_experiments import frozen_python as FP
from merlin_experiments import source_snapshot as SS

_ACTIVE = contextvars.ContextVar("qualification_verification_snapshot", default=None)
_ISSUED = weakref.WeakKeyDictionary()
PROTOCOL = "merlin.capsule_qualification_snapshot.v1"


class _Selection:
    pass


def _pin(path):
    path = Path(path)
    if not path.is_absolute() or path.resolve() != path or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("qualification verification sources must be canonical ordinary files")
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _critical_modules():
    from . import capsule_qualification_handoff as H
    from . import capsule_qualification_replay as R

    return (sys.modules[__name__], R, FP, SS, frozen_imports, H, H.C, H.G)


def _verify(selection, pointer=None):
    if type(selection) is not _Selection or selection not in _ISSUED:
        raise ValueError("qualification verification requires its live host selection")
    state = _ISSUED[selection]
    if os.getpid() != state["process"]:
        raise ValueError("qualification verification selection belongs to a different process")
    if FP.active_source_identity() != state["host_source"] or state["host_source"] is None:
        raise ValueError("qualification verification lost its current sealed host owner")
    if pointer is not None and pointer != state["qualification"]:
        raise ValueError("qualification verification snapshot belongs to a different receipt")
    from . import capsule_qualification_handoff as H

    H.C._read(state["qualification"])
    receipt = SS.verify(state["snapshot"])
    seal, _ = SS.load_seal(state["snapshot"], "snapshot")
    if _pin(seal)["sha256"] != state["seal_sha256"]:
        raise ValueError("qualification verification snapshot seal changed")
    for module, original, selected in state["sources"]:
        if _pin(Path(module.__file__)) != original or _pin(selected)["sha256"] != original["sha256"]:
            raise ValueError("qualification verification child or source guard differs from its host")
        if receipt["files"].get(selected.relative_to(state["snapshot"]).as_posix()) != original["sha256"]:
            raise ValueError("qualification verification source is outside its sealed inventory")
    return state


@contextmanager
def qualification_verification_snapshot(snapshot, *, seal_sha256, qualification):
    """Select only the fixed qualification child, within the current real FP owner.

    Prepare the historical public-profile bytes before sealing. The replay,
    transport, verifier and H/C/G leaves must be this host's exact source bytes.
    Defaults outside this explicit scope keep the inherited FP execution path.
    """
    inherited = _ACTIVE.get()
    foreign_process = False
    if inherited is not None:
        previous = _ISSUED.get(inherited) if type(inherited) is _Selection else None
        if previous is None or previous["process"] == os.getpid():
            raise ValueError("qualification verification snapshots cannot be nested")
        if (
            Path(snapshot) != previous["snapshot"]
            or seal_sha256 != previous["seal_sha256"]
            or qualification != previous["qualification"]
        ):
            raise ValueError("forked qualification reconstruction changed its host-selected inputs")
        # A fork may retain a foreign ContextVar, never its authority. Validate
        # all inputs and issue a fresh local owner BEFORE replacing that value.
        foreign_process = True
    if FP.active_source_identity() is None:
        raise ValueError("qualification verification requires the current sealed host owner")
    if (
        type(qualification) is not dict
        or set(qualification) != {"path", "sha256"}
        or type(seal_sha256) is not str
        or len(seal_sha256) != 64
        or any(c not in "0123456789abcdef" for c in seal_sha256)
    ):
        raise ValueError("qualification verification needs exact snapshot and receipt pins")
    snapshot = Path(snapshot)
    if (
        not snapshot.is_absolute()
        or snapshot.resolve() != snapshot
        or any(p.is_symlink() for p in (snapshot, *snapshot.parents))
    ):
        raise ValueError("qualification verification snapshot must be canonical")
    receipt = SS.verify(snapshot)
    sources = []
    for module in _critical_modules():
        original = _pin(Path(module.__file__))
        relative = module.__name__.replace(".", "/") + ".py"
        matches = [
            snapshot / root / relative for root in receipt["python_roots"] if (snapshot / root / relative).is_file()
        ]
        if len(matches) != 1:
            raise ValueError("qualification verification lacks one exact original child/source guard")
        sources.append((module, original, matches[0]))
    selection = _Selection()
    _ISSUED[selection] = {
        "snapshot": snapshot,
        "seal_sha256": seal_sha256,
        "qualification": json.loads(json.dumps(qualification)),
        "host_source": FP.active_source_identity(),
        "process": os.getpid(),
        "sources": tuple(sources),
    }
    token = None
    try:
        _verify(selection)
        token = _ACTIVE.set(selection)
        try:
            yield
        except BaseException as error:
            try:
                _verify(selection)
            except BaseException as post_error:
                error.add_note(f"Qualification snapshot final check also refused: {post_error}")
            raise
        else:
            _verify(selection)
    finally:
        if token is not None:
            _ACTIVE.reset(token)
            if foreign_process:
                _ACTIVE.set(None)
        _ISSUED.pop(selection, None)


def selected_command(pointer, native_argv):
    """Return a fresh guarded command only for the one fixed private child entry."""
    selection = _ACTIVE.get()
    if selection is None:
        return None
    state = _verify(selection, pointer)
    from . import capsule_qualification_replay as R

    if len(native_argv) != 4 or native_argv[:3] != [sys.executable, "-m", R.__name__]:
        raise ValueError("qualification snapshot permits only the fixed original replay entry")
    arguments = json.loads(native_argv[3])
    if (
        type(arguments) is not dict
        or set(arguments) != {"receipt", "descriptor", "selection"}
        or arguments["receipt"] != pointer
    ):
        raise ValueError("qualification snapshot child lost its exact original receipt")
    arguments["verification_snapshot_protocol"] = PROTOCOL
    command = [*native_argv[:3], json.dumps(arguments)]
    return FP.python_command(state["snapshot"], command, verifier_source=Path(SS.__file__))


def revalidate_selected(pointer):
    selection = _ACTIVE.get()
    if selection is not None:
        _verify(selection, pointer)


def selected_snapshot_active():
    return _ACTIVE.get() is not None


def capture_qualification_verification_snapshot():
    """Capture a live host-issued handle, never a serialized proof or verdict."""
    selection = _ACTIVE.get()
    if selection is not None:
        _verify(selection)
    return selection


@contextmanager
def reenter_qualification_verification_snapshot(selection):
    """Borrow the original lifetime on an explicitly joined broker/eval thread."""
    current = _ACTIVE.get()
    if selection is None:
        if current is not None:
            raise ValueError("qualification replay cannot inherit an unbound snapshot")
        yield
        return
    if current is not None and current is not selection:
        raise ValueError("qualification replay cannot replace another active snapshot")
    _verify(selection)
    token = _ACTIVE.set(selection)
    try:
        try:
            yield
        except BaseException as error:
            try:
                _verify(selection)
            except BaseException as post_error:
                error.add_note(f"Qualification snapshot borrowed final check also refused: {post_error}")
            raise
        else:
            _verify(selection)
    finally:
        _ACTIVE.reset(token)


@contextmanager
def original_payload_scope(record):
    """Use the original live snapshot transaction while H.verify still runs fully.

    Only already hash-bound historical metadata selects the operation. The
    original verifier independently repeats its policy/input joins inside it.
    External qualification products are never cached by this payload scope.
    """
    from . import capsule_qualification_handoff as H

    environment_pin, bundle_pin = record["environment"], record["bundle_manifest"]
    environment_path, bundle_path = Path(environment_pin["path"]), Path(bundle_pin["path"])
    if _pin(environment_path) != environment_pin or _pin(bundle_path) != bundle_pin:
        raise ValueError("original qualification environment or bundle changed")
    environment = H.yaml.safe_load(environment_path.read_bytes())
    bundle = H.yaml.safe_load(bundle_path.read_bytes())
    original = environment["bundle_input_snapshot"]
    workspace = Path(environment["workspace_path"])
    marker_path = Path(original["path"]) / "snapshot.json"
    if BW.bundle_snapshot_root(workspace) != marker_path.parent:
        raise ValueError("original qualification snapshot lost its workspace")
    marker_pin = _pin(marker_path)
    if marker_pin["sha256"] != original["manifest_sha256"]:
        raise ValueError("original qualification snapshot marker changed")
    repo = Path(json.loads(marker_path.read_bytes())["repo"])
    with BW.verified_snapshot_scope(workspace, bundle, repo=repo):
        yield
    if _pin(environment_path) != environment_pin or _pin(bundle_path) != bundle_pin or _pin(marker_path) != marker_pin:
        raise ValueError("original qualification scope inputs changed during replay")
