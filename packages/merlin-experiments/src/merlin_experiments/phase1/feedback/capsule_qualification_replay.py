"""Reopen an unchanged capsule verifier from its original observation path.

Only an active sealed Python owner can delegate to its byte-identical verifier.
The fresh child keeps that same import guard; no receipt or source path is edited.
"""

from __future__ import annotations

import json
import os
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

from merlin.common import invocation_record as IR
from merlin.common import owned_process
from merlin.common.cpu_pool import AUTHOR, PRIORITY_ENV
from merlin_experiments import frozen_python as FP
from merlin_experiments.phase1.feedback import capsule_qualification_handoff as H

from . import capsule_qualification_snapshot as _snapshot
from .capsule_qualification_snapshot import (
    capture_qualification_verification_snapshot as capture_qualification_verification_snapshot,
)
from .capsule_qualification_snapshot import qualification_verification_snapshot as qualification_verification_snapshot
from .capsule_qualification_snapshot import (
    reenter_qualification_verification_snapshot as reenter_qualification_verification_snapshot,
)


def _original_source(pointer):
    record = H.C._read(pointer)
    source = record["verifier_source"]
    current = H.C.pin(Path(H.__file__))
    if (
        FP.active_source_identity() is None
        or source["sha256"] != current["sha256"]
        or H.C.pin(Path(source["path"])) != source
    ):
        raise ValueError("capsule qualification replay requires the same sealed verifier source")
    observed = IR.verify(Path(record["verification"]["path"]))
    H.C._read(record["verification"])
    if observed.get("executable") != source or any(
        {
            "path": str(Path(source["path"]).parent / Path(module.__file__).name),
            "sha256": H.C.pin(Path(module.__file__))["sha256"],
        }
        not in observed["dependencies"]
        for module in (H.C, H.G)
    ):
        raise ValueError("capsule qualification replay differs from the observed verifier or sealed dependencies")
    return record, source


def _verify_original(pointer, descriptor, selection=None, *, verification_snapshot_protocol=None):
    """Fixed child entry; load only bytes already admitted by the sealed owner."""
    record, source = _original_source(pointer)
    if verification_snapshot_protocol not in (None, _snapshot.PROTOCOL):
        raise ValueError("unknown qualification verification snapshot protocol")
    verifier = FP._load(Path(source["path"]), H.__name__, source["sha256"])
    operation = _snapshot.original_payload_scope(record) if verification_snapshot_protocol else nullcontext()
    with operation:
        if selection is None:
            result = verifier.verify(Path(pointer["path"]), descriptor=Path(descriptor))
        else:
            arguments = dict(selection)
            arguments["context"] = SimpleNamespace(
                target=selection["context"]["target"],
                descriptor=Path(selection["context"]["descriptor"]),
                repo=Path(selection["context"]["repo"]),
            )
            arguments["candidate"] = Path(arguments["candidate"])
            run_dir = Path(arguments.pop("run_dir"))
            result = verifier.selected(run_dir, **arguments)
    _original_source(pointer)
    return result


def _delegate(pointer, *, descriptor, selection=None):
    record, _ = _original_source(pointer)
    native = [
        sys.executable,
        "-m",
        __name__,
        json.dumps({"receipt": pointer, "descriptor": str(descriptor), "selection": selection}),
    ]
    command = _snapshot.selected_command(pointer, native)
    separate_snapshot = command is not None
    if command is None:
        command = FP.inherited_python_command(native)
    try:
        # This fixed child only reopens historical qualification evidence.
        # Give its metadata leases author priority without mutating the shared
        # parent environment or escalating unrelated compiler/simulator jobs.
        environment = {**os.environ, PRIORITY_ENV: str(AUTHOR)}
        checked = owned_process.run(command, env=environment, timeout=180, text=True)
    except BaseException as error:
        if separate_snapshot:
            try:
                _snapshot.revalidate_selected(pointer)
                _original_source(pointer)
            except BaseException as post_error:
                error.add_note(f"Qualification replay final check also refused: {post_error}")
        raise
    else:
        if separate_snapshot:
            _snapshot.revalidate_selected(pointer)
            _original_source(pointer)
    if checked.returncode:
        raise ValueError("original capsule verifier refused replay: " + checked.stderr[-2000:])
    result = json.loads(checked.stdout)
    _original_source(pointer)
    if result != {**record, "receipt": pointer}:
        raise ValueError("original capsule verifier returned different qualification evidence")
    return result


def verify(pointer, *, descriptor):
    """Use ordinary H.verify, or exact original-path H in a guarded fresh child."""
    record = H.C._read(pointer)
    if not _snapshot.selected_snapshot_active() and record["verifier_source"] == H.C.pin(Path(H.__file__)):
        return H.verify(Path(pointer["path"]), descriptor=descriptor)
    return _delegate(pointer, descriptor=descriptor)


def selected(run_dir, *, context, candidate, capsules, hidden_capsules, contract, facts):
    """Preserve H.selected's exact new-run input joins in the original verifier."""
    raw = os.environ.get(H.ENV)
    if raw is None:
        return None
    pointer = json.loads(raw)
    if not isinstance(pointer, dict) or set(pointer) != {"path", "sha256"}:
        raise ValueError("imported capsule qualification requires an exact selected pin")
    arguments = dict(
        context=context,
        candidate=candidate,
        capsules=capsules,
        hidden_capsules=hidden_capsules,
        contract=contract,
        facts=facts,
    )
    record = H.C._read(pointer)
    if not _snapshot.selected_snapshot_active() and record["verifier_source"] == H.C.pin(Path(H.__file__)):
        return H.selected(run_dir, **arguments)

    def input_selection():
        return [
            H.C._selection(run_dir, capsules=roots, labels=labels, context=context, contract=contract, facts=facts)
            for roots, labels in ((capsules, {"public", "dev"}), (hidden_capsules, {"hidden"}))
        ] + [H.hash_tree(candidate)]

    before = input_selection()
    serialized = {
        **{key: str(value) if value is not None else None for key, value in arguments.items() if key != "context"},
        "run_dir": str(run_dir),
        "context": {"target": context.target, "descriptor": str(context.descriptor), "repo": str(context.repo)},
    }
    result = _delegate(pointer, descriptor=context.descriptor, selection=serialized)
    if before != input_selection():
        raise ValueError("capsule qualification selection changed during original-verifier replay")
    return result


if __name__ == "__main__":
    arguments = json.loads(sys.argv[1])
    print(
        json.dumps(
            _verify_original(
                arguments["receipt"],
                arguments["descriptor"],
                arguments.get("selection"),
                verification_snapshot_protocol=arguments.get("verification_snapshot_protocol"),
            )
        )
    )
