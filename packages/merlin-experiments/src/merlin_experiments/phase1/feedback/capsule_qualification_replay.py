"""Reopen an unchanged capsule verifier from its original observation path.

Only an active sealed Python owner can delegate to its byte-identical verifier.
The fresh child keeps that same import guard; no receipt or source path is edited.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

from merlin.common import invocation_record as IR
from merlin.common import owned_process
from merlin_experiments import frozen_python as FP
from merlin_experiments.phase1.feedback import capsule_qualification_handoff as H


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


def _verify_original(pointer, descriptor, selection=None):
    """Fixed child entry; load only bytes already admitted by the sealed owner."""
    _, source = _original_source(pointer)
    verifier = FP._load(Path(source["path"]), H.__name__, source["sha256"])
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
    command = FP.inherited_python_command(
        [
            sys.executable,
            "-m",
            __name__,
            json.dumps({"receipt": pointer, "descriptor": str(descriptor), "selection": selection}),
        ]
    )
    checked = owned_process.run(command, timeout=180, text=True)
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
    if record["verifier_source"] == H.C.pin(Path(H.__file__)):
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
    if record["verifier_source"] == H.C.pin(Path(H.__file__)):
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
    print(json.dumps(_verify_original(arguments["receipt"], arguments["descriptor"], arguments.get("selection"))))
