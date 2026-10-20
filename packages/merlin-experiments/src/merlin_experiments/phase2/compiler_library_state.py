"""Explicit reviewed-library selection, exact snapshots and sealed transport data.

Records retain evidence, not import authority. A final consumer supplies its own
reviewed host selection; only an already sealed original-parent child transport
may reconstruct that parent's selection without another command-line grant.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from pathlib import Path

from merlin.targetgen.compiler_library import (
    SCHEMA as CONTRACT_SCHEMA,
)
from merlin.targetgen.compiler_library import (
    CompilerLibraryContract,
    CompilerLibrarySelection,
    LibraryMember,
)

from .contracts import StageGateError, document_sha256, exact_tree_record

BINDING_SCHEMA = "merlin.phase2.compiler_library_binding.v1"
MAX_RECORD_BYTES = 4 << 20


def _contract(record):
    if (
        not isinstance(record, Mapping)
        or set(record) != {"schema", "review_id", "public_modules", "members"}
        or record["schema"] != CONTRACT_SCHEMA
        or type(record["public_modules"]) is not list
        or type(record["members"]) is not list
        or not 1 <= len(record["members"]) <= 8192
        or any(type(row) is not dict or set(row) != {"path", "sha256", "module"} for row in record["members"])
    ):
        raise StageGateError("additional compiler library requires its exact reviewed contract")
    try:
        return CompilerLibraryContract(
            record["review_id"],
            tuple(record["public_modules"]),
            tuple(LibraryMember(**row) for row in record["members"]),
        )
    except (ValueError, TypeError, KeyError) as error:
        raise StageGateError("additional compiler library contract is invalid") from error


def restore_selection(record):
    """Reconstruct data only under an independently authorized or sealed owner."""
    if not isinstance(record, Mapping) or set(record) != {"root", "contract_sha256", "contract"}:
        raise StageGateError("additional compiler library selection is incomplete")
    contract = _contract(record["contract"])
    if contract.sha256 != record["contract_sha256"] or not isinstance(record["root"], str):
        raise StageGateError("additional compiler library selection identity changed")
    try:
        return CompilerLibrarySelection(contract, Path(record["root"]))
    except (ValueError, OSError) as error:
        raise StageGateError("additional compiler library source selection is absent or changed") from error


def load_selection(*, root, contract_path, contract_sha256):
    """Host CLI selection; the expected reviewed identity is mandatory."""
    if root is None and contract_path is None and contract_sha256 is None:
        return None
    if root is None or contract_path is None or contract_sha256 is None:
        raise StageGateError("compiler library root, reviewed contract and contract SHA-256 are required together")
    path = Path(contract_path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_RECORD_BYTES:
        raise StageGateError("reviewed compiler library contract is absent, linked or oversized")
    try:
        record = json.loads(path.read_bytes())
    except (ValueError, OSError) as error:
        raise StageGateError("reviewed compiler library contract is unreadable") from error
    return restore_selection({"root": str(Path(root)), "contract_sha256": contract_sha256, "contract": record})


def add_selection_arguments(parser):
    """Explicit host review inputs; candidate records never select a library."""
    parser.add_argument("--compiler-library-root", type=Path, help="reviewed additional compiler source root")
    parser.add_argument("--compiler-library-contract", type=Path, help="exact reviewed compiler_library.v1 JSON")
    parser.add_argument("--compiler-library-contract-sha256", help="expected canonical reviewed contract SHA-256")


def selection_from_arguments(args):
    return load_selection(
        root=args.compiler_library_root,
        contract_path=args.compiler_library_contract,
        contract_sha256=args.compiler_library_contract_sha256,
    )


def validate_binding(record):
    """Pure version/identity checks; saved data does not authorize execution."""
    if (
        not isinstance(record, Mapping)
        or set(record) != {"schema", "origin", "snapshot", "snapshot_sha256", "sha256"}
        or record["schema"] != BINDING_SCHEMA
        or document_sha256({key: value for key, value in record.items() if key != "sha256"}) != record["sha256"]
    ):
        raise StageGateError("additional compiler library binding schema or identity changed")
    for key in ("origin", "snapshot"):
        row = record[key]
        if not isinstance(row, Mapping) or set(row) != {"root", "contract_sha256", "contract"}:
            raise StageGateError("additional compiler library binding selection is incomplete")
        contract = _contract(row["contract"])
        if (
            contract.sha256 != row["contract_sha256"]
            or not isinstance(row["root"], str)
            or not Path(row["root"]).is_absolute()
        ):
            raise StageGateError("additional compiler library binding has an invalid selection identity")
    if (
        record["origin"]["contract"] != record["snapshot"]["contract"]
        or type(record["snapshot_sha256"]) is not str
        or len(record["snapshot_sha256"]) != 64
        or any(character not in "0123456789abcdef" for character in record["snapshot_sha256"])
    ):
        raise StageGateError("additional compiler library snapshot is not the selected reviewed release")
    return copy.deepcopy(dict(record))


def _readonly_snapshot(selection):
    root = selection.root
    expected = {member.path for member in selection.contract.members}
    directories = {parent.as_posix() for name in expected for parent in Path(name).parents if parent != Path(".")}
    for path in (root, *root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if (
            path.is_symlink()
            or path.stat().st_mode & 0o222
            or not (path.is_file() or path.is_dir())
            or path != root
            and relative not in (directories if path.is_dir() else expected)
        ):
            raise StageGateError("additional compiler library snapshot is mutable or has foreign leaves")
    if {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()} != expected:
        raise StageGateError("additional compiler library snapshot lost exact member coverage")
    return exact_tree_record(root)["sha256"]


def freeze_selection(selection, destination):
    """Copy only reviewed leaves, preserving the original selection separately."""
    if type(selection) is not CompilerLibrarySelection:
        raise StageGateError("compiler library snapshot requires an explicit reviewed host owner")
    origin = selection.record()
    destination = Path(destination)
    if (
        destination.exists()
        or destination.is_symlink()
        or not destination.is_absolute()
        or destination.resolve() != destination
        or any(path.is_symlink() for path in destination.parents)
    ):
        raise StageGateError("additional compiler library snapshot needs a fresh absolute destination")
    if destination.is_relative_to(selection.root) or selection.root.is_relative_to(destination):
        raise StageGateError("additional compiler library snapshot overlaps its source release")
    destination.mkdir(parents=True)
    for member in selection.contract.members:
        source, target = selection.root / member.path, destination / member.path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
        target.chmod(0o444)
    for path in sorted(destination.rglob("*"), key=lambda path: len(path.parts), reverse=True):
        if path.is_dir():
            path.chmod(0o555)
    destination.chmod(0o555)
    snapshot = CompilerLibrarySelection(selection.contract, destination)
    if selection.record() != origin:
        raise StageGateError("additional compiler library changed while snapshotting")
    record = {
        "schema": BINDING_SCHEMA,
        "origin": origin,
        "snapshot": snapshot.record(),
        "snapshot_sha256": _readonly_snapshot(snapshot),
    }
    record["sha256"] = document_sha256(record)
    return validate_binding(record)


def verify_binding(record, selection):
    """Match an independently selected host owner, then reopen the exact snapshot."""
    record = validate_binding(record)
    if type(selection) is not CompilerLibrarySelection or selection.record() != record["origin"]:
        raise StageGateError("candidate library handoff requires the same independently reviewed host selection")
    snapshot = restore_selection(record["snapshot"])
    if _readonly_snapshot(snapshot) != record["snapshot_sha256"]:
        raise StageGateError("additional compiler library snapshot identity changed")
    return snapshot


def handoff_selection(handoff):
    """Recheck the same independently reviewed owner at every final consumer."""
    binding = getattr(handoff, "compiler_library_binding", None)
    selection = getattr(handoff, "compiler_library_selection", None)
    host = getattr(handoff, "compiler_library_host_selection", None)
    if binding is None:
        if selection is not None or host is not None:
            raise StageGateError("unbound compiler library on a legacy handoff")
        return None
    checked = verify_binding(binding, host)
    if type(selection) is not CompilerLibrarySelection or selection.record() != checked.record():
        raise StageGateError("final consumer lost the exact candidate compiler library snapshot owner")
    return selection


def freeze_stage(selection, root, diagnostic, refusal):
    """Finish one stage without altering legacy or diagnostic record meaning."""
    if selection is None:
        return None, refusal
    if diagnostic:
        return (
            None,
            refusal
            or "diagnostic additional-library authoring is not consumable by the legacy paired measurement handoff",
        )
    return freeze_selection(selection, root / "control/compiler_library/python"), refusal


def stage_record(selection, binding, diagnostic):
    from .candidate_record import LIBRARY_SCHEMA_VERSION, SCHEMA_VERSION

    if binding is not None:
        return {"schema_version": LIBRARY_SCHEMA_VERSION, "additional_compiler_library": binding}
    return {
        "schema_version": SCHEMA_VERSION,
        **(
            {"additional_compiler_library": selection.record(), "diagnostic_library_stage": True}
            if selection is not None and diagnostic
            else {}
        ),
    }


def verification_inputs(selection, binding):
    return {"compiler_library_selection": selection} if binding is not None else {}
