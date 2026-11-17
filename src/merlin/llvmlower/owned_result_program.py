"""Persist the compiler-owned native result program without invocation data.

Records retain actual product/implementation bytes. They do not establish
source equivalence or hermetic toolchain closure. All product filenames are
fixed members of the selected native build; serialized paths are never opened.
"""

import hashlib
import json
from pathlib import Path

from merlin.common.paths import runtime_dir
from merlin.common.strict_json import loads

from . import owned_results
from .owned_result_abi import PROFILE, bridge, prepare
from .owned_results import OwnedResultProgram, _limits, _pin

SCHEMA = "merlin.owned-result-program.v1"
FILENAME = "owned-result-program.json"
_ARTIFACTS = ("source.mlir", "prepared.mlir", "model.ll", "model.so", "owned_result_bridge.c")


def _implementations():
    root = Path(owned_results.__file__).parent
    return {
        "owned_results.py": root / "owned_results.py",
        "owned_result_abi.py": root / "owned_result_abi.py",
        "owned_result_c.py": root / "owned_result_c.py",
        "owned_result_lowering.py": root / "owned_result_lowering.py",
        "owned_result_allocations.py": root / "owned_result_allocations.py",
        "dealloc_lifetime.py": root / "dealloc_lifetime.py",
        "dealloc_promotions.py": root / "dealloc_promotions.py",
        "owned_result_program.py": Path(__file__),
        "owned_results.c": runtime_dir() / "abi/owned_results.c",
        "owned_result_views.c": runtime_dir() / "abi/owned_result_views.c",
        "owned_result_views.h": runtime_dir() / "abi/owned_result_views.h",
    }


def _record(path):
    raw = Path(path).read_bytes()
    return {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}


def save(program):
    if type(program) is not OwnedResultProgram:
        raise ValueError("owned result persistence requires an actual compiler program")
    program.verify()
    root = program.library.parent
    if program.source != root / "source.mlir" or program.library != root / "model.so":
        raise ValueError("owned result products differ from their fixed compiler members")
    record = {
        "schema": SCHEMA,
        "profile": PROFILE,
        "entry": program.entry,
        "limits": {name: getattr(program, name) for name in ("max_bytes", "max_allocations", "max_output_bytes")},
        "artifacts": {name: _record(root / name) for name in _ARTIFACTS},
        "implementations": {name: _record(path) for name, path in _implementations().items()},
    }
    path = root / FILENAME
    with path.open("x") as stream:
        json.dump(record, stream, sort_keys=True, indent=2)
        stream.write("\n")
    load(path).verify()
    return path


def load(path):
    path = Path(path).absolute()
    if path.name != FILENAME or path.resolve() != path or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("owned result record requires its exact ordinary compiler member")
    with path.open("rb") as stream:
        raw = stream.read((1 << 20) + 1)
    if len(raw) > 1 << 20:
        raise ValueError("owned result compiler record exceeds its byte bound")
    record = loads(raw)
    if (
        not isinstance(record, dict)
        or set(record) != {"schema", "profile", "entry", "limits", "artifacts", "implementations"}
        or record["schema"] != SCHEMA
        or record["profile"] != PROFILE
    ):
        raise ValueError("unsupported owned result compiler record")
    limits = record["limits"]
    if not isinstance(limits, dict) or set(limits) != {"max_bytes", "max_allocations", "max_output_bytes"}:
        raise ValueError("owned result compiler record lacks complete allocation limits")
    _limits(**limits)
    root = path.parent
    implementations = _implementations()
    if (
        not isinstance(record["artifacts"], dict)
        or set(record["artifacts"]) != set(_ARTIFACTS)
        or not isinstance(record["implementations"], dict)
        or set(record["implementations"]) != set(implementations)
    ):
        raise ValueError("owned result compiler record changes the complete member roster")
    artifacts = {name: root / name for name in _ARTIFACTS}
    for role, paths in (("artifacts", artifacts), ("implementations", implementations)):
        if any(p.is_symlink() or not p.is_file() or _record(p) != record[role][name] for name, p in paths.items()):
            raise ValueError("owned result compiler products/implementation bytes changed")
    prepared, inputs, outputs = prepare((root / "source.mlir").read_text(), entry=record["entry"])
    if (
        prepared != (root / "prepared.mlir").read_text()
        or bridge(inputs, outputs, entry=record["entry"]) != (root / "owned_result_bridge.c").read_text()
    ):
        raise ValueError("owned result prepared source/bridge differs from original typed source")
    program = OwnedResultProgram(
        root / "source.mlir",
        root / "model.so",
        inputs,
        outputs,
        tuple(_pin(p) for p in (*artifacts.values(), *implementations.values(), path)),
        entry=record["entry"],
        **limits,
    )
    program.verify()
    return program
