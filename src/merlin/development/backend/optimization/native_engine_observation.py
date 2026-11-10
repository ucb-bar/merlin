"""Byte-bound, source-deletion-safe native RTL observation bundles.

This is an evidence container, not an RTL synthesizer or a target capability
grant. A target example/OOT runner supplies the numerical bytes and the exact
native build members; this module freezes and verifies those bytes without
knowing accelerator names or command protocols.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

SCHEMA = "merlin.native-engine-observation.v1"


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _member_name(name: str) -> Path:
    relative = Path(name)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts or relative.as_posix() != name:
        raise ValueError(f"unsafe member path {name!r}")
    return relative


def _pin(root: Path, name: str) -> dict[str, Any]:
    root = root.resolve(strict=True)
    relative = _member_name(name)
    path = root
    for part in relative.parts:
        path = path / part
        if path.is_symlink():
            raise ValueError(f"symlinked bundle member or parent: {name!r}")
    if not path.is_file() or not path.resolve(strict=True).is_relative_to(root):
        raise ValueError(f"missing or symlinked bundle member {name!r}")
    return {"path": name, "sha256": _sha(path), "n_bytes": path.stat().st_size}


def seal(
    *,
    output: str | Path,
    members: Mapping[str, str | Path],
    case: Mapping[str, Any],
    selection: Mapping[str, Any],
    build_commands: list[dict[str, Any]],
    scope: str,
) -> Path:
    """Freeze explicit inputs/tools/driver/output bytes, then validate the bundle.

    Members use relative destination names as keys. The caller must include
    ``selection.json``, ``raw.fir``, ``core.hw.mlir``, ``model.so``,
    ``state.json``, ``case/golden.bin``, and ``case/observed.bin``. The
    selection's own raw/core digests are checked.
    Build commands document invocation provenance; the evidence is the frozen
    bytes and independently compared outputs, not an inference that command
    text alone proves any tool executed.
    """
    root = Path(output).resolve()
    if root.exists():
        raise FileExistsError(f"observation output already exists: {root}")
    required = {
        "selection.json",
        "raw.fir",
        "core.hw.mlir",
        "model.so",
        "state.json",
        "case/golden.bin",
        "case/observed.bin",
    }
    if not required.issubset(members):
        raise ValueError(f"missing observation members: {sorted(required - members.keys())}")
    if not scope.strip():
        raise ValueError("native-engine scope must be explicit")
    if "producer.py" in members:
        raise ValueError("producer.py is reserved for the exact observation verifier source")
    members = {**members, "producer.py": Path(__file__)}
    source_pins = selection.get("sources") or {}
    for name, source_key in (("raw.fir", "firrtl"), ("core.hw.mlir", "core_hw")):
        expected = (source_pins.get(source_key) or {}).get("sha256")
        if not expected or _sha(Path(members[name])) != expected:
            raise ValueError(f"{name} disagrees with source selection")
    shape = case.get("shape")
    if not isinstance(shape, list) or len(shape) != 2 or not all(isinstance(x, int) and x > 0 for x in shape):
        raise ValueError("case must declare a positive two-dimensional shape")
    count = shape[0] * shape[1]
    width = case.get("element_bytes")
    if not isinstance(width, int) or width <= 0 or count != case.get("compared_elements"):
        raise ValueError("case dtype or compared-element count differs")
    for name in ("case/golden.bin", "case/observed.bin"):
        if Path(members[name]).stat().st_size != count * width:
            raise ValueError(f"{name} does not contain all {count} elements")
    if Path(members["case/golden.bin"]).read_bytes() != Path(members["case/observed.bin"]).read_bytes():
        raise ValueError("native output differs from independent golden")
    if not isinstance(case.get("observations"), dict):
        raise ValueError("case must declare concrete interface observations")
    root.mkdir(parents=True)
    for name, origin in sorted(members.items()):
        target = root / _member_name(name)
        target.parent.mkdir(parents=True, exist_ok=True)
        if Path(origin).is_symlink():
            raise ValueError(f"member is a symlink: {origin}")
        source = Path(origin).resolve(strict=True)
        if not source.is_file():
            raise ValueError(f"member is not an ordinary source file: {origin}")
        shutil.copyfile(source, target)
    pins = [_pin(root, name) for name in sorted(members)]
    body = {
        "schema_version": SCHEMA,
        "scope": scope,
        "case": dict(case),
        "build_commands": build_commands,
        "members": pins,
    }
    (root / "observation.json").write_text(json.dumps(body, sort_keys=True, indent=2) + "\n")
    verify(root)
    return root / "observation.json"


def verify(root: str | Path) -> dict[str, Any]:
    """Validate every frozen byte; no original source/checkout is consulted."""
    root = Path(root).resolve(strict=True)
    receipt_path = root / "observation.json"
    if receipt_path.is_symlink() or not receipt_path.is_file():
        raise ValueError("native observation receipt is missing or symlinked")
    doc = json.loads(receipt_path.read_text())
    if doc.get("schema_version") != SCHEMA:
        raise ValueError("unknown native observation schema")
    pins = doc.get("members")
    if not isinstance(pins, list) or not all(
        isinstance(row, dict) and isinstance(row.get("path"), str) for row in pins
    ):
        raise ValueError("invalid native observation members")
    names = {row["path"] for row in pins}
    required = {
        "selection.json",
        "raw.fir",
        "core.hw.mlir",
        "model.so",
        "state.json",
        "case/golden.bin",
        "case/observed.bin",
        "producer.py",
    }
    if len(pins) != len(names) or not required.issubset(names):
        raise ValueError("invalid or duplicate native observation members")
    for row in pins:
        actual = _pin(root, row["path"])
        if actual != row:
            raise ValueError(f"frozen member changed: {row['path']}")
    selection = json.loads((root / "selection.json").read_text())
    for name, source_key in (("raw.fir", "firrtl"), ("core.hw.mlir", "core_hw")):
        if _sha(root / name) != selection["sources"][source_key]["sha256"]:
            raise ValueError(f"frozen {name} disagrees with selected source")
    case = doc["case"]
    if (
        not isinstance(case.get("shape"), list)
        or len(case["shape"]) != 2
        or not all(isinstance(x, int) and x > 0 for x in case["shape"])
    ):
        raise ValueError("invalid native case shape")
    count = case["shape"][0] * case["shape"][1]
    width = case["element_bytes"]
    if count != case["compared_elements"] or not isinstance(width, int) or width <= 0:
        raise ValueError("invalid native case element ABI")
    if len((root / "case/golden.bin").read_bytes()) != count * width:
        raise ValueError("golden element count changed")
    if (root / "case/golden.bin").read_bytes() != (root / "case/observed.bin").read_bytes():
        raise ValueError("observed native output differs from independent golden")
    return doc
