"""Host-pinned normalized model sources, separate from functional qualification.

Only the explicitly supplied spec and MLIR file are read. Provenance pins are
recorded, never interpreted as paths, callbacks, normalization or capture grants.
The host declares completeness; parsing cannot prove that a capture is complete.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import stat

from merlin.frontends.linalg_mlir import parse_mlir_text

OBJECTIVE_DIRECTORY = "_external_objective"
OBJECTIVES_DIRECTORY = "_external_objectives"


def objective_directory(objective_id: str) -> Path:
    """Collision-free grant location for a member of a multi-model external portfolio."""
    _require(isinstance(objective_id, str) and objective_id
             and all(c.isascii() and (c.isalnum() or c in "_-.") for c in objective_id),
             "external objective id must be a simple nonempty identifier")
    return Path(OBJECTIVES_DIRECTORY) / objective_id


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _pin(value):
    _require(isinstance(value, str) and len(value) == 64
             and all(c in "0123456789abcdef" for c in value), "objective pins must be lowercase SHA-256")


def _sha(payload):
    return hashlib.sha256(payload).hexdigest()


def _json(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+"\n").encode()


def _read(path: Path, *, limit: int, digest: str) -> bytes:
    _pin(digest)
    _require(path.is_absolute(), "external objective paths must be explicit absolute paths")
    _require(not any(part == ".." for part in path.parts), "objective paths cannot contain parent traversal")
    # Reject symlink components as well as the leaf; do not follow a surprising
    # alternate source tree before freezing host-granted bytes.
    _require(not any(part.is_symlink() for part in (path, *path.parents)), "external objective path is linked")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as stream:
        info = os.fstat(stream.fileno())
        _require(stat.S_ISREG(info.st_mode), "external objective requires a regular file")
        _require(0 < info.st_size <= limit, "external objective exceeds host input size bound or is empty")
        payload = stream.read(limit+1)
    _require(len(payload) <= limit and _sha(payload) == digest, "external objective bytes differ from the pinned hash")
    return payload


def _unique(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, "duplicate external objective spec key")
        result[key] = value
    return result


@dataclass(frozen=True)
class ExternalObjective:
    """Validated immutable bytes; contains no authority to execute source code."""
    objective_id: str
    source_sha256: str
    spec_sha256: str
    source_bytes: bytes
    record_bytes: bytes
    descriptor_bytes: bytes

    def files(self) -> tuple[tuple[str, bytes], ...]:
        return (("capsule.interface.mlir", self.source_bytes),
                ("capsule.yaml", self.descriptor_bytes), ("objective.json", self.record_bytes))

    def record(self) -> dict:
        return json.loads(self.record_bytes)


def load_external_objective(spec_path: Path, *, spec_sha256: str,
                            max_source_bytes: int) -> ExternalObjective:
    """Read one pinned host declaration and its exact already-normalized source.

    The spec contains ONLY schema/id/entry/source path+hash, nonempty provenance
    pins, and explicit normalized/complete-model assertions. It cannot introduce
    expected outputs, compiler options, scripts or additional readable files.
    """
    _require(type(max_source_bytes) is int and max_source_bytes > 0, "positive host source byte bound required")
    raw = _read(Path(spec_path), limit=1024*1024, digest=spec_sha256)
    spec = json.loads(raw, object_pairs_hook=_unique,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError("non-finite objective spec")))
    fields = {"schema", "id", "entry", "normalized_source", "source_sha256", "provenance_pins",
              "already_normalized", "complete_model"}
    _require(isinstance(spec, dict) and set(spec) == fields, "external objective spec fields are incomplete or unsupported")
    _require(spec["schema"] == "external_full_model_objective_spec_v1", "unsupported external objective schema")
    for key in ("id", "entry"):
        value = spec[key]
        _require(isinstance(value, str) and value and all(c.isascii() and (c.isalnum() or c in "_-." ) for c in value),
                 f"external objective {key} must be a simple nonempty identifier")
    _require(spec["already_normalized"] is True and spec["complete_model"] is True,
             "host must explicitly declare an already-normalized complete model")
    provenance = spec["provenance_pins"]
    _require(isinstance(provenance, dict) and provenance, "external objective requires explicit provenance pins")
    for label, digest in provenance.items():
        _require(isinstance(label, str) and label and all(c.isascii() and (c.isalnum() or c in "_-.") for c in label),
                 "provenance labels are identifiers, never paths")
        _pin(digest)
    _require(isinstance(spec["normalized_source"], str), "normalized source must be an explicit path")
    source = _read(Path(spec["normalized_source"]), limit=max_source_bytes, digest=spec["source_sha256"])
    module = parse_mlir_text(source.decode("utf-8"))
    module.verify()
    functions = [op for op in module.body.block.ops if op.name == "func.func"]
    symbols = {op.sym_name.data: op for op in functions}
    _require(len(symbols) == len(functions), "duplicate source function symbols")
    entry = symbols.get(spec["entry"])
    _require(entry is not None and len(entry.body.blocks) == 1, "normalized objective requires its defined single-block entry")
    _require(bool(entry.function_type.outputs), "external model entry must return declared outputs")
    operations = Counter(op.name for op in entry.walk() if op is not entry)
    _require(any(name != "func.return" for name in operations), "return-only source does not provide an optimization graph")
    for operation in module.walk():
        if operation.name == "func.call":
            callee = operation.callee.root_reference.data
            _require(callee in symbols and bool(symbols[callee].body.blocks),
                     "normalized objective has an unresolved external function call")
    record = {"schema": "external_full_model_objective_v1", "id": spec["id"], "entry": spec["entry"],
        "spec_sha256": spec_sha256, "source_sha256": spec["source_sha256"],
        "provenance_pins": provenance, "source_bytes": len(source),
        "entry_argument_types": [str(arg.type) for arg in entry.body.block.args],
        "entry_output_types": [str(ty) for ty in entry.function_type.outputs],
        "entry_operation_counts": dict(sorted(operations.items())),
        "complete_model": "host_declared_not_independently_proved",
        "already_normalized": "host_declared_no_normalization_executed",
        "phase1_qualification_applies_to_external_model": False,
        "phase1_action": "reuse_unchanged_existing_qualification_and_waivers",
        "numerical_equivalence": "UNPROVEN", "full_model_execution_allowed": False,
        "scope": "separate pinned compile/static optimization objective; bounded changed-mechanism probes only"}
    descriptor = {"id": spec["id"], "interface_mlir": "capsule.interface.mlir", "entry": spec["entry"],
        "required_oracle_tiers": [], "external_objective_record": "objective.json",
        "performance": {"global_objective": True, "qualification": "UNPROVEN_EXTERNAL_MODEL"}}
    return ExternalObjective(spec["id"], spec["source_sha256"], spec_sha256, source, _json(record), _json(descriptor))
