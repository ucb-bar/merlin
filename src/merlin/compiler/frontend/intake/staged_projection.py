"""Replay compiler-only source intake without capture or evaluator access.

The projection carries only fields read by compiler admission. Replaying the
same builder checks source, signature, effects and the exact member roster;
hashing a caller-resealed manifest alone cannot supply those relationships.
Original receipt identities remain observations, not authenticated authorship
or numerical qualification.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path

from merlin.common.digest import is_sha256
from merlin.common.digest import sha256_bytes as sha

SCHEMA = "merlin.compile-intake.v1"
MEMBER = "source/intake.json"
POLICY = "merlin.imported-source-policy.v1"
PROFILES = frozenset({"standalone-tensor-kernel", "standalone-effect-kernel"})
_RECEIPT = ("schema", "profile", "materialized_abi")


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def relative(name):
    if not isinstance(name, str):
        raise ValueError("compiler projection member needs a canonical relative path")
    path = Path(name)
    if not path.parts or path.is_absolute() or ".." in path.parts or str(path) != name:
        raise ValueError("compiler projection member needs a canonical relative path")
    return path


def finish_projection(manifest, members, bindings, receipt, *, result_mode=None, provenance_sha256=None):
    """Add explicit replay inputs and a non-transforming imported-source policy."""
    record = {
        "schema": SCHEMA,
        "source_profile": manifest["source_profile"],
        "receipt": {key: receipt[key] for key in _RECEIPT if key in receipt},
        "source_members": bindings,
        "result_mode": result_mode,
        "source_provenance_sha256": provenance_sha256,
    }
    members[MEMBER] = encoded(record)
    members["precision-policy.json"] = encoded({
        "schema": POLICY,
        "source_sha256": manifest["source"]["model_mlir_sha256"],
        "source_profile": manifest["source_profile"],
        "payload_profile": manifest.get("payload_profile"),
        "operation_semantics": "retain-imported-operations",
        "numerical_transforms": [],
        "scope": "imported source contract only; backend admission and numerical qualification required",
    })
    manifest["members"] = {name: sha(raw) for name, raw in members.items()}


def publish(target, manifest, members):
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.stage-", dir=target.parent))
    try:
        for name, raw in members.items():
            path = temporary / relative(name)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
        (temporary / "compile-inputs.json").write_bytes(encoded(manifest))
        if target.exists() or target.is_symlink():
            raise ValueError("compile-input destination appeared during staging")
        os.rename(temporary, target)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return manifest


def read_members(root, manifest):
    members = manifest.get("members")
    if not isinstance(members, dict) or not {MEMBER, "program.mlir", "signature.json"} <= set(members):
        raise ValueError("imported compiler stage lacks its replay projection; restage the capture")
    directories = set()
    for name, digest in members.items():
        path = relative(name)
        if name == "compile-inputs.json" or not isinstance(digest, str) or len(digest) != 64 or any(
                char not in "0123456789abcdef" for char in digest):
            raise ValueError("invalid compiler projection member identity")
        directories.update(str(parent) for parent in path.parents if parent != Path("."))
    seen = set()
    pending = [(root, "")]
    while pending:
        directory, prefix = pending.pop()
        for path in directory.iterdir():
            name = prefix + path.name
            if path.is_symlink():
                raise ValueError("compiler projection cannot contain symlinks: " + name)
            if path.is_dir() and name in directories:
                pending.append((path, name + "/"))
            elif path.is_file() and name in {"compile-inputs.json", *members}:
                seen.add(name)
            else:
                raise ValueError("compiler projection contains an undeclared member: " + name)
    if seen != {"compile-inputs.json", *members}:
        raise ValueError("compiler projection has absent declared members")
    raw_members = {}
    for name, expected in members.items():
        path = root / name
        if name == "weights.safetensors":
            # The standalone schema requires this exact sixteen-byte empty
            # container; do not allocate a caller-sized archive before checking.
            with path.open("rb") as stream:
                raw = stream.read(17)
            if len(raw) != 16:
                raise ValueError("standalone compiler projection needs its empty parameter container")
        else:
            raw = path.read_bytes()
        if sha(raw) != expected:
            raise ValueError("compiler projection member differs from manifest: " + name)
        raw_members[name] = raw
    return raw_members


def load_projection(root, manifest):
    """Reconstruct the entire admitted compiler view, including its typed ABI."""
    from merlin.compiler.frontend.inputs import StagedCompileInputs
    from merlin.compiler.frontend.intake.kernel_compile_inputs import prepare_kernel_inputs
    from merlin.compiler.frontend.linalg_mlir import parse_mlir_text

    raw_members = read_members(root, manifest)
    record = json.loads(raw_members[MEMBER])
    if (not isinstance(record, dict) or set(record) != {
            "schema", "source_profile", "receipt", "source_members", "result_mode", "source_provenance_sha256"}
            or record.get("schema") != SCHEMA or record.get("source_profile") != manifest.get("source_profile")
            or manifest.get("source_profile") not in PROFILES):
        raise ValueError("unsupported compiler source projection")
    receipt, bindings = record["receipt"], record["source_members"]
    source = manifest.get("source")
    if not isinstance(receipt, dict) or set(receipt) - set(_RECEIPT) or not isinstance(bindings, dict):
        raise ValueError("compiler source projection has unknown receipt or member fields")
    if (not isinstance(source, dict) or not is_sha256(source.get("capture_receipt_sha256"))
            or not is_sha256(record["source_provenance_sha256"])):
        raise ValueError("compiler source projection lacks its observed source receipt identity")
    requested = set()

    def checked(name):
        relative(name)
        member = bindings.get(name)
        if not isinstance(member, str) or member not in raw_members or member in {MEMBER, "precision-policy.json"}:
            raise ValueError("compiler source projection lacks a required source binding: " + name)
        requested.add(name)
        return raw_members[member]

    expected, values = prepare_kernel_inputs(checked, receipt, source["capture_receipt_sha256"],
        result_mode=record["result_mode"], provenance_sha256=record["source_provenance_sha256"])
    if set(bindings) != requested:
        raise ValueError("compiler projection contains unused source bindings")
    finish_projection(expected, values, bindings, receipt, result_mode=record["result_mode"],
                      provenance_sha256=record["source_provenance_sha256"])
    if expected != manifest or values != raw_members:
        raise ValueError("compiler source projection differs from independently reconstructed intake")
    signature = json.loads(raw_members["signature.json"])
    policy = json.loads(raw_members["precision-policy.json"])
    module = parse_mlir_text(raw_members["program.mlir"].decode())
    function = next(iter(module.ops))
    spec = json.loads(checked("spec.json"))
    names = [row["name"] for row in spec["buffers"] if row["effect"] != "write"]
    names.extend(row["name"] for row in spec.get("runtime_scalars", []))
    argument_bindings = tuple({"kind": "input", "name": name, "input_index": index,
        "dtype": row["dtype"], "shape": row["shape"],
        "abi_kind": row.get("kind", "tensor"), "mlir_type": str(argument.type)}
        for index, (name, row, argument) in enumerate(zip(names, signature["inputs"], function.args, strict=True)))
    return StagedCompileInputs(
        raw_members["program.mlir"].decode(), signature["entry"], policy, raw_members["precision-policy.json"],
        source["model_mlir_sha256"], manifest["members"]["program.mlir"],
        manifest["members"]["precision-policy.json"], argument_bindings,
        manifest["members"]["weights.safetensors"], source_profile=manifest["source_profile"],
        signature=signature)
