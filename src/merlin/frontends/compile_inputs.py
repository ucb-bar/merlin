"""Stage the declared compiler inputs from a materialized model capture.

This host-owned boundary is deliberately narrower than a capture bundle. The
compiler receives parsed IR, external weights, source correspondence, and a
static signature. Runtime inputs and reference outputs stay with the evaluator.
The capture producer's receipt establishes byte identity for copied members;
it does not establish source closure or numerical equivalence.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

from xdsl.dialects.builtin import StringAttr

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.xdsl_dialects._common import text as module_text

_COPIED = ("weights.safetensors", "weights.safetensors.manifest.json", "frontend-trace.json")
_SOURCE = "model.mlir"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_member(directory: Path, name: str) -> bytes:
    member = directory / name
    if member.is_symlink() or not member.is_file():
        raise ValueError(f"capture member is absent or a symlink: {name}")
    return member.read_bytes()


def _checked_member(directory: Path, receipt: dict, name: str) -> bytes:
    entry = (receipt.get("artifacts") or {}).get(name)
    if not isinstance(entry, dict):
        raise ValueError(f"capture receipt lacks {name}")
    raw = _read_member(directory, name)
    if entry.get("bytes") != len(raw) or entry.get("sha256") != _sha(raw):
        raise ValueError(f"capture member differs from receipt: {name}")
    return raw


def stage_compile_inputs(capture_dir: str | Path, destination: str | Path) -> dict:
    """Publish a new, immutable compiler-only directory for one captured model.

    The destination must be absent. The caller owns capture permission and the
    evaluator's separate runtime bundle. This function never reads the runtime
    input or golden members, even if the producer receipt lists them.
    """
    source = Path(capture_dir).absolute()
    target = Path(destination).absolute()
    if (
        source.is_symlink()
        or any(parent.is_symlink() for parent in source.parents)
        or target.exists()
        or target.is_symlink()
    ):
        raise ValueError("capture must be a real directory and destination must be absent")
    if not source.is_dir():
        raise ValueError("capture directory is absent")
    receipt_raw = _read_member(source, "capture_receipt.json")
    receipt = json.loads(receipt_raw)
    if receipt.get("schema") != "m2m.capture-receipt.v1":
        raise ValueError("unsupported capture receipt")
    materialized_abi = receipt.get("materialized_abi") or {}
    if materialized_abi.get("complete") is not True or receipt.get("lifted_constants"):
        raise ValueError("capture has incomplete ABI or unbound lifted constants")
    raw = _checked_member(source, receipt, _SOURCE)
    copied = {name: _checked_member(source, receipt, name) for name in _COPIED}
    metadata = json.loads(_checked_member(source, receipt, "meta.json"))
    if metadata.get("ok") is not True or metadata.get("opaque") != 0:
        raise ValueError("capture has opaque or failed frontend operations")
    if metadata.get("frontend_trace", {}).get("sha256") != _sha(copied["frontend-trace.json"]):
        raise ValueError("frontend trace differs from capture metadata")
    trace = json.loads(copied["frontend-trace.json"])
    if trace.get("status") != "complete" or trace.get("mlir", {}).get("sha256") != _sha(raw):
        raise ValueError("frontend trace does not bind complete captured IR")
    module = parse_mlir_text(raw.decode("utf-8"))
    old_reference = module.attributes.get("prov.weights_file")
    if not isinstance(old_reference, StringAttr) or old_reference.data != metadata.get("weights"):
        raise ValueError("parsed IR weights reference differs from capture metadata")
    module.attributes["prov.weights_file"] = StringAttr("weights.safetensors")
    module.verify()
    staged_text = module_text(module, generic=True)
    parsed = parse_mlir_text(staged_text)
    parsed.verify()
    if parsed.attributes.get("prov.weights_file") != StringAttr("weights.safetensors"):
        raise ValueError("staged IR lost its relative weights reference")
    staged = staged_text.encode("utf-8")
    signature = {
        "schema": "merlin.compile-signature.v1",
        "entry": metadata.get("func_name"),
        "inputs": metadata.get("input_abi"),
        "outputs": metadata.get("output_abi"),
        "shape_scope": "static_captured_signature",
        "capture_dtype": metadata.get("dtype"),
    }
    if (
        not isinstance(signature["entry"], str)
        or not isinstance(signature["inputs"], list)
        or not isinstance(signature["outputs"], list)
        or materialized_abi.get("inputs") != len(signature["inputs"])
    ):
        raise ValueError("capture has an incomplete invocation signature")
    signature_raw = (json.dumps(signature, sort_keys=True, indent=2) + "\n").encode()
    manifest = {
        "schema": "merlin.compile-inputs.v1",
        "scope": "compiler_only_static_capture; source_closure_and_numerical_admission_unverified",
        "source": {
            "capture_receipt_sha256": _sha(receipt_raw),
            "model_mlir_sha256": _sha(raw),
        },
        "members": {
            "program.mlir": _sha(staged),
            "signature.json": _sha(signature_raw),
            **{name: _sha(value) for name, value in copied.items()},
        },
        "excluded_evaluator_members": ["inputs.npz", "inputs.json", "golden.npy", "golden.json", "extra.npz"],
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.stage-", dir=target.parent))
    try:
        for name, value in {"program.mlir": staged, "signature.json": signature_raw, **copied}.items():
            (temporary / name).write_bytes(value)
        (temporary / "compile-inputs.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
        if target.exists() or target.is_symlink():
            raise ValueError("compile-input destination appeared during staging")
        os.rename(temporary, target)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return manifest
