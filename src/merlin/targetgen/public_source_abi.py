"""Observe neutral public ABI metadata enrichment before ordinary lowering.

The selected capsule remains the source/numerical authority. A derived source
adds only checked function attributes. Original and normalized source, complete
public metadata and the actual helper call stay pinned through candidate use.
The existing emission observer consumes the exact normalized bytes unchanged;
no saved receipt can select source bytes or bypass its ordinary checks.
"""
from __future__ import annotations

import contextlib
import hashlib
import inspect
import json
from pathlib import Path
import uuid

import yaml
from merlin.common import invocation_record
from .contract.linalg_iface import bind_logical_io_metadata, is_linalg_on_tensors


@contextlib.contextmanager
def public_abi_source(capsule, source, generated):
    source = Path(source)
    program = capsule.get("component_program")
    if capsule.get("label") != "public" or not isinstance(program, dict) or program.get("schema") != "merlin.component_program.v2":
        yield source
        return
    text = source.read_text(encoding="utf-8")
    if not is_linalg_on_tensors(text):
        yield source
        return
    original = source.resolve(strict=True)
    if source.is_symlink():
        raise ValueError("public ABI source cannot be a symlink")
    root = Path(capsule["__dir__"]).resolve(strict=True)
    metadata = root / "capsule.yaml"
    if metadata.is_symlink() or not original.is_relative_to(root):
        raise ValueError("public ABI metadata/source escapes its selected capsule")
    if metadata.stat().st_size > 2 << 20 or original.stat().st_size > 16 << 20:
        raise ValueError("public ABI enrichment exceeds its bounded input size")
    metadata_bytes = metadata.read_bytes()
    declared = yaml.safe_load(metadata_bytes)
    keys = ("label", "interface_mlir", "inputs", "component_program")
    if any(capsule.get(key) != declared.get(key) for key in keys):
        raise ValueError("public ABI metadata differs from the original loaded capsule")
    if (root / declared.get("interface_mlir", "capsule.interface.mlir")).resolve(strict=True) != original:
        raise ValueError("public ABI interface differs from the selected source")
    inputs, outputs = declared.get("inputs"), declared["component_program"].get("outputs")
    if type(inputs) is not list or type(outputs) is not list:
        raise ValueError("public ABI has no complete ordered inputs/outputs")
    work = Path(generated).resolve() / "public-source-abi" / uuid.uuid4().hex
    work.mkdir(parents=True, mode=0o700)
    normalized = work / "interface.mlir"
    receipt = work / "source_binding.json"
    with invocation_record.observe_call(
        work, stage="public_logical_io_enrichment", function=bind_logical_io_metadata,
        arguments={"scope": "public ordered ABI names, roles and exact tensor types only"},
        inputs=(original, metadata), outputs=(normalized, receipt),
        dependencies=(Path(__file__), Path(inspect.getsourcefile(bind_logical_io_metadata))),
    ) as call:
        if original.read_text(encoding="utf-8") != text or metadata.read_bytes() != metadata_bytes:
            raise ValueError("public ABI source/metadata changed before observed transformation")
        rendered, report = bind_logical_io_metadata(text, inputs, outputs)
        report.update(original_source=str(original), public_metadata=str(metadata),
                      public_metadata_sha256=hashlib.sha256(metadata_bytes).hexdigest())
        normalized.write_text(rendered, encoding="utf-8")
        receipt.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        call.returned()
    invocation_record.verify(call.path)
    # The normal lower_interface observer continues to check source==staged
    # input and complete target/object custody. This outer owner additionally
    # keeps the source-to-metadata transition live across that invocation.
    try:
        yield normalized
    finally:
        invocation_record.verify(call.path)
        if any(capsule.get(key) != declared.get(key) for key in keys):
            raise ValueError("public ABI metadata changed during compiler execution")
