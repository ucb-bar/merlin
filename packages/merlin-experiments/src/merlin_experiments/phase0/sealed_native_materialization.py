"""Explicit native reference composition from an original verified sealed source.

Original producer/stage bytes remain untouched. A separate ordinary bundle adds
the complete original integer reference to a versioned byte-only receipt, which
still grants no source, arithmetic, target, timing or cohort qualification.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from merlin.common import strict_json
from merlin.targetgen import native_reference_composition as C

from . import sealed_native_program as N


def source_binding(attestation, program, root):
    capture = attestation["capture"]
    return dict(selection_sha256=attestation["selection"]["sha256"],
                capture_tree_sha256=capture["capture_tree_sha256"], program=program,
                model_sha256=hashlib.sha256((root / "model.mlir").read_bytes()).hexdigest(),
                original_receipt_sha256=hashlib.sha256((root / "capture_receipt.json").read_bytes()).hexdigest())


def materialize(attestation, *, program, source_member, destination, max_payload_bytes):
    """Reverify original execution and compose exact bytes under a public bound."""
    if type(max_payload_bytes) is not int or max_payload_bytes < 1:
        raise ValueError("native materialization needs an explicit unchanged public payload bound")
    # The original reader verifies the complete seal first, then refuses the
    # public payload bound before loading arrays, metadata or the source body.
    artifact = N.read(attestation, program=program, source_member=source_member, max_payload_bytes=max_payload_bytes)
    root = N._root(attestation, program)
    N._payload_limit(root, max_payload_bytes)
    raw = (root / "capture_receipt.json").read_bytes()
    original = strict_json.loads(raw, max_bytes=1 << 20)
    if original.get("schema") != "m2m.capture-receipt.v1":
        raise ValueError("native composition selects one unchanged original producer receipt")
    leaves = set(original["artifacts"]) | {"golden.json"}
    leaves.update(path.name for path in root.iterdir() if path.name in {"golden.npy", "goldens.npz"})
    reference_leaves = set(leaves)
    catalog = (getattr(artifact, "meta", {}) or {}).get("framework_catalog")
    if catalog:
        leaves.add("pytorch-opset.json")  # Already owned by the original receipt-bound metadata digest.
    payload, total = {}, 0
    # Refuse source/reference cost BEFORE the original reader loads arrays.
    for name in leaves:
        path = root / name
        if Path(name).name != name or path.is_symlink() or not path.is_file():
            raise ValueError("native composition source leaves are absent or indirect")
        total += path.stat().st_size
        if total > max_payload_bytes:
            raise ValueError("native composition source/payload exceeds the unchanged public bound")
    total = 0
    for name in sorted(leaves):
        path = root / name
        if (Path(name).name != name or path.is_symlink() or not path.is_file()
                or path.stat().st_size > max_payload_bytes - total):
            raise ValueError("native composition source/payload exceeds the unchanged public bound")
        value = path.read_bytes()
        total += len(value)
        identity = dict(bytes=len(value), sha256=hashlib.sha256(value).hexdigest())
        if name == "pytorch-opset.json" and catalog and identity["sha256"] != catalog["sha256"]:
            raise ValueError("native composition metadata-owned framework catalog changed")
        if name in original["artifacts"] and identity != original["artifacts"][name]:
            raise ValueError("native composition original producer artifact changed")
        payload[name] = value
    document = copy.deepcopy(original)
    document["schema"] = C.SCHEMA
    added = sorted(reference_leaves - set(original["artifacts"]))
    if not added:
        raise ValueError("native composition requires a missing complete original reference")
    document["artifacts"].update({name: dict(bytes=len(payload[name]),
        sha256=hashlib.sha256(payload[name]).hexdigest()) for name in added})
    document["reference_composition"] = dict(schema=C.BINDING_SCHEMA,
        original_receipt=dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(), json_utf8=raw.decode("utf-8")),
        sealed_source=source_binding(attestation, program, root), added_reference_members=added)
    C.validate(document)
    payload["capture_receipt.json"] = (json.dumps(document, sort_keys=True, indent=2) + "\n").encode()
    if sum(map(len, payload.values())) > max_payload_bytes:
        raise ValueError("native composition complete receipt/payload exceeds the unchanged public bound")
    destination = Path(destination)
    if (not destination.is_absolute() or ".." in destination.parts or destination.exists()
            or any(path.is_symlink() for path in (destination, *destination.parents))):
        raise ValueError("native composition needs a fresh ordinary task-owned destination")
    destination.mkdir(mode=0o700)
    for name, value in payload.items():
        (destination / name).write_bytes(value)
    return dict(schema=C.SCHEMA, source_binding=document["reference_composition"]["sealed_source"],
                payload_bytes=sum(map(len, payload.values())), added_reference_members=added,
                metadata_owned_members=sorted(leaves - reference_leaves),
                receipt_sha256=hashlib.sha256(payload["capture_receipt.json"]).hexdigest(),
                scope="original sealed bytes reused; no new capture/reference execution or target admission")


def verify_against_original(directory, attestation, *, program, root):
    """Called only inside the original source proof AFTER live seal verification."""
    path = Path(directory) / "source-capture-receipt.json"
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("native reference composition has an indirect source receipt")
    if not path.exists():
        return
    document = strict_json.loads(path.read_bytes(), max_bytes=2 << 20)
    if document.get("schema") != C.SCHEMA:
        return
    C.validate(document)
    binding = document["reference_composition"]
    if (binding["original_receipt"]["json_utf8"].encode() != (root / "capture_receipt.json").read_bytes()
            or binding["sealed_source"] != source_binding(attestation, program, root)):
        raise ValueError("native reference composition differs from the actual original verified source")
    for name in binding["added_reference_members"]:
        source = root / name
        raw = source.read_bytes()
        if (source.is_symlink() or document["artifacts"][name] !=
                dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
                or (Path(directory) / ("source-capture-" + name)).read_bytes() != raw):
            raise ValueError("native reference composition differs from complete original sealed output bytes")
