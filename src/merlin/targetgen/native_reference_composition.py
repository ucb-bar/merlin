"""Byte-only composition of an original receipt and complete integer references.

This record never establishes sealed execution or source closure. Those remain
independent original-owner obligations at ordinary Phase0 source qualification.
Legacy producer receipts retain their exact schema and interpretation.
"""

from __future__ import annotations

import hashlib

from merlin.common import strict_json

SCHEMA = "merlin.native_integer_reference_composition.v1"
BINDING_SCHEMA = "merlin.original_sealed_native_reference_binding.v1"


def validate(document):
    binding = document.get("reference_composition") if isinstance(document, dict) else None
    if (document.get("schema") != SCHEMA or not isinstance(binding, dict)
            or set(binding) != {"schema", "original_receipt", "sealed_source", "added_reference_members"}
            or binding["schema"] != BINDING_SCHEMA):
        raise ValueError("native reference composition has no explicit original receipt binding")
    pin = binding["original_receipt"]
    if not isinstance(pin, dict) or set(pin) != {"bytes", "sha256", "json_utf8"}:
        raise ValueError("native reference composition lacks complete original receipt bytes")
    text = pin["json_utf8"]
    if not isinstance(text, str) or len(text) > 1 << 20:
        raise ValueError("native reference composition original receipt exceeds its metadata bound")
    raw = text.encode("utf-8")
    if len(raw) != pin["bytes"] or hashlib.sha256(raw).hexdigest() != pin["sha256"]:
        raise ValueError("native reference composition original receipt bytes changed")
    original = strict_json.loads(raw, max_bytes=1 << 20)
    if not isinstance(original, dict) or original.get("schema") != "m2m.capture-receipt.v1":
        raise ValueError("native reference composition needs its original producer receipt")
    if set(document) != set(original) | {"reference_composition"}:
        raise ValueError("native reference composition changes original producer metadata membership")
    for key in set(original) - {"schema", "artifacts"}:
        if document[key] != original[key]:
            raise ValueError("native reference composition changes original producer metadata")
    before, after = original.get("artifacts"), document.get("artifacts")
    added = binding["added_reference_members"]
    if (not isinstance(before, dict) or not isinstance(after, dict) or not isinstance(added, list)
            or not added or len(added) != len(set(added)) or "golden.json" not in after
            or any(not isinstance(name, str) or name not in {"golden.json", "golden.npy", "goldens.npz"}
                   for name in added)
            or set(after) != set(before) | set(added) or set(before) & set(added)
            or any(after[key] != value for key, value in before.items())):
        raise ValueError("native reference composition changes or omits original/reference artifacts")
    source = binding["sealed_source"]
    if (not isinstance(source, dict) or set(source) != {"selection_sha256", "capture_tree_sha256",
            "program", "model_sha256", "original_receipt_sha256"}
            or source["original_receipt_sha256"] != pin["sha256"]
            or source["model_sha256"] != before.get("model.mlir", {}).get("sha256")
            or any(not isinstance(source[key], str) or len(source[key]) != 64
                   or any(c not in "0123456789abcdef" for c in source[key])
                   for key in set(source) - {"program"})
            or source["program"] is not None and not isinstance(source["program"], str)):
        raise ValueError("native reference composition lacks its exact selected source identities")
    return original
