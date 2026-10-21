"""Exact raw source regions through the existing Merlin host/device compiler.

Selection grants no numerical credit. Frozen placement, complete private inputs,
all live outputs and the native owned-host auditor remain mandatory.
"""
from __future__ import annotations

import contextlib
import json
import tempfile
from pathlib import Path

import yaml


def selected_source_slice(capsule):
    if capsule.get("kind") != "model_slice" or not (capsule.get("semantic") or {}).get("must_accelerate"):
        return False
    # Source intake declares these raw linalg contracts explicitly. Ordinary
    # component interface capsules retain their existing kernel route.
    return (capsule.get("operation") or {}).get("op") in {"model", "host_island_seam"}


def require_slice_placement(capsule):
    from . import frozen_placement

    root = Path(capsule["__dir__"])
    declaration = yaml.safe_load((root / "capsule.yaml").read_bytes())
    screen = declaration.get("software_screen") or {}
    if screen.get("status") != "admitted":
        gaps = [{key: row.get(key) for key in ("operation", "mlir_operation", "status", "reason")}
                for row in screen.get("decisions", []) if row.get("status") != "admitted"]
        raise ValueError("frozen Phase 0 placement is not admitted: " + json.dumps(gaps, sort_keys=True))
    return frozen_placement.read(declaration, root)


def materialize_source_slice(capsule, *, target, output):
    from .native_model_execution import independent_frozen_source_eligibility
    from .owned_host_leaf_execution import _materialize_typed_source

    root = Path(capsule["__dir__"])
    if any(path.is_symlink() for path in (root, *root.parents)):
        raise ValueError("source slice root is indirect")
    declaration = yaml.safe_load((root / "capsule.yaml").read_bytes())
    if not selected_source_slice(declaration):
        raise ValueError("source slice lacks its frozen Merlin host route")
    source_name = declaration.get("linalg_mlir")
    if not isinstance(source_name, str) or Path(source_name).name != source_name or source_name != declaration.get("interface_mlir"):
        raise ValueError("source slice requires one exact local linalg program")
    source = root / source_name
    if not isinstance(target, str) or not target:
        raise ValueError("source slice needs its trusted target selection")
    require_slice_placement(capsule)
    certificate = independent_frozen_source_eligibility({**declaration, "__dir__": str(root)}, target=target)
    if certificate.get("n_eligible", 0) <= 0:
        raise ValueError("source slice has no independently eligible accelerator work")
    return _materialize_typed_source(declaration, source, certificate, output=output)


@contextlib.contextmanager
def runtime_bundle(capsule, *, target, artifact_root=None):
    from .owned_host_leaf_execution import _verify_source_identity

    with tempfile.TemporaryDirectory(prefix="source-slice-") as temporary:
        output = Path(artifact_root) if artifact_root is not None else Path(temporary) / "bundle"
        bound, certificate, identity = materialize_source_slice(capsule, target=target, output=output)
        provenance = {"bundle": {"artifacts": json.loads(bound.identity_bytes)},
                      "tool": json.loads(bound.validator_identity_bytes), "source": identity,
                      "scope": "exact frozen connected source and all live outputs; no full-model equivalence"}
        def verify():
            _verify_source_identity(capsule, identity)
            bound.verify(bound.root)
        verify()
        yield bound.root, provenance, verify
        verify()
