"""Host-only semantic search over the exact frozen Phase 0 and model inputs.

This receipt is a diagnostic sidecar. It is never an agent grant, a portfolio
objective, a timing observation, or qualification evidence.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import stat
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from merlin.development.backend.optimization import contracts as C
from merlin.development.backend.optimization.stage_inputs import StageE2ESentinel
from merlin.target.spec.instructions import validate_normalized_instruction_model
from merlin.targetgen.contract.linalg_iface import is_linalg_on_tensors, parse_linalg_mlir
from merlin.targetgen.semantic_search import SearchLimits, search_linalg_inventory

SCHEMA = "merlin.phase2.host_semantic_search.v1"
_MODEL = Path("software/instruction-semantics.json")
_MAX_MLIR_BYTES = 8_000_000
_MAX_PAYLOAD_OPERATIONS = 512
_LIMITS = SearchLimits(max_candidates=32, timeout_ms=50, solver_timeout_ms=50)


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _selected_file(directory: Path, name: object) -> Path:
    if not isinstance(name, str) or not name or Path(name).name != name or name in {".", ".."}:
        raise ValueError("capsule has no safe declared source MLIR member")
    path = directory / name
    if path.is_symlink() or not path.is_file():
        raise ValueError("declared source MLIR is absent or linked")
    return path


def inspect_member(
    sentinel: StageE2ESentinel, *, target: str, capsule_linked: bool, frozen_grants: Sequence[Path]
) -> dict[str, Any]:
    """Inspect one selected frozen capsule without consulting a live checkout."""
    row: dict[str, Any] = {
        "capsule": sentinel.capsule,
        "capsule_sha256": sentinel.capsule_sha256,
        "status": "unavailable",
        "qualification": "selection and modeled allocation only; no target code or execution proof",
    }
    if not capsule_linked:
        row.update(status="refused", reason="external objective has no frozen Phase 0 capsule linkage")
        return row
    try:
        source = Path(sentinel.frozen_source_path)
        if source.is_symlink() or not source.is_dir() or len(source.parents) < 2:
            raise ValueError("frozen model capsule is absent or linked")
        observed = C.exact_tree_record(source)
        if observed["sha256"] != sentinel.capsule_sha256:
            raise ValueError("frozen model capsule bytes changed after selection")
        descriptor = C.mapping_file(source / "capsule.yaml", yaml_file=True)
        if (
            descriptor.get("kind") != "model"
            or descriptor.get("label") != "public"
            or descriptor.get("name") != sentinel.capsule
        ):
            raise ValueError("selected capsule is not the declared public model")
        trace = descriptor.get("frontend_trace")
        source_name = (
            trace.get("source_mlir")
            if isinstance(trace, Mapping) and trace.get("source_mlir")
            else descriptor.get("interface_mlir")
        )
        mlir_path = _selected_file(source, source_name)
        if mlir_path.stat().st_size > _MAX_MLIR_BYTES:
            row.update(status="unavailable", reason="source MLIR exceeds host diagnostic byte bound")
            return row
        mlir_raw = mlir_path.read_bytes()
        mlir_sha256 = _sha256(mlir_raw)
        if (
            isinstance(trace, Mapping)
            and trace.get("source_mlir_sha256") is not None
            and trace["source_mlir_sha256"] != mlir_sha256
        ):
            raise ValueError("frozen source MLIR differs from capsule frontend trace")
        row["source_mlir"] = {"member": mlir_path.name, "sha256": mlir_sha256}

        # The Phase 0 evidence is selected by this capsule's own frozen corpus
        # location, not by target name or a mutable current-run lookup.
        evidence_root = source.parents[1] / "_evidence"
        if not any(
            not Path(grant).is_symlink()
            and Path(grant).is_dir()
            and source.is_relative_to(grant)
            and evidence_root.is_relative_to(grant)
            for grant in frozen_grants
        ):
            raise ValueError("Phase 0 evidence and model capsule lack one frozen functional corpus grant")
        manifest_path = evidence_root / "evidence-manifest.json"
        if (
            evidence_root.is_symlink()
            or (evidence_root / "software").is_symlink()
            or not evidence_root.is_dir()
            or manifest_path.is_symlink()
            or not manifest_path.is_file()
        ):
            row["reason"] = "frozen capsule corpus has no Phase 0 evidence manifest"
            return row
        manifest_raw = manifest_path.read_bytes()
        manifest = json.loads(manifest_raw)
        if (
            not isinstance(manifest, Mapping)
            or manifest.get("schema") != "phase0_evidence_v1"
            or manifest.get("target") != target
        ):
            raise ValueError("Phase 0 evidence target differs from selected Phase 2 target")
        artifacts = manifest.get("artifacts")
        if not isinstance(artifacts, Mapping):
            raise ValueError("Phase 0 evidence manifest has no artifact table")
        entry = artifacts.get(_MODEL.as_posix())
        if not isinstance(entry, Mapping):
            row["reason"] = "Phase 0 evidence manifest has no selected instruction model"
            return row
        model_path = evidence_root / _MODEL
        if model_path.is_symlink() or not model_path.is_file():
            raise ValueError("selected Phase 0 instruction model is absent or linked")
        model_raw = model_path.read_bytes()
        model_sha256 = _sha256(model_raw)
        if entry.get("sha256") != model_sha256 or entry.get("size_bytes") != len(model_raw):
            raise ValueError("selected Phase 0 instruction model differs from evidence manifest")
        model = json.loads(model_raw)
        if (
            not isinstance(model, dict)
            or model.get("target") != target
            or model.get("schema") != "merlin.instruction_semantics.v1"
        ):
            raise ValueError("selected Phase 0 instruction model target or schema differs")
        row["phase0"] = {
            "evidence_manifest_sha256": _sha256(manifest_raw),
            "instruction_model_sha256": model_sha256,
        }
        if model.get("status") == "UNKNOWN" and model.get("instructions") == []:
            row.update(status="unknown", reason="Phase 0 selected no described instruction semantics")
            return row
        model = validate_normalized_instruction_model(model, expected_target=target)
        mlir_text = mlir_raw.decode("utf-8")
        if not is_linalg_on_tensors(mlir_text):
            row.update(status="unavailable", reason="selected source is not linalg-on-tensors MLIR")
            return row
        parsed = parse_linalg_mlir(mlir_text)
        operations = parsed.get("ops")
        if not isinstance(operations, list) or len(operations) > _MAX_PAYLOAD_OPERATIONS:
            row.update(status="unavailable", reason="source MLIR exceeds host diagnostic operation bound")
            return row
        if not any(str(operation.get("operation", "")).startswith("linalg.") for operation in operations):
            row.update(status="unavailable", reason="source MLIR has no linalg payload operation")
            return row
        row.update(status="diagnostic", result=search_linalg_inventory(parsed, model, limits=_LIMITS))
    except ImportError as exc:
        row.update(status="unavailable", reason=f"semantic search dependency unavailable: {exc}")
    except Exception as exc:  # noqa: BLE001 - a diagnostic failure must not gate Phase 2
        row.update(status="refused", reason=f"{type(exc).__name__}: {exc}")
    return row


def write_portfolio_receipt(
    stage_root: Path,
    *,
    target: str,
    members: Sequence[tuple[StageE2ESentinel, bool]],
    frozen_grants: Sequence[Path],
) -> dict[str, Any]:
    """Atomically publish a private sidecar beside, outside, all agent mounts."""
    stage_root = Path(stage_root).absolute()
    if stage_root.is_symlink() or not stage_root.is_dir():
        raise ValueError("host semantic diagnostic needs a real stage root")
    stage_stat = stage_root.stat()
    if stage_stat.st_uid != os.geteuid() or stage_stat.st_mode & 0o022:
        raise ValueError("host semantic diagnostic stage root must be owner-controlled")
    if not members:
        raise ValueError("host semantic diagnostic needs selected portfolio members")
    private = stage_root / "_host_semantic_diagnostics"
    if private.exists() or private.is_symlink():
        raise ValueError("host semantic diagnostic requires a fresh private directory")
    private.mkdir(mode=0o700)
    private_stat = private.stat()
    if (
        private.is_symlink()
        or not stat.S_ISDIR(private_stat.st_mode)
        or private_stat.st_uid != os.geteuid()
        or private_stat.st_mode & 0o077
    ):
        raise ValueError("host semantic diagnostic directory is not owner-only")
    destination = private / "semantic_search.json"
    if destination.exists() or destination.is_symlink():
        raise ValueError("host semantic diagnostic requires a fresh sidecar destination")
    receipt = {
        "schema": SCHEMA,
        "target": target,
        "visibility": "host_private_diagnostic",
        "scope": "frozen selected Phase 0 instruction model and frozen selected model capsule source MLIR",
        "members": [
            inspect_member(member, target=target, capsule_linked=linked, frozen_grants=frozen_grants)
            for member, linked in members
        ],
        "score_effect": "none",
        "timing_effect": "none",
        "qualification_effect": "none",
    }
    payload = (json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    temporary = private / f".semantic_search.{secrets.token_hex(16)}.tmp"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
    descriptor = os.open(temporary, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, destination, follow_symlinks=False)
    finally:
        temporary.unlink(missing_ok=True)
    return receipt
