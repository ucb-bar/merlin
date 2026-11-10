"""Retain raw frontend evidence and exact selected producer source bytes."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path
from urllib.parse import unquote, urlparse

from merlin.capture.artifacts import CapsuleArtifacts, M2MUnavailable, _portable_weights_reference


def _write_frontend_evidence(
    art: CapsuleArtifacts, directory: Path, packaged_mlir: str, *, packaged_path: str = "capsule.interface.mlir"
) -> dict | None:
    """Bind raw trace bytes and each portable emitted MLIR to an exact edit receipt."""
    meta = art.meta or {}
    selected = meta.get("frontend_trace")
    if not isinstance(selected, dict):
        return None  # Historical diagnostic captures have no source-call proof.
    trace_path = Path(selected.get("path", ""))
    if trace_path.is_symlink() or not trace_path.is_file():
        raise M2MUnavailable("selected capture frontend trace is missing or symlinked")
    raw_trace = trace_path.read_bytes()
    if hashlib.sha256(raw_trace).hexdigest() != selected.get("sha256"):
        raise M2MUnavailable("selected capture frontend trace bytes changed")
    raw_path = meta.get("frontend_raw_mlir_path")
    if raw_path is not None:
        source_path = Path(raw_path)
        if source_path.is_symlink() or not source_path.is_file():
            raise M2MUnavailable("raw frontend MLIR is missing or symlinked")
        raw_source = source_path.read_bytes()
    else:
        raw_source = art.linalg_mlir.encode()
    declared_mlir = json.loads(raw_trace).get("mlir") or {}
    if declared_mlir.get("sha256") is not None and declared_mlir["sha256"] != hashlib.sha256(raw_source).hexdigest():
        raise M2MUnavailable("frontend trace does not match its exact source MLIR")
    if declared_mlir.get("bytes") is not None and declared_mlir["bytes"] != len(raw_source):
        raise M2MUnavailable("frontend trace source MLIR byte count changed")
    original_weight = meta.get("materialized_weights_reference") or getattr(art, "weights_path", None)
    has_sidecar = (directory / "capsule.weights.safetensors").is_file()
    expected_package, packaging_edit = _portable_weights_reference(
        art.linalg_mlir, original_weight, sidecar=has_sidecar
    )
    if packaged_mlir != expected_package:
        raise M2MUnavailable("capsule MLIR change is not the declared weights-reference portability edit")
    portable_source, source_edit = _portable_weights_reference(
        raw_source.decode("utf-8"), original_weight, sidecar=has_sidecar
    )
    if raw_source != art.linalg_mlir.encode() and not meta.get("capture_normalization"):
        raise M2MUnavailable("raw frontend and captured MLIR differ without a normalization receipt")
    source_bytes = portable_source.encode("utf-8")
    raw_hash = hashlib.sha256(raw_source).hexdigest()
    source_hash = hashlib.sha256(source_bytes).hexdigest()
    packaged_bytes = packaged_mlir.encode("utf-8")
    interface_path = directory / "capsule.interface.mlir"
    interface_hash = (
        hashlib.sha256(packaged_bytes).hexdigest()
        if packaged_path == interface_path.name
        else hashlib.sha256(interface_path.read_bytes()).hexdigest()
    )
    sidecars = None
    if has_sidecar:
        sidecars = {
            name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
            for name in ("capsule.weights.safetensors", "capsule.weights.safetensors.manifest.json")
        }
    capture_receipt_bytes = meta.get("materialized_receipt_bytes")
    (directory / "frontend-trace.json").write_bytes(raw_trace)
    (directory / "frontend-source.mlir").write_bytes(source_bytes)
    catalog = meta.get("framework_catalog") or {}
    if catalog:
        catalog_path = Path(catalog.get("path", ""))
        if catalog_path.is_symlink() or not catalog_path.is_file():
            raise M2MUnavailable("selected capture framework catalog is missing or symlinked")
        raw_catalog = catalog_path.read_bytes()
        if hashlib.sha256(raw_catalog).hexdigest() != catalog.get("sha256"):
            raise M2MUnavailable("selected capture framework catalog bytes changed")
        (directory / "pytorch-opset.json").write_bytes(raw_catalog)
    record = {
        "schema": "merlin.capsule_frontend_evidence.v2",
        "path": "frontend-trace.json",
        "sha256": hashlib.sha256(raw_trace).hexdigest(),
        "source_mlir": "frontend-source.mlir",
        "source_mlir_sha256": source_hash,
        "source_mlir_bytes": len(source_bytes),
        "raw_source_mlir_sha256": raw_hash,
        "raw_source_mlir_bytes": len(raw_source),
        "raw_source_trace_bound": (
            declared_mlir.get("sha256") == raw_hash and declared_mlir.get("bytes") == len(raw_source)
        ),
        "capture_mlir_sha256": hashlib.sha256(art.linalg_mlir.encode()).hexdigest(),
        "packaged_mlir": packaged_path,
        "packaged_mlir_sha256": hashlib.sha256(packaged_bytes).hexdigest(),
        "packaged_mlir_bytes": len(packaged_bytes),
        "interface_mlir": interface_path.name,
        "interface_mlir_sha256": interface_hash,
        "weights_sidecars": sidecars,
        "capture_receipt": (
            {"path": "source-capture-receipt.json", "sha256": hashlib.sha256(capture_receipt_bytes).hexdigest()}
            if capture_receipt_bytes is not None
            else None
        ),
        "normalization": meta.get("capture_normalization"),
        # The frontend revision, as verified against its software pin when the capture was made.
        "frontend_pin": (
            {k: meta["model2mlir_pin"][k] for k in ("pin", "ok", "citation")} if meta.get("model2mlir_pin") else None
        ),
        "packaging": packaging_edit["kind"],
        "weights_reference_relocation": (
            {"from_reference_sha256": packaging_edit["from_reference_sha256"], "to": packaging_edit["to"]}
            if packaging_edit["kind"] == "weights_reference_relocation"
            else None
        ),
        "source_portability": {
            **source_edit,
            "raw_sha256": raw_hash,
            "emitted_sha256": source_hash,
        },
        "qualification": (
            "trace-bound raw hash and exact portability edit; compiler execution unverified"
            if declared_mlir.get("sha256") == raw_hash and declared_mlir.get("bytes") == len(raw_source)
            else "raw digest observed without a complete trace byte declaration; compiler execution unverified"
        ),
    }
    (directory / "frontend-evidence.json").write_text(json.dumps(record, sort_keys=True, indent=2) + "\n")
    return record


def _freeze_selected_m2m_tool(m2m_root: Path, directory: Path) -> dict:
    """Keep the exact selected installed M2M package beside a live capture.

    A package receipt is not a proof about Torch, arbitrary loader imports, or
    runtime data.  Source checkouts without a wheel remain explicitly unverified.
    Installed distributions must match their originating wheel byte for byte;
    a missing or changed wheel is a failed capture, not an anonymous version.
    """
    distributions = sorted(m2m_root.glob("m2m-*.dist-info"))
    if not distributions:
        return {
            "status": "not_available",
            "reason": "selected M2M source has no installed-wheel identity",
            "scope": "selected M2M package only; Torch, loaders and data are not covered",
        }
    if len(distributions) != 1 or distributions[0].is_symlink():
        raise M2MUnavailable("selected M2M installation has ambiguous distribution identity")
    direct = distributions[0] / "direct_url.json"
    if direct.is_symlink() or not direct.is_file():
        raise M2MUnavailable("selected installed M2M distribution has no wheel origin")
    try:
        origin = json.loads(direct.read_text(encoding="utf-8"))
        url = urlparse(origin["url"])
        if url.scheme != "file" or url.netloc not in ("", "localhost"):
            raise ValueError("M2M wheel origin must be a local file")
        wheel = Path(unquote(url.path))
        if wheel.suffix != ".whl" or wheel.is_symlink() or not wheel.is_file():
            raise ValueError("M2M wheel origin is missing or not a regular wheel")
        payload = wheel.read_bytes()
        with zipfile.ZipFile(wheel) as archive:
            names = archive.namelist()
            modules = sorted(name for name in names if name.startswith("m2m/") and name.endswith(".py"))
            if not modules or len(names) != len(set(names)):
                raise ValueError("M2M wheel has no unique module inventory")
            for name in modules:
                if Path(name).is_absolute() or ".." in Path(name).parts:
                    raise ValueError("M2M wheel contains an unsafe module path")
            installed = sorted(path.relative_to(m2m_root).as_posix() for path in (m2m_root / "m2m").rglob("*.py"))
            if installed != modules:
                raise ValueError("installed M2M module set differs from its selected wheel")
            module_identity = {}
            for name in modules:
                path = m2m_root / name
                if path.is_symlink() or not path.is_file():
                    raise ValueError(f"installed M2M module is missing or symlinked: {name}")
                source = path.read_bytes()
                if source != archive.read(name):
                    raise ValueError(f"installed M2M module differs from its selected wheel: {name}")
                module_identity[name] = {"bytes": len(source), "sha256": hashlib.sha256(source).hexdigest()}
    except (OSError, KeyError, ValueError, zipfile.BadZipFile) as exc:
        raise M2MUnavailable(f"cannot prove selected installed M2M wheel identity: {exc}") from exc
    directory.mkdir(parents=True, exist_ok=True)
    wheel_member = directory / "capture-tool.whl"
    wheel_member.write_bytes(payload)
    record = {
        "schema": "merlin.capsule_capture_tool.v1",
        "status": "verified_selected_package",
        "wheel": {"path": wheel_member.name, "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()},
        "modules": module_identity,
        "scope": "selected M2M package only; Torch, loaders and data are not covered",
    }
    sidecar = directory / "capture-tool.json"
    sidecar.write_text(json.dumps(record, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return {
        "status": record["status"],
        "path": sidecar.name,
        "sha256": hashlib.sha256(sidecar.read_bytes()).hexdigest(),
    }
