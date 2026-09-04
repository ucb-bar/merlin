"""Deterministic readiness and execution gate for paper Merlin backend producers.

Prepared paper inputs are not compiler inputs.  A real producer needs the capture-complete MLIR
session, the promoted compiler authority, and backend-specific lowering/linking support.  This
module enumerates those prerequisites for every required package and refuses to route a cell
through the generic MRLNSES2 demonstration producer when its declared backend is unsupported.

Once a built-in backend producer exists, its final action must be
``paper_merlin_packages.register_backend_producer_input``.  Until then ``--execute`` is a
fail-closed audit, not a source of placeholder objects.
"""
from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from merlin.common.artifacts import ProductDir, new_product
from merlin.common.mlir_query import forward_signature
from merlin.common.paths import repo_root

from .freeze import sha256_paths
from .paper import MatrixCell, PaperStudySpec
from .paper_build_bundle import load_multi_toolchain_authority
from .paper_merlin_packages import (
    _capture_registration,
    _required_cells,
    _validate_promoted_campaign_binding,
    _validate_registered_captures,
)
from .paper_session_abi import SessionDescriptor, load_session_descriptor
from .session import validate_paper_input_binding

_CAPABILITY = {
    "hand_v0_int8": (
        "hand_w8a8_mrlnses2_producer_absent",
        "The MRLNSES2 worker does not apply the hand_v0_int8 normalization, quantization passes, "
        "schedule, feature set, or captured-weight binding.",
    ),
    "merlin_frozen": (
        "promoted_compiler_mrlnses2_adapter_absent",
        "The selected rvvhost-compile package has no closed adapter that emits the public "
        "MRLNSES2 object graph; the existing worker calls generic lower_model directly and ignores "
        "the promoted policy/compiler.",
    ),
    "merlin_xnnpack": (
        "xnnpack_mrlnses2_producer_absent",
        "The MRLNSES2 worker neither performs the complete XNNPACK matmul rewrite nor builds and "
        "links its bound RVV shim/library into the replayable object graph.",
    ),
    "merlin_openblas": (
        "openblas_mrlnses2_producer_absent",
        "The MRLNSES2 worker neither performs the complete OpenBLAS matmul rewrite nor builds and "
        "links its bound RVV shim/library into the replayable object graph.",
    ),
}


class ProducerPlanNotReady(RuntimeError):
    """No complete, genuine backend producer set can be published."""

    def __init__(self, reasons: Sequence[str], output_dir: Path):
        super().__init__("; ".join(reasons))
        self.reasons = tuple(reasons)
        self.output_dir = output_dir


@dataclass(frozen=True)
class Blocker:
    code: str
    detail: str

    def to_dict(self) -> dict[str, str]:
        return {"code": self.code, "detail": self.detail}


def _sha(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _package_cells(study: PaperStudySpec) -> tuple[MatrixCell, ...]:
    first_core = study.core_counts[0]
    cells = tuple(cell for cell in _required_cells(study) if cell.core_count == first_core)
    if len(cells) != 25:
        raise ValueError(f"paper producer roster has {len(cells)} packages, expected exactly 25")
    return cells


def _program_mlir(capture: Path, descriptor: SessionDescriptor) -> tuple[Path, ...]:
    contract = yaml.safe_load((capture / "session_contract.yaml").read_text(encoding="utf-8"))
    if not isinstance(contract, Mapping):
        raise ValueError("capture session contract is not a mapping")
    if descriptor.source_contract_version == 1:
        paths = (capture / "model.mlir",)
    else:
        rows = contract.get("programs")
        if not isinstance(rows, list) or len(rows) != len(descriptor.programs):
            raise ValueError("capture program list differs from its MRLNSES2 descriptor")
        paths = tuple(capture / str(row.get("bundle", "")) / "model.mlir"
                      for row in rows if isinstance(row, Mapping))
    if len(paths) != len(descriptor.programs):
        raise ValueError("capture omits a program MLIR file")
    for path in paths:
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(capture):
            raise ValueError(f"capture program MLIR is absent or unsafe: {path}")
    return paths


def _abi_blockers(capture: Path) -> list[Blocker]:
    try:
        descriptor = load_session_descriptor(capture)
        programs = _program_mlir(capture, descriptor)
    except (OSError, ValueError) as exc:
        return [Blocker("capture_session_descriptor_invalid", str(exc))]
    external = {(row.endpoint.program, row.endpoint.input) for row in descriptor.inputs}
    routes = {(row.target_program, row.target_input) for row in descriptor.routes}
    states = {(row.program, row.input) for row in descriptor.states}
    blockers: list[Blocker] = []
    for index, path in enumerate(programs):
        try:
            inputs, outputs = forward_signature(path)
        except (OSError, ValueError) as exc:
            blockers.append(Blocker(
                "capture_mlir_signature_invalid", f"program {index}: {exc}"))
            continue
        covered = {input_index for owner, input_index in external | routes | states
                   if owner == index}
        missing = sorted(set(range(len(inputs))) - covered)
        if missing:
            blockers.append(Blocker(
                "mrlnses2_unbound_mlir_inputs",
                f"program {index} has unbound MLIR input arguments {missing}; captured weights or "
                "immutable context need an explicit public binding/load recipe"))
        if not outputs:
            blockers.append(Blocker(
                "capture_mlir_output_absent", f"program {index} has no compiled output"))
    return blockers


def _capture_blockers(cell: MatrixCell, registration: Mapping[str, Any] | None) -> list[Blocker]:
    artifact = cell.model.artifacts[cell.precision]
    value, digest = str(artifact.get("path", "")), str(artifact.get("sha256", ""))
    if not value or value == "unresolved" or len(digest) != 64:
        return [Blocker(
            "capture_artifact_unresolved",
            f"{cell.model.name}/{cell.precision} has no capture-complete path and SHA-256")]
    capture = Path(value).resolve()
    if capture.is_symlink() or not capture.is_dir():
        return [Blocker("capture_artifact_absent", f"capture is absent or unsafe: {capture}")]
    if sha256_paths([capture]) != digest:
        return [Blocker("capture_artifact_digest_mismatch", f"capture bytes differ: {capture}")]
    if registration is None:
        return [Blocker(
            "capture_registration_absent", "no complete registration authorizes this capture")]
    rows = registration.get("captures")
    matches = [row for row in rows if isinstance(row, Mapping)
               and row.get("model") == cell.model.name
               and row.get("precision") == cell.precision] if isinstance(rows, list) else []
    if len(matches) != 1 or matches[0].get("sha256") != digest:
        return [Blocker(
            "capture_registration_cell_mismatch", "capture registration does not bind this cell")]
    return _abi_blockers(capture)


def _paper_input_prerequisites(study: PaperStudySpec) -> tuple[list[Blocker], dict[str, Any]]:
    value = Path(str(study.paper_inputs.get("path", "")))
    bundle = value.resolve() if value.is_absolute() else (repo_root() / value).resolve()
    expected = str(study.paper_inputs.get("sha256", ""))
    evidence: dict[str, Any] = {"path": str(bundle), "expected_sha256": expected}
    if bundle.is_symlink() or not bundle.is_dir():
        return [Blocker(
            "paper_input_bundle_absent", f"frozen paper input bundle is absent or unsafe: {bundle}")
        ], evidence
    actual = sha256_paths([bundle])
    evidence["actual_sha256"] = actual
    if actual != expected:
        return [Blocker(
            "paper_input_bundle_digest_mismatch",
            f"prepared-input bytes differ: study={expected} actual={actual}")], evidence
    binding_errors = validate_paper_input_binding(bundle, study.models)
    if binding_errors:
        return [Blocker("paper_input_bundle_binding_invalid", detail)
                for detail in binding_errors], evidence
    evidence["status"] = "validated"
    evidence["models"] = sorted(study.holdout_models)
    return [], evidence


def _global_prerequisites(
        study_path: Path, study: PaperStudySpec, capture_registration: Path | None,
        promoted_compiler: Path | None, runtime_artifact: Path | None,
        producer_authority: Path | None,
        ) -> tuple[Mapping[str, Any] | None, list[Blocker], dict[str, Any]]:
    blockers: list[Blocker] = []
    registration: Mapping[str, Any] | None = None
    evidence: dict[str, Any] = {}
    paper_input_blockers, paper_input_evidence = _paper_input_prerequisites(study)
    blockers.extend(paper_input_blockers)
    evidence["paper_inputs"] = paper_input_evidence
    if capture_registration is None:
        blockers.append(Blocker(
            "capture_registration_absent",
            "capture-complete staged study and capture-registration.json do not exist"))
    else:
        try:
            registration = _capture_registration(study_path, capture_registration)
            _validate_registered_captures(study, registration)
            evidence["capture_registration_sha256"] = _sha(capture_registration)
        except (OSError, ValueError) as exc:
            blockers.append(Blocker("capture_registration_invalid", str(exc)))
            registration = None
    if promoted_compiler is None:
        blockers.append(Blocker(
            "promoted_compiler_absent",
            "the generic CPU-host campaign has not published its selected compiler package"))
    elif registration is None:
        blockers.append(Blocker(
            "promoted_compiler_unauthorized",
            "a compiler package cannot be matched without the capture-authorizing registration"))
    else:
        try:
            _validate_promoted_campaign_binding(registration, promoted_compiler)
            evidence["promoted_compiler_sha256"] = sha256_paths([promoted_compiler])
        except (OSError, ValueError) as exc:
            blockers.append(Blocker("promoted_compiler_invalid", str(exc)))
    if runtime_artifact is None:
        blockers.append(Blocker(
            "runtime_artifact_absent", "no explicit frozen runtime artifact was supplied"))
    elif runtime_artifact.is_symlink() or not runtime_artifact.is_file():
        blockers.append(Blocker(
            "runtime_artifact_invalid", f"runtime artifact is absent or unsafe: {runtime_artifact}"))
    else:
        evidence["runtime_artifact_sha256"] = _sha(runtime_artifact)
    if producer_authority is None:
        blockers.append(Blocker(
            "producer_toolchain_authority_absent",
            "no explicit multi-toolchain authority for deterministic MRLNSES2 lowering was supplied"))
    else:
        try:
            authority = load_multi_toolchain_authority(producer_authority)
            evidence["producer_toolchain_authority_sha256"] = authority.sha256
        except (OSError, ValueError) as exc:
            blockers.append(Blocker("producer_toolchain_authority_invalid", str(exc)))
    return registration, blockers, evidence


def materialize(
        study_path: str | Path, *, capture_registration: str | Path | None = None,
        promoted_compiler: str | Path | None = None,
        runtime_artifact: str | Path | None = None,
        producer_authority: str | Path | None = None, execute: bool = False,
        product: ProductDir | None = None) -> Path:
    """Audit or execute the built-in producer set; never substitute a generic graph."""
    study_path = Path(study_path).resolve()
    study = PaperStudySpec.from_yaml(study_path)
    registration_path = Path(capture_registration).resolve() if capture_registration else None
    promoted_path = Path(promoted_compiler).resolve() if promoted_compiler else None
    runtime_path = Path(runtime_artifact).resolve() if runtime_artifact else None
    authority_path = Path(producer_authority).resolve() if producer_authority else None
    sources = [str(study_path), *(str(path) for path in (
        registration_path, promoted_path, runtime_path, authority_path) if path is not None)]
    product = product or new_product(
        "paper-merlin-producers", version=1, target=study.target, sources=sources)
    cells = _package_cells(study)
    registration, global_blockers, evidence = _global_prerequisites(
        study_path, study, registration_path, promoted_path, runtime_path, authority_path)
    rows: list[dict[str, Any]] = []
    for cell in cells:
        blockers = [*global_blockers, *_capture_blockers(cell, registration)]
        capability_code, capability_detail = _CAPABILITY[cell.backend.name]
        blockers.append(Blocker(capability_code, capability_detail))
        unique = {blocker.code: blocker for blocker in blockers}
        rows.append({
            "cell": {"model": cell.model.name, "backend": cell.backend.name,
                     "precision": cell.precision},
            "status": "blocked",
            "registered": False,
            "blockers": [value.to_dict() for value in unique.values()],
        })
    actual_hand = sorted({cell.precision for cell in cells
                          if cell.backend.name == "hand_v0_int8"})
    plan = {
        "schema_version": 1, "kind": "paper_merlin_backend_producer_plan_v1",
        "mode": "execute" if execute else "preflight", "status": "blocked",
        "study": {"path": str(study_path), "sha256": _sha(study_path)},
        "matrix_contract": {
            "packages": len(cells), "templates_after_packaging": len(cells) * len(study.core_counts),
            "hand_v0_int8_precisions": actual_hand,
            "requested_hand_fp32_conflict": actual_hand != ["fp32"],
            "resolution": "preserve_frozen_study",
        },
        "evidence": evidence,
        "capability_sources": {
            "generic_worker": "merlin/python/merlin/compare/paper_merlin_lower_worker.py",
            "hand_and_kernel_swap_reference": "merlin/python/merlin/mining/k1.py",
            "producer": "merlin/python/merlin/compare/paper_merlin_mlir_producer.py",
        },
        "cells": rows,
        "summary": {
            "required_packages": len(cells), "genuinely_produced": 0,
            "registered": 0, "runnable_templates": 0,
            "blocked": len(rows),
            "blocker_counts": {
                code: sum(any(blocker["code"] == code for blocker in row["blockers"])
                          for row in rows)
                for code in sorted({blocker["code"] for row in rows
                                    for blocker in row["blockers"]})
            },
        },
    }
    plan_path = product.add_artifact("producer-plan.json")
    _write_json(plan_path, plan)
    product.notes = (
        f"paper Merlin producers blocked; genuine=0/{len(cells)}; execute={execute}")
    product.write_manifest()
    raise ProducerPlanNotReady(
        [f"{row['cell']['model']}/{row['cell']['backend']}/{row['cell']['precision']}: "
         f"{','.join(blocker['code'] for blocker in row['blockers'])}" for row in rows],
        product.path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="merlin-paper-merlin-producers")
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--capture-registration", type=Path)
    parser.add_argument("--promoted-compiler", type=Path)
    parser.add_argument("--runtime-artifact", type=Path)
    parser.add_argument("--producer-authority", type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    try:
        output = materialize(
            args.study, capture_registration=args.capture_registration,
            promoted_compiler=args.promoted_compiler, runtime_artifact=args.runtime_artifact,
            producer_authority=args.producer_authority, execute=args.execute)
    except ProducerPlanNotReady as exc:
        print("merlin-paper-merlin-producers: BLOCKED — 0/25 genuine graphs")
        print(f"evidence: {exc.output_dir}")
        return 2
    print(f"merlin-paper-merlin-producers: wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
