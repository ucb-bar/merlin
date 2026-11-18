"""Read-only Phase 0 evidence selection and exact run-owned snapshots.
"""

from __future__ import annotations

import copy
import hashlib
import io
import json
import math
import os
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from types import SimpleNamespace
from typing import Any
import yaml
from merlin.development.testgen.evidence_output import _materialize_evidence


# ---- evidence ----------------------------------------------------------------------------------

def evidence_artifacts(selection: EvidenceSelection) -> dict[str, bytes]:
    """Calculate the existing evidence artifacts without writing files."""
    from merlin.development.testgen.evidence.observations import EvidenceSelection, _canonical_digest, _digest, _json
    if selection.archived_artifacts:
        # A newer implementation must not rewrite historical reports or widen
        # their qualification. Re-export the already validated original bytes.
        return dict(selection.archived_artifacts)
    outputs: dict[str, bytes] = {}
    outputs["software/selection.json"] = selection.views_json
    if selection.raw_facts is not None:
        outputs["hardware/circt/facts.json"] = selection.raw_facts
    hardware_views = {
        "loaded-facts": selection.loaded_facts,
        "refreshed-facts": selection.refreshed_facts,
        "target-profile": selection.target_profile,
        "execution-capabilities": selection.execution_capabilities,
        "performance-facts": selection.performance_facts,
        "readout-inputs": selection.readout_inputs,
        "readout-facets": selection.readout_facets,
        "readout-numerics": selection.readout_numerics,
        "isa-taxonomy": selection.isa_taxonomy,
        "toolchain-inputs": selection.toolchain_inputs,
        "quantization": selection.quantization_snapshot,
    }
    for name, value in hardware_views.items():
        outputs[f"hardware/effective-views/{name}.json"] = _json(value)
    for name in ("contract", "residual", "hardware_spec", "software_spec", "datapath", "diagnostics"):
        outputs[f"software/{name.replace('_', '-')}.json"] = _json(getattr(selection, name))
    outputs["software/instruction-semantics.json"] = _json(selection.instruction_semantics)
    if selection.instruction_semantics_source is not None:
        outputs["software/instruction-semantics-authored.yaml"] = selection.instruction_semantics_source
    from merlin.compiler.quantization.admission import build_quantization_contract, capture_recipe_candidates
    from merlin.development.targetgen.operation_accounting import build_operation_accounting
    from merlin.development.targetgen.performance_basis import build_performance_basis

    accounting = build_operation_accounting(
        getattr(selection, "application_inventory", None),
        selection.software_spec or None,
        capability_contract=selection.contract,
        framework_catalog=getattr(selection, "framework_catalog", None),
        framework_catalogs=getattr(selection, "framework_catalogs", None),
        frontend_traces=getattr(selection, "frontend_traces", None),
        application_graphs=getattr(selection, "application_graphs", None),
        host_capabilities=getattr(selection, "host_capabilities", None),
        capture_execution_attestations=getattr(selection, "capture_execution_attestations", None),
        application_sources={
            source.role.removeprefix("application-capture:"): source.content
            for source in selection.source_snapshots
            if source.role.startswith("application-capture:")
        }
        or None,
    )
    accounting["selected_inventory"] = getattr(
        selection,
        "application_inventory_identity",
        {
            "status": "not_available",
            "reason": "snapshot predates selected application inventory",
        },
    )
    native_baselines = getattr(selection, "native_baseline_observations", {})
    for label, observation in native_baselines.items():
        destination = f"software/native-baselines/{_digest(label.encode())}"
        observation["frozen_artifact_root"] = destination
        original_root = Path(observation["source"]).parent
        for source in selection.source_snapshots:
            if source.role == f"native-qualification:{label}":
                outputs[f"{destination}/qualification.json"] = source.content
            elif source.role == f"native-artifact:{label}":
                outputs[f"{destination}/{source.path.relative_to(original_root).as_posix()}"] = source.content
        application = accounting["applications"][label]
        application["native_baseline_observation"] = copy.deepcopy(observation)
        application["precision_execution"] = {
            "status": observation["status"],
            "executor": "native_cpu",
            "target_executed": False,
            "input_abi": observation["input_abi"],
            "output_abi": observation["output_abi"],
            "scope": "whole saved program; individual operations and RVV board support remain unqualified",
        }
    quantization = build_quantization_contract(
        selection.software_spec,
        {
            "contract": selection.contract,
            "quantization_candidates": selection.quantization_snapshot.get("quantization_candidates", []),
            "readout_facets": selection.readout_facets,
            "readout_numerics": selection.readout_numerics,
        },
        accounting,
    )
    if getattr(selection, "rtl_source_relocations", None) is not None:
        outputs["hardware/source-relocations.json"] = _json(selection.rtl_source_relocations)
    outputs["coverage/operation-accounting.json"] = _json(accounting)
    outputs["coverage/performance-basis.json"] = _json(
        build_performance_basis(
            accounting,
            selection.performance_facts,
            target=selection.target,
            operation_accounting_sha256=_digest(outputs["coverage/operation-accounting.json"]),
            performance_facts_artifact_sha256=_digest(outputs["hardware/effective-views/performance-facts.json"]),
            selected_hardware_spec_sha256=_canonical_digest(selection.hardware_spec),
            raw_rtl_facts_sha256=selection.raw_facts_sha256,
        )
    )
    outputs["software/quantization-contract.json"] = _json(quantization)
    from merlin.development.testgen.capture_recipe import recipe_outputs

    outputs.update(
        recipe_outputs(
            capture_recipe_candidates(selection.software_spec, quantization),
            target=selection.target,
            software_review=selection.software_spec.get("status"),
            encode=_json,
        )
    )
    outputs["software/host-capabilities.json"] = _json(getattr(selection, "host_capabilities", {}))
    outputs["software/native-baseline-observations.json"] = _json(native_baselines)
    for label, graph in sorted(getattr(selection, "application_graphs", {}).items()):
        identity = _digest(str(label).encode())
        outputs[f"coverage/application-graphs/{identity}.json"] = _json(graph)
    # Application labels are metadata, never paths: no traversal or collisions.
    for label, trace in sorted(getattr(selection, "frontend_traces", {}).items()):
        identity = _digest(str(label).encode())
        outputs[f"software/frontend/{identity}/frontend-trace.json"] = _json(trace)
    for label, catalog in sorted(getattr(selection, "framework_catalogs", {}).items()):
        identity = _digest(str(label).encode())
        outputs[f"software/frontend/{identity}/pytorch-opset.json"] = _json(catalog)
    outputs["software/frontend/index.json"] = _json(
        {
            "schema": "merlin.phase0.frontend_index.v1",
            "applications": {
                label: {
                    "frontend_trace": f"software/frontend/{_digest(label.encode())}/frontend-trace.json"
                    if label in getattr(selection, "frontend_traces", {})
                    else None,
                    "application_graph": f"coverage/application-graphs/{_digest(label.encode())}.json"
                    if label in getattr(selection, "application_graphs", {})
                    else None,
                    "framework_catalog": f"software/frontend/{_digest(label.encode())}/pytorch-opset.json"
                    if label in getattr(selection, "framework_catalogs", {})
                    else None,
                }
                for label in sorted((getattr(selection, "application_inventory", None) or {}).get("applications", {}))
            },
        }
    )
    from merlin.development.testgen.coverage_readme import render as render_coverage_readme

    outputs["coverage/README.md"] = render_coverage_readme(accounting, quantization)
    outputs["software/framework/pytorch-opset.json"] = _json(
        getattr(
            selection,
            "framework_catalog",
            {
                "schema": "merlin.pytorch_opset.v1",
                "status": "not_available",
                "reason": "snapshot predates framework catalog observation",
            },
        )
    )
    for source in selection.source_snapshots:
        if source.role == "application-inventory":
            outputs["coverage/application-inventory.json"] = source.content
    source_index = []
    for index, source in enumerate(selection.source_snapshots):
        destination = f"software/source-snapshots/{index:04d}-{source.sha256}.bin"
        outputs[destination] = source.content
        source_index.append(
            {
                "source": str(source.path),
                "role": source.role,
                "path": destination,
                "sha256": source.sha256,
                "size_bytes": len(source.content),
            }
        )
    manifest = {
        "schema": "phase0_evidence_v1",
        "target": selection.target,
        "status": selection.status,
        "raw_facts_sha256": selection.raw_facts_sha256,
        "target_profile_sha256": selection.performance_facts["target_profile_sha256"],
        "performance_facts_sha256": selection.performance_facts["sha256"],
        "diagnostics": selection.diagnostics,
        "sources": source_index,
        "qualification_blockers": selection.qualification_blockers,
        "artifacts": {name: {"sha256": _digest(raw), "size_bytes": len(raw)} for name, raw in outputs.items()},
        "consumers": {
            "corpus_binding": [
                "software/contract.json",
                "hardware/effective-views/loaded-facts.json",
                "software/datapath.json",
                "hardware/effective-views/isa-taxonomy.json",
            ],
            "performance_gates": ["hardware/effective-views/performance-facts.json"],
            "performance_basis": [
                "coverage/operation-accounting.json",
                "hardware/effective-views/performance-facts.json",
                "software/hardware-spec.json",
                "software/software-spec.json",
                "software/contract.json",
                *(["hardware/circt/facts.json"] if selection.raw_facts is not None else []),
                "coverage/performance-basis.json",
            ],
            "memory_regime_axes": ["hardware/effective-views/refreshed-facts.json"],
            "readout": [
                "hardware/effective-views/refreshed-facts.json",
                "hardware/effective-views/readout-inputs.json",
            ],
            "operation_accounting": [
                "software/software-spec.json",
                "software/contract.json",
                "software/framework/pytorch-opset.json",
                "coverage/operation-accounting.json",
                "software/host-capabilities.json",
                "software/native-baseline-observations.json",
                "software/frontend/index.json",
                *sorted(
                    name
                    for name in outputs
                    if name.startswith(("coverage/application-graphs/", "software/frontend/"))
                    and name != "software/frontend/index.json"
                ),
                *(["coverage/application-inventory.json"] if "coverage/application-inventory.json" in outputs else []),
            ],
            "quantization": [
                "software/datapath.json",
                "hardware/effective-views/quantization.json",
                "software/quantization-contract.json",
                "software/quantization-recipes.json",
                *sorted(name for name in outputs if name.startswith("software/quantization-recipes/")),
                "coverage/operation-accounting.json",
            ],
            "instruction_selection": [
                "software/software-spec.json",
                "software/contract.json",
                "software/instruction-semantics.json",
                *(
                    ["software/instruction-semantics-authored.yaml"]
                    if selection.instruction_semantics_source is not None
                    else []
                ),
                *(["hardware/circt/facts.json"] if selection.raw_facts is not None else []),
            ],
        },
        "qualification": "byte snapshots and declared/derived views only; no compiler or hardware verdict",
    }
    outputs["evidence-manifest.json"] = _json(manifest)
    return outputs


def export_evidence(selection: EvidenceSelection, artifact_root: str | Path) -> dict:
    """Write only already-observed bytes beneath the explicit Phase 0 run root."""
    from merlin.development.testgen.evidence.observations import EvidenceSelection
    outputs = evidence_artifacts(selection)
    _materialize_evidence(Path(artifact_root), outputs)
    return json.loads(outputs["evidence-manifest.json"])


def load_exported_evidence(artifact_root: str | Path) -> EvidenceSelection:
    """Verify and reopen saved views without reading any original source path."""
    from merlin.development.testgen.evidence.observations import EvidenceSelection, EvidenceSource, _digest
    root = Path(artifact_root).absolute()
    manifest_path = root / "evidence-manifest.json"
    if manifest_path.is_symlink() or any(parent.is_symlink() for parent in manifest_path.parents):
        raise ValueError("evidence manifest traverses a symlink")
    manifest_raw = manifest_path.read_bytes()
    manifest = json.loads(manifest_raw)
    if not isinstance(manifest, dict) or manifest.get("schema") != "phase0_evidence_v1":
        raise ValueError("unsupported Phase 0 evidence manifest")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("evidence manifest requires an artifact inventory")
    observed = {}
    for name, record in artifacts.items():
        member = Path(name)
        if member.is_absolute() or ".." in member.parts or not member.parts:
            raise ValueError(f"invalid evidence member: {name}")
        path = root / member
        if path.is_symlink() or any(parent.is_symlink() for parent in path.parents):
            raise ValueError(f"evidence member traverses a symlink: {name}")
        raw = path.read_bytes()
        if not isinstance(record, dict) or record.get("sha256") != _digest(raw) or record.get("size_bytes") != len(raw):
            raise ValueError(f"evidence member changed: {name}")
        observed[name] = raw
    views = observed.get("software/selection.json")
    if views is None:
        raise ValueError("evidence selection snapshot is missing")
    document = json.loads(views)
    if document.get("target") != manifest.get("target"):
        raise ValueError("evidence selection target differs from manifest")
    snapshots = []
    for source in manifest.get("sources") or []:
        raw = observed.get(source.get("path"))
        if raw is None or source.get("sha256") != _digest(raw) or source.get("size_bytes") != len(raw):
            raise ValueError("evidence source snapshot changed or is missing")
        snapshots.append(EvidenceSource(Path(source["source"]), source["role"], raw))
    selection = EvidenceSelection(
        manifest["target"],
        tuple(snapshots),
        views,
        observed.get("hardware/circt/facts.json"),
        tuple(sorted({**observed, "evidence-manifest.json": manifest_raw}.items())),
        observed.get("software/instruction-semantics-authored.yaml"),
    )
    if selection.raw_facts_sha256 != manifest.get("raw_facts_sha256"):
        raise ValueError("raw facts identity differs from evidence manifest")
    return selection
