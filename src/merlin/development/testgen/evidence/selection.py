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

def select_evidence(
    target: str,
    *,
    descriptor=None,
    capture_python=None,
    capability_contract_path=None,
    facts_path=None,
    hardware_spec=None,
    software_spec=None,
    conformance_spec=None,
    inventory_path=None,
    native_qualifications=None,
    prohibited_roles=(),
    hardware_intake=None,
    software_intake=None,
    source_components: bool = False,
    rtl_source_relocations=None,
) -> EvidenceSelection:
    """Observe selected source bytes once; never extract facts or modify a checkout.

    Missing native evidence produces diagnostic unknowns. Malformed explicitly
    selected documents fail rather than selecting another source implicitly.
    """
    from merlin.development.testgen.evidence.observations import EvidenceSelection, EvidenceSource, _canonical_digest, _digest, _json, _native_baseline_observations, _record_scaling_contradictions, _reference_inventory_members, _same_selected_file, _same_source_consistency
    from merlin.analysis.perf.profile import derive_profile
    from merlin.development.targetgen import readout_facet, target_registry
    from merlin.development.targetgen.rtl import facts as rtl_facts
    from merlin.runtime.backends import base, chipyard_rocc

    if hardware_intake is not None:
        from merlin.development.testgen.rtl_intake import IndependentHardwareIntake, RtlIntakeRefusal

        if type(hardware_intake) is not IndependentHardwareIntake or hardware_intake.target != target:
            raise RtlIntakeRefusal("fresh evidence requires the selected independently issued hardware authority")
        hardware_intake.verify()
        if facts_path is None:
            raise RtlIntakeRefusal("fresh evidence requires its explicit issued public facts member")
        hardware_intake.verify_public_facts(facts_path)
    if software_intake is not None:
        from merlin.development.testgen.software_intake import IndependentSoftwareIntake

        if type(software_intake) is not IndependentSoftwareIntake or software_intake.hardware is not hardware_intake:
            raise ValueError(
                "fresh software evidence requires the exact independent minimal source and hardware intake"
            )
        software_intake.verify_selected_source(software_spec)
    if source_components is not False:
        from merlin.development.testgen.component_source_binding import require_live_selection

        require_live_selection(source_components, software_intake, hardware_intake, capability_contract_path)

    sources: dict[Path, EvidenceSource] = {}
    diagnostics: list[dict[str, Any]] = []
    excluded = {".git", "__pycache__", "build", ".venv", "out"}
    extensions = {".py", ".json", ".yaml", ".yml", ".mlir", ".h", ".hpp", ".cpp", ".c", ".inc", ".S"}

    def observe(path, role, *, required=False) -> bytes | None:
        path = Path(path).absolute()
        if path.name == ".env" or path.name.startswith(".env."):
            raise ValueError("environment configuration must not enter a Phase 0 evidence bundle")
        if path in sources:
            return sources[path].content
        try:
            raw = path.read_bytes()
        except (FileNotFoundError, IsADirectoryError):
            if required:
                raise
            diagnostics.append({"component": role, "status": "unknown", "reason": f"unavailable: {path}"})
            return None
        sources[path] = EvidenceSource(path, role, raw)
        return raw

    def document(value, role, *, required=True) -> dict:
        if value is None:
            return {}
        if isinstance(value, Mapping):
            return copy.deepcopy(dict(value))
        raw = observe(value, role, required=required)
        if raw is None:
            return {}
        loaded = yaml.safe_load(raw)
        if not isinstance(loaded, dict):
            raise ValueError(f"{value}: {role} must be a mapping")
        return loaded

    descriptor_path = getattr(descriptor, "path", descriptor)
    descriptor_doc = document(descriptor_path, "descriptor") if descriptor_path is not None else {}
    if hardware_intake is not None and descriptor_doc.get("isa_headers"):
        raise RtlIntakeRefusal("fresh evidence may not select an ISA header transcription")
    application_inventory = None
    frontend_traces, application_graphs, framework_catalogs, host_capabilities = {}, {}, {}, {}
    inventory_identity = {"status": "not_available", "reason": "no selected detailed application inventory"}
    capture_attestations: dict[str, Any] = {}
    whole_program_admission = None
    if conformance_spec is not None or inventory_path is not None:
        from merlin.development.testgen.profiles import application_inventory_path

        requirement = (
            document(conformance_spec, "conformance-spec", required=False) if conformance_spec is not None else {}
        )
        if requirement.get("target", target) != target:
            raise ValueError("selected conformance requirement target differs from selected target")
        sidecar = (
            application_inventory_path(conformance_spec, document=requirement) if conformance_spec is not None else None
        )
        if inventory_path is not None:
            selected_inventory = Path(inventory_path).absolute()
            if conformance_spec is not None and (sidecar is None or sidecar.absolute() != selected_inventory):
                raise ValueError("explicit application inventory differs from selected conformance sidecar")
            if selected_inventory.is_symlink():
                raise ValueError("selected application inventory may not be a symlink")
            sidecar = selected_inventory
        declared_whole_program = requirement.get("whole_program_admission")
        if conformance_spec is not None and isinstance(declared_whole_program, Mapping):
            from merlin.development.testgen.program_admission import operations_digest

            name = declared_whole_program.get("sidecar")
            if not isinstance(name, str) or not name or Path(name).name != name:
                raise ValueError("whole-program admission sidecar must be one adjacent basename")
            location = Path(conformance_spec).parent / name
            if location.is_symlink():
                raise ValueError("whole-program admission sidecar may not be a symlink")
            raw_whole_program = observe(location, "whole-program-admission", required=True)
            whole_program_admission = json.loads(raw_whole_program)
            if operations_digest(whole_program_admission) != declared_whole_program.get("sha256"):
                raise ValueError("whole-program admission sidecar differs from the requirement's commitment")
        if conformance_spec is not None and requirement.get("public_program_sources") is not None:
            from merlin.development.testgen.public_programs import observe_inventory

            observe_inventory(requirement, conformance_spec, observe)
        if sidecar is not None:
            raw_inventory = observe(sidecar, "application-inventory", required=False)
            if raw_inventory is not None:
                application_inventory = json.loads(raw_inventory)
                if not isinstance(application_inventory, dict):
                    raise ValueError("selected detailed application inventory must be a mapping")
                actual = _canonical_digest(application_inventory)
                expected = (requirement.get("application_demands") or {}).get("full_inventory_sha256")
                if expected is not None and expected != actual:
                    raise ValueError("selected application inventory differs from conformance requirement")
                declared = (descriptor_doc.get("workload_spec") or {}).get("applications") or []
                observed = application_inventory.get("applications")
                if not isinstance(observed, dict):
                    raise ValueError("selected detailed application inventory requires an application mapping")
                # A directory selector is not an immutable list of model labels.
                # Do not enumerate it here or silently treat its characters as labels.
                labels = [str(label) for label in declared] if isinstance(declared, (list, tuple, dict)) else None
                roster_matches = labels is not None and len(labels) == len(set(labels)) and set(observed) == set(labels)
                inventory_identity = {
                    "status": "digest_bound" if expected is not None and roster_matches else "unverified",
                    "raw_sha256": _digest(raw_inventory),
                    "content_sha256": actual,
                    "requirement_digest_present": expected is not None,
                    "declared_roster_matches": roster_matches,
                    "declared_applications": sorted(labels) if labels is not None else None,
                    "observed_applications": sorted(observed),
                }
                # Attestations issued at derivation travel with the byte-bound
                # requirement; the coverage gate re-verifies each one from disk.
                capture_attestations = requirement.get("capture_execution_attestations") or {}
                if not isinstance(capture_attestations, dict) or (
                    capture_attestations and set(capture_attestations) != set(observed)
                ):
                    raise ValueError("capture execution attestations must cover exactly the selected applications")
                from merlin.development.targetgen.application_inventory import application_graph_inventory

                for label, application in sorted(observed.items()):
                    # Only fresh inventories retain the exact capture owner.
                    # Do not guess historical store paths from basename tags.
                    selected_capture = application.get("capture_source_path")
                    if not isinstance(selected_capture, str) or not selected_capture:
                        continue
                    capture_path = Path(selected_capture)
                    if not capture_path.is_absolute():
                        capture_path = sidecar.parent / capture_path
                    capture_raw = observe(capture_path, f"application-capture:{label}")
                    if capture_raw is None:
                        continue
                    if _digest(capture_raw) != application.get("capture_sha256"):
                        raise ValueError(f"selected application capture bytes changed: {label}")
                    application_graphs[label] = application_graph_inventory(capture_path)
                    if application_graphs[label]["capture_sha256"] != _digest(capture_raw):
                        raise ValueError(f"application graph was derived from changed source bytes: {label}")
                    normalized = (application.get("capture_normalization") or {}).get("output_sha256")
                    if normalized and application_graphs[label]["normalized_mlir_sha256"] != normalized:
                        raise ValueError(
                            f"selected application normalization differs from the derived inventory: {label}; "
                            "use the same frontend/MLIR environment for derivation and Phase 0 execution"
                        )
                    for name, filename, destination in (
                        ("frontend-trace", "frontend-trace.json", frontend_traces),
                        ("framework-catalog", "pytorch-opset.json", framework_catalogs),
                    ):
                        record = application.get(name.replace("-", "_")) or {}
                        location = (
                            (record.get("source_path") or record.get("path")) if isinstance(record, Mapping) else None
                        )
                        member = Path(location) if location else capture_path.with_name(filename)
                        if not member.is_absolute():
                            member = capture_path.parent / member
                        if member.is_symlink():
                            raise ValueError(f"selected {name} may not be symlinked: {label}")
                        raw_member = observe(member, f"application-{name}:{label}")
                        if raw_member is None:
                            continue
                        expected_member = (
                            (record.get("raw_sha256") or record.get("sha256")) if isinstance(record, Mapping) else None
                        )
                        if expected_member is not None and expected_member != _digest(raw_member):
                            raise ValueError(f"selected application {name} bytes changed: {label}")
                        parsed = json.loads(raw_member)
                        if not isinstance(parsed, dict):
                            raise ValueError(f"selected application {name} must be a mapping: {label}")
                        destination[label] = parsed
    if inventory_identity["status"] != "digest_bound":
        diagnostics.append(
            {
                "component": "application-inventory",
                "status": "unknown",
                "reason": inventory_identity.get("reason", "inventory digest or declared roster is unverified"),
            }
        )
    hardware_doc = document(hardware_spec, "hardware-spec")
    if hardware_doc.get("target", target) != target:
        raise ValueError("hardware spec target differs from selected target")
    if "schema" in hardware_doc and hardware_doc["schema"] != "merlin.hardware_selection.v1":
        raise ValueError("unsupported hardware selection schema")
    software_doc = document(software_spec, "software-spec")
    contract = (
        document(capability_contract_path, "target-contract", required=True)
        if capability_contract_path is not None
        else {}
    )
    # The shared runtime family selects data, not a compiler-bearing provider.
    data_only = (contract.get("runner") or {}).get("backend") == "chipyard_rocc"
    residual, provider = {}, None
    try:
        provider = target_registry.resolve(target) if hardware_intake is None and not data_only else None
    except (KeyError, FileNotFoundError, ValueError) as exc:
        diagnostics.append({"component": "support", "status": "unknown", "reason": str(exc)})
    if (
        provider is not None
        and getattr(provider, "kind", None) == "generated"
        and not provider.contract_path.is_file()
        and not provider.contract_path.is_symlink()
    ):
        # Registry discovery also names the *future* generated-package home. Until it has been
        # materialized it is not a selected support implementation, and reading it as one makes
        # Phase 0 inventory a phantom provider.
        diagnostics.append(
            {
                "component": "support",
                "status": "unknown",
                "reason": f"generated support contract is not materialized: {provider.contract_path}",
            }
        )
        provider = None
    if provider is not None:
        # Capability input selection is independent of executable support ownership.
        # The existing explicit contract selector must not be ignored merely because
        # support code was selected from an external provider.
        if capability_contract_path is None:
            contract = document(rtl_facts.target_contract_path(target), "target-contract", required=False)
        residual = document(provider.base / "contracts" / "residual.yaml", "residual", required=False)
        # Bind support code and its declarative/header dependencies before hooks
        # execute. This is an inventory, not a candidate grant or qualification.
        for path in sorted(provider.base.rglob("*")):
            if (
                path.is_file()
                and path.suffix in extensions
                and not excluded.intersection(path.relative_to(provider.base).parts)
            ):
                observe(path, "support-source")
    if contract and contract.get("name") != target:
        raise ValueError("selected backend capability contract differs from selected target")
    datapath = {}
    if software_spec is not None:
        from merlin.development.targetgen.software_spec import (
            capability_contract,
            numerical_datapath,
            software_spec_references,
            validate_software_spec,
        )

        software_doc = validate_software_spec(software_doc, target=target, source=str(software_spec))
        if not isinstance(software_spec, Mapping):
            try:
                references = software_spec_references(software_spec, document=software_doc)
            except ValueError as exc:
                references = {}
                diagnostics.append({"component": "numerical-model", "status": "unknown", "reason": str(exc)})
            for role, path in references.items():
                if path.is_dir():
                    for leaf in _reference_inventory_members(role, path, software_doc, extensions, excluded):
                        observe(leaf, f"software-reference:{role}")
                else:
                    observe(path, f"software-reference:{role}", required=True)
        if source_components:
            from merlin.development.testgen.component_source_binding import backend_unknown

            diagnostics.append(backend_unknown(software_doc, contract=contract))
        else:
            contract = capability_contract(software_doc, base_contract=contract)
        datapath = numerical_datapath(software_doc)
    if provider is not None and capability_contract_path is not None:
        _record_scaling_contradictions(
            diagnostics,
            contract,
            observe(provider.contract_path, "support-contract", required=True),
            target=target,
            selected_source=None
            if isinstance(capability_contract_path, Mapping)
            else sources.get(Path(capability_contract_path).absolute()),
        )
    # The hardware selection is a source declaration, not an inferred dtype.
    for field, value in hardware_doc.items():
        if field.endswith("_path") and isinstance(value, str):
            owner = Path(hardware_spec).parent if not isinstance(hardware_spec, Mapping) else Path.cwd()
            observe(owner / value, f"hardware-reference:{field}")
    try:
        selected_path = None if data_only and facts_path is None else rtl_facts.find_facts(target, explicit=facts_path)
    except FileNotFoundError as exc:
        selected_path = None
        diagnostics.append({"component": "facts", "status": "unknown", "reason": str(exc)})
    raw_facts = observe(selected_path, "rtl-facts") if selected_path is not None else None
    loaded_facts = json.loads(raw_facts) if raw_facts is not None else {}
    if not isinstance(loaded_facts, dict) or not isinstance(loaded_facts.get("facts", {}), dict):
        raise ValueError("selected RTL facts must contain a facts mapping")
    if not loaded_facts.get("facts"):
        diagnostics.append({"component": "facts", "status": "unknown", "reason": "no populated RTL facts selected"})
    instruction_semantics_source = None
    instruction_semantics = {
        "schema": "merlin.instruction_semantics.v1",
        "target": target,
        "status": "UNKNOWN",
        "unknowns": ["selected_target_contract_has_no_instruction_semantics_resource"],
        "instructions": [],
    }
    instruction_resource = contract.get("instruction_semantics")
    if hardware_intake is not None and instruction_resource is not None:
        raise RtlIntakeRefusal("fresh evidence may not select legacy instruction semantics resources")
    if instruction_resource is not None:
        if provider is None or not isinstance(instruction_resource, str) or not instruction_resource.strip():
            raise ValueError("instruction_semantics requires a selected support provider and relative resource")
        from merlin.compiler.backend.load import contained_resource
        from merlin.target.spec.instructions import normalize_instruction_semantics

        selected_instruction_path = contained_resource(provider.base, instruction_resource)
        instruction_semantics_source = observe(selected_instruction_path, "instruction-semantics", required=True)
        authored_instructions = yaml.safe_load(instruction_semantics_source)
        instruction_semantics = normalize_instruction_semantics(
            authored_instructions,
            software_spec=software_doc,
            rtl_facts=loaded_facts,
            target=target,
            source_bytes=instruction_semantics_source,
            software_source_bytes=(
                sources[Path(software_spec).absolute()].content
                if software_spec is not None and not isinstance(software_spec, Mapping)
                else None
            ),
            rtl_source_bytes=raw_facts,
        )
    # Only explicit full paths identify actual RTL owners. Basenames and widths
    # are not enough to reconstruct provenance or declare numeric support.
    inputs = loaded_facts.get("inputs") or {}
    body = loaded_facts.get("facts") or {}
    from merlin.development.testgen.evidence_rtl_sources import observe_sources

    source_relocations = observe_sources(
        loaded_facts,
        observe,
        diagnostics,
        relocations=rtl_source_relocations,
        facts_sha256=_digest(raw_facts) if raw_facts is not None else None,
    )
    for record in body.get("interfaces") or []:
        if isinstance(record, Mapping) and record.get("name") == "register_bundle_layouts" and record.get("source"):
            observe(record["source"], "readout-register-source")
    taxonomy = copy.deepcopy(body.get("isa_taxonomy") or {})
    if descriptor_path is not None and hardware_intake is None:
        from merlin.development.targetgen import isa_taxonomy
        from merlin.development.targetgen.target_experiment import load_target_experiment

        try:
            te = descriptor if hasattr(descriptor, "isa_headers") else load_target_experiment(descriptor_path)
            from merlin.common.paths import repo_root, resolve_grant

            source_root = te.source_root if te.source_root is not None else repo_root()
            board_catalog = te.selected_board_catalog()
            if board_catalog is not None:
                observe(board_catalog, "host-board-catalog", required=True)
            if te.host_lanes is not None:
                for name, lane in sorted(te.host_lanes.profiles.items()):
                    try:
                        capability_path, capabilities, identity = lane.resolve_capabilities(
                            root=source_root, descriptor=te.path
                        )
                        if capability_path is not None:
                            observe(capability_path, f"host-capability-spec:{name}", required=True)
                        host_capabilities[name] = {**identity, "capability_spec": capabilities}
                        # Preserve the selected immutable payload identity, not
                        # only the authored claim or a mutable checkout path.
                        package_root = source_root / lane.package
                        for member in sorted(package_root.rglob("*")):
                            if member.is_file():
                                observe(member, f"host-package:{name}", required=True)
                    except (OSError, ValueError) as exc:
                        host_capabilities[name] = {"status": "unknown", "reason": str(exc)}
                        diagnostics.append(
                            {"component": f"host-capabilities:{name}", "status": "unknown", "reason": str(exc)}
                        )
            if not data_only:
                headers = [str(resolve_grant(header, root=source_root)) for header in te.isa_headers]
                for header in headers:
                    observe(header, "isa-source")
                # The taxonomy's historical memo is path-keyed. A fresh selection
                # must derive from the source bytes just observed, not that memo.
                for source in sources.values():
                    if source.role == "isa-source":
                        isa_taxonomy._CACHE.pop(str(source.path), None)
                model = (contract.get("runner") or {}).get("model_ext")
                taxonomy = dict(
                    isa_taxonomy.derive_isa_taxonomy(
                        SimpleNamespace(target=target, isa_headers=headers), model_ext=model
                    )
                )
        except Exception as exc:  # noqa: BLE001 -- absent ISA tooling is unknown evidence
            taxonomy = {
                "status": "unknown",
                "by_class": {},
                "by_mnemonic": {},
                "asm_mnemonics": {},
                "unknown": {"taxonomy": f"{type(exc).__name__}: {exc}"},
            }
            diagnostics.append(
                {"component": "isa-taxonomy", "status": "unknown", "reason": f"{type(exc).__name__}: {exc}"}
            )
    # Replace physical shape declarations only where one extracted array
    # settles them. Never reinterpret a port width as numeric support.
    arrays = [array for array in body.get("arrays") or [] if isinstance(array, Mapping)]
    caps = contract.get("capabilities") or {}
    if isinstance(caps, dict) and any(key in caps for key in ("mesh", "tile")):
        for key in ("mesh", "tile"):
            declared = caps.get(key)
            if not isinstance(declared, dict):
                continue
            if len(arrays) == 1 and arrays[0].get("rows") and arrays[0].get("cols"):
                geometry = {field: arrays[0][field] for field in ("rows", "cols")}
                if any(declared.get(field) not in (None, value) for field, value in geometry.items()):
                    diagnostics.append(
                        {
                            "component": "physical-contract",
                            "status": "contradiction",
                            "reason": f"{key} declaration differs from selected array geometry",
                        }
                    )
                caps[key] = {**declared, **geometry}
            else:
                del caps[key]
                diagnostics.append(
                    {
                        "component": "physical-contract",
                        "status": "unknown",
                        "reason": f"{key} declaration has no unique selected array geometry",
                    }
                )
    snapshot_bytes = {str(path): source.content for path, source in sources.items()}
    for row in (source_relocations or {}).get("sources", []):
        snapshot_bytes[row["original_path"]] = snapshot_bytes[row["replacement_path"]]
    refreshed_facts = dict(readout_facet.with_current_register_layouts(loaded_facts, source_bytes=snapshot_bytes))
    readout_inputs = chipyard_rocc.readout_metadata(contract) if data_only else {}
    if hardware_intake is None and not data_only:
        with rtl_facts.observed_facts(target, refreshed_facts, selected_path):
            readout_inputs = readout_facet.capture_inputs(target, facts=refreshed_facts, include_taxonomy=False)
    readout_inputs["taxonomy"] = taxonomy
    facets = readout_facet.for_target(target, contract=contract, facts=refreshed_facts, readout_inputs=readout_inputs)
    # The selected contract's scale claim must agree with the selected RTL
    # readout. Keep a mismatch diagnostic: narrowing a declaration or certifying
    # the software semantics requires its own reviewed input and fresh run.
    units = [unit for unit in contract.get("compute_units") or () if isinstance(unit, Mapping)]
    selected_facets = facets if units else []
    for unit, facet in zip(units, selected_facets, strict=True):
        for finding in readout_facet.reconcile(unit, facet):
            diagnostics.append(
                {
                    "component": "readout-scaling",
                    "status": "contradiction" if finding["kind"] == "scaling_exceeds_readout" else "unknown",
                    "reason": finding["why"],
                    "finding": finding,
                    "contract_sha256": _canonical_digest(contract),
                    "raw_facts_sha256": _digest(raw_facts) if raw_facts is not None else None,
                    "readout_facet_sha256": _canonical_digest(facet.to_dict()),
                }
            )
    from merlin.compiler.quantization.capabilities import derive_candidates

    quantization_candidates = [candidate.to_dict() for candidate in derive_candidates(contract, facets)]
    # Fill the authored spec's fact-derived declarations from the same fact view the drift check
    # compares against, under the experiment's prohibited roles. Everything downstream of this
    # selection (screens, quantization contract, recipes, the frozen corpus) sees the resolved spec;
    # the authored bytes stay the selected source identity.
    software_derivation = None
    if software_spec is not None:
        from merlin.development.targetgen import spec_fact_drift

        # Some capability helpers still resolve by target name. Their reads
        # must use this selection too, never extract from an ambient provider.
        with (
            target_registry.observed_contract(target, contract),
            rtl_facts.observed_facts(target, refreshed_facts, selected_path),
        ):
            fact_view = spec_fact_drift.fact_capabilities(
                target=target,
                contract=contract,
                raw_facts=loaded_facts,
                readout_facets=[facet.to_dict() for facet in facets],
                quantization_candidates=quantization_candidates,
                taxonomy=taxonomy,
                prohibited_roles=prohibited_roles,
                **({"data_only": True} if data_only else {}),
            )
        software_doc, software_derivation = spec_fact_drift.resolve_spec(software_doc, fact_view)
        software_derivation["prohibited_instruction_roles"] = sorted(set(prohibited_roles))
        for row in software_derivation["unresolved"]:
            diagnostics.append(
                {
                    "component": "software-spec-derivation",
                    "status": "unknown",
                    "reason": row["reason"],
                    "declaration": row.get("id") or row.get("format"),
                }
            )
    target_profile = derive_profile(target, facts=refreshed_facts, residual=residual, contract=contract).to_dict()
    if hardware_intake is None:
        with rtl_facts.observed_facts(target, refreshed_facts, selected_path):
            execution = base.execution_capability_facts(
                target, **({"contract": contract, "facts": refreshed_facts} if data_only else {})
            )
    else:
        execution = base.independent_execution_capability_facts()
    performance_doc = {"target_profile": target_profile, "execution_capabilities": execution}
    performance_facts = {
        "target": target,
        "traits": target_profile["traits"],
        "execution_capabilities": execution,
        "target_profile_sha256": _canonical_digest(target_profile),
        "execution_capabilities_sha256": _canonical_digest(execution),
        "sha256": _canonical_digest(performance_doc),
        "raw_facts_sha256": _digest(raw_facts) if raw_facts is not None else None,
    }
    # These observations do not establish common elaboration or refmodel
    # equivalence. Keep absent production receipts visible in diagnostic runs.
    consistency = loaded_facts.get("source_consistency") or {}
    if inputs.get("source_bundle_path"):
        from merlin.development.targetgen.rtl import source_selection

        observe(Path(source_selection.__file__), "rtl-production-reader", required=True)
        try:
            production = source_selection.load_selection(inputs["source_bundle_path"], target=target)
            consistency = source_selection.production_consistency(production)
            from merlin.development.testgen.elaboration_evidence import snapshot_elaboration

            snapshot_elaboration(production, consistency, observe)
            # Generic serialization is a separate consumer edge after source
            # production. Validate it rather than discard its exact parser input
            # or falsely contradict a coherent source bundle with an added edge.
            genericization = (loaded_facts.get("source_consistency") or {}).get("genericization")
            if genericization is not None:
                from merlin.common.digest import sha256_file

                if not isinstance(genericization, Mapping):
                    raise ValueError("invalid CIRCT genericization receipt")
                source, output, tool = (genericization[key] for key in ("input", "output", "tool"))
                command = genericization["command"]
                if (
                    genericization.get("kind") != "circt_generic_serialization"
                    or genericization.get("returncode") != 0
                    or not _same_selected_file(source, production["sources"]["core_hw"])
                    or output != {"path": inputs.get("generic_hw_path"), "sha256": inputs.get("generic_hw_sha256")}
                    or not isinstance(command, list)
                    or len(command) != 5
                    or command[1] != "--mlir-print-op-generic"
                    or command[3] != "-o"
                    or Path(command[0]).resolve() != Path(tool["path"]).resolve()
                    or Path(command[2]).resolve() != Path(source["path"]).resolve()
                    or Path(command[4]).resolve() != Path(output["path"]).resolve()
                    or any(sha256_file(member["path"]) != member["sha256"] for member in (source, output, tool))
                ):
                    raise ValueError("CIRCT genericization source/output/tool binding differs")
                consistency["genericization"] = copy.deepcopy(genericization)
                consistency["sources"].append({"role": "core_hw_generic", **output})
            if not _same_source_consistency(consistency, loaded_facts.get("source_consistency") or {}):
                diagnostics.append(
                    {
                        "component": "source-consistency",
                        "status": "contradiction",
                        "reason": "recorded source consistency differs from independent validation",
                    }
                )
                consistency = {"status": "unverified"}
            historical = ((production.get("diagnostics") or {}).get("authored_hierarchy") or {}).get("path")
            if historical is not None:
                observe(historical, "rtl-historical-hierarchy", required=True)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            diagnostics.append(
                {
                    "component": "source-consistency",
                    "status": "contradiction",
                    "reason": f"production source validation failed: {exc}",
                }
            )
            consistency = {"status": "unverified"}
    if consistency.get("status") != "verified":
        diagnostics.append(
            {
                "component": "source-consistency",
                "status": "unknown",
                "reason": "no verified common-elaboration receipt selected",
            }
        )
    # Fresh facts are not enough: an rtl-source-audit report beside them must verify them, bound to the
    # exact facts and hardware-spec bytes selected here. And every selected application capture must
    # carry an admitted sealed-runner attestation that still matches its bytes on disk.
    from merlin.development.testgen.evidence_status import (
        AUDIT_MEMBER,
        capture_attestation_diagnostics,
        rtl_audit_diagnostic,
    )

    audit_path = Path(selected_path).with_name(AUDIT_MEMBER) if selected_path is not None else None
    audit_raw = observe(audit_path, "rtl-source-audit") if audit_path is not None and audit_path.is_file() else None
    hardware_source = (
        sources.get(Path(hardware_spec).absolute())
        if hardware_spec is not None and not isinstance(hardware_spec, Mapping)
        else None
    )
    audit_refusal = rtl_audit_diagnostic(
        audit_raw, facts_raw=raw_facts, hardware_raw=hardware_source.content if hardware_source else None
    )
    if audit_refusal is not None:
        diagnostics.append(audit_refusal)
    if application_inventory is not None:
        diagnostics.extend(
            capture_attestation_diagnostics(application_inventory.get("applications") or {}, capture_attestations)
        )
    if software_doc.get("status") != "reviewed":
        diagnostics.append(
            {
                "component": "software-review",
                "status": "unknown",
                "reason": "selected numerical software semantics are not reviewed",
            }
        )
    try:
        from merlin.development.targetgen.rtl.mlc_bridge import toolchain_identity

        observed_toolchain = toolchain_identity()
    except Exception as exc:  # noqa: BLE001 -- unavailable tool identity stays unknown
        observed_toolchain = {"unknown_reason": f"{type(exc).__name__}: {exc}"}
    toolchain_inputs = {
        "recorded_by_extractor": inputs.get("toolchain"),
        "observed_at_selection": observed_toolchain,
        "scope": "current tool observations do not identify the tools that produced unstamped facts",
    }
    from merlin.development.targetgen import aten_coverage

    # Observe the framework selected for capture, never a convenient host torch.
    # Replays use the serialized result below and do not relaunch this worker.
    framework_catalog = aten_coverage.observe_opset(python=capture_python, timeout=30)
    observe(Path(aten_coverage.__file__).with_name("_aten_opset_worker.py"), "framework-catalog-reader")
    if framework_catalog.get("status") != "available":
        diagnostics.append(
            {
                "component": "framework-catalog",
                "status": "unknown",
                "reason": "selected capture framework operator catalog is unavailable or incomplete",
            }
        )
    for field in hardware_doc.get("required_fields") or []:
        value = loaded_facts
        for part in str(field).split("."):
            value = value.get(part) if isinstance(value, Mapping) else None
        if value is None or value == [] or value == {}:
            diagnostics.append(
                {
                    "component": "required-hardware-field",
                    "status": "unknown",
                    "reason": f"selected facts do not establish {field}",
                }
            )
    native_baselines = (
        _native_baseline_observations(
            native_qualifications, (application_inventory or {}).get("applications", {}), observe
        )
        if native_qualifications
        else {}
    )
    for source in sources.values():
        if source.path.read_bytes() != source.content:
            raise ValueError(f"evidence source changed during selection: {source.path}")
    from merlin.development.testgen.evidence_status import status as evidence_status

    views = {
        "target": target,
        # Operator policy (evidence_status): verified only when nothing is on record against it.
        "status": evidence_status(diagnostics),
        "diagnostics": diagnostics,
        "descriptor": descriptor_doc,
        "hardware_spec": hardware_doc,
        "software_spec": software_doc,
        "software_spec_derivation": software_derivation,
        "instruction_semantics": instruction_semantics,
        "instruction_semantics_source_sha256": (
            _digest(instruction_semantics_source) if instruction_semantics_source is not None else None
        ),
        "datapath": datapath,
        "application_inventory": application_inventory,
        "application_inventory_identity": inventory_identity,
        "capture_execution_attestations": capture_attestations,
        "whole_program_admission": whole_program_admission,
        "frontend_traces": frontend_traces,
        "application_graphs": application_graphs,
        "framework_catalogs": framework_catalogs,
        "host_capabilities": host_capabilities,
        "native_baseline_observations": native_baselines,
        "loaded_facts": loaded_facts,
        "refreshed_facts": refreshed_facts,
        "contract": contract,
        "residual": residual,
        "target_profile": target_profile,
        "execution_capabilities": execution,
        "performance_facts": performance_facts,
        "readout_inputs": readout_inputs,
        "readout_facets": [facet.to_dict() for facet in facets],
        "readout_numerics": [facet.numerics_handoff() for facet in facets],
        "isa_taxonomy": taxonomy,
        "toolchain_inputs": toolchain_inputs,
        "framework_catalog": framework_catalog,
        "qualification_blockers": [
            "source-production consistency is checked when an explicit source bundle is selected; "
            "numerical reference-model qualification and reviewed operation support remain separate obligations"
        ],
        "quantization_snapshot": {
            "quantization_candidates": quantization_candidates,
            "numerical_semantics": copy.deepcopy(software_doc.get("numerical_semantics")),
            "readout_numerics": [facet.numerics_handoff() for facet in facets],
            "declared_units": [
                {key: copy.deepcopy(unit.get(key)) for key in ("name", "dtypes", "scaling", "requant")}
                for unit in contract.get("compute_units") or []
                if isinstance(unit, Mapping)
            ],
        },
    }
    if source_relocations is not None:
        views["rtl_source_relocations"] = source_relocations
    return EvidenceSelection(
        target,
        tuple(sources.values()),
        _json(views),
        raw_facts,
        instruction_semantics_source=instruction_semantics_source,
    )
