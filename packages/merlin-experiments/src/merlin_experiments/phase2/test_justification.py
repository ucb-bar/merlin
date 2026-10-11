"""Byte-bound rationale for generated Phase 2 tests; no measurements are inferred.

The receipt connects each development capsule to the Phase 0 machine observation,
workload census, falsifiable claim and planned measurement. It is an input audit,
not a claim analyzer or a simulator result.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path

import yaml

from merlin.targetgen.application_inventory import verified_static_integerization
from merlin.targetgen.performance_match import match_operation_form
from merlin_experiments.phase0.performance_scope import validate_performance_scope
from merlin_experiments.phase0.profiles import validate_host_source_basis
from merlin_experiments.phase2 import contracts as C
from merlin_experiments.phase2.contracts import StageGateError

_GENERATOR = "merlin/contract/capsules/generate_corpus.py"
_FACTS = "hardware/effective-views/performance-facts.json"
_ACCOUNTING = "coverage/operation-accounting.json"
_BASIS = "coverage/performance-basis.json"
_HARDWARE_SPEC = "software/hardware-spec.json"
_SOFTWARE_SPEC = "software/software-spec.json"
_CONTRACT = "software/contract.json"


def _host_basis(descriptor, directory: Path, gate) -> dict | None:
    """Reopen prepared host source premises without claiming executed host cost."""
    selected = gate.get("source_execution_basis")
    if selected is None:
        return None
    try:
        validate_host_source_basis(selected, owner="Phase 2 test rationale")
    except ValueError as exc:
        raise StageGateError(str(exc)) from exc
    from merlin_experiments.phase0.public_programs import BASIS_SCHEMA, verify_written

    if selected["schema"] == BASIS_SCHEMA:
        try:
            return verify_written(descriptor, directory)
        except (ValueError, KeyError, TypeError, OSError) as exc:
            raise StageGateError(
                "public program rationale differs from its original complete source/reference"
            ) from exc
    preparation = (descriptor.get("performance") or {}).get("source_preparation") or {}
    screen = descriptor.get("source_semantics_screen") or {}
    program = descriptor.get("component_program") or {}
    operation = descriptor.get("operation") or {}
    original = (operation.get("attributes") or {}).get("program")
    source = directory / "capsule.interface.mlir"
    from merlin.targetgen.component_program import render

    storage = program.get("selected_storage") or {}
    try:
        typed, expected = render(original, operand_dtype=storage["operand"], accumulator_dtype=storage["accumulator"])
    except (ValueError, KeyError, TypeError) as exc:
        raise StageGateError("host rationale cannot reopen its complete typed source") from exc
    operations = sorted({{"alias": "copy", "update": "add"}.get(node["op"], node["op"]) for node in typed["nodes"]})
    if (
        preparation.get("schema") != "merlin.phase0.source_performance_preparation.v2"
        or preparation.get("hardware_admission") != "not_established"
        or preparation.get("measurement_admission") != "not_established"
        or screen.get("status") != "source_admitted"
        or screen.get("software_intake_sha256") != selected["software_intake_sha256"]
        or sorted(selected["operations"]) != operations
        or operation.get("op") != "component_program"
        or typed != program
        or source.is_symlink()
        or source.read_bytes() != expected.encode()
        or screen.get("program_sha256") != C.sha256_file(source)
    ):
        raise StageGateError("host rationale differs from the original independently prepared source basis")
    golden = directory / "golden.yaml"
    if not golden.is_file() or golden.is_symlink():
        raise StageGateError("host rationale lacks its complete independent numerical reference")
    return {
        "selection": selected,
        "source_sha256": C.sha256_file(source),
        "typed_program_sha256": C.document_sha256(program),
        "reference_sha256": C.sha256_file(golden),
        "status": "source_prepared_execution_unknown",
        "scope": "original source/reference only; execution, placement, complete cost and physical effects unknown",
    }


def scope_projection_from_requirement_bytes(
    payload: bytes, *, target: str, raw_facts_sha256: str, selected_applications: set[str]
) -> dict:
    """Retain only public connected-slice classifications from a selected source requirement."""
    requirement = yaml.safe_load(payload)
    if not isinstance(requirement, Mapping) or requirement.get("target") != target:
        raise StageGateError("selected conformance requirement target differs")
    derivation = (requirement.get("derivation") or {}).get("phase0_execution") or {}
    if derivation.get("raw_facts_sha256") != raw_facts_sha256:
        raise StageGateError("selected conformance requirement RTL facts differ")
    scope = requirement.get("scope")
    if not isinstance(scope, dict):
        raise StageGateError("selected conformance requirement lacks connected-slice scope")
    try:
        performance = validate_performance_scope(scope)
        if any(
            not isinstance(row, Mapping) or row.get("application") not in selected_applications
            for row in scope["typed_required_instances"]["instances"]
        ):
            raise ValueError("typed source scope names an application outside selected public captures")
        if any(not set(row.get("observed_in") or []).issubset(selected_applications) for row in scope["required"]):
            raise ValueError("raw source scope names an application outside selected public captures")
        projection = {
            "required": [{key: row[key] for key in ("signature", "occurrences")} for row in scope["required"]],
            "typed_required_instances": {
                "schema": "merlin.phase0.typed_scope_instances.v1",
                "instances": [
                    {key: row[key] for key in ("instance_id", "signature")}
                    for row in scope["typed_required_instances"]["instances"]
                ],
            },
            "performance": {
                "schema": "merlin.phase0.performance_scope.v1",
                "status": performance["status"],
                "required": [
                    {key: row[key] for key in ("signature", "occurrences", "instance_ids")}
                    for row in performance["required"]
                ],
                "excluded": [
                    {key: row[key] for key in ("instance_id", "signature", "status", "reason")}
                    for row in performance["excluded"]
                ],
                "unresolved": [
                    {key: row[key] for key in ("instance_id", "signature", "status", "reason")}
                    for row in performance["unresolved"]
                ],
            },
        }
        validate_performance_scope(projection)
    except (KeyError, TypeError, ValueError) as exc:
        raise StageGateError(f"selected conformance requirement has invalid performance scope: {exc}") from exc
    return projection


def _selected_scope_source(
    root: Path, evidence: Mapping, *, target: str, raw_facts_sha256: str, selected_applications: set[str]
) -> tuple[dict, dict] | None:
    sources = evidence.get("sources") or []
    if not isinstance(sources, list):
        raise StageGateError("Phase 0 source snapshot index is malformed")
    selected = [row for row in sources if isinstance(row, Mapping) and row.get("role") == "conformance-spec"]
    if not selected:
        return None
    if len(selected) != 1:
        raise StageGateError("Phase 0 has ambiguous selected conformance snapshots")
    row = selected[0]
    relative = Path(str(row.get("path") or ""))
    if relative.is_absolute() or len(relative.parts) != 3 or relative.parts[:2] != ("software", "source-snapshots"):
        raise StageGateError("selected conformance snapshot path is unsafe")
    path = root / relative
    if any(parent.is_symlink() for parent in (path, path.parent, path.parent.parent)) or not path.is_file():
        raise StageGateError("selected conformance snapshot is absent or linked")
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    artifact = (evidence.get("artifacts") or {}).get(relative.as_posix())
    if (
        digest != row.get("sha256")
        or len(payload) != row.get("size_bytes")
        or not isinstance(artifact, Mapping)
        or artifact.get("sha256") != digest
        or artifact.get("size_bytes") != len(payload)
    ):
        raise StageGateError("selected conformance snapshot differs from Phase 0 evidence")
    projection = scope_projection_from_requirement_bytes(
        payload,
        target=target,
        raw_facts_sha256=raw_facts_sha256,
        selected_applications=selected_applications,
    )
    return projection, {
        "selected_requirement_sha256": digest,
        "selected_scope_sha256": hashlib.sha256(C.canonical_json(projection)).hexdigest(),
    }


def requires_justification(provenance: Mapping) -> bool:
    """Current Phase 0 output is gated; old synthetic fixtures stay inspectable."""
    return provenance.get("generated_by") == _GENERATOR


def selected_evidence_root(corpus) -> Path:
    """Resolve the two Phase 0 evidence layouts without trusting an arbitrary path.

    Installed Phase 0 can put the exported evidence beside ``capsules/``;
    checkout generation defaults to ``capsules/_evidence/``. The generated,
    hash-bound provenance must identify the selected layout exactly.
    """
    default = Path(corpus.corpus_root) / "_evidence"
    manifest = C.mapping_file(corpus.provenance_manifest, yaml_file=True)
    declaration = manifest.get("phase0_evidence") or {}
    selected = declaration.get("manifest") if isinstance(declaration, Mapping) else None
    if selected is None:
        root = default
    else:
        source = Path(str(selected))
        if not source.is_absolute():
            source = Path(corpus.corpus_root) / source
        root = source.parent
        if source.name != "evidence-manifest.json":
            raise StageGateError("Phase 0 evidence declaration has an unexpected manifest name")
    corpus_root = Path(corpus.corpus_root)
    allowed = {default.absolute(), corpus_root.absolute().parent}
    if (
        corpus_root.is_symlink()
        or corpus_root.resolve(strict=True) != corpus_root.absolute()
        or root.absolute() not in allowed
        or root.is_symlink()
        or not root.is_dir()
        or root.resolve(strict=True) != root.absolute()
    ):
        raise StageGateError("Phase 2 evidence root is absent or outside the Phase 0 output")
    manifest_path = root / "evidence-manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise StageGateError("Phase 2 evidence manifest is absent or linked")
    return root


def _selected_evidence(corpus) -> tuple[dict, dict, dict, dict, dict, dict | None]:
    root = selected_evidence_root(corpus)
    manifest_path = root / "evidence-manifest.json"
    evidence = C.mapping_file(manifest_path)
    facts_path, accounting_path, basis_path = root / _FACTS, root / _ACCOUNTING, root / _BASIS
    facts, accounting, basis = (
        C.mapping_file(facts_path),
        C.mapping_file(accounting_path),
        C.mapping_file(basis_path),
    )
    artifacts = evidence.get("artifacts")
    generation = corpus.performance_generation
    declared = generation.get("facts") if isinstance(generation, Mapping) else None
    template = generation.get("shared_template") if isinstance(generation, Mapping) else None
    if (
        evidence.get("target") != corpus.target
        or facts.get("target") != corpus.target
        or not isinstance(artifacts, Mapping)
        or not isinstance(declared, Mapping)
        or not isinstance(template, Mapping)
        or not isinstance(template.get("sha256"), str)
        or len(template["sha256"]) != 64
        or declared.get("target") != corpus.target
        or declared.get("sha256") != facts.get("sha256")
        or declared.get("raw_facts_sha256") != facts.get("raw_facts_sha256")
        or evidence.get("performance_facts_sha256") != facts.get("sha256")
        or evidence.get("raw_facts_sha256") != facts.get("raw_facts_sha256")
    ):
        raise StageGateError("Phase 2 test rationale lacks coherent frozen Phase 0 source/fact identity")
    for relative, path in ((_FACTS, facts_path), (_ACCOUNTING, accounting_path), (_BASIS, basis_path)):
        recorded = artifacts.get(relative)
        if (
            not isinstance(recorded, Mapping)
            or recorded.get("sha256") != C.sha256_file(path)
            or recorded.get("size_bytes") != path.stat().st_size
        ):
            raise StageGateError(f"Phase 2 test rationale evidence bytes changed: {relative}")
    if accounting.get("schema") != "merlin.phase0.operation_accounting.v1":
        raise StageGateError("Phase 2 test rationale lacks a Phase 0 operation census")
    basis_sources = basis.get("sources")
    consumers = evidence.get("consumers")
    selected_views = (_HARDWARE_SPEC, _SOFTWARE_SPEC, _CONTRACT)
    for relative in selected_views:
        path = root / relative
        recorded = artifacts.get(relative)
        if (
            not isinstance(recorded, Mapping)
            or recorded.get("sha256") != C.sha256_file(path)
            or recorded.get("size_bytes") != path.stat().st_size
        ):
            raise StageGateError(f"Phase 2 test rationale selected view bytes changed: {relative}")
    hardware_path, software_path, contract_path = (root / relative for relative in selected_views)
    if (
        basis.get("schema") != "merlin.phase0.performance_basis.v1"
        or basis.get("target") != corpus.target
        or basis.get("status") != accounting.get("status")
        or not isinstance(basis_sources, Mapping)
        or not isinstance(basis.get("applications"), Mapping)
        or not isinstance(basis.get("overall"), Mapping)
        or basis.get("selected_inventory") != accounting.get("selected_inventory")
        or not isinstance(consumers, Mapping)
        or _BASIS not in (consumers.get("performance_basis") or ())
        or basis_sources.get("operation_accounting_sha256") != C.sha256_file(accounting_path)
        or basis_sources.get("performance_facts_artifact_sha256") != C.sha256_file(facts_path)
        or basis_sources.get("performance_facts_sha256") != facts.get("sha256")
        or basis_sources.get("raw_rtl_facts_sha256") != facts.get("raw_facts_sha256")
        or basis_sources.get("selected_inventory_sha256") != accounting.get("inventory_sha256")
        or basis_sources.get("selected_software_spec_sha256") != accounting.get("selected_software_spec_sha256")
        or (
            accounting.get("selected_software_spec_sha256") is not None
            and accounting["selected_software_spec_sha256"] != C.document_sha256(C.mapping_file(software_path))
        )
        or basis_sources.get("selected_capability_contract_sha256")
        != accounting.get("selected_capability_contract_sha256")
        or (
            accounting.get("selected_capability_contract_sha256") is not None
            and accounting["selected_capability_contract_sha256"] != C.document_sha256(C.mapping_file(contract_path))
        )
        or basis_sources.get("selected_hardware_spec_sha256") != C.document_sha256(C.mapping_file(hardware_path))
    ):
        raise StageGateError("Phase 2 test rationale lacks a coherent frozen performance basis")
    if set(basis["applications"]) != set(accounting.get("applications") or {}):
        raise StageGateError("Phase 2 performance basis application roster differs from operation accounting")
    for label, selected in basis["applications"].items():
        recorded = accounting["applications"][label]
        if (
            not isinstance(selected, Mapping)
            or not isinstance(recorded, Mapping)
            or any(
                selected.get(key) != recorded.get(key)
                for key in ("capture_sha256", "capture_receipt", "capture_quantization", "capture_integerization")
            )
        ):
            raise StageGateError("Phase 2 performance basis capture evidence differs from operation accounting")
    selected_scope = (
        _selected_scope_source(
            root,
            evidence,
            target=corpus.target,
            raw_facts_sha256=facts["raw_facts_sha256"],
            selected_applications=set(basis["applications"]),
        )
        if basis.get("status") != "not_available"
        else None
    )
    return (
        evidence,
        facts,
        accounting,
        basis,
        {
            "manifest_sha256": C.sha256_file(manifest_path),
            "performance_facts_sha256": C.sha256_file(facts_path),
            "operation_accounting_sha256": C.sha256_file(accounting_path),
            "performance_basis_sha256": C.sha256_file(basis_path),
            "shared_template_sha256": template["sha256"],
            "derived_performance_facts_sha256": facts["sha256"],
            "raw_rtl_facts_sha256": facts["raw_facts_sha256"],
            **(selected_scope[1] if selected_scope is not None else {}),
        },
        selected_scope[0] if selected_scope is not None else None,
    )


def verify_live_justification(corpus, receipt: Mapping) -> dict | None:
    """Recheck all original inputs before a selected cohort is copied."""
    _, _, _, _, source, scope = _selected_evidence(corpus)
    expected = receipt.get("source") or {}
    if any(expected.get(key) != value for key, value in source.items()):
        raise StageGateError("Phase 2 test rationale source evidence changed before freeze")
    for key in ("selected_requirement_sha256", "selected_scope_sha256"):
        if (key in expected) != (key in source):
            raise StageGateError("Phase 2 selected performance scope source changed before freeze")
    if expected.get("provenance_sha256") != C.sha256_file(corpus.provenance_manifest):
        raise StageGateError("Phase 2 test rationale provenance changed before freeze")
    if expected.get("performance_generation_sha256") != C.document_sha256(corpus.performance_generation):
        raise StageGateError("Phase 2 test rationale generation changed before freeze")
    for row in receipt.get("members") or []:
        relative = Path(str(row.get("relative_path") or ""))
        if relative.is_absolute() or ".." in relative.parts or len(relative.parts) != 2:
            raise StageGateError("Phase 2 test rationale contains an unsafe member path")
        if C.exact_tree_record(Path(corpus.corpus_root) / relative)["sha256"] != row.get("capsule_tree_sha256"):
            raise StageGateError(f"Phase 2 test rationale member changed before freeze: {relative}")
    return scope


def _basis_rows(basis: Mapping) -> tuple[list[tuple[str, int, Mapping]], int | None, int | None]:
    """Check the numeric census while retaining rows with unpriced work."""
    rows: list[tuple[str, int, Mapping]] = []
    observed_known = observed_unknown = 0
    for application, record in sorted(basis["applications"].items()):
        if not isinstance(application, str) or not isinstance(record, Mapping):
            raise StageGateError("Phase 2 performance basis has a malformed application")
        signatures = record.get("rows")
        if not isinstance(signatures, list) or record.get("n_signatures") != len(signatures):
            raise StageGateError("Phase 2 performance basis has incomplete application rows")
        receipt = record.get("capture_receipt") or {}
        integerization = record.get("capture_integerization")
        capture_identity = integerization.get("capture") if isinstance(integerization, Mapping) else None
        source_integerization_byte_bound = (
            isinstance(receipt, Mapping)
            and receipt.get("status") == "verified_materialized"
            and receipt.get("errors") == []
            and isinstance(integerization, dict)
            and integerization.get("source_quantization") == record.get("capture_quantization")
            and isinstance(capture_identity, Mapping)
            and capture_identity.get("sha256") == record.get("capture_sha256")
            and verified_static_integerization(integerization, receipt_sha256=receipt.get("receipt_sha256"))
        )
        for index, row in enumerate(signatures):
            if not isinstance(row, Mapping) or type(row.get("count")) is not int or row["count"] <= 0:
                raise StageGateError("Phase 2 performance basis has a malformed demand row")
            if row.get("application") != application or row.get("capture_sha256") != record.get("capture_sha256"):
                raise StageGateError("Phase 2 performance basis demand source identity differs")
            macs = row.get("macs")
            if not isinstance(macs, Mapping):
                raise StageGateError("Phase 2 performance basis demand lacks MAC accounting")
            total = macs.get("total")
            if total is not None and (type(total) is not int or total < 0):
                raise StageGateError("Phase 2 performance basis has an invalid known MAC mass")
            if row.get("independent_compute_demand") is True:
                if total is not None:
                    observed_known += total
                elif row.get("semantic_family") == "contraction":
                    observed_unknown += row["count"]
            rows.append(
                (application, index, {**row, "source_integerization_byte_bound": source_integerization_byte_bound})
            )
        if sum(row["count"] for row in signatures) != record.get("n_operations"):
            raise StageGateError("Phase 2 performance basis application operation count differs")
    overall = basis["overall"]
    if basis.get("status") == "not_available":
        if rows or overall.get("known_macs") is not None or overall.get("unknown_macs_demands") is not None:
            raise StageGateError("Phase 2 performance basis invents workload mass without an inventory")
        return rows, None, None
    if (
        type(overall.get("known_macs")) is not int
        or type(overall.get("unknown_macs_demands")) is not int
        or overall["known_macs"] != observed_known
        or overall["unknown_macs_demands"] != observed_unknown
    ):
        raise StageGateError("Phase 2 performance basis MAC totals differ from its demand rows")
    return rows, observed_known, observed_unknown


def derive_justification(corpus) -> dict:
    """Derive a deterministic plan for every admitted generated performance member.

    The caller supplies the *complete* discovered Phase 2 cohort. A selected subset
    cannot silently omit a sibling needed by a comparison group or negative control.
    """
    provenance = C.mapping_file(corpus.provenance_manifest, yaml_file=True)
    if not requires_justification(provenance):
        raise StageGateError("test justification requires current generated Phase 0 provenance")
    if C.sha256_file(corpus.provenance_manifest) != corpus.provenance_sha256:
        raise StageGateError("performance provenance changed before test justification")
    evidence, facts, accounting, basis, source, _scope = _selected_evidence(corpus)
    phase = corpus.performance_generation.get("phase") or {}
    generated = {
        str(item) for item in provenance.get("generated") or [] if Path(str(item)).parts[:1] == (phase.get("category"),)
    }
    present = {member.source_relative_path for member in corpus.capsules}
    if not generated or present != generated:
        raise StageGateError("test justification requires every generated Phase 2 capsule")
    basis_rows, known_macs, unknown_macs_demands = _basis_rows(basis)
    represented_rows: set[tuple[str, int]] = set()
    candidate_rows: set[tuple[str, int]] = set()
    candidates_by_row: dict[tuple[str, int], dict[str, str]] = defaultdict(dict)
    roles_by_group: dict[tuple[str, str], list[tuple[str, str]]] = defaultdict(list)
    family_claims: dict[str, str] = {}
    for member in corpus.capsules:
        performance = member.descriptor.get("performance") or {}
        stable_fields = (
            "level",
            "family",
            "lever",
            "member_class",
            "claim",
            "comparand",
            "falsifier",
            "gate",
            "regime",
            "cost",
            "acceptance",
            "noise_band",
        )
        claim_sha = C.document_sha256({key: performance.get(key) for key in stable_fields})
        if member.family in family_claims and family_claims[member.family] != claim_sha:
            raise StageGateError(f"generated performance family has divergent claim contracts: {member.family}")
        family_claims[member.family] = claim_sha
        group = member.descriptor.get("comparison_group") or {}
        if isinstance(group, Mapping) and group.get("name") and group.get("role"):
            roles_by_group[(member.family, str(group["name"]))].append((str(group["role"]), member.capsule))
    rows = []
    has_host_basis = False
    for member in sorted(corpus.capsules, key=lambda item: (item.family, item.capsule)):
        if C.exact_tree_record(member.source_dir)["sha256"] != member.source_sha256:
            raise StageGateError(f"performance member changed before test justification: {member.capsule}")
        descriptor = member.descriptor
        performance = descriptor.get("performance") or {}
        gate, comparand, falsifier = (performance.get(key) or {} for key in ("gate", "comparand", "falsifier"))
        if any(not isinstance(value, Mapping) for value in (gate, comparand, falsifier)):
            raise StageGateError(f"malformed test claim for {member.capsule}")
        names = gate.get("traits") or []
        host_basis = _host_basis(descriptor, member.source_dir, gate)
        has_host_basis |= host_basis is not None
        if (
            (not names and host_basis is None)
            or not isinstance(names, list)
            or any(name not in (facts.get("traits") or {}) for name in names)
        ):
            raise StageGateError(f"test claim has no frozen RTL trait basis: {member.capsule}")
        selected_traits = {name: facts["traits"][name] for name in names}
        if host_basis is None and any(row.get("satisfied") is not True for row in selected_traits.values()):
            raise StageGateError(f"generated performance member has an unsatisfied RTL gate: {member.capsule}")
        capabilities = gate.get("execution_capabilities") or []
        selected_execution = {name: (facts.get("execution_capabilities") or {}).get(name) for name in capabilities}
        if host_basis is None and any(
            not isinstance(row, Mapping) or row.get("satisfied") is not True for row in selected_execution.values()
        ):
            raise StageGateError(f"generated performance member has an unsatisfied execution gate: {member.capsule}")
        if (
            not all(comparand.get(key) for key in ("kind", "against", "demand_equal"))
            or not all(falsifier.get(key) for key in ("observation", "fires_when", "negative_control"))
            or not gate.get("instrument")
        ):
            raise StageGateError(f"test claim lacks a comparator, falsifier or instrument: {member.capsule}")
        group = descriptor.get("comparison_group") or {}
        siblings = (
            sorted(
                ({"role": role, "capsule": name} for role, name in roles_by_group[(member.family, str(group["name"]))]),
                key=lambda item: (item["role"], item["capsule"]),
            )
            if isinstance(group, Mapping) and group.get("name")
            else []
        )
        expected_ops = ((performance.get("emitter") or {}).get("knobs") or {}).get("members") or []
        sibling_ops = (
            {
                ((other.descriptor.get("operation") or {}).get("op"))
                for other in corpus.capsules
                if other.family == member.family
                and isinstance(group, Mapping)
                and (other.descriptor.get("comparison_group") or {}).get("name") == group.get("name")
            }
            if siblings
            else set()
        )
        matching_status = (
            "incomplete_group"
            if (siblings and (len(siblings) < 2 or set(expected_ops) - sibling_ops))
            else "planned_unverified"
        )
        # A prose counterfactual is a *declared* control until an analyzer verifies
        # equal demand and the paired programs/results. Never label it established.
        acceptance = performance.get("acceptance") or {}
        measure = acceptance.get("evidence") or {} if isinstance(acceptance, Mapping) else {}
        repeats = acceptance.get("replicates") or {} if isinstance(acceptance, Mapping) else {}
        geometry = performance.get("shape_geometry") or {}
        relevant = []
        for application, index, demand in basis_rows:
            form = match_operation_form(demand, descriptor)
            if form["status"] not in {
                "exact_arithmetic_form",
                "structural_arithmetic_candidate",
                "shape_dtype_candidate",
            }:
                continue
            key = (application, index)
            if form["status"] == "exact_arithmetic_form":
                represented_rows.add(key)
            else:
                candidate_rows.add(key)
                candidates_by_row[key][member.capsule] = form["status"]
            relevant.append(
                {
                    "application": application,
                    "signature_index": index,
                    "operation": demand.get("operation"),
                    "form_match": form,
                    "count": demand["count"],
                    "known_macs": demand["macs"]["total"],
                }
            )
        exact_macs = sum(
            row["known_macs"] or 0 for row in relevant if row["form_match"]["status"] == "exact_arithmetic_form"
        )
        if any(row["form_match"]["status"] == "exact_arithmetic_form" for row in relevant):
            relevance_status = "exact_arithmetic_form"
        elif any(row["form_match"]["status"] == "structural_arithmetic_candidate" for row in relevant):
            relevance_status = "structural_arithmetic_candidate"
        elif relevant:
            relevance_status = "shape_dtype_candidate"
        elif basis.get("status") == "not_available":
            relevance_status = "basis_unavailable"
        elif basis.get("status") == "not_declared":
            relevance_status = "no_declared_workload"
        elif geometry.get("in_census") is False:
            relevance_status = "off_census"
        else:
            relevance_status = "no_exact_form_match"
        workload = {
            "status": relevance_status,
            "capsule_form": {
                "operation": (descriptor.get("operation") or {}).get("op"),
                "inputs": [
                    {key: tensor.get(key) for key in ("role", "shape", "dtype")}
                    for tensor in descriptor.get("inputs") or []
                    if isinstance(tensor, Mapping)
                ],
                "attributes": {
                    key: ((descriptor.get("operation") or {}).get("attributes") or {}).get(key)
                    for key in ("epilogue", "output_dtype")
                },
            },
            "matched_demands": relevant,
            "exact_known_macs": exact_macs,
            "known_macs_denominator": known_macs,
            "unknown_macs_demands": unknown_macs_demands,
            "geometry_class": geometry.get("geometry_class"),
            "class_in_census": geometry.get("in_census"),
            "census_mac_fraction": geometry.get("census_mac_fraction"),
            "class_fraction_scope": "shape class only; not this capsule's covered workload fraction",
            "selected_inventory_sha256": basis["sources"].get("selected_inventory_sha256"),
            "scope": accounting.get("scope"),
            "qualification": "exact arithmetic form is a diagnostic match, not complete operation or model coverage",
        }
        screen = descriptor.get("software_screen") or {}
        rows.append(
            {
                "family": member.family,
                "capsule": member.capsule,
                "relative_path": member.source_relative_path,
                "capsule_tree_sha256": member.source_sha256,
                "performance_sha256": C.document_sha256(performance),
                "purpose": {
                    "level": performance.get("level"),
                    "lever": performance.get("lever"),
                    "member_class": performance.get("member_class"),
                    "claim": performance.get("claim"),
                },
                "hardware_basis": {"traits": selected_traits, "execution_capabilities": selected_execution},
                **(
                    {
                        "source_execution_basis": host_basis,
                        "curriculum": performance.get("curriculum"),
                        "evaluation_tiers": {
                            "required_oracle_tiers": descriptor.get("required_oracle_tiers"),
                            "execution": "UNKNOWN",
                            "timing": "UNKNOWN",
                        },
                    }
                    if host_basis is not None
                    else {}
                ),
                "workload_need": workload,
                "hypothesis": {"observation": falsifier["observation"], "falsified_when": falsifier["fires_when"]},
                "matched_comparator": {
                    "declaration": comparand,
                    "group": group or None,
                    "candidate_members": siblings,
                    "matching_status": matching_status,
                },
                "negative_control": {"declaration": falsifier["negative_control"], "status": "planned_unverified"},
                "measurement": {
                    "instrument": gate["instrument"],
                    "timing_simulator": measure.get("timing_simulator"),
                    "timing_tier": measure.get("timing_tier"),
                    "status": "unmeasured",
                },
                "correctness": {
                    "required_oracle_tiers": descriptor.get("required_oracle_tiers") or [],
                    "simulator": measure.get("correctness_simulator"),
                    "tier": measure.get("correctness_tier"),
                    "status": "unverified",
                },
                "repeat_dispersion": {
                    "replicates": repeats,
                    "noise_band": performance.get("noise_band") or acceptance.get("band") or {},
                    "status": "planned_unmeasured",
                },
                "support": {
                    "software_screen": screen.get("status", "not_established"),
                    "status": "unsupported" if screen.get("status") == "unsupported" else "unverified",
                },
            }
        )
    represented_known_macs = sum(
        demand["macs"]["total"] or 0
        for application, index, demand in basis_rows
        if (application, index) in represented_rows
    )
    candidate_only_known_macs = sum(
        demand["macs"]["total"] or 0
        for application, index, demand in basis_rows
        if (application, index) in candidate_rows - represented_rows
    )
    uncovered_demands = [
        {
            "application": application,
            "capture_sha256": demand.get("capture_sha256"),
            "signature_index": index,
            "operation": demand.get("operation"),
            "mlir_operation": demand.get("mlir_operation"),
            "contraction_shape": demand.get("contraction_shape"),
            "shape_confidence": demand.get("shape_confidence"),
            "operand_format": demand.get("operand_format"),
            "ordered_operand_dtypes": [
                tensor.get("dtype")
                for tensor in demand.get("ordered_operand_types") or []
                if isinstance(tensor, Mapping)
            ],
            "accumulator_dtypes": demand.get("accumulator_dtypes"),
            "placement": demand.get("placement"),
            "count": demand["count"],
            "known_macs": demand["macs"]["total"],
            "unknown_macs_reason": demand["macs"].get("reason"),
            "candidate_capsules": [
                {"capsule": capsule, "status": status}
                for capsule, status in sorted(candidates_by_row.get((application, index), {}).items())
            ],
        }
        for application, index, demand in basis_rows
        if (application, index) not in represented_rows
        and demand.get("independent_compute_demand") is True
        and (demand.get("semantic_family") == "contraction" or demand["macs"]["total"] is not None)
    ]
    return {
        "schema": "merlin.phase2.test_justification.v2" if has_host_basis else "merlin.phase2.test_justification.v1",
        "target": corpus.target,
        "status": "diagnostic_unmeasured",
        "qualification": "test design and input identity only; no compiler correctness or measured speedup",
        "source": {
            **source,
            "provenance_sha256": corpus.provenance_sha256,
            "performance_generation_sha256": C.document_sha256(corpus.performance_generation),
            "evidence_status": evidence.get("status"),
        },
        "phase_roles": (provenance.get("phase_corpora") or {}).get(corpus.target),
        "families_not_generated": {
            "skipped_inapplicable": corpus.performance_generation.get("skipped_inapplicable") or [],
            "blocked_unimplemented": corpus.performance_generation.get("blocked_unimplemented") or [],
            "errors": corpus.performance_generation.get("errors") or [],
            **(
                {
                    "mandatory_unknowns": (
                        (corpus.performance_generation.get("component_generation") or {}).get("source_curriculum") or {}
                    ).get("required_unknowns", [])
                }
                if has_host_basis
                else {}
            ),
        },
        "workload_accounting": {
            "basis_status": basis.get("status"),
            "known_macs": known_macs,
            "exact_arithmetic_form_macs": represented_known_macs if known_macs is not None else None,
            "candidate_only_known_macs": candidate_only_known_macs if known_macs is not None else None,
            "uncovered_known_macs": known_macs - represented_known_macs if known_macs is not None else None,
            "unknown_macs_demands": unknown_macs_demands,
            "known_static_tensor_bytes": basis["overall"].get("known_static_tensor_bytes"),
            "unknown_static_tensor_bytes_demands": basis["overall"].get("unknown_static_tensor_bytes_demands"),
            "uncovered_demands": uncovered_demands,
            "qualification": "captured static MAC work; no execution frequency, cycle or speedup inference",
        },
        "members": rows,
    }
