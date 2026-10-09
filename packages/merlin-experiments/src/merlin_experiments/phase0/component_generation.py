"""Admission of independent component sweeps through the normal generator.

This declaration path does not certify hardware, timings or whole-model coverage.
The host reviews the selected SW declaration against byte-bound HW inputs; the
existing writer, numerical engines and written-program screen retain ownership.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path

import yaml

from merlin.targetgen import corpus_spec, phase_policy, software_spec
from merlin.targetgen.semantic_families import from_op

SCHEMA = "merlin.phase0.component_generation.v1"
DECLARATION = "merlin.component_performance.v1"
_OBJECTIVE_FIELDS = {"metric", "unit", "direction", "basis"}
_CAPTURE_FIELDS = {"micro_model", "materialized_capture", "model", "capture", "capture_dir", "quant_recipe"}


def digest(document):
    return hashlib.sha256(
        json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _source(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def require_inputs(*, descriptor, recipe, performance_template, profiles_root, evidence_mode, **sidecars):
    """Refuse model/hidden selectors before the ordinary loader can use them."""
    if descriptor is None or recipe is None or performance_template is None or profiles_root is not None:
        raise ValueError("component generation requires explicit descriptor, recipe and shared performance_template")
    if evidence_mode != "diagnostic":
        raise ValueError(
            "component generation requires explicit diagnostic evidence mode; whole coverage is not verified"
        )
    supplied = {name: value for name, value in sidecars.items() if value is not None}
    if "evidence_input" in supplied:
        from merlin_experiments.frozen_python import active_source_identity

        if active_source_identity() is not None:
            # The normal orchestrator selected and source-froze these bytes.
            # Ordinary direct calls still refuse historical evidence inputs.
            supplied.pop("evidence_input")
    if supplied:
        raise ValueError(
            "component generation refuses capture, synthesis, hidden and previously exported evidence inputs"
        )
    document = yaml.safe_load(Path(descriptor).read_bytes())
    if not isinstance(document, dict) or any(
        document.get(key) for key in ("workload_spec", "claim_boundary", "grading")
    ):
        raise ValueError(
            "component generation requires an independent descriptor without model/holdout grading selectors"
        )
    public = yaml.safe_load(Path(recipe).read_bytes()) or {}
    if not isinstance(public, dict) or public.get("capsules") or public.get("sweeps"):
        raise ValueError("component recipe may not author capsules or sweeps; use the shared independent template")
    template = yaml.safe_load(Path(performance_template).read_bytes()) or {}
    if not isinstance(template, dict) or any(row.get("requires_form_scope") for row in template.get("sweeps") or []):
        raise ValueError("component generation refuses capture-derived form-scope sweeps")


def bind_entries(
    entries,
    *,
    evidence,
    profile,
    recipe,
    performance_template,
    coverage_plan_sha256=None,
    execution_budget=None,
    execution_admission=None,
    semantic_basis=None,
    hardware_intake=None,
    software_intake=None,
    automatic_derivation=None,
    source_components=False,
):
    """Bind reviewed objectives and the actual selected generator/source identity."""
    if type(source_components) is not bool or (source_components and (evidence is None or evidence.contract)):
        raise ValueError("source component identity requires an explicit source-only mode without a backend contract")
    if evidence is None or evidence.software_spec.get("status") != "reviewed":
        raise ValueError("component generation requires selected reviewed software and hardware evidence")
    spec = evidence.software_spec
    recipe_document = yaml.safe_load(Path(recipe).read_bytes())
    if not isinstance(recipe_document, dict):
        raise ValueError("component objective recipe must be an explicit mapping")
    recipe_declaration = recipe_document.get("component_performance")
    legacy_declaration = spec.get("component_performance")
    if recipe_declaration is not None and legacy_declaration is not None:
        raise ValueError("component objective must have one explicit owner; recipe and software spec both declare it")
    declaration = recipe_declaration if recipe_declaration is not None else legacy_declaration
    if (
        not isinstance(declaration, dict)
        or set(declaration) != {"schema", "status", "hardware", "objectives"}
        or declaration.get("schema") != DECLARATION
        or declaration.get("status") != "reviewed"
    ):
        raise ValueError("component generation requires an explicit reviewed component_performance declaration")
    hardware = {key: evidence.derivation_identity[key] for key in ("contract_sha256", "raw_facts_sha256")}
    if any(value is None for value in hardware.values()) or declaration["hardware"] != hardware:
        raise ValueError("component objective declaration differs from selected hardware contract/facts")
    snapshots = [row for row in evidence.source_snapshots if row.role == "software-spec"]
    if len(snapshots) != 1 or snapshots[0].sha256 != (profile.get("_software_spec_identity") or {}).get("sha256"):
        raise ValueError("component declaration is not bound to the selected software source bytes")
    declaration_source = {
        "kind": "recipe" if recipe_declaration is not None else "legacy_software_spec",
        "sha256": _source(recipe)["sha256"] if recipe_declaration is not None else snapshots[0].sha256,
    }
    if evidence.application_inventory or evidence.whole_program_admission:
        raise ValueError("component generation refuses application/capture evidence")
    owners = {row["id"]: row for row in spec["operations"] if row.get("status", "reviewed") == "reviewed"}
    rows = declaration["objectives"]
    if not isinstance(rows, list):
        raise ValueError("component objectives must be an explicit list")
    families = {(entry.get("performance") or {}).get("family") for entry in entries}
    selected = {}
    provenance = (
        f"software-spec:{snapshots[0].sha256}",
        f"component-declaration:{digest(declaration)}",
        f"component-declaration-source:{declaration_source['kind']}:{declaration_source['sha256']}",
        *(f"{key}:{value}" for key, value in hardware.items()),
    )
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"family", "operations", "objective"}:
            raise ValueError("component objective requires exactly family, operations and objective")
        family, operations, objective = row["family"], row["operations"], row["objective"]
        if not isinstance(family, str) or family not in families or family in selected:
            raise ValueError("component objective names an unknown or duplicate generated family")
        if (
            not isinstance(operations, list)
            or not operations
            or any(not isinstance(name, str) or name not in owners for name in operations)
            or len(set(operations)) != len(operations)
        ):
            raise ValueError("component objective must name unique reviewed software operation declarations")
        if not isinstance(objective, dict) or set(objective) != _OBJECTIVE_FIELDS:
            raise ValueError("component objective needs only metric, unit, direction and basis; provenance is derived")
        typed = phase_policy.PerformanceObjective(
            **objective,
            provenance=(*provenance, *(f"operation:{name}:{digest(owners[name])}" for name in operations)),
        )
        selected[family] = {"operations": list(operations), "objective": typed.to_dict()}
    source_paths = [*Path(__file__).parent.glob("*.py")]
    source_paths += [Path(owner.__file__) for owner in (corpus_spec, phase_policy, software_spec)]
    if coverage_plan_sha256 is not None:
        from merlin.common.paths import module_source_path

        source_paths += [
            module_source_path(name)
            for name in (
                "merlin.targetgen.component_program",
                "merlin.targetgen.component_sources",
                "merlin.targetgen.capsule_results",
                "merlin.targetgen.capsule_golden",
                "merlin.targetgen.capsule_inputs",
                "merlin.targetgen.capsule_source",
                "merlin.targetgen._m2m_capture_worker",
                "merlin.runtime.tensor",
                "merlin.runtime.commandbuffer",
                "merlin.targetgen.operation_accounting",
                "merlin.targetgen.application_inventory",
                "merlin.targetgen.address_space",
                "merlin.targetgen.golden_store",
                "merlin.targetgen.corpus_operands",
                "merlin.targetgen.model_coverage",
                "merlin.targetgen.input_palette",
                "merlin.runtime.fp8_formats",
                "merlin.common.quant_formats",
            )
        ]
    identity = {
        "schema": SCHEMA,
        "status": "generated_from_reviewed_component_declarations",
        "scope": "independent_components_no_application_inputs",
        "evidence_status": evidence.status,
        "whole_coverage_verified": False,
        "timing_verified": False,
        "hardware": hardware,
        "software_spec_sha256": snapshots[0].sha256,
        "declaration_sha256": digest(declaration),
        "declaration_source": declaration_source,
        "recipe": _source(recipe),
        "shared_template": _source(performance_template),
        "generator_sources": [_source(path) for path in sorted(source_paths)],
        "families": selected,
    }
    if source_components:
        from .component_source_binding import identity as source_identity

        identity["source_semantics_admission"] = source_identity(software_intake, hardware=hardware_intake)
    if execution_budget is not None:
        from .component_execution_budget import validate

        identity["execution_budget_sha256"] = digest(validate(execution_budget))
        if execution_admission is None or execution_admission.get("policy") != execution_budget:
            raise ValueError("budgeted component identity requires its actual complete admission roster")
        identity["execution_admission_sha256"] = digest(execution_admission)
    elif execution_admission is not None:
        raise ValueError("component execution admission requires an explicit frozen policy")
    if hardware_intake is not None:
        from .rtl_intake import bind_component_hardware

        identity.update(bind_component_hardware(evidence, hardware_intake))
    if software_intake is not None:
        from .software_intake import bind_component_software

        identity.update(bind_component_software(evidence, software_intake))
    if coverage_plan_sha256 is not None:
        identity["component_coverage_plan_sha256"] = coverage_plan_sha256
        from merlin_experiments.frozen_python import active_source_identity

        routing = (
            json.loads(os.environ.get("MERLIN_PHASE0_FROZEN_SOURCE_MAP", "{}")) if active_source_identity() else {}
        )
        selected_sources = []
        for source in evidence.source_snapshots:
            path = routing.get(str(source.path), str(source.path))
            selected_source = _source(path)
            if selected_source["sha256"] != source.sha256:
                raise ValueError("selected component semantic/hardware/numerical source changed")
            selected_sources.append({**selected_source, "role": source.role})
        identity["selected_sources"] = selected_sources
    if semantic_basis is not None:
        identity["semantic_basis"] = semantic_basis.reviewed_semantics()
    if automatic_derivation is not None:
        identity["automatic_derivation_sha256"] = digest(automatic_derivation)
    bound = []
    for entry in entries:
        coverage = entry.get("component_coverage")
        covered = isinstance(coverage, dict) and coverage_plan_sha256 is not None
        if coverage is not None and (not covered or coverage.get("plan_sha256") != coverage_plan_sha256):
            raise ValueError("component coverage membership must come from the selected independent plan")
        functional = covered and coverage.get("cohort") in {"functional_guard", "withheld_transfer"}
        if (
            (not functional and (entry.get("cat") != "_perf" or entry.get("label") != "dev"))
            or entry.get("source_role") != "derived_sweep"
            or entry.get("kind") == "model"
            or any(entry.get(key) for key in _CAPTURE_FIELDS)
            or (entry.get("performance") or {}).get("global_objective")
        ):
            raise ValueError(
                "component generation admits only independent derived sweeps and selected coverage obligations"
            )
        value = copy.deepcopy(entry)
        if covered:
            value["component_coverage"]["generation_sha256"] = digest(identity)
        if functional:
            bound.append(value)
            continue
        performance = value["performance"]
        if "component_generation_sha256" in performance or "objective" in performance:
            raise ValueError("component objective identity is generated, never authored by the sweep")
        performance["component_generation_sha256"] = digest(identity)
        family = performance["family"]
        if family in selected:
            op = entry.get("op")
            semantic_family = from_op(op or "")
            declarations = [owners[name] for name in selected[family]["operations"]]
            if not any(
                op in row.get("ops", []) or (semantic_family is not None and semantic_family in row.get("families", []))
                for row in declarations
            ):
                raise ValueError("component objective owner does not declare the generated operation/family")
            performance["objective"] = copy.deepcopy(selected[family]["objective"])
        bound.append(value)
    return bound, identity


def require_written(capsule, *, source_software=None, source_hardware=None, directory=None):
    """Require the selected original-source or concrete target admission scope."""
    screen = capsule.get("software_screen") or {}
    coverage = capsule.get("component_coverage") or {}
    if coverage.get("expectation") == "unsupported_program":
        if screen.get("status") != "unsupported":
            raise ValueError(
                "declared refusal obligation was not independently refused by the concrete program screen: "
                + str(screen.get("status"))
                + ": "
                + str(screen.get("reason"))
            )
        return
    if source_software is not None or source_hardware is not None:
        from .component_source_binding import screen_written

        if directory is None or coverage.get("cohort") not in {"functional_guard", "withheld_transfer"}:
            raise ValueError(
                "source-only admission requires original bounded coverage members, not performance support"
            )
        capsule["source_semantics_screen"] = screen_written(
            capsule, directory, software=source_software, hardware=source_hardware
        )
        return
    if screen.get("status") != "admitted":
        raise ValueError("component generated program lacks reviewed concrete software/hardware admission")
    if coverage.get("cohort") in {"functional_guard", "withheld_transfer"}:
        return
    objective = (capsule.get("performance") or {}).get("objective")
    typed = None
    if objective is not None:
        value = dict(objective)
        value["provenance"] = tuple(value["provenance"])
        typed = phase_policy.PerformanceObjective(**value)
    verdict = phase_policy.priceable(capsule, performance_objective=typed)
    if verdict.value != phase_policy.YES:
        raise ValueError("component work admission refused: " + verdict.reason)
