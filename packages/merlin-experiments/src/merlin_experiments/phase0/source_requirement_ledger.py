"""Separate checked source test inputs from unevaluated compiler verdicts.

This diagnostic ledger preserves the original required denominator. It does not
repair missing RTL premises or issue a source release, compiler qualification,
runtime role or performance authority. Historical coverage meanings are intact.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import yaml

from .component_generation import digest

SCHEMA = "merlin.phase0.source_requirement_ledger.v1"
PREREQUISITE_SCHEMA = "merlin.phase0.source_requirement_ledger.v3"
PURPOSES = ("source_diagnostic", "source_preparation", "performance_campaign")

# These are compiler verdict owners, not evidence that a source case exists.
_CANDIDATE_EFFECTS = {"physical_interaction", "physical_effect", "effect_domain"}
_PREMISE_PRODUCERS = {
    "operation": "protected original call-to-semantic-owner correspondence",
    "operator_effect": "original public native schema/effect observation",
    "original_call_binding": "original call/default/bridge metadata observation",
    "original_operator_factory": "original typed operator source factory",
    "original_operator_admission": "original full source/reference numerical comparison",
    "source_operator_form": "original typed operator source factory",
    "numeric_datapath": "independent RTL arithmetic-to-source numerical premise",
    "numeric_domain": "original complete numerical-domain source/reference inputs",
    "packing_mapping": "independent command/instance/axis/address/capacity mapping",
    "packing_domain": "complete selected packing/resource source premise",
    "resource_role": "independent command/instance/axis/address/capacity mapping",
    "effect": "original operator effect semantics and checked source cases",
    "interaction": "original interaction source factory and complete reference",
    "physical_interaction": "original physical buffer/guard/order test contract and RTL effect premise",
    "physical_effect": "original physical buffer/guard/order test contract and RTL effect premise",
    "effect_domain": "complete original source effect contract and independent RTL primitive premise",
}


def _file(path):
    path = Path(path)
    if (
        not path.is_absolute()
        or ".." in path.parts
        or any(part.is_symlink() for part in (path, *path.parents))
        or not path.is_file()
    ):
        raise ValueError("requirement evidence needs canonical ordinary source/product files")
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _selected(pin):
    if not isinstance(pin, dict) or _file(pin["path"]) != {key: pin[key] for key in ("path", "sha256")}:
        raise ValueError("requirement generation source bytes changed")
    return Path(pin["path"])


def _relations(program):
    """Observe actual logical use-def predicates, never physical reuse."""
    produced = {node["name"] for node in program["nodes"]}
    consumers = {name: set() for name in produced}
    for node in program["nodes"]:
        for value in node["actual_inputs"]:
            if value in consumers:
                consumers[value].add(node["name"])
    published = {row["actual_value"] for row in program["outputs"]}
    return {
        "shared_producer_multiple_consumers": sorted(name for name, uses in consumers.items() if len(uses) > 1),
        "publication_and_further_use": sorted(name for name, uses in consumers.items() if uses and name in published),
    }


def _member(root, member, *, software, hardware):
    """Recompute all original values after the enclosing budget was verified."""
    from merlin.targetgen import golden_store
    from merlin_experiments.phase1.source_inputs import fingerprint

    from .component_numerics import evaluate
    from .component_source_binding import screen_written

    relative = Path(member["member"])
    if relative.is_absolute() or len(relative.parts) != 2 or any(part in {".", ".."} for part in relative.parts):
        raise ValueError("requirement source member escaped its generation owner")
    directory = root / relative
    capsule_path = directory / "capsule.yaml"
    _file(capsule_path)
    entries = sorted(directory.rglob("*"))
    if any(path.is_symlink() for path in entries):
        raise ValueError("requirement source/reference member contains an indirect product")
    files = [_file(path) for path in entries if path.is_file()]
    if fingerprint(directory) != member["sha256"]:
        raise ValueError("requirement original member bytes changed")
    capsule = yaml.safe_load(capsule_path.read_bytes())
    screen = screen_written(capsule, directory, software=software, hardware=hardware)
    if screen != capsule.get("source_semantics_screen"):
        raise ValueError("requirement source semantics changed")
    source = _file(directory / capsule["linalg_mlir"])
    stored = golden_store.load_golden(directory)
    original = evaluate(capsule)
    if (
        not original
        or digest(original) != digest(stored.get("outputs"))
        or sorted(original) != member["output_roster"]
        or screen["program_sha256"] != source["sha256"]
    ):
        raise ValueError("requirement complete original independent reference differs from written outputs")
    program = capsule["component_program"]
    return {
        "member": member["member"],
        "member_sha256": member["sha256"],
        "source": source,
        "original_input_contract_sha256": digest(
            {key: capsule.get(key) for key in ("inputs", "input_palette", "stimulus_range", "external_inputs")}
        ),
        "original_typed_program_sha256": digest(program),
        "complete_output_roster": sorted(original),
        "complete_reference_outputs_sha256": digest(original),
        "source_semantics_sha256": digest(screen),
        "logical_relations": _relations(program),
        "source_product_pins": files,
        "producer": "ordinary component writer/source screen and independent component_numerics.evaluate",
        "producer_phase": 0,
        "scope": (
            "complete original tensor source/reference; physical buffers/guards/order/device effects unestablished"
        ),
    }


def _performance(root, coverage, *, purpose):
    identity = coverage["generation_identity"]
    recipe_path = _selected(identity["recipe"])
    template_path = _selected(identity["shared_template"])
    recipe = yaml.safe_load(recipe_path.read_bytes())
    template = yaml.safe_load(template_path.read_bytes())
    declaration = recipe.get("component_performance")
    objectives = declaration.get("objectives") if isinstance(declaration, dict) else None
    rows = coverage["obligations"]
    counts = {
        cohort: sum(
            len([member for member in row["members"] if member["state"] == "source_generated"])
            for row in rows
            if row["cohort"] == cohort
        )
        for cohort in ("development", "functional_guard", "withheld_transfer")
    }
    missing = []
    if not isinstance(objectives, list) or not objectives:
        missing.append("explicit nonempty independent performance objectives")
    if not template.get("families") or not template.get("sweeps"):
        missing.append("independent source performance families and sweeps")
    for cohort, count in counts.items():
        if not count:
            missing.append("actual complete source/reference membership for " + cohort)
    # No existing producer closes these contracts merely by writing objectives.
    missing += [
        "independently checked resource-derived multi-size/tail/streaming/reuse corpus contract",
        "source-bound measurement/cold-warm/timer/domain and calibration/held contract",
        "independently qualified Phase2 guard producer (source-only guards remain unqualified)",
    ]
    return {
        "required_for_purpose": purpose == "performance_campaign",
        "recipe": _file(recipe_path),
        "shared_template": _file(template_path),
        "objectives_sha256": digest(objectives),
        "objective_count": len(objectives) if isinstance(objectives, list) else 0,
        "checked_source_members_by_cohort": counts,
        "missing_producers": missing,
        "producer_phase": 0,
        "verdict_phase": 2,
        "candidate_verdict": "not_evaluated",
        "scope": "source corpus/measurement design requirements; not measured optimization or timing qualification",
    }


@dataclass(frozen=True)
class SourceRequirementLedger:
    """Immutable diagnostic data; no issuer or conversion to release authority."""

    document_json: str

    def record(self):
        return json.loads(self.document_json)


def prepare_requirement_ledger(*, root, coverage, hardware, software, purpose, standard_ir=None):
    """Reopen actual sources, budgets and complete references before attribution."""
    from .component_execution_budget import verify as verify_budget
    from .component_source_binding import verify_prepared_sources

    if purpose not in PURPOSES:
        raise ValueError("source requirement ledger needs an explicit supported preparation purpose")
    root = Path(root).absolute()
    if any(path.is_symlink() for path in (root, *root.parents)):
        raise ValueError("source requirement generation owner is indirect")
    coverage = copy.deepcopy(coverage)
    # Replays exact required originals and the live HW/SW selection, including
    # unavailable rows. A re-signed missing-row list is not a new denominator.
    verify_prepared_sources(root, coverage, software=software, hardware=hardware)
    unknowns = {row["id"]: row for row in coverage["automatic_derivation"]["required_unknowns"]}
    if len(unknowns) != len(coverage["automatic_derivation"]["required_unknowns"]):
        raise ValueError("requirement original unknown IDs are duplicated")
    budget_error = None
    try:
        verify_budget(root, coverage)
    except ValueError as error:
        # A missing or denied requested case cannot trigger tensor/reference
        # allocation. Preserve its source blocker instead of dropping it.
        budget_error = str(error)
    witnesses, rows, seen = {}, [], set()
    for original in coverage["obligations"]:
        identity = original["id"]
        if identity in seen:
            raise ValueError("requirement original IDs are duplicated")
        seen.add(identity)
        selected = unknowns.get(identity)
        missing = []
        members = []
        for member in original["members"]:
            if member["state"] != "source_generated":
                missing.append("actual required source/reference member: " + member["name"])
            elif budget_error is not None:
                missing.append("complete source/reference budget replay: " + budget_error)
            else:
                witness = _member(root, member, software=software, hardware=hardware)
                witnesses[member["member"]] = witness
                members.append(member["member"])
        kind = selected["kind"] if selected else "generated_source_case"
        if selected:
            missing.append(_PREMISE_PRODUCERS.get(kind, "independent original source prerequisite producer: " + kind))
        elif not members:
            missing.append("actual complete original source testcase and independent reference")
        if original["errors"]:
            missing.extend("original source-generation refusal: " + error for error in original["errors"])
        rows.append(
            {
                "original_id": identity,
                "original_requirement_sha256": digest(original),
                "original_declaration_sha256": original["declaration_sha256"],
                "mandatory": original["mandatory"],
                "cohort": original["cohort"],
                "kind": kind,
                "original_selector": copy.deepcopy(selected["selector"]) if selected else None,
                "source_producer": _PREMISE_PRODUCERS.get(kind, "ordinary component source/reference writer"),
                "source_producer_phase": 0,
                "checked_source_members": members,
                "missing_source_producers": sorted(set(missing)),
                "source_input_state": "checked" if not missing else "unavailable",
                "candidate_producer": "ordinary Phase1 source/emission/ELF/full-output/effect qualification",
                "candidate_verdict_phase": 1,
                "candidate_verdict": "not_evaluated",
                "candidate_predicate": "physical_execution_effect"
                if kind in _CANDIDATE_EFFECTS
                else "original_compiler_requirement",
            }
        )
    if not set(unknowns) <= seen:
        raise ValueError("requirement ledger lost an original missing obligation")
    # Match logical source cases by an actual checked predicate. This only
    # establishes a testcase relation; physical ABI/guard/order premises above
    # remain missing until an independently sourced producer provides them.
    for row in rows:
        selector = row["original_selector"]
        row["logical_testcase_members"] = (
            sorted(member for member, witness in witnesses.items() if witness["logical_relations"].get(selector))
            if row["kind"] == "physical_interaction" and isinstance(selector, str)
            else []
        )
    performance = _performance(root, coverage, purpose=purpose)
    mandatory_missing = [row["original_id"] for row in rows if row["mandatory"] and row["missing_source_producers"]]
    document = {
        "schema": SCHEMA,
        "purpose": purpose,
        "coverage_sha256": coverage["sha256"],
        "hardware_intake_sha256": hardware.sha256,
        "software_intake_sha256": software.sha256,
        "original_required_ids": [row["original_id"] for row in rows],
        "original_mandatory_ids": [row["original_id"] for row in rows if row["mandatory"]],
        "requirements": rows,
        "checked_source_witnesses": sorted(witnesses.values(), key=lambda row: row["member"]),
        "mandatory_source_blockers": mandatory_missing,
        "performance_preparation": performance,
        "candidate_verdicts": "not_evaluated",
        "release_authority": "not_issued",
        "status": "diagnostic_incomplete"
        if mandatory_missing or purpose == "performance_campaign"
        else "diagnostic_source_inputs_checked",
        "scope": "source/candidate ownership diagnostic; exposes dependency cycle without declaring it fixed",
    }
    if standard_ir is not None:
        from .original_reference_requirements import join

        document = join(document, coverage=coverage, hardware=hardware, software=software, standard_ir=standard_ir)
    document["sha256"] = digest({key: value for key, value in document.items() if key != "sha256"})
    return SourceRequirementLedger(json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False))


def verify_requirement_ledger(ledger, *, root, coverage, hardware, software, purpose, standard_ir=None):
    """Recompute actual evidence; frozen data cannot grant or defer acceptance."""
    if type(ledger) is not SourceRequirementLedger:
        raise ValueError("requirement comparison needs the exact diagnostic data type")
    actual = prepare_requirement_ledger(
        root=root, coverage=coverage, hardware=hardware, software=software, purpose=purpose, standard_ir=standard_ir
    )
    if actual.record() != ledger.record():
        raise ValueError("requirement source/reference evidence or full denominator changed")
    return actual.record()


def prepare_prerequisite_ledger(
    *, root, coverage, hardware, software, purpose, schema_intake, semantic_basis, standard_ir=None
):
    """Retain fulfilled original factories beside the unchanged coverage projection.

    The explicit v3 domain includes every original call/cohort prerequisite.
    Its state describes source construction, never any admission or candidate
    verdict. Historical ledger APIs and their original required IDs are intact.
    """
    from merlin.common.jsonio import canonical_json

    from . import original_factory_prerequisites as F

    if (
        type(schema_intake) is not F.IndependentOperatorSchemaIntake
        or schema_intake.software is not software
        or software.hardware is not hardware
    ):
        raise ValueError("stable prerequisites need the identical live original schema/software/hardware owners")
    factory = F.prepare(
        schema_intake=schema_intake,
        basis=semantic_basis,
        source_record=coverage["automatic_derivation"]["original_call_sources"],
    )
    result = prepare_requirement_ledger(
        root=root, coverage=coverage, hardware=hardware, software=software, purpose=purpose, standard_ir=standard_ir
    ).record()
    original = result["original_required_ids"]
    coverage_ids = [row["id"] for row in coverage["obligations"]]
    if (
        original != coverage_ids
        or len(set(original)) != len(original)
        or [row["original_id"] for row in result["requirements"]] != original
    ):
        raise ValueError("stable prerequisites lost or duplicated an original coverage ID")
    observation = factory.record()
    factories = {row["id"]: row for row in observation["factory_prerequisites"]}
    for row in result["requirements"]:
        selected = factories.get(row["original_id"])
        if row["kind"] == "original_operator_factory" or selected is not None:
            if (
                selected is None
                or row["kind"] != selected["kind"]
                or canonical_json(row["original_selector"]) != canonical_json(selected["selector"])
                or type(row["mandatory"]) is not bool
                or row["mandatory"] is not selected["mandatory"]
            ):
                raise ValueError("stable prerequisite ID collides with an incompatible original coverage selector")
    union = sorted(set(original) | set(factories))
    if not set(original) <= set(union):
        raise ValueError("stable prerequisites omitted an original coverage ID")
    result.update(
        schema=PREREQUISITE_SCHEMA,
        coverage_projection_schema=result["schema"],
        original_prerequisite_ids=union,
        original_factory_prerequisites=observation,
        prerequisite_scope="complete original source factory prerequisites; coverage and candidate blockers unchanged",
    )
    result["sha256"] = digest({key: value for key, value in result.items() if key != "sha256"})
    return SourceRequirementLedger(json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False))


def verify_prerequisite_ledger(ledger, **inputs):
    """Recompute the fixed v3 roster; caller statuses cannot replace replay."""
    from merlin.common.jsonio import canonical_json

    if type(ledger) is not SourceRequirementLedger:
        raise ValueError("stable prerequisite comparison needs the exact diagnostic data type")
    actual = prepare_prerequisite_ledger(**inputs)
    if canonical_json(actual.record()) != canonical_json(ledger.record()):
        raise ValueError("stable original prerequisite roster, source bytes or unchanged coverage projection changed")
    return actual.record()
