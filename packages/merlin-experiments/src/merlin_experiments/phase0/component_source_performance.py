"""Prepare source performance cases without hardware or measured admission.

The explicit v1 selection uses only literal tensor DAG shapes. It replays the
ordinary sweep, aggregate budget, standard source and complete independent
reference owners. Data in this receipt cannot qualify a guard or a compiler.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import yaml

from merlin.targetgen import component_program, phase_policy
from merlin.targetgen.semantic_families import from_op

from .component_generation import digest

SCHEMA = "merlin.phase0.source_performance_preparation.v1"


def require_selection(selection, *, source_components):
    if selection is not None and (selection != SCHEMA or not source_components):
        raise ValueError("source performance preparation needs explicit v1 live backend-free inputs")


def sweep_refusal(sweep):
    """Unsupported requests remain named missing families; no device defaults."""
    allowed = {"id", "name", "axes", "fit_axes", "base", "source_reference"}
    base = sweep.get("base") or {}
    base_fields = {"kind", "cat", "label", "op", "program", "performance", "input_palette", "stimulus_range", "comment"}
    axes = sweep.get("axes")
    if (
        set(sweep) - allowed
        or not isinstance(base, dict)
        or set(base) - base_fields
        or base.get("op") != "component_program"
    ):
        return "source preparation supports only literal tensor DAG sweeps without capture or encoding selectors"
    if (
        not isinstance(axes, dict)
        or not axes
        or any(
            not isinstance(name, str)
            or not isinstance(values, list)
            or not values
            or any(type(value) is not int or value < 1 for value in values)
            for name, values in axes.items()
        )
    ):
        return "source preparation requires explicit positive literal axes; hardware/tile-derived extents are missing"
    return None


def materialize(entry, binding, *, point, gate):
    """Resolve axis objects only in original input shapes, then type the DAG."""
    value = copy.deepcopy(entry)
    program = value.get("program")
    if not isinstance(program, dict) or not isinstance(program.get("inputs"), list):
        raise ValueError("source performance preparation needs the complete original tensor DAG")
    used = set()
    for row in program["inputs"]:
        shape = row.get("shape")
        if not isinstance(shape, list):
            raise ValueError("source performance shapes must be explicit original input lists")
        for index, extent in enumerate(shape):
            if isinstance(extent, dict) and set(extent) == {"axis"} and extent["axis"] in point:
                used.add(extent["axis"])
                shape[index] = point[extent["axis"]]
            elif type(extent) is not int or extent < 1:
                raise ValueError("source performance shape has an unsupported axis or hardware-relative extent")
    if used != set(point):
        raise ValueError("source performance axes must each affect the actual typed tensor shapes")
    component_program.analyze(
        program,
        operand_dtype=binding.mlir_dtype(binding.operand_dtype),
        accumulator_dtype=binding.mlir_dtype(binding.accum_dtype),
    )
    value.pop("_derived_axes", None)
    value["source"] = "direct"
    value["operand_dtype"] = binding.operand_dtype
    value["performance"]["source_preparation"] = {
        "schema": SCHEMA,
        "hardware_gate": copy.deepcopy(gate),
        "hardware_admission": "not_established",
        "measurement_admission": "not_established",
    }
    value["performance"]["emitter"]["resolved"] = {
        "source": "direct",
        "operand_dtype": binding.operand_dtype,
        "accum_dtype": binding.accum_dtype,
        "datatype_basis": "independently selected source integer semantics",
        "builder": "merlin.targetgen.corpus_spec.build",
        "scope": "original source only",
    }
    return value


def node_owners(program, *, software):
    """Check every alias/update/copy/contraction and complete typed signature."""
    from .component_compile_plan import _owner

    owners = software.public_facts()["operations"]
    storage = program["selected_storage"]
    values = {row["name"]: row for row in program["inputs"] + program["nodes"]}
    rows = []
    for node in program["nodes"]:
        operation = {"alias": "copy", "update": "add"}.get(node["op"], node["op"])
        family = from_op(operation)
        selected = [
            owner
            for owner in owners
            if operation in owner.get("ops", []) or (not owner.get("ops") and family in owner.get("families", []))
        ]
        if len(selected) != 1:
            raise ValueError("source performance node has missing or ambiguous reviewed owners")
        if node["op"] == "alias" and "aliasing" in selected[0].get("signature", {}):
            raise ValueError("source performance preparation does not establish physical alias signatures")
        operands = [values[name]["dtype"] for name in node["actual_inputs"]]
        _owner(
            {"operation": operation, "operation_owner": selected[0]["id"]},
            software,
            storage,
            program,
            typed_inputs=operands,
            result_dtype=node["dtype"],
        )
        rows.append(
            {
                "node": node["name"],
                "source_operation": node["op"],
                "operation": operation,
                "family": family,
                "owner": selected[0]["id"],
                "ordered_operand_dtypes": operands,
                "ordered_result_dtypes": [node["dtype"]],
            }
        )
    if not rows:
        raise ValueError("source performance objective requires actual reviewed DAG operations")
    return rows


def _pin(path):
    from .source_requirement_ledger import _file

    return _file(Path(path))


def _binding(software, hardware):
    from .component_source_binding import derive

    numeric = software.public_facts()["numerical_semantics"]
    return derive(
        software,
        hardware=hardware,
        datapath={
            "operand_dtype": numeric["operand_dtype"],
            "accum_dtype": numeric["accumulator_dtype"],
            "subnormal_operand_flush": numeric["subnormal_operand_flush"],
            "numerical_semantics": numeric,
        },
    )


def _requests(coverage, *, software, hardware):
    """Replay actual selected shared sweeps and original coverage construction."""
    from .component_coverage_inputs import expand
    from .component_coverage_plan import ComponentCoveragePlan, ComponentObligation
    from .profiles import _merge_shared_perf, _select_performance_withdrawals
    from .sweeps import expand_sweeps

    identity = coverage["generation_identity"]
    mode = identity.get("source_performance_preparation")
    if not isinstance(mode, dict) or mode.get("schema") != SCHEMA:
        raise ValueError("source performance preparation selection is absent")
    paths = []
    for key in ("recipe", "shared_template"):
        pin = identity[key]
        if _pin(pin["path"]) != pin:
            raise ValueError("source performance generation inputs changed")
        paths.append(Path(pin["path"]))
    recipe = yaml.safe_load(paths[0].read_bytes())
    if not isinstance(recipe, dict) or recipe.get("capsules") or recipe.get("sweeps"):
        raise ValueError("source performance requests must come from the selected independent shared template")
    profile = copy.deepcopy(recipe)
    _select_performance_withdrawals(profile, source=paths[0])
    _merge_shared_perf(profile, source=paths[0], performance_template=paths[1])
    binding = _binding(software, hardware)
    blocked, errors, skipped = [], [], []
    development = expand_sweeps(
        profile,
        binding,
        trait_facts=mode["pending_gate_facts"],
        blocked_unimplemented=blocked,
        errors=errors,
        skipped=skipped,
        source_preparation=SCHEMA,
    )
    declaration = coverage["declaration"]
    obligations = tuple(
        ComponentObligation(row["id"], row["mandatory"], row["cohort"], json.dumps(row))
        for row in declaration["obligations"]
    )
    plan = ComponentCoveragePlan("", identity["component_coverage_plan_sha256"], json.dumps(declaration), obligations)
    functional, _ = expand(plan, binding=binding, evidence=None)
    return development + functional, development, binding, blocked + errors + skipped


def prepare_source_contracts(*, root, coverage, hardware, software):
    """Bind the full requested roster; recompute complete values after budgets.

    Missing sources and resource/measurement requirements remain explicit. This
    fixed replay is a source-data diagnostic, with no release/admission issuer.
    """
    from merlin_experiments.phase1.source_inputs import fingerprint

    from .component_execution_budget import admit, measure, source_for_capsule
    from .component_source_binding import verify_prepared_sources
    from .source_requirement_ledger import _member

    root = Path(root)
    verified = verify_prepared_sources(root, coverage, hardware=hardware, software=software)
    entries, development, binding, missing_families = _requests(coverage, software=software, hardware=hardware)
    admission = admit(entries, binding=binding, policy=coverage["declaration"]["execution_budget"])
    if admission != coverage["execution_admission"]:
        raise ValueError("source performance complete aggregate budget/request roster changed")
    identity = coverage["generation_identity"]
    declaration = yaml.safe_load(Path(identity["recipe"]["path"]).read_bytes())["component_performance"]
    if (
        not isinstance(declaration, dict)
        or set(declaration) != {"schema", "status", "hardware", "objectives"}
        or declaration["schema"] != "merlin.component_performance.v1"
        or declaration["status"] != "reviewed"
        or declaration["hardware"] != identity["hardware"]
    ):
        raise ValueError("source performance declaration differs from exact selected generation inputs")
    if not declaration["objectives"]:
        raise ValueError("source performance preparation requires nonempty original reviewed objectives")
    development_names = {entry["name"] for entry in development}
    expected_names = {row["requested_member"] for row in admission["decisions"]}
    actual_names = {
        path.parent.relative_to(root).as_posix()
        for path in root.glob("*/*/capsule.yaml")
        if path.parent.parent.name != "_evidence"
    }
    if actual_names - expected_names:
        raise ValueError("source performance member roster contains a foreign source")
    requested_by_name = {entry["name"]: entry for entry in entries}
    members, counts = [], dict.fromkeys(("development", "functional_guard", "withheld_transfer"), 0)
    for decision in admission["decisions"]:
        relative = decision["requested_member"]
        cohort = (
            "development"
            if decision["name"] in development_names
            else (requested_by_name[decision["name"]].get("component_coverage") or {}).get("cohort")
        )
        row = {
            "member": relative,
            "cohort": cohort,
            "budget": copy.deepcopy(decision),
            "candidate_verdict": "not_evaluated",
        }
        if decision["state"] != "admitted" or relative not in actual_names:
            row.update(state="unavailable", missing="original source/reference or aggregate budget admission")
            members.append(row)
            continue
        directory = root / relative
        capsule_pin = _pin(directory / "capsule.yaml")
        capsule = yaml.safe_load(Path(capsule_pin["path"]).read_bytes())
        if (
            measure(source_for_capsule(capsule)) != decision["cost"]
            or digest(source_for_capsule(capsule)) != decision["source_sha256"]
        ):
            raise ValueError("source performance written program differs from original budgeted source")
        stamp = (
            (capsule.get("performance") or {}).get("component_generation_sha256")
            if cohort == "development"
            else capsule["component_coverage"]["generation_sha256"]
        )
        if stamp != digest(identity):
            raise ValueError("source performance member is not bound to exact generation identity")
        owners = node_owners(capsule["component_program"], software=software)
        if cohort == "development":
            original_performance = requested_by_name[decision["name"]]["performance"]
            if (
                capsule["performance"].get("source_preparation") != original_performance["source_preparation"]
                or capsule["performance"]["emitter"] != original_performance["emitter"]
            ):
                raise ValueError("source performance hardware/measurement scope or original emitter changed")
            family = capsule["performance"]["family"]
            objective = identity["families"].get(family)
            selected = next((item for item in declaration["objectives"] if item["family"] == family), None)
            if (
                objective is None
                or selected is None
                or objective["operations"] != selected["operations"]
                or any(owner["owner"] not in selected["operations"] for owner in owners)
            ):
                raise ValueError("source performance objective differs from every actual reviewed node owner")
            typed = phase_policy.PerformanceObjective(
                **{**selected["objective"], "provenance": tuple(objective["objective"]["provenance"])}
            )
            if capsule["performance"]["objective"] != typed.to_dict() or objective["objective"] != typed.to_dict():
                raise ValueError("source performance original objective changed")
        member = {
            "member": relative,
            "sha256": fingerprint(directory),
            "output_roster": sorted(row["name"] for row in capsule["component_program"]["outputs"]),
        }
        source = _member(root, member, software=software, hardware=hardware)
        row.update(
            state="source_checked",
            original=source,
            typed_node_owners=owners,
            original_ordered_output_abi=copy.deepcopy(capsule["component_program"]["outputs"]),
        )
        members.append(row)
        counts[cohort] += 1
    missing = [
        "independent RTL command/axis/capacity/tail and streaming/reuse test premises",
        "complete original numerical-domain/effect/physical test premises",
        "source-bound timer/cold-warm/domain calibration and held measurement contract",
        "independently qualified hardware Phase2 guard and fresh measured baseline",
    ]
    missing += ["actual nonempty checked source membership: " + cohort for cohort, count in counts.items() if not count]
    data = {
        "schema": SCHEMA,
        "scope": "source and full reference preparation only",
        "status": "source_contracts_incomplete",
        "generation_identity_sha256": digest(identity),
        "coverage_sha256": coverage["sha256"],
        "execution_admission_sha256": digest(admission),
        "objective_count": len(declaration["objectives"]),
        "requested_members": members,
        "source_checked_counts": counts,
        "missing_families": missing_families,
        "original_required_ids": [row["id"] for row in coverage["obligations"]],
        "mandatory_missing_ids": verified["missing_mandatory_obligations"],
        "missing_producers": missing,
        "hardware_guard_link": "not_established",
        "candidate_verdict": "not_evaluated",
        "measured_baseline": "not_established",
        "release_authority": "not_issued",
    }
    data["sha256"] = digest(data)
    return data


def verify_source_contracts(record, **inputs):
    if record != prepare_source_contracts(**inputs):
        raise ValueError("source performance contracts changed from complete original source/reference replay")
    return copy.deepcopy(record)
