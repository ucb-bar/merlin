"""Prepare source performance cases without hardware or measured admission.

The explicit v1 selection uses literal tensor DAG shapes. The v2 selection
derives ordinary templates from live minimal software and RTL sources. Both
replay the ordinary sweep, aggregate budget, standard source and complete
independent reference owners. These data do not qualify a guard or a compiler.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import yaml

from merlin.targetgen import component_program, phase_policy
from merlin.targetgen.op_form_regimes import Bound, boundary_witnesses
from merlin.targetgen.semantic_families import from_op

from .component_generation import digest

SCHEMA = "merlin.phase0.source_performance_preparation.v1"
SCHEMA_V2 = "merlin.phase0.source_performance_preparation.v2"
CURRICULUM_SCHEMA = "merlin.phase0.source_curriculum.v1"


def validate_curriculum_policy(policy):
    """Closed construction budgets contain no application or query selector."""
    if (
        not isinstance(policy, dict)
        or set(policy) != {"schema", "probe_extents", "max_members", "max_extent_bits"}
        or policy["schema"] != CURRICULUM_SCHEMA
        or not isinstance(policy["probe_extents"], list)
        or len(policy["probe_extents"]) < 2
        or any(type(v) is not int or v < 1 for v in policy["probe_extents"])
        or len(set(policy["probe_extents"])) != len(policy["probe_extents"])
        or any(type(policy[k]) is not int or policy[k] < 1 for k in ("max_members", "max_extent_bits"))
    ):
        raise ValueError(
            "source curriculum requires explicit distinct positive teaching extents and finite metadata budgets"
        )
    if any(v.bit_length() > policy["max_extent_bits"] for v in policy["probe_extents"]):
        raise ValueError("source curriculum teaching extent exceeds its metadata budget")
    return copy.deepcopy(policy)


def derive_curriculum(*, software, hardware, policy):
    """Derive ordinary sweep inputs from live minimal semantics and RTL data.

    Geometry is a source probe until its tensor/resource/command mapping is
    proved. The complete requested family roster retains missing owners and
    missing hardware boundaries. No source count grants measured admission.
    """
    from .component_source_binding import identity
    from .profiles import HOST_SOURCE_BASIS

    policy = validate_curriculum_policy(policy)
    origin = identity(software, hardware=hardware)
    numeric = software.public_facts()["numerical_semantics"]
    binding = _binding(software, hardware)
    sweeps, unknowns, objectives = [], [], []
    requested = 0

    def missing(family, reason, **detail):
        unknowns.append({"family": family, "mandatory": True, "status": "UNKNOWN", "reason": reason, **detail})

    def add(family, original, axes, purpose, boundary=None):
        nonlocal requested
        count = 1
        for points in axes.values():
            count *= len(points)
        requested += count
        if (
            count > 128
            or requested > policy["max_members"]
            or any(point.bit_length() > policy["max_extent_bits"] for points in axes.values() for point in points)
        ):
            missing(
                family, "complete requested source family exceeds explicit metadata budget", requested_members=count
            )
            return
        prototype = copy.deepcopy(original)
        for row in prototype["inputs"]:
            row["shape"] = [axes[dim["axis"]][0] if isinstance(dim, dict) else dim for dim in row["shape"]]
        try:
            typed = component_program.analyze(
                prototype,
                operand_dtype=binding.mlir_dtype(binding.operand_dtype),
                accumulator_dtype=binding.mlir_dtype(binding.accum_dtype),
            )
            owners = node_owners(typed, software=software)
        except ValueError as exc:
            missing(family, "independent source semantics unavailable: " + str(exc), requested_members=count)
            return
        operations = sorted({row["operation"] for row in owners})
        source_basis = {
            "schema": HOST_SOURCE_BASIS,
            "kind": "host",
            "software_intake_sha256": software.sha256,
            "operations": operations,
            "numerical_reference": "integer_reference",
            "boundary": "complete_host_call",
        }
        limits = [
            "host/device placement and execution UNKNOWN",
            "physical allocation, alias, lifetime, fanout and completion UNKNOWN",
            "complete CPU/device cost, overlap and calibrated timing UNKNOWN",
            "finite source references do not establish whole-domain or whole-model support",
        ]
        if boundary is not None:
            limits.append("source geometry is conditional on the unproved tensor/resource/command mapping")
        performance = {
            "level": "L4_boundary" if "matmul" in operations else "L6_global",
            "family": family,
            "lever": purpose,
            "member_class": "OBJECTIVE",
            "claim": "DIFFERENTIAL",
            "comparand": {
                "kind": "candidate",
                "against": "same complete original source",
                "cancels": "identical input bytes and ordered published outputs",
                "demand_equal": "complete typed DAG",
            },
            "falsifier": {
                "observation": "complete source-bound call cost",
                "fires_when": "any output differs or complete cost increases",
                "negative_control": "unchanged compiler and identical inputs",
            },
            "gate": {
                "traits": [],
                "source_execution_basis": source_basis,
                "instrument": "independent complete host/device call observation required",
                "capacity": "UNKNOWN tensor/resource mapping",
                "on_missing": "skip_with_evidence",
            },
            "regime": {"separation": "independent source geometry", "layout": "explicit row-major tensor semantics"},
            "emitter": {"status": "existing", "entry": "component_program", "knobs": {}},
            "cost": {"tier": "L0 source/reference", "runs": 1, "projected_cycles": "unbounded", "basis": "unmeasured"},
            "curriculum": {
                "schema": CURRICULUM_SCHEMA,
                "purpose": purpose,
                "derivation": origin,
                "numerical_reference": copy.deepcopy(numeric),
                "evaluation_tiers": {"source_reference": "L0", "execution": "UNKNOWN", "timing": "UNKNOWN"},
                "resource_boundary": boundary
                or {"kind": "explicit independent teaching extents", "hardware_mapping": "UNKNOWN"},
                "limitations": limits,
            },
        }
        sweeps.append(
            {
                "id": family,
                "name": family + "_" + "_".join("{" + key + "}" for key in axes),
                "axes": axes,
                "fit_axes": [key for key, points in axes.items() if len(points) > 1],
                "source_reference": "live independent minimal SW and RTL source derivation",
                "base": {
                    "op": "component_program",
                    "kind": "model_slice",
                    "cat": "_perf",
                    "label": "dev",
                    "program": original,
                    "performance": performance,
                },
            }
        )
        objectives.append(
            {
                "family": family,
                "operations": sorted({row["owner"] for row in owners}),
                "objective": {
                    "metric": "complete_call_cycles",
                    "unit": "cycles",
                    "direction": "min",
                    "basis": "all host movement/arithmetic, device work and ordered output publication",
                },
            }
        )

    def graph(inputs, nodes, outputs):
        return {
            "schema": component_program.SCHEMA_V2,
            "inputs": [
                {"name": name, "role": role, "shape": shape, "dtype": dtype} for name, role, shape, dtype in inputs
            ],
            "nodes": nodes,
            "outputs": [{"name": name, "value": value} for name, value in outputs],
        }

    def node(name, op, inputs, **attrs):
        return {"name": name, "op": op, "inputs": inputs, **attrs}

    r, c = {"axis": "R"}, {"axis": "C"}
    axes = {"R": list(policy["probe_extents"]), "C": list(policy["probe_extents"])}
    copy_graph = graph(
        [("X", "input", [r, c], "operand")],
        [node("first", "copy", ["X"]), node("second", "copy", ["first"]), node("third", "copy", ["first"])],
        [("producer", "first"), ("left", "second"), ("right", "third")],
    )
    add("host_movement_fanout", copy_graph, axes, "movement with a published producer and two live consumers")
    add(
        "host_cast",
        graph(
            [("X", "input", [r, c], "operand")],
            [node("wide", "cast", ["X"], dtype="accumulator"), node("narrow", "cast", ["wide"], dtype="operand")],
            [("wide_output", "wide"), ("narrow_output", "narrow")],
        ),
        axes,
        "signed conversion and narrowing with both results checked",
    )
    add(
        "host_reduction",
        graph(
            [("X", "input", [r, c], "operand")],
            [
                node("rows", "reduce_sum", ["X"], axis=1, dtype="accumulator"),
                node("columns", "reduce_sum", ["X"], axis=0, dtype="accumulator"),
            ],
            [("row_sums", "rows"), ("column_sums", "columns")],
        ),
        axes,
        "true axis reductions over every source element",
    )
    mixed = graph(
        [("A", "input", [r, c], "operand"), ("W", "weight", [c, r], "operand"), ("V", "weight", [1, r], "operand")],
        [
            node("moved", "copy", ["A"]),
            node("product", "matmul", ["moved", "W"]),
            node("reduced", "reduce_sum", ["product"], axis=1, dtype="accumulator"),
            node("narrow", "cast", ["reduced"], dtype="operand"),
            node("final", "matmul", ["narrow", "V"]),
        ],
        [("producer", "product"), ("host_sum", "reduced"), ("converted", "narrow"), ("result", "final")],
    )
    add("mixed_closed_boundary", mixed, axes, "closed movement/contraction/reduction/cast/contraction composition")
    reuse = graph(
        [("A", "input", [r, c], "operand"), ("B", "input", [r, c], "operand"), ("W", "weight", [c, r], "operand")],
        [
            node("left", "matmul", ["A", "W"]),
            node("right", "matmul", ["B", "W"]),
            node("sum", "add", ["left", "right"]),
        ],
        [("left_result", "left"), ("right_result", "right"), ("joined", "sum")],
    )
    add(
        "immutable_reuse_live_outputs",
        reuse,
        axes,
        "immutable weight reuse with simultaneously published intermediate results",
    )
    state = graph(
        [("A", "input", [r, c], "operand"), ("W", "weight", [c, r], "operand")],
        [
            node("product", "matmul", ["A", "W"]),
            node("old", "copy", ["product"]),
            node("view", "alias", ["product"]),
            node("branch", "copy", ["view"]),
            node("updated", "update", ["product", "branch"]),
        ],
        [("old_epoch", "old"), ("new_epoch", "updated"), ("alias_read", "view"), ("escaped", "branch")],
    )
    add("logical_state_lifetimes", state, axes, "old/new logical state and escaped values across an update")
    facts = hardware.public_facts()["facts"]
    bits = int(binding.mlir_dtype(binding.operand_dtype)[1:])
    memories = facts.get("memories") or []
    if not memories:
        missing("joint_resource_boundaries", "no independently derived memory volume")
    arrays = facts.get("arrays") or []
    for ordinal, memory in enumerate(memories):
        capacity = memory.get("bytes") if isinstance(memory, dict) else None
        if type(capacity) is not int or capacity <= 0:
            missing("joint_resource_boundaries", "memory volume remains unknown", memory_ordinal=ordinal)
            continue
        if capacity * 8 % bits:
            missing(
                "source_volume_equality",
                "memory bit volume cannot be represented by an exact whole-element source",
                memory_ordinal=ordinal,
                requested_comparisons=["<", "=", ">"],
            )
            continue
        base = capacity * 8 // bits
        if base < 1:
            missing("joint_resource_boundaries", "memory volume cannot hold one source element", memory_ordinal=ordinal)
            continue
        volume_points = boundary_witnesses(Bound("source_elements", base, f"memories.{ordinal}.bytes / element_bits"))
        if any(value is None for value in volume_points.values()):
            missing("joint_resource_boundaries", "no positive below-boundary source geometry", memory_ordinal=ordinal)
            continue
        boundary = {
            "kind": "source_storage_volume",
            "fact": f"memories.{ordinal}.bytes",
            "observed_bytes": capacity,
            "element_bits": bits,
            "source_elements": base,
            "comparisons": ["<", "=", ">"],
            "hardware_mapping": "UNKNOWN",
        }
        add(
            f"source_volume_{ordinal}",
            copy_graph,
            {"R": [1], "C": list(volume_points.values())},
            "source storage below, at and above observed memory volume with live fanout",
            boundary,
        )
        depth = memory.get("depth")
        if type(depth) is int and depth > 1:
            depth_points = boundary_witnesses(Bound("source_rows", depth, f"memories.{ordinal}.depth"))
            add(
                f"joint_source_depth_volume_{ordinal}",
                copy_graph,
                {"R": list(depth_points.values()), "C": list(volume_points.values())},
                "joint source row-count and row-volume boundaries with published producer fanout",
                {
                    **boundary,
                    "kind": "joint_source_scalar_comparisons",
                    "source_axes": {
                        "R": {"fact": f"memories.{ordinal}.depth", "value": depth},
                        "C": {"fact": f"memories.{ordinal}.bytes", "source_elements": base},
                    },
                    "comparison_scope": "each source axis only; complete allocation and physical tails UNKNOWN",
                },
            )
        else:
            missing("joint_source_depth_volume", "memory depth lacks a positive lower witness", memory_ordinal=ordinal)
        if not arrays:
            missing("joint_tail_capacity", "no independently mapped array/tensor axes", memory_ordinal=ordinal)
        for array_ordinal, array in enumerate(arrays):
            rows, cols = (array.get(key) if isinstance(array, dict) else None for key in ("rows", "cols"))
            if any(type(v) is not int or v <= 1 for v in (rows, cols)):
                missing(
                    "joint_tail_capacity",
                    "array geometry remains unknown",
                    memory_ordinal=ordinal,
                    array_ordinal=array_ordinal,
                )
                continue
            # Row and column tails are crossed jointly. The volume boundary
            # remains an independently observed scalar source count; no
            # physical capacity or tensor axis role is inferred.
            joint = graph(
                [("X", "input", [r, c], "operand"), ("Volume", "input", [1, {"axis": "S"}], "operand")],
                copy_graph["nodes"] + [node("volume_copy", "copy", ["Volume"])],
                [("producer", "first"), ("left", "second"), ("right", "third"), ("volume", "volume_copy")],
            )
            add(
                f"joint_tail_capacity_{ordinal}_{array_ordinal}",
                joint,
                {
                    "R": list(boundary_witnesses(Bound("source_rows", rows, f"arrays.{array_ordinal}.rows")).values()),
                    "C": list(boundary_witnesses(Bound("source_cols", cols, f"arrays.{array_ordinal}.cols")).values()),
                    "S": list(volume_points.values()),
                },
                "joint row/column source tails and memory-volume boundary with live producer fanout",
                {
                    **boundary,
                    "array_facts": {"rows": f"arrays.{array_ordinal}.rows", "cols": f"arrays.{array_ordinal}.cols"},
                },
            )
    missing(
        "physical_resource_mapping", "source facts do not establish command/tensor/capacity/reuse/lifetime mappings"
    )
    missing("connected_stream_overlap", "connected engine and overlap execution facts remain independently required")
    derivation = {
        "schema": CURRICULUM_SCHEMA,
        "sources": origin,
        "policy": copy.deepcopy(policy),
        "requested_members": requested,
        "required_unknowns": unknowns,
        "objectives": objectives,
        "qualification": "ordinary source/reference derivation only; no phase or timing admission",
    }
    derivation["sha256"] = digest(derivation)
    return {"sweeps": sweeps, "curriculum_derivation": derivation}


def verify_curriculum_template(path, *, software, hardware):
    selected = yaml.safe_load(Path(path).read_bytes())
    declaration = selected.get("curriculum_derivation") if isinstance(selected, dict) else None
    if not isinstance(declaration, dict):
        raise ValueError("source curriculum v2 requires its original automatic template derivation")
    actual = derive_curriculum(software=software, hardware=hardware, policy=declaration.get("policy"))
    if selected != actual:
        raise ValueError("source curriculum template differs from live complete automatic derivation")
    return copy.deepcopy(actual["curriculum_derivation"])


def require_selection(selection, *, source_components):
    if selection is not None and (selection not in {SCHEMA, SCHEMA_V2} or not source_components):
        raise ValueError("source performance preparation needs an explicit supported live backend-free selection")


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


def materialize(entry, binding, *, point, gate, selection=SCHEMA):
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
        "schema": selection,
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
    if selection == SCHEMA_V2:
        value["performance"]["curriculum"]["point"] = copy.deepcopy(point)
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
    if not isinstance(mode, dict) or mode.get("schema") not in {SCHEMA, SCHEMA_V2}:
        raise ValueError("source performance preparation selection is absent")
    paths = []
    for key in ("recipe", "shared_template"):
        pin = identity[key]
        if _pin(pin["path"]) != pin:
            raise ValueError("source performance generation inputs changed")
        paths.append(Path(pin["path"]))
    if mode["schema"] == SCHEMA_V2:
        derivation = verify_curriculum_template(paths[1], software=software, hardware=hardware)
        if identity.get("source_curriculum") != derivation:
            raise ValueError("source curriculum generation identity differs from live ordinary template replay")
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
        source_preparation=mode["schema"],
    )
    declaration = coverage["declaration"]
    obligations = tuple(
        ComponentObligation(row["id"], row["mandatory"], row["cohort"], json.dumps(row))
        for row in declaration["obligations"]
    )
    plan = ComponentCoveragePlan("", identity["component_coverage_plan_sha256"], json.dumps(declaration), obligations)
    functional, _ = expand(plan, binding=binding, evidence=None)
    missing = blocked + errors + skipped
    if mode["schema"] == SCHEMA_V2:
        missing += derivation["required_unknowns"]
    return development + functional, development, binding, missing


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
        "schema": identity["source_performance_preparation"]["schema"],
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
    if identity["source_performance_preparation"]["schema"] == SCHEMA_V2:
        data["curriculum_derivation"] = copy.deepcopy(identity["source_curriculum"])
    data["sha256"] = digest(data)
    return data


def verify_source_contracts(record, **inputs):
    if record != prepare_source_contracts(**inputs):
        raise ValueError("source performance contracts changed from complete original source/reference replay")
    return copy.deepcopy(record)
