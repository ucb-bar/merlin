"""Source-driven fusion exploration around existing compiler APIs, not a pass runner.

Discovery grants no legality or speedup. Realization returns the actual native
product and original alternative for the caller's ordinary emission pipeline.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import math

from merlin.common.digest import is_sha256
from merlin.transforms.certificates import ir_digest


ROUTES = {
    "adjacent_elementwise": "merlin.compile.scheduling.fusion.plan_fusion",
    "epilogue": "merlin.xdsl_dialects.lowering.output_fusion.decisions",
    "shared_input": "merlin.compile.scheduling.shared_input_schedule.schedule_shared_input",
    "connected_residency": "merlin.compile.compiler_observations.lifetimes",
    "preparation_reuse": "merlin.compile.compiler_observations.install_prepared_observed",
}


def _boundary(operations, ordinals):
    # Regions may capture external SSA values without listing them as parent
    # operands. Include the entire owned tree, excluding its own block args.
    nested = tuple(dict.fromkeys(child for root in operations for child in root.walk()))
    members = set(nested)
    inputs, outputs = {}, {}
    for op in nested:
        for value in op.operands:
            owner = value.owner
            local_block_arg = hasattr(owner, "parent_op") and owner.parent_op() in members
            if owner not in members and not local_block_arg:
                inputs[value] = None
        for value in op.results:
            if any(use.operation not in members for use in value.uses):
                outputs[value] = None

    def value_record(value):
        owner = value.owner
        if owner in ordinals:
            identity = {"producer_ordinal": ordinals[owner], "result": list(owner.results).index(value)}
        else:
            parent = owner.parent_op() if hasattr(owner, "parent_op") else None
            identity = {"block_owner_ordinal": ordinals.get(parent), "argument": getattr(value, "index", None)}
        return {**identity, "type": str(value.type)}

    return {"live_inputs": [value_record(v) for v in inputs],
            "live_outputs": [value_record(v) for v in outputs],
            "source_ids": [{"ordinal": ordinals[op],
                            "region_id": getattr(op.attributes.get("prov.region_id"), "data", None),
                            "node_ids": [getattr(a, "data", None) for a in
                                         getattr(op.attributes.get("prov.source_node_ids"), "data", ())]}
                           for op in operations]}


def _structural_family(op):
    """Small public IR observation, never a hardware eligibility decision.

    Named matrix operations have declared contraction semantics. A generic is
    observed as elementwise only for parallel iterators and identity maps. Other
    forms remain unknown; selected realization readers own stronger proofs.
    """
    if op.name in {"linalg.matmul", "linalg.batch_matmul", "linalg.matvec", "linalg.vecmat", "linalg.dot"}:
        return "contraction"
    if op.name != "linalg.generic":
        return None
    iterators = op.properties.get("iterator_types")
    maps = op.properties.get("indexing_maps")
    if not iterators or not maps or not iterators.data or not maps.data:
        return None
    if (all(getattr(item.data, "value", None) == "parallel" for item in iterators.data)
            and all(item.data == item.data.identity(item.data.num_dims) for item in maps.data)):
        return "elementwise_map"
    return None


def discover(module, *, capability_contract, numerical_policy_sha256, preparation_requests=(),
             max_candidates=256):
    """Bounded typed-SSA candidates; facts do not substitute for native proofs.

Capability composition restrictions are retained. Absent facts stay unknown;
an available semantic family does not prove scalar equivalence or target effects.
No workload/model identifier or reference timing is a dispatch key.
"""
    from merlin.compile import compiler_observations as C
    from merlin.compile.scheduling.shared_input_groups import find_shared_input_groups
    from merlin.verify.lean_contract import document_digest

    if not is_sha256(numerical_policy_sha256) or type(max_candidates) is not int or not 0 < max_candidates <= 1024:
        raise ValueError("fusion discovery requires a numeric policy pin and bounded candidate count")
    members = tuple(module.walk())
    if len(members) > 10000:
        raise ValueError("fusion source operation budget exceeded")
    ordinals = {op: index for index, op in enumerate(members)}
    # Read only explicit public declarations. This advisory index deliberately
    # does not fold units, infer composite families, or consult eligibility.
    caps = {}
    for unit in capability_contract.get("compute_units", ()):
        if not isinstance(unit, dict):
            raise ValueError("public compute-unit declarations must be objects")
        for declaration in unit.get("semantic_capabilities", ()):
            if not isinstance(declaration, dict) or not isinstance(declaration.get("family"), str):
                raise ValueError("public semantic declarations must explicitly name a family")
            caps.setdefault(declaration["family"], []).append({
                "unit": unit.get("name"), "declaration": dict(declaration)})
    rows = []

    def add(route, operations, *, premise, detail=None):
        if len(rows) >= max_candidates:
            raise ValueError("fusion candidate budget exceeded; no truncated roster")
        families = sorted({_structural_family(op) for op in operations} - {None})
        selected = []
        for family in families:
            declarations = caps.get(family, [])
            selected.append({"family": family, "status": "declared" if declarations else "unknown",
                             "basis": "explicit_public_family_declaration_only",
                             "composed_with": [row["declaration"].get("composed_with") for row in declarations],
                             "forms": [row["declaration"].get("forms") for row in declarations],
                             "declarations": declarations,
                             "applicability": "unknown_until_native_legality_checks"})
        rows.append({"route": route, "entrypoint": ROUTES[route], "status": "unknown",
                     "reason": "source opportunity; native legality and selected realization still required",
                     "operation_ordinals": [ordinals[op] for op in operations],
                     **_boundary(operations, ordinals), "capability_observations": selected,
                     "legality_premises": premise,
                     "storage_lifetimes": "unproved", "schedule_sha256": None,
                     "whole_region_cost": None, "detail": detail})

    typed = {op: _structural_family(op) for op in members if any(hasattr(v.type, "get_shape") for v in op.results)}
    typed = {op: stage for op, stage in typed.items() if stage is not None}
    seen = set()
    for producer, stage in typed.items():
        for result in producer.results:
            for use in result.uses:
                consumer = use.operation
                if consumer not in typed or producer.parent is not consumer.parent or (producer, consumer) in seen:
                    continue
                seen.add((producer, consumer))
                block = tuple(producer.parent.ops)
                adjacent = block.index(consumer) == block.index(producer) + 1
                if stage == "contraction" and typed[consumer] != "contraction":
                    route = "epilogue"
                elif adjacent and stage != "contraction" and typed[consumer] != "contraction":
                    route = "adjacent_elementwise"
                else:
                    route = "connected_residency"
                add(route, (producer, consumer), premise=[
                    "complete live outputs and external observers retained",
                    "exact scalar/numeric order and selected standalone versus composed form",
                    "target storage ownership, lifetime, completion and capacity checked",
                    "actual transformed result consumed by ordinary emission"], detail={"adjacent": adjacent})
    shared_reader = None
    try:
        groups = find_shared_input_groups(module)
    except ValueError as error:
        groups = ()
        shared_reader = {"status": "unsupported", "scope": "current closed source reader",
                         "reason": str(error)[:240], "hardware_unsupported": False}
    if groups:
        from xdsl.dialects import func
        function = next(op for op in module.body.block.ops if type(op) is func.FuncOp)
        operations = tuple(function.body.block.ops)
        for group in groups:
            add("shared_input", tuple(operations[i] for i in group.source_op_indices), premise=[
                "native immutable shared SSA/zero-seed reader passed",
                "each output and epilogue retains independent state",
                "native shared-input storage/effect verification remains required"], detail=group.record())
    requests = tuple(preparation_requests)
    packing = C.packing(module, requests)
    fanout = C.connected_fusion(module, max_candidates=max_candidates)
    for opportunity in packing["opportunities"]:
        operations = tuple(members[i] for i in opportunity["consumer_ordinals"])
        add("preparation_reuse", operations, premise=[
            "explicit same format and actual SSA views",
            "source lifetime proof plus native physical preparation writer",
            "cross-invocation persistent ownership independently proved"], detail=opportunity)
    counts = Counter(row["route"] for row in rows)
    return {"planned": {"source_regions": len(rows)}, "emitted": {},
            "schema": "merlin.fusion_workflow.v1", "source_ir_sha256": ir_digest(module),
            "capability_sha256": document_digest(dict(capability_contract)),
            "numerical_policy_sha256": numerical_policy_sha256,
            "maturity": "source_opportunities_not_legalized", "physical_savings_claimed": False,
            "structural_reader_scope": "named matrix operations and parallel identity-map generics only",
            "unclassified_tensor_operations": [{"ordinal": ordinals[op], "operation": op.name}
                for op in members if any(hasattr(v.type, "get_shape") for v in op.results)
                and _structural_family(op) is None],
            "metric_definitions": {"planned.source_regions": "source opportunities, not legalized regions"},
            "routes": {route: {"status": "unknown" if counts[route] else "no-match", "candidates": counts[route]}
                       for route in ROUTES}, "shared_input_reader": shared_reader, "candidates": rows,
            "pointwise_fanout": fanout,
            "fallback": "preserve original source and unfused schedule; no discovery mutation",
            "missing": ["general connected graph residency realization", "native proof for each selected route",
                        "complete-region numerical checks and same-scope runtime cost"]}


@dataclass(frozen=True)
class Realization:
    artifact: object
    unfused: object
    receipt: dict
    objects: tuple = ()


def realize_adjacent(kernel, objects, banks, adapter, capabilities, **limits):
    """Return actual checked target calls, retaining the original unfused Kernel.

The existing finite physical-fusion verifier remains the owner. Counted-loop
regions must use their compact route; this helper never unrolls them.
"""
    from merlin.compile.scheduling.fusion import plan_fusion
    from merlin.sched.ir import Call

    before = kernel.digest()
    if any(type(node) is not Call for node in kernel.body):
        return Realization(kernel, kernel, {
            "route": "adjacent_elementwise", "status": "unsupported",
            "reason": "finite synchronous call route only; retain compact loops unchanged",
            "source_schedule_sha256": before, "schedule_sha256": before,
            "hardware_unsupported": False, "whole_region_cost": None,
            "planned": {}, "emitted": {},
        }, tuple(objects))
    candidate = plan_fusion(kernel, objects, banks, adapter, capabilities, **limits)
    if kernel.digest() != before:
        raise ValueError("native fusion mutated the retained unfused schedule")
    artifact = kernel if candidate is None else candidate.kernel
    return Realization(artifact, kernel, {
        "route": "adjacent_elementwise", "status": "no-match" if candidate is None else "applied",
        "planned": {} if candidate is None else {"source_regions": 1}, "emitted": {},
        "maturity": "checked_schedule_not_instruction_image",
        "source_schedule_sha256": before, "schedule_sha256": artifact.digest(),
        "checks": [] if candidate is None else list(candidate.checks),
        "objects": [asdict(obj) for obj in (objects if candidate is None else candidate.objects)],
        "emission": "actual returned Kernel must enter ordinary allocation/emission",
        "execution_qualification": "not_established", "whole_region_cost": None,
    }, tuple(objects if candidate is None else candidate.objects))


def realize_shared_input(module, group, **selection):
    """Return the native checked logical schedule; target encoding is still required.

    The original module is retained unchanged as the ordinary unfused alternative.
    Each output and later epilogue stays independent. Never call logical effects an
    emitted target image or claim that planned byte reduction is measured traffic.
    """
    from merlin.compile.scheduling.shared_input_schedule import schedule_shared_input
    from merlin.verify.lean_contract import document_digest

    before = ir_digest(module)
    candidate = schedule_shared_input(module, group, **selection)
    if ir_digest(module) != before:
        raise ValueError("shared-input scheduling mutated the retained source")
    record = None if candidate is None else asdict(candidate)
    return Realization(candidate if candidate is not None else module, module, {
        "route": "shared_input", "status": "no-match" if candidate is None else "applied",
        "planned": {} if candidate is None else {"source_regions": 1}, "emitted": {},
        "source_ir_sha256": before,
        "maturity": "logical_schedule_not_target_instruction_image",
        "schedule_sha256": None if record is None else document_digest(record),
        "schedule": record,
        "boundary": group.record(),
        "storage": {key: asdict(selection[key]) for key in ("resources", "storage")},
        "emission": "consume returned operations through target adapter and independent actual-effect checker",
        "execution_qualification": "not_established", "whole_region_cost": None,
    })


def realize_multioutput(module, operations, **selection):
    """Consume the existing closed fan-out compiler and its guarded actual images.

    Invalid effects/source are refused by the native compiler. This is a narrow
    pointwise route, not general contraction or whole-graph fusion.
    """
    from merlin.compile.scheduling.multioutput_fusion import compile_source_multioutput
    from merlin.compile.compiler_observations import emitted_multioutput

    before = ir_digest(module)
    product = compile_source_multioutput(module, operations, **selection)
    if ir_digest(module) != before:
        raise ValueError("multioutput compiler mutated the retained source")
    observed = emitted_multioutput(product)
    return Realization(product, module, {
        **observed, "route": "adjacent_elementwise", "form": "two_output_pointwise_fanout",
        "status": "applied", "source_ir_sha256": before,
        "emission": "consume the returned guarded compilation images in ordinary target emission",
        "whole_region_cost": None, "execution_qualification": "not_established",
    })


def compare_region_costs(baseline, candidate):
    """Compare explicit complete-region observations; issue no placement authority.

    Observation trust remains with the caller's timing/correctness owner. Missing
    measurements or unqualified models cannot turn planned transformations into
    a gain. Physical observations need no fictitious cycle-model identity.
    """
    keys = ("source_ir_sha256", "numeric_policy_sha256", "hardware_sha256", "scope_sha256")
    unknown = {"status": "unknown", "saved_cycles": None, "placement_authority": False}
    if not isinstance(baseline, dict) or not isinstance(candidate, dict):
        return {**unknown, "reason": "missing whole-region observations"}
    if any(baseline.get(k) != candidate.get(k) or not is_sha256(baseline.get(k)) for k in keys):
        return {**unknown, "reason": "source/numeric/hardware/scope mismatch"}
    method = baseline.get("method")
    if method not in {"physical", "model"} or candidate.get("method") != method:
        return {**unknown, "reason": "unknown or mismatched timing method"}
    if method == "model" and (
            not is_sha256(baseline.get("model_sha256"))
            or baseline.get("model_sha256") != candidate.get("model_sha256")
            or any(row.get("model_qualification") != "qualified" for row in (baseline, candidate))):
        return {**unknown, "reason": "unbound, mismatched or unqualified cycle model"}
    if any(row.get("scope") != "complete_region" or row.get("complete") is not True
           or row.get("correctness") != "pass"
           or type(row.get("cycles")) not in (int, float) or not math.isfinite(row["cycles"])
           or row["cycles"] < 0 or not is_sha256(row.get("artifact_sha256"))
           for row in (baseline, candidate)):
        return {**unknown, "reason": "partial, incorrect, unbound or invalid complete-region observations"}
    return {"status": "reported_comparison", "saved_cycles": baseline["cycles"] - candidate["cycles"],
            "baseline": baseline, "candidate": candidate, "placement_authority": False,
            "qualification": "supplied observations; native timing and correctness owners remain authoritative"}
