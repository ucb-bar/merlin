"""Source-driven fusion exploration around existing compiler APIs, not a pass runner.

Discovery grants no legality or speedup. Realization returns the actual native
product and original alternative for the caller's ordinary emission pipeline.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
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
    members = set(operations)
    inputs, outputs = {}, {}
    for op in operations:
        for value in op.operands:
            if value.owner not in members:
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


def discover(module, *, capability_contract, numerical_policy_sha256, preparation_requests=(),
             max_candidates=256):
    """Bounded typed-SSA candidates; facts do not substitute for native proofs.

Capability composition restrictions are retained. Absent facts stay unknown;
an available semantic family does not prove scalar equivalence or target effects.
No workload/model identifier or reference timing is a dispatch key.
"""
    from merlin.compile import compiler_observations as C
    from merlin.compile.scheduling.shared_input_groups import find_shared_input_groups
    from merlin.targetgen.eligibility import capability_map_from_contract, _family_support
    from merlin.verify.lean_contract import document_digest
    from merlin.xdsl_dialects.lowering import compute_groups as CG

    if not is_sha256(numerical_policy_sha256) or type(max_candidates) is not int or not 0 < max_candidates <= 1024:
        raise ValueError("fusion discovery requires a numeric policy pin and bounded candidate count")
    members = tuple(module.walk())
    if len(members) > 10000:
        raise ValueError("fusion source operation budget exceeded")
    ordinals = {op: index for index, op in enumerate(members)}
    caps = capability_map_from_contract(dict(capability_contract))
    rows = []

    def add(route, operations, *, premise, detail=None):
        if len(rows) >= max_candidates:
            raise ValueError("fusion candidate budget exceeded; no truncated roster")
        stages = [CG.classify(op) for op in operations]
        families = sorted({CG.FAMILY_OF_STAGE.get(stage.kind) for stage in stages if stage is not None} - {None})
        selected = []
        for family in families:
            declarations, _ = _family_support(family, caps)
            selected.append({"family": family, "status": "declared" if declarations else "unknown",
                             "composed_with": [list(cap.composed_with) for cap in declarations or ()],
                             "forms": [list(cap.forms) for cap in declarations or ()]})
        rows.append({"route": route, "entrypoint": ROUTES[route], "status": "unknown",
                     "reason": "source opportunity; native legality and selected realization still required",
                     "operation_ordinals": [ordinals[op] for op in operations],
                     **_boundary(operations, ordinals), "capability_observations": selected,
                     "legality_premises": premise,
                     "storage_lifetimes": "unproved", "schedule_sha256": None,
                     "whole_region_cost": None, "detail": detail})

    typed = {op: CG.classify(op) for op in members if any(hasattr(v.type, "get_shape") for v in op.results)}
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
                if stage.kind == CG.CONTRACTION and typed[consumer].kind != CG.CONTRACTION:
                    route = "epilogue"
                elif adjacent and stage.kind != CG.CONTRACTION and typed[consumer].kind != CG.CONTRACTION:
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
            "routes": {route: {"status": "unknown" if counts[route] else "no-match", "candidates": counts[route]}
                       for route in ROUTES}, "shared_input_reader": shared_reader, "candidates": rows,
            "fallback": "preserve original source and unfused schedule; no discovery mutation",
            "missing": ["general connected graph residency realization", "native proof for each selected route",
                        "complete-region numerical checks and same-scope runtime cost"]}


@dataclass(frozen=True)
class Realization:
    artifact: object
    unfused: object
    receipt: dict


def realize_adjacent(kernel, objects, banks, adapter, capabilities, **limits):
    """Return actual checked target calls, retaining the original unfused Kernel.

The existing finite physical-fusion verifier remains the owner. Counted-loop
regions must use their compact route; this helper never unrolls them.
"""
    from merlin.compile.scheduling.fusion import plan_fusion

    before = kernel.digest()
    candidate = plan_fusion(kernel, objects, banks, adapter, capabilities, **limits)
    if kernel.digest() != before:
        raise ValueError("native fusion mutated the retained unfused schedule")
    artifact = kernel if candidate is None else candidate.kernel
    return Realization(artifact, kernel, {
        "route": "adjacent_elementwise", "status": "no-match" if candidate is None else "applied",
        "source_schedule_sha256": before, "schedule_sha256": artifact.digest(),
        "checks": [] if candidate is None else list(candidate.checks),
        "objects": objects if candidate is None else candidate.objects,
        "emission": "actual returned Kernel must enter ordinary allocation/emission",
        "execution_qualification": "not_established", "whole_region_cost": None,
    })


def compare_region_costs(baseline, candidate):
    """Compare supplied complete-region observations; issue no placement authority."""
    keys = ("source_ir_sha256", "numeric_policy_sha256", "hardware_sha256",
            "scope_sha256", "model_sha256", "method")
    unknown = {"status": "unknown", "saved_cycles": None, "placement_authority": False}
    if not isinstance(baseline, dict) or not isinstance(candidate, dict):
        return {**unknown, "reason": "missing whole-region observations"}
    if any(baseline.get(k) != candidate.get(k) or not baseline.get(k) for k in keys):
        return {**unknown, "reason": "source/numeric/hardware/model/scope mismatch"}
    if any(not is_sha256(baseline[k]) for k in keys if k.endswith("sha256")):
        return {**unknown, "reason": "unbound observation identity"}
    if any(row.get("scope") != "complete_region" or row.get("complete") is not True
           or type(row.get("cycles")) not in (int, float) or not math.isfinite(row["cycles"])
           or row["cycles"] < 0 or not is_sha256(row.get("artifact_sha256"))
           for row in (baseline, candidate)):
        return {**unknown, "reason": "partial, unbound or invalid complete-region observations"}
    return {"status": "reported_comparison", "saved_cycles": baseline["cycles"] - candidate["cycles"],
            "baseline": baseline, "candidate": candidate, "placement_authority": False,
            "qualification": "supplied observations; native timing and correctness owners remain authoritative"}
