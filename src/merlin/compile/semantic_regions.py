"""Outline selected semantic regions into the existing application dispatch IR.

The target supplies legal ``RegionAlternative`` choices. This module preserves
their source slices and checks wiring; it neither chooses numerical policies
nor interprets missing operators. Allocation and C emission consume the normal
``MerlinProgram`` after outlining.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from hashlib import sha256

from merlin.semantic_compiler.model import KernelRequest, SemanticNode
from merlin.xdsl_dialects.lowering.dispatch_program import Buffer, DispatchProgram, Node
from merlin.xdsl_dialects.lowering.global_plan import GlobalPlan, verify_global_plan
from merlin.xdsl_dialects.lowering.global_plan_emission import (
    BoundaryMapping,
    EmittedComponent,
    GlobalPlanEmission,
    dispatch_digest,
    verify_global_plan_emission,
)


def semantic_dispatch(request: KernelRequest, *, entry: str) -> DispatchProgram:
    """Expose every computation and constant, without target placement decisions.

    This boundary is for logical graphs. Already allocated views need a separate
    physical representation transition and cannot be relabeled as dense buffers.
    Input ordering follows the declared request boundary, not sorted SSA names.
    """
    if not entry or request.physical_extents or request.boundary_layouts:
        raise ValueError("semantic outlining needs an entry and an unallocated logical graph")
    _output_storages(request)
    args = {key: i for i, (key, _) in enumerate(request.input_storages)}
    source_digest = request.digest()
    return DispatchProgram(
        entry,
        list(range(len(args))),
        {
            node.id: Buffer(
                node.id,
                list(node.type.shape),
                node.type.dtype,
                "arg" if node.effect == "input" else "const" if node.effect == "constant" else "intermediate",
                args.get(node.id),
            )
            for node in request.nodes
        },
        [
            Node(
                "dispatch",
                node.op,
                list(node.inputs),
                [node.id],
                {
                    "source_node": node.id,
                    "source_request_digest": source_digest,
                },
            )
            for node in request.nodes
            if node.effect == "pure"
        ],
        list(request.outputs),
    )


def extract_semantic_region(request: KernelRequest, node_ids: tuple[str, ...]) -> KernelRequest:
    """Retain exact operations, constants and every value escaping a source slice.

    An outside producer becomes a typed input. Constants stay exact byte-bound
    constants. No cast, layout change, initialization or arithmetic rewrite is
    implied by a cut. Outputs include original results before other escaping
    values; the application retains the original ordered result tuple.
    """
    inside = set(node_ids)
    if not inside or len(inside) != len(node_ids):
        raise ValueError("a semantic region needs distinct computation IDs")
    if request.physical_extents or request.boundary_layouts:
        raise ValueError("cannot cut an allocated physical graph as logical tensors")
    if any(request.node(key).effect != "pure" for key in inside):
        raise ValueError("region computation IDs cannot include boundary values or constants")
    reads = {key for node in request.nodes if node.id in inside for key in node.inputs} - inside
    outside_reads = {key for node in request.nodes if node.id not in inside for key in node.inputs}
    outputs = tuple(
        dict.fromkeys(
            [
                *(key for key in request.outputs if key in inside),
                *(node.id for node in request.nodes if node.id in inside and node.id in outside_reads),
            ]
        )
    )
    if not outputs:
        raise ValueError("a selected region has no escaping result; eliminate dead code explicitly")
    nodes, inputs, constants = [], [], []
    constant_bindings = {binding.node_id: binding for binding in request.constants}
    for node in request.nodes:
        if node.id in inside:
            nodes.append(node)
        elif node.id in reads:
            if node.effect == "constant":
                nodes.append(node)
                constants.append(constant_bindings[node.id])
            else:
                # A reusable application input may have later region readers.
                # Its region-local last read cannot authorize destruction of the
                # shared buffer. Domains remain valid; retention is conservative.
                nodes.append(
                    replace(
                        node,
                        attrs=tuple(
                            (key, "preserve" if key == "input_retention" else value) for key, value in node.attrs
                        ),
                    )
                    if node.effect == "input"
                    else SemanticNode(
                        node.id,
                        "input",
                        (),
                        node.type,
                        effect="input",
                    )
                )
                inputs.append((node.id, dict(request.input_storages).get(node.id, "external")))
    original_outputs = _output_storages(request)
    identity = sha256(
        json.dumps([request.digest(), [node.id for node in nodes]], separators=(",", ":")).encode()
    ).hexdigest()
    return KernelRequest(
        tuple(nodes),
        outputs,
        tuple(original_outputs.get(key, "external") for key in outputs),
        tuple(inputs),
        request.target_identity,
        request.lowering_policy,
        identity,
        tuple(constants),
    )


def _output_storages(request: KernelRequest) -> dict[str, str]:
    result = {}
    for key, storage in zip(request.outputs, request.output_storages, strict=True):
        if key in result and result[key] != storage:
            raise ValueError("distinct physical representations of one result need explicit transitions")
        result[key] = storage
    return result


def partition_semantic_dag(request: KernelRequest, *, max_nodes: int) -> tuple[KernelRequest, ...]:
    """Cut a pure DAG into bounded regions after a deterministic pressure order.

    Region boundaries are complete tensors. The ordering is a node-count
    heuristic; target compilation still decides register feasibility. No
    arithmetic, output order, representation or initialization is changed.
    """
    from merlin.semantic_compiler.allocate import CandidateGraph, Value, topological_orders

    if type(max_nodes) is not int or max_nodes <= 0:
        raise ValueError("region size must be a positive integer")
    if any(node.effect not in {"input", "constant", "pure"} for node in request.nodes):
        raise ValueError("staging requires a pure semantic DAG")
    indices = {node.id: index for index, node in enumerate(request.nodes)}
    graph = CandidateGraph(tuple(
        Value(index, node.op, "logical", 1, tuple(indices[key] for key in node.inputs), node.id,
              "instruction" if node.effect == "pure" else node.effect)
        for index, node in enumerate(request.nodes)
    ), tuple(indices[key] for key in request.outputs))
    order = next(topological_orders(graph, limit=1))
    return tuple(
        extract_semantic_region(request, tuple(request.nodes[index].id for index in order[start:start + max_nodes]))
        for start in range(0, len(order), max_nodes)
    )


def semantic_staging_buffers(
    request: KernelRequest,
    regions: tuple[KernelRequest, ...],
    *,
    storage: str,
    extents: dict[str, int],
    fixed: dict[str, int],
):
    """Describe buffers for sequential, completed region executions.

    A region's inputs remain live through its output writes, prohibiting
    within-region aliasing. Source inputs/constants remain immutable for the
    invocation; results remain live after their producing region. The target
    must preserve these completion events when composing physical programs.
    """
    from merlin.semantic_compiler.buffer_allocation import Buffer

    source = {node.id: node for node in request.nodes}
    available = {node.id for node in request.nodes if node.effect in {"input", "constant"}}
    births = dict.fromkeys(available, 0)
    ends = dict.fromkeys(available, len(regions) + 1)
    covered: set[str] = set()
    for event, region in enumerate(regions, 1):
        computations = tuple(node.id for node in region.nodes if node.effect == "pure")
        if covered.intersection(computations) or any(key not in source for key in computations):
            raise ValueError("staged regions duplicate or add a source computation")
        expected = extract_semantic_region(request, computations)
        if region.record() != expected.record():
            raise ValueError("staged region differs from its exact source slice")
        leaves = {node.id for node in region.nodes if node.effect in {"input", "constant"}}
        if not leaves <= available:
            raise ValueError("staged region reads a value before its producer completed")
        for key in leaves:
            ends[key] = max(ends[key], event)
        for key in region.outputs:
            if key in births:
                raise ValueError("staged buffer has more than one producer")
            births[key] = ends[key] = event
        available.update(region.outputs)
        covered.update(computations)
    if covered != {node.id for node in request.nodes if node.effect == "pure"}:
        raise ValueError("staged regions omit source computations")
    if not set(request.outputs) <= available or not set(fixed) <= births.keys():
        raise ValueError("staged buffers omit an output or name an unknown fixed buffer")
    for key in request.outputs:
        ends[key] = len(regions) + 1
    return tuple(
        Buffer(key, storage, extents[key], births[key], ends[key], fixed_start=fixed.get(key),
               persistent=source[key].effect in {"input", "constant"})
        for key in source if key in births
    )


@dataclass(frozen=True)
class SemanticRegionOutlining:
    """Selected source slices and the checked dispatch rewrite they implement."""

    source_digest: str
    logical: DispatchProgram
    plan: GlobalPlan
    emission: GlobalPlanEmission
    regions: tuple[tuple[str, KernelRequest], ...]

    def emit_global_plan(self, program: DispatchProgram, plan: GlobalPlan) -> GlobalPlanEmission:
        if dispatch_digest(program) != dispatch_digest(self.logical) or plan.digest != self.plan.digest:
            raise ValueError("semantic outlining differs from the selected source plan")
        return self.emission


def outline_semantic_regions(
    request: KernelRequest,
    *,
    entry: str,
    plan: GlobalPlan,
) -> SemanticRegionOutlining:
    """Check complete coverage, derive region ports and topologically order calls.

    Non-contiguous source regions are legal only when their quotient graph is
    acyclic. A region's position cannot be inferred from its first or last source
    operation. Explicit representation transitions must be lowered by their
    target emitter; this outliner supports unchanged boundary representations.
    """
    logical = semantic_dispatch(request, entry=entry)
    source_digest = request.digest()
    problems = verify_global_plan(logical, plan)
    if problems:
        raise ValueError("invalid source region plan: " + "; ".join(problems))
    if plan.transitions:
        raise ValueError("representation transitions need an explicit target emitter")
    owners = {logical.nodes[index].outputs[0]: item.id for item in plan.selected for index in item.nodes}
    selected = {item.id: item for item in plan.selected}
    slices = {}
    boundaries = {}
    dependencies = {}
    for item in plan.selected:
        keys = tuple(logical.nodes[index].outputs[0] for index in item.nodes)
        region = extract_semantic_region(request, keys)
        inputs = tuple(node.id for node in region.nodes if node.effect in {"input", "constant"})
        outputs = region.outputs
        for representations, expected in ((item.inputs, inputs), (item.outputs, outputs)):
            if {row.buffer for row in representations} != set(expected):
                raise ValueError("selected region representations omit or add a source boundary value")
            if any(row.representation.dtype != request.node(row.buffer).type.dtype for row in representations):
                raise ValueError("a source region boundary changes dtype without a conversion")
        slices[item.id] = region
        boundaries[item.id] = inputs, outputs
        dependencies[item.id] = {owners[key] for key in inputs if key in owners}
    pending, ordered = set(selected), []
    while pending:
        ready = [key for key in pending if not dependencies[key].intersection(pending)]
        if not ready:
            raise ValueError("selected fusion regions create a cyclic application graph")
        key = min(ready, key=lambda key: (selected[key].nodes, key))
        ordered.append(key)
        pending.remove(key)
    calls, receipts = [], []
    kept = {key for key, buffer in logical.buffers.items() if buffer.kind in {"arg", "const"}} | set(request.outputs)
    for key in ordered:
        inputs, outputs = boundaries[key]
        kept.update((*inputs, *outputs))
        calls.append(
            Node(
                "dispatch",
                selected[key].implementation,
                list(inputs),
                list(outputs),
                {
                    "source_request_digest": source_digest,
                    "selected_region": key,
                    "region_request_digest": slices[key].digest(),
                },
            )
        )
        receipts.append(
            EmittedComponent(
                key, (len(calls) - 1,), tuple((name, name) for name in inputs), tuple((name, name) for name in outputs)
            )
        )
    dispatch = DispatchProgram(
        entry,
        logical.args,
        {key: value for key, value in logical.buffers.items() if key in kept},
        calls,
        list(request.outputs),
    )
    mappings = tuple(
        BoundaryMapping("input", key, key) for key, buffer in logical.buffers.items() if buffer.kind in {"arg", "const"}
    ) + tuple(BoundaryMapping("output", key, key) for key in dict.fromkeys(request.outputs))
    emission = GlobalPlanEmission(
        dispatch,
        plan.digest,
        dispatch_digest(logical),
        tuple(receipts),
        (),
        mappings,
        ("source-semantic-request:" + source_digest,),
    )
    problems = verify_global_plan_emission(logical, plan, emission)
    if problems:
        raise ValueError("invalid semantic region wiring: " + "; ".join(problems))
    return SemanticRegionOutlining(source_digest, logical, plan, emission, tuple((key, slices[key]) for key in ordered))
