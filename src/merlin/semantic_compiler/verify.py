"""Replay a selected native graph against source semantics and target facts.

This checker is deliberately separate from rule generation and Z3 formula
construction. It validates only the declared finite semantic/placement model;
it does not certify a target instruction stream or numerical hardware model.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from .allocate import (
    AllocationResult,
    CandidateGraph,
    Reservation,
    StorageBank,
    check_assignment,
    instruction_schedule,
    lower_candidate,
)
from .egg_bridge import Exploration
from .extract import Candidate, Choice
from .model import KernelRequest
from .rules import ConcatViewDescriptor, InstructionDescriptor, RuleProgram, ViewDescriptor, bundle_catalog


@dataclass(frozen=True)
class SelectionCheck:
    valid: bool
    fingerprint: str
    reason: str = ""


def check_timing(
    graph: CandidateGraph, order: tuple[int, ...], issue_times: tuple[tuple[int, int], ...]
) -> tuple[bool, str]:
    """Replay each read and completion event from the selected instruction graph."""
    instructions = {value.id for value in graph.values if value.kind == "instruction"}
    if any(value.id != index for index, value in enumerate(graph.values)):
        return False, "physical schedule has invalid value identities"
    if any(child_id < 0 or child_id >= value.id for value in graph.values for child_id in value.children):
        return False, "physical schedule has invalid value dependencies"
    if any(
        value.projection_of is None or value.projection_of >= value.id
        or graph.value(value.projection_of).kind != "instruction"
        or value.instruction_instance != graph.value(value.projection_of).instruction_instance
        for value in graph.values if value.kind == "projection"
    ):
        return False, "physical schedule has an unowned multi-result projection"
    if len(order) != len(instructions) or set(order) != instructions:
        return False, "physical schedule omits or duplicates an instruction"
    if len(issue_times) != len(order) or tuple(value_id for value_id, _ in issue_times) != order:
        return False, "physical schedule omits, duplicates or reorders an instruction"

    def producer_ready(child_id: int, read_time: int) -> bool:
        """Follow a non-emitting view to the instruction that produced its bits."""
        if child_id < 0 or child_id >= len(graph.values):
            return False
        child = graph.value(child_id)
        if child.kind == "instruction":
            return child_id in completed and completed[child_id] <= read_time
        if child.kind == "projection":
            return child.projection_of in completed and completed[child.projection_of] <= read_time
        if child.kind in {"input", "constant"}:
            return not child.children
        if child.kind != "view" or not child.children or any(source >= child_id for source in child.children):
            return False
        return all(producer_ready(source, read_time) for source in child.children)

    completed: dict[int, int] = {}
    previous_completion = 0
    for value_id, issue in issue_times:
        value = graph.value(value_id)
        if value.kind != "instruction" or type(issue) is not int or issue <= previous_completion:
            return False, "physical schedule issues before prior completion"
        if value.completion_offset < 0 or (
            value.input_read_offsets and len(value.input_read_offsets) != len(value.children)
        ):
            return False, "physical schedule has malformed execution timing"
        for index, child_id in enumerate(value.children):
            read_offset = value.read_offset(index)
            if read_offset < 0 or read_offset > value.completion_offset:
                return False, "input read occurs outside the instruction execution interval"
            if not producer_ready(child_id, issue + read_offset):
                return False, "input read precedes producer completion"
        previous_completion = issue + value.completion_offset
        completed[value_id] = previous_completion
    return True, ""


def _realized_extent(program: RuleProgram, choice: Choice) -> int | None:
    """Physical extent the lowered graph gives this selected choice."""
    metadata = program.symbols.get(choice.symbol)
    if metadata is None:
        return None
    extent = (
        metadata["descriptor"].get("extent") if metadata["kind"] in {"instruction", "view"}
        else metadata.get("extent")
    )
    return extent if type(extent) is int else None


def check_selection(
    request: KernelRequest,
    descriptors: tuple[InstructionDescriptor, ...],
    program: RuleProgram,
    exploration: Exploration,
    candidate: Candidate,
    graph: CandidateGraph,
    allocation: AllocationResult,
    banks: tuple[StorageBank, ...],
    *,
    views: tuple[ViewDescriptor | ConcatViewDescriptor, ...] = (),
    fixed_inputs: dict[str, int] | None = None,
    reservations: tuple[Reservation, ...] = (),
    fixed_outputs: tuple[int | None, ...] | None = None,
) -> SelectionCheck:
    def fail(reason: str) -> SelectionCheck:
        return SelectionCheck(False, "", reason)

    if allocation.status != "feasible" or len(candidate.outputs) != len(request.outputs):
        return fail("selection lacks a feasible assignment or ordered roots")
    if not candidate.bundles_complete():
        return fail("selected multi-result instruction omitted a projection")
    try:
        expected_graph = lower_candidate(candidate, program)
    except (KeyError, ValueError) as exc:
        return fail(f"selected multi-result instruction is malformed: {exc}")
    if graph != expected_graph:
        return fail("selected instruction graph changed after extraction")
    by_name = {descriptor.name: descriptor for descriptor in descriptors}
    if len(by_name) != len(descriptors):
        return fail("target has duplicate instruction identities")
    try:
        bundles = bundle_catalog(descriptors)
    except ValueError as exc:
        return fail(str(exc))
    views_by_name = {view.name: view for view in views}
    if len(views_by_name) != len(views) or set(views_by_name) & set(by_name):
        return fail("target has duplicate realization identities")
    source_index = {node.id: index for index, node in enumerate(request.nodes)}
    boundaries = dict(request.input_storages)
    visited: set[tuple] = set()
    selected_bundles: dict[tuple, tuple[str, tuple[str, ...]]] = {}

    def visit(choice: Choice, required_layout: str = "") -> str:
        metadata = program.symbols.get(choice.symbol)
        if metadata is None:
            return "selected symbol has no generated metadata"
        kind = metadata["kind"]
        source_id = metadata.get("source_node")
        if source_id is None or source_id not in source_index:
            return "selected symbol lacks a source correspondence"
        actual_layout = (
            request.boundary_layout(source_id) if kind in {"input", "constant"}
            else metadata["descriptor"]["physical_layout"] if kind == "view"
            else metadata["descriptor"].get("output_layout", "") if kind == "instruction"
            else ""
        )
        if required_layout and actual_layout != required_layout:
            return "selected physical layout differs from its required representation"
        # A required layout propagates to operands, so a choice reached again
        # under a different requirement must be checked again.
        if (choice.key(), required_layout) in visited:
            return ""
        visited.add((choice.key(), required_layout))
        source = request.node(source_id)
        if choice.eclass != exploration.class_by_node[source_index[source_id]]:
            return "selected symbol is in the wrong semantic e-class"
        if kind == "view":
            recorded = metadata["descriptor"]
            view = views_by_name.get(recorded.get("name"))
            if view is None or view.record() != recorded:
                return "selected view differs from the target descriptor"
            sources = tuple(request.node(child) for child in source.inputs)
            if not sources or not view.accepts(source, sources):
                return "selected view does not preserve its declared layout"
            if len(choice.children) != len(sources) or choice.storage != view.storage:
                return "selected view has wrong operands or storage"
            if any(
                child.storage != view.storage or child.eclass != exploration.class_by_node[source_index[source_id]]
                for child, source_id in zip(choice.children, source.inputs, strict=True)
            ):
                return "selected view changed its source value or physical storage"
            # The view's offset and element order hold only for the source
            # geometry the target qualified, so a padded or otherwise
            # differently sized source realisation is not the same bits.
            expected_extents = (
                view.source_extents if isinstance(view, ConcatViewDescriptor)
                else (view.source_extent,) if view.kind == "slice"
                else (view.extent,)
            )
            if tuple(_realized_extent(program, child) for child in choice.children) != expected_extents:
                return "selected view source extent differs from the target descriptor"
        elif kind == "instruction":
            recorded = metadata["descriptor"]
            descriptor = by_name.get(recorded["name"])
            if descriptor is None or descriptor.record() != recorded:
                return "selected instruction differs from the target descriptor"
            if descriptor.bundle_name:
                if (
                    descriptor.bundle_name not in bundles
                    or choice.bundle_result_index != descriptor.bundle_result_index
                    or choice.bundle_result_count != descriptor.bundle_result_count
                    or choice.bundle_instance != metadata.get("bundle_instance")
                ):
                    return "selected multi-result instruction differs from its descriptor"
                instance = choice.physical_instance()
                assert instance is not None
                signature = (descriptor.bundle_name, source.inputs)
                previous = selected_bundles.setdefault(instance, signature)
                if previous != signature:
                    return "shared instruction projections have different semantic operands"
            elif choice.bundle_instance or "bundle_instance" in metadata:
                return "single-result instruction carries a multi-result instance"
            if request.boundary_layouts and len(descriptor.input_layouts) != len(choice.children):
                return "selected instruction lacks physical operand layouts"
            realization = metadata.get("realization")
            if realization == "value_preserving_copy_v1":
                if not descriptor.value_preserving_copy or descriptor.computation != "identity":
                    return "selected physical copy lacks a value-preserving descriptor"
                if len(choice.children) != 1 or choice.children[0].eclass != choice.eclass:
                    return "selected physical copy changed its semantic value"
                if choice.storage != descriptor.output_storage or (
                    choice.children[0].storage != descriptor.input_storages[0]
                ):
                    return "selected physical copy uses an undeclared storage transition"
                if (
                    source.type.dtype != descriptor.output_dtype
                    or (source.type.dtype != descriptor.input_dtypes[0])
                    or source.type.numerical_policy != descriptor.numerical_policy
                    or (source.type.numerical_policy != descriptor.input_numerical_policies[0])
                ):
                    return "selected physical copy changes tensor type or numerical policy"
                if (
                    len(source.type.shape) not in descriptor.ranks
                    or (descriptor.input_ranks and len(source.type.shape) != descriptor.input_ranks[0])
                    or not all(bound.accepts(source.type.shape) for bound in descriptor.output_axis_bounds)
                    or (
                        descriptor.input_axis_bounds
                        and not all(bound.accepts(source.type.shape) for bound in descriptor.input_axis_bounds[0])
                    )
                ):
                    return "selected physical copy violates its shape precondition"
            elif realization is not None:
                return "selected instruction has an unknown realization rule"
            else:
                if source.effect != "pure" or source.op != descriptor.computation:
                    return "selected instruction does not implement the source operation"
                if source.type.dtype != descriptor.output_dtype or (
                    source.type.numerical_policy != descriptor.numerical_policy
                ):
                    return "selected instruction changes output numerical policy"
                if len(source.type.shape) not in descriptor.ranks or source.index_maps != descriptor.index_maps:
                    return "selected instruction has incompatible rank or index maps"
                operand_shapes = tuple(request.node(child_id).type.shape for child_id in source.inputs)
                if descriptor.shape_contract == "equal":
                    if any(shape != source.type.shape for shape in operand_shapes):
                        return "selected instruction violates equal-shape contract"
                else:
                    shapes = {
                        "out": source.type.shape,
                        **{f"in{index}": shape for index, shape in enumerate(operand_shapes)},
                    }
                    if any(
                        shapes[item.lhs_port][item.lhs_axis] != shapes[item.rhs_port][item.rhs_axis]
                        for item in descriptor.shape_equalities
                    ):
                        return "selected instruction violates a dimension relation"
                if not all(bound.accepts(source.type.shape) for bound in descriptor.output_axis_bounds):
                    return "selected instruction violates a shape precondition"
                source_attrs = dict(source.attrs)
                fixed_attrs = dict(descriptor.required_attrs)
                parameter_attrs = {bound.name: bound for bound in descriptor.parameter_attrs}
                if set(source_attrs) != set(fixed_attrs) | set(parameter_attrs):
                    return "selected instruction differs in semantic attributes"
                if any(
                    type(source_attrs[key]) is not type(value) or source_attrs[key] != value
                    for key, value in fixed_attrs.items()
                ):
                    return "selected instruction differs in semantic attributes"
                if any(
                    type(source_attrs[key]) is not int or not bound.minimum <= source_attrs[key] <= bound.maximum
                    for key, bound in parameter_attrs.items()
                ):
                    return "selected instruction violates a computational attribute bound"
                if descriptor.parameter_attrs:
                    selected_attrs = metadata.get("instantiated_attrs")
                    if type(selected_attrs) is not dict or set(selected_attrs) != set(source_attrs) or any(
                        type(selected_attrs[key]) is not type(value) or selected_attrs[key] != value
                        for key, value in source_attrs.items()
                    ):
                        return "selected instruction has a changed attribute substitution"
                if not descriptor.parameter_attrs and "instantiated_attrs" in metadata:
                    return "selected instruction has an undeclared attribute substitution"
                if len(choice.children) != len(source.inputs) or choice.storage != descriptor.output_storage:
                    return "selected instruction has wrong operands or output storage"
                for index, (child, child_id) in enumerate(zip(choice.children, source.inputs)):
                    typed = request.node(child_id).type
                    if child.eclass != exploration.class_by_node[source_index[child_id]]:
                        return "selected instruction operand changed semantic value"
                    if child.storage != descriptor.input_storages[index]:
                        return "selected instruction operand is in the wrong storage"
                    if typed.dtype != descriptor.input_dtypes[index] or not descriptor.accepts_input_policy(
                        index, typed.numerical_policy
                    ):
                        return "selected instruction changes operand numerical policy"
                    if descriptor.input_ranks and len(typed.shape) != descriptor.input_ranks[index]:
                        return "selected instruction operand rank differs"
                    if descriptor.input_axis_bounds and not all(
                        bound.accepts(typed.shape) for bound in descriptor.input_axis_bounds[index]
                    ):
                        return "selected instruction violates an input axis precondition"
        elif kind == "input":
            if choice.children or choice.storage != boundaries.get(source_id):
                return "selected input lacks its declared boundary representation"
            if type(metadata.get("extent")) is not int or metadata["extent"] != request.physical_extent(source_id):
                return "selected input physical extent differs from its boundary"
            expected_preservation = dict(source.attrs).get("input_retention", "preserve") == "preserve"
            if type(metadata.get("preserve_input")) is not bool or (
                metadata["preserve_input"] != expected_preservation
            ):
                return "selected input retention differs from the source contract"
        elif kind == "constant":
            if choice.children or choice.storage != "external":
                return "selected constant has no legal boundary representation"
            if type(metadata.get("extent")) is not int or metadata["extent"] != request.physical_extent(source_id):
                return "selected constant physical extent differs from its boundary"
        else:
            return "pure semantic node was selected without a target realization"
        for index, child in enumerate(choice.children):
            child_layout = (
                metadata["descriptor"]["physical_layout"] if kind == "view"
                else descriptor.input_layouts[index] if (
                    kind == "instruction" and descriptor.input_layouts
                    and (request.boundary_layouts or required_layout)
                )
                else ""
            )
            problem = visit(child, child_layout)
            if problem:
                return problem
        return ""

    for index, choice in enumerate(candidate.outputs):
        if choice.eclass != exploration.roots[index] or choice.storage != request.output_storages[index]:
            return fail("selected root differs from an ordered source output")
        problem = visit(choice, request.output_layouts[index] if request.output_layouts else "")
        if problem:
            return fail(problem)
    checked, reason = check_assignment(
        graph,
        allocation.order,
        allocation.addresses,
        banks,
        fixed_inputs=fixed_inputs,
        reservations=reservations,
        fixed_outputs=fixed_outputs,
    )
    if not checked:
        return fail(f"selected physical assignment failed independent replay: {reason}")
    expected_schedule = instruction_schedule(graph, allocation.order)
    if allocation.issue_times != tuple((value_id, expected_schedule[value_id][0]) for value_id in allocation.order):
        return fail("selected physical schedule differs from the allocated live ranges")
    timed, reason = check_timing(graph, allocation.order, allocation.issue_times)
    if not timed:
        return fail(f"selected physical schedule failed independent replay: {reason}")
    evidence = {
        "source": request.digest(),
        "candidate": candidate.digest(),
        "target": [descriptor.record() for descriptor in descriptors],
        "views": [view.record() for view in views],
        "banks": [bank.record() for bank in banks],
        "reservations": [vars(reservation) for reservation in reservations],
        "order": list(allocation.order),
        "issue_times": list(allocation.issue_times),
        "addresses": sorted(allocation.addresses.items()),
        "fixed_inputs": sorted((fixed_inputs or {}).items()),
        "fixed_outputs": list(fixed_outputs) if fixed_outputs is not None else None,
    }
    fingerprint = hashlib.sha256(json.dumps(evidence, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return SelectionCheck(True, fingerprint)
