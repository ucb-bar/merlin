"""Replay a selected native graph against source semantics and target facts.

This checker is deliberately separate from rule generation and Z3 formula
construction. It validates only the declared finite semantic/placement model;
it does not certify an Atlas instruction stream or numerical hardware model.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from .allocate import (
    AllocationResult,
    CandidateGraph,
    StorageBank,
    check_assignment,
    instruction_schedule,
    lower_candidate,
)
from .egg_bridge import Exploration
from .extract import Candidate, Choice
from .model import KernelRequest
from .rules import InstructionDescriptor, RuleProgram


@dataclass(frozen=True)
class SelectionCheck:
    valid: bool
    fingerprint: str
    reason: str = ""


def check_timing(
    graph: CandidateGraph, order: tuple[int, ...], issue_times: tuple[tuple[int, int], ...]
) -> tuple[bool, str]:
    """Replay each read and completion event from the selected instruction graph."""
    if len(issue_times) != len(order) or tuple(value_id for value_id, _ in issue_times) != order:
        return False, "physical schedule omits, duplicates or reorders an instruction"
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
            child = graph.value(child_id)
            if child.kind == "instruction" and (child_id not in completed or completed[child_id] > issue + read_offset):
                return False, "input read precedes producer completion"
        previous_completion = issue + value.completion_offset
        completed[value_id] = previous_completion
    return True, ""


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
    fixed_inputs: dict[str, int] | None = None,
) -> SelectionCheck:
    def fail(reason: str) -> SelectionCheck:
        return SelectionCheck(False, "", reason)

    if allocation.status != "feasible" or len(candidate.outputs) != len(request.outputs):
        return fail("selection lacks a feasible assignment or ordered roots")
    if graph != lower_candidate(candidate, program):
        return fail("selected instruction graph changed after extraction")
    by_name = {descriptor.name: descriptor for descriptor in descriptors}
    if len(by_name) != len(descriptors):
        return fail("target has duplicate instruction identities")
    source_index = {node.id: index for index, node in enumerate(request.nodes)}
    boundaries = dict(request.input_storages)
    visited: set[tuple] = set()

    def visit(choice: Choice) -> str:
        if choice.key() in visited:
            return ""
        visited.add(choice.key())
        metadata = program.symbols.get(choice.symbol)
        if metadata is None:
            return "selected symbol has no generated metadata"
        kind = metadata["kind"]
        source_id = metadata.get("source_node")
        if source_id is None or source_id not in source_index:
            return "selected symbol lacks a source correspondence"
        source = request.node(source_id)
        if choice.eclass != exploration.class_by_node[source_index[source_id]]:
            return "selected symbol is in the wrong semantic e-class"
        if kind == "instruction":
            recorded = metadata["descriptor"]
            descriptor = by_name.get(recorded["name"])
            if descriptor is None or descriptor.record() != recorded:
                return "selected instruction differs from the target descriptor"
            if source.effect != "pure" or source.op != descriptor.computation:
                return "selected instruction does not implement the source operation"
            if source.type.dtype != descriptor.output_dtype or (
                source.type.numerical_policy != descriptor.numerical_policy
            ):
                return "selected instruction changes output numerical policy"
            if len(source.type.shape) not in descriptor.ranks or source.index_maps != descriptor.index_maps:
                return "selected instruction has incompatible rank or index maps"
            if not all(bound.accepts(source.type.shape) for bound in descriptor.output_axis_bounds):
                return "selected instruction violates a shape precondition"
            if dict(source.attrs) != dict(descriptor.required_attrs):
                return "selected instruction differs in semantic attributes"
            if len(choice.children) != len(source.inputs) or choice.storage != descriptor.output_storage:
                return "selected instruction has wrong operands or output storage"
            for index, (child, child_id) in enumerate(zip(choice.children, source.inputs)):
                typed = request.node(child_id).type
                if child.eclass != exploration.class_by_node[source_index[child_id]]:
                    return "selected instruction operand changed semantic value"
                if child.storage != descriptor.input_storages[index]:
                    return "selected instruction operand is in the wrong storage"
                if typed.dtype != descriptor.input_dtypes[index] or (
                    typed.numerical_policy != descriptor.input_numerical_policies[index]
                ):
                    return "selected instruction changes operand numerical policy"
                if descriptor.input_ranks and len(typed.shape) != descriptor.input_ranks[index]:
                    return "selected instruction operand rank differs"
        elif kind == "input":
            if choice.children or choice.storage != boundaries.get(source_id):
                return "selected input lacks its declared boundary representation"
        elif kind == "constant":
            if choice.children or choice.storage != "external":
                return "selected constant has no legal boundary representation"
        else:
            return "pure semantic node was selected without a target realization"
        for child in choice.children:
            problem = visit(child)
            if problem:
                return problem
        return ""

    for index, choice in enumerate(candidate.outputs):
        if choice.eclass != exploration.roots[index] or choice.storage != request.output_storages[index]:
            return fail("selected root differs from an ordered source output")
        problem = visit(choice)
        if problem:
            return fail(problem)
    checked, reason = check_assignment(graph, allocation.order, allocation.addresses, banks, fixed_inputs=fixed_inputs)
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
        "banks": [bank.record() for bank in banks],
        "order": list(allocation.order),
        "issue_times": list(allocation.issue_times),
        "addresses": sorted(allocation.addresses.items()),
    }
    fingerprint = hashlib.sha256(json.dumps(evidence, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return SelectionCheck(True, fingerprint)
