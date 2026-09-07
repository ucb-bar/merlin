"""Verified target boundary for lowering a :class:`GlobalPlan` to executable dispatch.

The shared planner chooses regions and explicit representation transitions.  A target plugin owns
their concrete lowering, but it must return a receipt that accounts for every emitted node and maps
the logical program boundary to the executable boundary.  This keeps target opcodes out of the core
without allowing a plugin to attach a plan as inert metadata while executing the old program.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from .dispatch_program import DispatchProgram, verify_program
from .global_plan import GlobalPlan, verify_global_plan


def dispatch_digest(program: DispatchProgram) -> str:
    body = json.dumps(program.to_dict(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class EmittedComponent:
    """Emitted node indices owned by one selected region or representation transition."""

    plan_id: str
    node_indices: tuple[int, ...]
    # Logical buffer -> emitted buffer at this component's boundary. Internal temporaries are
    # deliberately private to its emitter; crossing values are part of the checked contract.
    inputs: tuple[tuple[str, str], ...] = ()
    outputs: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if not self.plan_id.strip():
            raise ValueError("an emitted component must name its plan id")
        if tuple(sorted(set(self.node_indices))) != self.node_indices:
            raise ValueError("emitted component node indices must be sorted and unique")
        for direction, mappings in (("input", self.inputs), ("output", self.outputs)):
            if any(not logical.strip() or not emitted.strip() for logical, emitted in mappings):
                raise ValueError(f"component {direction} mappings must name both buffers")
            if len({logical for logical, _ in mappings}) != len(mappings):
                raise ValueError(f"component repeats a logical {direction} buffer")

    def to_dict(self) -> dict[str, Any]:
        return {"plan_id": self.plan_id, "node_indices": list(self.node_indices),
                "inputs": dict(self.inputs), "outputs": dict(self.outputs)}


@dataclass(frozen=True)
class BoundaryMapping:
    """One logical model input/result and its emitted ABI buffer."""

    direction: str
    logical_buffer: str
    emitted_buffer: str

    def __post_init__(self) -> None:
        if self.direction not in ("input", "output"):
            raise ValueError("boundary mapping direction must be input or output")
        if not self.logical_buffer.strip() or not self.emitted_buffer.strip():
            raise ValueError("boundary mapping must name both buffers")

    def to_dict(self) -> dict[str, str]:
        return {"direction": self.direction, "logical_buffer": self.logical_buffer,
                "emitted_buffer": self.emitted_buffer}


@dataclass(frozen=True)
class GlobalPlanEmission:
    """Executable program plus a complete accounting receipt for its selected plan."""

    dispatch: DispatchProgram
    plan_digest: str
    logical_dispatch_digest: str
    regions: tuple[EmittedComponent, ...]
    transitions: tuple[EmittedComponent, ...]
    boundaries: tuple[BoundaryMapping, ...]
    provenance: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.plan_digest.strip() or not self.logical_dispatch_digest.strip():
            raise ValueError("global-plan emission must carry plan and logical-dispatch digests")
        if not self.provenance or any(not item.strip() for item in self.provenance):
            raise ValueError("global-plan emission requires non-empty provenance")

    def receipt(self) -> dict[str, Any]:
        return {
            "schema": "global_plan_emission_v2",
            "plan_digest": self.plan_digest,
            "logical_dispatch_digest": self.logical_dispatch_digest,
            "emitted_dispatch_digest": dispatch_digest(self.dispatch),
            "regions": [item.to_dict() for item in self.regions],
            "transitions": [item.to_dict() for item in self.transitions],
            "boundaries": [item.to_dict() for item in self.boundaries],
            "provenance": list(self.provenance),
        }


@runtime_checkable
class GlobalPlanEmitter(Protocol):
    """Target plugin seam; concrete encodings and fused kernels live behind this protocol."""

    def emit_global_plan(self, program: DispatchProgram,
                         plan: GlobalPlan) -> GlobalPlanEmission: ...


def _boundary_buffers(program: DispatchProgram, direction: str) -> set[str]:
    if direction == "output":
        return set(program.results)
    return {name for name, buffer in program.buffers.items() if buffer.kind == "arg"}


def _component_boundary(program: DispatchProgram, indices: tuple[int, ...]
                        ) -> tuple[set[str], set[str]]:
    inside = set(indices)
    produced = {buffer for index in inside for buffer in program.nodes[index].outputs}
    read = {buffer for index in inside for buffer in program.nodes[index].inputs}
    external_reads = set(program.results)
    external_reads.update(buffer for index, node in enumerate(program.nodes) if index not in inside
                          for buffer in node.inputs)
    return read - produced, produced & external_reads


def _verify_dataflow(program: DispatchProgram, plan: GlobalPlan,
                     emission: GlobalPlanEmission) -> list[str]:
    """Check component boundaries and every edge; counting owned nodes is insufficient.

    This proves wiring, not kernel arithmetic equivalence. A selected implementation still needs
    its independent semantic qualification. In particular, a self-reported component name cannot
    stand in for a proof that a fused kernel computes its declared operations.
    """
    problems: list[str] = []
    regions = {row.plan_id: row for row in emission.regions}
    transitions = {row.plan_id: row for row in emission.transitions}
    owners = {index: item.id for item in plan.selected for index in item.nodes}
    producers = {buffer: owners[index] for index, node in enumerate(program.nodes)
                 for buffer in node.outputs}
    boundary = {(row.direction, row.logical_buffer): row.emitted_buffer
                for row in emission.boundaries}

    for item in plan.selected:
        row = regions[item.id]
        expected_in, expected_out = _component_boundary(program, item.nodes)
        if set(dict(row.inputs)) != expected_in or set(dict(row.outputs)) != expected_out:
            problems.append(f"region {item.id!r} does not map its complete logical boundary")
        if {r.buffer for r in item.inputs} != expected_in \
                or {r.buffer for r in item.outputs} != expected_out:
            problems.append(f"region {item.id!r} representations differ from its graph boundary")

    for item in plan.transitions:
        row = transitions[item.id]
        if set(dict(row.inputs)) != {item.buffer} or set(dict(row.outputs)) != {item.buffer}:
            problems.append(f"transition {item.id!r} must map its logical value on both sides")

    for row in (*emission.regions, *emission.transitions):
        actual_in, actual_out = _component_boundary(emission.dispatch, row.node_indices)
        mapped_in, mapped_out = set(dict(row.inputs).values()), set(dict(row.outputs).values())
        if row.node_indices:
            if actual_in != mapped_in or actual_out != mapped_out:
                problems.append(f"component {row.plan_id!r} emitted boundary differs from its mapping")
        elif dict(row.inputs) != dict(row.outputs):
            problems.append(f"zero-node transition {row.plan_id!r} cannot change the value buffer")

    by_edge: dict[tuple[str, str | None, str | None], list] = {}
    for item in plan.transitions:
        by_edge.setdefault((item.buffer, item.producer, item.consumer), []).append(item)
    visited: set[tuple[str, str | None, str | None]] = set()
    selected = {item.id: item for item in plan.selected}

    def connect(buffer: str, producer: str | None, consumer: str | None) -> None:
        key = (buffer, producer, consumer)
        if key in visited:
            return
        visited.add(key)
        source = (dict(regions[producer].outputs).get(buffer) if producer is not None else
                  boundary.get(("input", buffer)))
        destination = (dict(regions[consumer].inputs).get(buffer) if consumer is not None else
                       boundary.get(("output", buffer)))
        matches = by_edge.get(key, [])
        if source is None or destination is None:
            problems.append(f"logical edge {key!r} has no emitted endpoint")
        elif len(matches) > 1:
            problems.append(f"logical edge {key!r} has duplicate transitions")
        elif matches:
            item = matches[0]
            row = transitions[item.id]
            if dict(row.inputs).get(buffer) != source or dict(row.outputs).get(buffer) != destination:
                problems.append(f"transition {item.id!r} bypasses its logical producer or consumer")
            if producer is not None and selected[producer].output_representation(buffer) != item.source:
                problems.append(f"transition {item.id!r} source encoding disagrees with producer")
            if consumer is not None and selected[consumer].input_representation(buffer) != item.destination:
                problems.append(f"transition {item.id!r} destination encoding disagrees with consumer")
        else:
            if source != destination:
                problems.append(f"logical edge {key!r} is disconnected in emitted dataflow")
            if producer is not None and consumer is not None \
                    and selected[producer].output_representation(buffer) \
                    != selected[consumer].input_representation(buffer):
                problems.append(f"logical edge {key!r} changes encoding without a transition")

    for item in plan.selected:
        logical_inputs, _ = _component_boundary(program, item.nodes)
        for buffer in sorted(logical_inputs):
            connect(buffer, producers.get(buffer), item.id)
    for buffer in program.results:
        connect(buffer, producers.get(buffer), None)
    for edge in by_edge.keys() - visited:
        problems.append(f"transition describes a nonexistent logical edge {edge!r}")
    return problems


def verify_global_plan_emission(program: DispatchProgram, plan: GlobalPlan,
                                emission: GlobalPlanEmission) -> list[str]:
    """Return every inconsistency between a logical plan and its executable lowering."""
    problems = list(verify_global_plan(program, plan))
    problems.extend(f"logical dispatch: {problem}" for problem in verify_program(program))
    problems.extend(f"emitted dispatch: {problem}" for problem in verify_program(emission.dispatch))
    if emission.plan_digest != plan.digest:
        problems.append("emission plan digest differs from the selected global plan")
    if emission.logical_dispatch_digest != dispatch_digest(program):
        problems.append("emission logical-dispatch digest differs from the planned program")

    selected = {alternative.id: alternative for alternative in plan.selected}
    transitions = {transition.id: transition for transition in plan.transitions}
    region_rows = {row.plan_id: row for row in emission.regions}
    transition_rows = {row.plan_id: row for row in emission.transitions}
    if len(region_rows) != len(emission.regions):
        problems.append("emission repeats a selected-region receipt")
    if len(transition_rows) != len(emission.transitions):
        problems.append("emission repeats a transition receipt")
    if set(region_rows) != set(selected):
        problems.append("emission region receipts do not exactly match selected alternatives")
    if set(transition_rows) != set(transitions):
        problems.append("emission transition receipts do not exactly match selected transitions")

    owners: dict[int, str] = {}
    for kind, rows in (("region", emission.regions), ("transition", emission.transitions)):
        for row in rows:
            if kind == "region" and not row.node_indices:
                problems.append(f"selected region {row.plan_id!r} emitted no executable node")
            transition = transitions.get(row.plan_id) if kind == "transition" else None
            if transition is not None and transition.materializes and not row.node_indices:
                problems.append(f"materializing transition {row.plan_id!r} emitted no node")
            for index in row.node_indices:
                if index < 0 or index >= len(emission.dispatch.nodes):
                    problems.append(
                        f"{kind} {row.plan_id!r} accounts for absent emitted node {index}")
                elif index in owners:
                    problems.append(
                        f"emitted node {index} is owned by both {owners[index]!r} and "
                        f"{row.plan_id!r}")
                else:
                    owners[index] = row.plan_id
    unowned = sorted(set(range(len(emission.dispatch.nodes))) - set(owners))
    if unowned:
        problems.append(f"emission leaves executable node(s) unaccounted: {unowned}")

    # DispatchProgram's general verifier is intentionally permissive for older runtime writers.
    # A proof-carrying emitter must additionally establish SSA ownership and explicit captures.
    for label, dispatch in (("logical", program), ("emitted", emission.dispatch)):
        defined = {key for key, buffer in dispatch.buffers.items() if buffer.kind == "arg"}
        for index, node in enumerate(dispatch.nodes):
            if node.regions and node.captures is None:
                problems.append(f"{label} node {index} has an unknown region capture set")
            if not set(node.captures or ()).issubset(node.inputs):
                problems.append(f"{label} node {index} omits captured inputs")
            if not set(node.inputs).issubset(defined):
                problems.append(f"{label} node {index} reads a value without an earlier definition")
            for output in node.outputs:
                if output not in dispatch.buffers or output in defined:
                    problems.append(f"{label} node {index} has an absent or multiply defined output")
                defined.add(output)

    for direction in ("input", "output"):
        rows = [row for row in emission.boundaries if row.direction == direction]
        logical = [row.logical_buffer for row in rows]
        emitted = [row.emitted_buffer for row in rows]
        expected_logical = _boundary_buffers(program, direction)
        expected_emitted = _boundary_buffers(emission.dispatch, direction)
        if len(logical) != len(set(logical)) or set(logical) != expected_logical:
            problems.append(f"emission does not map every logical {direction} exactly once")
        if len(emitted) != len(set(emitted)) or set(emitted) != expected_emitted:
            problems.append(f"emission does not map every emitted {direction} exactly once")
        for row in rows:
            logical_spec = program.buffers.get(row.logical_buffer)
            emitted_spec = emission.dispatch.buffers.get(row.emitted_buffer)
            if (logical_spec is not None and emitted_spec is not None
                    and (logical_spec.shape != emitted_spec.shape
                         or logical_spec.dtype != emitted_spec.dtype)):
                problems.append(
                    f"boundary {direction} {row.logical_buffer!r}->{row.emitted_buffer!r} "
                    "changes the external shape or dtype")
            if (direction == "input" and logical_spec is not None and emitted_spec is not None
                    and logical_spec.arg_index != emitted_spec.arg_index):
                problems.append(f"boundary input {row.logical_buffer!r} changes its ABI argument index")
    output_map = {row.logical_buffer: row.emitted_buffer for row in emission.boundaries
                  if row.direction == "output"}
    if [output_map.get(name) for name in program.results] != emission.dispatch.results:
        problems.append("emission changes model result order")
    # Only traverse indices and owner maps after their structural checks succeeded.
    if not problems:
        problems.extend(_verify_dataflow(program, plan, emission))
    return problems


def emit_global_plan(program: DispatchProgram, plan: GlobalPlan,
                     emitter: GlobalPlanEmitter) -> GlobalPlanEmission:
    """Invoke a target emitter and refuse any incomplete or inert accounting receipt."""
    emission = emitter.emit_global_plan(program, plan)
    if not isinstance(emission, GlobalPlanEmission):
        raise TypeError("global-plan emitter did not return GlobalPlanEmission")
    problems = verify_global_plan_emission(program, plan, emission)
    if problems:
        raise ValueError("invalid global-plan emission: " + "; ".join(problems))
    return emission
