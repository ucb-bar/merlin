"""Bounded known-bit value cones with exact original conditional boundaries.

A boundary value is supplied independently for each observation. It is never
a memory read, reached state, clock event or observed opaque implementation.
Complete source membership and unknowns survive conditional composition.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

from merlin.common.jsonio import canonical_json
from merlin.targetgen.contract.mlir_source_admission import admit_mlir_source

from .hw_combinational import (
    EvaluationLimits,
    PreparedCombinationalObservation,
    ScalarPort,
    _Expression,
    _expression,
    _width,
)
from .hw_graph import parse_generic_hw
from .hw_observations import _module_name, _name
from .hw_value_bindings import OriginalValueSelection, ValueBindingLimits, hierarchical_value_bindings

SCHEMA = "merlin.hw_conditional_value_cones.v1"
_BOUNDARIES = {"state_result", "memory_read_result", "opaque_instance_result"}


@dataclass(frozen=True)
class ConditionalValueCut:
    """One exact original producer result; its known bits are conditional."""

    selection: OriginalValueSelection
    boundary: str

    def __post_init__(self):
        if (
            type(self.selection) is not OriginalValueSelection
            or self.selection.kind != "operation_result"
            or type(self.boundary) is not str
            or self.boundary not in _BOUNDARIES
        ):
            raise ValueError("Conditional cuts require exact original result identities and genuine boundary kinds.")


@dataclass(frozen=True)
class ConditionalConeInput:
    port: ScalarPort
    selection: OriginalValueSelection
    boundary: str


@dataclass(frozen=True)
class PreparedConditionalValueCones:
    """Prepared source expressions, with no temporal or admission authority."""

    inputs: tuple[ConditionalConeInput, ...]
    selections: tuple[OriginalValueSelection, ...]
    cuts: tuple[ConditionalValueCut, ...]
    expression: PreparedCombinationalObservation
    _source_record: bytes

    def evaluate(self, cases):
        """Observe complete outputs under every explicitly supplied known-bit input."""
        return self.expression.evaluate(cases)

    def record(self):
        """Return fresh metadata; this neither reopens source nor issues authority."""
        return {
            "schema": SCHEMA,
            "source": json.loads(self._source_record),
            "inputs": [asdict(row) for row in self.inputs],
            "selections": [asdict(row) for row in self.selections],
            "conditional_cuts": [asdict(row) for row in self.cuts],
            "outputs": [asdict(row) for row in self.expression.outputs],
            "evaluation_limits": asdict(self.expression.limits),
            "per_case_bit_work": self.expression.per_case_bit_work,
            "unknowns": [
                "definedness_and_four_state_values",
                "initialization_reset_clock_events_and_state_reachability",
                "memory_history_collision_and_opaque_implementation",
                "source_effect_execution_and_completion",
                "source_sdk_runtime_and_physical_correspondence",
                "semantic_roles_allocation_capacity_and_complete_costs",
            ],
            "scope": "conditional source known-bit expressions only",
            "admission_authority": False,
        }


def _identity(source_sha256, frame, original):
    return OriginalValueSelection(
        source_sha256,
        tuple(frame["path"]),
        frame["module"],
        original["kind"],
        original["ordinal"],
        original.get("slot", 0),
        original["type"],
    )


def prepare_conditional_value_cones(
    text: str,
    *,
    root: str,
    selections: tuple[OriginalValueSelection, ...],
    cuts: tuple[ConditionalValueCut, ...],
    binding_limits: ValueBindingLimits,
    evaluation_limits: EvaluationLimits,
) -> PreparedConditionalValueCones:
    """Freshly bind source and compose selected cones with an exact cut roster.

    Whole-source lexical and hierarchy bounds precede parsing/expansion. The
    structural reader discovers cuts; declarations cannot turn arbitrary logic
    into inputs. Only integer state, read and opaque results may be supplied.
    Unselected operation/effect membership remains UNKNOWN, never discarded.
    """
    if (
        type(binding_limits) is not ValueBindingLimits
        or type(evaluation_limits) is not EvaluationLimits
        or type(text) is not str
        or len(text) > min(binding_limits.source_bytes, evaluation_limits.source_bytes)
        or type(root) is not str
        or not root
        or type(selections) is not tuple
        or not selections
        or len(selections) > binding_limits.selections
        or any(type(row) is not OriginalValueSelection for row in selections)
        or len(set(selections)) != len(selections)
        or type(cuts) is not tuple
        or len(cuts) > binding_limits.nodes
        or any(type(row) is not ConditionalValueCut for row in cuts)
        or len({row.selection for row in cuts}) != len(cuts)
    ):
        raise ValueError("Conditional cones require bounded exact source, selection and cut rosters.")
    admit_mlir_source(
        text,
        max_source_bytes=min(binding_limits.source_bytes, evaluation_limits.source_bytes),
        max_nesting=64,
        max_integer_bits=max(64, binding_limits.scalar_bits),
        allow_dense=False,
        allow_dense_resource=False,
    )
    source_sha256 = hashlib.sha256(text.encode()).hexdigest()
    if any(row.source_sha256 != source_sha256 for row in (*selections, *(cut.selection for cut in cuts))):
        raise ValueError("Conditional source identity differs from its selections or cuts.")
    parsed = parse_generic_hw(text, reject_dense_literals=True)
    graph = hierarchical_value_bindings(parsed, root=root, selections=selections, limits=binding_limits)
    graph.update(source_sha256=source_sha256, source_byte_binding="exact supplied source bytes")
    frames = graph["frames"]
    nodes = graph["expressions"]
    if len(nodes) + len(selections) > evaluation_limits.nodes:
        raise ValueError("Conditional source cone exceeds its complete evaluation node budget.")
    work = graph["cost"]["bit_work"]
    for selected in graph["selections"]:
        width = nodes[selected["value"]]["width"]
        if type(width) is not int or not 0 < width <= evaluation_limits.scalar_bits:
            raise ValueError("Conditional outputs require bounded signless integer values.")
        work += width
    if work > evaluation_limits.bit_work:
        raise ValueError("Conditional source cone exceeds its per-case bit-work budget.")
    declarations = {cut.selection: cut.boundary for cut in cuts}
    reached, roots = {}, []
    for row in nodes:
        width = row["width"]
        if type(width) is not int or not 0 < width <= evaluation_limits.scalar_bits:
            raise ValueError("Conditional source cone encounters noninteger or oversized values.")
        kind = row["kind"]
        if kind in _BOUNDARIES:
            identity = _identity(source_sha256, frames[row["frame"]], row["original_value"])
            if declarations.get(identity) != kind:
                raise ValueError("Conditional source boundary lacks its exact declared cut.")
            reached[identity] = kind
        elif kind not in {"root_input", "instance_input_binding", "instance_output_binding", "combinational"}:
            raise ValueError("Conditional source cone encounters unsupported reachable logic.")
        if kind == "root_input" or kind in _BOUNDARIES:
            identity = _identity(source_sha256, frames[row["frame"]], row["original_value"])
            roots.append(ConditionalConeInput(ScalarPort("input_" + str(len(roots)), width), identity, kind))
    if reached != declarations:
        raise ValueError("Conditional cut roster has extra, stale or non-boundary declarations.")

    modules = {_module_name(op): op for op in parsed.walk() if _name(op) in {"hw.module", "hw.module.extern"}}
    operations = {
        name: tuple(module.regions[0].block.ops) for name, module in modules.items() if _name(module) == "hw.module"
    }
    # Structural node IDs are discovery order, not evaluator order. Recheck the
    # live primitive and compose dependencies before their uses without parsing
    # a saved graph or manufacturing a module with substituted source operands.
    indices, widths, expressions, active = {}, {}, [], set()
    root_ids = [row["id"] for row in nodes if row["kind"] == "root_input" or row["kind"] in _BOUNDARIES]
    for index, node_id in enumerate(root_ids):
        indices[node_id] = index
        widths[node_id] = nodes[node_id]["width"]
    for selected in graph["selections"]:
        stack = [(selected["value"], False)]
        while stack:
            node_id, ready = stack.pop()
            if node_id in indices:
                continue
            row = nodes[node_id]
            operands = row["operands"]
            if not ready:
                if node_id in active:
                    raise ValueError("Conditional source cone contains a cyclic expression.")
                active.add(node_id)
                stack.append((node_id, True))
                stack.extend((operand, False) for operand in reversed(operands))
                continue
            active.remove(node_id)
            if row["kind"] in {"instance_input_binding", "instance_output_binding"}:
                if len(operands) != 1 or widths[operands[0]] != row["width"]:
                    raise ValueError("Conditional hierarchical value binding changes its original type.")
                indices[node_id] = indices[operands[0]]
                widths[node_id] = row["width"]
                continue
            op = operations[frames[row["frame"]]["module"]][row["original_value"]["ordinal"]]
            if op.regions or len(op.results) != 1 or _width(op.results[0], evaluation_limits) != row["width"]:
                raise ValueError("Conditional primitive differs from its original result roster.")
            operand_widths = [_width(value, evaluation_limits) for value in op.operands]
            kind, parameter = _expression(op, operand_widths, row["width"], conditional_logic=True)
            if (
                operand_widths != [widths[index] for index in operands]
                or kind != row["expression"]
                or parameter != row["parameter"]
            ):
                raise ValueError("Conditional primitive differs from its original typed dependencies.")
            expressions.append(_Expression(kind, row["width"], tuple(indices[index] for index in operands), parameter))
            indices[node_id] = len(roots) + len(expressions) - 1
            widths[node_id] = row["width"]
    expression = PreparedCombinationalObservation(
        source_sha256,
        root,
        tuple(row.port for row in roots),
        tuple(
            ScalarPort("value_" + str(index), nodes[row["value"]]["width"])
            for index, row in enumerate(graph["selections"])
        ),
        tuple(expressions),
        tuple(indices[row["value"]] for row in graph["selections"]),
        work,
        evaluation_limits,
    )
    source_record = canonical_json(graph)
    if len(source_record) > binding_limits.metadata_bytes:
        raise ValueError("Conditional source record exceeds its complete metadata byte budget.")
    prepared = PreparedConditionalValueCones(tuple(roots), selections, cuts, expression, source_record)
    if len(canonical_json(prepared.record())) > binding_limits.metadata_bytes:
        raise ValueError("Conditional returned record exceeds its complete metadata byte budget.")
    return prepared
