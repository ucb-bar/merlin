"""Generate typed computational instruction rules from a selected descriptor.

The selector has no target-name or opcode cases. A descriptor declares its
computational operation and valid signatures; rules are specialized to a
request's verified static types before being handed to the general e-graph.
Physical slot/address choices remain symbolic after selection.
"""

from __future__ import annotations

from .view_descriptors import ViewDescriptor, ConcatViewDescriptor

from .instruction import AddressConstraint, AxisBound, AxisEquality, IntegerAttributeBound, InstructionDescriptor

import json
from dataclasses import dataclass
from hashlib import sha256
from math import prod
from typing import Any

from .model import IndexMap, KernelRequest, SemanticNode, TensorType


def _digest(value: object) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()[:24]


@dataclass(frozen=True)
class Rule:
    name: str
    lhs: str
    rhs: str
    source_node: str
    descriptor_name: str
    descriptor_digest: str

    def record(self) -> dict[str, str]:
        return vars(self).copy()


@dataclass(frozen=True)
class RuleProgram:
    nodes: tuple[dict[str, Any], ...]
    roots: tuple[int, ...]
    rewrites: tuple[Rule, ...]
    symbols: dict[str, dict[str, Any]]

    def record(self, *, iterations: int, node_limit: int) -> dict[str, Any]:
        return {
            "schema": "merlin.egg_request.v1",
            "nodes": list(self.nodes),
            "roots": list(self.roots),
            "rewrites": [rule.record() for rule in self.rewrites],
            "iterations": iterations,
            "node_limit": node_limit,
        }


def bundle_catalog(descriptors: tuple[InstructionDescriptor, ...]) -> dict[str, tuple[InstructionDescriptor, ...]]:
    """Validate complete multi-result signatures without target-specific rules."""
    groups: dict[str, dict[int, InstructionDescriptor]] = {}
    for descriptor in descriptors:
        if not descriptor.bundle_name:
            continue
        members = groups.setdefault(descriptor.bundle_name, {})
        if descriptor.bundle_result_index in members:
            raise ValueError("multi-result instruction has duplicate result index")
        members[descriptor.bundle_result_index] = descriptor
    complete: dict[str, tuple[InstructionDescriptor, ...]] = {}
    for name, members in groups.items():
        count = next(iter(members.values())).bundle_result_count
        if set(members) != set(range(count)) or any(
            item.bundle_result_count != count for item in members.values()
        ):
            raise ValueError("multi-result instruction descriptor is incomplete")
        ordered = tuple(members[index] for index in range(count))
        first = ordered[0]
        def common(item: InstructionDescriptor) -> tuple:
            return (
                item.input_storages, item.input_dtypes, item.input_numerical_policies,
                item.input_policy_alternatives, item.input_ranks, item.input_axis_bounds,
                item.input_layouts, item.input_read_offsets, item.completion_offset,
                item.index_maps[:-1] if item.index_maps else (),
            )
        if any(common(item) != common(first) for item in ordered[1:]):
            raise ValueError("multi-result instruction projections disagree on shared operands or timing")
        complete[name] = ordered
    return complete


def generate_rules(
    request: KernelRequest,
    descriptors: tuple[InstructionDescriptor, ...],
    views: tuple[ViewDescriptor | ConcatViewDescriptor, ...] = (),
) -> RuleProgram:
    """Make exact typed rules for each eligible source operation.

    The descriptor's computational signature, attrs, rank and policy decide
    eligibility. Generating a rule does not decide any physical binding.
    """
    bundles = bundle_catalog(descriptors)
    bundle_matches: dict[tuple[str, tuple[str, ...]], set[int]] = {}
    for node in request.nodes:
        if node.effect != "pure":
            continue
        inputs = tuple(request.node(child) for child in node.inputs)
        for descriptor in descriptors:
            if descriptor.bundle_name and descriptor.accepts(node, inputs):
                bundle_matches.setdefault((descriptor.bundle_name, node.inputs), set()).add(
                    descriptor.bundle_result_index
                )
    symbols: dict[str, dict[str, Any]] = {}
    source_symbol: dict[str, str] = {}
    index: dict[str, int] = {}
    nodes: list[dict[str, Any]] = []
    rules: list[Rule] = []
    for node in request.nodes:
        operand_types = [request.node(child).type.record() for child in node.inputs]
        if node.effect == "pure":
            # The symbol identifies the full pure expression, including its
            # operands. A signature-only symbol lets a rule generated for one
            # source use fire on an unrelated use with different inputs.
            symbol = "s_" + _digest((node.semantic_key(), tuple(source_symbol[child] for child in node.inputs)))
            symbols[symbol] = {"kind": "semantic", "op": node.op, "type": node.type.record()}
        else:
            symbol = "b_" + _digest(node.record())
            symbols[symbol] = {"kind": node.effect, "source_node": node.id, "type": node.type.record()}
            if node.effect == "input":
                symbols[symbol]["preserve_input"] = dict(node.attrs).get("input_retention", "preserve") == "preserve"
            if node.effect in {"input", "constant"}:
                symbols[symbol]["extent"] = request.physical_extent(node.id)
        source_symbol[node.id] = symbol
        index[node.id] = len(nodes)
        nodes.append({"symbol": symbol, "children": [index[child] for child in node.inputs]})
        lhs = f"({symbol} {' '.join(f'?a{i}' for i in range(len(node.inputs)))})" if node.inputs else symbol
        for descriptor in descriptors:
            if not descriptor.value_preserving_copy:
                continue
            source_attrs = dict(node.attrs)
            if any(key not in source_attrs or source_attrs[key] != value
                   for key, value in descriptor.source_required_attrs):
                continue
            copy_value = SemanticNode("copy_value", "identity", (node.id,), node.type)
            if not descriptor.accepts(copy_value, (node,)):
                continue
            descriptor_digest = _digest(descriptor.record())
            instruction_symbol = "i_" + _digest(("value_preserving_copy_v1", descriptor_digest, symbol))
            symbols.setdefault(
                instruction_symbol,
                {
                    "kind": "instruction",
                    "descriptor": descriptor.record(),
                    "source_node": node.id,
                    "type": node.type.record(),
                    "realization": "value_preserving_copy_v1",
                },
            )
            rules.append(
                Rule(
                    name=f"materialize_{descriptor.name}_{node.id}_{descriptor_digest}",
                    lhs=lhs,
                    rhs=f"({instruction_symbol} {lhs})",
                    source_node=node.id,
                    descriptor_name=descriptor.name,
                    descriptor_digest=descriptor_digest,
                )
            )
        if node.effect != "pure":
            continue
        sources = tuple(request.node(child) for child in node.inputs)
        for view in views:
            if not view.accepts(node, sources):
                continue
            descriptor_digest = _digest(view.record())
            view_symbol = "v_" + _digest((descriptor_digest, symbol))
            symbols.setdefault(
                view_symbol,
                {
                    "kind": "view",
                    "descriptor": view.record(),
                    "source_node": node.id,
                    "type": node.type.record(),
                },
            )
            arguments = " ".join(f"?a{i}" for i in range(len(sources)))
            rules.append(
                Rule(
                    name=f"view_{view.name}_{node.id}_{descriptor_digest}",
                    lhs=lhs,
                    rhs=f"({view_symbol} {arguments})",
                    source_node=node.id,
                    descriptor_name=view.name,
                    descriptor_digest=descriptor_digest,
                )
            )
        # This is a bit-preserving semantic identity, including the numerical
        # policy and every static dimension. Its realization can still require
        # a physical copy; extraction decides that from the requested storage.
        if node.op == "identity" and not node.attrs and not node.index_maps and len(node.inputs) == 1:
            child = request.node(node.inputs[0])
            if child.type == node.type:
                rules.append(
                    Rule(
                        name=f"structural_identity_v1_{node.id}",
                        lhs=lhs,
                        rhs="?a0",
                        source_node=node.id,
                        descriptor_name="<structural>",
                        descriptor_digest=_digest(("structural_identity_v1", node.type.record())),
                    )
                )
        for descriptor in descriptors:
            if descriptor.value_preserving_copy:
                # The source-independent rule above already realizes this
                # instruction, including when the source operation is identity.
                continue
            if not descriptor.accepts(node, tuple(request.node(child) for child in node.inputs)):
                continue
            if descriptor.bundle_name and bundle_matches.get((descriptor.bundle_name, node.inputs)) != set(
                range(descriptor.bundle_result_count)
            ):
                continue
            descriptor_digest = _digest(descriptor.record())
            # Identical pure source expressions may share one realization.
            # Distinct operand graphs have distinct semantic symbols above.
            instruction_symbol = "i_" + _digest((descriptor_digest, symbol, node.type.record(), operand_types))
            symbols.setdefault(
                instruction_symbol,
                {
                    "kind": "instruction",
                    "descriptor": descriptor.record(),
                    "source_node": node.id,
                    "type": node.type.record(),
                    **({"instantiated_attrs": dict(node.attrs)} if descriptor.parameter_attrs else {}),
                    **({
                        "bundle_instance": "m_" + _digest((
                            descriptor.bundle_name,
                            [item.record() for item in bundles[descriptor.bundle_name]],
                            tuple(source_symbol[child] for child in node.inputs),
                        )),
                        "bundle_result_index": descriptor.bundle_result_index,
                        "bundle_result_count": descriptor.bundle_result_count,
                    } if descriptor.bundle_name else {}),
                },
            )
            rhs = (
                f"({instruction_symbol} {' '.join(f'?a{i}' for i in range(len(node.inputs)))})"
                if node.inputs
                else instruction_symbol
            )
            rules.append(
                Rule(
                    name=f"select_{descriptor.name}_{node.id}_{descriptor_digest}",
                    lhs=lhs,
                    rhs=rhs,
                    source_node=node.id,
                    descriptor_name=descriptor.name,
                    descriptor_digest=descriptor_digest,
                )
            )
    return RuleProgram(
        nodes=tuple(nodes),
        roots=tuple(index[output] for output in request.outputs),
        rewrites=tuple(rules),
        symbols=symbols,
    )
