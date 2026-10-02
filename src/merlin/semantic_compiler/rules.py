"""Generate typed computational instruction rules from a selected descriptor.

The selector has no target-name or opcode cases. A descriptor declares its
computational operation and valid signatures; rules are specialized to a
request's verified static types before being handed to the general e-graph.
Physical slot/address choices remain symbolic after selection.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from typing import Any

from .model import KernelRequest, SemanticNode


def _digest(value: object) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()[:24]


@dataclass(frozen=True)
class AddressConstraint:
    """A restricted hardware validity/address-map relation over instruction ports.

    Ports are `out` or `inN`. No Python expression or eval is admitted.
    `eq_offset` means lhs = rhs + value; `aligned` means lhs mod value = 0.
    """

    kind: str
    lhs: str
    rhs: str = ""
    value: int = 0

    def __post_init__(self) -> None:
        if self.kind not in {"eq_offset", "aligned"}:
            raise ValueError("unknown address constraint kind")
        if self.kind == "eq_offset" and not self.rhs:
            raise ValueError("eq_offset needs a right-hand port")
        if self.kind == "aligned" and (self.rhs or self.value <= 0):
            raise ValueError("aligned needs a positive modulus and no right-hand port")

    def record(self) -> dict[str, Any]:
        return {"kind": self.kind, "lhs": self.lhs, "rhs": self.rhs, "value": self.value}


@dataclass(frozen=True)
class InstructionDescriptor:
    name: str
    computation: str
    input_storages: tuple[str, ...]
    output_storage: str
    output_dtype: str
    numerical_policy: str
    ranks: tuple[int, ...]
    required_attrs: tuple[tuple[str, str | int | float | bool], ...] = ()
    # Physical width is expressed in named storage units, never inferred from bytes.
    extent: int = 1
    validity: tuple[AddressConstraint, ...] = ()
    input_dtypes: tuple[str, ...] = ()
    input_numerical_policies: tuple[str, ...] = ()
    input_ranks: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not all((self.name, self.computation, self.output_storage, self.output_dtype, self.numerical_policy)):
            raise ValueError("instruction descriptor needs semantic, storage and numerical identity")
        if not self.ranks or any(rank <= 0 for rank in self.ranks) or self.extent <= 0:
            raise ValueError("instruction ranks and extent must be positive")
        if len({key for key, _ in self.required_attrs}) != len(self.required_attrs):
            raise ValueError("duplicate required attribute")
        arity = len(self.input_storages)
        if any(len(signature) not in {0, arity} for signature in (
            self.input_dtypes, self.input_numerical_policies, self.input_ranks,
        )):
            raise ValueError("input signature must match instruction arity")
        if any(rank <= 0 for rank in self.input_ranks):
            raise ValueError("input ranks must be positive")
        ports = {"out", *(f"in{index}" for index in range(len(self.input_storages)))}
        for condition in self.validity:
            if condition.lhs not in ports or (condition.rhs and condition.rhs not in ports):
                raise ValueError("address constraint refers to an undeclared instruction port")

    def accepts(self, node: SemanticNode, inputs: tuple[SemanticNode, ...]) -> bool:
        return (
            node.effect == "pure"
            and node.op == self.computation
            and len(inputs) == len(self.input_storages)
            and node.type.dtype == self.output_dtype
            and node.type.numerical_policy == self.numerical_policy
            and len(node.type.shape) in self.ranks
            and all(dict(node.attrs).get(key) == value for key, value in self.required_attrs)
            and (not self.input_dtypes or all(n.type.dtype == dtype for n, dtype in zip(inputs, self.input_dtypes)))
            and (
                not self.input_numerical_policies
                or all(n.type.numerical_policy == policy for n, policy in zip(inputs, self.input_numerical_policies))
            )
            and (not self.input_ranks or all(len(n.type.shape) == rank for n, rank in zip(inputs, self.input_ranks)))
        )

    def record(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "computation": self.computation,
            "input_storages": list(self.input_storages),
            "output_storage": self.output_storage,
            "output_dtype": self.output_dtype,
            "numerical_policy": self.numerical_policy,
            "ranks": list(self.ranks),
            "required_attrs": dict(self.required_attrs),
            "extent": self.extent,
            "validity": [condition.record() for condition in self.validity],
            "input_dtypes": list(self.input_dtypes),
            "input_numerical_policies": list(self.input_numerical_policies),
            "input_ranks": list(self.input_ranks),
        }


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


def generate_rules(request: KernelRequest, descriptors: tuple[InstructionDescriptor, ...]) -> RuleProgram:
    """Make exact typed rules for each eligible source operation.

    The descriptor's computational signature, attrs, rank and policy decide
    eligibility. Generating a rule does not decide any physical binding.
    """
    symbols: dict[str, dict[str, Any]] = {}
    source_symbol: dict[str, str] = {}
    index: dict[str, int] = {}
    nodes: list[dict[str, Any]] = []
    rules: list[Rule] = []
    for node in request.nodes:
        operand_types = [request.node(child).type.record() for child in node.inputs]
        if node.effect == "pure":
            symbol = "s_" + _digest((node.semantic_key(), operand_types))
            symbols[symbol] = {"kind": "semantic", "op": node.op, "type": node.type.record()}
        else:
            symbol = "b_" + _digest(node.record())
            symbols[symbol] = {"kind": node.effect, "source_node": node.id, "type": node.type.record()}
        source_symbol[node.id] = symbol
        index[node.id] = len(nodes)
        nodes.append({"symbol": symbol, "children": [index[child] for child in node.inputs]})
        if node.effect != "pure":
            continue
        lhs = f"({symbol} {' '.join(f'?a{i}' for i in range(len(node.inputs)))})" if node.inputs else symbol
        for descriptor in descriptors:
            if not descriptor.accepts(node, tuple(request.node(child) for child in node.inputs)):
                continue
            descriptor_digest = _digest(descriptor.record())
            instruction_symbol = "i_" + _digest((descriptor_digest, node.type.record(), operand_types))
            symbols[instruction_symbol] = {
                "kind": "instruction",
                "descriptor": descriptor.record(),
                "source_node": node.id,
                "type": node.type.record(),
            }
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
