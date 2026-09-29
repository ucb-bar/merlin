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

from .model import IndexMap, KernelRequest, SemanticNode


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

    @classmethod
    def from_record(cls, row: dict[str, Any]) -> AddressConstraint:
        if set(row) != {"kind", "lhs", "rhs", "value"}:
            raise ValueError("address constraint has missing or unknown fields")
        return cls(**row)


@dataclass(frozen=True)
class AxisBound:
    """A decidable per-axis computational precondition for a static tensor."""

    axis: int
    minimum: int = 1
    maximum: int | None = None
    multiple: int = 1

    def __post_init__(self) -> None:
        if any(type(value) is not int for value in (self.axis, self.minimum, self.multiple)):
            raise ValueError("axis bound fields must be integers")
        if self.maximum is not None and type(self.maximum) is not int:
            raise ValueError("axis bound maximum must be an integer")
        if self.axis < 0 or self.minimum <= 0 or self.multiple <= 0:
            raise ValueError("axis bound needs nonnegative axis and positive dimensions")
        if self.maximum is not None and self.maximum < self.minimum:
            raise ValueError("axis bound maximum is below minimum")

    def accepts(self, shape: tuple[int, ...]) -> bool:
        if self.axis >= len(shape):
            return False
        dimension = shape[self.axis]
        return (
            dimension >= self.minimum
            and (self.maximum is None or dimension <= self.maximum)
            and (dimension % self.multiple == 0)
        )

    def record(self) -> dict[str, int | None]:
        return {"axis": self.axis, "minimum": self.minimum, "maximum": self.maximum, "multiple": self.multiple}

    @classmethod
    def from_record(cls, row: dict[str, Any]) -> AxisBound:
        if set(row) != {"axis", "minimum", "maximum", "multiple"}:
            raise ValueError("axis bound has missing or unknown fields")
        return cls(**row)


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
    output_axis_bounds: tuple[AxisBound, ...] = ()
    index_maps: tuple[IndexMap, ...] = ()
    # Offsets use issued instruction cycles. A delayed operand read keeps its
    # physical source live; completion is also the conservative result-ready
    # event. Zero defaults are only suitable for synchronous descriptions.
    input_read_offsets: tuple[int, ...] = ()
    completion_offset: int = 0

    def __post_init__(self) -> None:
        if not all((self.name, self.computation, self.output_storage, self.output_dtype, self.numerical_policy)):
            raise ValueError("instruction descriptor needs semantic, storage and numerical identity")
        if not self.ranks or any(rank <= 0 for rank in self.ranks) or self.extent <= 0:
            raise ValueError("instruction ranks and extent must be positive")
        if len({key for key, _ in self.required_attrs}) != len(self.required_attrs):
            raise ValueError("duplicate required attribute")
        arity = len(self.input_storages)
        if len(self.input_dtypes) != arity or len(self.input_numerical_policies) != arity:
            raise ValueError("each instruction operand needs an explicit dtype and numerical policy")
        if len(self.input_ranks) not in {0, arity}:
            raise ValueError("input signature must match instruction arity")
        if any(not dtype or not policy for dtype, policy in zip(self.input_dtypes, self.input_numerical_policies)):
            raise ValueError("input dtype and numerical policy must be nonempty")
        if any(rank <= 0 for rank in self.input_ranks):
            raise ValueError("input ranks must be positive")
        if len({bound.axis for bound in self.output_axis_bounds}) != len(self.output_axis_bounds):
            raise ValueError("duplicate output axis bound")
        if self.index_maps and len(self.index_maps) != arity + 1:
            raise ValueError("instruction index maps need one map per operand and result")
        if self.input_read_offsets and len(self.input_read_offsets) != arity:
            raise ValueError("input read offsets must match instruction arity")
        if type(self.completion_offset) is not int or self.completion_offset < 0:
            raise ValueError("instruction completion offset must be nonnegative")
        if any(
            type(offset) is not int or offset < 0 or offset > self.completion_offset
            for offset in self.input_read_offsets
        ):
            raise ValueError("input read offset must precede instruction completion")
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
            and all(bound.accepts(node.type.shape) for bound in self.output_axis_bounds)
            and node.index_maps == self.index_maps
            and all(dict(node.attrs).get(key) == value for key, value in self.required_attrs)
            and all(n.type.dtype == dtype for n, dtype in zip(inputs, self.input_dtypes))
            and all(n.type.numerical_policy == policy for n, policy in zip(inputs, self.input_numerical_policies))
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
            "output_axis_bounds": [bound.record() for bound in self.output_axis_bounds],
            "index_maps": [index_map.record() for index_map in self.index_maps],
            "input_read_offsets": list(self.input_read_offsets),
            "completion_offset": self.completion_offset,
        }

    @classmethod
    def from_record(cls, row: dict[str, Any]) -> InstructionDescriptor:
        expected = {
            "name",
            "computation",
            "input_storages",
            "output_storage",
            "output_dtype",
            "numerical_policy",
            "ranks",
            "required_attrs",
            "extent",
            "validity",
            "input_dtypes",
            "input_numerical_policies",
            "input_ranks",
            "output_axis_bounds",
            "index_maps",
            "input_read_offsets",
            "completion_offset",
        }
        if set(row) != expected:
            raise ValueError("instruction descriptor has missing or unknown fields")
        return cls(
            name=row["name"],
            computation=row["computation"],
            input_storages=tuple(row["input_storages"]),
            output_storage=row["output_storage"],
            output_dtype=row["output_dtype"],
            numerical_policy=row["numerical_policy"],
            ranks=tuple(row["ranks"]),
            required_attrs=tuple(sorted(row["required_attrs"].items())),
            extent=row["extent"],
            validity=tuple(AddressConstraint.from_record(item) for item in row["validity"]),
            input_dtypes=tuple(row["input_dtypes"]),
            input_numerical_policies=tuple(row["input_numerical_policies"]),
            input_ranks=tuple(row["input_ranks"]),
            output_axis_bounds=tuple(AxisBound.from_record(item) for item in row["output_axis_bounds"]),
            index_maps=tuple(IndexMap.from_record(item) for item in row["index_maps"]),
            input_read_offsets=tuple(row["input_read_offsets"]),
            completion_offset=row["completion_offset"],
        )


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
            # The symbol identifies the full pure expression, including its
            # operands. A signature-only symbol lets a rule generated for one
            # source use fire on an unrelated use with different inputs.
            symbol = "s_" + _digest((node.semantic_key(), tuple(source_symbol[child] for child in node.inputs)))
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
            if not descriptor.accepts(node, tuple(request.node(child) for child in node.inputs)):
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
