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
from math import prod
from typing import Any

from .model import IndexMap, KernelRequest, SemanticNode, TensorType


def _digest(value: object) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()[:24]


@dataclass(frozen=True)
class AddressConstraint:
    """A restricted hardware validity/address-map relation over instruction ports.

    Ports are `out` or `inN`. No Python expression or eval is admitted.
    `eq_offset` means lhs = rhs + value; `aligned` means lhs mod value = 0.
    `at_least` and `at_most` bound a port in its declared storage unit.
    """

    kind: str
    lhs: str
    rhs: str = ""
    value: int = 0

    def __post_init__(self) -> None:
        if self.kind not in {"eq_offset", "aligned", "at_least", "at_most"}:
            raise ValueError("unknown address constraint kind")
        if type(self.value) is not int:
            raise ValueError("address constraint value must be an integer")
        if self.kind == "eq_offset" and not self.rhs:
            raise ValueError("eq_offset needs a right-hand port")
        if self.kind == "aligned" and (self.rhs or self.value <= 0):
            raise ValueError("aligned needs a positive modulus and no right-hand port")
        if self.kind in {"at_least", "at_most"} and (self.rhs or self.value < 0):
            raise ValueError("address bound needs a nonnegative value and no right-hand port")

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
class AxisEquality:
    """A declared equality between dimensions of instruction ports."""

    lhs_port: str
    lhs_axis: int
    rhs_port: str
    rhs_axis: int

    def __post_init__(self) -> None:
        if any(not isinstance(port, str) or not port for port in (self.lhs_port, self.rhs_port)):
            raise ValueError("shape equality needs named instruction ports")
        if any(type(axis) is not int or axis < 0 for axis in (self.lhs_axis, self.rhs_axis)):
            raise ValueError("shape equality axes must be nonnegative integers")

    def record(self) -> dict[str, str | int]:
        return vars(self).copy()

    @classmethod
    def from_record(cls, row: dict[str, Any]) -> AxisEquality:
        if set(row) != {"lhs_port", "lhs_axis", "rhs_port", "rhs_axis"}:
            raise ValueError("shape equality has missing or unknown fields")
        return cls(**row)


@dataclass(frozen=True)
class IntegerAttributeBound:
    """A computational integer attribute solved from the source operation."""

    name: str
    minimum: int
    maximum: int

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("integer attribute bound needs a name")
        if any(type(value) is not int for value in (self.minimum, self.maximum)) or self.minimum > self.maximum:
            raise ValueError("integer attribute bound needs an ordered finite interval")

    def accepts(self, value: object) -> bool:
        return type(value) is int and self.minimum <= value <= self.maximum

    def record(self) -> dict[str, str | int]:
        return vars(self).copy()

    @classmethod
    def from_record(cls, row: dict[str, Any]) -> IntegerAttributeBound:
        if set(row) != {"name", "minimum", "maximum"}:
            raise ValueError("integer attribute bound has missing or unknown fields")
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
    # A target may read the same encoded input bits from values produced under
    # other named policies. This changes instruction eligibility only: source
    # e-classes, the selected source type, and its numerical history stay intact.
    input_policy_alternatives: tuple[tuple[str, ...], ...] = ()
    input_ranks: tuple[int, ...] = ()
    output_axis_bounds: tuple[AxisBound, ...] = ()
    input_axis_bounds: tuple[tuple[AxisBound, ...], ...] = ()
    index_maps: tuple[IndexMap, ...] = ()
    # Offsets use issued instruction cycles. A delayed operand read keeps its
    # physical source live; completion is also the conservative result-ready
    # event. Zero defaults are only suitable for synchronous descriptions.
    input_read_offsets: tuple[int, ...] = ()
    completion_offset: int = 0
    shape_contract: str = "equal"
    shape_equalities: tuple[AxisEquality, ...] = ()
    # An explicitly qualified instruction may overwrite one produced input
    # only after its read and at completion. Boundary inputs are never mutable.
    in_place_inputs: tuple[int, ...] = ()
    # A physical copy of any already computed value. This may introduce a
    # storage realization without requiring a copy node in semantic source IR.
    value_preserving_copy: bool = False
    # Empty layouts mean the instruction's physical element order is unknown.
    # A layout-qualified request cannot use it as a no-copy view source or sink.
    input_layouts: tuple[str, ...] = ()
    output_layout: str = ""
    # Preconditions on the source value of a physical copy. They do not
    # change the copied value or become attributes of an identity node.
    source_required_attrs: tuple[tuple[str, str | int | float | bool], ...] = ()
    parameter_attrs: tuple[IntegerAttributeBound, ...] = ()
    # Projections of one physical instruction share the same bundle name and
    # operand realization. Every declared result must be selected together.
    bundle_name: str = ""
    bundle_result_index: int = 0
    bundle_result_count: int = 1

    def __post_init__(self) -> None:
        if not all((self.name, self.computation, self.output_storage, self.output_dtype, self.numerical_policy)):
            raise ValueError("instruction descriptor needs semantic, storage and numerical identity")
        if not isinstance(self.bundle_name, str) or type(self.bundle_result_index) is not int or (
            type(self.bundle_result_count) is not int
        ):
            raise ValueError("multi-result instruction fields need typed names and indexes")
        if self.bundle_name:
            if (
                self.bundle_result_count <= 1
                or not 0 <= self.bundle_result_index < self.bundle_result_count
                or self.value_preserving_copy
                or self.parameter_attrs
                or self.in_place_inputs
            ):
                raise ValueError("multi-result instruction needs a fixed, complete pure result signature")
        elif self.bundle_result_index != 0 or self.bundle_result_count != 1:
            raise ValueError("single-result instruction cannot declare a bundle result")
        if not self.ranks or any(rank <= 0 for rank in self.ranks) or self.extent <= 0:
            raise ValueError("instruction ranks and extent must be positive")
        if len({key for key, _ in self.required_attrs}) != len(self.required_attrs):
            raise ValueError("duplicate required attribute")
        parameter_names = [bound.name for bound in self.parameter_attrs]
        if len(set(parameter_names)) != len(parameter_names) or set(parameter_names) & {
            key for key, _ in self.required_attrs
        }:
            raise ValueError("duplicate or conflicting parameter attribute")
        if len({key for key, _ in self.source_required_attrs}) != len(self.source_required_attrs):
            raise ValueError("duplicate source required attribute")
        if self.source_required_attrs and not self.value_preserving_copy:
            raise ValueError("source attributes require a value-preserving copy")
        arity = len(self.input_storages)
        if not isinstance(self.input_layouts, tuple) or self.input_layouts and (
            len(self.input_layouts) != arity
            or any(not isinstance(layout, str) or not layout for layout in self.input_layouts)
        ):
            raise ValueError("instruction input layouts must cover every operand")
        if not isinstance(self.output_layout, str):
            raise ValueError("instruction output layout must be a string")
        if bool(self.input_layouts) != bool(self.output_layout) and arity:
            raise ValueError("instruction input and output layouts must be declared together")
        if len(self.input_dtypes) != arity or len(self.input_numerical_policies) != arity:
            raise ValueError("each instruction operand needs an explicit dtype and numerical policy")
        if not isinstance(self.input_policy_alternatives, tuple) or (
            self.input_policy_alternatives and len(self.input_policy_alternatives) != arity
        ):
            raise ValueError("input policy alternatives must match instruction arity")
        if any(
            not isinstance(alternatives, tuple)
            or any(not isinstance(policy, str) or not policy for policy in alternatives)
            or len(set(alternatives)) != len(alternatives)
            or self.input_numerical_policies[index] in alternatives
            for index, alternatives in enumerate(self.input_policy_alternatives)
        ):
            raise ValueError("input policy alternatives must be distinct, nonempty policies")
        if len(self.input_ranks) not in {0, arity}:
            raise ValueError("input signature must match instruction arity")
        if any(not dtype or not policy for dtype, policy in zip(self.input_dtypes, self.input_numerical_policies)):
            raise ValueError("input dtype and numerical policy must be nonempty")
        if any(rank <= 0 for rank in self.input_ranks):
            raise ValueError("input ranks must be positive")
        if len({bound.axis for bound in self.output_axis_bounds}) != len(self.output_axis_bounds):
            raise ValueError("duplicate output axis bound")
        if self.input_axis_bounds and (len(self.input_axis_bounds) != arity or len(self.input_ranks) != arity):
            raise ValueError("input axis bounds need an explicit rank for every operand")
        for index, bounds in enumerate(self.input_axis_bounds):
            if len({bound.axis for bound in bounds}) != len(bounds) or any(
                bound.axis >= self.input_ranks[index] for bound in bounds
            ):
                raise ValueError("input axis bound is duplicated or exceeds operand rank")
        if self.index_maps and len(self.index_maps) != arity + 1:
            raise ValueError("instruction index maps need one map per operand and result")
        if self.shape_contract not in {"equal", "relations", "bounded"}:
            raise ValueError("unknown instruction shape contract")
        if self.shape_contract == "equal" and self.shape_equalities:
            raise ValueError("equal-shape contract cannot add independent shape relations")
        if self.shape_contract == "relations" and (not self.shape_equalities or len(self.input_ranks) != arity):
            raise ValueError("relational shape contract needs equalities and explicit input ranks")
        if self.shape_contract == "bounded":
            # Without a dimension relation, an instruction can only be
            # admitted at one exact shape per port. Wider intervals would
            # claim unknown input/output shape combinations are equivalent.
            def exact_axes(bounds: tuple[AxisBound, ...], rank: int) -> bool:
                return {bound.axis for bound in bounds} == set(range(rank)) and all(
                    bound.maximum == bound.minimum for bound in bounds
                )

            if (
                self.shape_equalities
                or len(self.ranks) != 1
                or len(self.input_ranks) != arity
                or len(self.input_axis_bounds) != arity
                or not exact_axes(self.output_axis_bounds, self.ranks[0])
                or any(not exact_axes(bounds, rank) for bounds, rank in zip(self.input_axis_bounds, self.input_ranks))
            ):
                raise ValueError("bounded shape contract needs exact bounds on every port axis")
        if self.input_read_offsets and len(self.input_read_offsets) != arity:
            raise ValueError("input read offsets must match instruction arity")
        if type(self.completion_offset) is not int or self.completion_offset < 0:
            raise ValueError("instruction completion offset must be nonnegative")
        if any(
            type(offset) is not int or offset < 0 or offset > self.completion_offset
            for offset in self.input_read_offsets
        ):
            raise ValueError("input read offset must precede instruction completion")
        if len(set(self.in_place_inputs)) != len(self.in_place_inputs) or any(
            type(index) is not int or index < 0 or index >= arity for index in self.in_place_inputs
        ):
            raise ValueError("in-place operand index is invalid")
        if self.in_place_inputs and self.completion_offset == 0:
            raise ValueError("in-place writes need a later completion event")
        if type(self.value_preserving_copy) is not bool:
            raise ValueError("value-preserving copy flag must be boolean")
        if self.value_preserving_copy and (
            self.computation != "identity"
            or arity != 1
            or self.required_attrs
            or self.parameter_attrs
            or self.index_maps
            or self.shape_contract != "equal"
            or self.shape_equalities
            or self.output_dtype != self.input_dtypes[0]
            or self.numerical_policy != self.input_numerical_policies[0]
            or self.input_policy_alternatives
        ):
            raise ValueError("value-preserving copy must preserve one complete tensor type")
        if any(
            self.input_read_offsets and self.input_read_offsets[index] >= self.completion_offset
            for index in self.in_place_inputs
        ):
            raise ValueError("in-place input must be read before output completion")
        ports = {"out", *(f"in{index}" for index in range(len(self.input_storages)))}
        for condition in self.validity:
            if condition.lhs not in ports or (condition.rhs and condition.rhs not in ports):
                raise ValueError("address constraint refers to an undeclared instruction port")
        port_ranks = {"out": min(self.ranks), **{f"in{index}": rank for index, rank in enumerate(self.input_ranks)}}
        for condition in self.shape_equalities:
            if condition.lhs_port not in port_ranks or condition.rhs_port not in port_ranks:
                raise ValueError("shape equality refers to an undeclared instruction port")
            if condition.lhs_axis >= port_ranks[condition.lhs_port] or (
                condition.rhs_axis >= port_ranks[condition.rhs_port]
            ):
                raise ValueError("shape equality axis exceeds instruction port rank")

    def accepts_input_policy(self, index: int, policy: str) -> bool:
        if not 0 <= index < len(self.input_numerical_policies):
            return False
        return policy == self.input_numerical_policies[index] or (
            bool(self.input_policy_alternatives)
            and policy in self.input_policy_alternatives[index]
        )

    def accepts(self, node: SemanticNode, inputs: tuple[SemanticNode, ...]) -> bool:
        attributes = dict(node.attrs)
        fixed = dict(self.required_attrs)
        return (
            node.effect == "pure"
            and node.op == self.computation
            and len(inputs) == len(self.input_storages)
            and node.type.dtype == self.output_dtype
            and node.type.numerical_policy == self.numerical_policy
            and len(node.type.shape) in self.ranks
            and all(bound.accepts(node.type.shape) for bound in self.output_axis_bounds)
            and (
                not self.input_axis_bounds
                or all(
                    all(bound.accepts(value.type.shape) for bound in bounds)
                    for value, bounds in zip(inputs, self.input_axis_bounds)
                )
            )
            and node.index_maps == self.index_maps
            # Semantic attributes are part of the computation. An undeclared
            # attribute cannot silently become a hardware don't-care.
            and set(attributes) == set(fixed) | {bound.name for bound in self.parameter_attrs}
            and all(type(attributes[key]) is type(value) and attributes[key] == value for key, value in fixed.items())
            and all(bound.accepts(attributes[bound.name]) for bound in self.parameter_attrs)
            and all(n.type.dtype == dtype for n, dtype in zip(inputs, self.input_dtypes))
            and all(self.accepts_input_policy(index, n.type.numerical_policy)
                    for index, n in enumerate(inputs))
            and (not self.input_ranks or all(len(n.type.shape) == rank for n, rank in zip(inputs, self.input_ranks)))
            and (
                all(n.type.shape == node.type.shape for n in inputs)
                if self.shape_contract == "equal"
                else self._relations_accept(node, inputs)
                if self.shape_contract == "relations"
                else True
            )
        )

    def _relations_accept(self, node: SemanticNode, inputs: tuple[SemanticNode, ...]) -> bool:
        shapes = {"out": node.type.shape, **{f"in{index}": value.type.shape for index, value in enumerate(inputs)}}
        return all(
            shapes[item.lhs_port][item.lhs_axis] == shapes[item.rhs_port][item.rhs_axis]
            for item in self.shape_equalities
        )

    def record(self) -> dict[str, Any]:
        result = {
            "name": self.name,
            "computation": self.computation,
            "input_storages": list(self.input_storages),
            "output_storage": self.output_storage,
            "output_dtype": self.output_dtype,
            "numerical_policy": self.numerical_policy,
            "ranks": list(self.ranks),
            "required_attrs": dict(self.required_attrs),
            "source_required_attrs": dict(self.source_required_attrs),
            "extent": self.extent,
            "validity": [condition.record() for condition in self.validity],
            "input_dtypes": list(self.input_dtypes),
            "input_numerical_policies": list(self.input_numerical_policies),
            "input_ranks": list(self.input_ranks),
            "output_axis_bounds": [bound.record() for bound in self.output_axis_bounds],
            "input_axis_bounds": [[bound.record() for bound in bounds] for bounds in self.input_axis_bounds],
            "index_maps": [index_map.record() for index_map in self.index_maps],
            "input_read_offsets": list(self.input_read_offsets),
            "completion_offset": self.completion_offset,
            "shape_contract": self.shape_contract,
            "shape_equalities": [condition.record() for condition in self.shape_equalities],
            "in_place_inputs": list(self.in_place_inputs),
            "value_preserving_copy": self.value_preserving_copy,
        }
        if self.parameter_attrs:
            result["parameter_attrs"] = [bound.record() for bound in self.parameter_attrs]
        if self.input_policy_alternatives:
            result["input_policy_alternatives"] = [list(policies) for policies in self.input_policy_alternatives]
        if self.input_layouts or self.output_layout:
            result["input_layouts"] = list(self.input_layouts)
            result["output_layout"] = self.output_layout
        if self.bundle_name:
            result["bundle_name"] = self.bundle_name
            result["bundle_result_index"] = self.bundle_result_index
            result["bundle_result_count"] = self.bundle_result_count
        return result

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
            "source_required_attrs",
            "extent",
            "validity",
            "input_dtypes",
            "input_numerical_policies",
            "input_ranks",
            "output_axis_bounds",
            "input_axis_bounds",
            "index_maps",
            "input_read_offsets",
            "completion_offset",
            "shape_contract",
            "shape_equalities",
            "in_place_inputs",
            "value_preserving_copy",
        }
        required = expected - {"source_required_attrs"}
        allowed = expected | {
            "parameter_attrs", "input_policy_alternatives", "input_layouts", "output_layout",
            "bundle_name", "bundle_result_index", "bundle_result_count",
        }
        if not required <= set(row) <= allowed:
            raise ValueError("instruction descriptor has missing or unknown fields")
        if any(key in row for key in ("bundle_name", "bundle_result_index", "bundle_result_count")) and not all(
            key in row for key in ("bundle_name", "bundle_result_index", "bundle_result_count")
        ):
            raise ValueError("multi-result instruction record needs all bundle fields")
        if ("input_layouts" in row) != ("output_layout" in row):
            raise ValueError("instruction physical layouts must be recorded together")
        raw_alternatives = row.get("input_policy_alternatives", [])
        raw_layouts = row.get("input_layouts", [])
        if not isinstance(raw_layouts, list) or not isinstance(row.get("output_layout", ""), str):
            raise ValueError("instruction physical layouts must have typed fields")
        if not isinstance(raw_alternatives, list) or any(
            not isinstance(policies, list) for policies in raw_alternatives
        ):
            raise ValueError("input policy alternatives must be lists of policy names")
        return cls(
            name=row["name"],
            computation=row["computation"],
            input_storages=tuple(row["input_storages"]),
            output_storage=row["output_storage"],
            output_dtype=row["output_dtype"],
            numerical_policy=row["numerical_policy"],
            ranks=tuple(row["ranks"]),
            required_attrs=tuple(sorted(row["required_attrs"].items())),
            parameter_attrs=tuple(IntegerAttributeBound.from_record(item) for item in row.get("parameter_attrs", ())),
            source_required_attrs=tuple(sorted(row.get("source_required_attrs", {}).items())),
            extent=row["extent"],
            validity=tuple(AddressConstraint.from_record(item) for item in row["validity"]),
            input_dtypes=tuple(row["input_dtypes"]),
            input_numerical_policies=tuple(row["input_numerical_policies"]),
            input_policy_alternatives=tuple(
                tuple(policies) for policies in raw_alternatives
            ),
            input_ranks=tuple(row["input_ranks"]),
            output_axis_bounds=tuple(AxisBound.from_record(item) for item in row["output_axis_bounds"]),
            input_axis_bounds=tuple(
                tuple(AxisBound.from_record(item) for item in bounds) for bounds in row["input_axis_bounds"]
            ),
            index_maps=tuple(IndexMap.from_record(item) for item in row["index_maps"]),
            input_read_offsets=tuple(row["input_read_offsets"]),
            completion_offset=row["completion_offset"],
            shape_contract=row["shape_contract"],
            shape_equalities=tuple(AxisEquality.from_record(item) for item in row["shape_equalities"]),
            in_place_inputs=tuple(row["in_place_inputs"]),
            value_preserving_copy=row["value_preserving_copy"],
            input_layouts=tuple(raw_layouts),
            output_layout=row.get("output_layout", ""),
            bundle_name=row.get("bundle_name", ""),
            bundle_result_index=row.get("bundle_result_index", 0),
            bundle_result_count=row.get("bundle_result_count", 1),
        )


@dataclass(frozen=True)
class ViewDescriptor:
    """Target assertion for a contiguous reshape or outermost-axis slice view.

    Exact types and the extent in the named bank's unit are intentional: a
    shape-changing tensor operation is not a free view merely because its
    logical elements have the same count. The target must qualify this layout
    for both shapes before publishing the descriptor. A slice also declares
    its source extent and offset in that same physical unit.
    """

    name: str
    storage: str
    source_type: TensorType
    result_type: TensorType
    extent: int
    physical_layout: str = "contiguous_element_order"
    kind: str = "reshape"
    source_extent: int | None = None
    physical_offset: int = 0

    def __post_init__(self) -> None:
        if (
            not self.name or not self.storage or type(self.extent) is not int or self.extent <= 0
            or type(self.physical_offset) is not int
        ):
            raise ValueError("view needs a name, storage and positive physical extent")
        if self.physical_layout != "contiguous_element_order":
            raise ValueError("view needs a declared contiguous element order")
        if self.kind == "reshape":
            if (
                self.source_extent is not None or self.physical_offset != 0
                or self.source_type.dtype != self.result_type.dtype
                or self.source_type.numerical_policy != self.result_type.numerical_policy
                or prod(self.source_type.shape) != prod(self.result_type.shape)
            ):
                raise ValueError("reshape view must preserve element bits and count")
        elif self.kind == "slice":
            if (
                self.source_type.dtype != self.result_type.dtype
                or self.source_type.numerical_policy != self.result_type.numerical_policy
            ):
                raise ValueError("slice view must preserve element bits and numerical policy")
            if (
                len(self.source_type.shape) != len(self.result_type.shape)
                or self.source_type.shape[1:] != self.result_type.shape[1:]
                or type(self.source_extent) is not int or self.source_extent <= 0
                or type(self.physical_offset) is not int or self.physical_offset < 0
                or self.source_extent % prod(self.source_type.shape)
            ):
                raise ValueError("slice view needs outermost-axis contiguous physical geometry")
            units_per_element = self.source_extent // prod(self.source_type.shape)
            units_per_outer_slice = units_per_element * prod(self.source_type.shape[1:])
            if (
                self.extent != prod(self.result_type.shape) * units_per_element
                or self.physical_offset % units_per_outer_slice
                or self.physical_offset + self.extent > self.source_extent
            ):
                raise ValueError("slice view offset or extent differs from contiguous elements")
        else:
            raise ValueError("unknown view kind")

    def accepts(self, node: SemanticNode, source: SemanticNode | tuple[SemanticNode, ...]) -> bool:
        if isinstance(source, tuple):
            if len(source) != 1:
                return False
            source = source[0]
        common = (
            node.effect == "pure"
            and not node.attrs
            and node.inputs == (source.id,)
            and source.type == self.source_type
            and node.type == self.result_type
        )
        if not common:
            return False
        if self.kind == "reshape":
            return node.op == "reshape" and not node.index_maps
        rank = len(self.source_type.shape)
        units_per_element = self.source_extent // prod(self.source_type.shape)
        start = self.physical_offset // (units_per_element * prod(self.source_type.shape[1:]))
        identity = tuple(tuple(int(axis == dim) for axis in range(rank)) for dim in range(rank))
        return node.op == "gather" and node.index_maps == (
            IndexMap(rank, identity, (start, *(0 for _ in range(rank - 1)))),
            IndexMap(rank, identity, (0,) * rank),
        )

    def record(self) -> dict[str, Any]:
        row = {
            "name": self.name,
            "storage": self.storage,
            "source_type": self.source_type.record(),
            "result_type": self.result_type.record(),
            "extent": self.extent,
            "physical_layout": self.physical_layout,
        }
        if self.kind == "slice":
            row.update({"kind": self.kind, "source_extent": self.source_extent,
                        "physical_offset": self.physical_offset})
        return row

    @classmethod
    def from_record(cls, row: dict[str, Any]) -> ViewDescriptor | ConcatViewDescriptor:
        if row.get("kind") == "concat":
            return ConcatViewDescriptor.from_record(row)
        base = {"name", "storage", "source_type", "result_type", "extent", "physical_layout"}
        kind = row.get("kind", "reshape")
        expected = base if kind == "reshape" else base | {"kind", "source_extent", "physical_offset"}
        if set(row) != expected:
            raise ValueError("view has missing or unknown fields")
        return cls(
            row["name"], row["storage"], TensorType.from_record(row["source_type"]),
            TensorType.from_record(row["result_type"]), row["extent"], row["physical_layout"],
            kind, row.get("source_extent"), row.get("physical_offset", 0),
        )


@dataclass(frozen=True)
class ConcatViewDescriptor:
    """Outermost-axis concatenation of contiguous row-major storage spans.

    Every source and the result use the same bank and physical unit width per
    logical element. The allocator keeps this view until all adjacency
    obligations are checked; it never initializes or copies a child implicitly.
    """

    name: str
    storage: str
    source_types: tuple[TensorType, ...]
    result_type: TensorType
    source_extents: tuple[int, ...]
    extent: int
    physical_layout: str = "contiguous_element_order"

    def __post_init__(self) -> None:
        if not self.name or not self.storage or self.physical_layout != "contiguous_element_order":
            raise ValueError("concat view needs named contiguous storage")
        if (
            len(self.source_types) < 2 or len(self.source_types) != len(self.source_extents)
            or not self.result_type.shape
            or any(len(source.shape) != len(self.result_type.shape) for source in self.source_types)
            or any(type(size) is not int or size <= 0 for size in self.source_extents)
            or type(self.extent) is not int or self.extent <= 0
        ):
            raise ValueError("concat view needs equal-rank positive source extents")
        if (
            any((source.dtype, source.numerical_policy)
                != (self.result_type.dtype, self.result_type.numerical_policy)
                for source in self.source_types)
            or any(source.shape[1:] != self.result_type.shape[1:] for source in self.source_types)
            or sum(source.shape[0] for source in self.source_types) != self.result_type.shape[0]
        ):
            raise ValueError("concat view changes element order, type, or numerical policy")
        widths = []
        for source, size in zip(self.source_types, self.source_extents, strict=True):
            if size % prod(source.shape):
                raise ValueError("concat view has a nonintegral physical element width")
            widths.append(size // prod(source.shape))
        if len(set(widths)) != 1 or sum(self.source_extents) != self.extent:
            raise ValueError("concat view extents disagree with contiguous source elements")

    def accepts(self, node: SemanticNode, sources: tuple[SemanticNode, ...]) -> bool:
        return (
            node.effect == "pure"
            and node.op == "concat"
            and node.attrs == (("axis", 0),)
            and not node.index_maps
            and node.inputs == tuple(source.id for source in sources)
            and tuple(source.type for source in sources) == self.source_types
            and node.type == self.result_type
        )

    def record(self) -> dict[str, Any]:
        return {
            "kind": "concat",
            "name": self.name,
            "storage": self.storage,
            "source_types": [source.record() for source in self.source_types],
            "result_type": self.result_type.record(),
            "source_extents": list(self.source_extents),
            "extent": self.extent,
            "physical_layout": self.physical_layout,
        }

    @classmethod
    def from_record(cls, row: dict[str, Any]) -> ConcatViewDescriptor:
        if set(row) != {
            "kind", "name", "storage", "source_types", "result_type",
            "source_extents", "extent", "physical_layout",
        } or row["kind"] != "concat":
            raise ValueError("concat view has missing or unknown fields")
        if not isinstance(row["source_types"], list) or not isinstance(row["source_extents"], list):
            raise ValueError("concat view sources must be lists")
        return cls(
            row["name"], row["storage"],
            tuple(TensorType.from_record(source) for source in row["source_types"]),
            TensorType.from_record(row["result_type"]), tuple(row["source_extents"]),
            row["extent"], row["physical_layout"],
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
