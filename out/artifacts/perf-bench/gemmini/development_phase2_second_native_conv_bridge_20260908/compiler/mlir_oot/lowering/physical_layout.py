"""Target-neutral physical-layout planning over a producer-consumer graph.

The planner deliberately knows nothing about a particular accelerator.  A caller supplies the
legal layouts, the preferred layout of accelerated operators, and the semantic classes that can
carry a physical layout without moving data.  The solver then:

* joins values across layout-preserving operators;
* joins every input/output of a residual merge, making branch agreement a hard invariant;
* selects one legal layout for each joined component; and
* emits conversions only at ports of operators that cannot consume the selected layout.

This module is a planning contract.  It does not rewrite storage until a backend has provided a
lowering for every conversion and every selected operator layout.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import reduce
from operator import mul
from typing import Any, Iterable


class LayoutPlanningError(ValueError):
    """The supplied graph or layout constraints have no sound solution."""


def _elements(shape: Iterable[int]) -> int:
    return reduce(mul, (int(dim) for dim in shape), 1)


def _dtype_bytes(dtype: str) -> int:
    text = str(dtype)
    if len(text) > 1 and text[0] in ("i", "u", "f") and text[1:].isdigit():
        bits = int(text[1:])
        if bits > 0:
            return max(1, (bits + 7) // 8)
    if text == "bf16":
        return 2
    raise LayoutPlanningError(f"cannot derive storage width for dtype {dtype!r}")


@dataclass(frozen=True)
class LayoutValue:
    """One logical tensor whose physical axis order is selected by the planner."""

    name: str
    shape: tuple[int, ...]
    dtype: str

    @property
    def nbytes(self) -> int:
        return _elements(self.shape) * _dtype_bytes(self.dtype)

    def to_dict(self) -> dict[str, Any]:
        return {"shape": list(self.shape), "dtype": self.dtype, "nbytes": self.nbytes}


@dataclass(frozen=True)
class LayoutOp:
    """Physical-layout behavior of one logical operator.

    ``couple`` means that all listed input and output values have the same physical layout.  It is
    used for layout-polymorphic convolution, elementwise chains, and residual merges.  A non-coupled
    op is a true boundary: each port is converted if the surrounding value does not use one of the
    op's ``supported_layouts``.
    """

    name: str
    kind: str
    inputs: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()
    couple: bool = False
    supported_layouts: tuple[str, ...] = ()
    preferred_layout: str | None = None
    preference_weight: int = 1
    reason: str = ""

    @property
    def ports(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys((*self.inputs, *self.outputs)))

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "name": self.name,
            "kind": self.kind,
            "inputs": list(self.inputs),
            "outputs": list(self.outputs),
            "couple": self.couple,
            "supported_layouts": list(self.supported_layouts),
            "preference_weight": self.preference_weight,
        }
        if self.preferred_layout is not None:
            result["preferred_layout"] = self.preferred_layout
        if self.reason:
            result["reason"] = self.reason
        return result

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "LayoutOp":
        return cls(
            name=str(value["name"]),
            kind=str(value["kind"]),
            inputs=tuple(str(item) for item in value.get("inputs", ())),
            outputs=tuple(str(item) for item in value.get("outputs", ())),
            couple=bool(value.get("couple", False)),
            supported_layouts=tuple(str(item) for item in value.get("supported_layouts", ())),
            preferred_layout=(str(value["preferred_layout"])
                              if value.get("preferred_layout") is not None else None),
            preference_weight=int(value.get("preference_weight", 1)),
            reason=str(value.get("reason", "")),
        )


@dataclass(frozen=True)
class LayoutGraph:
    """Serializable input to :func:`solve`."""

    layouts: tuple[str, ...]
    canonical_layout: str
    values: dict[str, LayoutValue]
    ops: tuple[LayoutOp, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "target_neutral_physical_layout_graph_v1",
            "layouts": list(self.layouts),
            "canonical_layout": self.canonical_layout,
            "values": {name: value.to_dict() for name, value in sorted(self.values.items())},
            "ops": [op.to_dict() for op in self.ops],
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "LayoutGraph":
        values = {
            str(name): LayoutValue(str(name), tuple(int(d) for d in desc["shape"]),
                                   str(desc["dtype"]))
            for name, desc in value["values"].items()
        }
        return cls(
            layouts=tuple(str(item) for item in value["layouts"]),
            canonical_layout=str(value["canonical_layout"]),
            values=values,
            ops=tuple(LayoutOp.from_dict(item) for item in value["ops"]),
        )


@dataclass(frozen=True)
class LayoutConversion:
    """One required physical conversion at a non-propagating operator port."""

    op: str
    port: str
    value: str
    source_layout: str
    target_layout: str
    nbytes: int
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "op": self.op,
            "port": self.port,
            "value": self.value,
            "source_layout": self.source_layout,
            "target_layout": self.target_layout,
            "nbytes": self.nbytes,
            "reason": self.reason,
        }


@dataclass
class LayoutPlan:
    """Chosen layouts and the exact conversions needed to realize them."""

    assignments: dict[str, str]
    components: list[dict[str, Any]]
    conversions: list[LayoutConversion]
    op_counts: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "target_neutral_physical_layout_plan_v1",
            "assignments": dict(sorted(self.assignments.items())),
            "components": self.components,
            "conversions": [item.to_dict() for item in self.conversions],
            "summary": {
                "values": len(self.assignments),
                "components": len(self.components),
                "conversions": len(self.conversions),
                "conversion_bytes": sum(item.nbytes for item in self.conversions),
                "op_counts": dict(sorted(self.op_counts.items())),
            },
        }


class _DisjointSet:
    def __init__(self, names: Iterable[str]):
        self.parent = {name: name for name in names}

    def find(self, name: str) -> str:
        root = name
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[name] != name:
            parent = self.parent[name]
            self.parent[name] = root
            name = parent
        return root

    def union(self, lhs: str, rhs: str) -> None:
        left, right = self.find(lhs), self.find(rhs)
        if left == right:
            return
        # Stable roots make serialized plans reproducible across processes.
        if right < left:
            left, right = right, left
        self.parent[right] = left


def _validate(graph: LayoutGraph) -> None:
    if not graph.layouts:
        raise LayoutPlanningError("layout domain is empty")
    if len(set(graph.layouts)) != len(graph.layouts):
        raise LayoutPlanningError("layout domain contains duplicates")
    if graph.canonical_layout not in graph.layouts:
        raise LayoutPlanningError("canonical layout is outside the layout domain")
    for op in graph.ops:
        missing = sorted(set(op.ports) - set(graph.values))
        if missing:
            raise LayoutPlanningError(f"operator {op.name!r} references unknown values {missing}")
        if not op.supported_layouts:
            raise LayoutPlanningError(f"operator {op.name!r} declares no supported layouts")
        unsupported = sorted(set(op.supported_layouts) - set(graph.layouts))
        if unsupported:
            raise LayoutPlanningError(
                f"operator {op.name!r} supports layouts outside the graph domain: {unsupported}")
        if op.preferred_layout is not None and op.preferred_layout not in op.supported_layouts:
            raise LayoutPlanningError(
                f"operator {op.name!r} prefers unsupported layout {op.preferred_layout!r}")
        if op.preference_weight < 0:
            raise LayoutPlanningError(f"operator {op.name!r} has a negative preference weight")


def _native_layout(op: LayoutOp, canonical: str) -> str:
    if op.preferred_layout in op.supported_layouts:
        return op.preferred_layout  # type: ignore[return-value]
    if canonical in op.supported_layouts:
        return canonical
    return op.supported_layouts[0]


def solve(graph: LayoutGraph) -> LayoutPlan:
    """Solve ``graph`` with lexicographic accelerator coverage then conversion cost.

    The primary objective maximizes accelerated operators in their preferred layout.  This avoids
    selecting a globally slow layout merely because the model has one large external tensor.  Among
    plans with equal preferred-op coverage, conversion bytes and then conversion count are minimized.
    The final tie follows the caller's layout-domain order and is deterministic.
    """
    _validate(graph)
    joined = _DisjointSet(graph.values)
    for op in graph.ops:
        if not op.couple or len(op.ports) < 2:
            continue
        head = op.ports[0]
        for name in op.ports[1:]:
            joined.union(head, name)

    members: dict[str, list[str]] = {}
    for name in graph.values:
        members.setdefault(joined.find(name), []).append(name)
    for values in members.values():
        values.sort()

    coupled: dict[str, list[LayoutOp]] = {root: [] for root in members}
    boundaries: dict[str, list[tuple[LayoutOp, str, str]]] = {root: [] for root in members}
    for op in graph.ops:
        if op.couple and op.ports:
            coupled[joined.find(op.ports[0])].append(op)
        elif not op.couple:
            for name in op.inputs:
                boundaries[joined.find(name)].append((op, "input", name))
            for name in op.outputs:
                boundaries[joined.find(name)].append((op, "output", name))

    assignments: dict[str, str] = {}
    component_rows: list[dict[str, Any]] = []
    for root in sorted(members):
        legal = list(graph.layouts)
        for op in coupled[root]:
            legal = [layout for layout in legal if layout in op.supported_layouts]
        if not legal:
            names = [op.name for op in coupled[root]]
            raise LayoutPlanningError(
                f"joined component {members[root]} has incompatible operator layouts: {names}")

        choices: list[tuple[tuple[int, int, int, int], str, int, int, int]] = []
        for domain_index, layout in enumerate(legal):
            preferred = sum(op.preference_weight for op in coupled[root]
                            if op.preferred_layout == layout)
            mismatches = [(op, port, name) for op, port, name in boundaries[root]
                          if layout not in op.supported_layouts]
            conversion_bytes = sum(graph.values[name].nbytes for _, _, name in mismatches)
            score = (-preferred, conversion_bytes, len(mismatches), domain_index)
            choices.append((score, layout, preferred, conversion_bytes, len(mismatches)))
        _, selected, preferred, conversion_bytes, conversion_count = min(choices)
        for name in members[root]:
            assignments[name] = selected
        component_rows.append({
            "id": root,
            "layout": selected,
            "values": members[root],
            "coupled_ops": [op.name for op in coupled[root]],
            "preferred_weight_satisfied": preferred,
            "boundary_conversion_bytes": conversion_bytes,
            "boundary_conversions": conversion_count,
        })

    conversions: list[LayoutConversion] = []
    for op in graph.ops:
        if op.couple:
            continue
        native = _native_layout(op, graph.canonical_layout)
        for port, names in (("input", op.inputs), ("output", op.outputs)):
            for name in names:
                selected = assignments[name]
                if selected in op.supported_layouts:
                    continue
                source, target = ((selected, native) if port == "input" else (native, selected))
                conversions.append(LayoutConversion(
                    op=op.name,
                    port=port,
                    value=name,
                    source_layout=source,
                    target_layout=target,
                    nbytes=graph.values[name].nbytes,
                    reason=op.reason or "operator does not support the surrounding physical layout",
                ))

    op_counts: dict[str, int] = {}
    for op in graph.ops:
        op_counts[op.kind] = op_counts.get(op.kind, 0) + 1
    return LayoutPlan(assignments, component_rows, conversions, op_counts)
