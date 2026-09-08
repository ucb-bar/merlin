"""Policy-driven physical-layout topology extraction from provenance-annotated tensor IR.

Semantic names are caller data.  The extractor therefore neither knows model names nor assumes one
spelling for a convolution family.  It distinguishes activation tensors from rank-equal parameters
using each region's externally visible output batch extent, records every omitted region with a
reason, and returns ``not_applicable`` when the requested activation rank does not occur.
"""
from __future__ import annotations

from collections import Counter, OrderedDict
from dataclasses import dataclass
from typing import Any, Callable, Sequence

from merlin.perf.physical_layout import (
    CapacityLimit,
    ConversionCapability,
    LayoutGraph,
    LayoutOp,
    LayoutRefusal,
    LayoutValue,
    OperatorEncoding,
    PhysicalEncoding,
    PhysicalLayoutError,
)


EncodingProvider = Callable[[str, tuple[int, ...], str], Sequence[PhysicalEncoding]]


@dataclass(frozen=True)
class RegionLayoutPolicy:
    """Target-neutral semantic classes and externally supplied physical facts."""

    canonical_encoding: str
    preferred_accelerator_encoding: str
    accelerator_ops: frozenset[str]
    preserving_ops: frozenset[str]
    residual_ops: frozenset[str]
    view_ops: frozenset[str]
    fixed_layout_ops: frozenset[str]
    weight_ops: frozenset[str]
    encoding_provider: EncodingProvider
    capabilities: tuple[str, ...] = ()
    conversions: tuple[ConversionCapability, ...] = ()
    capacity_limits: tuple[CapacityLimit, ...] = ()
    activation_rank: int = 4
    max_assignments: int = 100_000

    def __post_init__(self) -> None:
        if not self.canonical_encoding.strip() or not self.preferred_accelerator_encoding.strip():
            raise PhysicalLayoutError("capture policy requires canonical and preferred encodings")
        if type(self.activation_rank) is not int or self.activation_rank <= 0:
            raise PhysicalLayoutError("capture activation rank must be positive")
        semantic_sets = (
            self.accelerator_ops, self.preserving_ops, self.residual_ops, self.view_ops,
            self.fixed_layout_ops, self.weight_ops,
        )
        if any(not name.strip() for names in semantic_sets for name in names):
            raise PhysicalLayoutError("capture semantic classes require nonempty names")
        seen: set[str] = set()
        for names in semantic_sets:
            overlap = seen & names
            if overlap:
                raise PhysicalLayoutError(
                    f"capture semantic classes overlap for {sorted(overlap)}")
            seen.update(names)


@dataclass(frozen=True)
class LayoutOmission:
    region: str
    semantic: str
    reason: str
    input_shapes: tuple[tuple[int, ...], ...]
    output_shapes: tuple[tuple[int, ...], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "region": self.region,
            "semantic": self.semantic,
            "reason": self.reason,
            "input_shapes": [list(shape) for shape in self.input_shapes],
            "output_shapes": [list(shape) for shape in self.output_shapes],
        }


@dataclass(frozen=True)
class LayoutCaptureResult:
    status: str
    graph: LayoutGraph | None
    census: tuple[tuple[str, Any], ...]
    omissions: tuple[LayoutOmission, ...]
    refusals: tuple[LayoutRefusal, ...]

    @property
    def applicable(self) -> bool:
        return self.graph is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "target_neutral_physical_layout_capture_v1",
            "status": self.status,
            "applicable": self.applicable,
            "graph": self.graph.to_dict() if self.graph is not None else None,
            "census": dict(self.census),
            "omissions": [item.to_dict() for item in self.omissions],
            "refusals": [item.to_dict() for item in self.refusals],
        }


def _text_attr(op: Any, name: str) -> str:
    for attributes in (op.attributes, getattr(op, "properties", {}) or {}):
        value = attributes.get(name)
        if value is not None:
            return str(getattr(value, "data", "") or "")
    return ""


def _shape(value: Any) -> tuple[int, ...]:
    try:
        return tuple(int(dim) for dim in value.type.get_shape())
    except (AttributeError, TypeError, ValueError):
        return ()


def _dtype(value: Any) -> str:
    try:
        return str(value.type.get_element_type())
    except AttributeError:
        return str(value.type)


def _entry_block(module: Any) -> Any:
    for op in module.walk():
        if op.name == "func.func":
            return op.regions[0].blocks[0]
    raise PhysicalLayoutError("module has no function entry")


def _semantic(ops: Sequence[Any]) -> str:
    names = [_text_attr(op, "prov.op") for op in ops if _text_attr(op, "prov.op")]
    if not names:
        return ""
    counts = Counter(names)
    first = tuple(dict.fromkeys(names))
    return max(first, key=lambda name: (counts[name], -first.index(name)))


def _helper_producer(value: Any) -> bool:
    owner = getattr(value, "owner", None)
    return getattr(owner, "name", "") in {"arith.constant", "tensor.empty", "tensor.splat"}


def _safe_view(inputs: Sequence[Any], outputs: Sequence[Any]) -> bool:
    return len(inputs) == 1 and len(outputs) == 1 and _shape(inputs[0]) == _shape(outputs[0])


def _topological_rows(rows: Sequence[tuple[int, LayoutOp]]) -> tuple[LayoutOp, ...]:
    """Stable topological order for provenance regions whose payloads may interleave."""
    producer: dict[str, str] = {}
    by_name = {op.name: (position, op) for position, op in rows}
    for _, op in rows:
        for output in op.outputs:
            previous = producer.get(output)
            if previous is not None and previous != op.name:
                raise PhysicalLayoutError(
                    f"captured value {output!r} has producers {previous!r} and {op.name!r}")
            producer[output] = op.name
    dependencies: dict[str, set[str]] = {name: set() for name in by_name}
    followers: dict[str, set[str]] = {name: set() for name in by_name}
    for _, op in rows:
        for value in op.inputs:
            source = producer.get(value)
            if source is not None and source != op.name:
                dependencies[op.name].add(source)
                followers[source].add(op.name)
    ready = sorted((name for name, deps in dependencies.items() if not deps),
                   key=lambda name: (by_name[name][0], name))
    ordered: list[LayoutOp] = []
    while ready:
        name = ready.pop(0)
        ordered.append(by_name[name][1])
        for follower in sorted(followers[name]):
            dependencies[follower].discard(name)
            if not dependencies[follower] and follower not in ready:
                ready.append(follower)
        ready.sort(key=lambda item: (by_name[item][0], item))
    if len(ordered) != len(rows):
        blocked = sorted(name for name, deps in dependencies.items() if deps)
        raise PhysicalLayoutError(
            f"captured physical-layout graph has a dependency cycle among {blocked}")
    return tuple(ordered)


def _region_ports(ops: Sequence[Any], rank: int, *, require_output_batch: bool) -> tuple[
        list[Any], list[Any], str | None, tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...]]:
    owned = set(ops)
    raw_inputs: list[Any] = []
    raw_outputs: list[Any] = []
    for op in ops:
        for value in op.operands:
            if (getattr(value, "owner", None) not in owned and value not in raw_inputs
                    and not _helper_producer(value)):
                raw_inputs.append(value)
        for value in op.results:
            if (value not in raw_outputs
                    and any(use.operation not in owned for use in value.uses)):
                raw_outputs.append(value)
    ranked_inputs = [value for value in raw_inputs if len(_shape(value)) == rank]
    ranked_outputs = [value for value in raw_outputs if len(_shape(value)) == rank]
    input_shapes = tuple(_shape(value) for value in raw_inputs)
    output_shapes = tuple(_shape(value) for value in raw_outputs)
    output_batches = {shape[0] for shape in map(_shape, ranked_outputs)}
    if not output_batches:
        if require_output_batch:
            return [], [], "accelerator_activation_output_unproven", input_shapes, output_shapes
        input_batches = {shape[0] for shape in map(_shape, ranked_inputs)}
        if len(input_batches) != 1:
            return [], [], "activation_batch_unproven", input_shapes, output_shapes
        output_batches = input_batches
    if len(output_batches) != 1:
        return [], [], "ambiguous_activation_batch", input_shapes, output_shapes
    batch = next(iter(output_batches))
    inputs = [value for value in ranked_inputs if _shape(value)[0] == batch]
    outputs = [value for value in ranked_outputs if _shape(value)[0] == batch]
    if not inputs and not outputs:
        return [], [], "no_external_activation_ports", input_shapes, output_shapes
    return inputs, outputs, None, input_shapes, output_shapes


def extract_region_layout_graph(module: Any, policy: RegionLayoutPolicy) -> LayoutCaptureResult:
    """Extract exact activation topology or return a deterministic not-applicable result."""

    block = _entry_block(module)
    top_ops = [op for op in block.ops if op.name != "func.return"]
    position = {op: index for index, op in enumerate(block.ops)}
    regions: OrderedDict[str, list[Any]] = OrderedDict()
    region_of: dict[Any, str] = {}
    for op in top_ops:
        region = _text_attr(op, "prov.region_id")
        if region:
            regions.setdefault(region, []).append(op)
            region_of[op] = region
    # model-to-IR capture marks several anchor operations in a lowered source region, while helper
    # operations between them may be unmarked.  Infer ownership only for non-overlapping contiguous
    # anchor intervals; overlapping intervals remain unattributed and are refused below.
    anchored_positions = {
        region: (min(position[op] for op in ops), max(position[op] for op in ops))
        for region, ops in regions.items()
    }
    inferred_interior_operations = 0
    for op in top_ops:
        if op in region_of:
            continue
        candidates = [region for region, (first, last) in anchored_positions.items()
                      if first <= position[op] <= last]
        if len(candidates) == 1:
            region = candidates[0]
            regions[region].append(op)
            regions[region].sort(key=position.__getitem__)
            region_of[op] = region
            inferred_interior_operations += 1
    semantic_of = {name: _semantic(ops) for name, ops in regions.items()}
    semantic_counts = Counter(semantic_of.values())

    names: dict[Any, str] = {}
    for index, argument in enumerate(block.args):
        names[argument] = f"arg{index}"
    for op_index, op in enumerate(block.ops):
        for result_index, result in enumerate(op.results):
            names[result] = f"v{op_index}_{result_index}"

    graph_values: dict[str, LayoutValue] = {}

    def add_value(value: Any) -> str:
        name = names[value]
        if name not in graph_values:
            shape, dtype = _shape(value), _dtype(value)
            encodings = tuple(policy.encoding_provider(name, shape, dtype))
            graph_values[name] = LayoutValue(name, shape, dtype, encodings)
        return name

    rows: list[tuple[int, LayoutOp]] = []
    omissions: list[LayoutOmission] = []
    capture_refusals: list[LayoutRefusal] = []
    included = Counter()
    for region, ops in regions.items():
        semantic = semantic_of[region]
        if semantic in policy.weight_ops:
            omissions.append(LayoutOmission(region, semantic, "parameter_only_region", (), ()))
            continue
        inputs, outputs, reason, input_shapes, output_shapes = _region_ports(
            ops, policy.activation_rank, require_output_batch=semantic in policy.accelerator_ops)
        if reason is not None:
            omissions.append(LayoutOmission(region, semantic or "unclassified", reason,
                                             input_shapes, output_shapes))
            if semantic in (policy.accelerator_ops | policy.preserving_ops | policy.residual_ops
                            | policy.view_ops | policy.fixed_layout_ops):
                capture_refusals.append(LayoutRefusal(
                    region, "classified_activation_region_omitted",
                    f"{semantic!r} was classified but omitted: {reason}",
                    ("provenance-region physical-layout capture",), True))
            continue
        if semantic in policy.accelerator_ops and (not inputs or not outputs):
            omissions.append(LayoutOmission(
                region, semantic, "accelerator_activation_ports_incomplete",
                input_shapes, output_shapes))
            capture_refusals.append(LayoutRefusal(
                region, "classified_activation_region_omitted",
                "accelerator activation ports are incomplete",
                ("provenance-region physical-layout capture",), True))
            continue
        if semantic in policy.residual_ops and (len(inputs) < 2 or not outputs):
            omissions.append(LayoutOmission(
                region, semantic, "residual_activation_ports_incomplete",
                input_shapes, output_shapes))
            capture_refusals.append(LayoutRefusal(
                region, "classified_activation_region_omitted",
                "residual activation ports are incomplete",
                ("provenance-region physical-layout capture",), True))
            continue
        if semantic in policy.preserving_ops and (not inputs or not outputs):
            omissions.append(LayoutOmission(
                region, semantic, "preserving_activation_ports_incomplete",
                input_shapes, output_shapes))
            capture_refusals.append(LayoutRefusal(
                region, "classified_activation_region_omitted",
                "layout-preserving activation ports are incomplete",
                ("provenance-region physical-layout capture",), True))
            continue
        input_names = tuple(add_value(value) for value in inputs)
        output_names = tuple(add_value(value) for value in outputs)
        safe_view = semantic in policy.view_ops and _safe_view(inputs, outputs)
        if semantic in policy.accelerator_ops:
            kind, couple = "accelerator", bool(input_names and output_names)
        elif semantic in policy.residual_ops:
            kind, couple = "residual", len(input_names) >= 2 and bool(output_names)
        elif semantic in policy.preserving_ops:
            kind, couple = "layout_preserving", bool(input_names and output_names)
        elif safe_view:
            kind, couple = "layout_preserving_view", True
        elif semantic in policy.view_ops:
            kind, couple = "unsafe_view_boundary", False
        elif semantic in policy.fixed_layout_ops:
            kind, couple = "fixed_boundary", False
        else:
            kind, couple = "unknown_boundary", False

        sample = graph_values[(input_names or output_names)[0]]
        available = tuple(encoding.id for encoding in sample.encodings)
        if couple:
            options = tuple(OperatorEncoding(
                f"{kind}:{identifier}", identifier,
                1 if kind == "accelerator" and identifier == policy.preferred_accelerator_encoding
                else 0,
                provenance=("capture policy classified layout-polymorphic activation ports",),
            ) for identifier in available)
        else:
            options = (OperatorEncoding(
                f"{kind}:{policy.canonical_encoding}", policy.canonical_encoding,
                provenance=("capture policy classified a fixed activation-layout boundary",),
            ),)
        rows.append((min(position[op] for op in ops), LayoutOp(
            region, kind, input_names, output_names, couple, options,
            (f"semantic operation {semantic or 'unclassified'}",),
        )))
        included[kind] += 1

    # An unattributed operation touching a proved activation cannot be assigned to a provenance
    # region soundly.  Record a blocking refusal instead of inventing ownership or making it
    # transparent; payload operations often interleave with their attributed source region.
    known_values = {value for value, name in names.items() if name in graph_values}
    for op in block.ops:
        if op in region_of or op.name in {
                "func.return", "arith.constant", "tensor.empty", "tensor.splat"}:
            continue
        touching = [value for value in (*op.operands, *op.results) if value in known_values]
        if not touching:
            continue
        subject = f"unattributed_{position[op]}_{op.name}"
        shapes = tuple(_shape(value) for value in touching)
        omissions.append(LayoutOmission(
            subject, op.name, "unattributed_activation_operation", shapes, ()))
        capture_refusals.append(LayoutRefusal(
            subject, "unattributed_activation_layout_unproven",
            "activation operation has no provenance-region ownership or layout contract",
            ("top-level source operation census",), True))

    if not graph_values:
        refusal = LayoutRefusal(
            "capture", "no_activation_of_required_rank",
            f"no provenance region exposes a proved rank-{policy.activation_rank} activation",
            blocks_plan=True)
        census = (
            ("activation_rank", policy.activation_rank),
            ("included_op_counts", {}),
            ("inferred_interior_operations", inferred_interior_operations),
            ("omission_reason_counts", dict(sorted(Counter(
                item.reason for item in omissions).items()))),
            ("provenance_regions", len(regions)),
            ("semantic_region_counts", dict(sorted(semantic_counts.items()))),
        )
        return LayoutCaptureResult("not_applicable", None, census, tuple(omissions), (refusal,))

    graph_inputs: list[LayoutOp] = []
    for index, argument in enumerate(block.args):
        name = names[argument]
        if name not in graph_values:
            continue
        graph_inputs.append(LayoutOp(
            f"graph_input_{index}", "graph_input", (), (name,), False,
            (OperatorEncoding(
                f"graph_input:{policy.canonical_encoding}", policy.canonical_encoding,
                provenance=("entry ABI physical encoding",),
            ),),
            ("entry ABI physical encoding",),
        ))
        included["graph_input"] += 1
    graph_outputs: list[LayoutOp] = []
    returns = [op for op in block.ops if op.name == "func.return"]
    if returns:
        for index, value in enumerate(returns[0].operands):
            name = names.get(value)
            if name not in graph_values:
                continue
            graph_outputs.append(LayoutOp(
                f"graph_output_{index}", "graph_output", (name,), (), False,
                (OperatorEncoding(
                    f"graph_output:{policy.canonical_encoding}", policy.canonical_encoding,
                    provenance=("result ABI physical encoding",),
                ),),
                ("result ABI physical encoding",),
            ))
            included["graph_output"] += 1

    ordered_ops = tuple((*graph_inputs, *_topological_rows(rows), *graph_outputs))
    graph = LayoutGraph(
        tuple(graph_values[name] for name in sorted(graph_values)), ordered_ops,
        policy.capabilities, policy.conversions, policy.capacity_limits,
        tuple(capture_refusals),
        max_assignments=policy.max_assignments,
    )
    omission_counts = Counter(item.reason for item in omissions)
    census = (
        ("activation_rank", policy.activation_rank),
        ("included_op_counts", dict(sorted(included.items()))),
        ("inferred_interior_operations", inferred_interior_operations),
        ("layout_graph_ops", len(ordered_ops)),
        ("layout_graph_values", len(graph_values)),
        ("omission_reason_counts", dict(sorted(omission_counts.items()))),
        ("provenance_regions", len(regions)),
        ("semantic_region_counts", dict(sorted(semantic_counts.items()))),
    )
    status = "captured_with_refusals" if capture_refusals else "captured"
    return LayoutCaptureResult(
        status, graph, census, tuple(omissions), tuple(capture_refusals))
