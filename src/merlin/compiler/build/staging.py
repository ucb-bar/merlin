"""Plan a bench-staged on-chip schedule when one invocation cannot fit.

The bench host keeps every application tensor and the unchanged parameter blob.
Each step places one selected region, or one slice of a region, in the window
through the existing single-call placement and provider facts. A slice changes
no arithmetic: an output column of a prequantized contraction reads only its
weight row and exponent, a batch element of a batched one reads only its own
operands, and a column of an index-mapped elementwise region reads only its
operand columns, and a leading-row chunk of any other region reads only its
own rows of every split operand. A band changes no arithmetic either: an
output row of a prequantized contraction reads only its activation row, a band
of a static activation quantizer only its own input elements, and a band of any
output axis of an index-mapped region (convolution patch gathers, window
reductions such as pooling, layout and elementwise regions) only the operand
rows its maps name, including the halo rows its stride, kernel, and dilation
need. A tile of a contraction is one output-column slice of one row band,
used only when no single-axis option places. K order is never split. Staging
chooses, among admitted options, the fewest device launches, then the fewest
placed bytes. All slices of a
region bind one compiled program, and each step package is published as an
ordinary generated single-call application. ELF linking and execution are not
rendered here.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import replace
from hashlib import sha256
from math import prod
from pathlib import Path
from typing import NamedTuple

from merlin.compiler.lower.c.runtime import DT_BYTES
from merlin.compiler.lower.c.layout import bounded_dense_read_map
from merlin.compiler.semantic.linalg.source_compositions import row_coupled_axes
from merlin.compiler.semantic.target_binding import NativeCompilationError


class Split(NamedTuple):
    """One split that keeps source arithmetic; axes maps each split node to (axis, stride, footprint)."""

    name: str
    axis: int
    extent: int
    axes: dict


class LoopBound(NamedTuple):
    """A later-trip region that can bind the body of the earlier region at index; nodes maps its nodes to ours."""

    index: int
    nodes: dict


def _alternative(item, nodes=None):
    """A selected alternative without its region identity, buffers renamed through nodes, in buffer order.

    Port order is not compared: a bound region binds every port by name through the node correspondence.
    """
    row = item.to_dict()
    del row["id"], row["nodes"]
    for side in ("inputs", "outputs"):
        for port in row[side]:
            name, dot, part = port["buffer"].partition(".")
            port["buffer"] = (nodes or {}).get(name, name) + dot + part
        row[side].sort(key=lambda port: port["buffer"])
    return json.dumps(row, sort_keys=True)


def loop_bindings(outlined, plan, parameter_attributes):
    """Return the region loop and, for each later-trip region that can bind an earlier body, its LoopBound.

    The loop is the longest tandem repeat of regions equal up to the target's
    declared parameter attributes. A region in a later trip binds the first
    earlier region at its body position whose request is exactly equal (no
    attribute ignored) and whose selected alternative matches under the node
    correspondence. Any other region, for example one with a different
    calibrated value, compiles itself.
    """
    from merlin.compiler.schedule.region_loop import analyze_region_loop, correspondence, structural_signature
    from merlin.compiler.transforms.dense_views import IMPLEMENTATION as VIEW

    records = [region.record() for _, region in outlined.regions]
    loop = analyze_region_loop(records, parameter_attributes=parameter_attributes)
    if loop is None:
        return None, {}
    offset, period, trips = loop["offset"], loop["period"], loop["trip_count"]
    selected = {item.id: item for item in plan.selected}
    exact = [structural_signature(row) for row in records]
    representatives, bound = [[offset + j] for j in range(period)], {}
    for index in range(offset + period, offset + period * trips):
        key = outlined.regions[index][0]
        if selected[key].implementation == VIEW:
            continue
        for candidate in representatives[(index - offset) % period]:
            if exact[candidate] == exact[index]:
                nodes = correspondence(records[candidate], records[index])
                if _alternative(selected[outlined.regions[candidate][0]], nodes) == _alternative(selected[key]):
                    bound[index] = LoopBound(candidate, nodes)
                    break
        else:
            representatives[(index - offset) % period].append(index)
    return dict(region_offset=offset, region_period=period, trip_count=trips), bound


def _bytes(node):
    return prod(node.type.shape) * DT_BYTES[node.type.dtype]


def _sources(region):
    return ", ".join(node.id for node in region.nodes if node.effect == "pure")


def _refuse(key, region, message, status="resource_limit"):
    raise NativeCompilationError(status, f"application region {key} (source: {_sources(region)}): {message}")


def _single(region):
    pure = [node for node in region.nodes if node.effect == "pure"]
    if (len(pure) != 1 or region.outputs != (pure[0].id,) or region.physical_extents or region.boundary_layouts
            or region.output_layouts):
        return None
    return pure[0]


def _prequantized(region):
    """(contraction, (lhs, lhs exponent, rhs, rhs exponent)) of a transposed prequantized contraction."""
    node = _single(region)
    if node is None:
        return None
    attrs = dict(node.attrs)
    if (attrs.get("semantic_kind") != "prequantized_contraction" or attrs.get("rhs_transposed") != "true"
            or len(node.inputs) != 4 or node.index_maps or len(set(node.inputs)) != 4):
        return None
    operands = tuple(region.node(key) for key in node.inputs)
    return (node, operands) if all(item.effect == "input" for item in operands) else None


def output_columns(region):
    """Return (contraction, weight, weight exponent, N, K) when slices keep source arithmetic."""
    found = _prequantized(region)
    if found is None:
        return None
    node, (lhs, lhs_exponent, rhs, rhs_exponent) = found
    if len(rhs.type.shape) != 2:
        return None
    n, k = rhs.type.shape
    rows = node.type.shape[:-1]
    if (not node.type.shape or node.type.shape[-1] != n or lhs.type.shape != (*rows, k)
            or rhs_exponent.type.shape not in {(n,), (n, 1)} or lhs_exponent.type.shape not in {(), (*rows, 1)}):
        return None
    return node, rhs, rhs_exponent, n, k


def _batch(region):
    """Outermost nontrivial batch axis of a batched prequantized contraction.

    Each batch element reads only its own operands and exponents, with the
    full K order unchanged; a rank-0 exponent is shared by every slice.
    """
    found = _prequantized(region)
    if found is None:
        return None
    node, (lhs, lhs_exponent, rhs, rhs_exponent) = found
    shape = node.type.shape
    batch, rank = shape[:-2], len(shape)
    if rank < 3 or len(rhs.type.shape) != rank or len(lhs.type.shape) != rank:
        return None
    m, n, k = shape[-2], shape[-1], rhs.type.shape[-1]
    if lhs.type.shape != (*batch, m, k) or rhs.type.shape != (*batch, n, k):
        return None
    axis = next((i for i, extent in enumerate(batch) if extent > 1), None)
    if axis is None:
        return None
    axes = {node.id: axis, lhs.id: axis, rhs.id: axis}
    for exponent in (lhs_exponent, rhs_exponent):
        if exponent.type.shape == ():
            continue
        if len(exponent.type.shape) != rank or exponent.type.shape[:-2] != batch:
            return None
        axes[exponent.id] = axis
    return "batch", batch[axis], axes


def _column_axis(index_map, rank):
    """The one operand axis that reads loop column j at j, with no other loop axis or offset; else None."""
    reads = [i for i, row in enumerate(index_map.coefficients) if row[-1]]
    if index_map.loop_rank != rank or len(reads) != 1:
        return None
    axis = reads[0]
    unit = tuple(int(j == rank - 1) for j in range(rank))
    return axis if index_map.coefficients[axis] == unit and not index_map.offsets[axis] else None


def _broadcast_columns(index_map, rank):
    return index_map.loop_rank == rank and not any(row[-1] for row in index_map.coefficients)


def _elementwise_columns(region):
    """Columns of an index-mapped region whose loops are all output axes.

    Every result element reads only the operand elements its maps name, so a
    column chunk reads its own operand columns and any column-broadcast operand
    whole. Constants that vary by column, reductions and permutations of the
    column axis are not split.
    """
    pure = [node for node in region.nodes if node.effect == "pure"]
    if not pure or region.physical_extents or region.boundary_layouts or region.output_layouts:
        return None
    shape = pure[0].type.shape
    rank = len(shape)
    if rank == 0 or shape[-1] < 2:
        return None
    axes = {}
    for node in pure:
        maps = node.index_maps
        if node.type.shape != shape or not maps or maps[-1].coefficients != tuple(
                tuple(int(i == j) for j in range(rank)) for i in range(rank)) or any(maps[-1].offsets):
            return None
        axes[node.id] = rank - 1
        for key, index_map in zip(node.inputs, maps[:-1], strict=True):
            operand = region.node(key)
            # Each operand is sliced along its own axis that its map ties to the loop column.
            axis = _column_axis(index_map, rank)
            if axis is not None and operand.type.shape[axis] == shape[-1]:
                if operand.effect == "constant" or axes.get(key, axis) != axis:
                    return None
                axes[key] = axis
            elif not _broadcast_columns(index_map, rank) or key in axes:
                return None
    # An operand read both whole and by column cannot be split.
    for node in pure:
        for key, index_map in zip(node.inputs, node.index_maps[:-1]):
            if key in axes and _broadcast_columns(index_map, rank):
                return None
    return "columns", shape[-1], axes


def split_rule(region):
    """(axis name, extent, {node id: split axis}) of the one split that keeps source arithmetic, or None."""
    found = output_columns(region)
    if found is not None:
        node, rhs, exponent, n, _ = found
        return "output_columns", n, {rhs.id: 0, exponent.id: 0, node.id: len(node.type.shape) - 1}
    return _batch(region) or _elementwise_columns(region)


def slice_request(region, width):
    """The shared slice request: the split axis becomes width, every identity and other port is unchanged."""
    _, extent, axes = split_rule(region)
    if type(width) is not int or not 0 < width < extent or extent % width:
        raise ValueError("slice width must divide the split extent")

    def narrowed(item):
        shape = list(item.type.shape)
        shape[axes[item.id]] = width
        return replace(item, type=replace(item.type, shape=tuple(shape)))
    return replace(region, nodes=tuple(narrowed(item) if item.id in axes else item for item in region.nodes))


def _axis_range(node, axis, start, width):
    shape, element = node.type.shape, DT_BYTES[node.type.dtype]
    inner = prod(shape[axis + 1:]) * element
    return dict(offset=start * inner, bytes=width * inner, count=prod(shape[:axis]),
                stride=shape[axis] * inner if axis else 0)


def _slice_ranges(region, width, index):
    """Byte ranges of slice index within each dense application tensor of the region."""
    _, _, axes = split_rule(region)
    return {key: _axis_range(region.node(key), axis, index * width, width) for key, axis in axes.items()
            if region.node(key).effect == "input" or key in region.outputs}


def _whole(node):
    return dict(offset=0, bytes=_bytes(node), count=1, stride=0)


def _summary(key, request, lowered, label):
    return {"id": key, "source_request_digest": request.digest(), "placement": None, "context_index": 0,
            "description": lowered.description, "native_programs": list(lowered.programs),
            "inputs": list(lowered.inputs), "outputs": list(lowered.outputs), "package": label}


def _ports(key, region, lowered):
    # Same port rule as whole-application rendering: only constants may stay embedded.
    if (set(lowered.outputs) != set(region.outputs) or len(lowered.outputs) != len(region.outputs)
            or any(region.node(port).effect not in {"input", "constant"} for port in lowered.inputs)
            or any(node.effect == "input" and node.id not in lowered.inputs for node in region.nodes)):
        raise ValueError(f"compiled C ports of {key} omit a runtime source value or a required result")


def _declared_columns(adapter, summary, native_root, rhs):
    """Columns per provider-declared refill group of the weight, or None when none are declared.

    A provider that refuses the whole region for size declares no groups; each
    width is then admitted by compiling and placing it.
    """
    try:
        provider = adapter.deployment_requirements([summary], native_root) or {}
    except NativeCompilationError as error:
        if error.status != "resource_limit":
            raise
        return None
    groups = [row for row in provider.get("refill_groups", []) if row.get("region_operand") == rhs.id]
    if not groups:
        return None
    row = _bytes(rhs) // rhs.type.shape[0]
    widths = [group["bytes"] // row for group in groups[0]["groups"]]
    if any(group["bytes"] % row for group in groups[0]["groups"]) or any(w != widths[0] for w in widths[:-1]):
        raise NativeCompilationError("unsupported_semantics", "provider output-column groups are not uniform rows")
    return widths[0]


def _widths(region, capacity, group):
    """Dense-feasible slice widths, widest first; group-aligned when the provider declares groups."""
    _, extent, axes = split_rule(region)
    fixed = sum(_bytes(item) for item in region.nodes if item.effect == "input" and item.id not in axes)
    per_unit = sum(_bytes(region.node(key)) for key in axes
                   if region.node(key).effect == "input" or key in region.outputs) // extent
    return [w for w in range(extent - 1, 0, -1)
            if not extent % w and (group is None or not w % group) and fixed + w * per_unit <= capacity]


def _step_application(key, alternative, request, lowered, label, adapter, capacity, native_root):
    """Render one step package as an ordinary single-call on-chip application.

    Its generated driver, deployment and handoff come from the existing
    single-invocation renderer, and its deployment is the step's placement;
    an overflow raises that renderer's named refusal. Instruction image paths
    are relative to the staged publication root.
    """
    from merlin.compiler.partition.global_plan import GlobalPlan

    from .regions import _render

    pure = sum(node.effect == "pure" for node in request.nodes)
    plan = GlobalPlan((replace(alternative, id=key, nodes=tuple(range(pure))),), (), alternative.cycles)
    # The application name differs from the region package's own C symbols.
    _, _, interface, _, files = _render(request, adapter, "step_" + label, capacity, [lowered],
                                        native_root=native_root, plan=plan)
    return interface["deployment"], {f"steps/{label}/{member}": data for member, data in files.items()}


def _row_chunk(shape, count):
    """(axis, shape) of one of count equal row-major chunks when each is a box, else None."""
    before = 1
    for axis, extent in enumerate(shape):
        if not count % before and not (before * extent) % count:
            return axis, (1,) * axis + (before * extent // count, *shape[axis + 1:])
        before *= extent
    return None


def _quantizer_groups(shape, attrs):
    """Row counts of a runtime exponent's declared reduction groups, row-major, or None.

    Rows are the source's leading indices; a head is its last two axes. Block
    groups hold the declared rows over all rows and head blocks within each
    head, the last one short when rows does not divide.
    """
    reduce, size = attrs.get("reduce"), attrs.get("rows")
    total = prod(shape[:-1])
    head = shape[-2] if len(shape) > 1 else None
    if type(size) is not int or size < 1 or not shape:
        return None
    if reduce == "tensor":
        return (total,)
    if reduce == "token":
        return (1,) * total
    if reduce == "head" and head:
        return (head,) * (total // head)
    if reduce == "block":
        return (size,) * (total // size) + ((total % size,) if total % size else ())
    if reduce == "head_block" and head:
        return ((size,) * (head // size) + ((head % size,) if head % size else ())) * (total // head)
    return None


def _broadcast_role(item, rank, lead):
    """Split when an operand keeps the leading rows, whole when it broadcasts over them, else None."""
    if len(item.type.shape) > rank:
        return None
    aligned = (1,) * (rank - len(item.type.shape)) + tuple(item.type.shape)
    if aligned[:len(lead)] == lead:
        return "split"
    return "whole" if all(extent == 1 for extent in aligned[:len(lead)]) else None


def _row_uses(region, node, count):
    """(tensor, role) pairs under which count leading-row chunks keep node's source arithmetic, or None."""
    attrs, inputs = dict(node.attrs), [region.node(key) for key in node.inputs]
    if attrs.get("semantic_kind") == "dynamic_activation_quantizer" and not node.index_maps and len(inputs) == 1:
        (source,) = inputs
        shape = tuple(source.type.shape)
        chunk = _row_chunk(shape, count)
        results = [item for item in region.nodes if item.effect == "pure" and item.op.rsplit(".", 1)[0]
                   == node.op.rsplit(".", 1)[0] and item.inputs == node.inputs and item.attrs == node.attrs]
        if chunk is None or len(results) != 2:
            return None
        # A runtime exponent reduces each declared group of rows: chunks must
        # hold whole rows and repeat the source's groups, in order.
        groups = _quantizer_groups(shape, attrs)
        if (chunk[1][-1:] != shape[-1:] or groups is None
                or _quantizer_groups(chunk[1], attrs) * count != groups
                or any(_row_chunk(item.type.shape, count) is None for item in results)):
            return None
        return [(node.id, "split"), (source.id, "split")]
    chunk = _row_chunk(node.type.shape, count)
    if chunk is None:
        return None
    axis, rank = chunk[0], len(node.type.shape)
    lead = tuple(node.type.shape[:axis + 1])
    uses = [(node.id, "split")]
    if node.op == "reshape" and not node.index_maps and len(inputs) == 1:
        return uses + [(inputs[0].id, "split")] if _row_chunk(inputs[0].type.shape, count) else None
    if node.op == "concat" and not node.index_maps:
        # Concatenation along a later axis keeps every leading row of each operand.
        if node.attrs != (("axis", attrs.get("axis")),) or type(attrs["axis"]) is not int or attrs["axis"] <= axis or any(
                len(item.type.shape) != rank or tuple(item.type.shape[:axis + 1]) != lead for item in inputs):
            return None
        return uses + [(item.id, "split") for item in inputs]
    if attrs.get("semantic_kind") == "prequantized_contraction" and not node.index_maps:
        # Batch axes precede the two matrix axes; code and exponent operands
        # broadcast over them, and each batch keeps its whole K order.
        if axis >= rank - 2:
            return None
        roles = [_broadcast_role(item, rank, lead) for item in inputs]
        return None if None in roles else uses + [(item.id, role) for item, role in zip(inputs, roles)]
    if len(node.index_maps) != len(inputs) + 1 or not (
            "numeric_semantics" in attrs or (len(inputs) == 1 and bounded_dense_read_map(inputs[0], node))):
        return None
    *maps, result = node.index_maps
    rows = tuple(tuple(int(i == j) for j in range(result.loop_rank)) for i in range(axis + 1))
    if result.coefficients[:axis + 1] != rows or any(result.offsets[:axis + 1]):
        return None
    for item, mapping in zip(inputs, maps, strict=True):
        if not any(any(row[:axis + 1]) for row in mapping.coefficients):
            uses.append((item.id, "whole"))
        elif (mapping.coefficients[:axis + 1] == rows and not any(mapping.offsets[:axis + 1])
                and tuple(item.type.shape[:axis + 1]) == lead
                and not any(any(row[:axis + 1]) for row in mapping.coefficients[axis + 1:])):
            uses.append((item.id, "split"))
        else:
            return None
    return uses


def leading_rows(region, count):
    """Split or whole role of every region tensor when count leading-row chunks keep source arithmetic.

    A chunk is the same contiguous row-major fraction of every split tensor.
    Reshapes keep row-major order, and concatenations along a later axis keep
    each operand's rows. Computations with index maps declare numeric semantics
    or are byte-preserving reads, and their leading result rows are the leading
    loop axes, so a chunk reads only its own rows of a split operand, or a whole
    operand that no split loop axis indexes. Prequantized contractions split
    only batch axes, and runtime activation quantizers only whole declared
    reduction groups.
    Reductions over a later axis keep their order. A split constant must repeat
    across chunks so one compiled program serves every chunk. Returns None
    otherwise.
    """
    if count < 2 or region.physical_extents or region.boundary_layouts or region.output_layouts:
        return None
    roles = {}
    for node in (item for item in region.nodes if item.effect == "pure"):
        uses = _row_uses(region, node, count)
        if uses is None:
            return None
        for key, role in uses:
            if roles.setdefault(key, role) != role:
                return None
    if any(roles.get(key) != "split" for key in region.outputs):
        return None
    for binding in region.constants:
        if roles.get(binding.node_id) == "split":
            data = bytes.fromhex(binding.data_hex)
            if data != data[:len(data) // count] * count:
                return None
    return roles


def row_slice(region, count):
    """The shared chunk request: split tensors keep one leading-row chunk, every identity is unchanged."""
    roles = leading_rows(region, count)
    if roles is None:
        raise ValueError("leading-row count does not keep source arithmetic")
    split = {key for key, role in roles.items() if role == "split"}
    nodes = tuple(replace(node, type=replace(node.type, shape=_row_chunk(node.type.shape, count)[1]))
                  if node.id in split else node for node in region.nodes)
    constants = tuple(replace(row, data_hex=row.data_hex[:len(row.data_hex) // count]) if row.node_id in split
                      else row for row in region.constants)
    return replace(region, nodes=nodes, constants=constants)


def _row_ranges(region, count, index):
    roles = leading_rows(region, count)
    ranges = {}
    for key, role in roles.items():
        if role == "split":
            size = _bytes(region.node(key)) // count
            ranges[key] = dict(offset=index * size, bytes=size, count=1, stride=0)
    return ranges


def leading_row_counts(region, capacity):
    """Dense-feasible leading-row chunk counts, fewest chunks first."""
    outputs = [region.node(key) for key in region.outputs]
    if not outputs or any(not node.type.shape for node in outputs):
        return []
    rows = max(prod(node.type.shape) for node in outputs)
    divisors = {d for i in range(1, int(rows ** 0.5) + 1) if not rows % i for d in (i, rows // i)}
    ports = [node for node in region.nodes if node.effect == "input" or node.id in region.outputs]
    counts = []
    for count in sorted(divisors - {1}):
        roles = leading_rows(region, count)
        if roles is not None and sum(_bytes(node) // count if roles.get(node.id) == "split" else _bytes(node)
                                     for node in ports) <= capacity:
            counts.append(count)
    return counts


def _rows(region):
    """Row axes of a prequantized contraction: each output row reads only its activation row.

    The weight and its exponents are read whole, and a rank-0 activation
    exponent is shared by every band, so K order and scaling are unchanged.
    """
    found = _prequantized(region)
    if found is None:
        return []
    node, (lhs, lhs_exponent, rhs, rhs_exponent) = found
    if len(rhs.type.shape) != 2 or not node.type.shape:
        return []
    rows, (n, k) = node.type.shape[:-1], rhs.type.shape
    if (node.type.shape[-1] != n or lhs.type.shape != (*rows, k) or rhs_exponent.type.shape not in {(n,), (n, 1)}
            or lhs_exponent.type.shape not in {(), (*rows, 1)}):
        return []
    splits = []
    for axis, extent in enumerate(rows):
        if extent > 1:
            axes = {node.id: (axis, 1, 1), lhs.id: (axis, 1, 1)}
            if lhs_exponent.type.shape:
                axes[lhs_exponent.id] = (axis, 1, 1)
            splits.append(Split("rows", axis, extent, axes))
    return splits


def _static_quantizer(region):
    """Bands of any axis of a static activation quantizer.

    Its exponent is declared, not computed from the data, so each code reads
    only its own input element and every band writes the same exponent.
    """
    pure = [node for node in region.nodes if node.effect == "pure"]
    if (not pure or region.physical_extents or region.boundary_layouts or region.output_layouts
            or len({node.inputs for node in pure}) != 1 or len(pure[0].inputs) != 1):
        return []
    source = region.node(pure[0].inputs[0])
    codes = [node for node in pure if node.type.shape]
    if (source.effect != "input" or any(node.index_maps or dict(node.attrs).get("semantic_kind")
                                        != "static_activation_quantizer" or dict(node.attrs).get("mode") != "static"
                                        for node in pure)
            or len(codes) != 1 or codes[0].type.shape != source.type.shape):
        return []
    return [Split("bands", axis, extent, {source.id: (axis, 1, 1), codes[0].id: (axis, 1, 1)})
            for axis, extent in enumerate(source.type.shape) if extent > 1]


def _bands(region, axis):
    """Bands of one output axis of an index-mapped region, with the halo each operand read needs.

    Every pure node writes its loops in order (any further loops reduce) and
    shares the band axis. An operand read whose row on that loop has stride s
    and nonnegative window terms (kernel and dilation) needs s * (w - 1) + f
    rows for a band of w results, where f is the operand extent less
    s * (extent - 1); bands start s * w rows apart. All bands therefore bind
    one program with unchanged maps. Operands that do not vary along the band
    loop are read whole. A negative offset (padding), a band loop in two
    operand axes, an offset past the band's rows, a band-varying constant, a
    strided or haloed intermediate, and an operand read both ways refuse.
    """
    pure = [node for node in region.nodes if node.effect == "pure"]
    if not pure or region.physical_extents or region.boundary_layouts or region.output_layouts:
        return None
    shape = pure[0].type.shape
    rank = len(shape)
    if not 0 <= axis < rank or shape[axis] < 2:
        return None
    extent, band = shape[axis], (axis, 1, 1)
    uses = {node.id: band for node in pure}
    for node in pure:
        maps = node.index_maps
        if not maps or len(node.type.shape) != rank or node.type.shape[axis] != extent or axis in row_coupled_axes(node):
            return None
        loops = maps[-1].loop_rank
        if (maps[-1].coefficients != tuple(tuple(int(i == j) for j in range(loops)) for i in range(rank))
                or any(maps[-1].offsets)
                or any(not any(row[loop] for item in maps[:-1] for row in item.coefficients)
                       for loop in range(rank, loops))):
            return None
        for key, index_map in zip(node.inputs, maps[:-1], strict=True):
            operand = region.node(key)
            rows = [b for b, row in enumerate(index_map.coefficients) if row[axis]]
            use = None
            if len(rows) > 1:
                return None
            if rows:
                (b,) = rows
                row, stride = index_map.coefficients[b], index_map.coefficients[b][axis]
                footprint = operand.type.shape[b] - stride * (extent - 1)
                if stride < 1 or min(row) < 0 or not 0 <= index_map.offsets[b] < footprint:
                    return None
                use = (b, stride, footprint)
                if operand.effect == "constant":
                    return None
            if uses.setdefault(key, use) != use:
                return None
    return Split("bands", axis, extent, {key: use for key, use in uses.items() if use is not None})


def band_rules(region):
    """Band splits that keep source arithmetic: (name, output axis, extent, {node id: (axis, stride, footprint)}).

    Contraction row axes, static activation quantizer axes, and the halo bands
    of every output axis of an index-mapped region. Staging ranks their options
    together with the split rule's and leading-row options.
    """
    pure = [node for node in region.nodes if node.effect == "pure"]
    rules = [*_rows(region), *_static_quantizer(region)]
    rules += [_bands(region, axis) for axis in range(len(pure[0].type.shape) if pure else 0)]
    return [rule for rule in rules if rule is not None]


def _span(use, width):
    """Rows a band of width results reads along the operand axis."""
    _, stride, footprint = use
    return stride * (width - 1) + footprint


def band_request(region, split, width):
    """The shared band request: each split axis takes its band span, every identity and other port is unchanged."""
    if type(width) is not int or not 0 < width < split.extent or split.extent % width:
        raise ValueError("slice width must divide the split extent")

    def narrowed(item):
        use = split.axes[item.id]
        shape = list(item.type.shape)
        shape[use[0]] = _span(use, width)
        return replace(item, type=replace(item.type, shape=tuple(shape)))
    return replace(region, nodes=tuple(narrowed(item) if item.id in split.axes else item for item in region.nodes))


def _ported(region, key):
    return region.node(key).effect == "input" or key in region.outputs


def _band_ranges(region, split, width, index):
    """Byte ranges of band index within each dense application tensor of the region; halos overlap."""
    return {key: _axis_range(region.node(key), use[0], use[1] * index * width, _span(use, width))
            for key, use in split.axes.items() if _ported(region, key)}


def _step_bytes(region, split, width):
    """Dense bytes one band step places: its band of each split port and every other port whole."""
    total = 0
    for item in region.nodes:
        if _ported(region, item.id):
            use = split.axes.get(item.id)
            total += _bytes(item) if use is None else _bytes(item) // item.type.shape[use[0]] * _span(use, width)
    return total


def _band_candidates(region, capacity):
    """(split, width) pairs to try: every band rule in rule order, each with its dense-feasible widths widest first."""
    return [(split, width) for split in band_rules(region) for width in range(split.extent - 1, 0, -1)
            if not split.extent % width and _step_bytes(region, split, width) <= capacity]


def _tile_candidates(region, capacity, group):
    """(width, split, band) options: output-column slices of a contraction, each banded by activation rows.

    A tile reads only its weight rows and exponents and its activation rows,
    with the full K order unchanged. Only row axes after unit axes are banded,
    so each tile writes one strided box of the output.
    """
    candidates, extent = [], split_rule(region)[1]
    for width in (w for w in range(extent - 1, 0, -1) if not extent % w and (group is None or not w % group)):
        request = slice_request(region, width)
        node = output_columns(request)[0]
        for split in _rows(request):
            if prod(node.type.shape[:split.axis]) == 1:
                candidates += [(width, split, band) for band in range(split.extent - 1, 0, -1)
                               if not split.extent % band and _step_bytes(request, split, band) <= capacity]
    return candidates


def _tile_ranges(region, width, split, band, column, row):
    """Byte ranges of one tile: its weight columns, its activation rows, and their output box."""
    ranges = {**_band_ranges(region, split, band, row), **_slice_ranges(region, width, column)}
    node = output_columns(region)[0]
    rows = prod(node.type.shape[split.axis + 1:-1])
    out = ranges[node.id]
    ranges[node.id] = dict(out, offset=out["offset"] + row * band * rows * out["stride"], count=band * rows)
    return ranges


def _option(region, label, kind, option):
    """(request, package label, refusal tag) of one staging option."""
    if kind == "width":
        return slice_request(region, option), f"{label}_w{option}", f"width {option}"
    if kind == "rows":
        return row_slice(region, option), f"{label}_r{option}", f"rows {option}"
    if kind == "tile":
        width, split, band = option
        return (band_request(slice_request(region, width), split, band), f"{label}_w{width}_rows{split.axis}_b{band}",
                f"width {width} rows axis {split.axis} band {band}")
    split, width = option
    return (band_request(region, split, width), f"{label}_{split.name}{split.axis}_w{width}",
            f"{split.name} axis {split.axis} width {width}")


def _placed(region, kind, option):
    """(dense bytes all steps of one staging option place, step count); every step places its request's ports."""
    request = _option(region, "", kind, option)[0]
    count = {"width": lambda: split_rule(region)[1] // option, "rows": lambda: option,
             "band": lambda: option[0].extent // option[1],
             "tile": lambda: split_rule(region)[1] // option[0] * (option[1].extent // option[2])}[kind]()
    return count * sum(_bytes(node) for node in request.nodes if _ported(request, node.id)), count


def _published(kind, option):
    if kind == "tile":
        return option[0], option[1].axis, option[2]
    return (option[0].name, option[0].axis, option[1]) if kind == "band" else option


def _readonly_state(source, state):
    """Authenticate captured read-only state without changing source input roles."""
    if state is None:
        return {}
    state.check(source)
    ports = {}
    for item in state.ports:
        if (item.port.mode != "read_only" or item.port.initialization != "captured"
                or item.initial is None):
            raise NativeCompilationError(
                "unsupported_semantics", "bench staging requires captured read-only state")
        if (not item.node_id or Path(item.node_id).name != item.node_id
                or item.node_id in {".", ".."} or "\\" in item.node_id):
            raise ValueError("state payload needs a direct source identifier")
        ports[item.node_id] = item
    return ports


def stage(source, adapter, outlined, plan, compiled, *, capacity, native_root, compile_slice, parameters=None,
          state=None, widths=None, counts=None, bands=None, tiles=None):
    """Return the ordered steps, regions, shared slice packages, aliases and step application files.

    compile_slice(index, key, region, request, label, root) compiles or replays
    one slice request under root/segments. A compiled entry that is a
    NativeCompilationError is a region whose whole compilation exceeded a
    resource limit; it is split or refused. Its options are the split rule's
    group-aligned widths (or, without a split rule, leading-row counts) and
    every band rule's widths. The chosen option launches the device the fewest
    times, then places the fewest dense bytes over all its steps, then has the
    fewest steps, then comes first in rule order. Launches come from each
    option's own compiled placement, so options are compiled in byte order
    until no untried option could launch fewer times. A contraction's column
    slices banded by activation rows are chosen the same way, only when no
    single-axis option places.
    widths, counts, bands, and tiles fix published choices on replay; rejected
    options are removed before publication.
    """
    from merlin.compiler.transforms.dense_views import IMPLEMENTATION as VIEW

    selected = {item.id: item for item in plan.selected}
    parameter_entries = dict(zip(parameters.node_ids, parameters.entries, strict=True)) if parameters else {}
    state_ports = _readonly_state(source, state)
    if set(parameter_entries) & set(state_ports):
        raise ValueError("a source input cannot be both a parameter and state")
    inputs = [key for key, _ in source.input_storages if key not in parameter_entries and key not in state_ports]
    producers, aliases, steps, regions, packages, applications = {}, {}, [], [], [], {}

    def origin(key, window):
        key = aliases.get(key, key)
        node = source.node(key)
        if key in parameter_entries:
            entry = parameter_entries[key]
            # A band of a parameter is the same gathered runs as any other band, offset into the blob.
            return dict(kind="parameter", path="handoff/parameters.bin", value=key,
                        **{**window, "offset": entry.offset + window["offset"]})
        if key in state_ports:
            item = state_ports[key]
            return dict(kind="parameter", path=f"handoff/state/{key}.bin", value=key,
                        role="read_only_state", state=dict(name=item.port.name, mode="read_only",
                            initialization="captured", source_input_index=item.input_index), **window)
        if node.effect == "input":
            index = inputs.index(key)
            return dict(kind="application_input", value=key, index=index, path=f"inputs/input_{index:03d}.bin", **window)
        return dict(kind="step_output", value=key, producer_steps=producers[key], **window)

    def step(key, index, region, request, lowered, label, placement, ranges, slice_record):
        loads = [dict(port=row["id"], address=row["address"], bytes=row["bytes"],
                      source=origin(row["id"], ranges.get(row["id"], _whole(region.node(row["id"])))))
                 for row in placement["inputs"]]
        # An output every slice writes whole (a declared exponent) is read back once, from the first slice.
        reads = [dict(port=row["id"], address=row["address"], bytes=row["bytes"],
                      destination=dict(value=row["id"], **ranges.get(row["id"], _whole(region.node(row["id"])))))
                 for row in placement["outputs"]
                 if slice_record is None or row["id"] in ranges or slice_record["index"] == 0]
        call = placement["calls"][0]
        record = dict(step=len(steps), region=key, context_index=index, placement=call["placement"], package=label,
                      request_digest=request.digest(), native_programs=call["native_programs"],
                      images=[dict(path=row["path"], sha256=row["sha256"], bytes=row["bytes"])
                              for row in placement["provider"]["images"]],
                      launches=call["launches_per_invocation"], slice=slice_record, loads=loads, reads=reads,
                      allocations=placement["allocations"], reservations=placement["provider"]["reservations"],
                      window_required_bytes=placement["window_required_bytes"],
                      window_remaining_bytes=placement["window_remaining_bytes"])
        steps.append(record)
        return record

    for index, ((key, region), lowered) in enumerate(zip(outlined.regions, compiled, strict=True)):
        label = f"region_{index:03d}"
        summary = None
        if isinstance(lowered, NativeCompilationError):
            whole_error = lowered.reason
        elif selected[key].implementation == VIEW:
            (parent,), (child,) = lowered.inputs, lowered.outputs
            aliases[child] = aliases.get(parent, parent)
            regions.append(dict(id=key, context_index=index, staging="alias", value=child, root=aliases[child]))
            continue
        else:
            _ports(key, region, lowered)
            summary = {**_summary(key, region, lowered, label), "placement": selected[key].placement}
            try:
                placement, applications[label] = _step_application(key, selected[key], region, lowered, label,
                                                                   adapter, capacity, native_root)
            except NativeCompilationError as error:
                if error.status != "resource_limit":
                    _refuse(key, region, error.reason, error.status)
                whole_error = error.reason
            else:
                first = step(key, index, region, region, lowered, label, placement, {}, None)
                regions.append(dict(id=key, context_index=index, staging="whole", steps=[first["step"]]))
                for port in lowered.outputs:
                    producers[port] = [first["step"]]
                continue
        rule = split_rule(region)
        group = None
        if rule is not None:
            axis, extent, _ = rule
            contraction = output_columns(region)
            group = (None if contraction is None or summary is None
                     else _declared_columns(adapter, summary, native_root, contraction[1]))
            options = [("width", width) for width in _widths(region, capacity, group)]
        else:
            options = [("rows", count) for count in leading_row_counts(region, capacity)]
        options += [("band", option) for option in _band_candidates(region, capacity)]
        # Every rule's options are tried in order of placed bytes, then steps, then rule order.
        options.sort(key=lambda row: _placed(region, *row))
        # Column-row tiles of a contraction, in the same order, only after every single-axis option.
        if rule is not None and contraction is not None:
            options += sorted((("tile", option) for option in _tile_candidates(region, capacity, group)),
                              key=lambda row: _placed(region, *row))
        published = {"width": widths, "rows": counts, "band": bands, "tile": tiles}
        replay = any(value is not None for value in published.values())
        if replay:
            kind = next((kind for kind, value in published.items() if key in (value or {})),
                        "width" if rule else "rows")
            options = [(k, option) for k, option in options
                       if k == kind and key in (published[k] or {}) and published[k][key] == _published(k, option)]
            if not options:
                name = {"width": "slice width", "rows": "leading-row count", "band": "band", "tile": "tile"}[kind]
                raise ValueError(f"published {name} differs from the declared split rules")
        elif not options:
            _refuse(key, region, whole_error)
        # Every step of a device region launches at least once, so an option's step count bounds its launches.
        floors = [_placed(region, *row)[1] if selected[key].placement != "host" else 0 for row in options]
        best, reasons = None, [whole_error]
        for position, (kind, option) in enumerate(options):
            # Tiles come last and are tried only when no single-axis option placed, so they bound only each other.
            if best is not None and (kind == "tile") != (best[1][0] == "tile"):
                break
            if best is not None and best[0] <= min(floor for floor, row in zip(floors[position:], options[position:])
                                                   if (row[0] == "tile") == (kind == "tile")):
                break
            request, slice_label, tag = _option(region, label, kind, option)
            try:
                # A slice can fail at compilation or at placement; both try the next option.
                shared = compile_slice(index, key, region, request, slice_label, Path(native_root))
                _ports(key, request, shared)
                slice_summary = {**_summary(key, request, shared, slice_label), "placement": selected[key].placement}
                placement, application = _step_application(key, selected[key], request, shared, slice_label,
                                                           adapter, capacity, native_root)
            except NativeCompilationError as error:
                # Any named refusal of one option (size, or a shape the provider cannot cover) rejects
                # only that option; a published option on replay must compile as published.
                if replay:
                    _refuse(key, region, f"{tag}: {error.reason}", error.status)
                # Only the chosen option is published.
                shutil.rmtree(Path(native_root) / "segments" / slice_label, ignore_errors=True)
                reasons.append(f"{tag}: {error.reason}")
                continue
            launches = _placed(region, kind, option)[1] * placement["calls"][0]["launches_per_invocation"]
            # Only the chosen option is published; an earlier one with no more launches keeps its place.
            if best is not None and best[0] <= launches:
                shutil.rmtree(Path(native_root) / "segments" / slice_label, ignore_errors=True)
                continue
            if best is not None:
                shutil.rmtree(Path(native_root) / "segments" / best[3], ignore_errors=True)
                del applications[best[3]]
            best = (launches, (kind, option), request, slice_label, shared, slice_summary, placement)
            applications[slice_label] = application
        if best is None:
            names = dict.fromkeys([axis if rule else "leading-row", *(split.name for split in band_rules(region))])
            _refuse(key, region, f"no provider-admitted {', '.join(names)} slice fits the declared window; "
                    + "; ".join(reasons))
        _, (kind, option), request, slice_label, shared, slice_summary, placement = best
        packages.append((slice_label, request, shared, slice_summary))
        if kind == "width":
            count = extent // option
            records = [step(key, index, region, request, shared, slice_label, placement,
                            _slice_ranges(region, option, s), dict(index=s, count=count, axis=axis,
                                                                   range=[s * option, (s + 1) * option]))
                       for s in range(count)]
            regions.append(dict(id=key, context_index=index, staging=axis, extent=extent, width=option,
                                slices=count, provider_group_columns=group, shared_package=slice_label,
                                shared_request_digest=request.digest(), steps=[row["step"] for row in records],
                                unsplit_refusal=whole_error))
        elif kind == "tile":
            width, split, band = option
            across, down = extent // width, split.extent // band
            records = [step(key, index, region, request, shared, slice_label, placement,
                            _tile_ranges(region, width, split, band, s // down, s % down),
                            dict(index=s, count=across * down, rule="tiles", axis=split.axis,
                                 columns=[s // down * width, (s // down + 1) * width],
                                 range=[s % down * band, (s % down + 1) * band]))
                       for s in range(across * down)]
            regions.append(dict(id=key, context_index=index, staging="tiles", extent=extent, columns=width,
                                row_axis=split.axis, row_extent=split.extent, rows=band, slices=across * down,
                                provider_group_columns=group, shared_package=slice_label,
                                shared_request_digest=request.digest(), steps=[row["step"] for row in records],
                                unsplit_refusal=whole_error))
        elif kind == "rows":
            records = [step(key, index, region, request, shared, slice_label, placement,
                            _row_ranges(region, option, s), dict(index=s, count=option, axis="leading_rows"))
                       for s in range(option)]
            regions.append(dict(id=key, context_index=index, staging="leading_rows", slices=option,
                                shared_package=slice_label, shared_request_digest=request.digest(),
                                steps=[row["step"] for row in records], unsplit_refusal=whole_error))
        else:
            split, width = option
            count = split.extent // width
            records = [step(key, index, region, request, shared, slice_label, placement,
                            _band_ranges(region, split, width, s),
                            dict(index=s, count=count, rule=split.name, axis=split.axis,
                                 range=[s * width, (s + 1) * width]))
                       for s in range(count)]
            regions.append(dict(id=key, context_index=index, staging=split.name, axis=split.axis,
                                extent=split.extent, width=width, slices=count, shared_package=slice_label,
                                shared_request_digest=request.digest(), steps=[row["step"] for row in records],
                                unsplit_refusal=whole_error))
        for port in region.outputs:
            producers[port] = [row["step"] for row in records if port in {read["port"] for read in row["reads"]}]
    return steps, regions, packages, aliases, applications


def schedule_record(source, steps, regions, aliases, payload=None, state=None):
    """Existing deployment family: the ordered steps, their byte ranges and transfer totals.

    payload is the published parameter blob; each parameter load records the
    SHA256 of its exact source runs, concatenated, for authentication before upload.
    """
    state_ports = _readonly_state(source, state)
    state_paths = {f"handoff/state/{key}.bin": item.initial for key, item in state_ports.items()}
    for row in steps:
        for load in row["loads"]:
            item = load["source"]
            if item["kind"] == "parameter":
                # The hash covers every gathered run, in upload order.
                digest = sha256()
                for run in range(item["count"]):
                    offset = item["offset"] + run * item["stride"]
                    if item["path"] in state_paths:
                        data = state_paths[item["path"]][offset:offset + item["bytes"]]
                    else:
                        payload.seek(offset)
                        data = payload.read(item["bytes"])
                    if len(data) != item["bytes"]:
                        raise ValueError("parameter or state source range exceeds its authenticated payload")
                    digest.update(data)
                item["sha256"] = digest.hexdigest()
    producers = {read["destination"]["value"]: [] for row in steps for read in row["reads"]}
    for row in steps:
        for read in row["reads"]:
            producers[read["destination"]["value"]].append(row["step"])
    outputs = [dict(value=key, root=aliases.get(key, key), producer_steps=producers.get(aliases.get(key, key), []))
               for key in source.outputs]
    loads = [load for row in steps for load in row["loads"]]
    reads = [read for row in steps for read in row["reads"]]
    shared = {}
    for row in steps:
        if shared.setdefault(row["package"], row["images"]) != row["images"]:
            raise ValueError("steps of one shared package bind different images")
    return {
        "schema": "merlin.onchip_deployment.v3", "mode": "bench_staged", "request_digest": source.digest(),
        "target_identity": source.target_identity, "steps": steps, "regions": regions, "application_outputs": outputs,
        "storage_policy": "bench host holds every application tensor and the unchanged parameter blob; "
                          "each step places only its own ports in the window",
        "totals": {
            "steps": len(steps), "device_steps": sum(row["placement"] != "host" for row in steps),
            "host_steps": sum(row["placement"] == "host" for row in steps),
            "alias_regions": sum(row["staging"] == "alias" for row in regions),
            "split_regions": sum("slices" in row for row in regions),
            "shared_packages": len(shared), "launches": sum(row["launches"] for row in steps),
            "upload_bytes": sum(load["bytes"] for load in loads),
            "parameter_upload_bytes": sum(load["bytes"] for load in loads if load["source"]["kind"] == "parameter"),
            "inter_step_upload_bytes": sum(load["bytes"] for load in loads if load["source"]["kind"] == "step_output"),
            "readback_bytes": sum(read["bytes"] for read in reads),
            "max_window_required_bytes": max((row["window_required_bytes"] for row in steps), default=0),
        },
        "unproven": "linked host code and stack fit, step drivers, transport, and execution",
    }


def schedule_json(record):
    return (json.dumps(record, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def _package_files(label, lowered):
    files = {}
    for member, data in lowered.files.items():
        relative = Path(member)
        if relative.is_absolute() or relative.as_posix() != member or ".." in relative.parts:
            raise ValueError("region C export contains an unsafe path")
        files[f"c-package/regions/{label}/{member}"] = data
    return files


def render(source, adapter, outlined, plan, compiled, *, capacity, native_root, compile_slice, reason,
           parameters=None, payload=None, state=None, widths=None, counts=None, bands=None, tiles=None):
    """Publish region packages, shared slice packages, one step application per package and the schedule."""
    steps, regions, packages, aliases, applications = stage(
        source, adapter, outlined, plan, compiled, capacity=capacity, native_root=native_root,
        compile_slice=compile_slice, parameters=parameters, state=state, widths=widths, counts=counts, bands=bands,
        tiles=tiles)
    selected = {item.id: item for item in plan.selected}
    files = {member: data for application in applications.values() for member, data in application.items()}
    summaries = []
    for index, ((key, region), lowered) in enumerate(zip(outlined.regions, compiled, strict=True)):
        if isinstance(lowered, NativeCompilationError):
            continue
        files.update(_package_files(f"region_{index:03d}", lowered))
        summaries.append({**_summary(key, region, lowered, f"region_{index:03d}"),
                          "placement": selected[key].placement, "context_index": index})
    for label, request, lowered, summary in packages:
        files.update(_package_files(label, lowered))
        summaries.append({**summary, "context_index": next(row["context_index"] for row in regions
                                                          if row.get("shared_package") == label)})
    record = schedule_record(source, steps, regions, aliases, payload, state)
    record["staging_reason"] = reason
    record["window_bytes"] = capacity
    parameter_ids = set(parameters.node_ids) if parameters else set()
    state_ports = _readonly_state(source, state)
    interface = {"staging": "bench_staged", "deployment": "deployment.json",
                 "inputs": [key for key, _ in source.input_storages
                            if key not in parameter_ids and key not in state_ports],
                 "outputs": list(source.outputs),
                 **({"model_parameters": parameters.record()} if parameters else {}),
                 **({"model_state": state.record()} if state is not None else {})}
    if state is not None:
        files["application-state.json"] = schedule_json(state.record())
        files.update({f"handoff/state/{key}.bin": item.initial for key, item in state_ports.items()})
    files.update({
        "source-request.json": schedule_json(source.record()),
        "selected-plan.json": schedule_json(plan.to_dict()),
        "application-c.json": schedule_json(interface),
        "compiled-regions.json": schedule_json(summaries),
        "deployment.json": schedule_json(record),
        "c-package/README.md": (
            "# Staged application\n\nEach deployment.json step binds one region package under regions/. "
            "Slices of one region share one package. steps/<package>/ holds that package's generated "
            "single-call application (driver, deployment.json and handoff); instruction image paths are "
            "relative to this publication root. The bench host keeps every tensor and parameters.bin, uploads "
            "each step's ranges, and reads its results back. Linked ELFs and execution are not part of this "
            "publication.\n").encode(),
    })
    return summaries, interface, [], files
