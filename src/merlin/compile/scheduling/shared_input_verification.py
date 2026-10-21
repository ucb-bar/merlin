"""Independent finite source/lifetime induction over actual sibling effects.

This checks the synchronous logical effect family, not target admission or
overlap. An adapter must decode actual emitted addresses and source bindings;
retaining planned operations instead of projecting emitted ones is no proof.
"""

from __future__ import annotations

from .block_schedule import Compute, Load, Preload, Store
from .compact_dense import DenseLoopStorage
from .shared_input_groups import SharedInputGroup
from .shared_input_schedule import SharedInputEffect


def verify_shared_input(
    operations,
    *,
    module,
    group,
    primitive_tile,
    resources,
    storage,
    max_commands=1_000_000,
    max_rows_visited=16 << 20,
    max_source_operations=10_000,
    max_source_elements=64 << 20,
):
    """Check all source products, separate results and actual private addresses.

    Loaded row intervals carry original tensor identities. Every consumed
    rectangle is reconstructed from them, including retained array contents.
    Accumulator rows carry output ownership and the next required K position.
    This proves increasing, complete products without expanding M*K*N scalar
    products. Complete output intervals are checked independently at the end.
    """
    from merlin.transforms.dense_planning import DenseResources

    if type(group) is not SharedInputGroup or type(resources) is not DenseResources:
        raise ValueError("shared-input checking requires original source and selected resources")
    group.verify_current(module, max_source_operations=max_source_operations, max_source_elements=max_source_elements)
    if (
        type(operations) is not tuple
        or type(primitive_tile) is not tuple
        or len(primitive_tile) != 3
        or any(type(v) is not int or v <= 0 for v in (*primitive_tile, max_commands, max_rows_visited))
        or len(operations) > max_commands
        or type(storage) is not DenseLoopStorage
        or storage.accumulator is None
    ):
        raise ValueError("shared-input checking requires bounded actual effects and assigned private storage")
    tm, tk, tn = primitive_tile
    views = {"lhs": storage.lhs, "weight": storage.rhs, "accumulator": storage.accumulator}
    for role, width in (("lhs", tk), ("weight", tn), ("accumulator", 4 * tn)):
        if views[role].unit_bytes < width:
            raise ValueError("selected primitive exceeds assigned physical row width")
    intervals = [(v.backing, v.base * v.unit_bytes, (v.base + v.capacity) * v.unit_bytes) for v in views.values()]
    if any(
        a[0] == b[0] and a[1] < b[2] and b[1] < a[2]
        for index, a in enumerate(intervals)
        for b in intervals[index + 1 :]
    ):
        raise ValueError("shared-input assigned physical storage aliases")
    operands, accumulators, outputs = {}, {}, {}
    pending, stationary = None, None
    visited = products = lhs_loads = rhs_loads = requested = retained = 0
    high = dict.fromkeys(views, 0)

    def rectangle(role, row, rows, cols):
        nonlocal visited
        view = views[role]
        max_r, max_c = (tk, tn) if role == "weight" else (tm, tk if role == "lhs" else tn)
        if (
            any(type(v) is not int for v in (row, rows, cols))
            or not 0 < rows <= max_r
            or not 0 < cols <= max_c
            or row < view.base
            or row + rows > view.base + view.capacity
            or row % view.alignment
        ):
            raise ValueError("actual sibling rectangle exceeds primitive or assigned private storage")
        if not resources.partial_output_tiles and (
            (role in ("lhs", "accumulator") and rows != tm) or (role in ("weight", "accumulator") and cols != tn)
        ):
            raise ValueError("actual sibling rectangle requires unselected partial output support")
        visited += rows
        if visited > max_rows_visited:
            raise ValueError("actual sibling effects exceed independent row-checking budget")
        high[role] = max(high[role], row + rows - view.base)
        if (
            high["lhs"] * storage.lhs.unit_bytes + high["weight"] * storage.rhs.unit_bytes > resources.operand_bytes
            or high["accumulator"] * storage.accumulator.unit_bytes > resources.accumulator_bytes
        ):
            raise ValueError("actual sibling high-water marks exceed selected resource bytes")

    for effect in operations:
        if type(effect) is not SharedInputEffect:
            raise ValueError("unknown sibling effect cannot be erased")
        member, op = effect.member, effect.operation
        if member is not None and (type(member) is not int or not 0 <= member < len(group.widths)):
            raise ValueError("actual sibling effect selects unknown original result")
        if pending is not None and type(op) is not Compute:
            raise ValueError("synchronous sibling preload must have its immediate matching compute")
        if type(op) is Load:
            if op.role not in ("lhs", "weight") or (member is None) != (op.role == "lhs") or op.gather is not None:
                raise ValueError("actual sibling load has wrong source ownership or unknown gather")
            rectangle(op.role, op.row, op.rows, op.cols)
            source_rows, source_cols = (group.m, group.k) if member is None else (group.k, group.widths[member])
            if (
                any(type(v) is not int or v < 0 for v in (op.dram_row, op.dram_col))
                or op.dram_row + op.rows > source_rows
                or op.dram_col + op.cols > source_cols
            ):
                raise ValueError("actual sibling load exceeds original tensor")
            for r in range(op.rows):
                operands[op.role, op.row + r] = member, op.dram_row + r, op.dram_col, op.cols
            requested += op.rows * op.cols
            lhs_loads += member is None
            rhs_loads += member is not None
        elif type(op) is Preload:
            if member is None or type(op.accumulate) is not bool or type(op.weight_rows) is not int:
                raise ValueError("actual sibling descriptor has unknown source or arithmetic continuation")
            rectangle("accumulator", op.accumulator_row, op.rows, op.cols)
            if not 0 < op.weight_rows <= tk:
                raise ValueError("actual sibling descriptor has invalid reduction depth")
            if op.weight_row is not None:
                rectangle("weight", op.weight_row, op.weight_rows, op.cols)
                weights = tuple(operands.get(("weight", op.weight_row + r)) for r in range(op.weight_rows))
                if None in weights:
                    raise ValueError("actual sibling preload reads uninitialized weight rows")
                owner, k0, n0, width = weights[0]
                if (
                    owner != member
                    or width < op.cols
                    or any(row != (member, k0 + r, n0, width) for r, row in enumerate(weights))
                    or op.weight_block != (k0 // tk, n0 // tn)
                ):
                    raise ValueError("actual sibling weights have wrong source identity or rectangle")
                stationary = member, k0, n0, op.weight_rows, op.cols
            elif op.weight_block is not None:
                raise ValueError("retained sibling descriptor supplies an unrelated weight identity")
            if stationary is None or stationary[0] != member or stationary[3:] != (op.weight_rows, op.cols):
                raise ValueError("actual retained weights have wrong sibling ownership or geometry")
            pending = member, op, stationary
        elif type(op) is Compute:
            if (
                member is None
                or pending is None
                or member != pending[0]
                or type(op.fresh_weights) is not bool
                or op.input_gather is not None
            ):
                raise ValueError("actual sibling compute lacks its matching descriptor")
            _, descriptor, weights = pending
            if (op.rows, op.cols) != (descriptor.rows, descriptor.weight_rows) or op.fresh_weights != (
                descriptor.weight_row is not None
            ):
                raise ValueError("actual sibling compute changes primitive or retained-weight semantics")
            rectangle("lhs", op.input_row, op.rows, op.cols)
            _, k0, n0, depth, width = weights
            for r in range(op.rows):
                source = operands.get(("lhs", op.input_row + r))
                if source is None:
                    raise ValueError("actual sibling compute reads uninitialized shared input")
                owner, m, k, cols = source
                if (
                    owner is not None
                    or k != k0
                    or cols < depth
                    or op.input_block != ((m - r) // tm, k0 // tk)
                    or (r and m != operands["lhs", op.input_row][1] + r)
                ):
                    raise ValueError("actual sibling product differs from original input/weight geometry")
                address = descriptor.accumulator_row + r
                expected = member, m, n0, width, k0
                if descriptor.accumulate:
                    if k0 == 0 or accumulators.get(address) != expected:
                        raise ValueError("actual sibling continuation loses products or aliases a live output")
                elif k0 != 0:
                    raise ValueError("actual sibling initialization skips required source products")
                elif address in accumulators:
                    raise ValueError("actual sibling initialization overwrites an unread output")
                accumulators[address] = member, m, n0, width, k0 + depth
            products += op.rows * depth * width
            retained += not op.fresh_weights
            pending = None
        elif type(op) is Store:
            if member is None or op.readout is not None:
                raise ValueError("sibling raw-core checker cannot replace an independent original epilogue")
            rectangle("accumulator", op.accumulator_row, op.rows, op.cols)
            if (
                any(type(v) is not int or v < 0 for v in (op.dram_row, op.dram_col))
                or op.dram_row + op.rows > group.m
                or op.dram_col + op.cols > group.widths[member]
            ):
                raise ValueError("actual sibling readout exceeds original result")
            for r in range(op.rows):
                expected = member, op.dram_row + r, op.dram_col, op.cols, group.k
                if accumulators.get(op.accumulator_row + r) != expected:
                    raise ValueError("actual sibling readout changes or omits original products")
                outputs.setdefault((member, op.dram_row + r), []).append((op.dram_col, op.dram_col + op.cols))
                # Completion releases ownership; a different output may now
                # reuse the row, while a duplicate store still fails coverage.
                del accumulators[op.accumulator_row + r]
        else:
            raise ValueError("unknown sibling effects cannot be erased or approximated")
    if pending is not None or accumulators or len(outputs) != group.m * len(group.widths):
        raise ValueError("actual sibling final state leaves pending or missing original outputs")
    for (member, _), intervals in outputs.items():
        cursor = 0
        for start, end in sorted(intervals):
            if start != cursor:
                raise ValueError("actual sibling outputs contain gaps or duplicate intervals")
            cursor = end
        if cursor != group.widths[member]:
            raise ValueError("actual sibling outputs omit required source columns")
    if products != group.m * group.k * sum(group.widths):
        raise ValueError("actual sibling products differ from complete original work")
    return {
        "status": "verified_finite_shared_input_effects",
        "products": products,
        "output_elements": group.m * sum(group.widths),
        "lhs_loads": lhs_loads,
        "rhs_loads": rhs_loads,
        "requested_operand_bytes": requested,
        "requested_store_bytes": group.m * sum(group.widths) * 4,
        "retained_computes": retained,
        "commands": len(operations),
        "rows_visited": visited,
        "machine_emission": "not_established",
        "timing_authority": "NONE",
    }
