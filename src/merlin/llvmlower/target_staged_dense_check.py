"""Conservative ordering envelope for the first staged LLVM integration.

This checks preservation of existing completion placements, not a hardware
completion law. Only physical-row reassignment, whole-output-tile permutation
and deletion of RHS loads are admitted. The independent staged math checker
must still establish that the actual retained operands compute every product.
"""

from merlin.compile.scheduling import staged_dense as staged


def _normalized(operation):
    """Forget assigned local rows only; keep all logical work and flags."""
    kind = type(operation)
    if kind is staged.OperandLoad:
        values = (operation.role, operation.source_row, operation.source_col, operation.rows, operation.cols)
    elif kind is staged.AccumulatorZero:
        values = (operation.rows, operation.cols)
    elif kind is staged.DescriptorIssue:
        values = (operation.rhs_row is None, operation.m, operation.k, operation.n, operation.continuation)
    elif kind is staged.ComputeConsume:
        values = (operation.m, operation.k, operation.retain_weights)
    elif kind is staged.AccumulatorReadout:
        values = (operation.output_row, operation.output_col, operation.rows, operation.cols)
    elif kind is staged.CompletionFence:
        values = ()
    else:
        raise ValueError("staged LLVM completion preservation has unknown effects")
    return kind, tuple((type(value), value) for value in values)


def _tiles(operations):
    tiles, cursor = {}, 0
    while cursor < len(operations):
        start = cursor
        if type(operations[cursor]) is not staged.AccumulatorZero:
            raise ValueError("staged LLVM completion preservation needs independently zeroed output tiles")
        cursor += 1
        while cursor < len(operations) and type(operations[cursor]) is not staged.AccumulatorReadout:
            if type(operations[cursor]) is staged.AccumulatorZero:
                raise ValueError("staged LLVM completion preservation cannot split live output tiles")
            _normalized(operations[cursor])
            cursor += 1
        if cursor == len(operations):
            raise ValueError("staged LLVM output tile has no complete readout")
        readout = operations[cursor]
        key = readout.output_row, readout.output_col
        cursor += 1
        if cursor == len(operations) or type(operations[cursor]) is not staged.CompletionFence:
            raise ValueError("staged LLVM output tile lost its original readout completion")
        while cursor < len(operations) and type(operations[cursor]) is staged.CompletionFence:
            cursor += 1
        if key in tiles:
            raise ValueError("staged LLVM completion preservation found duplicate output tiles")
        tiles[key] = operations[start:cursor]
    if not tiles:
        raise ValueError("staged LLVM completion preservation needs complete output work")
    return tiles


def verify_staged_completion_preservation(original, actual, *, max_commands=1_000_000):
    """Check actual order against source, independently of the selected passes.

    The source defines the leading prefix and final all-fence suffix (after the
    last readout's mandatory first fence). Strip exactly these counts from both
    streams; inferring a new candidate suffix would misclassify a moved tile's
    longer closing run as global completion. No synchronization is removed or
    hoisted by this first integration route.
    """
    if (
        type(max_commands) is not int
        or not 0 < max_commands <= 1 << 30
        or type(original) is not tuple
        or type(actual) is not tuple
        or len(original) > max_commands
        or len(actual) > max_commands
    ):
        raise ValueError("staged LLVM completion preservation requires bounded immutable operations")
    for operation in (*original, *actual):
        _normalized(operation)
    prefix = 0
    while prefix < len(original) and type(original[prefix]) is staged.CompletionFence:
        prefix += 1
    last = next((i for i in range(len(original) - 1, -1, -1) if type(original[i]) is staged.AccumulatorReadout), None)
    if (
        last is None
        or last + 1 == len(original)
        or any(type(op) is not staged.CompletionFence for op in original[last + 1 :])
    ):
        raise ValueError("original staged LLVM source lacks final readout completion")
    suffix = len(original) - last - 2
    if prefix + suffix >= len(actual):
        raise ValueError("staged LLVM candidate overlaps anchored completion boundaries")
    if any(type(op) is not staged.CompletionFence for op in actual[:prefix]) or (
        suffix and any(type(op) is not staged.CompletionFence for op in actual[-suffix:])
    ):
        raise ValueError("staged LLVM candidate changed anchored completion boundaries")
    before = _tiles(original[prefix : len(original) - suffix if suffix else None])
    after = _tiles(actual[prefix : len(actual) - suffix if suffix else None])
    if before.keys() != after.keys():
        raise ValueError("staged LLVM candidate changed complete output-tile membership")
    removed = 0
    for key, candidate in after.items():
        source, cursor = before[key], 0
        for operation in candidate:
            while cursor < len(source) and _normalized(source[cursor]) != _normalized(operation):
                if type(source[cursor]) is not staged.OperandLoad or source[cursor].role != "rhs":
                    raise ValueError("staged LLVM candidate moved or removed a retained effect/completion fence")
                cursor += 1
                removed += 1
            if cursor == len(source):
                raise ValueError("staged LLVM candidate added or changed an effect/completion fence")
            cursor += 1
        for operation in source[cursor:]:
            if type(operation) is not staged.OperandLoad or operation.role != "rhs":
                raise ValueError("staged LLVM candidate dropped original work/completion")
            removed += 1
    return {
        "output_tiles": len(before),
        "removed_rhs_loads": removed,
        "anchored_prefix_fences": prefix,
        "anchored_suffix_fences": suffix,
        "scope": "preserved source synchronization placement; original completion law remains required",
    }
