"""Captured input storage specialization shared by bufferization and callers."""

from __future__ import annotations


def has_overlapping_elements(record):
    """Decide injectivity of a static strided address map, ignoring empty axes."""
    shape, strides = record["shape"], record["stride"]
    if len(shape) != len(strides) or any(type(v) is not int or v < 0 for v in [*shape, *strides]):
        raise ValueError("cannot prove input storage injectivity without static nonnegative shape and strides")
    if 0 in shape:
        return False
    axes = sorted((s, n) for n, s in zip(shape, strides) if n > 1)
    span = 0
    for stride, size in axes:
        if stride == 0:
            return True
        if stride <= span:
            break
        span += (size - 1) * stride
    else:
        return False
    # Separated axis envelopes prove ordinary/padded/permuted layouts cheaply.
    # Interleaved envelopes need an exact address proof, not a dense-layout guess.
    addresses = {0}
    for stride, size in axes:
        expanded = set()
        for i in range(size):
            for address in addresses:
                value = address + i * stride
                if value in expanded:
                    return True
                expanded.add(value)
        addresses = expanded
    return False


def static_storage_span(record):
    """Return a proved static address envelope, or None for unknown layouts."""
    if record is None:
        return None
    shape, strides, offset = record["shape"], record["stride"], record["storage_offset"]
    if len(shape) != len(strides) or any(type(v) is not int or v < 0 for v in [*shape, *strides, offset]):
        return None
    if 0 in shape:
        return 0
    return offset + 1 + sum((n - 1) * s for n, s in zip(shape, strides))


def refuse_overlapping_input_aliases(contract):
    """Borrow overlapping inputs only while their storage remains read-only.

    Input-relative result views do not write their backing storage. Mutation
    results and result-to-result copies do, even when their destination is a
    borrowed predecessor rather than a direct input alias.
    """
    roots = []
    mutation_roots = {}
    writes = set()
    for ordinal, result in enumerate(contract["results"]):
        if result is None:
            roots.append(set())
            continue
        declared = set(result.get("alias_inputs", []))
        for index in declared:
            if not 0 <= index < len(contract["inputs"]):
                raise ValueError("invalid result base argument")
        aliases = set().union(*(mutation_roots.get(index, {index}) for index in declared))
        predecessors = result.get("alias_results", [])
        if not declared and predecessors:
            leader = predecessors[0]
            if type(leader) is not int or not 0 <= leader < ordinal:
                raise ValueError("invalid result storage predecessor")
            aliases.update(roots[leader])
            writes.update(aliases)
        if result.get("role") == "user_input_mutation":
            base = result["input_index"]
            if not 0 <= base < len(contract["inputs"]):
                raise ValueError("invalid result base argument")
            if base in mutation_roots:
                raise ValueError("duplicate mutation result for one input")
            aliases.update(mutation_roots.get(base, {base}))
            need, capacity = static_storage_span(result), static_storage_span(contract["inputs"][base])
            # The result-layout lowering allocates fresh storage when a static
            # mutation outgrows its caller allocation. Its returned views and
            # later result copies therefore cannot write overlapping caller bytes.
            if declared == {base} and need is not None and capacity is not None and need > capacity:
                aliases.clear()
            writes.update(aliases)
            mutation_roots[base] = aliases.copy()
        roots.append(aliases)
    for index in sorted(writes):
        record = contract["inputs"][index]
        if record is not None and has_overlapping_elements(record):
            raise ValueError(
                f"input {index} has overlapping logical elements: caller-storage write-back "
                "requires an injective input layout"
            )


def _apply_input_layout(module, contract):
    from torch_mlir import ir

    functions = [
        op
        for op in module.body.operations
        if op.operation.name == "func.func" and ir.StringAttr(op.attributes["sym_name"]).value == "forward"
    ]
    if len(functions) != 1:
        raise ValueError("input layout requires one forward definition")
    function = functions[0]
    arguments = function.regions[0].blocks[0].arguments
    records = contract["inputs"]
    if len(arguments) != len(records):
        raise ValueError("input layout ABI count mismatch")
    existing = function.attributes.get("arg_attrs")
    attrs = list(ir.ArrayAttr(existing)) if existing is not None else [ir.DictAttr.get({}) for _ in arguments]
    for i, (argument, record) in enumerate(zip(arguments, records)):
        if record is None:
            continue
        shape, strides, offset = record["shape"], record["stride"], record["storage_offset"]
        typ = ir.RankedTensorType(argument.type)
        if list(typ.shape) != shape or len(strides) != typ.rank:
            raise ValueError("captured input shape differs from the compiled input ABI")
        if any(type(v) is not int or v < 0 for v in [*strides, offset]):
            raise ValueError("captured input ABI requires static nonnegative strides and offset")
        updated = {entry.name: entry.attr for entry in ir.DictAttr(attrs[i])}
        updated["bufferization.buffer_layout"] = ir.StridedLayoutAttr.get(offset, strides)
        attrs[i] = ir.DictAttr.get(updated)
    function.attributes["arg_attrs"] = ir.ArrayAttr.get(attrs)


def is_identity_layout(record):
    if record is None:
        return True
    stride = 1
    expected = []
    for size in reversed(record["shape"]):
        expected.append(stride)
        stride *= size
    return record["storage_offset"] == 0 and record["stride"] == list(reversed(expected))


def storage_array(logical, record):
    """Scatter transport values into captured storage; refuse inconsistent overlap."""
    import numpy as np

    if record is None:
        return logical
    shape, strides, offset = record["shape"], record["stride"], record["storage_offset"]
    if list(logical.shape) != shape or len(strides) != len(shape):
        raise ValueError("runtime input shape differs from captured input ABI")
    if any(type(v) is not int or v < 0 for v in [*shape, *strides, offset]):
        raise ValueError("runtime input ABI requires static nonnegative layout")
    if is_identity_layout(record) and record.get("minimum_storage_elements", 0) <= logical.size:
        return logical
    capacity = offset if 0 in shape else offset + 1 + sum((n - 1) * s for n, s in zip(shape, strides))
    minimum = record.get("minimum_storage_elements", 0)
    if type(minimum) is not int or minimum < 0:
        raise ValueError("runtime input minimum backing span must be a nonnegative integer")
    physical = np.zeros(max(1, capacity, minimum), dtype=logical.dtype)
    seen = {}
    for coordinate in np.ndindex(*shape):
        address = offset + sum(i * s for i, s in zip(coordinate, strides))
        value = logical[coordinate]
        raw = value.tobytes()
        if address in seen and seen[address] != raw:
            raise ValueError("runtime input values disagree at overlapping captured storage addresses")
        seen[address] = raw
        physical[address] = value
    return physical


def descriptor_lines(records):
    """Bind rank-exact descriptors to the same storage ABI used by the compiler."""
    lines = []
    for i, record in enumerate(records):
        if record is None:
            continue
        rank = len(record["shape"])
        lines.append(f"((int64_t *)d[{i}])[2] = {record['storage_offset']};")
        for axis, stride in enumerate(record["stride"]):
            lines.append(f"((int64_t *)d[{i}])[{3 + rank + axis}] = {stride};")
    return lines
