"""Observe complete local bit partitions without assigning tensor/resource roles."""

from __future__ import annotations

from collections import defaultdict

from xdsl.ir import BlockArgument, OpResult

from .hw_observations import _inputs, _instance_output, _integer, _module_name, _name, _width

SCHEMA = "merlin.hw_local_equal_partitions.v1"


def equal_partitions(module):
    """Trace only extracts/contiguous concats to inputs or opaque instance results.

    Every returned partition covers the complete original integer bitvector with
    equal, nonoverlapping slices. A root's identity and source ordinals are kept;
    names, vector length and port width confer no signedness, tensor axis, memory
    capacity, protocol, instruction or physical-tail interpretation.
    """
    partitions, incomplete = [], 0
    for op in module.walk():
        if _name(op) != "hw.module":
            continue
        inputs = _inputs(op)
        children = list(op.regions[0].block.ops)
        ordinals = {child: ordinal for ordinal, child in enumerate(children)}

        def trace(value, seen=frozenset()):
            width = _width(value)
            if width is None or width < 1 or value in seen:
                return None
            if isinstance(value, BlockArgument) and value in inputs:
                return value, 0, width
            if not isinstance(value, OpResult):
                return None
            parent, seen = value.owner, seen | {value}
            if _name(parent) == "hw.instance" and _instance_output(value) is not None:
                return value, 0, width
            if _name(parent) == "comb.extract" and len(parent.operands) == 1 and len(parent.results) == 1:
                base, low = trace(parent.operands[0], seen), _integer(parent, "lowBit")
                if base is not None and low is not None and 0 <= low and low + width <= base[2]:
                    return base[0], base[1] + low, width
            if _name(parent) == "comb.concat" and len(parent.results) == 1:
                parts = [trace(part, seen) for part in reversed(parent.operands)]
                if not parts or any(part is None for part in parts):
                    return None
                root, low, total = parts[0]
                for part in parts[1:]:
                    if part[0] is not root or part[1] != low + total:
                        return None
                    total += part[2]
                if total == width:
                    return root, low, total
            return None

        slices = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        for child in children:
            if _name(child) != "comb.extract":
                continue
            for value in child.results:
                observed = trace(value)
                if observed is not None:
                    root, low, width = observed
                    if width != _width(root):
                        slices[root][width][low].append(ordinals[child])
        for root, widths in slices.items():
            for width, lows in sorted(widths.items()):
                root_width = _width(root)
                if (
                    root_width % width
                    or len(lows) != root_width // width
                    or any(low != index * width for index, low in enumerate(sorted(lows)))
                ):
                    incomplete += 1
                    continue
                if isinstance(root, BlockArgument):
                    identity = {"kind": "module_input", "ordinal": root.index, "name": inputs[root]}
                else:
                    instance, selected_module, output, _ = _instance_output(root)
                    identity = {
                        "kind": "opaque_instance_output",
                        "producer_ordinal": ordinals[root.owner],
                        "output_ordinal": root.index,
                        "instance": instance,
                        "module": selected_module,
                        "output": output,
                    }
                partitions.append(
                    {
                        "module": _module_name(op),
                        "root": identity,
                        "packed_width": root_width,
                        "slice_width": width,
                        "slice_count": root_width // width,
                        "slices": [{"low_bit": low, "op_ordinals": sorted(lows[low])} for low in sorted(lows)],
                    }
                )
    return {
        "schema": SCHEMA,
        "scope": "complete local equal-width bitvector partitions only",
        "partitions": partitions,
        "incomplete_candidates": incomplete,
        "complete_resource_domain": False,
    }
