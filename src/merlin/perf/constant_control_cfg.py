"""Bounded exact control trace when every branch depends only on constants.

This fallback observes no tensor values: loads, calls and opaque results remain
unknown. A branch consuming any unknown refuses the whole count. LLVM widths,
signed comparisons, casts and integer wrap are preserved; possible poison refuses.
"""

from collections import Counter

from xdsl.dialects.builtin import IntegerAttr, IntegerType, UnitAttr
from xdsl.dialects.llvm import ICmpPredicateFlag
from xdsl.ir import Operation

from merlin.xdsl_dialects.lowering.integer_constant_eval import constant_integer

# A tiled whole-model kernel can execute tens of millions of host instructions
# while still having fully constant control. Keep a finite proof bound above
# that scale; data-dependent branches remain unknown regardless of the bound.
MAX_BLOCK_VISITS = 4_000_000
MAX_OPERATIONS = 100_000_000


def _width(value):
    ty = value.type
    if not isinstance(ty, IntegerType) or not 1 <= ty.width.data <= 128:
        raise ValueError("constant control requires a bounded integer width")
    return ty.width.data


def _signed(value, width):
    value %= 1 << width
    return value - (1 << width) if value >= 1 << (width - 1) else value


def _prove_poison_flags(op, unsigned, signed, width, result):
    """Check LLVM no-wrap assertions using the exact operands, before wrapping.

    Only the scalar add/sub/mul/shift flags understood here are admitted. This
    is a proof for one visited instruction, not permission to summarize a flagged
    induction update or to assume that a flag holds on another iteration.
    """
    for key in ("overflowFlags", "overflow_flags", "isExact", "isDisjoint"):
        attribute = op.properties.get(key)
        if attribute is None:
            continue
        if key == "isDisjoint" and isinstance(attribute, UnitAttr) and op.name == "llvm.or":
            if unsigned[0] & unsigned[1]:
                raise ValueError("constant control violates disjoint-or poison flag")
            continue
        if not isinstance(attribute, IntegerAttr):
            raise ValueError("constant control arithmetic has unproved poison flags")
        flags = attribute.value.data
        if flags == 0:
            continue
        if key != "overflowFlags" or flags not in (1, 2, 3):
            raise ValueError("constant control arithmetic has unproved poison flags")
        a, b = signed
        if op.name == "llvm.add":
            signed_result = a + b
        elif op.name == "llvm.sub":
            signed_result = a - b
        elif op.name == "llvm.mul":
            signed_result = a * b
        elif op.name == "llvm.shl":
            signed_result = a << unsigned[1]
        else:
            raise ValueError("constant control arithmetic has unproved poison flags")
        if flags & 1 and not -(1 << (width - 1)) <= signed_result < 1 << (width - 1):
            raise ValueError("constant control arithmetic violates signed no-wrap poison flags")
        if flags & 2 and not 0 <= result < 1 << width:
            raise ValueError("constant control arithmetic violates unsigned no-wrap poison flags")


def integer_operation(op, args):
    """One exact integer instruction; unknown operands never become constants."""
    name = op.op_name.data if op.name == "builtin.unregistered" else op.name
    if name == "llvm.mlir.constant":
        return constant_integer(op.results[0])
    if name == "llvm.select":
        if args[0] is None:
            return None
        return args[1] if args[0] else args[2]
    if not args or any(type(v) is not int for v in args):
        return None
    width = _width(op.operands[0])
    unsigned = [v % (1 << width) for v in args]
    signed = [_signed(v, width) for v in args]
    if name == "llvm.icmp":
        predicate = ICmpPredicateFlag.from_int(op.properties["predicate"].value.data).value
        a, b = signed if predicate.startswith("s") else unsigned
        comparison = {
            "eq": a == b,
            "ne": a != b,
            "slt": a < b,
            "ult": a < b,
            "sle": a <= b,
            "ule": a <= b,
            "sgt": a > b,
            "ugt": a > b,
            "sge": a >= b,
            "uge": a >= b,
        }
        return int(comparison[predicate])
    if name in {"llvm.sext", "llvm.zext", "llvm.trunc"}:
        prove_cast_flags(op, unsigned[0], signed[0])
        result = signed[0] if name == "llvm.sext" else unsigned[0]
    elif name in {"llvm.intr.umin", "llvm.intr.umax", "llvm.intr.smin", "llvm.intr.smax"}:
        if _poison_flags(op):
            raise ValueError("integer minimum/maximum has unproved poison flags")
        operands = unsigned if name in {"llvm.intr.umin", "llvm.intr.umax"} else signed
        result = min(operands) if name.endswith("min") else max(operands)
    else:
        a = unsigned[0]
        b = unsigned[1] if len(unsigned) > 1 else None
        if name == "llvm.add":
            result = a + b
        elif name == "llvm.sub":
            result = a - b
        elif name == "llvm.mul":
            result = a * b
        elif name == "llvm.and":
            result = a & b
        elif name == "llvm.or":
            result = a | b
        elif name == "llvm.xor":
            result = a ^ b
        elif name in {"llvm.udiv", "llvm.urem", "llvm.sdiv", "llvm.srem"}:
            if not b:
                raise ValueError("constant control integer division by zero")
            if name.startswith("llvm.s"):
                a, b = signed
                if a == -(1 << (width - 1)) and b == -1:
                    raise ValueError("constant control signed division overflow")
                quotient = (abs(a) // abs(b)) * (-1 if (a < 0) != (b < 0) else 1)
                result = quotient if name == "llvm.sdiv" else a - quotient * b
            else:
                result = a // b if name == "llvm.udiv" else a % b
        elif name in {"llvm.shl", "llvm.lshr", "llvm.ashr"}:
            if b >= width:
                raise ValueError("constant control shift is poison")
            result = a << b if name == "llvm.shl" else a >> b if name == "llvm.lshr" else signed[0] >> b
        else:
            return None
        _prove_poison_flags(op, unsigned, signed, width, result)
    return _signed(result, _width(op.results[0]))


def prove_cast_flags(op, unsigned, signed):
    """Check one concrete LLVM narrowing/nonnegative assertion before conversion.

    https://llvm.org/docs/LangRef.html#conversion-operations: trunc nuw preserves
    all unsigned bits; trunc nsw preserves the signed value. zext nneg requires
    a nonnegative signed source. No assertion may authorize its own operand.
    """
    destination = _width(op.results[0])
    for key in ("overflowFlags", "overflow_flags", "isExact", "isDisjoint"):
        attribute = op.properties.get(key)
        if attribute is None:
            continue
        if isinstance(attribute, IntegerAttr):
            flags = attribute.value.data
        elif isinstance(getattr(attribute, "data", None), frozenset):
            names = {value.value for value in attribute.data}
            if names - {"nsw", "nuw"}:
                raise ValueError("integer conversion has unproved poison flags")
            flags = (1 if "nsw" in names else 0) | (2 if "nuw" in names else 0)
        else:
            raise ValueError("integer conversion has unproved poison flags")
        if not flags:
            continue
        if key != "overflowFlags" or op.name != "llvm.trunc" or flags not in {1, 2, 3}:
            raise ValueError("integer conversion has unproved poison flags")
        if flags & 1 and not -(1 << (destination - 1)) <= signed < 1 << (destination - 1):
            raise ValueError("integer truncation violates signed no-wrap poison flags")
        if flags & 2 and not 0 <= unsigned < 1 << destination:
            raise ValueError("integer truncation violates unsigned no-wrap poison flags")
    nonnegative = op.properties.get("nonNeg")
    if nonnegative is not None and (
        op.name != "llvm.zext" or not isinstance(nonnegative, UnitAttr) or signed < 0
    ):
        raise ValueError("integer conversion violates nonnegative poison flag")


def _poison_flags(op):
    for key in ("overflowFlags", "overflow_flags", "isExact", "isDisjoint"):
        flags = op.properties.get(key)
        flag_value = getattr(getattr(flags, "value", None), "data", getattr(flags, "data", None))
        if isinstance(flag_value, frozenset) and not flag_value:
            continue  # xDSL's canonical #llvm.overflow<none> carries an empty flag set.
        if flags is not None and flag_value not in ((), 0, False):
            return True
    return False


def _straight_line_loop(header, block_set, predecessors):
    """Recognize ``for (iv = entry; iv < bound; iv += step)`` over straight-line blocks.

    Only the induction is carried; the header holds just its comparison and branch.
    The body is a chain of single-predecessor blocks with no other control, and no
    body value is used outside the loop, so its values cannot reach any branch.
    The bound and step are read from the traced values on each entry, so a bound
    that varies with an enclosing loop (a tile remainder) is still exact.
    """
    ops = tuple(header.ops)
    if len(header.args) != 1 or not isinstance(header.args[0].type, IntegerType) or len(ops) != 2:
        return None
    induction = header.args[0]
    compare, branch = ops
    if (
        compare.name != "llvm.icmp"
        or branch.name != "llvm.cond_br"
        or len(compare.operands) != 2
        or compare.operands[0] is not induction
        or branch.operands[0] is not compare.results[0]
    ):
        return None
    predicate = ICmpPredicateFlag.from_int(compare.properties["predicate"].value.data).value
    sizes = list(branch.properties["operandSegmentSizes"].get_values())
    body, exit_block = branch.successors
    if predicate not in {"slt", "ult"} or sizes[1] or body is header or exit_block is header:
        return None
    chain, block = [], body
    while True:
        if (
            block is header
            or block in chain
            or block not in block_set
            or block.args
            or predecessors.get(block) != [chain[-1] if chain else header]
        ):
            return None
        chain.append(block)
        terminal = block.last_op
        if terminal is None or terminal.name != "llvm.br":
            return None
        if any(op.successors for op in block.ops if op is not terminal):
            return None
        if terminal.successors[0] is header:
            break
        if terminal.operands:
            return None
        block = terminal.successors[0]
    inside = {header, *chain}
    if exit_block in inside:
        return None
    for block in chain:
        for op in block.ops:
            for result in op.results:
                if any(use.operation.parent_block() not in inside for use in result.uses):
                    return None
    latch = chain[-1].last_op
    if len(latch.operands) != 1:
        return None
    update = latch.operands[0].owner
    if (
        not isinstance(update, Operation)
        or update.name != "llvm.add"
        or update.parent_block() not in chain
        or _poison_flags(update)
        or len(update.operands) != 2
        or [value is induction for value in update.operands].count(True) != 1
    ):
        return None
    step = next(value for value in update.operands if value is not induction)
    bound = compare.operands[1]
    for value in (bound, step):
        owner = value.owner
        if (owner.parent_block() if isinstance(owner, Operation) else owner) in inside:
            return None
    return {
        "compare": compare,
        "branch": branch,
        "bound": bound,
        "step": step,
        "chain": tuple(chain),
        "cleared": dict.fromkeys(result for block in chain for op in block.ops for result in op.results),
        "unsigned": predicate == "ult",
    }


def _loop_trips(loop, entry, values):
    """Exact trip count and exit induction value, or ``None`` when the update could wrap."""
    width = _width(loop["compare"].operands[0])
    bound, step = values.get(loop["bound"]), values.get(loop["step"])
    if type(entry) is not int or type(bound) is not int or type(step) is not int:
        return None
    step = _signed(step, width)
    if loop["unsigned"]:
        start, limit, bound = entry % (1 << width), (1 << width) - 1, bound % (1 << width)
    else:
        start, limit, bound = _signed(entry, width), (1 << (width - 1)) - 1, _signed(bound, width)
    if step <= 0:
        return None
    trips = max(0, (bound - start + step - 1) // step)
    if start + trips * step > limit:
        return None
    return trips, _signed(start + trips * step, width)


def constant_control_counts(
    function, *, max_blocks=MAX_BLOCK_VISITS, max_operations=MAX_OPERATIONS, summarize_loops=True
):
    """Prove execution counts by terminating bounded constant-only CFG traversal.

    A straight-line counted loop (see :func:`_straight_line_loop`) is entered once
    and advanced by its exact trip count instead of being stepped: its body has no
    control and none of its values leave the loop, so the counts are the same.
    """
    blocks = tuple(function.body.blocks)
    if (
        not blocks
        or type(max_blocks) is not int
        or type(max_operations) is not int
        or min(max_blocks, max_operations) <= 0
    ):
        raise ValueError("constant control has no declared finite proof budget")
    if sum(len(tuple(b.ops)) for b in blocks) > max_operations:
        raise ValueError("constant control static operation budget exhausted")
    block_set = frozenset(blocks)
    predecessors = {}
    for source in blocks:
        for destination in source.last_op.successors if source.last_op is not None else ():
            predecessors.setdefault(destination, []).append(source)
    loops, constants = {}, {}
    if summarize_loops:
        from merlin.verify.input_copy_batch import execute as batch_control
        from merlin.verify.input_copy_loops import control_dependencies, copy_regions, region_cost

        batched_regions = copy_regions(function)
    else:
        batched_regions = {}
    control_cones, count_cache = {}, {}
    values = {arg: None for arg in blocks[0].args}
    counts = Counter()
    visits = operations = 0
    block, incoming = blocks[0], ()
    while block is not None:
        visits += 1
        if visits > max_blocks or block not in block_set:
            raise ValueError("constant control block budget exhausted or branch leaves function")
        if len(incoming) != len(block.args) and block is not blocks[0]:
            raise ValueError("constant control block argument mismatch")
        if summarize_loops and incoming:
            if block not in loops:
                loops[block] = _straight_line_loop(block, block_set, predecessors)
            loop = loops[block]
            trips = _loop_trips(loop, incoming[0], values) if loop is not None else None
            if trips is not None:
                count, final = trips
                counts[block] += count + 1
                for member in loop["chain"]:
                    counts[member] += count
                values.update(loop["cleared"])
                operations += 2
                values[block.args[0]] = final
                values[loop["compare"].results[0]] = 0
                args = [values.get(v) for v in loop["branch"].operands]
                successor, incoming = loop["branch"].successors[1], tuple(args[1:])
                block = successor
                continue
        if block in batched_regions and incoming:
            region = batched_regions[block]
            cost = region_cost(region)
            if cost > 1000:
                if incoming != (region[0].start,):
                    raise ValueError("constant control batch entry differs from original loop induction")
                if block not in control_cones:
                    control_cones[block] = control_dependencies(region)[1]
                key = (block, tuple(values.get(value) for value in control_cones[block]))
                cached = count_cache.get(key)
                if cached is None:
                    if cost + operations > max_operations:
                        raise ValueError("constant control batched operation budget exhausted")
                    successor, witnessed, _ = batch_control(
                        region, values, incoming, None, {}, set(), control_only=True
                    )
                    count_cache[key] = successor, witnessed
                    operations += cost
                else:
                    successor, witnessed = cached
                    operations += 1
                counts.update(witnessed)
                values.update((arg, None) for member in region[0].nodes for arg in member.args)
                values.update(
                    (result, None) for member in region[0].nodes for op in member.ops for result in op.results
                )
                block, incoming = successor, ()
                continue
        counts[block] += 1
        if incoming:
            values.update(zip(block.args, incoming, strict=True))
        successor = None
        for op in block.ops:
            operations += 1
            if operations > max_operations:
                raise ValueError("constant control operation budget exhausted")
            args = [values.get(v) for v in op.operands]
            if op.name == "llvm.br":
                successor, incoming = op.successors[0], tuple(args)
            elif op.name == "llvm.cond_br":
                if type(args[0]) is not int:
                    raise ValueError("constant control branch depends on tensor data or opaque result")
                sizes = list(op.properties["operandSegmentSizes"].get_values())
                taken = args[0] != 0
                successor = op.successors[0 if taken else 1]
                incoming = tuple(args[1 : 1 + sizes[1]] if taken else args[1 + sizes[1] :])
            elif op.name == "llvm.return":
                pass
            elif op.name == "llvm.mlir.constant":
                if op not in constants:
                    constants[op] = integer_operation(op, args)
                values[op.results[0]] = constants[op]
            elif len(op.results) == 1:
                values[op.results[0]] = integer_operation(op, args)
            else:
                values.update((result, None) for result in op.results)
        if block.last_op.name not in {"llvm.br", "llvm.cond_br", "llvm.return"}:
            raise ValueError("constant control has unsupported terminator")
        block = successor
    return {b: counts[b] for b in blocks}
