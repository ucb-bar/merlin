"""Bounded scalar observations for compact emitted LLVM command schedules.

These are static observations, not an expanded execution trace or a hazard
proof. Only a proved single-iteration loop body can bind an induction value to
one constant. A varying induction of a proved counted loop is known only by its
value RANGE (initial through its final update), which bounds the high bits of
the expressions it feeds (:func:`integer_range`); exit/header values and other
phi arguments remain unknown. Each block's executions per call come from the
same counted-loop proof (:func:`counted_loop_facts`). No work is proportional to
a loop's trip count.
"""

from xdsl.dialects.builtin import IntegerAttr, IntegerType
from xdsl.ir import Operation


def fold_integer(value, bindings):
    """Fold closed fixed-width integer expressions, refusing poison flags/shifts."""
    cache = {}

    def visit(v, depth=0):
        if v in cache:
            return cache[v]
        if depth > 64 or len(cache) >= 4096 or not isinstance(v.type, IntegerType):
            return None
        width = v.type.width.data
        if not 0 < width <= 256:
            return None
        cache[v] = None
        if v in bindings:
            result = bindings[v]
        else:
            op = v.owner
            if not isinstance(op, Operation):
                return None
            flags = op.properties.get("overflowFlags")
            if flags is not None and (not isinstance(flags, IntegerAttr) or flags.value.data != 0):
                return None
            if op.name == "llvm.mlir.constant":
                attr = op.properties.get("value")
                if not isinstance(attr, IntegerAttr):
                    return None
                result = attr.value.data
            else:
                args = [visit(arg, depth + 1) for arg in op.operands]
                if not args or any(arg is None for arg in args):
                    return None
                a = args[0]
                if op.name in {"llvm.trunc", "llvm.sext", "llvm.zext"} and len(args) == 1:
                    result = a
                    if op.name == "llvm.zext":
                        result &= (1 << op.operands[0].type.width.data) - 1
                elif len(args) == 2:
                    b = args[1]
                    if op.name == "llvm.add":
                        result = a + b
                    elif op.name == "llvm.sub":
                        result = a - b
                    elif op.name == "llvm.mul":
                        result = a * b
                    elif op.name == "llvm.and":
                        result = a & b
                    elif op.name == "llvm.or":
                        result = a | b
                    elif op.name == "llvm.xor":
                        result = a ^ b
                    elif op.name in {"llvm.shl", "llvm.lshr", "llvm.ashr"} and 0 <= b < width:
                        result = (a << b if op.name == "llvm.shl" else
                                  a >> b if op.name == "llvm.ashr" else
                                  (a & ((1 << width) - 1)) >> b)
                    else:
                        return None
                else:
                    return None
        result &= (1 << width) - 1
        if result >= 1 << (width - 1):
            result -= 1 << width
        cache[v] = result
        return result

    return visit(value)


def _overflow_flags_only(op) -> bool:
    """Whether ``op``'s only poison promise is a no-wrap flag (refuted by a non-negative signed range)."""
    from merlin.perf.constant_control_cfg import _poison_flags

    if not _poison_flags(op):
        return False
    return not any(op.properties.get(key) is not None for key in ("isExact", "isDisjoint", "nonNeg", "sameSign"))


def _incoming(block_argument):
    """Every value reaching ``block_argument`` along its function's branches, or ``None``."""
    block, parent = block_argument.owner, block_argument.owner.parent_op()
    if parent is None or parent.name != "llvm.func" or len(parent.body.blocks) > 4096:
        return None
    if block is parent.body.blocks.first:
        return None
    incoming = []
    for pred in parent.body.blocks:
        end = pred.last_op
        if end is None:
            continue
        for i, successor in enumerate(end.successors):
            if successor is not block:
                continue
            if end.name == "llvm.br":
                args = tuple(end.operands)
            elif end.name == "llvm.cond_br":
                sizes = tuple(end.properties["operandSegmentSizes"].get_values())
                args = tuple(end.operands[1:1 + sizes[1]] if i == 0 else end.operands[1 + sizes[1]:])
            else:
                return None
            if len(args) != len(block.args) or args[block_argument.index].type != block_argument.type:
                return None
            incoming.append(args[block_argument.index])
    return incoming or None


def integer_range(value, ranges, bindings=None):
    """``(lo, hi)``: a proved non-negative range of an integer SSA value, or ``None``.

    ``ranges`` holds the induction values of proved counted loops; constants and ``bindings`` are
    exact. Additions, products, shifts by constants and bitwise combinations propagate a range only
    while it stays below the signed limit of its width, so no wrap (and no no-wrap poison) occurs.
    Anything else -- ABI arguments, loads, calls, other phis -- has no range.
    """
    from xdsl.ir import BlockArgument

    bindings = bindings or {}
    cache, active = {}, set()

    def visit(v, depth=0):
        if not isinstance(v.type, IntegerType) or not 0 < (width := v.type.width.data) <= 256:
            return None
        if v in cache:
            return cache[v]
        if v in active or depth > 64 or len(cache) + len(active) >= 4096:
            return None
        limit = (1 << (width - 1)) - 1 if width > 1 else 1
        active.add(v)
        result = None
        if v in bindings:
            exact = bindings[v] & ((1 << width) - 1)
            result = (exact, exact)
        elif v in ranges:
            result = ranges[v]
        elif isinstance(v, BlockArgument):
            incoming = _incoming(v)
            if incoming:
                arms = [visit(arg, depth + 1) for arg in incoming]
                if all(arm is not None for arm in arms):
                    result = (min(a[0] for a in arms), max(a[1] for a in arms))
        elif isinstance(v.owner, Operation):
            op = v.owner
            from merlin.perf.constant_control_cfg import _poison_flags

            if not _poison_flags(op) or _overflow_flags_only(op):
                name = op.name
                if name == "llvm.mlir.constant" and isinstance(attr := op.properties.get("value"), IntegerAttr):
                    exact = attr.value.data & ((1 << width) - 1)
                    result = (exact, exact)
                elif name in {"llvm.zext", "llvm.sext", "llvm.trunc"} and len(op.operands) == 1:
                    result = visit(op.operands[0], depth + 1)
                elif name == "llvm.select" and len(op.operands) == 3:
                    arms = [visit(arg, depth + 1) for arg in op.operands[1:]]
                    if all(arm is not None for arm in arms):
                        result = (min(a[0] for a in arms), max(a[1] for a in arms))
                elif name == "llvm.and" and len(op.operands) == 2:
                    arms = [arm for arm in (visit(arg, depth + 1) for arg in op.operands) if arm is not None]
                    if arms:
                        result = (0, min(arm[1] for arm in arms))
                elif len(op.operands) == 2:
                    a, b = (visit(arg, depth + 1) for arg in op.operands)
                    if a is not None and b is not None:
                        if name == "llvm.add":
                            result = (a[0] + b[0], a[1] + b[1])
                        elif name == "llvm.sub" and a[0] >= b[1]:
                            result = (a[0] - b[1], a[1] - b[0])
                        elif name == "llvm.mul":
                            result = (a[0] * b[0], a[1] * b[1])
                        elif name == "llvm.shl" and b[0] == b[1] and b[0] < width:
                            result = (a[0] << b[0], a[1] << b[0])
                        elif name == "llvm.lshr" and b[0] == b[1] and b[0] < width:
                            result = (a[0] >> b[0], a[1] >> b[0])
                        elif name in {"llvm.or", "llvm.xor"}:
                            top = (1 << max(a[1].bit_length(), b[1].bit_length())) - 1
                            result = (max(a[0], b[0]) if name == "llvm.or" else 0, top)
        if result is not None and not 0 <= result[0] <= result[1] <= limit:
            result = None
        active.discard(v)
        cache[v] = result
        return result

    return visit(value)


def known_integer_bits(value, bindings, ranges=None):
    """Bounded bit facts for defined modular scalar SSA, never a guessed whole operand.

    Unknown memory/calls, poison promises and unproved shifts refuse. Phi cycles
    contribute no bits and every incoming definition is still checked. Work is
    bounded by the original SSA graph, independently of loop trip counts. With
    ``ranges`` (counted-loop induction ranges), a value whose :func:`integer_range`
    is proved also knows every bit above its upper bound to be zero.
    """
    from xdsl.ir import BlockArgument

    from merlin.perf.constant_control_cfg import _poison_flags

    cache, active = {}, set()

    def visit(v, depth=0):
        if not isinstance(v.type, IntegerType) or not 0 < (width := v.type.width.data) <= 256:
            return None
        full = (1 << width) - 1
        if depth > 64 or len(cache) + len(active) >= 4096:
            return None
        if v in active:
            return 0, 0, width
        if v in cache:
            return cache[v]
        active.add(v)
        result = None
        if v in bindings:
            result = full, bindings[v] & full, width
        elif isinstance(v, BlockArgument):
            block, parent = v.owner, v.owner.parent_op()
            if parent is not None and parent.name == "llvm.func" and len(parent.body.blocks) <= 4096:
                if block is parent.body.blocks.first:
                    result = 0, 0, width  # Original native scalar ABI argument.
                else:
                    incoming = []
                    valid = True
                    for pred in parent.body.blocks:
                        end = pred.last_op
                        if end is None:
                            continue
                        for i, successor in enumerate(end.successors):
                            if successor is not block:
                                continue
                            if end.name == "llvm.br":
                                args = tuple(end.operands)
                            elif end.name == "llvm.cond_br":
                                sizes = tuple(end.properties["operandSegmentSizes"].get_values())
                                args = tuple(end.operands[1:1 + sizes[1]] if i == 0 else end.operands[1 + sizes[1]:])
                            else:
                                valid = False
                                continue
                            if len(args) != len(block.args) or args[v.index].type != v.type:
                                valid = False
                            else:
                                incoming.append(visit(args[v.index], depth + 1))
                    if valid and incoming and all(item is not None for item in incoming):
                        mask, bits, _ = incoming[0]
                        for other_mask, other_bits, _ in incoming[1:]:
                            mask &= other_mask & ~(bits ^ other_bits)
                            bits &= mask
                        result = mask, bits, width
        else:
            op = v.owner
            if isinstance(op, Operation) and not _poison_flags(op) and not any(
                op.properties.get(key) is not None for key in ("nonNeg", "sameSign")
            ):
                if op.name == "llvm.mlir.constant" and isinstance(attr := op.properties.get("value"), IntegerAttr):
                    result = full, attr.value.data & full, width
                elif op.name in {
                    "llvm.add", "llvm.sub", "llvm.mul", "llvm.and", "llvm.or", "llvm.xor", "llvm.icmp",
                    "llvm.select", "llvm.shl", "llvm.lshr", "llvm.trunc", "llvm.zext", "llvm.sext",
                }:
                    arity = (3 if op.name == "llvm.select" else
                             1 if op.name in {"llvm.trunc", "llvm.zext", "llvm.sext"} else 2)
                    args = [visit(arg, depth + 1) for arg in op.operands] if len(op.operands) == arity else []
                    if args and all(arg is not None for arg in args):
                        widths = [arg[2] for arg in args]
                        if op.name == "llvm.select":
                            typed = widths == [1, width, width]
                        elif op.name == "llvm.icmp":
                            typed = width == 1 and len(widths) == 2 and widths[0] == widths[1]
                        elif op.name in {"llvm.trunc", "llvm.zext", "llvm.sext"}:
                            typed = len(widths) == 1
                        else:
                            typed = widths == [width, width]
                        if not typed:
                            active.remove(v)
                            cache[v] = None
                            return None
                        mask, bits, source_width = args[0]
                        if op.name in {"llvm.add", "llvm.sub", "llvm.mul", "llvm.icmp"} and len(args) == 2:
                            result = 0, 0, width
                        elif op.name == "llvm.select" and len(args) == 3:
                            left, right = args[1:]
                            if mask == 1:
                                result = left if bits else right
                            else:
                                common = left[0] & right[0] & ~(left[1] ^ right[1])
                                result = common, left[1] & common, width
                        elif op.name in {"llvm.and", "llvm.or", "llvm.xor"} and len(args) == 2:
                            other_mask, other_bits, _ = args[1]
                            if op.name == "llvm.or":
                                ones = bits | other_bits
                                zeros = (mask & ~bits) & (other_mask & ~other_bits)
                            elif op.name == "llvm.and":
                                ones = bits & other_bits
                                zeros = (mask & ~bits) | (other_mask & ~other_bits)
                            else:
                                common = mask & other_mask
                                ones = (bits ^ other_bits) & common
                                zeros = common & ~ones
                            result = (ones | zeros) & full, ones & full, width
                        elif op.name in {"llvm.shl", "llvm.lshr"} and len(args) == 2:
                            shift_mask, shift, shift_width = args[1]
                            if shift_mask == (1 << shift_width) - 1 and 0 <= shift < width:
                                if op.name == "llvm.shl":
                                    shifted_mask = ((mask << shift) | ((1 << shift) - 1)) & full
                                    result = shifted_mask, (bits << shift) & full, width
                                else:
                                    high = full ^ ((1 << (width - shift)) - 1)
                                    result = (mask >> shift) | high, bits >> shift, width
                        elif op.name == "llvm.trunc" and len(args) == 1 and width <= source_width:
                            result = mask & full, bits & full, width
                        elif op.name in {"llvm.zext", "llvm.sext"} and len(args) == 1 and width >= source_width:
                            high = full ^ ((1 << source_width) - 1)
                            if op.name == "llvm.zext":
                                result = mask | high, bits, width
                            elif mask & (1 << (source_width - 1)):
                                result = mask | high, bits | (high if bits & (1 << (source_width - 1)) else 0), width
                            else:
                                result = mask, bits, width
        if ranges:
            bounded = integer_range(v, ranges, bindings)
            if bounded is not None:
                full = (1 << width) - 1
                zeros = full & ~((1 << bounded[1].bit_length()) - 1)
                mask, bits, _ = result if result is not None else (0, 0, width)
                result = (mask | zeros) & full, bits & ~zeros & full, width
        active.remove(v)
        cache[v] = result
        return result

    return visit(value)


def counted_loop_observations(module):
    """Reuse the independent constant-loop proof, with no dynamic enumeration."""
    facts = counted_loop_facts(module)
    return facts["observation"], facts["bindings"]


def _callers(module):
    """``{callee symbol: [calling operations]}`` over every direct call of the module."""
    calls = {}
    for op in module.walk():
        if op.name in ("llvm.call", "llvm.invoke"):
            callee = op.properties.get("callee")
            name = getattr(getattr(callee, "root_reference", None), "data", None)
            calls.setdefault(name, []).append(op)
    return calls


def counted_loop_facts(module):
    """The counted-loop proof of every function: its observation, single-trip bindings, the value
    range of every proved induction, and each block's executions per kernel call.

    A block's executions are its function's proved block multiplicity
    (:func:`merlin.perf.host_cfg_activity.block_multiplicity`) for a function nothing in the module
    calls (the harness invokes it once); a function called from inside the module, or one whose CFG is
    not proved, leaves its blocks' executions ``None`` (unknown), never a guess.
    """
    from merlin.perf.host_cfg_activity import _constant_loop, block_multiplicity
    from merlin.perf.host_cfg_index import prepare_host_cfg

    functions, bindings, ranges, executions = [], {}, {}, {}
    callers = _callers(module)
    for function in module.body.block.ops:
        if function.name != "llvm.func" or not function.body.blocks:
            continue
        blocks = tuple(function.body.blocks)
        called = bool(callers.get(function.sym_name.data))
        if len(blocks) == 1:
            executions[blocks[0]] = None if called else 1
            continue
        record = {"function": function.sym_name.data, "loops": [], "unavailable": []}
        functions.append(record)
        if len(blocks) > 4096:
            record["unavailable"].append("CFG exceeds bounded static observation")
            continue
        try:
            cfg = prepare_host_cfg(function)
        except (ValueError, KeyError, TypeError, AttributeError) as error:
            record["unavailable"].append(str(error))
            continue
        try:
            multiplicity = {} if called else block_multiplicity(function, prepared_cfg=cfg)
        except (ValueError, KeyError, TypeError, AttributeError):
            multiplicity = {}
        executions.update({block: multiplicity.get(block) for block in blocks})
        positions = {block: i for i, block in enumerate(blocks)}
        for header in blocks:
            for latch in cfg.predecessors[header]:
                if not cfg.dominance.dominates(header, latch):
                    continue
                nodes, pending = {header, latch}, [latch] if header is not latch else []
                while pending:
                    for pred in cfg.predecessors[pending.pop()]:
                        if pred not in nodes:
                            nodes.add(pred)
                            pending.append(pred)
                try:
                    row, nodes = _constant_loop(header, latch, nodes, cfg.predecessors, positions)
                except (ValueError, KeyError, TypeError, AttributeError) as error:
                    record["unavailable"].append(f"loop at block {positions[header]}: {error}")
                    continue
                record["loops"].append(row)
                induction_value = header.args[row["induction_argument_index"]]
                final = row["initial"] + row["trip_count"] * row["step"]
                if row["initial"] >= 0:
                    ranges[induction_value] = (row["initial"], max(row["initial"], final))
                if row["trip_count"] == 1:
                    induction = header.args[row["induction_argument_index"]]
                    # A pretested header observes both initial and final IV. A
                    # posttested singleton header executes its body once.
                    body = nodes if row.get("post_tested") else nodes - {header}
                    for block in body:
                        bindings.setdefault(block, {})[induction] = row["initial"]
    observation = {"basis": "original_llvm_cfg_constant_induction_v1", "functions": functions,
                   "instruction_multiplicity": "executions_per_kernel_call",
                   "varying_operands": "induction ranges only; loop bounds are not operand or hazard proof"}
    return {"observation": observation, "bindings": bindings, "ranges": ranges, "executions": executions}
