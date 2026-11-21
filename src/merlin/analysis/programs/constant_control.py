"""Bounded exact control trace when every branch depends only on constants.

This fallback observes no tensor values: loads, calls and opaque results remain
unknown. A branch consuming any unknown refuses the whole count. LLVM widths,
signed comparisons, casts and integer wrap are preserved; possible poison refuses.
"""

from __future__ import annotations

from collections import Counter, OrderedDict
from dataclasses import dataclass
from math import prod

from xdsl.dialects import llvm
from xdsl.dialects.builtin import IntegerAttr, IntegerType, UnitAttr
from xdsl.dialects.llvm import ICmpPredicateFlag
from xdsl.ir import Operation

from merlin.analysis.programs.cfg_activity import _constant_loop
from merlin.analysis.programs.host_cfg_index import prepare_host_cfg
from merlin.compiler.dialects.lowering.analyses import constant_integer

# ---- constant_control_cfg ----------------------------------------------------------------------

# A tiled whole-model kernel can execute tens of millions of host instructions
# while still having fully constant control. Keep a finite proof bound above
# that scale; data-dependent branches remain unknown regardless of the bound.
MAX_BLOCK_VISITS = 4_000_000
MAX_OPERATIONS = 100_000_000
MAX_COUNT_CACHE_KEYS = 4096


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
    if nonnegative is not None and (op.name != "llvm.zext" or not isinstance(nonnegative, UnitAttr) or signed < 0):
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
    function,
    *,
    max_blocks=MAX_BLOCK_VISITS,
    max_operations=MAX_OPERATIONS,
    summarize_loops=True,
    observe=None,
    observed_operations=None,
    summarizable_assembly=(),
):
    """Prove execution counts by terminating bounded constant-only CFG traversal.

    A straight-line counted loop (see :func:`_straight_line_loop`) is entered once
    and advanced by its exact trip count instead of being stepped: its body has no
    control and none of its values leave the loop, so the counts are the same.
    An observer may additionally supply original result-free assembly sites
    whose straight-line transport it has independently validated. Only unwatched
    sites can be summarized; this supplies no instruction or memory semantics.
    """
    # An explicit watched roster allows summaries only outside that roster.
    # Callers must still discard every observation on an incomplete traversal.
    if observe is not None and observed_operations is None:
        summarize_loops = False
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
    watched = None if observed_operations is None else frozenset(observed_operations)
    assembly = frozenset(summarizable_assembly)
    if any(
        op.name != "llvm.inline_asm" or op.results or op.regions or op.parent_block() not in block_set
        for op in assembly
    ):
        raise ValueError("summarized assembly must be original result-free instructions")
    if watched is not None and (observe is None or any(op.parent_block() not in block_set for op in watched)):
        raise ValueError("constant control observation roster leaves the original function")

    def unobserved_closed(nodes):
        if watched is None:
            return True  # Preserve the existing count-only summary contract.
        nodes = frozenset(nodes)
        for block in nodes:
            for item in (*block.args, *(result for op in block.ops for result in op.results)):
                if any(use.operation.parent_block() not in nodes for use in item.uses):
                    return False
            for op in block.ops:
                if (
                    op in watched
                    or op.regions
                    or op.name in {"llvm.call", "llvm.invoke"}
                    or op.name == "llvm.inline_asm"
                    and op not in assembly
                ):
                    return False
                # A count-only summary cannot discharge arithmetic assertions or
                # possible division/shift poison that the scalar visitor checks.
                if (
                    _poison_flags(op)
                    or op.properties.get("nonNeg") is not None
                    or op.name in {"llvm.udiv", "llvm.urem", "llvm.sdiv", "llvm.srem"}
                ):
                    return False
                if op.name in {"llvm.shl", "llvm.lshr", "llvm.ashr"}:
                    amount = constant_integer(op.operands[1])
                    if amount is None or not 0 <= amount < _width(op.operands[0]):
                        return False
        return True

    predecessors = {}
    for source in blocks:
        for destination in source.last_op.successors if source.last_op is not None else ():
            predecessors.setdefault(destination, []).append(source)
    loops, constants = {}, {}
    if summarize_loops:
        from merlin.verification.host.copy_domains import control_dependencies, copy_regions, region_cost
        from merlin.verification.host.copy_proofs import execute as batch_control

        batched_regions = {
            **control_regions(function, include_rectangular=watched is not None),
            **copy_regions(function),
        }
        batched_regions = {
            header: region for header, region in batched_regions.items() if unobserved_closed(region[0].nodes)
        }
    else:
        batched_regions = {}
    control_cones, count_cache = {}, OrderedDict()
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
                if loops[block] is not None and not unobserved_closed((block, *loops[block]["chain"])):
                    loops[block] = None
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
            rectangular = rectangular_counts(region) if watched is not None else None
            if rectangular is not None:
                if incoming != (region[0].start,):
                    raise ValueError("constant control rectangular entry differs from original induction")
                operations += len(region[0].nodes) + len(region[1])
                if operations > max_operations:
                    raise ValueError("constant control rectangular proof budget exhausted")
                counts.update(rectangular)
                values.update((arg, None) for member in region[0].nodes for arg in member.args)
                values.update(
                    (result, None) for member in region[0].nodes for op in member.ops for result in op.results
                )
                block, incoming = region[0].exit, ()
                continue
            cost = region_cost(region)
            if cost > 1000 or getattr(region[0], "count_only", False):
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
                    if len(count_cache) > MAX_COUNT_CACHE_KEYS:
                        count_cache.popitem(last=False)
                    operations += cost
                else:
                    successor, witnessed = cached
                    count_cache.move_to_end(key)
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
            if observe is not None and (watched is None or op in watched):
                observe(op, tuple(args))
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


# ---- constant_control_regions ------------------------------------------------------------------
# Bounded count-only domains for pretested loops with constant internal branches.
#
# Every nested natural loop must be independently proved. Its body is acyclic
# after removing exactly those backedges, and no value escapes. The masked integer
# interpreter still evaluates every selected control lane; opaque results remain unknown.
# These regions confer no memory, arithmetic, instruction or effect permission.


@dataclass(frozen=True)
class _ControlLoop:
    header: object
    latch: object
    nodes: frozenset
    exit: object
    start: int
    step: int
    trips: int  # Header visits, including the final false condition.
    update: object
    count_only: bool = True


def control_regions(function, *, include_rectangular=False):
    """Derive closed pretested rectangular domains without trusting annotations."""
    cfg = prepare_host_cfg(function)
    positions = {block: index for index, block in enumerate(cfg.blocks)}
    backedges = {
        (latch, header)
        for latch in cfg.blocks
        for header in cfg.successors[latch]
        if cfg.dominance.dominates(header, latch)
    }
    loops = {}
    for latch, header in backedges:
        if latch is header or len(header.args) != 1 or len(tuple(header.ops)) != 2:
            continue
        if not isinstance(header.args[0].type, IntegerType):
            continue
        nodes, pending = {header}, [latch]
        while pending:
            block = pending.pop()
            if block in nodes:
                continue
            nodes.add(block)
            pending.extend(cfg.predecessors[block])
        if any(block is not header and cfg.predecessors[block] - nodes for block in nodes):
            continue
        try:
            row, _ = _constant_loop(header, latch, nodes, cfg.predecessors, positions)
        except ValueError:
            continue
        if row.get("other_carried_values") or not 1 <= header.args[0].type.width.data <= 64:
            continue
        # The vector integer owner uses exact signed-host integers. Wider
        # unsigned domains remain available to the ordinary scalar interpreter.
        final = row["initial"] + row["trip_count"] * row["step"]
        if not 0 <= row["initial"] <= final < 1 << (header.args[0].type.width.data - 1):
            continue
        if any(
            use.operation.parent_block() not in nodes
            for block in nodes
            for value in (*block.args, *(r for op in block.ops for r in op.results))
            for use in value.uses
        ):
            continue
        loop = _ControlLoop(
            header,
            latch,
            frozenset(nodes),
            header.last_op.successors[1],
            row["initial"],
            row["step"],
            row["trip_count"] + 1,
            latch.last_op.operands[0],
        )
        if not loop.exit.args:
            loops[header] = loop

    from merlin.verification.host.copy_domains import control_dependencies, region_cost

    structural = {}
    for header, loop in loops.items():
        nested = sorted(
            (child for child in loops.values() if child.nodes <= loop.nodes), key=lambda child: -len(child.nodes)
        )
        edges = {(child.latch, child.header) for child in nested}
        if edges != {(a, b) for a, b in backedges if a in loop.nodes and b in loop.nodes}:
            continue  # An unsupported inner cycle cannot become a finite domain.
        if any(a.nodes & b.nodes and not (a.nodes <= b.nodes or b.nodes <= a.nodes) for a in nested for b in nested):
            continue
        pending, order = set(loop.nodes), []
        while pending:
            ready = [
                block
                for block in cfg.blocks
                if block in pending
                and not any(pred in pending and (pred, block) not in edges for pred in cfg.predecessors[block])
            ]
            if not ready:
                break
            order.extend(ready)
            pending.difference_update(ready)
        if pending:
            continue
        structural[header] = (loop, tuple(nested), tuple(order))
    rectangular = {
        header: region
        for header, region in structural.items()
        if include_rectangular and rectangular_counts(region) is not None
    }
    leaves = {header: region for header, region in structural.items() if len(region[1]) == 1}
    if _leaf_trace_fits(cfg, loops, backedges, leaves, MAX_BLOCK_VISITS) is not False:
        return {**leaves, **rectangular}
    found = dict(leaves)
    for header, region in structural.items():
        loop = region[0]
        if len(region[1]) == 1:
            continue  # Preserve the original leaf selection, including derived external keys.
        # Bound all distinct external-control keys, not only one invocation's
        # lanes. Otherwise a large outer region can consume the whole budget
        # before a cheaper inner region gets a chance. Unrecognized keys keep
        # the ordinary scalar fallback; this is selection, never count credit.
        external = control_dependencies(region)[1]
        varying = {value for value in external if constant_integer(value) is None}
        owners = {parent.header.args[0]: parent for parent in loops.values() if loop.nodes < parent.nodes}
        if varying - owners.keys():
            continue
        key_bound = prod(owners[value].trips for value in varying)
        if key_bound > MAX_COUNT_CACHE_KEYS or region_cost(region) * key_bound > MAX_OPERATIONS:
            continue
        found[header] = region
    return {**found, **rectangular}


def _leaf_trace_fits(cfg, loops, backedges, leaves, limit):
    """Prefer the original leaf path when its conservative scalar visits fit.

    This is a CFG/domain bound, never a scalar retrace. Unknown cycles retain
    the old path. Count every acyclic branch arm, all final-false headers and
    one entry per summarized leaf; omitted leaf bodies cost no scalar visits.
    """
    if {(loop.latch, loop.header) for loop in loops.values()} != backedges:
        return None
    pending = set(cfg.blocks)
    while pending:
        ready = {
            block
            for block in pending
            if not any(pred in pending and (pred, block) not in backedges for pred in cfg.predecessors[block])
        }
        if not ready:
            return None  # Irreducible/unrecognized cycles have no finite forest bound.
        pending.difference_update(ready)
    if any(
        a.header is not b.header and a.nodes & b.nodes and not (a.nodes < b.nodes or b.nodes < a.nodes)
        for a in loops.values()
        for b in loops.values()
    ):
        return None
    total = 0
    for block in cfg.blocks:
        if any(block in region[0].nodes and block is not header for header, region in leaves.items()):
            continue
        total += prod(
            loop.trips if block is loop.header else loop.trips - 1
            for loop in loops.values()
            if block in loop.nodes and not (block is loop.header and block in leaves)
        )
        if total > limit:
            return False
    return True


def rectangular_counts(region):
    """Exact block multiplicities for closed constant rectangular loop forests.

    The original region owner proves every backedge/domain and SSA closure. This
    narrower projection admits only those loop guards as conditional branches,
    with no early exits. It proves control multiplicities, never memory effects.
    """
    root, loops, order = region
    if not getattr(root, "count_only", False):
        return None
    headers = {loop.header for loop in loops}
    for block in root.nodes:
        terminal = block.last_op
        if terminal.name == "llvm.cond_br":
            if block not in headers:
                return None
        elif terminal.name != "llvm.br":
            return None
        if any(
            destination not in root.nodes and not (block is root.header and destination is root.exit)
            for destination in terminal.successors
        ):
            return None
    return {
        block: prod(loop.trips if block is loop.header else loop.trips - 1 for loop in loops if block in loop.nodes)
        for block in order
    }


# ---- finite_pointer_trace ----------------------------------------------------------------------
# Bounded emitted-LLVM operand projection with symbolic ABI pointer identity.
#
# No concrete addresses or tensor values are invented. This source-only observer
# supports constant metadata control and byte-addressed pointer arithmetic. Calls,
# poison, data-dependent control and exhausted bounds refuse the complete trace.
# It is not a native execution, alias authority or numerical qualification.


@dataclass(frozen=True)
class ArgumentAddress:
    argument: int
    offset: int = 0


def _value(op, args, pointer_bits):
    if _poison_flags(op):
        raise ValueError("pointer projection refuses unproved poison assertions")
    if op.name == "llvm.getelementptr":
        # Only an actual one-dimensional byte displacement has a layout-free
        # interpretation. Other element layouts, assertion flags and indices
        # remain unknown; no object bounds or nonalias premise follows here.
        element = op.properties.get("elem_type")
        flags = op.properties.get("noWrapFlags")
        indices = op.properties.get("rawConstantIndices")
        if (
            not isinstance(element, IntegerType)
            or element.width.data != 8
            or flags is None
            or flags.value.data != 0
            or indices is None
            or type(args[0]) is not ArgumentAddress
        ):
            return None
        raw = tuple(indices.get_values())
        if len(raw) != 1:
            return None
        index = raw[0]
        if index == llvm.GEP_USE_SSA_VAL:
            if (
                len(args) != 2
                or type(args[1]) is not int
                or not isinstance(op.operands[1].type, IntegerType)
                or op.operands[1].type.width.data != pointer_bits
            ):
                return None
            index = args[1] % (1 << pointer_bits)
            if index >= 1 << (pointer_bits - 1):
                index -= 1 << pointer_bits
        elif len(args) != 1:
            return None
        return ArgumentAddress(args[0].argument, args[0].offset + index)
    if op.name == "llvm.ptrtoint":
        if (
            len(args) != 1
            or not isinstance(op.results[0].type, IntegerType)
            or op.results[0].type.width.data != pointer_bits
        ):
            raise ValueError("pointer projection requires the original exact pointer integer width")
        return args[0] if type(args[0]) is ArgumentAddress else None
    if op.name == "llvm.inttoptr":
        return args[0] if len(args) == 1 and type(args[0]) is ArgumentAddress else None
    if op.name == "llvm.select":
        if type(args[0]) is not int:
            return None
        return args[1] if args[0] else args[2]
    if op.name in ("llvm.add", "llvm.sub") and len(args) == 2:
        a, b = args
        if type(a) is ArgumentAddress and type(b) is int:
            return ArgumentAddress(a.argument, a.offset + (b if op.name == "llvm.add" else -b))
        if op.name == "llvm.add" and type(a) is int and type(b) is ArgumentAddress:
            return ArgumentAddress(b.argument, b.offset + a)
        if op.name == "llvm.sub" and type(a) is type(b) is ArgumentAddress and a.argument == b.argument:
            return a.offset - b.offset
    if any(type(a) is ArgumentAddress for a in args):
        return None
    return integer_operation(op, args)


def project_finite_pointer_trace(function, *, observe, pointer_bits, max_operations=200_000, max_blocks=20_000):
    """Observe the actual scalar operands at every visited original operation.

    A pointer is its original formal argument identity plus a signed byte
    displacement. Selected addresses still need original array-bound and
    nonalias checks. Unsupported pointer transforms stay unknown. No partial
    observation is returned when any later control or resource bound refuses.
    """
    if (
        function.name != "llvm.func"
        or not function.body.blocks
        or not callable(observe)
        or type(pointer_bits) is not int
        or not 1 <= pointer_bits <= 128
        or type(max_operations) is not int
        or not 1 <= max_operations <= 1_000_000
        or type(max_blocks) is not int
        or not 1 <= max_blocks <= 100_000
    ):
        raise ValueError("bounded original function, pointer width and observer required")
    function.verify()
    blocks = tuple(function.body.blocks)
    block_set = set(blocks)
    values = {
        arg: ArgumentAddress(i) if type(arg.type) is llvm.LLVMPointerType else None
        for i, arg in enumerate(blocks[0].args)
    }
    pending, operations, visits, observations = blocks[0], 0, 0, []
    while pending is not None:
        block = pending
        visits += 1
        if visits > max_blocks or block not in block_set:
            raise ValueError("finite pointer projection block budget exhausted or branch leaves function")
        pending = None
        for op in block.ops:
            operations += 1
            if operations > max_operations:
                raise ValueError("finite pointer projection operation budget exhausted")
            if op.regions or op.name in ("llvm.call", "llvm.invoke"):
                raise ValueError("finite pointer projection has opaque control/effects")
            args = tuple(values.get(value) for value in op.operands)
            observed = observe(op, args)
            if observed is not None:
                observations.append(observed)
            if op.name == "llvm.br":
                incoming = args
                pending = op.successors[0]
            elif op.name == "llvm.cond_br":
                if type(args[0]) is not int:
                    raise ValueError(
                        "finite pointer projection control depends on tensor data or unknown pointer facts"
                    )
                sizes = tuple(op.properties["operandSegmentSizes"].get_values())
                taken = bool(args[0])
                pending = op.successors[0 if taken else 1]
                incoming = args[1 : 1 + sizes[1]] if taken else args[1 + sizes[1] :]
            elif op.name == "llvm.return":
                continue
            elif len(op.results) == 1:
                values[op.results[0]] = _value(op, args, pointer_bits)
            else:
                values.update((v, None) for v in op.results)
        if block.last_op is None or block.last_op.name not in ("llvm.br", "llvm.cond_br", "llvm.return"):
            raise ValueError("finite pointer projection requires explicit original LLVM control")
        if pending is not None:
            if pending not in block_set or len(incoming) != len(pending.args):
                raise ValueError("finite pointer projection has malformed branch arguments")
            values.update(zip(pending.args, incoming, strict=True))
    return tuple(observations)
