"""The integer softmax, restructured over its captured IR: the runner half.

This file is SOURCE for the lowering runner, not a module Merlin imports: :mod:`.int_softmax_table`
reads it and splices it into every runner variant, which executes it in the compiler's Python (the
MLIR bindings live there). It is therefore self-contained -- only ``torch_mlir`` and the standard
library, imported inside the functions -- and every top-level name starts with ``_ist_`` so it cannot
collide with another rewrite spliced into the same script.

What it matches is the STRUCTURE an integer softmax is captured as, never a model or a target:

    m  = rowmax(x)                      linalg.reduce maximumf over the last dimension
    q  = fptosi(clamp(roundeven((x - m) / S), lo, hi))     S > 0, lo <= 0 <= hi, lo integral
    p  = f(q)                           any pure integer elementwise DAG of q and constants
    T  = rowsum(p)                      linalg.reduce addi over the last dimension

and, when it follows, the per-row symmetric int8 quantization of ``P = p / T`` that the next
contraction runs. Each rewrite is the same value as what it replaces, and why is stated where it
is done:

* ``p = f(q)`` becomes a read of a table holding ``f`` at every ``q`` in ``[lo, 0]``, computed here
  by evaluating the captured integer ops with the IR's own integer semantics (a domain point where
  one would be poison refuses the match). ``q <= 0`` because ``m`` is the row's maximum, so the
  clamp's upper bound never binds and is dropped; its lower bound becomes a compare-and-select,
  which agrees with the clamp on every non-NaN value and also sends NaN (a row with no finite
  maximum, where the original converts NaN to an integer, which is poison) to the bound -- so every
  table read is in bounds.
* the row sum accumulates in i32 when the table's largest magnitude times the row length cannot
  overflow it, then is widened back.
* ``P``'s per-row quantization depends on the row only through ``T``: ``P = k / T`` for ``k`` in the
  table's range, the row's largest ``P`` is ``max / T`` (the row maximum's ``q`` is 0, and the table
  is largest there -- checked), the smallest is clamped with 0 by the quantization (checked), and
  every per-element step reads ``P`` and per-row values only (checked). So the quantization runs once
  per row on the ``K`` candidate values and each element reads its int8 at ``p``.

Anything that does not match leaves the IR untouched and is reported with the reason.
"""

_IST_TOKEN = "OK int_softmax_table "
#: The largest grid a table may cover (entries). The integer softmax's grid is ~11k.
_IST_TABLE_LIMIT = 1 << 16
#: How far the forward search for the row sum may walk from ``q`` (elementwise ops).
_IST_SEARCH_LIMIT = 256
#: Integer ops the table evaluator implements, with the IR's semantics.
_IST_INT_OPS = (
    "arith.constant",
    "arith.addi",
    "arith.subi",
    "arith.muli",
    "arith.andi",
    "arith.ori",
    "arith.xori",
    "arith.shli",
    "arith.shrsi",
    "arith.shrui",
    "arith.floordivsi",
    "arith.ceildivsi",
    "arith.divsi",
    "arith.divui",
    "arith.remsi",
    "arith.remui",
    "arith.minsi",
    "arith.maxsi",
    "arith.minui",
    "arith.maxui",
    "arith.extsi",
    "arith.extui",
    "arith.trunci",
    "arith.cmpi",
    "arith.select",
)
#: Ops whose results may be erased when nothing reads them any more.
_IST_PURE = (
    "linalg.generic",
    "linalg.reduce",
    "tensor.empty",
    "tensor.splat",
    "tensor.collapse_shape",
    "tensor.expand_shape",
    "arith.constant",
)


class _IstRefusal(Exception):
    """A matched prefix whose remainder is not the integer softmax; the message says what differs."""


def _ist_ir():
    from torch_mlir import ir

    return ir


# ---- IR inspection ---------------------------------------------------------------------------


def _ist_operation(x):
    return getattr(x, "operation", x)


def _ist_producer(value):
    ir = _ist_ir()
    owner = value.owner
    if isinstance(owner, ir.Block):
        return None
    return _ist_operation(owner)


def _ist_uses(value):
    return [(_ist_operation(use.owner), int(use.operand_number)) for use in value.uses]


def _ist_users(value):
    return [op for op, _ in _ist_uses(value)]


def _ist_tensor(value):
    """``(shape, element type text)`` of a statically shaped ranked tensor, else None."""
    ir = _ist_ir()
    try:
        t = ir.RankedTensorType(value.type)
    except (TypeError, ValueError):
        return None
    if not t.has_static_shape:
        return None
    return [int(d) for d in t.shape], str(t.element_type)


def _ist_width(type_text):
    """The bit width of a signless integer type's text (``i64`` -> 64), else None."""
    if len(type_text) > 1 and type_text[0] == "i" and type_text[1:].isdecimal():
        return int(type_text[1:])
    return None


def _ist_attr_number(attr):
    ir = _ist_ir()
    for cls in (ir.IntegerAttr, ir.FloatAttr):
        try:
            return cls(attr).value
        except (TypeError, ValueError):
            pass
    try:
        dense = ir.DenseElementsAttr(attr)
    except (TypeError, ValueError):
        return None
    if not dense.is_splat:
        return None
    return _ist_attr_number(dense.get_splat_value())


def _ist_constant(value):
    """The number a scalar constant, or a splat tensor of one, holds; None for anything else."""
    op = _ist_producer(value)
    if op is None:
        return None
    if op.name == "tensor.splat" and len(op.operands) == 1:
        return _ist_constant(op.operands[0])
    if op.name != "arith.constant":
        return None
    return _ist_attr_number(op.attributes["value"])


class _IstGeneric:
    """A ``linalg.generic``'s operands, maps and body, read once."""

    def __init__(self, op):
        ir = _ist_ir()
        self.op = op
        n_out = len(op.results)
        operands = list(op.operands)
        self.inputs = operands[: len(operands) - n_out]
        self.outputs = operands[len(operands) - n_out :]
        self.maps = [ir.AffineMapAttr(a).value for a in op.attributes["indexing_maps"]]
        self.parallel = all("parallel" in str(a) for a in op.attributes["iterator_types"])
        self.block = op.regions[0].blocks[0]
        self.args = list(self.block.arguments)
        self.body = [_ist_operation(o) for o in self.block.operations]
        self.names = [o.name for o in self.body[:-1]]
        self.yielded = list(self.body[-1].operands)

    def arg_index(self, value):
        for i, arg in enumerate(self.args):
            if arg == value:
                return i
        return None

    def body_value(self, value):
        """A body operand as ``("arg", input index)`` or ``("const", number)``; None otherwise."""
        i = self.arg_index(value)
        if i is not None:
            if i >= len(self.inputs):
                return None
            c = _ist_constant(self.inputs[i])
            return ("const", c) if c is not None else ("arg", i)
        c = _ist_constant(value)
        return ("const", c) if c is not None else None

    def outputs_unread(self):
        return all(not list(self.args[len(self.inputs) + i].uses) for i in range(len(self.outputs)))


def _ist_generic(op):
    if op is None or op.name != "linalg.generic" or len(op.results) != 1:
        return None
    return _IstGeneric(op)


def _ist_identity(amap, rank):
    ir = _ist_ir()
    return amap == ir.AffineMap.get_identity(rank)


def _ist_rowbcast(amap, rank):
    """``(d0, ..., d{r-2}, 0)``: one value per row, read by every element of the row."""
    ir = _ist_ir()
    exprs = [ir.AffineDimExpr.get(i) for i in range(rank - 1)] + [ir.AffineConstantExpr.get(0)]
    return amap == ir.AffineMap.get(rank, 0, exprs)


def _ist_elementwise(g, rank):
    """All parallel, one result, every map the identity over ``rank`` dimensions."""
    return g is not None and g.parallel and all(_ist_identity(m, rank) for m in g.maps)


def _ist_single_op(g, name):
    """The generic's body is exactly one ``name`` op (plus constants) yielding its result."""
    real = [o for o in g.body[:-1] if o.name != "arith.constant"]
    if len(real) != 1 or real[0].name != name or len(g.yielded) != 1:
        return None
    if g.yielded[0] != real[0].results[0]:
        return None
    return real[0]


def _ist_reduce(op, rank, combiner):
    """``op`` is a ``linalg.reduce`` over the last of ``rank`` dims whose body is ``combiner``."""
    ir = _ist_ir()
    if op is None or op.name != "linalg.reduce" or len(op.results) != 1 or len(op.operands) != 2:
        return False
    dims = list(ir.DenseI64ArrayAttr(op.attributes["dimensions"]))
    if dims != [rank - 1]:
        return False
    body = [_ist_operation(o) for o in op.regions[0].blocks[0].operations]
    if [o.name for o in body] != [combiner, "linalg.yield"]:
        return False
    args = list(op.regions[0].blocks[0].arguments)
    operands = list(body[0].operands)
    return sorted(i for i, a in enumerate(args) if any(a == v for v in operands)) == [0, 1]


def _ist_back_through_reshapes(value, limit=8):
    for _ in range(limit):
        op = _ist_producer(value)
        if op is None or op.name not in ("tensor.collapse_shape", "tensor.expand_shape"):
            return value
        value = op.operands[0]
    return value


# ---- the table evaluator -----------------------------------------------------------------------


def _ist_signed(v, w):
    v &= (1 << w) - 1
    return v - (1 << w) if v >> (w - 1) else v


def _ist_unsigned(v, w):
    return v & ((1 << w) - 1)


def _ist_overflow_checked(op):
    try:
        flags = str(op.attributes["overflowFlags"])
    except KeyError:
        return False
    return "nsw" in flags or "nuw" in flags


def _ist_compile_slice(generics, root):
    """Compile the integer DAG ``generics`` (topological) into steps over integer slots.

    Slot 0 is the root's value. Returns ``(steps, slot of the last generic's result)``. A body op the
    evaluator does not implement, a non-integer value, or a body that reads its output refuses.
    """
    slots = {}
    steps = []
    slots[root] = 0
    count = 1

    def constant_slot(value, what):
        """A slot holding ``value``'s integer constant, read at its own width."""
        nonlocal count
        c = _ist_constant(value)
        tv = _ist_tensor(value)
        w = _ist_width(tv[1] if tv is not None else str(value.type))
        if c is None or isinstance(c, float) or w is None:
            raise _IstRefusal(what)
        steps.append(("const", count, (), 0, _ist_signed(int(c), w)))
        count += 1
        return count - 1

    def slot_of(value):
        for known, slot in slots.items():
            if known == value:
                return slot
        return constant_slot(value, "an operand of the integer DAG is neither computed in it nor an integer constant")

    last = None
    for g in generics:
        if not g.outputs_unread():
            raise _IstRefusal("an integer op reads its output buffer")
        local = []
        for i, arg in enumerate(g.args[: len(g.inputs)]):
            local.append((arg, slot_of(g.inputs[i])))
        for op in g.body[:-1]:
            if op.name not in _IST_INT_OPS:
                raise _IstRefusal(f"{op.name} in the integer DAG has no evaluator")
            ins = []
            for operand in op.operands:
                found = next((s for a, s in local if a == operand), None)
                if found is None:
                    found = constant_slot(operand, f"{op.name} reads a value the evaluator cannot resolve")
                ins.append(found)
            widths = []
            for v in [*op.operands, *op.results]:
                w = _ist_width(str(v.type))
                if w is None:
                    raise _IstRefusal(f"{op.name} on {v.type} is not a signless integer op")
                widths.append(w)
            extra = None
            if op.name == "arith.constant":
                extra = _ist_attr_number(op.attributes["value"])
                if extra is None or isinstance(extra, float):
                    raise _IstRefusal("a non-integer constant in the integer DAG")
                extra = _ist_signed(int(extra), widths[-1])
            elif op.name == "arith.cmpi":
                extra = int(_ist_ir().IntegerAttr(op.attributes["predicate"]).value)
            elif _ist_overflow_checked(op):
                extra = "checked"
            steps.append((op.name, count, tuple(ins), tuple(widths), extra))
            local.append((op.results[0], count))
            count += 1
        out = next((s for a, s in local if a == g.yielded[0]), None)
        if out is None:
            raise _IstRefusal("an integer op yields a value it does not compute")
        slots[g.op.results[0]] = out
        last = out
    return steps, last, count


def _ist_run(steps, size, q):
    v = [0] * size
    v[0] = q
    for name, out, ins, widths, extra in steps:
        if name == "const":
            v[out] = extra
            continue
        w = widths[-1]
        a = v[ins[0]] if ins else None
        b = v[ins[1]] if len(ins) > 1 else None
        if name == "arith.constant":
            r = extra
        elif name in ("arith.addi", "arith.subi", "arith.muli", "arith.shli"):
            if name == "arith.shli":
                if _ist_unsigned(b, w) >= w:
                    raise _IstRefusal("a shift amount reaches the width (poison) inside the domain")
                exact = a << _ist_unsigned(b, w)
            else:
                exact = a + b if name == "arith.addi" else a - b if name == "arith.subi" else a * b
            r = _ist_signed(exact, w)
            if extra == "checked" and r != exact:
                raise _IstRefusal(f"{name} overflows a no-wrap flag (poison) inside the domain")
        elif name in ("arith.andi", "arith.ori", "arith.xori"):
            ua, ub = _ist_unsigned(a, w), _ist_unsigned(b, w)
            r = _ist_signed(ua & ub if name == "arith.andi" else ua | ub if name == "arith.ori" else ua ^ ub, w)
        elif name in ("arith.shrsi", "arith.shrui"):
            amount = _ist_unsigned(b, w)
            if amount >= w:
                raise _IstRefusal("a shift amount reaches the width (poison) inside the domain")
            r = a >> amount if name == "arith.shrsi" else _ist_signed(_ist_unsigned(a, w) >> amount, w)
        elif name in ("arith.floordivsi", "arith.ceildivsi", "arith.divsi", "arith.remsi"):
            if b == 0 or (a == -(1 << (w - 1)) and b == -1):
                raise _IstRefusal(f"{name} divides by zero or overflows (poison) inside the domain")
            if name == "arith.floordivsi":
                r = a // b
            elif name == "arith.ceildivsi":
                r = -((-a) // b)
            else:
                t = abs(a) // abs(b)
                t = t if (a >= 0) == (b >= 0) else -t
                r = t if name == "arith.divsi" else a - b * t
            r = _ist_signed(r, w)
        elif name in ("arith.divui", "arith.remui"):
            ua, ub = _ist_unsigned(a, w), _ist_unsigned(b, w)
            if ub == 0:
                raise _IstRefusal(f"{name} divides by zero (poison) inside the domain")
            r = _ist_signed(ua // ub if name == "arith.divui" else ua % ub, w)
        elif name in ("arith.minsi", "arith.maxsi"):
            r = min(a, b) if name == "arith.minsi" else max(a, b)
        elif name in ("arith.minui", "arith.maxui"):
            pick_a = _ist_unsigned(a, w) <= _ist_unsigned(b, w)
            r = (a if pick_a else b) if name == "arith.minui" else (b if pick_a else a)
        elif name == "arith.extsi":
            r = a
        elif name == "arith.extui":
            r = _ist_signed(_ist_unsigned(a, widths[0]), w)
        elif name == "arith.trunci":
            r = _ist_signed(a, w)
        elif name == "arith.cmpi":
            wi = widths[0]
            sa, sb, ua, ub = a, b, _ist_unsigned(a, wi), _ist_unsigned(b, wi)
            r = (sa == sb, sa != sb, sa < sb, sa <= sb, sa > sb, sa >= sb, ua < ub, ua <= ub, ua > ub, ua >= ub)[extra]
            r = -1 if r else 0
        elif name == "arith.select":
            r = v[ins[1]] if a != 0 else v[ins[2]]
        else:
            raise _IstRefusal(f"{name} has no evaluator")
        v[out] = r
    return v


# ---- matching ----------------------------------------------------------------------------------


def _ist_match(sub_op):
    """Match one integer softmax from its ``x - rowmax(x)`` op; None when that prefix is not there.

    Returns the plan dict; raises :class:`_IstRefusal` for a softmax-shaped prefix whose remainder
    differs."""
    g_sub = _ist_generic(sub_op)
    if g_sub is None or len(g_sub.inputs) != 2 or not g_sub.parallel:
        return None
    sub = _ist_single_op(g_sub, "arith.subf")
    if sub is None or not g_sub.outputs_unread():
        return None
    x, mb = g_sub.inputs
    tx, tm = _ist_tensor(x), _ist_tensor(mb)
    if tx is None or tm is None:
        return None
    shape, ftype = tx
    rank = len(shape)
    if rank < 1 or tm[0] != shape[:-1] + [1]:
        return None
    if not (_ist_identity(g_sub.maps[0], rank) and _ist_rowbcast(g_sub.maps[1], rank)):
        return None
    if not (_ist_identity(g_sub.maps[2], rank) and list(sub.operands) == g_sub.args[:2]):
        return None
    red_max = _ist_producer(_ist_back_through_reshapes(mb))
    if not _ist_reduce(red_max, rank, "arith.maximumf") or red_max.operands[0] != x:
        return None
    # From here on the prefix IS ``x - rowmax(x)``: anything else differing is reported.
    plan = {"x": x, "shape": shape, "red_max": red_max}
    plan["max_attained"] = _ist_constant(red_max.operands[1]) == float("-inf")

    def single_user(value, what):
        users = _ist_users(value)
        if len(users) != 1:
            raise _IstRefusal(f"{what} has {len(users)} readers, not one")
        return users[0]

    def elementwise_one(op, name, what):
        g = _ist_generic(op)
        if not _ist_elementwise(g, rank) or not g.outputs_unread():
            raise _IstRefusal(f"{what} is not one elementwise op")
        inner = _ist_single_op(g, name)
        if inner is None:
            raise _IstRefusal(f"{what} is not {name}")
        return g, inner

    g_div, div = elementwise_one(single_user(g_sub.op.results[0], "x - max"), "arith.divf", "(x - max) / S")
    lhs, rhs = (g_div.body_value(v) for v in div.operands)
    if lhs != ("arg", 0) or rhs is None or rhs[0] != "const" or not (0 < rhs[1] < float("inf")):
        raise _IstRefusal("the exponent grid step is not a positive constant divisor")
    g_round, rnd = elementwise_one(single_user(g_div.op.results[0], "(x - max) / S"), "math.roundeven", "round")
    clamp_op = single_user(g_round.op.results[0], "the rounded grid index")
    g_clamp = _ist_generic(clamp_op)
    if not _ist_elementwise(g_clamp, rank) or len(g_clamp.inputs) != 1:
        raise _IstRefusal("the grid index clamp is not one elementwise op")
    real = [o for o in g_clamp.body[:-1] if o.name != "arith.constant"]
    if [o.name for o in real] != ["arith.maximumf", "arith.minimumf"] or g_clamp.yielded[0] != real[1].results[0]:
        raise _IstRefusal("the grid index clamp is not max-then-min")
    lo_v = [g_clamp.body_value(v) for v in real[0].operands]
    hi_v = [g_clamp.body_value(v) for v in real[1].operands]
    if lo_v[0] != ("arg", 0) or lo_v[1] is None or lo_v[1][0] != "const":
        raise _IstRefusal("the clamp's lower bound is not a constant")
    if real[1].operands[0] != real[0].results[0] or hi_v[1] is None or hi_v[1][0] != "const":
        raise _IstRefusal("the clamp's upper bound is not a constant")
    lo, hi = float(lo_v[1][1]), float(hi_v[1][1])
    if not (lo <= 0.0 <= hi) or lo != int(lo) or -lo + 1 > _IST_TABLE_LIMIT:
        raise _IstRefusal(f"the clamp [{lo}, {hi}] does not bound a table-sized grid at or below 0")
    g_cast, _ = elementwise_one(single_user(g_clamp.op.results[0], "the clamped grid index"), "arith.fptosi", "cast")
    qi = g_cast.op.results[0]
    qw = _ist_width(_ist_tensor(qi)[1])
    if qw is None or int(lo) < -(1 << (qw - 1)):
        raise _IstRefusal("the grid index type cannot hold the clamp's bound")
    plan.update(g_clamp=g_clamp, g_cast=g_cast, qi=qi, lo=int(lo), ftype=ftype)

    # The numerator: the nearest value downstream of q that a row sum reads.
    p = red_sum = None
    seen, frontier = [], [qi]
    while frontier and p is None and len(seen) < _IST_SEARCH_LIMIT:
        value = frontier.pop(0)
        for op in _ist_users(value):
            if _ist_reduce(op, rank, "arith.addi") and op.operands[0] == value:
                p, red_sum = value, op
                break
            g = _ist_generic(op)
            if _ist_elementwise(g, rank) and _ist_width(_ist_tensor(g.op.results[0])[1]) is not None:
                if all(other != g.op for other in seen):
                    seen.append(g.op)
                    frontier.append(g.op.results[0])
    if p is None:
        raise _IstRefusal("no row sum reads an integer function of the grid index")
    # Its backward slice must close on q and integer constants.
    order, visiting = [], []

    def visit(value):
        if value == qi or _ist_constant(value) is not None:
            return
        op = _ist_producer(value)
        if any(done.op == op for done in order):
            return
        if op is None or any(v == op for v in visiting):
            raise _IstRefusal("the numerator is not a function of the grid index alone")
        g = _ist_generic(op)
        if not _ist_elementwise(g, rank):
            raise _IstRefusal(f"the numerator reads {op.name}, not an elementwise integer op")
        visiting.append(op)
        for inp in g.inputs:
            visit(inp)
        visiting.pop()
        order.append(g)

    visit(p)
    if not order:
        raise _IstRefusal("the row sum reads the grid index itself")
    steps, out_slot, size = _ist_compile_slice(order, qi)
    table = [_ist_run(steps, size, q)[out_slot] for q in range(int(lo), 1)]
    plan.update(p=p, red_sum=red_sum, slice=order, table=table)
    return plan


def _ist_match_quantization(plan):
    """The per-row int8 quantization of ``p / T`` that follows; returns its plan or raises."""
    shape, p, red_sum, table = plan["shape"], plan["p"], plan["red_sum"], plan["table"]
    rank, n = len(shape), shape[-1]
    if not plan["max_attained"]:
        raise _IstRefusal("the row maximum's initial value is not -inf, so it need not be an element")
    if min(table) < 0 or max(table) <= 0 or table[-1] != max(table):
        raise _IstRefusal("the numerator table is not non-negative with its maximum at q = 0")
    # T as a float, one per row.
    tf = None
    for op in _ist_users(red_sum.results[0]):
        value = op.results[0] if op.name in ("tensor.collapse_shape", "tensor.expand_shape") else None
        while value is not None and _ist_tensor(value)[0] != shape[:-1] + [1]:
            nxt = [u for u in _ist_users(value) if u.name in ("tensor.collapse_shape", "tensor.expand_shape")]
            value = nxt[0].results[0] if len(nxt) == 1 else None
        if value is None:
            continue
        for user in _ist_users(value):
            g = _ist_generic(user)
            if _ist_elementwise(g, rank) and _ist_single_op(g, "arith.sitofp") is not None:
                tf = g.op.results[0]
    if tf is None:
        raise _IstRefusal("the row sum is not converted to a per-row float")
    probs = g_pf = g_pr = None
    for op in _ist_users(p):
        g = _ist_generic(op)
        if not _ist_elementwise(g, rank) or _ist_single_op(g, "arith.sitofp") is None:
            continue
        for user in _ist_users(g.op.results[0]):
            gd = _ist_generic(user)
            if gd is None or len(gd.inputs) != 2 or not gd.parallel or not gd.outputs_unread():
                continue
            div = _ist_single_op(gd, "arith.divf")
            if div is None or gd.inputs[0] != g.op.results[0] or gd.inputs[1] != tf:
                continue
            if not (_ist_identity(gd.maps[0], rank) and _ist_rowbcast(gd.maps[1], rank)):
                continue
            if _ist_identity(gd.maps[2], rank) and list(div.operands) == gd.args[:2]:
                probs, g_pf, g_pr = gd.op.results[0], g, gd
    if probs is None:
        raise _IstRefusal("the probabilities are not p / T")
    # Monotone value steps and row-preserving reshapes to the quantized operand.
    chain, value = [], probs
    while True:
        users = _ist_users(value)
        if len(users) != 1:
            break
        op = users[0]
        if op.name in ("tensor.collapse_shape", "tensor.expand_shape"):
            nxt = op.results[0]
        else:
            g = _ist_generic(op)
            vrank = len(_ist_tensor(value)[0])
            if not _ist_elementwise(g, vrank) or len(g.inputs) != 1 or not g.outputs_unread():
                break
            real = [o for o in g.body[:-1]]
            if real and not (len(real) == 1 and real[0].name in ("arith.truncf", "arith.extf")):
                break
            if not real and g.yielded[0] != g.args[0]:
                break
            nxt = g.op.results[0]
        tv = _ist_tensor(nxt)
        if tv is None or tv[0][-1] % n:
            raise _IstRefusal("a reshape of the probabilities moves the key dimension")
        chain.append(op)
        value = nxt
    xq = value
    sq = _ist_tensor(xq)[0]
    if sq[-1] != n:
        raise _IstRefusal("the quantized operand's last dimension is not the softmax row")
    rq = len(sq)
    red_min = red_max = head = None
    for op in _ist_users(xq):
        if _ist_reduce(op, rq, "arith.minimumf") and op.operands[0] == xq and red_min is None:
            red_min = op
        elif _ist_reduce(op, rq, "arith.maximumf") and op.operands[0] == xq and red_max is None:
            red_max = op
        elif head is None and _ist_generic(op) is not None:
            head = op
        else:
            raise _IstRefusal(f"the quantized operand is also read by {op.name}")
    if red_min is None or red_max is None or head is None:
        raise _IstRefusal("the probabilities are not quantized per row (min, max and a per-element step)")
    for op in _ist_users(red_min.results[0]):
        g = _ist_generic(op)
        mn = _ist_single_op(g, "arith.minimumf") if g is not None and g.parallel else None
        kinds = [g.body_value(v) for v in mn.operands] if mn is not None else []
        consts = [k[1] for k in kinds if k is not None and k[0] == "const"]
        if mn is None or len(consts) != 1 or not (consts[0] <= 0.0) or not g.outputs_unread():
            raise _IstRefusal("the row minimum is read other than as min(row minimum, c <= 0)")
    # The per-element steps: each reads the previous value (identity) and per-row values or constants.
    steps, value = [], xq
    op = head
    while True:
        g = _ist_generic(op)
        if g is None or not g.parallel or not g.outputs_unread() or not _ist_identity(g.maps[-1], rq):
            break
        ok, saw = True, False
        for inp, amap in zip(g.inputs, g.maps):
            tv = _ist_tensor(inp)
            if inp == value and _ist_identity(amap, rq):
                saw = True
            elif tv is not None and tv[0] == sq[:-1] + [1] and _ist_rowbcast(amap, rq):
                pass
            elif _ist_constant(inp) is not None and _ist_identity(amap, rq):
                pass
            else:
                ok = False
        if (
            not ok
            or not saw
            or any(not (o.name.startswith("arith.") or o.name.startswith("math.")) for o in g.body[:-1])
        ):
            break
        steps.append(g)
        value = g.op.results[0]
        users = _ist_users(value)
        if len(users) != 1:
            break
        op = users[0]
    if not steps or steps[0].op != head:
        raise _IstRefusal("the per-element quantization step does not read the operand and per-row values only")
    for g in steps[:-1]:
        if len(_ist_users(g.op.results[0])) != 1:
            raise _IstRefusal("an intermediate quantization value has other readers")
    return {
        "tf": tf,
        "g_pf": g_pf,
        "g_pr": g_pr,
        "chain": chain,
        "xq": xq,
        "sq": sq,
        "red_min": red_min,
        "red_max": red_max,
        "steps": steps,
        "k": max(table) + 1,
    }


# ---- rewriting ---------------------------------------------------------------------------------


def _ist_splice(ctx, anchor, args, body, result_types, *, after=False):
    """Parse ``body`` (MLIR, reading ``%a0..``) as a function, move its ops before ``anchor`` (after it,
    with ``after``) with its arguments bound to ``args``, and return the values it returns."""
    ir = _ist_ir()
    sig = ", ".join(f"%a{i}: {v.type}" for i, v in enumerate(args))
    rets = ", ".join(str(t) for t in result_types)
    text = "module {\nfunc.func @__ist_splice(" + sig + ") -> (" + rets + ") {\n" + body + "\n}\n}\n"
    tmp = ir.Module.parse(text, ctx)
    func = _ist_operation(tmp.body.operations[0])
    block = func.regions[0].blocks[0]
    ops = [_ist_operation(o) for o in block.operations]
    results = list(ops[-1].operands)
    ops[-1].erase()
    if after:
        for op in reversed(ops[:-1]):
            op.move_after(anchor)
    else:
        for op in ops[:-1]:
            op.move_before(anchor)
    for arg, value in zip(block.arguments, args):
        arg.replace_all_uses_with(value)
    return results


def _ist_tensor_type(shape, elem):
    return "tensor<" + "x".join(str(d) for d in shape) + ("x" if shape else "") + elem + ">"


def _ist_maps(rank, count):
    dims = ", ".join(f"d{i}" for i in range(rank))
    ident = f"affine_map<({dims}) -> ({dims})>"
    return "[" + ", ".join([ident] * count) + "]", "[" + ", ".join(['"parallel"'] * rank) + "]"


def _ist_head(maps, iters, ins, outs):
    """``linalg.generic {...} ins(...) outs(...) {`` for ``ins``/``outs`` lists of ``(name, type)``."""
    text = f"linalg.generic {{indexing_maps = {maps}, iterator_types = {iters}}}"
    for word, operands in (("ins", ins), ("outs", outs)):
        if operands:
            text += f" {word}(" + ", ".join(n for n, _ in operands) + " : " + ", ".join(t for _, t in operands) + ")"
    return text + " {"


def _ist_carry(src, dst):
    """The replaced op's provenance on its replacement, marked as this rewrite's."""
    ir = _ist_ir()
    for key in list(src.attributes):
        name = getattr(key, "name", key)
        if name.startswith("prov."):
            dst.attributes[name] = src.attributes[name]
    dst.attributes["prov.rewrite"] = ir.StringAttr.get("int_softmax_table")


def _ist_narrow_type(values):
    lo, hi = min(values), max(values)
    for w in (8, 16, 32, 64):
        if -(1 << (w - 1)) <= lo and hi < (1 << (w - 1)):
            return w
    raise _IstRefusal("the table's values do not fit 64 bits")


def _ist_float_text(v):
    return repr(float(v)) if v == v and v not in (float("inf"), float("-inf")) else None


def _ist_rewrite(ctx, plan, quant):
    ir = _ist_ir()
    shape, rank = plan["shape"], len(plan["shape"])
    lo, table = plan["lo"], plan["table"]
    g_clamp, g_cast, qi, p, red_sum = plan["g_clamp"], plan["g_cast"], plan["qi"], plan["p"], plan["red_sum"]
    ptype = _ist_tensor(p)[1]
    pw = _ist_width(ptype)
    qtype = _ist_tensor(qi)[1]
    n = shape[-1]
    init = _ist_constant(red_sum.operands[1])
    bound = max(abs(min(table)), abs(max(table))) * n + abs(int(init or 0))
    narrow = init is not None and not isinstance(init, float) and bound < (1 << 31) and pw >= 32
    gw = 32 if narrow else pw
    gtype = f"i{gw}"
    kw = _ist_narrow_type(table)
    maps1, iters = _ist_maps(rank, 2)
    report = {"table_entries": len(table), "table_bits": kw, "int32_sum": narrow, "row_quantization": False}

    # 1. the clamp as compare-and-select; the cast reads it.
    lo_text = _ist_float_text(lo)
    ft = plan["ftype"]
    (sel,) = _ist_splice(
        ctx,
        g_cast.op,
        [g_clamp.inputs[0]],
        f"""  %lo = arith.constant {lo_text} : {ft}
  %e = tensor.empty() : {_ist_tensor_type(shape, ft)}
  %r = {_ist_head(maps1, iters, [("%a0", _ist_tensor_type(shape, ft))], [("%e", _ist_tensor_type(shape, ft))])}
  ^bb0(%v: {ft}, %o: {ft}):
    %c = arith.cmpf oge, %v, %lo : {ft}
    %s = arith.select %c, %v, %lo : {ft}
    linalg.yield %s : {ft}
  }} -> {_ist_tensor_type(shape, ft)}
  return %r : {_ist_tensor_type(shape, ft)}""",
        [g_clamp.op.results[0].type],
    )
    _ist_carry(g_clamp.op, _ist_producer(sel))
    g_cast.op.operands[0] = sel

    # 2. the table read: tab[q - lo], at the gather's width.
    values = ", ".join(str(v) for v in table)
    widen = f"%w = arith.extsi %t : i{kw} to {gtype}" if kw < gw else ""
    out = "%w" if kw < gw else "%t"
    (pg,) = _ist_splice(
        ctx,
        g_cast.op,
        [qi],
        f"""  %tab = arith.constant dense<[{values}]> : tensor<{len(table)}xi{kw}>
  %lo = arith.constant {lo} : {qtype}
  %e = tensor.empty() : {_ist_tensor_type(shape, gtype)}
  %r = {_ist_head(maps1, iters, [("%a0", _ist_tensor_type(shape, qtype))], [("%e", _ist_tensor_type(shape, gtype))])}
  ^bb0(%q: {qtype}, %o: {gtype}):
    %d = arith.subi %q, %lo : {qtype}
    %i = arith.index_cast %d : {qtype} to index
    %t = tensor.extract %tab[%i] : tensor<{len(table)}xi{kw}>
    {widen}
    linalg.yield {out} : {gtype}
  }} -> {_ist_tensor_type(shape, gtype)}
  return %r : {_ist_tensor_type(shape, gtype)}""",
        [ir.RankedTensorType.get(shape, ir.IntegerType.get_signless(gw))],
        after=True,
    )
    _ist_carry(p.owner.operation if hasattr(p.owner, "operation") else p.owner, _ist_producer(pg))
    # 3. p at its own type for every other reader.
    if gw != pw:
        (p_wide,) = _ist_splice(
            ctx,
            _ist_producer(pg),
            [pg],
            f"""  %e = tensor.empty() : {_ist_tensor_type(shape, ptype)}
  %r = {_ist_head(maps1, iters, [("%a0", _ist_tensor_type(shape, gtype))], [("%e", _ist_tensor_type(shape, ptype))])}
  ^bb0(%v: {gtype}, %o: {ptype}):
    %w = arith.extsi %v : {gtype} to {ptype}
    linalg.yield %w : {ptype}
  }} -> {_ist_tensor_type(shape, ptype)}
  return %r : {_ist_tensor_type(shape, ptype)}""",
            [p.type],
            after=True,
        )
    else:
        p_wide = pg
    # 4. the row sum, at the gather's width, widened back.
    rows = shape[:-1]
    if narrow:
        maps_r, iters_r = _ist_maps(len(rows), 2)
        (t_wide,) = _ist_splice(
            ctx,
            red_sum,
            [pg],
            f"""  %c = arith.constant {int(init)} : {gtype}
  %init = tensor.splat %c : {_ist_tensor_type(rows, gtype)}
  %s = linalg.reduce ins(%a0 : {_ist_tensor_type(shape, gtype)})
    outs(%init : {_ist_tensor_type(rows, gtype)}) dimensions = [{rank - 1}]
    (%x: {gtype}, %y: {gtype}) {{
      %z = arith.addi %x, %y : {gtype}
      linalg.yield %z : {gtype}
    }}
  %e = tensor.empty() : {_ist_tensor_type(rows, ptype)}
  %r = {_ist_head(maps_r, iters_r, [("%s", _ist_tensor_type(rows, gtype))], [("%e", _ist_tensor_type(rows, ptype))])}
  ^bb0(%v: {gtype}, %o: {ptype}):
    %w = arith.extsi %v : {gtype} to {ptype}
    linalg.yield %w : {ptype}
  }} -> {_ist_tensor_type(rows, ptype)}
  return %r : {_ist_tensor_type(rows, ptype)}""",
            [red_sum.results[0].type],
        )
        _ist_carry(red_sum, _ist_producer(t_wide))
        red_sum.results[0].replace_all_uses_with(t_wide)
    for op, index in _ist_uses(p):
        if op != red_sum:
            op.operands[index] = p_wide
    if not narrow:
        red_sum.operands[0] = p_wide

    if quant is not None:
        _ist_rewrite_quantization(ctx, plan, quant, pg)
        report["row_quantization"] = True
    return report


def _ist_rewrite_quantization(ctx, plan, quant, pg):
    ir = _ist_ir()
    shape, rank, n = plan["shape"], len(plan["shape"]), plan["shape"][-1]
    k, sq, rq = quant["k"], quant["sq"], len(quant["sq"])
    ptype = _ist_tensor(plan["p"])[1]
    first = quant["red_min"]
    for op in (quant["red_max"], quant["steps"][0].op):
        if op.is_before_in_block(first):
            first = op

    def widened(t, elem=None):
        tv = _ist_tensor(t)
        return ir.RankedTensorType.get(
            tv[0][:-1] + [tv[0][-1] // n * k],
            ir.RankedTensorType(t.type).element_type if elem is None else elem,
        )

    def empty_like(anchor, rtype):
        (e,) = _ist_splice(ctx, anchor, [], f"  %e = tensor.empty() : {rtype}\n  return %e : {rtype}", [rtype])
        return e

    def clone_onto(op, anchor, replacements, rtype):
        new = op.clone(ip=ir.InsertionPoint(anchor))
        for index, value in replacements.items():
            new.operands[index] = value
        new.results[0].set_type(rtype)
        if new.name == "tensor.expand_shape":
            static = list(ir.DenseI64ArrayAttr(new.attributes["static_output_shape"]))
            new.attributes["static_output_shape"] = ir.DenseI64ArrayAttr.get(static[:-1] + [static[-1] // n * k])
        if new.name in ("linalg.generic",):
            new.operands[len(new.operands) - 1] = empty_like(new, rtype)
        _ist_carry(op, new)
        return new

    # The K candidates k / T per row, by the probabilities' own ops on k.
    kshape = shape[:-1] + [k]
    maps1, iters = _ist_maps(rank, 1)
    (kidx,) = _ist_splice(
        ctx,
        first,
        [],
        f"""  %e = tensor.empty() : {_ist_tensor_type(kshape, ptype)}
  %r = {_ist_head(maps1, iters, [], [("%e", _ist_tensor_type(kshape, ptype))])}
  ^bb0(%o: {ptype}):
    %i = linalg.index {rank - 1} : index
    %c = arith.index_cast %i : index to {ptype}
    linalg.yield %c : {ptype}
  }} -> {_ist_tensor_type(kshape, ptype)}
  return %r : {_ist_tensor_type(kshape, ptype)}""",
        [ir.RankedTensorType.get(kshape, ir.RankedTensorType(plan["p"].type).element_type)],
    )
    g_pf, g_pr = quant["g_pf"], quant["g_pr"]
    pf = clone_onto(g_pf.op, first, {0: kidx}, widened(g_pf.op.results[0]))
    cand = clone_onto(g_pr.op, first, {0: pf.results[0], 1: quant["tf"]}, widened(g_pr.op.results[0])).results[0]
    index_view = pg
    for op in quant["chain"]:
        rtype = widened(op.results[0])
        cand = clone_onto(op, first, {0: cand}, rtype).results[0]
    quant["red_min"].operands[0] = cand
    quant["red_max"].operands[0] = cand
    # The per-element steps run on the candidates.
    steps = quant["steps"]
    last = steps[-1].op
    old_uses = _ist_uses(last.results[0])
    value_old, value_new = quant["xq"], cand
    for g in steps:
        op = g.op
        rtype = widened(op.results[0])
        for index, inp in enumerate(g.inputs):
            if inp == value_old:
                op.operands[index] = value_new
            elif _ist_tensor(inp)[0] == quant["sq"]:
                c = _ist_producer(inp)
                splat_t = ir.RankedTensorType.get(quant["sq"][:-1] + [k], ir.RankedTensorType(inp.type).element_type)
                (s,) = _ist_splice(
                    ctx,
                    op,
                    [c.operands[0]] if c.name == "tensor.splat" else [],
                    (f"  %s = tensor.splat %a0 : {splat_t}\n  return %s : {splat_t}")
                    if c.name == "tensor.splat"
                    else f"  %s = arith.constant {c.attributes['value']}\n  return %s : {splat_t}",
                    [splat_t],
                )
                quant.setdefault("dead", []).append(c)
                op.operands[index] = s
        quant.setdefault("dead", []).append(_ist_producer(op.operands[len(op.operands) - 1]))
        op.operands[len(op.operands) - 1] = empty_like(op, rtype)
        value_old = op.results[0]
        op.results[0].set_type(rtype)
        value_new = op.results[0]
    qc = last.results[0]
    # The element's int8 at its numerator: the numerator viewed as the operand is.
    anchor = last
    for op in quant["chain"]:
        if op.name in ("tensor.collapse_shape", "tensor.expand_shape"):
            new = op.clone(ip=ir.InsertionPoint(last))  # a detached op cannot be moved
            new.move_after(anchor)
            anchor = new
            new.operands[0] = index_view
            tv = _ist_tensor(op.results[0])
            new.results[0].set_type(ir.RankedTensorType.get(tv[0], ir.RankedTensorType(pg.type).element_type))
            index_view = new.results[0]
    gtype = _ist_tensor(pg)[1]
    dtype = str(ir.RankedTensorType(qc.type).element_type)
    idx = "\n    ".join(f"%i{j} = linalg.index {j} : index" for j in range(rq - 1))
    coords = ", ".join([f"%i{j}" for j in range(rq - 1)] + ["%k"])
    maps2, iters2 = _ist_maps(rq, 2)
    (gathered,) = _ist_splice(
        ctx,
        anchor,
        [index_view, qc],
        f"""  %e = tensor.empty() : {_ist_tensor_type(sq, dtype)}
  %r = {_ist_head(maps2, iters2, [("%a0", _ist_tensor_type(sq, gtype))], [("%e", _ist_tensor_type(sq, dtype))])}
  ^bb0(%p: {gtype}, %o: {dtype}):
    {idx}
    %k = arith.index_cast %p : {gtype} to index
    %v = tensor.extract %a1[{coords}] : {_ist_tensor_type(sq[:-1] + [k], dtype)}
    linalg.yield %v : {dtype}
  }} -> {_ist_tensor_type(sq, dtype)}
  return %r : {_ist_tensor_type(sq, dtype)}""",
        [ir.RankedTensorType.get(sq, ir.RankedTensorType(qc.type).element_type)],
        after=True,
    )
    _ist_carry(last, _ist_producer(gathered))
    for op, index in old_uses:
        op.operands[index] = gathered


def _ist_match_scale(plan):
    """A constant elementwise step on the scores (the attention scale) applied BEHIND a reshape.

    Walks up from the softmax input through elementwise ops' first operand. The step found must read
    a pure reshape of a ``linalg.generic`` result and constants only; applied before the reshape it is
    the same op on the same values, and it can then fuse into the op that produced them. Returns
    ``(step, reshapes, source)`` or None."""
    value = plan["x"]
    for _ in range(4):
        op = _ist_producer(value)
        g = _ist_generic(op)
        if g is None or not g.parallel or not g.outputs_unread() or not g.inputs:
            return None
        rank = len(_ist_tensor(value)[0])
        if not all(_ist_identity(m, rank) for m in g.maps[:1] + g.maps[-1:]):
            return None
        first = g.inputs[0]
        others_const = all(_ist_constant(v) is not None for v in g.inputs[1:])
        reshapes, src = [], first
        while True:
            p = _ist_producer(src)
            if p is None or p.name not in ("tensor.collapse_shape", "tensor.expand_shape"):
                break
            if len(_ist_users(src)) != 1:
                return None
            reshapes.insert(0, p)
            src = p.operands[0]
        if (
            others_const
            and reshapes
            and all(_ist_identity(m, rank) for m in g.maps)
            and len(_ist_users(first)) == 1
            and _ist_generic(_ist_producer(src)) is not None
            and all(o.name.startswith("arith.") or o.name.startswith("math.") for o in g.body[:-1])
        ):
            return g, reshapes, src
        value = first
    return None


def _ist_rewrite_scale(ctx, match):
    """Apply the constant step to the reshape's source and reshape its result instead."""
    ir = _ist_ir()
    g, reshapes, src = match
    op = g.op
    shape = _ist_tensor(src)[0]
    elem = ir.RankedTensorType(op.results[0].type).element_type
    rtype = ir.RankedTensorType.get(shape, elem)
    old_uses = _ist_uses(op.results[0])
    new = op.clone(ip=ir.InsertionPoint(op))
    new.operands[0] = src
    for index, inp in enumerate(g.inputs[1:], start=1):
        c = _ist_producer(inp)
        stype = ir.RankedTensorType.get(shape, ir.RankedTensorType(inp.type).element_type)
        if c is not None and c.name == "tensor.splat":
            (s,) = _ist_splice(
                ctx, new, [c.operands[0]], f"  %s = tensor.splat %a0 : {stype}\n  return %s : {stype}", [stype]
            )
        else:
            (s,) = _ist_splice(
                ctx, new, [], f"  %s = arith.constant {c.attributes['value']}\n  return %s : {stype}", [stype]
            )
        new.operands[index] = s
    (e,) = _ist_splice(ctx, new, [], f"  %e = tensor.empty() : {rtype}\n  return %e : {rtype}", [rtype])
    new.operands[len(new.operands) - 1] = e
    new.results[0].set_type(rtype)
    # Every map was the identity (checked), so the step is the identity over the source's rank.
    rank = len(shape)
    new.attributes["indexing_maps"] = ir.ArrayAttr.get(
        [ir.AffineMapAttr.get(ir.AffineMap.get_identity(rank))] * len(new.operands)
    )
    new.attributes["iterator_types"] = ir.ArrayAttr.get([ir.Attribute.parse("#linalg.iterator_type<parallel>")] * rank)
    _ist_carry(op, new)
    value = new.results[0]
    for r in reshapes:
        moved = r.clone(ip=ir.InsertionPoint(op))
        moved.operands[0] = value
        moved.results[0].set_type(ir.RankedTensorType.get(_ist_tensor(r.results[0])[0], elem))
        value = moved.results[0]
    for user, index in old_uses:
        user.operands[index] = value
    return [op, *reversed(reshapes)]


def _ist_erase_dead(ops):
    """Erase every op of ``ops`` (and what only they read) that nothing reads any more."""
    pending = list(ops)
    while pending:
        progressed = False
        for op in list(pending):
            if op.name not in _IST_PURE:
                pending.remove(op)
                continue
            if any(list(r.uses) for r in op.results):
                continue
            feeders = [_ist_producer(v) for v in op.operands]
            pending.remove(op)
            op.erase()
            pending.extend(f for f in feeders if f is not None and all(f != q for q in pending))
            progressed = True
        if not progressed:
            break


def _int_softmax_table(ctx, module):
    """Rewrite every integer softmax in ``module``; returns the report dict the runner prints."""
    with ctx, _ist_ir().Location.unknown():
        report = {"softmax": 0, "int32_sums": 0, "row_quantizations": 0, "scales_moved": 0, "tables": [], "refused": []}
        # Every match is made before anything is edited, and nothing is erased until every rewrite is
        # done: an erased op's binding object is invalid, so no plan may still hold one.
        plans = []
        for op in [op for op in _ist_walk(module.operation) if op.name == "linalg.generic"]:
            try:
                plan = _ist_match(op)
            except _IstRefusal as why:
                report["refused"].append(str(why))
                continue
            if plan is None:
                continue
            quant = None
            try:
                quant = _ist_match_quantization(plan)
            except _IstRefusal as why:
                report["refused"].append("row quantization: " + str(why))
            plans.append((plan, quant, _ist_match_scale(plan)))
        dead = []
        for plan, quant, scale in plans:
            done = _ist_rewrite(ctx, plan, quant)
            report["softmax"] += 1
            report["int32_sums"] += int(done["int32_sum"])
            report["row_quantizations"] += int(done["row_quantization"])
            report["tables"].append([done["table_entries"], done["table_bits"]])
            dead += [plan["red_sum"], plan["g_clamp"].op, *(g.op for g in reversed(plan["slice"]))]
            if quant is not None:
                dead += [quant["g_pr"].op, *quant["chain"], *(op for op in quant.get("dead", ()) if op is not None)]
            if scale is not None:
                dead += _ist_rewrite_scale(ctx, scale)
                report["scales_moved"] += 1
        _ist_erase_dead(dead)
        module.operation.verify()
    return report


def _ist_walk(op):
    for region in op.regions:
        for block in region.blocks:
            for inner in list(block.operations):
                inner = _ist_operation(inner)
                yield inner
                yield from _ist_walk(inner)
