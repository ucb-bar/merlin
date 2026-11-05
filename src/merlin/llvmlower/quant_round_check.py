"""Independent expression reconstruction for bounded quantization legalization.

Only pure, straight-line typed scalar bodies are interpreted. Outer operations,
interfaces, effects and dataflow must remain structurally identical. This is a
conditional arithmetic check at tensor IR, not machine-code or hardware proof.
"""

from __future__ import annotations

import hashlib
import json
import math

from xdsl.dialects import arith, builtin
from xdsl.ir import Operation
from xdsl.traits import get_effects

from merlin.common.mlir_query import op_name


def bounded_round_input(value, integer_type):
    """Read literal integral clamp bounds for a defined narrow conversion.

    No producer, source label or selected plan supplies these bounds. Unknown
    scalars stop the walk, and both bounds must already have been established.
    """
    from .quant_round_chain import _float_width, _int_range, _integral_limit

    if not isinstance(integer_type, builtin.IntegerType) or integer_type.width.data < 2:
        return False
    if _float_width(value.type) is None:
        return False
    lo, hi = -math.inf, math.inf
    current = value
    for _ in range(8):
        owner = current.owner
        if not isinstance(owner, Operation) or owner.name not in {"arith.minimumf", "arith.maximumf"}:
            break
        for constant, other in (owner.operands, tuple(reversed(owner.operands))):
            producer = constant.owner
            attr = producer.properties.get("value") if isinstance(producer, Operation) else None
            if (isinstance(producer, Operation) and producer.name == "arith.constant"
                    and isinstance(attr, builtin.FloatAttr)):
                bound = attr.value.data
                if not math.isfinite(bound) or bound != int(bound):
                    return False
                if owner.name == "arith.minimumf":
                    hi = min(hi, bound)
                else:
                    lo = max(lo, bound)
                current = other
                break
        else:
            break
    imin, imax = _int_range(integer_type)
    limit = _integral_limit(value.type)
    return imin <= lo <= hi <= imax and abs(lo) < limit and abs(hi) < limit


def residual_round_expression(cur, ft, typ):
    """Declarative ties-even identity, independent of the instruction emitter."""
    def floating(number):
        return _term("arith.constant", ft, parameter=str(builtin.FloatAttr(number, ft)))

    def integer(number):
        return _term("arith.constant", typ, parameter=str(builtin.IntegerAttr(number, typ)))

    def node(name, dtype, *args, parameter=None):
        return _term("arith." + name, dtype, args, parameter)

    truncated = node("fptosi", typ, cur)
    residual = node("subf", ft, cur, node("sitofp", ft, truncated))
    magnitude = node("maximumf", ft, residual, node("negf", ft, residual))
    greater = node("cmpf", builtin.i1, magnitude, floating(0.5), parameter=2)
    tie = node("cmpf", builtin.i1, magnitude, floating(0.5), parameter=1)
    odd = node("cmpi", builtin.i1, node("andi", typ, truncated, integer(1)), integer(0), parameter=1)
    bump = node("ori", builtin.i1, greater, node("andi", builtin.i1, tie, odd))
    negative = node("cmpf", builtin.i1, cur, floating(0.0), parameter=4)
    direction = node("select", typ, negative, integer(-1), integer(1))
    return node("addi", typ, truncated, node("select", typ, bump, direction, integer(0)))


def pure_scalar_body(block, *, bounded_intrinsics=False, integer_extensions=False):
    """Canonical value semantics exclude opaque effects and constrained FP."""
    if not block.ops or block.last_op.name != "linalg.yield":
        return False
    for op in block.ops:
        if op.name == "linalg.yield":
            continue
        if bounded_intrinsics and op_name(op) == "llvm.intr.roundeven":
            if (len(op.operands) != 1 or len(op.results) != 1 or op.operands[0].type != op.results[0].type
                    or op.regions or op.successors or op.properties or set(op.attributes) - {"op_name__"}):
                return False
            continue
        # These registered widening operations lack an effect trait in some
        # xDSL versions. The optional source family admits only their exact
        # integer value semantics; unknown or narrowing operations do not gain
        # purity from a name prefix.
        extension = (integer_extensions and type(op) in (arith.ExtSIOp, arith.ExtUIOp)
                     and len(op.operands) == len(op.results) == 1
                     and all(type(value.type) is builtin.IntegerType for value in (*op.operands, *op.results))
                     and op.operands[0].type.width.data < op.results[0].type.width.data)
        if (
            isinstance(op, builtin.UnregisteredOp)
            or op.regions
            or op.successors
            or len(op.results) != 1
            or get_effects(op) != set() and not extension
            or not op.name.startswith(("arith.", "math."))
        ):
            return False
        for name, attr in (*op.attributes.items(), *op.properties.items()):
            if name.startswith("prov.") or name in {"value", "predicate"}:
                continue
            if name in {"fastmath", "overflowFlags"} and hasattr(attr, "data") and not attr.data:
                continue
            return False
    return True


def _term(name, typ, operands=(), parameter=None):
    return hashlib.sha256(json.dumps((name, str(typ), operands, parameter), separators=(",", ":")).encode()).hexdigest()


def _outputs(block, external, *, legalize, bounded_intrinsics=False, values=None):
    from .quant_round_chain import _match_chain

    memo = {arg: _term("argument", arg.type, parameter=i) for i, arg in enumerate(block.args)}

    def expression(value):
        if value in memo:
            return memo[value]
        owner = value.owner
        if not isinstance(owner, Operation) or owner.parent is not block:
            if value not in external:
                raise ValueError("quantization checker encountered an unmapped external value")
            return external[value]
        typ = value.type
        if bounded_intrinsics and owner.name == "arith.fptosi":
            rounded = owner.operands[0].owner
            if isinstance(rounded, Operation) and op_name(rounded) == "llvm.intr.roundeven":
                operand = rounded.operands[0]
                if not bounded_round_input(operand, typ):
                    raise ValueError("intrinsic rounding lacks actual bounded integer observations")
                answer = residual_round_expression(expression(operand), operand.type, typ)
                memo[value] = answer
                return answer
        if legalize and owner.name == "arith.fptosi":
            matched = _match_chain(owner)
            if not isinstance(matched, str):
                rounding, clamps, _lo, _hi = matched
                ft = rounding.operands[0].type
                cur = expression(rounding.operands[0])

                def floating(number):
                    return _term("arith.constant", ft, parameter=str(builtin.FloatAttr(number, ft)))

                # A separate declarative expression, not replay of the emitter.
                # The original matcher supplies only admitted bounds and order.
                for name, bound, _operation in reversed(clamps):
                    if name != "arith.addf":
                        cur = _term(name, ft, (cur, floating(bound)))
                answer = residual_round_expression(cur, ft, typ)
                memo[value] = answer
                return answer
        parameter = None
        if owner.name == "arith.constant":
            parameter = str(owner.properties["value"])
        elif owner.name in {"arith.cmpi", "arith.cmpf"}:
            parameter = owner.properties["predicate"].value.data
        answer = _term(owner.name, typ, tuple(expression(arg) for arg in owner.operands), parameter)
        memo[value] = answer
        return answer

    return tuple(expression(value) for value in (block.last_op.operands if values is None else values))


def verify_quantization_rewrite(before, after, *, bounded_intrinsics=False):
    """Reconstruct actual output expressions and all unchanged outer effects."""
    before.verify()
    after.verify()
    from merlin.transforms.certificates import ir_text

    if ir_text(before) == ir_text(after):
        return
    correspondence = {}
    external_before = {}
    external_after = {}
    serial = 0

    def bind(left, right):
        nonlocal serial
        if left.type != right.type:
            raise ValueError("quantization legalization changed a value type")
        correspondence[left] = right
        identity = _term("outer", left.type, parameter=serial)
        external_before[left] = identity
        external_after[right] = identity
        serial += 1

    def block(left, right, scalar=False):
        if len(left.args) != len(right.args):
            raise ValueError("quantization legalization changed block arguments")
        for a, b in zip(left.args, right.args, strict=True):
            bind(a, b)
        if scalar and pure_scalar_body(left) and pure_scalar_body(right, bounded_intrinsics=bounded_intrinsics):
            actual = _outputs(right, external_after, legalize=False, bounded_intrinsics=bounded_intrinsics)
            if (
                _outputs(left, external_before, legalize=False) != actual
                and _outputs(left, external_before, legalize=True) != actual
            ):
                raise ValueError("emitted quantization arithmetic differs from the admitted source expression")
            return
        a_ops, b_ops = tuple(left.ops), tuple(right.ops)
        if len(a_ops) != len(b_ops):
            raise ValueError("quantization legalization changed unmodeled operations or effects")
        for a, b in zip(a_ops, b_ops, strict=True):
            operation(a, b)

    def operation(left, right):
        if (
            left.name != right.name
            or left.attributes != right.attributes
            or left.properties != right.properties
            or left.successors
            or right.successors
            or len(left.results) != len(right.results)
            or len(left.regions) != len(right.regions)
            or tuple(correspondence.get(value) for value in left.operands) != tuple(right.operands)
        ):
            raise ValueError("quantization legalization changed outer interfaces, semantics or effects")
        for a, b in zip(left.results, right.results, strict=True):
            bind(a, b)
        for a, b in zip(left.regions, right.regions, strict=True):
            if len(a.blocks) != len(b.blocks):
                raise ValueError("quantization legalization changed control flow")
            for x, y in zip(a.blocks, b.blocks, strict=True):
                block(x, y, scalar=left.name == "linalg.generic")

    operation(before, after)
