"""A semantics for the ``linalg`` SOURCE module, given by the same lowering-to-``smt`` idea.

This is the missing half of translation validation. :mod:`merlin.verify.smt_semantics` encodes the
``interface`` program a pass *produced*; this module encodes the ``linalg`` program that pass
*consumed*. With both, :func:`merlin.verify.refine.validate_pass` states a per-compilation theorem
about the pass itself — *this source and this output compute the same function on every integer
input at this shape* — rather than the weaker target-vs-respecification claim that comes from
re-deriving what the pass "should have done" from the function signature.

**Why the distinction is not pedantry.** A specification re-derived from the signature shares the
compiler author's mental model of the workload. If that model is wrong — the workload is not actually
``A @ W``, or the zero points are not actually zero — a re-derived spec and a buggy pass agree, and
the query returns ``unsat`` while the program is wrong. Encoding the source IR removes the author
from the loop: the only inputs are the two artifacts.

**Scope.** Same as the interface encoder, and for the same reasons: rank-2 integer tensors, concrete
extents taken from the IR's own types (so every query stays quantifier-free), no poison, no memory
model, no float. Anything outside it raises :class:`UnsupportedSemantics` naming the construct —
never skipped, because a skipped op is a silently weakened theorem.

**Fail closed on zero points in particular.** ``linalg.quantized_matmul`` computes
``out[m][n] += (a[m][k] - zp_a) * (w[k][n] - zp_w)``. Reading the zero points off the actual operands
is the whole point: assuming they are zero would reintroduce the author's model through the back
door, and would silently validate a pass that dropped a *non-zero* zero point — the exact defect the
op exists to express. A zero point that is not a resolvable integer constant is an ABSTENTION.
"""
from __future__ import annotations

from typing import Any

from .smt_semantics import (Encoded, Encoder, Tensor, UnsupportedSemantics, _elem_width, _shape)

#: Ops with a value semantics here. Listed so a reader can see the encoded surface at a glance; the
#: walk below is still the authority, and anything absent raises rather than falling through.
ENCODABLE_OPS = frozenset({
    "linalg.quantized_matmul", "linalg.matmul", "tensor.empty", "arith.constant", "func.return",
})


def _const_int(op) -> int:
    """The integer an ``arith.constant`` carries, or raise.

    Read structurally off the value attribute rather than off the printed form: a float or an index
    constant has no bitvector reading here and must abstain instead of being coerced.
    """
    attr = op.properties.get("value")
    if attr is None:
        attr = op.attributes.get("value")
    value = getattr(attr, "value", None)
    data = getattr(value, "data", None)
    if not isinstance(data, int) or isinstance(data, bool):
        raise UnsupportedSemantics(
            f"arith.constant carries {attr!r}, which has no integer bitvector reading here")
    return data


class _Uninitialized:
    """Marker for a ``tensor.empty`` result: a destination-passing init with no value.

    ``linalg`` adds into its ``outs`` operand, so the init genuinely participates in the arithmetic
    when it holds a value. ``tensor.empty`` holds none — it is an allocation, and MLIR gives its
    contents no definition — so the contraction's accumulator starts from nothing, which is
    additively the same as starting from zero. That is a definition this module makes explicitly
    rather than a fact it reads off the IR, and it is the ONE place here where the encoding is a
    choice; it is named so a reviewer can see it instead of having to find it.
    """

    __slots__ = ("rows", "cols", "width")

    def __init__(self, rows: int, cols: int, width: int) -> None:
        self.rows, self.cols, self.width = rows, cols, width


def _sub_const(enc: Encoder, term, value: int, width: int):
    """``term - value`` at ``width`` bits, via addition of the negated constant.

    The ``smt`` dialect xDSL ships has no subtract op, and the zero point is a known integer, so the
    subtraction is done as ``term + (-value)`` — exact in two's complement, mod 2**width, which is
    exactly what ``arith.subi`` on an ``i<width>`` does.
    """
    if not -(2 ** (width - 1)) <= value < 2 ** (width - 1):
        raise UnsupportedSemantics(
            f"zero point {value} is not a signed {width}-bit value")
    return enc.smt.BVAddOp(term, enc.const((-value) & ((1 << width) - 1), width)).results[0]


def _contract(enc: Encoder, lhs: Tensor, rhs: Tensor, zp_lhs: int, zp_rhs: int,
              acc_width: int) -> Tensor:
    """``out[m][n] = sum_k (lhs[m][k] - zp_lhs) * (rhs[k][n] - zp_rhs)``.

    Two paths, because the cheap one is only *exact* when the zero points vanish.

    * Both zero points zero: this is a plain contraction, so :meth:`Encoder.matmul` applies, and it
      multiplies at twice the OPERAND width rather than at the accumulator width. That is exact (a
      product of two w-bit signed values fits in 2w bits) and it is the difference between a query
      that refutes and one that never returns — bit-blasted multiplier area scales as width**2.
    * Either zero point non-zero: the subtraction can leave the operand's own width, so the whole
      computation is done at ``acc_width``, mirroring what ``linalg`` itself does (extend, subtract,
      multiply and accumulate in the result's integer type, wrapping). Exact, and much more
      expensive; the shape has to be correspondingly smaller for a refutation to land.
    """
    if lhs.cols != rhs.rows:
        raise UnsupportedSemantics(
            f"contraction extent mismatch: {lhs.rows}x{lhs.cols} @ {rhs.rows}x{rhs.cols}")
    if zp_lhs == 0 and zp_rhs == 0:
        return enc.matmul(lhs, rhs, acc_width=acc_width)

    lhs_w = {(m, k): _sub_const(enc, enc.sign_extend(lhs.at(m, k), lhs.width, acc_width),
                                zp_lhs, acc_width)
             for m in range(lhs.rows) for k in range(lhs.cols)}
    rhs_w = {(k, n): _sub_const(enc, enc.sign_extend(rhs.at(k, n), rhs.width, acc_width),
                                zp_rhs, acc_width)
             for k in range(rhs.rows) for n in range(rhs.cols)}
    out: dict[tuple[int, int], Any] = {}
    for m in range(lhs.rows):
        for n in range(rhs.cols):
            acc = None
            for k in range(lhs.cols):
                prod = enc.smt.BVMulOp(lhs_w[(m, k)], rhs_w[(k, n)]).results[0]
                acc = prod if acc is None else enc.smt.BVAddOp(acc, prod).results[0]
            out[(m, n)] = acc if acc is not None else enc.const(0, acc_width)
    return Tensor(lhs.rows, rhs.cols, acc_width, out)


def _add_init(enc: Encoder, acc: Tensor, init) -> Tensor:
    """Fold the destination-passing ``outs`` operand into the contraction result."""
    if isinstance(init, _Uninitialized):
        return acc
    if (init.rows, init.cols) != (acc.rows, acc.cols):
        raise UnsupportedSemantics(
            f"init is {init.rows}x{init.cols} but the contraction is {acc.rows}x{acc.cols}")
    if init.width != acc.width:
        raise UnsupportedSemantics(
            f"init is {init.width} bits but the accumulator is {acc.width}; refusing to widen an "
            f"initialised accumulator implicitly")
    out = {k: enc.smt.BVAddOp(v, init.at(*k)).results[0] for k, v in acc.elems.items()}
    return Tensor(acc.rows, acc.cols, acc.width, out)


def encode_linalg(enc: Encoder, module, *, acc_width: int = 32) -> Encoded:
    """Interpret a ``linalg`` source module, returning its returned values and its symbolic inputs.

    This is the SPECIFICATION side of source-to-target validation, so it declares the leaves: the
    source module's block arguments are the program's inputs, and the target encoding is bound onto
    them (``encode_interface(..., shared=...)``). Two sides encoded over independent symbols would
    make the refinement query trivially satisfiable, so which side declares the leaves is not an
    implementation detail — it is what makes the result mean anything.

    ``inputs`` lists the RANK-2 TENSOR block arguments in order. A scalar argument (a runtime zero
    point is exactly this) is legal ``linalg`` but is not a tensor input; it gets no symbolic tensor
    and is not a constant, so anything that reads it abstains rather than guessing a value.

    Outputs are keyed ``ret0, ret1, ...`` in ``func.return`` operand order, which is the order the
    function signature declares and therefore the only correspondence a consumer can rely on.
    """
    env: dict[Any, Any] = {}
    consts: dict[Any, int] = {}
    outputs: dict[str, Tensor] = {}
    inputs: list[Tensor] = []

    func = None
    for op in module.walk():
        if op.name == "func.func":
            func = op
            break
    if func is None:
        raise UnsupportedSemantics("no func.func in module")

    block = func.body.block
    for i, arg in enumerate(block.args):
        if getattr(arg.type, "get_shape", None) is None:
            # A scalar leaf is legal linalg (a runtime zero point is exactly this) but it is not a
            # tensor input, so it gets no symbolic tensor and is NOT a constant. Anything that reads
            # it must abstain, which is what the zero-point resolution below does.
            continue
        rows, cols = _shape(arg.type)
        env[arg] = enc.symbolic_tensor(f"arg{i}", rows, cols, _elem_width(arg.type), acc_width)
        inputs.append(env[arg])

    def _zero_point(value, which: str) -> int:
        if value not in consts:
            raise UnsupportedSemantics(
                f"the {which} zero point of a quantized contraction is not a resolvable integer "
                f"constant (it is produced by "
                f"{getattr(getattr(value, 'owner', None), 'name', 'a block argument')!r}); "
                f"abstaining rather than assuming it is zero")
        return consts[value]

    def _tensor(value, which: str) -> Tensor:
        t = env.get(value)
        if isinstance(t, _Uninitialized):
            raise UnsupportedSemantics(
                f"the {which} operand of a contraction is an uninitialised tensor.empty; it has no "
                f"value to contract")
        if t is None:
            raise UnsupportedSemantics(f"the {which} operand of a contraction is undefined here")
        return t

    for op in block.ops:
        name = op.name
        if name == "arith.constant":
            consts[op.results[0]] = _const_int(op)
        elif name == "tensor.empty":
            rows, cols = _shape(op.results[0].type)
            env[op.results[0]] = _Uninitialized(rows, cols, _elem_width(op.results[0].type))
        elif name in ("linalg.quantized_matmul", "linalg.matmul"):
            res_width = _elem_width(op.results[0].type)
            if res_width != acc_width:
                raise UnsupportedSemantics(
                    f"{name} returns an i{res_width} tensor but the accumulator is i{acc_width}; "
                    f"refusing rather than silently re-widening the contraction")
            if name == "linalg.quantized_matmul":
                if len(op.operands) != 5:
                    raise UnsupportedSemantics(
                        f"linalg.quantized_matmul with {len(op.operands)} operands; expected "
                        f"lhs, rhs, zp_lhs, zp_rhs, init")
                zp_lhs = _zero_point(op.operands[2], "lhs")
                zp_rhs = _zero_point(op.operands[3], "rhs")
                init = env.get(op.operands[4])
            else:
                if len(op.operands) != 3:
                    raise UnsupportedSemantics(
                        f"linalg.matmul with {len(op.operands)} operands; expected lhs, rhs, init")
                zp_lhs = zp_rhs = 0
                init = env.get(op.operands[2])
            if init is None:
                raise UnsupportedSemantics(f"{name} init operand is undefined here")
            acc = _contract(enc, _tensor(op.operands[0], "lhs"), _tensor(op.operands[1], "rhs"),
                            zp_lhs, zp_rhs, acc_width)
            env[op.results[0]] = _add_init(enc, acc, init)
        elif name == "func.return":
            for i, value in enumerate(op.operands):
                outputs[f"ret{i}"] = _tensor(value, f"return operand {i}")
        else:
            raise UnsupportedSemantics(
                f"no semantics for {name!r} in a linalg source module "
                f"(encodable: {sorted(ENCODABLE_OPS)})")

    if not outputs:
        raise UnsupportedSemantics("the source module returns no tensors; nothing to validate")
    return Encoded(outputs=outputs, inputs=inputs)
