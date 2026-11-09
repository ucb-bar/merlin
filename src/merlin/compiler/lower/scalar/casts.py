"""Legalize scalar Boolean/float conversion without introducing poison.

Preserve source ordering, numerical/effect obligations and explicit selection.
"""

from __future__ import annotations

from merlin.compiler.ir.provenance import carry_provenance


def fix_bool_sitofp(module) -> int:
    """Rewrite ``arith.sitofp`` of an ``i1`` (a bool) to ``arith.uitofp``; returns the count.

    PyTorch casts a bool tensor to float as ``True -> 1.0`` (unsigned). model2MLIR, however,
    emits a *signed* ``arith.sitofp %b : i1 to fN`` for these casts. ``sitofp`` interprets the
    i1 as a signed 1-bit integer, whose ``true`` bit pattern is the value ``-1`` (two's
    complement), so the compiled kernel produces ``-1.0`` instead of ``+1.0``. This is silent
    on the driver's scalar interpreter (Python ``float(True) == 1.0``) but wrong in every
    *compiled* kernel (host clang and RVV alike).

    The canonical place it bites: the eager-attention causal mask, built as
    ``mask_f32 * sitofp(bool)``; the sign flip turns the ``-FLT_MAX`` fill into ``+FLT_MAX``,
    so softmax attends to *future* tokens (molmoact decoder cos 0.10). Rewriting to
    ``uitofp`` (``true -> 1.0``) matches torch. ``module.walk()`` recurses into linalg.generic
    bodies, so casts inside outlined kernels are covered. Apply before outlining/lowering.
    """
    from xdsl.dialects import arith
    from xdsl.dialects.builtin import IntegerType

    n = 0
    for op in list(module.walk()):
        if op.name != "arith.sitofp":
            continue
        src = op.operands[0]
        st = src.type
        if not (isinstance(st, IntegerType) and st.width.data == 1):
            continue
        new = arith.UIToFPOp(src, op.results[0].type)
        carry_provenance(new, op, "bool_sitofp_uitofp")
        op.parent_block().insert_op_before(new, op)
        op.results[0].replace_all_uses_with(new.results[0])
        op.parent_block().detach_op(op)
        n += 1
    return n


def fix_bool_fptosi(module) -> int:
    """Rewrite a float->``i1`` ``arith.fptosi``/``arith.fptoui`` into ``x != 0``; returns the count.

    The mirror image of :func:`fix_bool_sitofp`, and far worse than a sign flip. model2MLIR emits
    a cast to a bool tensor (``aten._to_copy`` with ``prov.orig_dtype = "bool"``) as
    ``arith.fptosi %x : f32 to i1``. Signed ``i1`` holds only ``{-1, 0}``, so *every* float whose
    truncation is not one of those two values — ``1.0`` included — is **poison** in LLVM, not a
    wrong number. mlir-translate emits ``fptosi float 1.0 to i1``, instcombine folds it to
    ``poison``, and the poison propagates out of the bool tensor into whatever consumes it.

    Where that lands is not local. In smolvla the poisoned mask feeds a masked-select whose result
    count sizes a ``malloc`` and bounds a data-dependent loop, so ``br i1 poison`` becomes a
    self-branch, ``simplifycfg`` deletes every block after it, and ``forward`` compiles to 3,654
    bytes with a call set of ``malloc``/``memset``/``roundevenf`` — a whole 500M-parameter model
    erased while the link succeeds and the build reports success.

    PyTorch's bool cast is ``x != 0`` (NaN included, which is why the predicate is the *unordered*
    ``une``), so that is what replaces it. ``module.walk()`` recurses into linalg.generic bodies.
    Apply before outlining/lowering.
    """
    from xdsl.dialects import arith
    from xdsl.dialects.builtin import AnyFloat, FloatAttr, IntegerType

    n = 0
    for op in list(module.walk()):
        if op.name not in ("arith.fptosi", "arith.fptoui"):
            continue
        res_t = op.results[0].type
        if not (isinstance(res_t, IntegerType) and res_t.width.data == 1):
            continue
        src = op.operands[0]
        if not isinstance(src.type, AnyFloat):
            continue
        zero = arith.ConstantOp(FloatAttr(0.0, src.type))
        # `une` = unordered-or-not-equal: NaN compares TRUE, matching torch's `bool(nan) is True`.
        cmp = arith.CmpfOp(src, zero.results[0], "une")
        carry_provenance(cmp, op, "bool_fptosi_cmpf_une")
        block = op.parent_block()
        block.insert_op_before(zero, op)
        block.insert_op_before(cmp, op)
        op.results[0].replace_all_uses_with(cmp.results[0])
        block.detach_op(op)
        n += 1
    return n
