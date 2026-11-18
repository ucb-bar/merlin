"""A semantics for the ``interface`` dialect, given by lowering it to the ``smt`` dialect.

This is the paper's idea applied to our IR (Fehr et al., PLDI 2025): the meaning of a program dialect
is defined by a *compilation pass* into a semantic dialect that bottoms out in SMT-LIB. Compiler
authors express semantics with the tooling they already use, and the solver work is reusable.

**Scope, chosen to be the smallest honest one.**

* Values are bitvectors. A tensor is a finite map ``(row, col) -> !smt.bv<w>``; extents are
  CONCRETE, taken from the IR's own tensor types. That is what keeps every obligation
  quantifier-free (QF_BV) — there is no ``smt.forall`` anywhere in this module.
* **No poison.** The paper needs it because ``arith`` carries ``nsw``/``nuw`` and undef. The
  ``interface`` dialect has no fast-math flags, no wrapping annotations, and no undef; adding a
  poison bit would double every value and every operator to propagate a state nothing can produce.
* **No memory or UB model.** ``interface`` is value-typed SSA over tensors. Residency lifetime is the
  one stateful property, and it is a syntactic def-use question already answered by the structural
  layer — it does not need a heap.
* Integer accumulation is *defined*, not undefined: it wraps, and ``smt.bv.add`` is already mod-2^w.

**Float is deliberately absent.** For a float datapath, reassociation is a legal backend choice, so a
bit-exact float refinement check would reject *correct* backends — it is the wrong specification, not
merely an expensive one. Float contractions belong in a structural/uninterpreted encoding; this
module raises rather than pretending, so a float target can never be silently reported as verified.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from merlin.common.paths import contract_dir, repo_root
from merlin.verification import HAS_XDSL, HAS_Z3

# ---- smt_semantics -----------------------------------------------------------------------------

#: Element widths we can encode, by MLIR type spelling.
_INT_WIDTH = {"i8": 8, "i16": 16, "i32": 32, "i64": 64}


class UnsupportedSemantics(RuntimeError):
    """The construct has no encoding here. Raised, never silently skipped."""


@dataclass
class Tensor:
    """A 2-D tensor of SMT bitvector terms, with concrete extents."""

    rows: int
    cols: int
    width: int
    elems: dict[tuple[int, int], Any]

    def at(self, r: int, c: int):
        return self.elems[(r, c)]


def _elem_width(mlir_type) -> int:
    """Element width of a ranked tensor type, or raise. Floats are refused explicitly."""
    et = getattr(mlir_type, "element_type", None)
    name = str(et) if et is not None else str(mlir_type)
    for spelling, width in _INT_WIDTH.items():
        if name == spelling:
            return width
    raise UnsupportedSemantics(
        f"element type {name!r} has no bitvector encoding here. Float datapaths are handled "
        "structurally, not bit-exactly: reassociation is a legal backend choice, so a bit-exact "
        "float refinement would reject correct backends."
    )


def _shape(mlir_type) -> tuple[int, int]:
    shape = [int(d) for d in mlir_type.get_shape()]
    if len(shape) != 2:
        raise UnsupportedSemantics(f"only rank-2 tensors are encoded; got shape {shape}")
    if any(extent <= 0 for extent in shape):
        raise UnsupportedSemantics(f"only concrete positive extents are encoded; got shape {shape}")
    return shape[0], shape[1]


class Encoder:
    """Builds ``smt`` ops for one module. Construct inside an ``ImplicitBuilder``."""

    def __init__(self) -> None:
        if not HAS_XDSL:
            raise UnsupportedSemantics("xDSL is not installed")
        from xdsl.dialects import smt

        self.smt = smt
        self._n = 0

        # -- primitives ------------------------------------------------------------------------------

    def _fresh(self, prefix: str, width: int):
        self._n += 1
        return self.smt.DeclareFunOp(self.smt.BitVectorType(width), f"{prefix}_{self._n}").results[0]

    def const(self, value: int, width: int):
        """A ``bv<width>`` literal. Accepts SIGNED values and stores the two's complement.

        ``BitVectorAttr`` requires an unsigned value in ``[0, 2**width)``, but the constants this
        encoder needs are naturally signed — the i8 saturation bounds are -128 and 127. Wrapping here
        keeps every caller in the sign domain it actually reasons about, instead of making each one
        remember to convert.
        """
        if not -(2 ** (width - 1)) <= value < 2**width:
            raise UnsupportedSemantics(f"constant {value} is not representable in {width} bits")
        return self.smt.BvConstantOp(self.smt.BitVectorAttr(value & ((1 << width) - 1), width)).results[0]

    def symbolic_tensor(self, name: str, rows: int, cols: int, elem_width: int, acc_width: int = 32) -> Tensor:
        """A tensor of fresh symbolic elements — the universally-quantified input, expressed as free
        constants (which is what keeps the query quantifier-free).

        Declared at the element's OWN width. Earlier revisions declared every element at
        ``acc_width`` and constrained it down (see :meth:`in_range`), because the dialect ships no
        widening op. That made an i8 x i8 product a full 32x32 bit-blasted multiplier carrying eight
        meaningful bits, and partial-product area scales as terms x width^2. Measured 2026-09-05: at
        16x16x16 such a query never refutes (``unknown`` at a 1829 s budget) while the same query at
        an 8-bit multiplier width refutes in 37 s. :meth:`sign_extend` now supplies the widening, so
        the wasted width is gone and the range constraints with it.
        """
        elems = {}
        for r in range(rows):
            for c in range(cols):
                elems[(r, c)] = self._fresh(f"{name}_{r}_{c}", elem_width)
        return Tensor(rows, cols, elem_width, elems)

    def sign_extend(self, term, from_w: int, to_w: int):
        """Sign-extend a ``bv<from_w>`` term to ``bv<to_w>`` by repeated doubling.

        ``concat(ashr(x, from_w - 1), x)`` is exactly the sign-extension of ``x`` to ``2*from_w``
        bits: an arithmetic right shift by ``from_w - 1`` leaves every bit equal to the sign bit, and
        concatenating that as the high half is what sign extension means. Doubling repeatedly reaches
        any power-of-two multiple; a non-multiple pads to the next doubling and is then exact because
        the extra high bits are still sign bits.
        """
        if to_w <= from_w:
            return term
        cur, w = term, from_w
        while w < to_w:
            high = self.smt.BVAShrOp(cur, self.const(w - 1, w)).results[0]
            cur = self._concat(high, cur)
            w *= 2
        if w != to_w:
            raise UnsupportedSemantics(
                f"cannot sign-extend {from_w} -> {to_w}: reached {w} by doubling. Widths must be "
                f"related by a power of two; refusing rather than silently truncating."
            )
        return cur

    def _concat(self, high, low):
        """Built like every other op here: constructed inside the ImplicitBuilder, which inserts it."""

        return BVConcatOp.get(high, low).results[0]

    def in_range(self, term, from_w: int, to_w: int):
        """Assert ``term`` (a ``bv<to_w>``) is the sign-extension of a ``from_w``-bit value.

        xDSL 0.68's ``smt`` dialect ships no extract/concat/extend op, so elements are declared
        directly at the accumulator width and CONSTRAINED instead, via the standard shift identity
        ``x == (x << s) >>a s`` with ``s = to_w - from_w``. Both shifts exist in the dialect, and the
        identity is exact — this narrows the input domain to the real element range rather than
        over-approximating it, so a counterexample is always a representable tensor.
        """
        if from_w >= to_w:
            return
        s = self.const(to_w - from_w, to_w)
        shifted = self.smt.BVShlOp(term, s).results[0]
        back = self.smt.BVAShrOp(shifted, s).results[0]
        self.smt.AssertOp(self.smt.EqOp(term, back).results[0])

    def matmul(self, lhs: Tensor, rhs: Tensor, acc_width: int = 32) -> Tensor:
        """``out[m][n] = sum_k lhs[m][k] * rhs[k][n]``, accumulated at ``acc_width``.

        The multiply happens at TWICE the operand width, not at the accumulator width. That is both
        exact and much cheaper: a product of two ``w``-bit signed values always fits in ``2w`` bits
        (the extreme case ``-2^(w-1) * -2^(w-1) = 2^(2w-2)`` is representable), so nothing is lost,
        while bit-blasted multiplier area scales as ``width^2``. Multiplying two sign-extended i8
        operands at 32 bits — what this did before :meth:`sign_extend` existed — spends 16x the
        partial-product area of a 16-bit multiply to compute the same product, and that factor is the
        difference between a query that refutes at a real mesh tile and one that never returns.

        Cost is ``M*N*K`` multiplications, which is why the driver quantifies over small concrete
        extents rather than symbolic ones.
        """
        if lhs.cols != rhs.rows:
            raise UnsupportedSemantics(f"contraction extent mismatch: {lhs.rows}x{lhs.cols} @ {rhs.rows}x{rhs.cols}")
        prod_w = max(lhs.width, rhs.width) * 2
        if prod_w > acc_width:
            # Refuse rather than truncate: a product the accumulator cannot hold is a semantics
            # question about the target, not something this encoder may silently decide.
            raise UnsupportedSemantics(
                f"a {lhs.width}x{rhs.width}-bit product needs {prod_w} bits but the accumulator is "
                f"{acc_width}; refusing rather than truncating"
            )
            # Widen each ELEMENT once, not once per use. lhs[m][k] is read for every output column and
            # rhs[k][n] for every output row, so extending inside the k-loop emits M*N*K widenings where
            # M*K + K*N suffice — at 16x16x16 that is 4096 against 512, an 8x term blowup in the exported
            # query for no semantic gain. The products themselves are genuinely M*N*K and are not cached.
        lhs_w = {
            (m, k): self.sign_extend(lhs.at(m, k), lhs.width, prod_w) for m in range(lhs.rows) for k in range(lhs.cols)
        }
        rhs_w = {
            (k, n): self.sign_extend(rhs.at(k, n), rhs.width, prod_w) for k in range(rhs.rows) for n in range(rhs.cols)
        }
        out: dict[tuple[int, int], Any] = {}
        for m in range(lhs.rows):
            for n in range(rhs.cols):
                acc = None
                for k in range(lhs.cols):
                    prod = self.smt.BVMulOp(lhs_w[(m, k)], rhs_w[(k, n)]).results[0]
                    wide = self.sign_extend(prod, prod_w, acc_width)
                    acc = wide if acc is None else self.smt.BVAddOp(acc, wide).results[0]
                out[(m, n)] = acc if acc is not None else self.const(0, acc_width)
        return Tensor(lhs.rows, rhs.cols, acc_width, out)

        # -- readout epilogue, mirroring merlin.runtime.engine.command_buffer exactly -------------------------------
        #    Every method below has a counterpart in the reference engine, and the refinement check is
        #    only meaningful if they agree bit for bit. Where they could drift, the divergence is named.

    def _slt(self, a, b):

        return BVCmpOp.get("slt", a, b).results[0]

    def _ite(self, cond, then_v, else_v):
        return self.smt.IteOp(cond, then_v, else_v).results[0]

    def relu(self, t: Tensor) -> Tensor:
        """``x if x > 0 else 0`` — ``Tensor.relu`` (tensor.py:182).

        The reference uses a STRICT ``>``; ``max(x, 0)`` agrees on every value including 0, so this
        encodes as ``ite(x < 0, 0, x)``.
        """
        zero = self.const(0, t.width)
        out = {k: self._ite(self._slt(v, zero), zero, v) for k, v in t.elems.items()}
        return Tensor(t.rows, t.cols, t.width, out)

    def requant(self, t: Tensor, shift: int) -> Tensor:
        """``(x + (1 << (shift-1))) >> shift`` — ``Tensor.requant`` (tensor.py:141), round-half-up.

        Python's ``>>`` on a negative int is an arithmetic floor shift, which is exactly
        ``smt.bv.ashr``; a logical shift here would silently disagree on negatives.
        ``shift <= 0`` is the identity, as in the reference.
        """
        if shift <= 0:
            return t
        half = self.const(1 << (shift - 1), t.width)
        amt = self.const(shift, t.width)
        out = {}
        for k, v in t.elems.items():
            biased = self.smt.BVAddOp(v, half).results[0]
            out[k] = self.smt.BVAShrOp(biased, amt).results[0]
        return Tensor(t.rows, t.cols, t.width, out)

    def add_bias(self, t: Tensor, bias: Tensor) -> Tensor:
        """``out[i][j] += bias[j]`` — ``Tensor.add_bias`` (tensor.py:129).

        The reference requires the bias to be a length-``n`` vector; anything else is a shape error
        there, so it is refused here rather than broadcast in some other way.
        """
        n = t.cols
        flat = {}
        for (r, c), v in bias.elems.items():
            flat[r * bias.cols + c] = v
        if len(flat) != n:
            raise UnsupportedSemantics(
                f"bias has {len(flat)} elements but the accumulator has {n} columns; the reference "
                f"engine requires a length-{n} bias vector"
            )
        out = {}
        for (r, c), v in t.elems.items():
            b = flat[c]
            if bias.width != t.width:
                b = self.sign_extend(b, bias.width, t.width)
            out[(r, c)] = self.smt.BVAddOp(v, b).results[0]
        return Tensor(t.rows, t.cols, t.width, out)

    def saturate(self, t: Tensor, lo: int, hi: int, width: int) -> Tensor:
        """Clamp to ``[lo, hi]`` and narrow — ``Tensor.to_i8`` / ``_i8_clamp`` (tensor.py:305).

        The clamp happens at the ACCUMULATOR width (so the comparisons are meaningful) and the result
        keeps that width: narrowing the bitvector would need an extract op, and the clamped value is
        already exactly representable, so the extra width loses no information.

        It does NOT follow that the terms stay comparable — this docstring used to claim that, and the
        claim was false. A ``bv<32>`` clamped into i8 range and a ``bv<8>`` leaf denote the same
        integer but are different SMT types, and ``smt.eq`` rejects the pair. :meth:`any_differs` is
        where that is reconciled; do not assume it anywhere else.
        """
        lo_c, hi_c = self.const(lo, t.width), self.const(hi, t.width)
        out = {}
        for k, v in t.elems.items():
            v = self._ite(self._slt(v, lo_c), lo_c, v)
            v = self._ite(self._slt(hi_c, v), hi_c, v)
            out[k] = v
        return Tensor(t.rows, t.cols, t.width, out)

    def any_differs(self, a: Tensor, b: Tensor):
        """A boolean term: the two tensors disagree somewhere.

        Asserting this is the refinement obligation's negation — ``unsat`` therefore means the two
        programs agree on ALL inputs at this shape, which is the verification result.

        WIDTHS ARE EQUALISED HERE, NOT ASSUMED EQUAL. :meth:`saturate` clamps at the accumulator width
        and keeps it, so a program reaching an output through a contraction carries ``bv<32>`` while
        one that moves a leaf through unchanged carries ``bv<8>``. Upstream ``smt.eq`` carries
        ``SameTypeOperands``, so comparing those two exported an ill-formed module and the whole query
        died with no verdict at all — three archived movement submissions were lost that way, counted
        as checker errors rather than as the abstentions or refutations they should have been.
        Sign-extending the narrower side is sound precisely BECAUSE ``saturate`` already clamped the
        wider one into the narrow range; the two terms denote the same integer when they agree.
        """
        if (a.rows, a.cols) != (b.rows, b.cols):
            raise UnsupportedSemantics(f"shape mismatch {a.rows}x{a.cols} vs {b.rows}x{b.cols}")
        width = max(a.width, b.width)
        diffs = []
        for r in range(a.rows):
            for c in range(a.cols):
                x, y = a.at(r, c), b.at(r, c)
                if a.width != width:
                    x = self.sign_extend(x, a.width, width)
                if b.width != width:
                    y = self.sign_extend(y, b.width, width)
                eq = self.smt.EqOp(x, y).results[0]
                diffs.append(self.smt.NotOp(eq).results[0])
        term = diffs[0]
        for d in diffs[1:]:
            term = self.smt.OrOp(term, d).results[0]
        return term

        # -- walking the interface dialect ----------------------------------------------------------------


@dataclass
class Encoded:
    """What one interpretation of a module yields.

    ``inputs`` is kept because the specification side must be built over the SAME symbolic terms —
    two sides encoded over independent symbols would make the query trivially satisfiable and the
    check meaningless.
    """

    outputs: dict[str, Tensor]
    inputs: list[Tensor]


def encode_interface(enc: Encoder, module, acc_width: int = 32, shared: list[Tensor] | None = None) -> Encoded:
    """Interpret an ``interface`` module, returning its committed outputs and its symbolic inputs.

    This walks the ACTUAL IR: a matmul the pass failed to emit, an operand it swapped, or a commit it
    duplicated all change the terms produced here, which is what makes the refinement check
    meaningful rather than a tautology.

    ``shared`` binds the block arguments to tensors another encoding already declared, BY POSITION.
    Passing it is what lets this program be compared against the source it was compiled from: two
    sides encoded over independent symbols would make the query trivially satisfiable. A shape or
    width mismatch is refused rather than coerced — it means the two artifacts are not about the same
    program.
    """
    env: dict[Any, Tensor] = {}
    outputs: dict[str, Tensor] = {}
    inputs: list[Tensor] = []
    n_commit = 0
    commits: list[Any] = []
    saw_return = False

    funcs = [op for op in module.walk() if op.name == "func.func"]
    if len(funcs) != 1:
        raise UnsupportedSemantics(f"expected exactly one func.func; found {len(funcs)}")
    func = funcs[0]
    if len(func.body.blocks) != 1:
        raise UnsupportedSemantics(f"interface function has {len(func.body.blocks)} blocks; only one is encoded")

        # Block arguments are the symbolic inputs.
    block = func.body.block
    operations = list(block.ops)
    if not operations or operations[-1].name != "func.return":
        raise UnsupportedSemantics("interface function must end in func.return")
    if shared is not None and len(shared) != len(block.args):
        raise UnsupportedSemantics(
            f"the interface module has {len(block.args)} block arguments but {len(shared)} shared "
            f"leaves were offered; the two artifacts are not the same program"
        )
    for i, arg in enumerate(block.args):
        rows, cols = _shape(arg.type)
        width = _elem_width(arg.type)
        if shared is not None:
            bound = shared[i]
            if (bound.rows, bound.cols, bound.width) != (rows, cols, width):
                raise UnsupportedSemantics(
                    f"argument {i} is {rows}x{cols}x{width}b in the interface module but the shared "
                    f"leaf is {bound.rows}x{bound.cols}x{bound.width}b"
                )
            env[arg] = bound
        else:
            env[arg] = enc.symbolic_tensor(f"arg{i}", rows, cols, width, acc_width)
        inputs.append(env[arg])

    for op in block.ops:
        name = op.name
        if name == "interface.resident_pack":
            # Residency is a placement decision, not a value transformation: the packed weight holds
            # the same numbers. (A packer that permuted values would be a layout bug, which the
            # structural layer checks; it is not expressible in this value algebra.)
            env[op.results[0]] = env[op.operands[0]]
        elif name == "interface.matmul":
            lhs, rhs = env[op.operands[0]], env[op.operands[1]]
            env[op.results[0]] = enc.matmul(lhs, rhs, acc_width=acc_width)
        elif name == "interface.commit":
            epilogue = op.properties.get("epilogue")
            stages = [str(s) for s in epilogue] if epilogue is not None else []
            if stages:
                raise UnsupportedSemantics(f"epilogue stages not encoded yet: {stages}")
            acc = env[op.operands[0]]
            env[op.results[0]] = acc
            outputs[f"commit{n_commit}"] = acc
            commits.append(op.results[0])
            n_commit += 1
        elif name == "func.return":
            if saw_return:
                raise UnsupportedSemantics("multiple func.return operations are not encoded")
            saw_return = True
            if list(op.operands) != commits:
                raise UnsupportedSemantics(
                    "func.return must deliver every interface.commit result in commit order; "
                    "the returned function is otherwise different from the values compared"
                )
        elif name == "interface.resident_evict":
            continue
        else:
            raise UnsupportedSemantics(f"no semantics for {name!r}")
    if not saw_return:
        raise UnsupportedSemantics("interface function has no func.return")
    return Encoded(outputs=outputs, inputs=inputs)

    # Preserve historical object lookup; implementation and completion state live here.


# ---- smt_ops -----------------------------------------------------------------------------------
# The ``smt`` ops xDSL 0.68 does not ship: the solver scope, and bitvector concatenation.
#
# xDSL vendors upstream MLIR's ``smt`` dialect (bitvectors, quantifiers, ``declare_fun``, ``assert``,
# ``yield``) but not ``smt.solver`` / ``smt.check``. Upstream's ``mlir-translate --export-smtlib``
# requires the solver scope — it is what delimits one query — so without it a module of ``smt`` ops
# cannot be exported at all.
#
# ``smt.check`` is genuinely unnecessary here: the solver API supplies ``(check-sat)``. Only the scope
# is missing, and it is a region-holding op with no operands or results, so this is twenty lines rather
# than a fork.
#
# Measured: a module built this way prints in generic form and is accepted verbatim by upstream
# ``mlir-translate --export-smtlib``.
#
# ``smt.bv.concat`` is the second gap, and it is the expensive one. Without it the encoder had to
# declare every tensor element at the ACCUMULATOR width and constrain it down to its real element
# range, which makes an i8 x i8 product a full 32x32 bit-blasted multiplier carrying eight meaningful
# bits. Measured 2026-09-05: at 16x16x16 that query never refutes (``unknown`` at a 1829 s budget),
# while the identical query at an 8-bit multiplier width refutes in 37 s -- partial-product area scales
# as terms x width^2, so the wasted width was the whole cost. Upstream's exporter emits ``(concat a b)``
# for this op, verified by running it.

if HAS_XDSL:
    from xdsl.dialects.builtin import IntegerAttr, i64
    from xdsl.dialects.smt import BitVectorType, BoolType
    from xdsl.ir import Dialect, Region
    from xdsl.irdl import IRDLOperation, irdl_op_definition, operand_def, prop_def, region_def, result_def

    @irdl_op_definition
    class SolverOp(IRDLOperation):
        """``smt.solver`` — the scope one SMT query lives in.

        Mirrors upstream MLIR's op of the same name. The body is terminated by the existing
        ``smt.yield``; assertions inside the region become the query's assertions.
        """

        name = "smt.solver"
        body = region_def()

        @staticmethod
        def from_region(region: Region) -> "SolverOp":
            return SolverOp(regions=[region])

    @irdl_op_definition
    class BVConcatOp(IRDLOperation):
        """``smt.bv.concat`` — bitvector concatenation, exported as SMT-LIB ``(concat lhs rhs)``.

        The result width is the sum of the operand widths. Mirrors upstream MLIR's op of the same
        name; xDSL 0.68 ships no extract/concat/extend, which is why this is defined here rather than
        imported. With it, an ``n``-bit value sign-extends to ``2n`` bits as
        ``concat(ashr(x, n-1), x)`` — the high half of an arithmetic right shift by ``n-1`` is all
        sign bits — so no separate extend op is needed.
        """

        name = "smt.bv.concat"
        lhs = operand_def(BitVectorType)
        rhs = operand_def(BitVectorType)
        result = result_def(BitVectorType)

        @staticmethod
        def get(lhs, rhs) -> "BVConcatOp":
            width = lhs.type.width.data + rhs.type.width.data
            return BVConcatOp(operands=[lhs, rhs], result_types=[BitVectorType(width)])

            #: Predicate ordinals for :class:`BVCmpOp`, MEASURED rather than assumed.
            #:
            #: Upstream spells the predicate as a keyword in custom syntax but stores it as an ``i64``
            #: ordinal, which is what xDSL must emit since it prints generic form. The mapping below was
            #: derived by round-tripping every keyword through ``mlir-opt --mlir-print-op-generic`` and
            #: reading the ordinal back; :func:`bv_cmp_exports_as` and the tests assert the exported SMT-LIB
            #: still says ``bvslt``, so a change in upstream's enum order fails loudly instead of silently
            #: inverting a comparison.

    BV_CMP_PREDICATES = {"slt": 0, "sle": 1, "sgt": 2, "sge": 3, "ult": 4, "ule": 5, "ugt": 6, "uge": 7}

    @irdl_op_definition
    class BVCmpOp(IRDLOperation):
        """``smt.bv.cmp`` — a bitvector comparison, exported as ``(bvslt a b)`` and friends.

        Needed for the readout epilogue: ``relu`` is ``ite(slt(x, 0), 0, x)`` and the saturating
        narrow to i8 is two clamps, neither expressible with the arithmetic ops xDSL ships.
        """

        name = "smt.bv.cmp"
        lhs = operand_def(BitVectorType)
        rhs = operand_def(BitVectorType)
        result = result_def(BoolType)
        pred = prop_def(IntegerAttr)

        @staticmethod
        def get(pred: str, lhs, rhs) -> "BVCmpOp":
            if pred not in BV_CMP_PREDICATES:
                raise ValueError(f"unknown bv compare predicate {pred!r}; known: {sorted(BV_CMP_PREDICATES)}")
            return BVCmpOp(
                operands=[lhs, rhs],
                result_types=[BoolType()],
                properties={"pred": IntegerAttr(BV_CMP_PREDICATES[pred], i64)},
            )

            #: Registered separately from xDSL's own ``smt`` Dialect object so nothing upstream is mutated.

    SMT_SOLVER = Dialect("smt", [SolverOp, BVConcatOp, BVCmpOp], [])
else:  # pragma: no cover - import guard
    SolverOp = None  # type: ignore[assignment]
    BVConcatOp = None  # type: ignore[assignment]
    BVCmpOp = None  # type: ignore[assignment]
    BV_CMP_PREDICATES = {}  # type: ignore[assignment]
    SMT_SOLVER = None  # type: ignore[assignment]

    # Conditional dialect definitions retain their historical pickle lookup too.


# ---- tools -------------------------------------------------------------------------------------
# Locating the external binaries the verification layers need.
#
# One resolver, shared by the lit harness and the SMT exporter, so a tool is never found by one layer
# and missed by another. ``None`` is a first-class answer: the caller reports "unavailable" and skips.
# It must never silently degrade into a pass.
#
# ``FileCheck`` resolution delegates to the existing RTL screen's candidate list rather than starting a
# second one — that resolver already knows the in-tree LLVM build and the chipyard toolchain.


def _llvm_bin() -> Path:
    """The in-tree LLVM build's bin dir (FileCheck/llvm-lit live here; the install tree omits them —
    it is configured with LLVM_INSTALL_UTILS=OFF)."""
    return repo_root() / "third_party" / "llvm-build" / "bin"


def find_filecheck() -> str | None:
    """Reuse the RTL screen's resolver; fall back to our own candidates if it is unavailable."""
    try:
        from merlin.development.targetgen.checks.rtl_check_runner import find_filecheck as _fc

        found = _fc()
        if found:
            return found
    except Exception:
        pass
    cand = _llvm_bin() / "FileCheck"
    return str(cand) if cand.is_file() else shutil.which("FileCheck")


def find_lit() -> str | None:
    cand = _llvm_bin() / "llvm-lit"
    return str(cand) if cand.is_file() else shutil.which("llvm-lit") or shutil.which("lit")


def find_mlir_tool(name: str) -> str | None:
    """Resolve an MLIR binary (``mlir-opt``, ``mlir-translate``) from the in-tree build or install."""
    for base in (_llvm_bin(), repo_root() / "third_party" / "llvm-install" / "bin"):
        cand = base / name
        if cand.is_file():
            return str(cand)
    return shutil.which(name)


def iface_irdl() -> Path:
    """The IRDL definition of the frozen ``merlin_iface`` grammar.

    Registering this into upstream ``mlir-opt --irdl-file`` gives a structural verifier for the one
    dialect that is contractually frozen, with no code of ours in the checking path.
    """
    return contract_dir() / "merlin_iface.irdl.mlir"


def availability() -> dict[str, str | None]:
    """Everything the verification layers depend on, resolved. For reporting a skip honestly."""
    return {
        "FileCheck": find_filecheck(),
        "llvm-lit": find_lit(),
        "mlir-opt": find_mlir_tool("mlir-opt"),
        "mlir-translate": find_mlir_tool("mlir-translate"),
    }

    # --- merlin_iface IRDL normalization (WORKAROUND, see docs/design/compiler_verification.md) --------
    # The tracked IRDL is generated by `tblgen-to-irdl`, which emits a custom type's symbol name WITH its
    # `!` sigil: `irdl.type @"!resident"`. That name is unspellable in MLIR text, so `mlir-opt
    # --irdl-file` rejects EVERY module using `!merlin_iface.resident` or `!merlin_iface.acc` -- which is
    # every real capsule interface module. The generator already normalizes a different tblgen artifact
    # (`irdl.base "!builtin.string"` -> `irdl.any`) but not this one.
    #
    # Stripping the sigil from the declarations AND the `::@` references restores full verification,
    # operand type checking included (measured: an `evict` handed a tensor is rejected with
    # "expected base type 'merlin_iface.resident' but got 'builtin.tensor'").
    #
    # This normalization is IDEMPOTENT and becomes a no-op the moment the generator is fixed, so it can
    # stay until then without hiding the defect. Fixing the generator is tracked separately; it lives
    # under a package this work does not own.


_SIGIL_FIXES = (('irdl.type @"!', 'irdl.type @"'), ('::@"!', '::@"'))


def normalize_irdl_text(text: str) -> tuple[str, int]:
    """Return ``(normalized, n_substitutions)``. Zero substitutions means the source is already
    clean -- report it rather than assuming the workaround is still needed."""
    n = 0
    for bad, good in _SIGIL_FIXES:
        n += text.count(bad)
        text = text.replace(bad, good)
    return text, n


def normalized_iface_irdl() -> Path:
    """The frozen grammar's IRDL, sigil-normalized, cached under the generated-output root."""
    from merlin.common.artifacts import cache_dir

    src = iface_irdl()
    out = Path(cache_dir("verify")) / "merlin_iface.normalized.irdl.mlir"
    text, _ = normalize_irdl_text(src.read_text(encoding="utf-8"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    return out

    # Preserve historical object lookup; implementation and completion state live here.


# ---- smt_export --------------------------------------------------------------------------------
# Export an ``smt``-dialect module to SMT-LIB and solve it.
#
# The whole chain is upstream infrastructure, which is the point: our dialect semantics is a
# compilation pass to MLIR's own ``smt`` dialect, and the SMT-LIB comes out of upstream's
# ``ExportSMTLIB`` translation rather than an encoder of ours.
#
#     xDSL smt module  ->  mlir-translate --export-smtlib  ->  z3
#
# One trap, learned by measurement and guarded by a test: the exporter emits a trailing ``(reset)``
# per solver scope. Handed to z3 verbatim the query still answers ``sat``, but the model comes back
# EMPTY — the reset has already discarded it, so the counterexample silently disappears. It is
# stripped here. An empty assignment set can still be legitimate if the outputs differ for every
# input, so consumers must keep the solver status and model object together.

#: The exporter's per-scope terminator. Stripping it is what keeps a counterexample readable.
_RESET = "(reset)"
_MAX_INPUT_BYTES = 16 * 1024 * 1024
_MAX_OUTPUT_BYTES = 64 * 1024 * 1024


class SmtUnavailable(RuntimeError):
    """Raised when the SMT chain cannot run. Never degrades into a verdict."""


@dataclass(frozen=True)
class Verdict:
    """The outcome of one query. Three states, never two.

    ``unsat``   the refinement obligation holds (this is the PASS for a verification query)
    ``sat``     refuted, with a counterexample in ``model``
    ``unknown`` the solver gave up (timeout / incompleteness) — NOT a pass
    """

    status: str
    model: str | None = None
    smt2: str = field(default="", repr=False)
    #: The counterexample as structured data: declared constant name -> signed integer value.
    #: Kept alongside the printed model because a counterexample is only actionable if it can be
    #: turned back into a concrete input tensor, and re-parsing a solver's pretty-printed model is
    #: exactly the brittle spelling-match this repo forbids.
    model_values: dict = field(default_factory=dict, repr=False)
    reason: str | None = None

    @property
    def verified(self) -> bool:
        return self.status == "unsat"

    @property
    def refuted(self) -> bool:
        return self.status == "sat"


def module_text(module) -> str:
    """Print an xDSL module. Generic form is what the upstream exporter parses."""
    if not HAS_XDSL:
        raise SmtUnavailable("xDSL is not installed")
    import io

    from xdsl.printer import Printer

    buf = io.StringIO()
    Printer(stream=buf, print_generic_format=True).print_op(module)
    return buf.getvalue()


def bounded_tool_output(
    argv: list[str], *, input_text: str | None = None, timeout_s: int = 120, max_output_bytes: int = _MAX_OUTPUT_BYTES
) -> tuple[int, str, str]:
    """Run an explicit executable with bounded time, input, captured output, and no shell."""
    if input_text is not None and len(input_text.encode("utf-8")) > _MAX_INPUT_BYTES:
        raise SmtUnavailable(f"translator input exceeds {_MAX_INPUT_BYTES} bytes")

        # The child writes to temporary files so its output never accumulates in process memory.
        # RLIMIT_FSIZE also bounds the files themselves on the supported Unix toolchain.

    def limit_files() -> None:
        import resource

        resource.setrlimit(resource.RLIMIT_FSIZE, (max_output_bytes, max_output_bytes))

    try:
        with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
            result = subprocess.run(
                argv,
                input=input_text.encode("utf-8") if input_text is not None else None,
                stdout=stdout,
                stderr=stderr,
                timeout=timeout_s,
                preexec_fn=limit_files,
                check=False,
            )
            stdout.seek(0)
            stderr.seek(0)
            out = stdout.read(max_output_bytes + 1)
            err = stderr.read(max_output_bytes + 1)
    except subprocess.TimeoutExpired as exc:
        raise SmtUnavailable(f"tool timed out after {timeout_s}s: {argv[0]!r}") from exc
    except OSError as exc:
        raise SmtUnavailable(f"tool cannot run at {argv[0]!r}: {exc}") from exc
    if len(out) > max_output_bytes or len(err) > max_output_bytes:
        raise SmtUnavailable(f"tool output exceeds {max_output_bytes} bytes: {argv[0]!r}")
    return result.returncode, out.decode("utf-8", errors="replace"), err.decode("utf-8", errors="replace")


def to_smtlib(module, *, translator: str | Path | None = None) -> str:
    """Export an ``smt``-dialect module to SMT-LIB text via upstream ``mlir-translate``."""
    tool = str(translator) if translator is not None else find_mlir_tool("mlir-translate")
    if tool is None:
        raise SmtUnavailable("mlir-translate not found; cannot export SMT-LIB")
    text = module_text(module)
    returncode, stdout, stderr = bounded_tool_output([tool, "--export-smtlib"], input_text=text)
    if returncode != 0:
        raise SmtUnavailable(f"mlir-translate --export-smtlib failed (exit {returncode}):\n{stderr[:4000]}")
    return stdout


def strip_reset(smt2: str) -> str:
    """Drop the exporter's per-scope ``(reset)``, which would discard the model."""
    return "\n".join(l for l in smt2.splitlines() if l.strip() != _RESET)


def solve(smt2: str, *, timeout_ms: int = 30_000) -> Verdict:
    """Solve SMT-LIB text with z3.

    ``timeout_ms`` bounds one query; a timeout is reported as ``unknown``, which is explicitly not a
    pass — an abstracted or timed-out obligation must never be counted as verified.
    """
    if not HAS_Z3:
        raise SmtUnavailable("z3 is not installed (pip install 'merlin[verify]')")
    import z3

    body = strip_reset(smt2)
    solver = z3.Solver()
    solver.set("timeout", timeout_ms)
    solver.from_string(body)
    res = solver.check()
    if res == z3.sat:
        m = solver.model()
        values = {}
        for decl in m.decls():
            val = m[decl]
            try:
                # as_signed_long() gives the two's-complement reading of a bitvector, which is the
                # element value the program actually computes with.
                values[decl.name()] = val.as_signed_long()
            except Exception:
                values[decl.name()] = str(val)
        return Verdict("sat", model=str(m), smt2=body, model_values=values)
    if res == z3.unsat:
        return Verdict("unsat", smt2=body)
    return Verdict("unknown", smt2=body, reason=solver.reason_unknown())


def check_module(module, *, timeout_ms: int = 30_000, translator: str | Path | None = None) -> Verdict:
    """Export and solve in one step."""
    return solve(to_smtlib(module, translator=translator), timeout_ms=timeout_ms)

    # Preserve historical object lookup; implementation and completion state live here.
