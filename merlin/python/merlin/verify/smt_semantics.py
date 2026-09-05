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

from dataclasses import dataclass
from typing import Any

from . import HAS_XDSL

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
        "float refinement would reject correct backends.")


def _shape(mlir_type) -> tuple[int, int]:
    shape = [int(d) for d in mlir_type.get_shape()]
    if len(shape) != 2:
        raise UnsupportedSemantics(f"only rank-2 tensors are encoded; got shape {shape}")
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
        if not -(2 ** (width - 1)) <= value < 2 ** width:
            raise UnsupportedSemantics(
                f"constant {value} is not representable in {width} bits")
        return self.smt.BvConstantOp(
            self.smt.BitVectorAttr(value & ((1 << width) - 1), width)).results[0]

    def symbolic_tensor(self, name: str, rows: int, cols: int, elem_width: int,
                        acc_width: int = 32) -> Tensor:
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
                f"related by a power of two; refusing rather than silently truncating.")
        return cur

    def _concat(self, high, low):
        """Built like every other op here: constructed inside the ImplicitBuilder, which inserts it."""
        from .smt_ops import BVConcatOp

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
            raise UnsupportedSemantics(
                f"contraction extent mismatch: {lhs.rows}x{lhs.cols} @ {rhs.rows}x{rhs.cols}")
        prod_w = max(lhs.width, rhs.width) * 2
        if prod_w > acc_width:
            # Refuse rather than truncate: a product the accumulator cannot hold is a semantics
            # question about the target, not something this encoder may silently decide.
            raise UnsupportedSemantics(
                f"a {lhs.width}x{rhs.width}-bit product needs {prod_w} bits but the accumulator is "
                f"{acc_width}; refusing rather than truncating")
        # Widen each ELEMENT once, not once per use. lhs[m][k] is read for every output column and
        # rhs[k][n] for every output row, so extending inside the k-loop emits M*N*K widenings where
        # M*K + K*N suffice — at 16x16x16 that is 4096 against 512, an 8x term blowup in the exported
        # query for no semantic gain. The products themselves are genuinely M*N*K and are not cached.
        lhs_w = {(m, k): self.sign_extend(lhs.at(m, k), lhs.width, prod_w)
                 for m in range(lhs.rows) for k in range(lhs.cols)}
        rhs_w = {(k, n): self.sign_extend(rhs.at(k, n), rhs.width, prod_w)
                 for k in range(rhs.rows) for n in range(rhs.cols)}
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

    # -- readout epilogue, mirroring merlin.runtime.tensor exactly -------------------------------
    #    Every method below has a counterpart in the reference engine, and the refinement check is
    #    only meaningful if they agree bit for bit. Where they could drift, the divergence is named.

    def _slt(self, a, b):
        from .smt_ops import BVCmpOp

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
                f"engine requires a length-{n} bias vector")
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
        already exactly representable, so the extra width is harmless and the terms stay comparable.
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
        """
        if (a.rows, a.cols) != (b.rows, b.cols):
            raise UnsupportedSemantics(f"shape mismatch {a.rows}x{a.cols} vs {b.rows}x{b.cols}")
        diffs = []
        for r in range(a.rows):
            for c in range(a.cols):
                eq = self.smt.EqOp(a.at(r, c), b.at(r, c)).results[0]
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


def encode_interface(enc: Encoder, module, acc_width: int = 32,
                    shared: list[Tensor] | None = None) -> Encoded:
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

    func = None
    for op in module.walk():
        if op.name == "func.func":
            func = op
            break
    if func is None:
        raise UnsupportedSemantics("no func.func in module")

    # Block arguments are the symbolic inputs.
    block = func.body.block
    if shared is not None and len(shared) != len(block.args):
        raise UnsupportedSemantics(
            f"the interface module has {len(block.args)} block arguments but {len(shared)} shared "
            f"leaves were offered; the two artifacts are not the same program")
    for i, arg in enumerate(block.args):
        rows, cols = _shape(arg.type)
        width = _elem_width(arg.type)
        if shared is not None:
            bound = shared[i]
            if (bound.rows, bound.cols, bound.width) != (rows, cols, width):
                raise UnsupportedSemantics(
                    f"argument {i} is {rows}x{cols}x{width}b in the interface module but the shared "
                    f"leaf is {bound.rows}x{bound.cols}x{bound.width}b")
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
            n_commit += 1
        elif name in ("interface.resident_evict", "func.return"):
            continue
        else:
            raise UnsupportedSemantics(f"no semantics for {name!r}")
    return Encoded(outputs=outputs, inputs=inputs)
