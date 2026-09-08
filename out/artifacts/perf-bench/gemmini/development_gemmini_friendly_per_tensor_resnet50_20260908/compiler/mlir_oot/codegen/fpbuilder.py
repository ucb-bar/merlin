"""Branch-free f32 primitives for the compiler-generated CPU lane.

The emitted kernel is a single basic block with no `select` and no branch (see
`docs/iteration_notes.md` R1.4), so every predicate here is built from the IEEE-754 bit
pattern: a comparison becomes an arithmetic difference whose SIGN BIT, broadcast by an
arithmetic shift, is the blend mask.  The transcendentals are likewise *generated* — a
range-reduced polynomial written out as ordinary `llvm.f*` ops — rather than a call into a
math library the bare-metal harness does not link.

Nothing in this module is keyed on a capsule: it is the scalar instruction set the linalg
lowering in `host_linalg.py` targets.
"""
from __future__ import annotations

from xdsl.dialects import llvm
from xdsl.dialects.builtin import Float32Type, FloatAttr, IntegerType, i16, i32, i64
from xdsl.ir import SSAValue
from xdsl.irdl import IRDLOperation, irdl_op_definition, operand_def, result_def

F32 = Float32Type()


@irdl_op_definition
class FPToSIOp(IRDLOperation):
    """LLVM's ordinary floating-point to signed-integer conversion.

    The pinned xDSL LLVM dialect omits this operation class, even though the
    downstream MLIR/LLVM toolchain implements ``llvm.fptosi``. Keeping the
    standard LLVM operation in target-neutral IR lets each CPU backend select
    its native conversion instead of expanding IEEE fields by hand.
    """

    name = "llvm.fptosi"
    value = operand_def()
    result = result_def()

    def __init__(self, value: SSAValue, result_type: IntegerType):
        super().__init__(operands=[value], result_types=[result_type])


@irdl_op_definition
class RoundEvenOp(IRDLOperation):
    """The standard LLVM round-to-integer, ties-to-even intrinsic."""

    name = "llvm.intr.roundeven"
    value = operand_def()
    result = result_def()

    def __init__(self, value: SSAValue):
        super().__init__(operands=[value], result_types=[value.type])

#: exp() range reduction.  `MAGIC` is 1.5 * 2**23: adding it to a float smaller than 2**22
#: forces the fractional bits out through round-to-nearest-even, so the mantissa's low bits
#: hold the rounded integer.  That is how the exponent is extracted without an fp->int op.
_LOG2E = 1.4426950408889634
_MAGIC = 12582912.0
_LN2_HI = 0.693359375
_LN2_LO = -2.1219444005469057e-4
#: exp(r) on the reduced interval |r| <= ln2/2: the Taylor series to r**5 (< 1 ulp there).
_EXP_POLY = (1.0, 1.0, 0.5, 1.0 / 6.0, 1.0 / 24.0, 1.0 / 120.0)
#: Abramowitz & Stegun 7.1.26 -- erf to 1.5e-7 absolute, in terms of exp() and one divide.
_ERF_P = 0.3275911
_ERF_A = (0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429)
#: The classic inverse-square-root seed; three Newton steps take it below 1e-7 relative.
_RSQRT_SEED = 0x5F3759DF


class FpBuilder:
    """f32 scalar arithmetic emitted into one `FnBuilder`'s single block."""

    def __init__(self, fb):
        self.fb = fb
        self._fconsts: dict[float, SSAValue] = {}

    # -- constants ---------------------------------------------------------------------------
    def fconst(self, value: float) -> SSAValue:
        key = float(value)
        hit = self._fconsts.get(key)
        if hit is not None:
            return hit
        op = llvm.ConstantOp(FloatAttr(key, F32), F32)
        self.fb.prologue(op)
        self._fconsts[key] = op.results[0]
        return op.results[0]

    def i32c(self, value: int) -> SSAValue:
        return self.fb.const(int(value), i32)

    # -- arithmetic --------------------------------------------------------------------------
    def fadd(self, a: SSAValue, b: SSAValue) -> SSAValue:
        return self.fb.add(llvm.FAddOp(a, b)).results[0]

    def fsub(self, a: SSAValue, b: SSAValue) -> SSAValue:
        return self.fb.add(llvm.FSubOp(a, b)).results[0]

    def fmul(self, a: SSAValue, b: SSAValue) -> SSAValue:
        return self.fb.add(llvm.FMulOp(a, b)).results[0]

    def fdiv(self, a: SSAValue, b: SSAValue) -> SSAValue:
        return self.fb.add(llvm.FDivOp(a, b)).results[0]

    def fneg(self, a: SSAValue) -> SSAValue:
        return self.fsub(self.fconst(0.0), a)

    # -- bit-level ---------------------------------------------------------------------------
    def bits(self, a: SSAValue) -> SSAValue:
        return self.fb.add(llvm.BitcastOp(a, i32)).results[0]

    def unbits(self, a: SSAValue) -> SSAValue:
        return self.fb.add(llvm.BitcastOp(a, F32)).results[0]

    def and32(self, a: SSAValue, b: SSAValue) -> SSAValue:
        return self.fb.add(llvm.AndOp(a, b)).results[0]

    def or32(self, a: SSAValue, b: SSAValue) -> SSAValue:
        return self.fb.add(llvm.OrOp(a, b)).results[0]

    def xor32(self, a: SSAValue, b: SSAValue) -> SSAValue:
        return self.fb.add(llvm.XOrOp(a, b)).results[0]

    def add32(self, a: SSAValue, b: SSAValue) -> SSAValue:
        return self.fb.add(llvm.AddOp(a, b)).results[0]

    def sub32(self, a: SSAValue, b: SSAValue) -> SSAValue:
        return self.fb.add(llvm.SubOp(a, b)).results[0]

    def shl32(self, a: SSAValue, n: int) -> SSAValue:
        return self.fb.add(llvm.ShlOp(a, self.i32c(n))).results[0]

    def ashr32(self, a: SSAValue, n: int) -> SSAValue:
        return self.fb.add(llvm.AShrOp(a, self.i32c(n))).results[0]

    def lshr32(self, a: SSAValue, n: int) -> SSAValue:
        return self.fb.add(llvm.LShrOp(a, self.i32c(n))).results[0]

    def not32(self, a: SSAValue) -> SSAValue:
        return self.xor32(a, self.i32c(-1))

    def blend32(self, mask: SSAValue, when_set: SSAValue, when_clear: SSAValue) -> SSAValue:
        return self.or32(self.and32(when_set, mask), self.and32(when_clear, self.not32(mask)))

    def smax32(self, a: SSAValue, b: SSAValue) -> SSAValue:
        d = self.sub32(a, b)
        return self.add32(b, self.and32(d, self.not32(self.ashr32(d, 31))))

    def smin32(self, a: SSAValue, b: SSAValue) -> SSAValue:
        d = self.sub32(a, b)
        return self.sub32(a, self.and32(d, self.not32(self.ashr32(d, 31))))

    # -- select-free float predicates --------------------------------------------------------
    def fmax(self, a: SSAValue, b: SSAValue) -> SSAValue:
        """`max(a, b)` for finite (or -inf) operands, with no compare and no select.

        `a - b` is negative exactly when `a < b`, so its sign bit -- broadcast by an
        arithmetic shift -- selects between the two operands' bit patterns.
        """
        mask = self.ashr32(self.bits(self.fsub(a, b)), 31)
        return self.unbits(self.blend32(mask, self.bits(b), self.bits(a)))

    def fmin(self, a: SSAValue, b: SSAValue) -> SSAValue:
        mask = self.ashr32(self.bits(self.fsub(a, b)), 31)
        return self.unbits(self.blend32(mask, self.bits(a), self.bits(b)))

    def fabs(self, a: SSAValue) -> SSAValue:
        return self.unbits(self.and32(self.bits(a), self.i32c(0x7FFFFFFF)))

    def copysign(self, magnitude: SSAValue, sign_of: SSAValue) -> SSAValue:
        sign = self.and32(self.bits(sign_of), self.i32c(-0x80000000))
        return self.unbits(self.or32(self.and32(self.bits(magnitude),
                                                self.i32c(0x7FFFFFFF)), sign))

    def fclamp(self, a: SSAValue, lo: float, hi: float) -> SSAValue:
        return self.fmin(self.fmax(a, self.fconst(lo)), self.fconst(hi))

    # -- transcendentals ---------------------------------------------------------------------
    def exp(self, x: SSAValue, *, bounded: bool = False) -> SSAValue:
        """`exp(x)` as a generated 2**k * poly(r) sequence (no libm call).

        ``bounded`` is reserved for compiler-generated callers that have already
        established the stricter ``[-87, 88]`` input range with the same f32
        operations.  General source ``exp`` operations retain the defensive clamp.
        """
        if not bounded:
            x = self.fclamp(x, -87.0, 88.0)
        y = self.fadd(self.fmul(x, self.fconst(_LOG2E)), self.fconst(_MAGIC))
        kbits = self.sub32(self.and32(self.bits(y), self.i32c(0x7FFFFF)), self.i32c(0x400000))
        kbits = self.smin32(self.smax32(kbits, self.i32c(-126)), self.i32c(127))
        kf = self.fsub(y, self.fconst(_MAGIC))
        r = self.fsub(self.fsub(x, self.fmul(kf, self.fconst(_LN2_HI))),
                      self.fmul(kf, self.fconst(_LN2_LO)))
        acc = self.fconst(_EXP_POLY[-1])
        for c in reversed(_EXP_POLY[:-1]):
            acc = self.fadd(self.fmul(acc, r), self.fconst(c))
        scale = self.unbits(self.shl32(self.add32(kbits, self.i32c(127)), 23))
        return self.fmul(acc, scale)

    def erf(self, x: SSAValue) -> SSAValue:
        ax = self.fabs(x)
        t = self.fdiv(self.fconst(1.0),
                      self.fadd(self.fconst(1.0), self.fmul(self.fconst(_ERF_P), ax)))
        acc = self.fconst(_ERF_A[-1])
        for c in reversed(_ERF_A[:-1]):
            acc = self.fadd(self.fmul(acc, t), self.fconst(c))
        poly = self.fmul(t, acc)
        e = self.exp(self.fneg(self.fmul(ax, ax)))
        y = self.fsub(self.fconst(1.0), self.fmul(poly, e))
        return self.copysign(y, x)

    def rsqrt(self, x: SSAValue) -> SSAValue:
        """`1/sqrt(x)` by the reciprocal-sqrt seed plus three Newton steps."""
        half = self.fmul(self.fconst(0.5), x)
        y = self.unbits(self.sub32(self.i32c(_RSQRT_SEED), self.ashr32(self.bits(x), 1)))
        for _ in range(3):
            y = self.fmul(y, self.fsub(self.fconst(1.5),
                                       self.fmul(half, self.fmul(y, y))))
        return y

    def sqrt(self, x: SSAValue) -> SSAValue:
        return self.fmul(x, self.rsqrt(self.fadd(x, self.fconst(0.0))))

    def log(self, x: SSAValue) -> SSAValue:
        """`log(x)` for x > 0: split the exponent off the bit pattern, then an atanh series.

        `m = x / 2**e` lands in [1, 2), so `s = (m-1)/(m+1)` is in [0, 1/3] and the odd series
        `2*(s + s^3/3 + ... + s^9/9)` is accurate to ~1e-7 there.
        """
        u = self.bits(x)
        e = self.sub32(self.and32(self.lshr32(u, 23), self.i32c(0xFF)), self.i32c(127))
        mantissa = self.unbits(self.or32(self.and32(u, self.i32c(0x807FFFFF)),
                                         self.i32c(0x3F800000)))
        s = self.fdiv(self.fsub(mantissa, self.fconst(1.0)),
                      self.fadd(mantissa, self.fconst(1.0)))
        s2 = self.fmul(s, s)
        acc = self.fconst(1.0 / 9.0)
        for c in (1.0 / 7.0, 1.0 / 5.0, 1.0 / 3.0, 1.0):
            acc = self.fadd(self.fmul(acc, s2), self.fconst(c))
        log_m = self.fmul(self.fconst(2.0), self.fmul(s, acc))
        ef = self.unbits(self.shl32(self.add32(e, self.i32c(127)), 23))
        # 2**e as a float, then log(2**e) = e * ln2 -- e is recovered from that float rather than
        # converted, so no fp<->int instruction is needed.
        e_as_float = self.fmul(self.fconst(1.0), self.log2_of_pow2(ef))
        return self.fadd(self.fmul(e_as_float, self.fconst(0.6931471805599453)), log_m)

    def log2_of_pow2(self, p: SSAValue) -> SSAValue:
        """`log2(p)` where p is exactly a power of two: recover the exponent as a float.

        `(bits(p) >> 23) - 127` is the integer exponent; adding it to the 1.5*2**23 magic and
        subtracting the magic back turns it into a float without an int->fp instruction.
        """
        e = self.sub32(self.lshr32(self.bits(p), 23), self.i32c(127))
        y = self.unbits(self.add32(self.bits(self.fconst(_MAGIC)), e))
        return self.fsub(y, self.fconst(_MAGIC))

    def powf(self, x: SSAValue, y: SSAValue) -> SSAValue:
        """`x ** y` for x > 0, as exp(y * log(x)) -- the only form a generated lane can take."""
        return self.exp(self.fmul(y, self.log(x)))

    def tanh(self, x: SSAValue) -> SSAValue:
        # The first clamp maps every f32 bit pattern to a finite value in
        # [-15, 15] under this emitter's existing bitwise selection semantics.
        # Multiplication by exactly representable 2.0 therefore gives [-30, 30],
        # so exp's wider [-87, 88] clamp would be an identity.  Keep the first
        # clamp because it is part of the tanh approximation's saturation policy.
        z = self.exp(self.fmul(self.fclamp(x, -15.0, 15.0), self.fconst(2.0)),
                     bounded=True)
        return self.fsub(self.fconst(1.0),
                         self.fdiv(self.fconst(2.0), self.fadd(z, self.fconst(1.0))))

    def recip(self, x: SSAValue) -> SSAValue:
        return self.fdiv(self.fconst(1.0), x)

    # -- integer bridging --------------------------------------------------------------------
    def sitofp(self, v: SSAValue) -> SSAValue:
        """The f32 nearest to the i64 integer value `v` (the IEEE conversion)."""
        return self.fb.add(llvm.SIToFPOp(v, F32)).results[0]

    def fptosi(self, x: SSAValue) -> SSAValue:
        """`(i64) x`, truncated toward zero by standard LLVM semantics.

        Values reaching this bridge are range-checked by the source program
        (quantization clamps before narrowing). Emitting the target-neutral
        operation exposes a native conversion to the selected CPU backend and
        avoids rebuilding it from many integer operations per tensor element.
        """
        return self.fb.add(FPToSIOp(x, i64)).results[0]

    def round_near_even(self, x: SSAValue) -> SSAValue:
        """Round to an integral float, ties to even, with the LLVM intrinsic.

        The intrinsic exactly represents the source operation across targets;
        no fast-math flags, reassociation, or approximation is used.
        """
        return self.fb.add(RoundEvenOp(x)).results[0]

    # -- storage formats ---------------------------------------------------------------------
    def round_bf16(self, x: SSAValue) -> SSAValue:
        """Round an f32 to bf16 precision (round-half-to-even) and keep it in f32."""
        u = self.bits(x)
        bias = self.add32(self.and32(self.lshr32(u, 16), self.i32c(1)), self.i32c(0x7FFF))
        return self.unbits(self.and32(self.add32(u, bias), self.i32c(-0x10000)))

    def load(self, ptr: SSAValue, index: SSAValue, dtype: str) -> SSAValue:
        """Load one element of `dtype` from `ptr[index]`, widened to f32."""
        if dtype == "f32":
            return self.fb.add(llvm.LoadOp(self.fb.gep(ptr, index, F32), F32)).results[0]
        if dtype == "bf16":
            raw = self.fb.add(llvm.LoadOp(self.fb.gep(ptr, index, i16), i16)).results[0]
            wide = self.fb.add(llvm.ZExtOp(raw, i32)).results[0]
            return self.unbits(self.shl32(wide, 16))
        raise ValueError(f"no CPU-lane load for element type {dtype!r}")

    def store(self, value: SSAValue, ptr: SSAValue, index: SSAValue, dtype: str) -> None:
        if dtype == "f32":
            self.fb.add(llvm.StoreOp(value, self.fb.gep(ptr, index, F32)))
            return
        if dtype == "bf16":
            u = self.bits(value)
            bias = self.add32(self.and32(self.lshr32(u, 16), self.i32c(1)), self.i32c(0x7FFF))
            packed = self.lshr32(self.add32(u, bias), 16)
            half = self.fb.add(llvm.TruncOp(packed, i16)).results[0]
            self.fb.add(llvm.StoreOp(half, self.fb.gep(ptr, index, i16)))
            return
        raise ValueError(f"no CPU-lane store for element type {dtype!r}")
