"""Prove ordered binary32 requantization and signed-i8 output transitions.

These target-independent numeric contracts are evaluated over the complete stated
signed-i32 accumulator domain. They never use calibration samples or model names.
Callers select any allowed local error explicitly and retain model accuracy gates.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass

from xdsl.dialects.builtin import TensorType
from xdsl.ir import Operation, SSAValue

# ---- requantization ----------------------------------------------------------------------------


def f32(value):
    try:
        return struct.unpack("<f", struct.pack("<f", value))[0]
    except OverflowError:
        return math.copysign(math.inf, value)


def quantized(acc, scales, relu=False):
    value = f32(acc)
    for scale in scales:
        value = f32(value * scale)
    low = 0 if relu else -128
    if value >= 127:
        return 127
    if value <= low:
        return low
    return max(low, min(127, round(value)))


def prove_scale(scales, lo=-(1 << 31), hi=(1 << 31) - 1, relu=False):
    """Prove exact int8 equality at every representable accumulator in [lo,hi]."""
    scales = tuple(f32(x) for x in scales)
    if not scales or any(not math.isfinite(x) or x <= 0 for x in scales):
        raise ValueError("all scales must be finite positive f32 constants")
    if not (-(1 << 31) <= lo <= hi < (1 << 31)):
        raise ValueError("invalid i32 proof domain")
    combined = 1.0
    for scale in scales:
        combined = f32(combined * scale)
    if not math.isfinite(combined) or combined <= 0:
        raise ValueError("combined scale is not finite positive f32")

    def first_at_least(sequence, q):
        left, right = lo, hi + 1
        while left < right:
            mid = (left + right) // 2
            if quantized(mid, sequence, relu) >= q:
                right = mid
            else:
                left = mid + 1
        return left

    low = 0 if relu else -128
    for q in range(low + 1, 128):
        source = first_at_least(scales, q)
        target = first_at_least((combined,), q)
        if source != target:
            witness = min(source, target)
            raise ValueError(
                f"f32 reassociation changes output at accumulator {witness}: source={quantized(witness, scales, relu)}, store={quantized(witness, (combined,), relu)}"
            )
    return dict(
        scale=combined,
        source_scales=list(scales),
        accumulator_min=lo,
        accumulator_max=hi,
        rounding="nearest_even",
        output_min=low,
        output_max=127,
        proof="exhaustive monotone integer-output transition comparison",
        transitions=127 - low,
    )


def prove_scale_bound(scales, lo=-(1 << 31), hi=(1 << 31) - 1, relu=False):
    """Compute the exact worst output error over the complete accumulator domain.

    Both saturated int8 outputs are monotone step functions. Their values are
    constant between the union of their transition points, so comparing each
    transition and the domain endpoints covers every possible accumulator.
    This reports a bound; callers must explicitly select an error policy before
    replacing source arithmetic. It does not establish model-level quality.
    """
    scales = tuple(f32(s) for s in scales)
    if not scales or any(not math.isfinite(s) or s <= 0 for s in scales):
        raise ValueError("all scales must be finite positive f32 constants")
    if not (-(1 << 31) <= lo <= hi < (1 << 31)):
        raise ValueError("invalid i32 proof domain")
    scale = 1.0
    for s in scales:
        scale = f32(scale * s)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("combined scale is not finite positive f32")
    low = 0 if relu else -128
    boundaries = {lo, hi}
    transitions = []
    for q in range(low + 1, 128):
        thresholds = []
        for sequence in (scales, (scale,)):
            left, right = lo, hi + 1
            while left < right:
                mid = (left + right) // 2
                if quantized(mid, sequence, relu) >= q:
                    right = mid
                else:
                    left = mid + 1
            thresholds.append(left)
            if left <= hi:
                boundaries.add(left)
        if thresholds[0] != thresholds[1]:
            transitions.append(dict(output=q, source=thresholds[0], target=thresholds[1]))
    error = 0
    witness = None
    for acc in sorted(boundaries):
        source = quantized(acc, scales, relu)
        target = quantized(acc, (scale,), relu)
        delta = abs(source - target)
        if delta > error:
            error = delta
            witness = dict(accumulator=acc, source=source, target=target)
    return dict(
        scale=scale,
        source_scales=list(scales),
        accumulator_min=lo,
        accumulator_max=hi,
        rounding="nearest_even",
        output_min=low,
        output_max=127,
        proof="complete union of monotone integer-output transition points and endpoints",
        transitions=127 - low,
        differing_transitions=transitions,
        max_output_lsb_error=error,
        exact=error == 0,
        witness=witness,
        scope="Local readout error only; requires an explicit numerical policy and full-model quality gate.",
    )


def synthesize_store_scale(scales, lo=-(1 << 31), hi=(1 << 31) - 1, relu=False):
    """Solve exact readout constraints over every positive finite f32 scale.

    Source output transitions constrain target(T)>=q and target(T-1)<q.
    Each constraint is monotone in the ordered positive-float bit pattern
    (reversed for a negative accumulator). Their intersection is therefore one
    exact interval, obtained by integer binary searches without real rounding
    relaxations. A contradictory interval proves this zero-bias form infeasible.
    """
    scales = tuple(f32(s) for s in scales)
    if not scales or any(not math.isfinite(s) or s <= 0 for s in scales):
        raise ValueError("all scales must be finite positive f32 constants")
    if not (-(1 << 31) <= lo <= hi < (1 << 31)):
        raise ValueError("invalid i32 proof domain")

    def from_bits(bits):
        return struct.unpack("<f", struct.pack("<I", bits))[0]

    first, last = 1, 0x7F7FFFFF
    lower, upper = first, last
    low = 0 if relu else -128
    witnesses = {}
    transitions = []

    def constrain(acc, q, reached):
        nonlocal lower, upper

        def valid(bits):
            return (quantized(acc, (from_bits(bits),), relu) >= q) == reached

        if acc == 0:
            if not valid(first):
                lower = last + 1
                witnesses["constant"] = dict(accumulator=acc, output=q, reached=reached)
            return
        increasing = (acc > 0) == reached
        left, right = first, last + 1
        while left < right:
            mid = (left + right) // 2
            if valid(mid) == increasing:
                right = mid
            else:
                left = mid + 1
        if increasing:
            if left > lower:
                lower = left
                witnesses["lower"] = dict(accumulator=acc, output=q, reached=reached, scale_bits=left)
        else:
            if left - 1 < upper:
                upper = left - 1
                witnesses["upper"] = dict(accumulator=acc, output=q, reached=reached, scale_bits=left - 1)

    for q in range(low + 1, 128):
        left, right = lo, hi + 1
        while left < right:
            mid = (left + right) // 2
            if quantized(mid, scales, relu) >= q:
                right = mid
            else:
                left = mid + 1
        transitions.append((q, left))
        if left <= hi:
            constrain(left, q, True)
        if left > lo:
            constrain(left - 1, q, False)
        if lower > upper:
            return dict(
                exact=False,
                source_scales=list(scales),
                accumulator_min=lo,
                accumulator_max=hi,
                relu=relu,
                scale_bits_min=lower,
                scale_bits_max=upper,
                proof="contradictory exact constraints across every positive finite f32 store scale",
                conflicting_output=q,
                witnesses=witnesses,
            )
    combined = 1.0
    for s in scales:
        combined = f32(combined * s)
    bits = struct.unpack("<I", struct.pack("<f", combined))[0]
    bits = max(lower, min(upper, bits))
    scale = from_bits(bits)
    # Independently compare all target transitions before publishing a solution.
    for q, threshold in transitions:
        left, right = lo, hi + 1
        while left < right:
            mid = (left + right) // 2
            if quantized(mid, (scale,), relu) >= q:
                right = mid
            else:
                left = mid + 1
        if left != threshold:
            raise AssertionError("synthesized scale transition validation failed")
    return dict(
        exact=True,
        scale=scale,
        source_scales=list(scales),
        accumulator_min=lo,
        accumulator_max=hi,
        relu=relu,
        scale_bits_min=lower,
        scale_bits_max=upper,
        rounding="nearest_even",
        output_min=low,
        output_max=127,
        transitions=127 - low,
        proof="exact intersection of all source transition constraints over positive finite f32 scales; independently revalidated",
        witnesses=witnesses,
    )


def synthesize_bias(scales, biases, output_reciprocal, lo, hi, relu=False):
    """Solve all integer-bias transition constraints for a constant channel table.

    Source order is f32(acc), sequential f32 multiplications, f32 bias add,
    optional ReLU, f32 reciprocal multiply, nearest-even and signed saturation.
    Each target channel uses i32(acc+bias) then one f32 scale/round/saturate operation.
    The accepted bias range excludes accumulator overflow.
    """
    import numpy as np

    scales = tuple(f32(x) for x in scales)
    reciprocal = f32(output_reciprocal)
    if not scales or any(not math.isfinite(x) or x <= 0 for x in (*scales, reciprocal)):
        raise ValueError("positive finite scalar scales required")
    biases = np.asarray(biases, dtype=np.float32)
    if biases.ndim != 1 or not np.isfinite(biases).all():
        raise ValueError("finite channel bias vector required")
    if not (-(1 << 31) <= lo <= hi < (1 << 31)):
        raise ValueError("invalid accumulator range")
    scale = 1.0
    for s in (*scales, reciprocal):
        scale = f32(scale * s)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("unrepresentable store scale")
    qlo = 0 if relu else -128

    def source(xs):
        values = xs.astype(np.float32)
        for s in scales:
            values = np.multiply(values, np.float32(s), dtype=np.float32)
        values = np.add(values, biases, dtype=np.float32)
        if relu:
            values = np.maximum(values, np.float32(0))
        values = np.multiply(values, np.float32(reciprocal), dtype=np.float32)
        return np.clip(np.rint(values), qlo, 127)

    # Every transition gives an equality or bound on the integer bias.
    lower = np.full(len(biases), max(-(1 << 31), -(1 << 31) - lo), dtype=np.int64)
    upper = np.full(len(biases), min((1 << 31) - 1, (1 << 31) - 1 - hi), dtype=np.int64)
    first_failure = np.full(len(biases), -999, dtype=np.int32)
    for q in range(qlo + 1, 128):
        left = np.full(len(biases), lo, dtype=np.int64)
        right = np.full(len(biases), hi + 1, dtype=np.int64)
        for _ in range((hi - lo + 1).bit_length()):
            active = left < right
            mid = (left + right) // 2
            reached = source(mid) >= q
            right = np.where(active & reached, mid, right)
            left = np.where(active & ~reached, mid + 1, left)
        source_threshold = left
        tl, tr = -(1 << 31), (1 << 31)
        while tl < tr:
            mid = (tl + tr) // 2
            if quantized(mid, (scale,), relu) >= q:
                tr = mid
            else:
                tl = mid + 1
        candidate = tl - source_threshold
        interior = (source_threshold > lo) & (source_threshold <= hi)
        lower = np.maximum(lower, np.where(source_threshold == lo, tl - lo, np.where(interior, candidate, lower)))
        upper = np.minimum(
            upper, np.where(source_threshold == hi + 1, tl - hi - 1, np.where(interior, candidate, upper))
        )
        first_failure = np.where((first_failure == -999) & (lower > upper), q, first_failure)
    accepted = lower <= upper
    return dict(
        scale=scale,
        source_scales=list(scales),
        output_reciprocal=reciprocal,
        accumulator_min=lo,
        accumulator_max=hi,
        relu=relu,
        proof="all monotone output transition constraints, including original f32 bias ordering",
        channels=len(biases),
        accepted_channels=int(accepted.sum()),
        integer_bias=[int(max(low, min(high, 0))) if ok else None for low, high, ok in zip(lower, upper, accepted)],
        refused_channels=[
            {
                "channel": i,
                "conflicting_transition": int(first_failure[i]),
                "bias_lower": int(lower[i]),
                "bias_upper": int(upper[i]),
            }
            for i in range(len(biases))
            if not accepted[i]
        ],
    )


# ---- integer_gemm ------------------------------------------------------------------------------
# Exact integer GEMM patterns in upstream linalg-on-tensors IR.
#
# Only dense row-major, from-zero i8 x i8 -> i32 contractions match.  A
# provenance label is never sufficient: indexing maps, iterator kinds, the
# integer region body, operand shapes and accumulator initialization are checked.


@dataclass(frozen=True)
class IntegerGemm:
    batch: int
    m: int
    n: int
    k: int


_MAPS_2D = ((0, 2), (2, 1), (0, 1))
_MAPS_3D = ((0, 1, 3), (0, 3, 2), (0, 1, 2))


def _shape(value: SSAValue, dtype: str) -> tuple[int, ...] | None:
    ty = value.type
    if not isinstance(ty, TensorType) or str(ty.get_element_type()) != dtype:
        return None
    return tuple(int(x) for x in ty.get_shape())


def _zero_init(value: SSAValue) -> bool:
    producer = value.owner
    if not isinstance(producer, Operation) or producer.name not in ("linalg.fill", "tensor.splat"):
        return False
    scalar = producer.operands[0]
    src = scalar.owner
    if not isinstance(src, Operation) or src.name != "arith.constant":
        return False
    attr = src.properties.get("value")
    if attr is None:
        attr = src.attributes.get("value")
    data = getattr(getattr(attr, "value", None), "data", None)
    return data is not None and int(data) == 0


def _canonical_maps(op: Operation, rank: int) -> bool:
    maps = op.properties.get("indexing_maps")
    iters = op.properties.get("iterator_types")
    expected = _MAPS_2D if rank == 2 else _MAPS_3D
    if maps is None or iters is None or len(maps.data) != 3 or len(iters.data) != rank + 1:
        return False
    if [x.data.value for x in iters.data] != ["parallel"] * rank + ["reduction"]:
        return False
    for amap, indices in zip(maps.data, expected):
        data = amap.data
        if data.num_dims != rank + 1 or data.num_symbols != 0:
            return False
        if tuple(getattr(expr, "position", None) for expr in data.results) != indices:
            return False
    return True


def _canonical_body(op: Operation) -> bool:
    if len(op.regions) != 1 or len(op.regions[0].blocks) != 1:
        return False
    block = op.regions[0].blocks[0]
    ops = list(block.ops)
    if [x.name for x in ops] != ["arith.extsi", "arith.extsi", "arith.muli", "arith.addi", "linalg.yield"]:
        return False
    if len(block.args) != 3 or [str(x.type) for x in block.args] != ["i8", "i8", "i32"]:
        return False
    ext_a, ext_b, mul, add, result = ops
    return (
        list(ext_a.operands) == [block.args[0]]
        and list(ext_b.operands) == [block.args[1]]
        and str(ext_a.results[0].type) == "i32"
        and str(ext_b.results[0].type) == "i32"
        and list(mul.operands) == [ext_a.results[0], ext_b.results[0]]
        and str(mul.results[0].type) == "i32"
        and set(add.operands) == {mul.results[0], block.args[2]}
        and str(add.results[0].type) == "i32"
        and list(result.operands) == [add.results[0]]
    )


def match_integer_gemm(op: Operation) -> IntegerGemm | None:
    """Return logical batch/M/N/K only when this op has exact GEMM semantics."""
    if op.name not in ("linalg.matmul", "linalg.generic") or len(op.operands) != 3 or len(op.results) != 1:
        return None
    lhs = _shape(op.operands[0], "i8")
    rhs = _shape(op.operands[1], "i8")
    init = _shape(op.operands[2], "i32")
    out = _shape(op.results[0], "i32")
    if lhs is None or rhs is None or init is None or out != init or not _zero_init(op.operands[2]):
        return None
    if op.name == "linalg.matmul":
        if len(lhs) != 2 or len(rhs) != 2:
            return None
    else:
        if len(lhs) not in (2, 3) or len(rhs) != len(lhs) or not _canonical_maps(op, len(lhs)):
            return None
        if not _canonical_body(op):
            return None
    if len(lhs) == 2:
        m, k = lhs
        rk, n = rhs
        batch = 1
        expected_out = (m, n)
    else:
        batch, m, k = lhs
        rb, rk, n = rhs
        if rb != batch:
            return None
        expected_out = (batch, m, n)
    if min(batch, m, n, k) <= 0 or k != rk or out != expected_out:
        return None
    return IntegerGemm(batch, m, n, k)


# ---- requantization_match ----------------------------------------------------------------------
# Typed source requantization analysis without device schedule selection.


def _constant(value):
    op = value.owner
    if not isinstance(op, Operation) or op.name != "arith.constant" or str(value.type) != "f32":
        raise ValueError("epilogue needs uniform f32 constants")
    attr = op.properties.get("value", op.attributes.get("value"))
    data = getattr(getattr(attr, "value", None), "data", None)
    if data is None:
        raise ValueError("not a scalar float constant")
    return f32(data)


@dataclass
class RequantizedContraction:
    contraction: Operation
    epilogue: Operation
    dimensions: object
    relu: bool
    proof: dict
    integer_bias: list[int] | None = None


def match(epilogue):
    """Retain the legacy uniform readout proof over the shared source matcher."""
    from merlin.compiler.lower.quant.readout_match import match_f32_readout
    from merlin.compiler.ir.readout import constant_parameter

    selected = match_f32_readout(epilogue)
    before, after, biases = [], [], None
    for kind, index in selected.steps:
        values, shape = constant_parameter(selected.parameters[index])
        if kind == "bias":
            biases = list(values)
        else:
            if shape:
                raise ValueError("legacy uniform readout proof requires scalar scales")
            (before if biases is None else after).append(values[0])
    if len(after) > 1:
        raise ValueError("legacy bias proof supports at most one post-bias scale")
    dims = selected.dimensions
    bound = dims.k * 128 * 128
    low, high = max(-(1 << 31), -bound), min((1 << 31) - 1, bound)
    proof = (
        prove_scale(before, low, high, selected.relu)
        if biases is None
        else synthesize_bias(before, biases, after[0] if after else 1.0, low, high, selected.relu)
    )
    if biases is not None and proof["accepted_channels"] != dims.n:
        raise ValueError(f"integer bias cannot preserve all channels: {proof['accepted_channels']}/{dims.n}")
    return RequantizedContraction(selected.contraction, epilogue, dims, selected.relu, proof, proof.get("integer_bias"))


# ---- integer_producer_range --------------------------------------------------------------------
# Conservative integer sum-of-products domains supplied by a typed producer.
#
# The caller must bind these facts to the actual producer, output storage and
# consumer. This arithmetic certificate does not establish that source binding.
# It admits exact signed-i32 accumulation only, with every prefix in range.


@dataclass(frozen=True)
class IntegerSumProductsRange:
    terms: int
    lhs_min: int
    lhs_max: int
    rhs_min: int
    rhs_max: int
    initial_min: int = 0
    initial_max: int = 0

    def interval(self):
        values = (
            self.terms,
            self.lhs_min,
            self.lhs_max,
            self.rhs_min,
            self.rhs_max,
            self.initial_min,
            self.initial_max,
        )
        if any(type(value) is not int for value in values) or self.terms < 0:
            raise ValueError("integer bounds and nonnegative term count required")
        for low, high in (
            (self.lhs_min, self.lhs_max),
            (self.rhs_min, self.rhs_max),
            (self.initial_min, self.initial_max),
        ):
            if not -(1 << 31) <= low <= high < (1 << 31):
                raise ValueError("ordered signed-i32 source bounds required")
        products = [a * b for a in (self.lhs_min, self.lhs_max) for b in (self.rhs_min, self.rhs_max)]
        low = self.initial_min + self.terms * min(products)
        high = self.initial_max + self.terms * max(products)
        prefix_low = min(self.initial_min, low)
        prefix_high = max(self.initial_max, high)
        if prefix_low < -(1 << 31) or prefix_high >= (1 << 31):
            raise ValueError("possible signed-i32 accumulation overflow")
        if self.terms and (min(products) < -(1 << 31) or max(products) >= (1 << 31)):
            raise ValueError("possible signed-i32 product overflow")
        return low, high

    def require_contained(self, low, high):
        actual_low, actual_high = self.interval()
        if actual_low < low or actual_high > high:
            raise ValueError("readout domain does not cover producer interval")
        return actual_low, actual_high


@dataclass(frozen=True)
class ConstantIntegerSumProductsRange:
    """Ordered integer contraction with an immutable constant weight vector.

    Every activation may independently take any integer in ``[lhs_min,lhs_max]``.
    The interval includes source products and every increasing-K prefix. A
    caller still binds the constants, activation type/range, seed, source order,
    and consumer to its actual typed source; this certificate changes no IR.
    """

    weights: tuple[int, ...]
    lhs_min: int
    lhs_max: int
    initial_min: int = 0
    initial_max: int = 0

    def _intervals(self):
        if type(self.weights) is not tuple:
            raise ValueError("immutable integer weight tuple required")
        bounds = (self.lhs_min, self.lhs_max, self.initial_min, self.initial_max)
        if any(type(value) is not int for value in (*bounds, *self.weights)):
            raise ValueError("literal integer weights and bounds required")
        for low, high in ((self.lhs_min, self.lhs_max), (self.initial_min, self.initial_max)):
            if not -(1 << 31) <= low <= high < (1 << 31):
                raise ValueError("ordered signed-i32 source bounds required")
        if any(not -(1 << 31) <= weight < (1 << 31) for weight in self.weights):
            raise ValueError("signed-i32 constant weights required")
        low, high = self.initial_min, self.initial_max
        prefix_low, prefix_high = low, high
        for weight in self.weights:
            products = (self.lhs_min * weight, self.lhs_max * weight)
            product_low, product_high = min(products), max(products)
            if product_low < -(1 << 31) or product_high >= (1 << 31):
                raise ValueError("possible signed-i32 product overflow")
            low += product_low
            high += product_high
            prefix_low, prefix_high = min(prefix_low, low), max(prefix_high, high)
            if prefix_low < -(1 << 31) or prefix_high >= (1 << 31):
                raise ValueError("possible signed-i32 accumulation prefix overflow")
        return (low, high), (prefix_low, prefix_high)

    def interval(self):
        """Final integer enclosure, after checking all source prefixes."""
        return self._intervals()[0]

    def prefix_interval(self):
        """Integer enclosure of the seed and every source prefix."""
        return self._intervals()[1]

    def require_exact_binary_conversion(self, *, significand_bits: int):
        """Sufficient exact-cast proof for a binary floating significand.

        All integers through +/-2**p are exactly representable with p bits.
        This check is conservative above that range. It grants no permission
        to move or reassociate any subsequent rounded floating arithmetic.
        """
        if type(significand_bits) is not int or not 2 <= significand_bits <= 64:
            raise ValueError("binary significand width in [2,64] required")
        low, high = self.interval()
        limit = 1 << significand_bits
        if low < -limit or high > limit:
            raise ValueError("integer domain may require an inexact floating conversion")
        return low, high
