"""Prove ordered binary32 requantization and signed-i8 output transitions.

These target-independent numeric contracts are evaluated over the complete stated
signed-i32 accumulator domain. They never use calibration samples or model names.
Callers select any allowed local error explicitly and retain model accuracy gates.
"""

import math
import struct


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
