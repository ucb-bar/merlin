"""Source-domain proofs required before fusing contraction readouts."""
import math

from frontend_composition import _dense_constant, _scalar_constant, recognize_mac


def _uniform_seed(value):
    owner = value.owner
    if getattr(owner, 'name', None) in ('linalg.fill', 'tensor.splat'):
        return _scalar_constant(owner.operands[0])
    values = _dense_constant(value)
    if values and all(v == values[0] for v in values):
        return values[0]
    return None


def _column_ranges(weights, k, n, transpose, seed):
    for column in range(n):
        coefficients = weights[column*k:(column+1)*k] if transpose else weights[column::n]
        lower = seed + sum(min(-128*w, 127*w) for w in coefficients)
        upper = seed + sum(max(-128*w, 127*w) for w in coefficients)
        yield lower, upper


def contraction_readout_refusal(readout):
    """Return a failed admission premise, using original typed SSA only.

    The source carrier has no semantic dataflow selector. Keep the conservative
    literal-weight mesh admission used for this carrier, even when the chosen
    device schedule splits K into WS passes. Power-of-two readouts also require
    exact integer-to-f32 conversion over the complete source accumulator domain;
    a power-of-two multiplier alone does not establish rational equality.
    """
    producer = readout['src'].owner
    if getattr(producer, 'name', None) not in ('linalg.generic', 'linalg.matmul'):
        return None
    mac = recognize_mac(producer)
    if mac is None:
        return None
    seed = _uniform_seed(mac['seed'])
    if seed is None:
        return None
    k = mac['lhs'].type.get_shape()[-1]
    n = readout['src'].type.get_shape()[-1]
    weights = _dense_constant(mac['rhs'])
    if weights is not None:
        for lower, upper in _column_ranges(weights, k, n, mac['transpose_rhs'], seed):
            if lower < -(1 << 19) or upper >= 1 << 19:
                return (f'constant-weight contraction partial sum range [{lower}, {upper}] '
                        'exceeds the signed 20-bit mesh bound')
    else:
        scales = readout.get('scales', [])
        bound = max(abs(seed - k*16256), abs(seed + k*16384))
        if bound > 1 << 24 and any(s != 0 and math.frexp(abs(s))[0] == 0.5 for s in scales):
            return (f'power-of-two contraction readout with runtime weights has accumulator bound {bound}; '
                    'exact integer-to-f32 equality requires a bound at most 2^24')
    return None
