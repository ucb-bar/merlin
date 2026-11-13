"""Independent correctly rounded rotary binary32 arithmetic, using integers only.

No host/framework trigonometric implementation participates. Machin's identity
and alternating arctangent series enclose pi as exact Fractions. All subsequent
range reduction and Taylor operations use outward-rounded dyadic intervals.
The mathematical interval must fit one binary32 RNE cell; ambiguity refuses.
"""
from fractions import Fraction
from functools import lru_cache
from hashlib import sha256
from pathlib import Path

import numpy as np

POLICY = 'rotary-k1-f32-rne-cr-trig-scale1-bf16-rne-v3'
INV_FREQ_SHA256 = '734f9e6b629f7baccc4de81b7e8d2982035ee1b81f73f158e7d33a4dcf13fa0e'
PRECISION = 160
SCALE = 1 << PRECISION
TERMS = 28


def checked(path):
    path = Path(path)
    return path


def inv_freq(path=None):
    path = checked(path or Path(__file__).resolve().parents[1] / 'scripts/inv_freq.hex')
    data = bytes.fromhex(path.read_text().strip())
    if len(data) != 128 or sha256(data).hexdigest() != INV_FREQ_SHA256:
        raise ValueError('captured inverse-frequency bytes differ')
    return np.frombuffer(data, dtype='<u4').copy()


def _rne_div(n, d):
    q, r = divmod(n, d)
    return q + (2*r > d or (2*r == d and q & 1))


def ratio_to_f32(n, d=1):
    """Round an exact rational to finite binary32, with integer ties-to-even."""
    if d <= 0:
        raise ValueError('positive denominator required')
    sign = 0x80000000 if n < 0 else 0
    n = abs(n)
    if not n:
        return sign
    e = n.bit_length() - d.bit_length()
    below_power = n < (d << e) if e >= 0 else (n << -e) < d
    if below_power:
        e -= 1
    shift = 149 if e < -126 else 23-e
    sig = _rne_div(n << shift, d) if shift >= 0 else _rne_div(n, d << -shift)
    if e < -126:
        return sign | sig
    if sig == 1 << 24:
        sig >>= 1
        e += 1
    if e > 127:
        raise ValueError('nonfinite binary32 result is unqualified')
    return sign | ((e+127) << 23) | (sig & 0x7fffff)


def f32_ratio(raw):
    raw = int(raw)
    if not 0 <= raw <= 0xffffffff or raw & 0x7f800000 == 0x7f800000:
        raise ValueError('finite raw binary32 required')
    exponent, mantissa = (raw >> 23) & 255, raw & 0x7fffff
    n, e = (mantissa, -149) if exponent == 0 else ((1 << 23) | mantissa, exponent-150)
    if raw >> 31:
        n = -n
    return (n << e, 1) if e >= 0 else (n, 1 << -e)


def multiply_f32(left, right):
    a, b = np.broadcast_arrays(np.asarray(left), np.asarray(right))
    if a.dtype != np.uint32 or b.dtype != np.uint32:
        raise ValueError('binary32 multiply requires raw uint32 words')
    out = np.empty(a.shape, np.uint32)
    for index in np.ndindex(a.shape):
        x, y = int(a[index]), int(b[index])
        n, d = f32_ratio(x)
        m, e = f32_ratio(y)
        out[index] = ratio_to_f32(n*m, d*e) if n*m else ((x ^ y) & 0x80000000)
    return out


def _atan_interval(inverse):
    # Alternating terms decrease strictly. Consecutive partial sums bracket atan.
    total, power, index = Fraction(), inverse, 0
    while True:
        term = Fraction(1 if index % 2 == 0 else -1, (2*index+1)*power)
        following = total + term
        if abs(term) < Fraction(1, SCALE << 20):
            return min(total, following), max(total, following)
        total, power, index = following, power*inverse*inverse, index+1


@lru_cache(maxsize=1)
def _pi_over_two_interval():
    alo, ahi = _atan_interval(5)
    blo, bhi = _atan_interval(239)
    lo, hi = 8*alo-2*bhi, 8*ahi-2*blo
    return lo.numerator*SCALE//lo.denominator, -(-hi.numerator*SCALE//hi.denominator)


def _multiply_interval(a, b):
    values = [a[0]*b[0], a[0]*b[1], a[1]*b[0], a[1]*b[1]]
    return min(values)//SCALE, -(-max(values)//SCALE)


def _negate(a):
    return -a[1], -a[0]


def _series(midpoint, cosine):
    """Taylor enclosure at exact dyadic midpoint; every arithmetic error bounded."""
    x = (midpoint, midpoint)
    negative_square = _negate(_multiply_interval(x, x))
    term = (SCALE, SCALE) if cosine else x
    total = term
    for j in range(1, TERMS+1):
        term = _multiply_interval(term, negative_square)
        divisor = (2*j-1)*(2*j) if cosine else (2*j)*(2*j+1)
        term = term[0]//divisor, -(-term[1]//divisor)
        total = total[0]+term[0], total[1]+term[1]
    # |x|<=1. Lagrange remainder <=1/(2*TERMS+1)! for either series.
    # That factorial exceeds SCALE; one fixed-point unit bounds the remainder.
    factorial = 1
    for j in range(2, 2*TERMS+2):
        factorial *= j
    if factorial <= SCALE:
        raise ValueError('Taylor remainder precision proof failed')
    return total[0]-1, total[1]+1


def round_interval(lo, hi, denominator=SCALE):
    if lo > hi:
        raise ValueError('reversed interval')
    left, right = ratio_to_f32(lo, denominator), ratio_to_f32(hi, denominator)
    if left != right:
        raise ValueError('ambiguous binary32 rounding: interval crosses an RNE boundary')
    return left


@lru_cache(maxsize=65536)
def sincos_f32(raw):
    """Correctly rounded sin/cos for qualified nonnegative rotary f32 angles."""
    n, d = f32_ratio(raw)
    if int(raw) & 0x80000000 or n < 0 or n > 2047*d or SCALE % d:
        raise ValueError('rotary angle outside qualified positions 0..2047 domain')
    if n == 0:
        return 0, 0x3f800000
    value = n*(SCALE//d)
    pl, ph = _pi_over_two_interval()
    quadrant = _rne_div(2*value, pl+ph)
    lo, hi = value-quadrant*ph, value-quadrant*pl
    if not -SCALE <= lo <= hi <= SCALE:
        raise ValueError('range-reduction bound failed')
    midpoint = (lo+hi)//2
    radius = max(midpoint-lo, hi-midpoint)
    # |sin'|,|cos'|<=1: input uncertainty adds at most radius/SCALE.
    s, c = _series(midpoint, False), _series(midpoint, True)
    s, c = (s[0]-radius, s[1]+radius), (c[0]-radius, c[1]+radius)
    s, c = ((s, c), (c, _negate(s)), (_negate(s), _negate(c)), (_negate(c), s))[quadrant % 4]
    return round_interval(*s), round_interval(*c)


def trig_f32(raw, operation):
    raw = np.asarray(raw)
    if raw.dtype != np.uint32 or operation not in ('sin', 'cos'):
        raise ValueError('rotary trig requires raw binary32 and sin/cos')
    lane = 0 if operation == 'sin' else 1
    return np.array([sincos_f32(int(x))[lane] for x in raw.flat], np.uint32).reshape(raw.shape)


def angles(positions, frequency=None):
    positions = np.asarray(positions)
    if positions.dtype != np.int64 or np.any(positions < 0) or np.any(positions > 2047):
        raise ValueError('rotary positions require int64 values in 0..2047')
    frequency = inv_freq() if frequency is None else np.asarray(frequency)
    if frequency.dtype != np.uint32 or frequency.shape != (32,) or sha256(frequency.astype('<u4').tobytes()).hexdigest() != INV_FREQ_SHA256:
        raise ValueError('captured inverse-frequency words differ')
    encoded = np.array([ratio_to_f32(int(p)) for p in positions.flat], np.uint32).reshape(positions.shape)
    return multiply_f32(encoded[..., None], frequency)


def prepare(positions, frequency=None):
    angle = angles(positions, frequency)
    cat = np.concatenate((angle, angle), axis=-1)
    result = {'angles_f32': angle, 'cat_f32': cat}
    for name in ('sin', 'cos'):
        bits = trig_f32(cat, name)
        # Source attention scale is exactly +1: finite binary32 values unchanged.
        result[name+'_f32'] = bits
        result[name+'_bf16'] = ((bits+np.uint32(0x7fff)+((bits >> 16)&1)) >> 16).astype(np.uint16)
    return result
