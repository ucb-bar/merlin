"""Rigorous, format-derived operand synthesis for the generated capsule corpus.

A capsule can only expose a compiler bug if its operands make that bug CHANGE THE OUTPUT. Row-identical or
near-symmetric operands hide whole bug classes: a wrong row stride, a swapped base offset, or a transposed
load all produce the SAME result on a matrix whose rows repeat or which equals its own transpose. The old
generator filled from 11 hand-picked magnitudes via a flat scalar hash, giving only ~6 distinct values and
~11 distinct rows of 32 — weak.

This module derives operands with three GUARANTEED properties, for any float format and any shape, with no
target literal and no regex:
  * every row is distinct and every column is distinct  -> row/col addressing & stride bugs change output;
  * the matrix is asymmetric (A != A^T)                 -> a transpose/layout bug is visible;
  * values span the format's representable range (a subnormal, mid, near-cap; both signs), exactly
    representable so the golden stays exact.

Construction: with a palette of P distinct representable values and P a PRIME >= max(rows, cols),
``v[r][c] = palette[(r + 2*c + salt) mod P]``. Prime P makes the row-stride 1 and col-stride 2 invertible
mod P, so rows differ (r1!=r2 => shifts differ), columns differ (2 coprime to P), and for any r!=c the
(1-2)*(r-c) offset is nonzero -> ``v[r][c] != v[c][r]`` (asymmetric). ``salt`` shifts the pattern per capsule
without breaking any property.
"""

from __future__ import annotations

import math

from merlin.common import target_data as _stimulus
from merlin.common.fp8_formats import canonical_float, representable_values

# ---- corpus_operands ---------------------------------------------------------------------------

# Keep magnitudes modest so a long-K reduction cannot overflow bf16 accumulation, while still spanning the
# dynamic range (subnormal -> a few powers of two). Widened automatically if a format is too sparse here.
_MAG_CAP = 16.0


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    i = 2
    while i * i <= n:
        if n % i == 0:
            return False
        i += 1
    return True


# The avalanche hash now lives in merlin.common.target_data, so the float operand synthesis here and the
# integer leaf fill in merlin.runtime.engine.command_buffer (and its C emitters) share ONE definition. Re-exported under
# the original private name because this module's callers and tests already use it.
_mix = _stimulus.mix


def _prime_at_least(k: int) -> int:
    n = max(2, k)
    while not _is_prime(n):
        n += 1
    return n


def derive_palette(fmt: str, n: int, *, mag_cap: float = _MAG_CAP) -> list[float]:
    """``n`` DISTINCT, exactly-representable values for ``fmt``, spread across a safe magnitude window and
    including both signs and the smallest positive (subnormal) value — derived from the format's own
    representable set, never a hand-picked list. Raises if the format cannot supply ``n`` distinct values."""
    canonical_float(fmt)  # fail closed on an unknown float format
    finite = representable_values(fmt)
    cap = mag_cap
    while True:
        pool = sorted({v for v in finite if 0.0 < abs(v) <= cap})  # nonzero, in-window
        if len(pool) >= n or cap >= max(abs(v) for v in finite):
            break
        cap *= 2.0  # sparse format -> widen the window
    if len(pool) < n:
        raise ValueError(f"format {fmt!r} has only {len(pool)} usable values, need {n}")
    # evenly-spaced picks across the sorted pool -> a genuine spread (subnormal..near-cap, both signs)
    step = len(pool) / n
    return [pool[int(i * step)] for i in range(n)]


def _usable_pool(fmt: str, max_alphabet: int | None, mag_cap: float = _MAG_CAP) -> list[float]:
    """Nonzero, exactly-representable values in a safe magnitude window (widened if the format is sparse),
    optionally capped to an evenly-spread ``max_alphabet`` values — the alphabet the operand is drawn from.
    A low-bit format (fp4 has 14 nonzero values; fp6 is LUT-limited to 16 per row) legitimately supplies a
    small alphabet; that is a datapath fact, not a degeneracy."""
    canonical_float(fmt)  # fail closed on an unknown float format
    finite = representable_values(fmt)
    cap = mag_cap
    while True:
        pool = sorted({v for v in finite if 0.0 < abs(v) <= cap})
        if len(pool) >= 2 or cap >= max(abs(v) for v in finite):
            break
        cap *= 2.0
    if max_alphabet is not None and len(pool) > max_alphabet:
        step = len(pool) / max_alphabet
        pool = [pool[int(i * step)] for i in range(max_alphabet)]
    return pool


def operand_values(
    shape: tuple[int, int], fmt: str, salt: int, *, max_alphabet: int | None = None, mag_cap: float = _MAG_CAP
) -> list[float]:
    """Flat row-major values for a 2-D operand of ``shape`` in ``fmt``, with distinct rows, distinct
    columns, and asymmetry guaranteed (see module docstring). ``salt`` (an int derived from the tensor
    name) shifts the pattern per capsule so operands vary but the guarantees hold.

    Two regimes: a format rich enough to supply a PRIME-sized palette (fp8/fp16/bf16/f32) uses the prime
    construction (byte-identical to before); a low-bit format whose alphabet is smaller than the shape (fp4,
    or LUT-capped fp6) uses a deterministic mixed construction that still yields distinct row/column TUPLES
    over a small alphabet, verified by :func:`rigor_findings` with a salt search (fail closed if none is
    rigorous). ``max_alphabet`` caps the alphabet (e.g. 16 for the fp6 LUT); ``mag_cap`` bounds operand
    magnitude (MX operands stay small so a block-scaled bf16 accumulate cannot overflow)."""
    rows, cols = shape
    p = _prime_at_least(max(rows, cols, 2))
    if max_alphabet is None:
        try:
            pal = derive_palette(fmt, p, mag_cap=mag_cap)  # prime construction (strongest guarantees)
            s = salt % p
            return [pal[(r + 2 * c + s) % p] for r in range(rows) for c in range(cols)]
        except ValueError:
            pass  # alphabet too small -> mixed construction below
    pool = _usable_pool(fmt, max_alphabet, mag_cap)
    a = len(pool)
    for attempt in range(256):  # deterministic salt search over the small alphabet
        s = salt + attempt
        vals = [pool[_mix(r, c, s) % a] for r in range(rows) for c in range(cols)]
        if not rigor_findings(vals, shape):
            return vals
    raise ValueError(f"format {fmt!r} alphabet ({a} values) too small to build a rigorous {rows}x{cols} operand")


def _achievable(n: int, alphabet: int, depth: int) -> int:
    """The most distinct ``depth``-long tuples an ``alphabet``-value stimulus can produce, capped at ``n``.

    The bound that makes this a rigor check rather than an impossibility check. A tuple of ``depth``
    values drawn from ``alphabet`` values has at most ``alphabet ** depth`` distinct forms, so demanding
    ``n`` of them when ``n`` exceeds that is not a standard an operand can meet -- it is arithmetic.
    Measured: a ``1 x 32`` gamma vector over the shared 4-value stimulus alphabet was reported as a
    degenerate operand, and no stimulus over that alphabet could have made it pass.

    The exponent is capped because ``alphabet ** rows`` for a tall operand is an enormous integer whose
    only use here is to be larger than ``n``.
    """
    if alphabet <= 1:
        return 1
    return min(n, alphabet ** min(depth, 20))


def rigor_limits(values: list[float], shape: tuple[int, int]) -> list[str]:
    """What this operand's own alphabet makes UNREACHABLE, as opposed to what it got wrong.

    Separate from :func:`rigor_findings` because the two are acted on differently: a finding is fixed by
    choosing a better stimulus, a limit is fixed only by widening the alphabet (or the shape), which is a
    decision about the dialect's operand types. Reporting them together would either fail a gate nothing
    can satisfy or lose the fact entirely; they are both recorded, in the place each belongs.
    """
    rows, cols = shape
    alphabet = len({float(v) for v in values})
    out: list[str] = []
    if _achievable(cols, alphabet, rows) < cols:
        out.append(
            f"{cols} columns over a {alphabet}-value alphabet in {rows} row(s): at most "
            f"{_achievable(cols, alphabet, rows)} can differ, so some column swaps are invisible "
            f"to any stimulus over this alphabet"
        )
    if _achievable(rows, alphabet, cols) < rows:
        out.append(
            f"{rows} rows over a {alphabet}-value alphabet in {cols} column(s): at most "
            f"{_achievable(rows, alphabet, cols)} can differ"
        )
    return out


def rigor_findings(values: list[float], shape: tuple[int, int]) -> list[str]:
    """Advisory: report the ways an operand would HIDE a bug — duplicate rows, duplicate columns, or
    symmetry (A == A^T). Empty list == rigorous. Target-agnostic; used by the corpus-rigor gate to keep a
    regeneration from silently degrading operand quality.

    Distinctness is judged against what the operand's own alphabet can ACHIEVE, not against its extent:
    see :func:`_achievable`. Nothing that passed before passes differently -- a fully distinct operand
    meets both bars -- but an operand that is as varied as its encoding permits is no longer reported as
    degenerate. What it cannot reach is recorded by :func:`rigor_limits` instead of being lost.
    """
    rows, cols = shape
    grid = [tuple(values[r * cols : (r + 1) * cols]) for r in range(rows)]
    alphabet = len({float(v) for v in values})
    out: list[str] = []
    want_rows = _achievable(rows, alphabet, cols)
    if len(set(grid)) < want_rows:
        out.append(
            f"duplicate rows: only {len(set(grid))} distinct of an achievable {want_rows} "
            f"({rows} rows) (row-addressing bugs invisible)"
        )
    colset = {tuple(grid[r][c] for r in range(rows)) for c in range(cols)}
    want_cols = _achievable(cols, alphabet, rows)
    if len(colset) < want_cols:
        out.append(
            f"duplicate columns: only {len(colset)} distinct of an achievable {want_cols} "
            f"({cols} columns) (col/stride bugs invisible)"
        )
    if rows == cols and all(grid[r][c] == grid[c][r] for r in range(rows) for c in range(cols)):
        out.append("operand is symmetric (A == A^T): a transpose/layout bug produces identical output")
    if len({v for row in grid for v in row}) <= 1:
        out.append("operand is constant (degenerate): almost any bug is invisible")
    return out


# E8M0 exponent half-window: block scales sweep +/- this many powers of two around the bias (scale 1.0).
# Kept SMALL on purpose — one distinct exponent per lane over a wide tile would span 2**lanes of dynamic
# range and overflow the bf16 accumulate to inf (a golden any broken kernel matches). +/-3 = scale in
# [2**-3, 2**3], enough distinct exponents (a 7-value alphabet) that a mis-indexed scale changes the output.
_E8M0_SPAN = 3


def e8m0_scale_codes(shape: tuple[int, int], salt: int, *, center: int = 127) -> list[list[int]]:
    """A rigorous, numerically-safe E8M0 block-scale stream for a ``(groups, lanes)`` shape (A: lanes=M rows;
    B: lanes=N cols). An MX matmul applies one block scale per (K-group, lane); if a lane's scale is
    mis-indexed the output must CHANGE, so the per-lane exponents VARY (they sweep a small window centred on
    ``center`` == the E8M0 bias 127, i.e. scale 1.0), with no two ADJACENT lanes equal and each lane also
    varying across groups. The window is only ``+/-_E8M0_SPAN`` wide so ``2**(code-127)`` stays modest and a
    bf16 accumulate cannot overflow. ``salt`` shifts the pattern per capsule; codes are clamped to the valid
    ``[0, 254]`` E8M0 range (255 is the NaN code)."""
    groups, lanes = shape
    period = 2 * _E8M0_SPAN + 1  # a `period`-value exponent alphabet
    codes = []
    for g in range(groups):
        row = []
        for li in range(lanes):
            off = ((li * 5 + g * 3 + salt) % period) - _E8M0_SPAN  # step 5 (coprime to 7) -> neighbours differ
            row.append(max(0, min(254, center + off)))
        codes.append(row)
    return codes


def scale_rigor_findings(codes: list[list[int]]) -> list[str]:
    """Advisory rigor for an E8M0 scale stream: a constant stream, or one where adjacent lanes never differ,
    hides scale-addressing bugs. Target-agnostic; used by the corpus-rigor gate so a regeneration cannot ship
    an all-equal (inert) scale stream. (An all-distinct check is deliberately NOT required: at wide tiles it
    would force a 2**lanes dynamic range — see :data:`_E8M0_SPAN`.)"""
    if not codes:
        return ["empty scale stream"]
    out: list[str] = []
    flat = [c for row in codes for c in row]
    if len(set(flat)) <= 1:
        out.append("scale stream is constant (degenerate): a mis-indexed block scale is invisible")
    if not any(any(row[c] != row[c + 1] for c in range(len(row) - 1)) for row in codes):
        out.append("no adjacent lanes differ in any group: lane-swap scale bugs invisible")
    return out


# ---- input_palette -----------------------------------------------------------------------------
# Explicit deterministic stress inputs, independent of numerical answers or schedules.
#
# Patterns select exactly representable scalar values along a declared tensor axis.
# Format landmarks are derived from the shared format registry/codecs. They never
# change an oracle's arithmetic, rounding order, tolerance or acceptance policy.

SCHEMA = "merlin.component_input_palette.v1"


def validate(declaration):
    if not isinstance(declaration, dict) or set(declaration) != {"schema", "inputs"} or declaration["schema"] != SCHEMA:
        raise ValueError("input palette requires the closed component_input_palette v1 schema")
    rows = declaration["inputs"]
    if not isinstance(rows, list) or not rows:
        raise ValueError("input palette requires explicit input patterns")
    names = set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"name", "axis", "values", "offset"}:
            raise ValueError("input pattern requires name, axis, values and offset")
        name = row["name"]
        if not isinstance(name, str) or not name or name in names:
            raise ValueError("input pattern names must be unique and explicit")
        names.add(name)
        if row["axis"] != "linear" and type(row["axis"]) is not int:
            raise ValueError("input pattern axis must be a dimension index or linear")
        if not isinstance(row["values"], list) or not row["values"]:
            raise ValueError("input pattern requires a nonempty scalar palette")
        if type(row["offset"]) is not int or not 0 <= row["offset"] < len(row["values"]):
            raise ValueError("input pattern offset must select a palette position")
        for value in row["values"]:
            if type(value) in (float, int) and math.isfinite(value):
                continue
            if isinstance(value, str) and value:
                continue
            raise ValueError("input palette values must be finite numbers or explicit format landmarks")
    return declaration


def scalar(value, dtype):
    from merlin.common import fp8_formats as formats
    from merlin.common.target_data import get

    descriptor = get(dtype)
    negative = isinstance(value, str) and value.startswith("-")
    token = value[1:] if negative else value
    if isinstance(value, str):
        if descriptor.is_float:
            exponent, mantissa, bias, scheme = formats.float_format_params(dtype)
            minimum, maximum = formats.normal_range(dtype)
            landmarks = {
                "zero": 0.0,
                "one": 1.0,
                "min_subnormal": 2.0 ** (1 - bias - mantissa),
                "min_normal": minimum,
                "max_finite": maximum,
                "half_ulp_one": 2.0 ** (-mantissa - 1),
                "cancellation_large": 2.0 ** (mantissa + 1),
                "infinity": float("inf"),
                "nan": float("nan"),
            }
            if token == "infinity" and scheme != "ieee":
                raise ValueError("selected float storage format cannot represent infinity")
            if token == "nan" and scheme == "mx_finite":
                raise ValueError("selected float storage format cannot represent NaN")
        else:
            bits = descriptor.element_bits
            landmarks = {"zero": 0, "one": 1, "max_signed": (1 << (bits - 1)) - 1, "min_signed": -(1 << (bits - 1))}
        if token not in landmarks:
            raise ValueError(f"input palette landmark {value!r} is unknown for {dtype!r}")
        value = landmarks[token]
        if negative:
            value = -value
    if descriptor.is_float:
        import numpy as np

        raw = formats.float_to_codes([value], dtype)
        decoded = float(formats._decode(np.asarray(raw), dtype)[0])
        if math.isnan(value):
            if not math.isnan(decoded):
                raise ValueError("input palette NaN cannot be represented in the selected storage format")
        elif decoded != value or (value == 0 and math.copysign(1, decoded) != math.copysign(1, value)):
            raise ValueError(f"input palette value {value!r} is not exactly representable in {dtype!r}")
        return value
    bits = descriptor.element_bits
    if type(value) not in (int, float) or int(value) != value or not -(1 << (bits - 1)) <= value < (1 << (bits - 1)):
        raise ValueError(f"input palette value exceeds signed {dtype!r} storage")
    return int(value)


def pattern(declaration, *, name, dtype, index=None):
    validate(declaration)
    candidates = [row for row in declaration["inputs"] if row["name"] == name]
    if index is not None and name != f"arg{index}":
        candidates += [row for row in declaration["inputs"] if row["name"] == f"arg{index}"]
    if len(candidates) > 1:
        raise ValueError("input palette names one tensor through conflicting named and positional selectors")
    if not candidates:
        candidates = [row for row in declaration["inputs"] if row["name"] == "*"]
    if not candidates:
        return None
    row = candidates[0]
    return {**row, "values": [scalar(value, dtype) for value in row["values"]]}


def realize(declaration, *, name, shape, dtype, index=None):
    selected = pattern(declaration, name=name, dtype=dtype, index=index)
    if selected is None:
        return None
    if not shape or any(type(dim) is not int or dim < 1 for dim in shape):
        raise ValueError("input palette needs a positive static tensor shape")
    count = math.prod(shape)
    axis = selected["axis"]
    if axis == "linear":
        stride, extent = 1, count
    else:
        if not -len(shape) <= axis < len(shape):
            raise ValueError("input palette axis is outside its tensor rank")
        axis %= len(shape)
        stride, extent = math.prod(shape[axis + 1 :]), shape[axis]
    values, offset = selected["values"], selected["offset"]
    if extent < len(values):
        raise ValueError("input palette axis is too short to exercise every declared value")
    return [values[((position // stride) % extent + offset) % len(values)] for position in range(count)]


def findings(declaration, *, name, shape, dtype, index=None):
    """Observable operand landmarks; execution/numerical equivalence is unproved."""
    values = realize(declaration, name=name, shape=shape, dtype=dtype, index=index)
    if values is None:
        return []
    from merlin.common.fp8_formats import normal_range
    from merlin.common.target_data import get

    result = set()
    finite = [value for value in values if math.isfinite(value)]
    if any(not math.isfinite(value) for value in values):
        result.add("nonfinite_input")
    if any(value == 0 and math.copysign(1, value) < 0 for value in values):
        result.add("signed_zero_input")
    if finite and min(finite) < 0 < max(finite):
        result.add("signed_input")
    descriptor = get(dtype)
    if descriptor.is_float:
        minimum, maximum = normal_range(dtype)
        if any(0 < abs(value) < minimum for value in finite):
            result.add("subnormal_input")
        if any(abs(value) == maximum for value in finite):
            result.add("maximum_finite_input")
        alphabet = set(finite)
        if 1.0 in alphabet and scalar("half_ulp_one", dtype) in alphabet:
            result.add("rounding_tie_input")
        if any(
            -value in alphabet and any(0 < abs(small) < abs(value) for small in alphabet) for value in alphabet if value
        ):
            result.add("cancellation_input")
    return sorted(result)


def render_source_inputs(declaration, names, dtype):
    """Emit one closed input-only override for an ordinary builtin Torch loader."""
    palettes = [pattern(declaration, name=name, dtype=dtype, index=index) for index, name in enumerate(names)]
    used = {row["name"] for row in palettes if row is not None}
    if any(row["name"] not in used for row in declaration["inputs"]):
        raise ValueError("input palette names a tensor absent from the generated PyTorch program")
    # Source contains the selected concrete values and a closed input-only
    # materializer. Capture binds its bytes and the actual typed operands.
    literals = []
    for palette in palettes:
        if palette is None:
            literals.append("None")
            continue
        values = [
            repr(value) if math.isfinite(value) else "float(" + repr(str(value)) + ")" for value in palette["values"]
        ]
        fields_literal = [repr(key) + ": " + repr(value) for key, value in palette.items() if key != "values"]
        literals.append("{" + ", ".join(fields_literal) + ", 'values': [" + ", ".join(values) + "]}")
    return _PALETTE_INPUTS.replace("__PALETTES__", "[" + ", ".join(literals) + "]")


_PALETTE_INPUTS = """
_PALETTES = __PALETTES__
_original_get_model_and_inputs = get_model_and_inputs
def get_model_and_inputs():
    model, inputs = _original_get_model_and_inputs()
    selected = []
    for tensor, palette in zip(inputs, _PALETTES, strict=True):
        if palette is None:
            selected.append(tensor)
            continue
        shape = tuple(tensor.shape)
        axis = palette["axis"]
        if axis == "linear":
            stride, extent = 1, tensor.numel()
        else:
            if not -len(shape) <= axis < len(shape):
                raise ValueError("input palette axis is outside its tensor rank")
            axis %= len(shape)
            stride, extent = math.prod(shape[axis + 1:]), shape[axis]
        values, offset = palette["values"], palette["offset"]
        if extent < len(values):
            raise ValueError("input palette axis cannot exercise every declared value")
        flat = [values[((position // stride) % extent + offset) % len(values)]
                for position in range(tensor.numel())]
        selected.append(torch.tensor(flat, dtype=tensor.dtype, device=tensor.device).reshape(shape))
    return model, tuple(selected)
"""
