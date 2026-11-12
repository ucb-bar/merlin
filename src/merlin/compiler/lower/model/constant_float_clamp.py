"""Preserve IEEE constant-clamp values with one original-input NaN guard.

This is an opt-in legalization of ordinary LLVM maximum/minimum call groups.
The helper retains the original two calls on the NaN path. Its non-NaN path uses
maxnum/minnum: nonzero finite ordered bounds exclude their signed-zero tie
ambiguity. No input finiteness assumption, reassociation, or memory motion is
introduced. Constrained, strict, or explicit floating-environment scopes refuse.
The guard freezes its input first, preserving defined values and safely refining
undef/poison without introducing a branch on poison.
"""

from __future__ import annotations

import hashlib
from decimal import Decimal

from merlin.compiler.lower.model.late_quant_rne import _functions, _identity, _instruction, _tokens

_GENERATED = "__merlin_ieee_const_clamp_"
_ENV_NAMES = frozenset(
    (
        "strictfp",
        "fegetround",
        "fesetround",
        "fetestexcept",
        "feclearexcept",
        "feraiseexcept",
        "feholdexcept",
        "feupdateenv",
        "fegetenv",
        "fesetenv",
        "fflags",
        "fcsr",
        "frflags",
        "fsflags",
        "frrm",
        "fsrm",
    )
)


def _has_environment_scope(tokens) -> bool:
    for token in tokens:
        identity = _identity(token.text)
        if identity.startswith("llvm.experimental.constrained."):
            return True
        # Quoted inline-assembly strings can name environment registers. Decode
        # their LLVM bytes and inspect identifier words, never comment text.
        words = "".join(c if c.isalnum() or c in "_." else " " for c in identity).split()
        if any(word in _ENV_NAMES for word in words):
            return True
    return False


def _call(line: str) -> dict | None:
    tokens = _tokens(line)
    op = _instruction(tokens, 0) if tokens else None
    if (
        op is None
        or op.opcode != "call"
        or op.dtype != "float"
        or op.callee not in ("llvm.maximum.f32", "llvm.minimum.f32")
        or not op.operands[0].startswith("%")
        or len(tokens) != 12
    ):
        return None
    return dict(
        indent=line[: tokens[0].start],
        result=op.result,
        kind=op.callee.split(".")[1],
        value=op.operands[0],
        bound=op.operands[1],
    )


def _ssa_key(value: str) -> tuple[str, str | int]:
    # LLVM unnamed numeric slots and quoted names containing the same digits
    # are distinct. Ordinary quoted/unquoted named identifiers share identity.
    spelling = value[1:]
    if spelling and all(c in "0123456789" for c in spelling):
        return "slot", int(spelling)
    return "name", _identity(value)


def _decimal_literal(text: str) -> bool:
    def digits(part: str) -> bool:
        return bool(part) and all(c in "0123456789" for c in part)

    def unsigned(part: str) -> str:
        return part[1:] if part.startswith(("+", "-")) else part

    mantissa, marker, exponent = text.lower().partition("e")
    if marker and not digits(unsigned(exponent)):
        return False
    mantissa = unsigned(mantissa)
    whole, dot, fraction = mantissa.partition(".")
    return bool(dot and (whole or fraction) and (not whole or digits(whole)) and (not fraction or digits(fraction)))


def _finite_f32(text: str) -> tuple[int, str] | None:
    """Decode a typed literal with integer RNE, independent of host FTZ/DAZ."""
    try:
        if text.startswith("0x") and len(text) == 18 and all(c in "0123456789abcdefABCDEF" for c in text[2:]):
            binary64 = int(text[2:], 16)
            sign = binary64 >> 63
            exponent = (binary64 >> 52) & 2047
            fraction = binary64 & ((1 << 52) - 1)
            if exponent == 2047:
                return None
            coefficient = fraction if exponent == 0 else fraction + (1 << 52)
            power = -1074 if exponent == 0 else exponent - 1023 - 52
            numerator = coefficient << max(power, 0)
            denominator = 1 << max(-power, 0)
        elif _decimal_literal(text):
            sign, digits, exponent = Decimal(text).as_tuple()
            if len(digits) > 256 or abs(exponent) > 1000:
                return None
            coefficient = int("".join(map(str, digits)))
            numerator = coefficient * 10 ** max(exponent, 0)
            denominator = 10 ** max(-exponent, 0)
        else:
            return None
    except ValueError:
        return None
    if numerator == 0:
        return None
    power = numerator.bit_length() - denominator.bit_length()
    below = (numerator < denominator << power) if power >= 0 else (numerator << -power < denominator)
    power -= bool(below)
    if power < -150 or power > 127:
        return None
    quantum = max(power - 23, -149)
    scaled_numerator = numerator << max(-quantum, 0)
    scaled_denominator = denominator << max(quantum, 0)
    mantissa, remainder = divmod(scaled_numerator, scaled_denominator)
    twice = 2 * remainder
    mantissa += twice > scaled_denominator or (twice == scaled_denominator and mantissa & 1)
    if not mantissa:
        return None
    if mantissa >= 1 << 24:
        mantissa >>= 1
        quantum += 1
    if mantissa < 1 << 23:
        assert quantum == -149
        bits = (sign << 31) | mantissa
    else:
        exponent32 = quantum + 23 + 127
        if exponent32 >= 255:
            return None
        bits = (sign << 31) | (exponent32 << 23) | (mantissa - (1 << 23))
    return bits, f"{bits:08x}"


def _ordered_f32_bits(lo_hex: str, hi_hex: str) -> bool:
    def key(bits: int) -> int:
        return (~bits & 0xFFFFFFFF) if bits >> 31 else bits ^ 0x80000000

    return key(int(lo_hex, 16)) <= key(int(hi_hex, 16))


def _helper(name: str, lo: str, hi: str) -> str:
    return f"""define internal float @{name}(float %x) alwaysinline nounwind memory(none) {{
entry:
  %frozen = freeze float %x
  %isnan = fcmp uno float %frozen, %frozen
  br i1 %isnan, label %nan, label %number
number:
  %number_lo = call float @llvm.maxnum.f32(float %frozen, float {lo})
  %number_hi = call float @llvm.minnum.f32(float %number_lo, float {hi})
  ret float %number_hi
nan:
  %original_lo = call float @llvm.maximum.f32(float %frozen, float {lo})
  %original_hi = call float @llvm.minimum.f32(float %original_lo, float {hi})
  ret float %original_hi
}}
"""


def rewrite(source: str) -> tuple[str, dict]:
    """Return a source-bound opt-in rewrite; unsupported scopes stay unchanged.

    Only contiguous groups of typed maximum/minimum intrinsics are eligible.
    Intervening intrinsic calls must be independent of the original maximum,
    and no operation crosses arithmetic, memory, unknown calls or basic blocks.
    If maximum has another use its exact original definition remains in place.
    Every minimum use retains its SSA
    definition. The ordinary source's existing floating policy is retained.
    """
    source_sha = hashlib.sha256(source.encode()).hexdigest()
    proof = dict(
        schema="constant_ieee_clamp_legalization_v1",
        source_sha256=source_sha,
        routes=[],
        policy="Existing ordinary LLVM floating semantics only; no additional FP environment or reassociation permission.",
    )
    tokens = _tokens(source)
    if _has_environment_scope(tokens):
        proof["refusal"] = "strict, constrained, or explicit floating-environment scope"
        proof["selected_sha256"] = source_sha
        return source, proof
    helpers = {}

    edits = []
    indices = {token.start: i for i, token in enumerate(tokens)}
    symbols = {_identity(token.text) for token in tokens if token.text.startswith("@")}
    for body_tokens in _functions(tokens):
        if not body_tokens:
            continue
        first_token = indices[body_tokens[0].start]
        last_token = indices[body_tokens[-1].start]
        opening, closing = tokens[first_token - 1], tokens[last_token + 1]
        assert opening.text == "{" and closing.text == "}"
        header_start = first_token - 2
        while header_start >= 0 and tokens[header_start].text != "define":
            header_start -= 1
        assert header_start >= 0
        function_name = next(token.text[1:] for token in tokens[header_start:first_token] if token.text.startswith("@"))
        if _identity("@" + function_name).startswith(_GENERATED):
            continue
        start = opening.end + (source[opening.end : opening.end + 1] == "\n")
        body = source[start : closing.start]
        lines = body.splitlines(keepends=True)
        typed = [_call(line) for line in lines]
        uses = {}
        for token in body_tokens:
            if token.text.startswith("%"):
                name = _ssa_key(token.text)
                uses[name] = uses.get(name, 0) + 1
        maxima = {}
        for i, minimum in enumerate(typed):
            if minimum is None:
                maxima.clear()
                continue
            if minimum["kind"] == "maximum":
                maxima[_ssa_key(minimum["result"])] = i
                continue
            if _ssa_key(minimum["value"]) not in maxima:
                continue
            first = maxima[_ssa_key(minimum["value"])]
            maximum = typed[first]
            between = typed[first + 1 : i]
            old_identity = _ssa_key(maximum["result"])
            if any(
                _ssa_key(call["value"]) == old_identity
                or (call["bound"].startswith("%") and _ssa_key(call["bound"]) == old_identity)
                for call in between
            ):
                continue
            lo = _finite_f32(maximum["bound"].strip())
            hi = _finite_f32(minimum["bound"].strip())
            if lo is None or hi is None or not _ordered_f32_bits(lo[1], hi[1]):
                continue
            identity = hashlib.sha256((lo[1] + ":" + hi[1]).encode()).hexdigest()[:16]
            name = _GENERATED + identity
            helper = _helper(name, maximum["bound"], minimum["bound"])
            if name not in helpers and name in symbols:
                raise ValueError("constant-clamp generated symbol already exists")
            # The physical helper identity uses decoded typed f32 bits, so
            # distinct valid literal spellings share the same exact endpoints.
            helpers.setdefault(name, helper)
            old = maximum["result"]
            retained = uses[old_identity] - 1 != 1
            if not retained:
                lines[first] = ""
            lines[i] = (
                minimum["indent"] + minimum["result"] + " = call float @" + name + "(float " + maximum["value"] + ")\n"
            )
            proof["routes"].append(
                dict(
                    input=maximum["value"],
                    maximum=old,
                    minimum=minimum["result"],
                    lower_f32_hex=lo[1],
                    upper_f32_hex=hi[1],
                    source_function=function_name,
                    source_body_line=i + 1,
                    intervening_independent_ieee_calls=len(between),
                    original_maximum_retained_for_other_uses=retained,
                    helper=name,
                    proof="Frozen input preserves every defined value and safely refines undef/poison, so the guard cannot introduce branch-on-poison UB. Non-NaN input plus finite nonzero ordered endpoints has identical maximum/minimum and maxnum/minnum values including signed zero; defined NaN executes original operations. Only independent typed IEEE intrinsic calls intervene; no other operation is crossed and all SSA uses are preserved.",
                )
            )
        edits.append((start, closing.start, "".join(lines)))

    selected = source
    for start, end, replacement in reversed(edits):
        selected = selected[:start] + replacement + selected[end:]
    if helpers:
        for name in ("maxnum", "minnum"):
            if "llvm." + name + ".f32" not in symbols:
                selected += f"\ndeclare float @llvm.{name}.f32(float, float)\n"
        selected += "\n" + "\n".join(helpers.values())
    proof["selected_sha256"] = hashlib.sha256(selected.encode()).hexdigest()
    return selected, proof
