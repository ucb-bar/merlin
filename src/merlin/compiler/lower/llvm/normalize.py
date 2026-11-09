"""Normalize MLIR printer LLVM literals for the selected LLVM parser."""

import struct as _struct

# The MLIR printer emits float immediates in MLIR's own hex spelling, e.g. the real line
#     %9 = fmul float f0x3F4C422A, %8
# from a bf16 gelu capsule's model.ll. LLVM's textual parser has no `f0x` literal, so the
# module fails to parse. LLVM IR wants a bare `0x…`, and a *float* hex literal must carry
# the 64-bit pattern of the double the f32 represents — so an 8-hex (f32) payload is
# widened, a 16-hex (f64) payload only sheds the `f`.
_F0X = "f0x"
_HEX_DIGITS = frozenset("0123456789abcdefABCDEF")


def _is_word_char(ch: str) -> bool:
    """The `\\b` before `f0x`: a preceding identifier character means this is the tail of a
    longer token (e.g. `named_f0x…`), not a float literal. `%` is not one, so `%f0x…` — an
    LLVM local whose name happens to start that way — IS treated as a literal, matching the
    behaviour this replaced."""
    return ch.isalnum() or ch == "_"


def _f0x_payload_to_llvm(hexs: str) -> str:
    if len(hexs) == 16:  # already a double bit pattern
        return f"0x{hexs.upper()}"
    bits = int(hexs, 16)  # f32 -> widen to double
    f32 = _struct.unpack("<f", _struct.pack("<I", bits))[0]
    dbits = _struct.unpack("<Q", _struct.pack("<d", f32))[0]
    return f"0x{dbits:016X}"


def normalize_hex_literals(ll_text: str) -> str:
    """Rewrite every `f0x<8 or 16 hex>` token to its LLVM `0x…` spelling.

    Scanned structurally: find the literal `f0x`, require an identifier boundary before it,
    then take the maximal run of hex digits after it. Only a run of EXACTLY 8 (f32) or 16
    (f64) digits is a float literal — any other length is not something this repair
    understands, so it is left untouched for the LLVM parser to reject loudly rather than
    being half-rewritten into a plausible-looking wrong constant.
    """
    out: list[str] = []
    i = 0
    while True:
        j = ll_text.find(_F0X, i)
        if j < 0:
            out.append(ll_text[i:])
            break
        out.append(ll_text[i:j])
        i = j + len(_F0X)
        if j > 0 and _is_word_char(ll_text[j - 1]):
            out.append(_F0X)  # inside a longer token — not a literal
            continue
        k = i
        while k < len(ll_text) and ll_text[k] in _HEX_DIGITS:
            k += 1
        if k - i not in (8, 16):
            out.append(_F0X)  # unknown payload width — leave verbatim
            continue
        out.append(_f0x_payload_to_llvm(ll_text[i:k]))
        i = k
    return "".join(out)


def normalize_float_literals(ll_text: str) -> str:
    """The MLIR LLVM-IR printer emits non-LLVM float literals (bare inf/-inf/nan,
    and `f0x..` f32 hex) that the textual parser rejects — canonicalize them."""
    ll_text = normalize_hex_literals(ll_text)
    # half (fp16) uses the LLVM `0xH<4hex>` literal form, bfloat (bf16) uses `0xR<4hex>` — the MLIR
    # printer emits bare inf/-inf/nan for these too (e.g. an fp16 causal-mask -inf), which the LLVM
    # textual parser rejects. Order matters: replace the widest names first so `float`↛`bfloat` etc.
    return (
        ll_text.replace("bfloat -inf", "bfloat 0xRFF80")
        .replace("bfloat inf", "bfloat 0xR7F80")
        .replace("bfloat nan", "bfloat 0xR7FC0")
        .replace("half -inf", "half 0xHFC00")
        .replace("half inf", "half 0xH7C00")
        .replace("half nan", "half 0xH7E00")
        .replace("float -inf", "float 0xFFF0000000000000")
        .replace("float inf", "float 0x7FF0000000000000")
        .replace("float nan", "float 0x7FF8000000000000")
        .replace("double -inf", "double 0xFFF0000000000000")
        .replace("double inf", "double 0x7FF0000000000000")
        .replace("double nan", "double 0x7FF8000000000000")
    )
