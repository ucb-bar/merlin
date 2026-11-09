"""Scalar BF16 splats preserve authored finite words and both zero signs."""

from struct import unpack

import pytest

from merlin.semantic_compiler.linalg_bridge import LinalgBridgeError, translate_linalg_text
from merlin.semantic_compiler.reference import evaluate_graph


def _splat_source(literals):
    body = []
    for index, literal in enumerate(literals):
        body += [
            f"%c{index} = arith.constant {literal} : bf16",
            f"%s{index} = tensor.splat %c{index} : tensor<2x3xbf16>",
        ]
    types = ", ".join("tensor<2x3xbf16>" for _ in literals)
    results = ", ".join(f"%s{index}" for index in range(len(literals)))
    return f"module {{ func.func @forward() -> ({types}) {{ " + "\n".join(body) + (
        f"\nreturn {results} : {types} }} }}"
    )


@pytest.mark.parametrize("word", [0xFF7F, 0x7F7F, 0x3F80, 0xBF80, 0x0001, 0x8001, 0x0000, 0x8000])
@pytest.mark.parametrize("spelling", ["raw", "decimal"])
def test_finite_bf16_splat_preserves_exact_raw_word(word, spelling):
    literal = f"0x{word:04X}" if spelling == "raw" else repr(
        unpack("<f", (word << 16).to_bytes(4, "little"))[0]
    )
    translated = translate_linalg_text(
        _splat_source([literal]), entry="forward", target_identity="synthetic",
    )
    binding, = translated.request.constants
    assert binding.encoding == "bf16-le-raw"
    assert bytes.fromhex(binding.data_hex) == word.to_bytes(2, "little") * 6
    result, = evaluate_graph(translated.request, {}, constants=translated.constants)
    assert result.elements == (word,) * 6


def test_bf16_zero_signs_remain_distinct_constants():
    translated = translate_linalg_text(
        _splat_source(["0.0", "-0.0", "-0.0"]), entry="forward", target_identity="synthetic",
    )
    assert len(translated.request.constants) == 2
    assert translated.request.outputs[0] != translated.request.outputs[1]
    assert translated.request.outputs[1] == translated.request.outputs[2]
    assert [value.elements for value in evaluate_graph(
        translated.request, {}, constants=translated.constants,
    )] == [(0,) * 6, (0x8000,) * 6, (0x8000,) * 6]


@pytest.mark.parametrize("literal", ["0x7F80", "0xFF80", "0x7FC0", "0x7F81"])
def test_nonfinite_bf16_scalar_refuses(literal):
    with pytest.raises(LinalgBridgeError, match="BF16 scalar needs an exact finite typed literal"):
        translate_linalg_text(_splat_source([literal]), entry="forward", target_identity="synthetic")


def test_live_minimum_finite_decimal_spelling_preserves_raw_word():
    translated = translate_linalg_text(
        _splat_source(["-3.389530e+38"]), entry="forward", target_identity="synthetic",
    )
    binding, = translated.request.constants
    assert binding.encoding == "bf16-le-raw"
    assert bytes.fromhex(binding.data_hex) == b"\x7f\xff" * 6
