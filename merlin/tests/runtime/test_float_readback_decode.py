"""A FLOAT result handed back over an integer console decodes to its VALUE, keyed on the declared dtype.

The shared backend console protocol (``OUT <name> <rows> <cols> <v...>``) carries whatever the device
harness printed. A bare-metal harness has no float formatting -- and would not want it, since a decimal
round-trip is the lossy step -- so it prints a float destination buffer's stored CONTAINER WORD. That
hand-back is lossless; what was missing was the reader. Ten gemmini capsules whose regions all belong on
the host lane were DECLINED by their backend for exactly this reason ("its result cannot be HANDED BACK
... an f32 store is read back as its raw i32 word, not as its value"), so a correctly compiled CPU-lane
program had no route to a grade at all.

The decode is keyed on the dtype the OUTPUT TENSOR IS DECLARED IN -- the same fact the harness sized the
buffer from -- and on nothing target-specific. These tests pin both directions: the right dtype recovers
the value, a wrong one does not (a check that cannot fail is worth nothing), and an integer readback is
returned untouched so every existing integer capsule parses byte-identically.
"""
from __future__ import annotations

import pytest

from merlin.runtime import fp8_formats as ff
from merlin.runtime.backends import base as bk
from merlin.runtime.commandbuffer import declared_output_dtypes

#: Values chosen to be EXACT in every format under test (bf16 has 8 significand bits), so a decode
#: failure is a decode failure and not a rounding argument.
VALUES = [-1.5, 0.25, 3.0, -0.125, 7.5, 2.0, -0.5, 1.0]


def _printed_as_signed(codes, bits: int) -> list[int]:
    """The code patterns as a SIGNED container's printf would put them on the wire (an ``int16_t``
    holding 0xBF80 prints -16512). The decode has to survive that; the harness's own float containers
    are unsigned, but a console is text and nothing downstream can tell which it was."""
    return [int(c) - (1 << bits) if int(c) >= (1 << (bits - 1)) else int(c) for c in codes]


# --- the primitive: code patterns <-> values, for every registered float format --------------------
@pytest.mark.parametrize("fmt", ["f32", "bf16", "fp16", "fp8_e4m3", "fp8_e5m2"])
def test_codes_to_f32_inverts_float_to_codes(fmt):
    """Round-trip on values the format itself declares representable, so the assertion is about the
    code<->value mapping and not about whether e5m2's two mantissa bits can hold 7.5 (they cannot)."""
    grid = ff.representable_values(fmt)
    values = [v for v in grid if 2 ** -8 <= abs(v) <= 2 ** 8][:32]
    assert values, fmt
    assert [float(v) for v in ff.codes_to_f32(ff.float_to_codes(values, fmt), fmt)] == values


@pytest.mark.parametrize("fmt", ["f32", "bf16", "fp16"])
def test_a_sign_extended_container_still_decodes(fmt):
    """The pattern survives a signed print: it is masked to the format's own storage width first."""
    codes = ff.float_to_codes(VALUES, fmt)
    wire = _printed_as_signed(codes, ff.storage_bits(fmt))
    assert any(w < 0 for w in wire), "the fixture must actually exercise sign extension"
    assert [float(v) for v in ff.codes_to_f32(wire, fmt)] == VALUES


def test_an_unregistered_format_fails_closed():
    with pytest.raises(KeyError):
        ff.codes_to_f32([0, 1], "not_a_format")


# --- which dtype a readback is decoded against ----------------------------------------------------
def test_declared_output_dtypes_reads_the_tensor_table():
    cb = {"tensors": {"X": {"shape": [2, 2], "dtype": "f32", "role": "input"},
                      "Y0": {"shape": [2, 2], "dtype": "bf16", "role": "output"}},
          "commands": []}
    assert declared_output_dtypes(cb) == {"X": "f32", "Y0": "bf16"}


def test_a_commands_own_output_dtype_wins_for_the_destination_it_names():
    """A movement/commit DECLARES the container its result lands in, and that is what a harness sizes
    the buffer from -- so the readback must read the same declaration, not the tensor's operand dtype."""
    cb = {"tensors": {"X": {"shape": [2, 2], "dtype": "i8", "role": "input"},
                      "Y0": {"shape": [2, 2], "dtype": "i8", "role": "output"}},
          "commands": [{"opcode": "VECTOR_MAP", "operands": {"lhs": "X", "dst": "Y0"},
                        "attributes": {"combine": "identity", "output_dtype": "i32"}}]}
    assert declared_output_dtypes(cb)["Y0"] == "i32"
    assert declared_output_dtypes(cb)["X"] == "i8", "a source is not re-declared by the command"


# --- the decode at the readback -------------------------------------------------------------------
def _rows(values, cols):
    return [values[i:i + cols] for i in range(0, len(values), cols)]


@pytest.mark.parametrize("dtype", ["f32", "bf16", "fp16"])
def test_a_float_declared_output_decodes_to_its_value(dtype):
    codes = _printed_as_signed(ff.float_to_codes(VALUES, dtype), ff.storage_bits(dtype))
    got = bk.decode_float_readback({"Y0": _rows(codes, 4)}, {"Y0": dtype})
    assert got == {"Y0": _rows(VALUES, 4)}


@pytest.mark.parametrize("wrong", ["i32", "i16", "i8"])
def test_declaring_the_wrong_dtype_does_not_decode(wrong):
    """MUTATION. The decode is not a formality that always fires: an output the buffer declares as an
    INTEGER is handed back as the raw word, so a mis-declared dtype produces numbers nobody computed
    rather than silently right answers. This is what makes the passing case evidence."""
    codes = _printed_as_signed(ff.float_to_codes(VALUES, "f32"), 32)
    got = bk.decode_float_readback({"Y0": _rows(codes, 4)}, {"Y0": wrong})
    assert got == {"Y0": _rows(codes, 4)}
    assert got != {"Y0": _rows(VALUES, 4)}


def test_the_wrong_float_dtype_decodes_to_the_wrong_values():
    """MUTATION, the other way: f32 patterns read as bf16 are not the f32 values. A decoder keyed on
    the declared dtype has to be WRONG when the declaration is wrong, or the declaration is not
    load-bearing and nothing was actually verified."""
    codes = _printed_as_signed(ff.float_to_codes(VALUES, "f32"), 32)
    got = bk.decode_float_readback({"Y0": _rows(codes, 4)}, {"Y0": "bf16"})
    assert got != {"Y0": _rows(VALUES, 4)}


def test_an_integer_output_is_returned_untouched():
    """NON-REGRESSION: every existing integer capsule reads back exactly as it did."""
    outputs = {"Y0": [[32, 20, 37], [-5, 0, 49]]}
    assert bk.decode_float_readback(outputs, {"Y0": "i32"}) == outputs
    assert bk.decode_float_readback(outputs, {"Y0": "i8"}) == outputs
    assert bk.decode_float_readback(outputs, {}) == outputs, "an undeclared output is not guessed at"


def test_a_backend_that_already_prints_decimals_is_left_alone():
    """The fp SIMT consoles parse with ``value_parser=float`` and hand back VALUES, not containers.
    Which one a backend did is visible in the values themselves, so no target has to be named."""
    outputs = {"Y0": [[-1.5, 0.25], [3.0, -0.125]]}
    assert bk.decode_float_readback(outputs, {"Y0": "f32"}) == outputs


def test_float_format_of_answers_for_integers_without_raising():
    assert bk.float_format_of("f32") == "f32"
    assert bk.float_format_of("bf16") == "bf16"
    assert bk.float_format_of("i32") is None
    assert bk.float_format_of("not_a_dtype") is None
