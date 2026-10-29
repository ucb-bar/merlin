"""Bounded IEEE evidence for the already supported unary/FMA float interpretation."""

import json
import math
import struct

import pytest
from test_lean_integer_fold import canonicalize

from merlin.verify.lean import differential as diff
from merlin.verify.lean.check import check_definedness, check_typed_stage

NAMES = [
    "arith.negf",
    "math.absf",
    "math.sqrt",
    "math.ceil",
    "math.floor",
    "math.round",
    "math.roundeven",
    "math.trunc",
]


def _bits(width, value):
    return int.from_bytes(struct.pack("<f" if width == 32 else "<d", value), "little")


def _is_nan(width, bits):
    fraction = 23 if width == 32 else 52
    exponent = 8 if width == 32 else 11
    return (
        bits & (((1 << exponent) - 1) << fraction) == (((1 << exponent) - 1) << fraction)
        and bits & ((1 << fraction) - 1) != 0
    )


def _reference(width, value):
    def rounded(fn):
        if not math.isfinite(value):
            return value
        result = float(fn(value))
        return math.copysign(0.0, value) if result == 0 else result

    root = math.sqrt(value) if value >= 0 else math.nan
    results = [
        -value,
        abs(value),
        root,
        rounded(math.ceil),
        rounded(math.floor),
        rounded(lambda x: math.copysign(math.floor(abs(x) + 0.5), x)),
        rounded(round),
        rounded(math.trunc),
        value * 2.0 + 0.5,
    ]
    return [_bits(width, x) for x in results]


def _source(width):
    kind = f"f{width}"
    body = [f"%v{n} = {name} %x : {kind}" for n, name in enumerate(NAMES)]
    body.append(f"%v8 = math.fma %x, %y, %z : {kind}")
    return f"""func.func @forward(%x: {kind}, %y: {kind}, %z: {kind}) -> ({", ".join([kind] * 9)}) {{
      {" ".join(body)}
      return {", ".join(f"%v{n}" for n in range(9))} : {", ".join([kind] * 9)}
    }}"""


@pytest.mark.parametrize("width", [32, 64])
def test_defined_floats_match_native_and_independent_references(tmp_path, fidelity_tools, width):
    program = tmp_path / "source.mlir"
    program.write_text(_source(width))
    receipt = check_definedness(program, "forward", tmp_path / "defined", m2m_python=fidelity_tools.compiler_python)
    assert receipt["verdict"] == "verified", receipt.get("reason")
    values = [0.0, -0.0, 0.25, -0.25, 0.5, -0.5, 1.5, -1.5, 2.5, -2.5, 4.0, -4.0, math.inf, -math.inf, math.nan]
    samples = [[[_bits(width, x)], [_bits(width, 2.0)], [_bits(width, 0.5)]] for x in values]
    work = tmp_path / "comparison"
    diff.compare(program.read_text(), samples, work, fidelity_tools, function="forward", arithmetic_nans=True)
    actual = json.loads((work / "native-results.json").read_text())["results"]
    for value, result in zip(values, actual, strict=True):
        expected = _reference(width, value)
        assert len(result) == len(expected)
        for observed, reference in zip(result, expected, strict=True):
            if _is_nan(width, reference):
                assert _is_nan(width, int(observed))
            else:
                assert int(observed) == reference


def test_changed_float_operation_is_rejected_and_observably_wrong(tmp_path, lean_tools, fidelity_tools):
    stage, before, after = canonicalize(tmp_path, lean_tools, _source(32))
    assert check_typed_stage(stage, m2m_python=lean_tools.m2m_python)["verdict"] == "verified"
    text = after.read_text()
    assert "arith.negf" in text
    after.write_text(text.replace("arith.negf", "math.absf", 1))
    assert check_typed_stage(stage, m2m_python=lean_tools.m2m_python)["verdict"] == "abstracted"
    samples = [[[_bits(32, 0.25)], [_bits(32, 2.0)], [_bits(32, 0.5)]]]
    results = []
    for name, program in [("source", before.read_text()), ("wrong", after.read_text())]:
        work = tmp_path / name
        diff.compare(program, samples, work, fidelity_tools, function="forward")
        results.append(json.loads((work / "native-results.json").read_text())["results"])
    assert results[0] != results[1]
