"""Explicit pure scalar proof declarations and bounded structural refusals."""

import hashlib
from dataclasses import replace

import pytest

from merlin.llvmlower.integer_scalar_contract import (
    IntegerScalarLimits,
    IntegerScalarNumerics,
    IntegerScalarSlot,
    OriginalIntegerScalarAbi,
    OriginalIntegerScalarSource,
)
from merlin.llvmlower.integer_scalar_correspondence import _Dags, _parse, _source

LIMITS = IntegerScalarLimits(1_000_000, 200_000, 64, 128, 100)
NUMERICS = IntegerScalarNumerics("modular_bitvector", "typed_integer_predicates", "none")


def abi(inputs=(64,), outputs=(64,)):
    return OriginalIntegerScalarAbi(
        "forward",
        "_mlir_ciface_forward",
        tuple(IntegerScalarSlot("arg" + str(i), bits) for i, bits in enumerate(inputs)),
        tuple(IntegerScalarSlot("result" + str(i), bits) for i, bits in enumerate(outputs)),
    )


def test_original_declared_empty_inputs_and_complete_repeated_result_slots(tmp_path):
    source = tmp_path / "original.mlir"
    source.write_text("module {}")
    selection = OriginalIntegerScalarSource(
        source, hashlib.sha256(source.read_bytes()).hexdigest(), abi((), (64, 64)), NUMERICS
    )
    record = selection.record(LIMITS)
    assert record["abi"]["inputs"] == []
    assert [slot["name"] for slot in record["abi"]["outputs"]] == ["result0", "result1"]


@pytest.mark.parametrize("field", ("source_bytes", "receipt_bytes", "nesting", "integer_bits", "operations"))
@pytest.mark.parametrize("value", (True, 0, -1, 1.0))
def test_exact_explicit_reader_limits(field, value):
    with pytest.raises(ValueError, match="explicit positive"):
        replace(LIMITS, **{field: value}).record()


@pytest.mark.parametrize(
    "selection",
    (
        abi(outputs=()),
        replace(abi(), inputs=[IntegerScalarSlot("arg0", 64)]),
        replace(abi(), outputs=(IntegerScalarSlot("same", 64), IntegerScalarSlot("same", 64))),
        abi(inputs=(True,)),
        abi(inputs=(129,)),
        replace(abi(), c_interface_symbol="forward"),
    ),
)
def test_malformed_or_incomplete_original_abi_refuses(selection):
    with pytest.raises(ValueError):
        selection.record(LIMITS)


@pytest.mark.parametrize(
    "selection",
    (
        replace(NUMERICS, arithmetic="bounded_exact"),
        replace(NUMERICS, overflow_promises="nsw"),
        replace(NUMERICS, comparisons="sampled"),
    ),
)
def test_numerical_contract_is_explicit_and_not_inferred(selection):
    with pytest.raises(ValueError, match="explicitly selected"):
        selection.record()


@pytest.mark.parametrize(
    "body,signature",
    (
        ("%x = arith.constant dense<0> : tensor<100000000xi64>\nfunc.return", "()"),
        ("%x = arith.constant 1 : i10000000000\nfunc.return", "()"),
    ),
)
def test_aggregate_or_huge_integer_source_refuses_before_parser(body, signature):
    with pytest.raises(ValueError):
        _parse(f"module {{ func.func @forward{signature} {{ {body} }} }}".encode(), LIMITS)


def test_complete_ordered_source_roots_preserve_repeated_result_indices():
    source = b"""module { func.func @forward(%x: i64, %y: i64) -> (i64,i64) {
      %v = arith.muli %x, %x : i64
      func.return %v, %y : i64, i64
    } }"""
    dags = _Dags()
    first = _source(_parse(source, LIMITS), abi((64, 64), (64, 64)), LIMITS, dags)
    swapped = _source(_parse(source.replace(b"%v, %y :", b"%y, %v :"), LIMITS), abi((64, 64), (64, 64)), LIMITS, dags)
    repeated = _source(_parse(source.replace(b"%v, %y :", b"%v, %v :"), LIMITS), abi((64, 64), (64, 64)), LIMITS, dags)
    assert first != swapped and first != repeated and repeated[0] == repeated[1]


def test_integer_constant_identity_keeps_bits_above_binary64_precision():
    def read(value, dags):
        source = (
            f"module {{ func.func @forward() -> i64 {{ %v = arith.constant {value} : i64 func.return %v : i64 }} }}"
        )
        return _source(_parse(source.encode(), LIMITS), abi((), (64,)), LIMITS, dags)

    dags = _Dags()
    assert read((1 << 53) + 3, dags) != read((1 << 53) + 4, dags)
