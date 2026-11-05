"""Neutral source-only tests for closed static floating-point math regions."""

from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256

import pytest

from merlin.common import mlir_query as mq
from merlin.frontends.linalg_composite_math import (
    SCHEMA,
    SOURCE_ONLY_SCOPE,
    STATIC_COMPOSITE_MATH_SOURCE_BODY_SCHEMA,
    recognize_static_composite_math_body,
    screen_static_composite_math_source,
    static_composite_math_ordered_types,
    validate_serialized_static_composite_math_pattern,
    validate_static_composite_math_source_body,
)
from merlin.frontends.linalg_patterns import InvalidLinalgPattern
from merlin.targetgen.host_capabilities import admit_host_operation, validate_host_capabilities
from merlin.targetgen.host_linkage_contract import SCHEMA as LINKAGE_SCHEMA
from merlin.targetgen.host_linkage_contract import SUPPLIER


def _body(kind: str) -> str:
    literal = '%bound = "arith.constant"() <{value = 2.250000e+00 : f32}> : () -> f32'
    if kind in ("clamp_upper_literal", "clamp_lower_literal"):
        scalar = "arith.minimumf" if kind == "clamp_upper_literal" else "arith.maximumf"
        return (
            f'{literal}\n        %value = "{scalar}"(%input, %bound) '
            "<{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32"
        )
    if kind == "reciprocal_f32":
        return (
            '%one = "arith.constant"() <{value = 1.000000e+00 : f32}> : () -> f32\n'
            '        %value = "arith.divf"(%one, %input) '
            "<{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32"
        )
    if kind == "reciprocal_signed_i64_to_f32":
        return (
            '%cast = "arith.sitofp"(%input) : (i64) -> f32\n'
            '        %one = "arith.constant"() <{value = 1.000000e+00 : f32}> : () -> f32\n'
            '        %value = "arith.divf"(%one, %cast) '
            "<{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32"
        )
    if kind == "literal_base_pow_f32":
        return (
            '%base = "arith.constant"() <{value = 3.250000e+00 : f32}> : () -> f32\n'
            '        %value = "math.powf"(%base, %input) '
            "<{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32"
        )
    if kind == "sigmoid_f32":
        return "\n        ".join((
            '%one = "arith.constant"() <{value = 1.000000e+00 : f32}> : () -> f32',
            '%negative = "arith.negf"(%input) : (f32) -> f32',
            '%exponential = "math.exp"(%negative) : (f32) -> f32',
            '%denominator = "arith.addf"(%one, %exponential) : (f32, f32) -> f32',
            '%value = "arith.divf"(%one, %denominator) : (f32, f32) -> f32',
        ))
    if kind == "tanh_gelu_f32":
        # Public tanh approximation coefficients; no captured model source.
        return "\n        ".join(
            (
                '%half = "arith.constant"() <{value = 5.000000e-01 : f32}> : () -> f32',
                '%one = "arith.constant"() <{value = 1.000000e+00 : f32}> : () -> f32',
                '%cubic = "arith.constant"() <{value = 4.471500e-02 : f32}> : () -> f32',
                '%scale = "arith.constant"() <{value = 7.9788456e-01 : f32}> : () -> f32',
                '%square = "arith.mulf"(%input, %input) : (f32, f32) -> f32',
                '%cube = "arith.mulf"(%square, %input) : (f32, f32) -> f32',
                '%scaled = "arith.mulf"(%cubic, %cube) : (f32, f32) -> f32',
                '%plus = "arith.addf"(%input, %scaled) : (f32, f32) -> f32',
                '%inner = "arith.mulf"(%scale, %plus) : (f32, f32) -> f32',
                '%tanh = "math.tanh"(%inner) : (f32) -> f32',
                '%shifted = "arith.addf"(%one, %tanh) : (f32, f32) -> f32',
                '%half_input = "arith.mulf"(%half, %input) : (f32, f32) -> f32',
                '%value = "arith.mulf"(%half_input, %shifted) : (f32, f32) -> f32',
            )
        )
    raise AssertionError(kind)


def _program(kind: str, *, shape: str = "2x3", body: str | None = None, map_result: str | None = None) -> str:
    input_type = "i64" if kind == "reciprocal_signed_i64_to_f32" else "f32"
    dims = ", ".join(f"d{i}" for i in range(len(shape.split("x"))))
    identity = f"({dims})"
    first_map = map_result or identity
    maps = ", ".join(f"affine_map<({dims}) -> {result}>" for result in (first_map, identity))
    iterators = ", ".join("#linalg.iterator_type<parallel>" for _ in shape.split("x"))
    return f"""builtin.module {{
  func.func @forward(%x: tensor<{shape}x{input_type}>,
                     %init: tensor<{shape}xf32>) -> tensor<{shape}xf32> {{
    %out = "linalg.generic"(%x, %init) <{{indexing_maps = [{maps}],
      iterator_types = [{iterators}], operandSegmentSizes = array<i32: 1, 1>}}> ({{
      ^bb0(%input: {input_type}, %acc: f32):
        {body or _body(kind)}
        "linalg.yield"(%value) : (f32) -> ()
    }}) : (tensor<{shape}x{input_type}>, tensor<{shape}xf32>) -> tensor<{shape}xf32>
    func.return %out : tensor<{shape}xf32>
  }}
}}"""


def _generic(text: str):
    return next(mq.walk(mq.parse(text), "linalg.generic"))


@pytest.mark.parametrize(
    "kind",
    (
        "clamp_upper_literal",
        "clamp_lower_literal",
        "reciprocal_f32",
        "reciprocal_signed_i64_to_f32",
        "literal_base_pow_f32",
        "sigmoid_f32",
        "tanh_gelu_f32",
    ),
)
@pytest.mark.parametrize("shape", ("7", "2x3"))
def test_closed_source_forms_are_typed_literal_bit_preserving_and_not_admission(kind, shape):
    pattern = recognize_static_composite_math_body(_generic(_program(kind, shape=shape)))
    declaration = {"schema": STATIC_COMPOSITE_MATH_SOURCE_BODY_SCHEMA, "operation": kind}
    assert pattern.schema == SCHEMA
    assert pattern.operation == kind
    assert pattern.shape == tuple(int(dim) for dim in shape.split("x"))
    assert pattern.ordered_types == (("i64", "f32", "f32") if kind == "reciprocal_signed_i64_to_f32" else ("f32",) * 3)
    assert validate_serialized_static_composite_math_pattern(asdict(pattern)) == pattern
    assert validate_static_composite_math_source_body(declaration) == declaration
    assert static_composite_math_ordered_types(declaration) == pattern.ordered_types
    assert "numerical equivalence unproved" in SOURCE_ONLY_SCOPE
    if kind == "literal_base_pow_f32":
        assert pattern.lowering_intrinsic_obligations == ("math.powf",)
    elif kind == "sigmoid_f32":
        assert pattern.lowering_intrinsic_obligations == ("math.exp",)
    elif kind == "tanh_gelu_f32":
        assert pattern.lowering_intrinsic_obligations == ("math.tanh",)
    else:
        assert pattern.lowering_intrinsic_obligations == ()


@pytest.mark.parametrize(
    ("kind", "old", "new"),
    (
        ("clamp_upper_literal", "(%input, %bound)", "(%bound, %input)"),
        ("clamp_lower_literal", "arith.maximumf", "arith.minimumf"),
        ("reciprocal_f32", "(%one, %input)", "(%input, %one)"),
        ("reciprocal_signed_i64_to_f32", "arith.sitofp", "arith.uitofp"),
        ("reciprocal_f32", "1.000000e+00", "2.000000e+00"),
        ("literal_base_pow_f32", "(%base, %input)", "(%input, %base)"),
        ("sigmoid_f32", "1.000000e+00", "2.000000e+00"),
        ("sigmoid_f32", "math.exp", "math.erf"),
        ("sigmoid_f32", "(%one, %exponential)", "(%exponential, %one)"),
        ("sigmoid_f32", "(%one, %denominator)", "(%denominator, %one)"),
        ("tanh_gelu_f32", "4.471500e-02", "4.471600e-02"),
        ("tanh_gelu_f32", "(%scale, %plus)", "(%plus, %scale)"),
        ("tanh_gelu_f32", "math.tanh", "math.erf"),
        ("tanh_gelu_f32", "(%half_input, %shifted)", "(%shifted, %half_input)"),
    ),
)
def test_changed_source_form_does_not_prove_the_original(kind, old, new):
    original = recognize_static_composite_math_body(_generic(_program(kind)))
    changed = _program(kind).replace(old, new)
    try:
        observed = recognize_static_composite_math_body(_generic(changed))
    except InvalidLinalgPattern:
        return
    assert observed != original  # Lower clamp is a distinct closed operation.


@pytest.mark.parametrize(
    ("kind", "old", "new"),
    (
        ("clamp_upper_literal", "<none>", "<reassoc>"),
        ("literal_base_pow_f32", "<none>", "<reassoc>"),
        ("reciprocal_f32", "(%one, %input)", "(%one, %acc)"),
        ("tanh_gelu_f32", '%value = "arith.mulf"', '%value = "arith.addf"'),
    ),
)
def test_flags_init_reads_and_changed_body_refuse(kind, old, new):
    with pytest.raises(InvalidLinalgPattern):
        recognize_static_composite_math_body(_generic(_program(kind).replace(old, new)))


def test_nonidentity_map_and_dynamic_shape_refuse():
    with pytest.raises(InvalidLinalgPattern):
        recognize_static_composite_math_body(_generic(_program("reciprocal_f32", map_result="(d1, d0)")))
    with pytest.raises(InvalidLinalgPattern):
        recognize_static_composite_math_body(_generic(_program("reciprocal_f32", shape="?x3")))


def test_literal_source_bits_preserve_signed_zero_and_refuse_extra_metadata():
    text = _program("clamp_upper_literal").replace("2.250000e+00", "0.000000e+00")
    positive = recognize_static_composite_math_body(_generic(text))
    negative = recognize_static_composite_math_body(_generic(text.replace("0.000000e+00", "-0.000000e+00")))
    assert positive.literal_bits == ("0x00000000",)
    assert negative.literal_bits == ("0x80000000",)
    unexpected = text.replace(
        "<{fastmath = #arith.fastmath<none>}>",
        "<{fastmath = #arith.fastmath<none>}> {unknown = true}",
    )
    with pytest.raises(InvalidLinalgPattern):
        recognize_static_composite_math_body(_generic(unexpected))


def test_serialized_literal_roster_is_rechecked():
    original = asdict(recognize_static_composite_math_body(_generic(_program("tanh_gelu_f32"))))
    for changed in (
        {**original, "schema": "another"},
        {**original, "shape": [2.0, 3]},
        {**original, "literal_bits": [*original["literal_bits"][:3], "0x3f4c422b"]},
        {**original, "literal_bits": [*original["literal_bits"][:3], "0x7fc00000"]},
        {**original, "lowering_intrinsic_obligations": ["tanhf"]},
        {**original, "unexpected": True},
    ):
        with pytest.raises(InvalidLinalgPattern):
            validate_serialized_static_composite_math_pattern(changed)


def test_declaration_requires_only_known_schema_and_operation():
    correct = {"schema": STATIC_COMPOSITE_MATH_SOURCE_BODY_SCHEMA, "operation": "clamp_upper_literal"}
    for changed in (
        {**correct, "unexpected": True},
        {**correct, "operation": "math.powf"},
        {**correct, "operation": 3},
        {**correct, "schema": SCHEMA},
        {"operation": correct["operation"]},
    ):
        with pytest.raises(InvalidLinalgPattern):
            validate_static_composite_math_source_body(changed)


def test_source_screen_pins_raw_and_normalized_bytes_and_ordinals(tmp_path):
    path = tmp_path / "neutral.mlir"
    path.write_text(_program("literal_base_pow_f32"))
    ordinal = next(i for i, op in enumerate(mq.walk(mq.parse(path.read_text()))) if op.name == "linalg.generic")
    record = screen_static_composite_math_source(path, (ordinal,))
    assert record.raw_sha256 == sha256(path.read_bytes()).hexdigest()
    assert record.normalized_sha256
    assert record.ordinals[0][1].operation == "literal_base_pow_f32"
    path.write_text(_program("literal_base_pow_f32").replace("3.250000e+00", "4.250000e+00"))
    assert screen_static_composite_math_source(path, (ordinal,)).raw_sha256 != record.raw_sha256
    for bad in ((True,), (-1,), (ordinal, ordinal), (10000,)):
        with pytest.raises(InvalidLinalgPattern):
            screen_static_composite_math_source(path, bad)


def _synthetic_host(kind: str, *, symbol: str | None = None):
    proof = recognize_static_composite_math_body(_generic(_program(kind)))
    declaration = {
        "id": "neutral_math",
        "ops": ["linalg.generic"],
        "placement": "host",
        "signature": {
            "ordered_operand_dtypes": list(proof.ordered_types[:-1]),
            "ordered_result_dtypes": [proof.ordered_types[-1]],
            "ranks": [2],
        },
        "source_body": {"schema": STATIC_COMPOSITE_MATH_SOURCE_BODY_SCHEMA, "operation": kind},
        "numerical_contract": {"status": "unreviewed"},
    }
    if symbol is not None:
        declaration["linkage_contract"] = {
            "schema": LINKAGE_SCHEMA,
            "supplier": SUPPLIER,
            "archive_sha256": "c" * 64,
            "symbols": [symbol],
        }
    selected = {
        "host": {
            "package_sha256": "a" * 64,
            "capability_spec_sha256": "b" * 64,
            "dtype_strategy": "int8_w8a8",
            "capability_spec": {
                "schema": "merlin.host_capabilities.v1",
                "status": "reviewed",
                "compiler": {"package_sha256": "a" * 64, "dtype_strategy": "int8_w8a8"},
                "operations": [declaration],
                "evidence": {"scope": "neutral placement declaration only"},
            },
        }
    }
    return selected, proof


@pytest.mark.parametrize(
    ("kind", "symbol"),
    (
        ("clamp_upper_literal", None),
        ("reciprocal_signed_i64_to_f32", None),
        ("literal_base_pow_f32", "powf"),
        ("tanh_gelu_f32", "tanhf"),
    ),
)
def test_host_declaration_requires_every_exact_parsed_body_and_only_pending_linkage(kind, symbol):
    selected, proof = _synthetic_host(kind, symbol=symbol)
    validate_host_capabilities(selected["host"]["capability_spec"])
    row = {"mlir_operation": "linalg.generic", "count": 2}
    signature = {
        "family": "elementwise_map",
        "ordered_operand_dtypes": list(proof.ordered_types[:-1]),
        "ordered_result_dtypes": [proof.ordered_types[-1]],
        "rank": 2,
    }
    good = (_generic(_program(kind)), _generic(_program(kind)))
    assert admit_host_operation(selected, row, signature)["status"] == "unknown"
    admitted = admit_host_operation(selected, row, signature, source_operations=good)
    assert admitted["status"] == "admitted"  # Synthetic reviewed placement only.
    assert admitted["source_body_proof"]["patterns"][0]["literal_bits"] == proof.literal_bits
    assert admitted["source_body_proof"]["patterns"][0]["lowering_intrinsic_obligations"] == (
        proof.lowering_intrinsic_obligations
    )
    assert (admitted.get("linkage_requirement") is not None) == (symbol is not None)
    assert "numerical" not in admitted["qualification"]
    changed = (
        _program(kind).replace("2.250000e+00", "2.750000e+00")
        if symbol is None
        else _program(kind).replace("3.250000e+00", "4.250000e+00")
    )
    if changed != _program(kind):
        differing_literals = admit_host_operation(
            selected, row, signature, source_operations=(good[0], _generic(changed))
        )
        assert differing_literals["status"] == "admitted"
        assert (
            differing_literals["source_body_proof"]["patterns"][0]["literal_bits"]
            != differing_literals["source_body_proof"]["patterns"][1]["literal_bits"]
        )
    assert admit_host_operation(selected, row, signature, source_operations=good[:1])["status"] == "unsupported"


@pytest.mark.parametrize(("kind", "symbol"), (("literal_base_pow_f32", "powf"), ("tanh_gelu_f32", "tanhf")))
def test_selected_math_symbol_contract_is_mandatory_and_exact(kind, symbol):
    selected, _ = _synthetic_host(kind, symbol=symbol)
    row = selected["host"]["capability_spec"]["operations"][0]
    for contract in (None, {**row["linkage_contract"], "symbols": ["other"]}):
        if contract is None:
            row.pop("linkage_contract")
        else:
            row["linkage_contract"] = contract
        with pytest.raises(ValueError, match="linkage"):
            validate_host_capabilities(selected["host"]["capability_spec"])
    clamp, _ = _synthetic_host("clamp_upper_literal", symbol="powf")
    with pytest.raises(ValueError, match="linkage"):
        validate_host_capabilities(clamp["host"]["capability_spec"])
