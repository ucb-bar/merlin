"""Neutral source-only tests for closed static floating-point math regions."""

# ruff: noqa: E501 - emitted MLIR lines retain the producer's operation form.

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


def test_ordered_compare_select_clamp_is_distinct_and_source_exact():
    body = """%bound = "arith.constant"() <{value = 2.250000e+00 : f32}> : () -> f32
        %condition = "arith.cmpf"(%input, %bound) <{predicate = 2 : i64, fastmath = #arith.fastmath<none>}> : (f32, f32) -> i1
        %value = "arith.select"(%condition, %bound, %input) : (i1, f32, f32) -> f32"""
    text = _program("clamp_upper_literal", body=body)
    pattern = recognize_static_composite_math_body(_generic(text))
    assert pattern.operation == "clamp_upper_literal_compare_select"
    validate_serialized_static_composite_math_pattern(asdict(pattern))
    for changed in (
        text.replace("predicate = 2", "predicate = 4"),
        text.replace("(%condition, %bound, %input)", "(%condition, %input, %bound)"),
        text.replace("(%input, %bound)", "(%acc, %bound)"),
    ):
        with pytest.raises(InvalidLinalgPattern):
            recognize_static_composite_math_body(_generic(changed))


_MIN_NORMAL_BODY = """%bound = "arith.constant"() <{value = 1.17549435e-38 : f32}> : () -> f32
        %condition = "arith.cmpf"(%input, %bound) <{predicate = 4 : i64, fastmath = #arith.fastmath<none>}> : (f32, f32) -> i1
        %value = "arith.select"(%condition, %bound, %input) : (i1, f32, f32) -> f32"""
_INT8_GRID_BODY = """%low = "arith.constant"() <{value = -127 : i64}> : () -> i64
        %lowf = "arith.sitofp"(%low) : (i64) -> f32
        %below = "arith.cmpf"(%input, %lowf) <{predicate = 4 : i64, fastmath = #arith.fastmath<none>}> : (f32, f32) -> i1
        %raised = "arith.select"(%below, %lowf, %input) : (i1, f32, f32) -> f32
        %high = "arith.constant"() <{value = 127 : i64}> : () -> i64
        %highf = "arith.sitofp"(%high) : (i64) -> f32
        %above = "arith.cmpf"(%raised, %highf) <{predicate = 2 : i64, fastmath = #arith.fastmath<none>}> : (f32, f32) -> i1
        %value = "arith.select"(%above, %highf, %raised) : (i1, f32, f32) -> f32"""
_QUANTIZATION_CLAMPS = {
    "clamp_lower_f32_min_normal_compare_select": (_MIN_NORMAL_BODY, ("0x00800000",)),
    "clamp_symmetric_int8_grid_compare_select": (_INT8_GRID_BODY, ("0xc2fe0000", "0x42fe0000")),
}


@pytest.mark.parametrize("kind", tuple(_QUANTIZATION_CLAMPS))
@pytest.mark.parametrize("shape", ("7", "1x32x64x160"))
def test_exact_quantization_clamps_are_recognized_with_their_exact_bounds(kind, shape):
    body, bits = _QUANTIZATION_CLAMPS[kind]
    pattern = recognize_static_composite_math_body(_generic(_program("clamp_upper_literal", shape=shape, body=body)))
    assert pattern.operation == kind
    assert pattern.ordered_types == ("f32", "f32", "f32")
    assert pattern.literal_bits == bits
    assert pattern.lowering_intrinsic_obligations == ()
    assert validate_serialized_static_composite_math_pattern(asdict(pattern)) == pattern
    declaration = {"schema": STATIC_COMPOSITE_MATH_SOURCE_BODY_SCHEMA, "operation": kind}
    assert validate_static_composite_math_source_body(declaration) == declaration
    for changed in (("0x00800001",), ("0xc3000000", "0x42fe0000"), ("0xc2fe0000", "0x43000000")):
        if len(changed) == len(bits):
            with pytest.raises(InvalidLinalgPattern, match="exact literals"):
                validate_serialized_static_composite_math_pattern({**asdict(pattern), "literal_bits": list(changed)})


@pytest.mark.parametrize(
    ("kind", "old", "new"),
    (
        # Any other lower literal: the next f32 above FLT_MIN, twice FLT_MIN, or zero.
        ("clamp_lower_f32_min_normal_compare_select", "1.17549435e-38", "1.17549449e-38"),
        ("clamp_lower_f32_min_normal_compare_select", "1.17549435e-38", "2.3509887e-38"),
        ("clamp_lower_f32_min_normal_compare_select", "1.17549435e-38", "0.000000e+00"),
        ("clamp_lower_f32_min_normal_compare_select", "predicate = 4", "predicate = 5"),
        ("clamp_lower_f32_min_normal_compare_select", "predicate = 4", "predicate = 12"),
        ("clamp_lower_f32_min_normal_compare_select", "(%condition, %bound, %input)", "(%condition, %input, %bound)"),
        ("clamp_lower_f32_min_normal_compare_select", "(%input, %bound)", "(%bound, %input)"),
        ("clamp_lower_f32_min_normal_compare_select", "#arith.fastmath<none>", "#arith.fastmath<nnan>"),
        ("clamp_symmetric_int8_grid_compare_select", "value = -127 : i64", "value = -128 : i64"),
        ("clamp_symmetric_int8_grid_compare_select", "value = 127 : i64", "value = 126 : i64"),
        ("clamp_symmetric_int8_grid_compare_select", "predicate = 4", "predicate = 5"),
        ("clamp_symmetric_int8_grid_compare_select", "predicate = 2", "predicate = 3"),
        ("clamp_symmetric_int8_grid_compare_select", '"arith.sitofp"(%low)', '"arith.uitofp"(%low)'),
        ("clamp_symmetric_int8_grid_compare_select", "(%below, %lowf, %input)", "(%below, %input, %lowf)"),
        ("clamp_symmetric_int8_grid_compare_select", "(%above, %highf, %raised)", "(%above, %highf, %input)"),
        ("clamp_symmetric_int8_grid_compare_select", '"arith.cmpf"(%raised, %highf)', '"arith.cmpf"(%input, %highf)'),
        ("clamp_symmetric_int8_grid_compare_select", "#arith.fastmath<none>", "#arith.fastmath<fast>"),
    ),
)
def test_quantization_clamps_refuse_any_other_literal_predicate_or_dataflow(kind, old, new):
    body, _ = _QUANTIZATION_CLAMPS[kind]
    assert old in body
    with pytest.raises(InvalidLinalgPattern):
        recognize_static_composite_math_body(_generic(_program("clamp_upper_literal", body=body.replace(old, new, 1))))


def test_int8_grid_clamp_refuses_swapped_bound_order_and_other_literal_types():
    upper_first = (
        _INT8_GRID_BODY.replace("value = -127", "value = UPPER")
        .replace("value = 127", "value = -127")
        .replace("value = UPPER", "value = 127")
    )
    narrow = _INT8_GRID_BODY.replace(": i64}> : () -> i64", ": i32}> : () -> i32").replace(
        "(i64) -> f32", "(i32) -> f32"
    )
    for body in (upper_first, narrow):
        with pytest.raises(InvalidLinalgPattern):
            recognize_static_composite_math_body(_generic(_program("clamp_upper_literal", body=body)))


@pytest.mark.parametrize("kind", tuple(_QUANTIZATION_CLAMPS))
def test_quantization_clamp_declaration_admits_only_its_exact_body(kind):
    body, _ = _QUANTIZATION_CLAMPS[kind]
    selected, _ = _synthetic_host("clamp_upper_literal")
    declaration = selected["host"]["capability_spec"]["operations"][0]
    declaration["source_body"]["operation"] = kind
    validate_host_capabilities(selected["host"]["capability_spec"])
    row = {"mlir_operation": "linalg.generic", "count": 1}
    signature = {
        "family": "elementwise_map",
        "ordered_operand_dtypes": ["f32", "f32"],
        "ordered_result_dtypes": ["f32"],
        "rank": 2,
    }
    exact = (_generic(_program("clamp_upper_literal", body=body)),)
    assert admit_host_operation(selected, row, signature, source_operations=exact)["status"] == "admitted"
    other = next(name for name in _QUANTIZATION_CLAMPS if name != kind)
    for source in (
        _program("clamp_upper_literal", body=_QUANTIZATION_CLAMPS[other][0]),
        _program("clamp_upper_literal"),
    ):
        refused = admit_host_operation(selected, row, signature, source_operations=(_generic(source),))
        assert refused["status"] == "unsupported"


_IDENTITY_2X3 = (
    "indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], "
    "iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], "
    "operandSegmentSizes = array<i32: 1, 1>"
)


def _quantize_chain(*, round_body=None, clamp_body=None, cast_input="%clamped", extra="") -> str:
    """x -> roundeven -> int8-grid clamp -> fptosi i8, each the sole result of its own generic."""
    round_body = round_body or (
        '%r = "math.roundeven"(%a) <{fastmath = #arith.fastmath<none>}> : (f32) -> f32\n'
        '        "linalg.yield"(%r) : (f32) -> ()'
    )
    clamp_body = clamp_body or (
        _INT8_GRID_BODY.replace("%input", "%input") + '\n        "linalg.yield"(%value) : (f32) -> ()'
    )
    return f"""builtin.module {{
  func.func @forward(%x: tensor<2x3xf32>) -> tensor<2x3xi8> {{
    %e0 = "tensor.empty"() : () -> tensor<2x3xf32>
    %rounded = "linalg.generic"(%x, %e0) <{{{_IDENTITY_2X3}}}> ({{
      ^bb0(%a: f32, %b: f32):
        {round_body}
    }}) : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
    %e1 = "tensor.empty"() : () -> tensor<2x3xf32>
    %clamped = "linalg.generic"(%rounded, %e1) <{{{_IDENTITY_2X3}}}> ({{
      ^bb0(%input: f32, %acc: f32):
        {clamp_body}
    }}) : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
{extra}    %e2 = "tensor.empty"() : () -> tensor<2x3xi8>
    %q = "linalg.generic"({cast_input}, %e2) <{{{_IDENTITY_2X3}}}> ({{
      ^bb0(%v: f32, %w: i8):
        %c = "arith.fptosi"(%v) : (f32) -> i8
        "linalg.yield"(%c) : (i8) -> ()
    }}) : (tensor<2x3xf32>, tensor<2x3xi8>) -> tensor<2x3xi8>
    func.return %q : tensor<2x3xi8>
  }}
}}"""


def _cast(text: str):
    return [op for op in mq.walk(mq.parse(text), "linalg.generic")][-1]


def test_int8_cast_is_proved_only_behind_roundeven_and_the_int8_grid_clamp():
    pattern = recognize_static_composite_math_body(_cast(_quantize_chain()))
    assert pattern.operation == "fptosi_i8_of_rounded_int8_grid_clamp"
    assert pattern.ordered_types == ("f32", "i8", "i8")
    assert pattern.literal_bits == ("0xc2fe0000", "0x42fe0000")
    assert validate_serialized_static_composite_math_pattern(asdict(pattern)) == pattern
    with pytest.raises(InvalidLinalgPattern, match="exact literals"):
        validate_serialized_static_composite_math_pattern(
            {**asdict(pattern), "literal_bits": ["0xc3000000", "0x42fe0000"]}
        )
    declaration = {"schema": STATIC_COMPOSITE_MATH_SOURCE_BODY_SCHEMA, "operation": pattern.operation}
    assert static_composite_math_ordered_types(declaration) == ("f32", "i8", "i8")


@pytest.mark.parametrize(
    "text",
    (
        # The operand comes straight from the input or from roundeven, skipping the clamp.
        _quantize_chain(cast_input="%x"),
        _quantize_chain(cast_input="%rounded"),
        # A wider or one-sided clamp does not bound the conversion to the int8 grid.
        _quantize_chain(
            clamp_body=_INT8_GRID_BODY.replace("value = -127", "value = -129")
            + '\n        "linalg.yield"(%value) : (f32) -> ()'
        ),
        _quantize_chain(clamp_body=_MIN_NORMAL_BODY + '\n        "linalg.yield"(%value) : (f32) -> ()'),
        # The clamp input is not integral: no roundeven, a different rounding, or fast-math.
        _quantize_chain(round_body='"linalg.yield"(%a) : (f32) -> ()'),
        _quantize_chain(
            round_body='%r = "math.round"(%a) <{fastmath = #arith.fastmath<none>}> : (f32) -> f32\n        "linalg.yield"(%r) : (f32) -> ()'
        ),
        _quantize_chain(
            round_body='%r = "math.floor"(%a) <{fastmath = #arith.fastmath<none>}> : (f32) -> f32\n        "linalg.yield"(%r) : (f32) -> ()'
        ),
        _quantize_chain(
            round_body='%r = "math.roundeven"(%a) <{fastmath = #arith.fastmath<nnan>}> : (f32) -> f32\n        "linalg.yield"(%r) : (f32) -> ()'
        ),
        _quantize_chain(
            round_body='%r = "math.roundeven"(%b) <{fastmath = #arith.fastmath<none>}> : (f32) -> f32\n        "linalg.yield"(%r) : (f32) -> ()'
        ),
    ),
)
def test_int8_cast_without_the_proved_producer_domain_is_refused(text):
    with pytest.raises(InvalidLinalgPattern):
        recognize_static_composite_math_body(_cast(text))


def test_int8_cast_refuses_another_conversion_or_result_type():
    for old, new in (
        ('"arith.fptosi"(%v) : (f32) -> i8', '"arith.fptoui"(%v) : (f32) -> i8'),
        ("tensor<2x3xi8>", "tensor<2x3xi16>"),
    ):
        text = (
            _quantize_chain()
            .replace(old, new)
            .replace("(f32) -> i8", "(f32) -> i16" if "i16" in new else "(f32) -> i8")
        )
        text = text.replace("%w: i8", "%w: i16" if "i16" in new else "%w: i8").replace(
            "(i8) -> ()", "(i16) -> ()" if "i16" in new else "(i8) -> ()"
        )
        with pytest.raises(InvalidLinalgPattern):
            recognize_static_composite_math_body(_cast(text))


def test_int8_cast_declaration_admits_only_the_proved_chain():
    selected, _ = _synthetic_host("clamp_upper_literal")
    declaration = selected["host"]["capability_spec"]["operations"][0]
    declaration["source_body"]["operation"] = "fptosi_i8_of_rounded_int8_grid_clamp"
    declaration["signature"] = {"ordered_operand_dtypes": ["f32", "i8"], "ordered_result_dtypes": ["i8"], "ranks": [2]}
    validate_host_capabilities(selected["host"]["capability_spec"])
    row = {"mlir_operation": "linalg.generic", "count": 1}
    signature = {
        "family": "elementwise_map",
        "ordered_operand_dtypes": ["f32", "i8"],
        "ordered_result_dtypes": ["i8"],
        "rank": 2,
    }
    proved = admit_host_operation(selected, row, signature, source_operations=(_cast(_quantize_chain()),))
    assert proved["status"] == "admitted"
    unproved = admit_host_operation(
        selected, row, signature, source_operations=(_cast(_quantize_chain(cast_input="%rounded")),)
    )
    assert unproved["status"] == "unsupported"
