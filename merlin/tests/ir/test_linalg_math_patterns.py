"""Neutral source-only checks for closed unary floating-point math regions."""

from __future__ import annotations

from hashlib import sha256

import pytest

from merlin.common import mlir_query as mq
from merlin.frontends.linalg_math_patterns import (
    SOURCE_ONLY_SCOPE,
    STATIC_F32_MATH_SOURCE_BODY_SCHEMA,
    recognize_static_f32_exp_rsqrt_body,
    recognize_static_f32_math_body,
    screen_static_f32_math_source,
    static_f32_math_ordered_types,
    validate_static_f32_math_source_body,
)
from merlin.frontends.linalg_patterns import InvalidLinalgPattern
from merlin.targetgen.host_capabilities import admit_host_operation, validate_host_capabilities


def _program(
    *,
    scalar: str = '"math.sin"(%input) <{fastmath = #arith.fastmath<none>}> : (f32) -> f32',
    shape: str = "2x3",
    input_dtype: str = "f32",
    scalar_body: str | None = None,
    yielded: str = "%value",
    input_map: str | None = None,
    iterators: str | None = None,
    generic_attributes: str = "",
) -> str:
    dimensions = ", ".join(f"d{i}" for i in range(len(shape.split("x"))))
    identity = f"({dimensions})"
    maps = ", ".join(f"affine_map<({dimensions}) -> {mapping}>" for mapping in (input_map or identity, identity))
    iterator_list = iterators or ", ".join("#linalg.iterator_type<parallel>" for _ in shape.split("x"))
    body = scalar_body or f"%value = {scalar}"
    return f"""builtin.module {{
  func.func @forward(%x: tensor<{shape}x{input_dtype}>,
                     %init: tensor<{shape}xf32>) -> tensor<{shape}xf32> {{
    %out = "linalg.generic"(%x, %init) <{{indexing_maps = [{maps}],
      iterator_types = [{iterator_list}], operandSegmentSizes = array<i32: 1, 1>}}> ({{
      ^bb0(%input: {input_dtype}, %acc: f32):
        {body}
        "linalg.yield"({yielded}) : (f32) -> ()
    }}) {generic_attributes} : (tensor<{shape}x{input_dtype}>, tensor<{shape}xf32>) -> tensor<{shape}xf32>
    func.return %out : tensor<{shape}xf32>
  }}
}}"""


def _generic(text: str):
    return next(mq.walk(mq.parse(text), "linalg.generic"))


@pytest.mark.parametrize("kind", ("sin", "cos"))
@pytest.mark.parametrize("shape", ("7", "2x3", "2x3x4"))
def test_registered_unary_f32_math_is_source_only(kind, shape):
    text = _program(scalar=f'"math.{kind}"(%input) <{{fastmath = #arith.fastmath<none>}}> : (f32) -> f32', shape=shape)
    pattern = recognize_static_f32_math_body(_generic(text))
    assert pattern.operation == f"math.{kind}"
    assert pattern.shape == tuple(int(dim) for dim in shape.split("x"))
    assert pattern.ordered_types == ("f32", "f32", "f32")
    assert pattern.scope == SOURCE_ONLY_SCOPE
    assert "selected-libm equivalence" in pattern.scope


@pytest.mark.parametrize("kind", ("exp", "rsqrt"))
def test_exp_and_rsqrt_have_separate_source_only_recognition(kind):
    scalar = f'"math.{kind}"(%input) <{{fastmath = #arith.fastmath<none>}}> : (f32) -> f32'
    pattern = recognize_static_f32_exp_rsqrt_body(_generic(_program(scalar=scalar)))
    assert pattern.operation == f"math.{kind}"
    assert pattern.shape == (2, 3)
    with pytest.raises(InvalidLinalgPattern, match="closed unary f32 math"):
        validate_static_f32_math_source_body({
            "schema": STATIC_F32_MATH_SOURCE_BODY_SCHEMA, "operation": f"math.{kind}",
        })
    for changed in (
        _program(scalar=f'"math.{kind}"(%acc) : (f32) -> f32'),
        _program(scalar=f'"math.{kind}"(%input) '
                 '<{fastmath = #arith.fastmath<fast>}> : (f32) -> f32'),
        _program(scalar=f'"math.{kind}"(%input) : (f32) -> f32', input_map="(d1, d0)"),
    ):
        with pytest.raises(InvalidLinalgPattern):
            recognize_static_f32_exp_rsqrt_body(_generic(changed))


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        (
            _program(
                input_dtype="f64",
                scalar='"math.sin"(%input) <{fastmath = #arith.fastmath<none>}> : (f64) -> f32',
            ),
            "f32 input",
        ),
        (_program(shape="?x3"), "static"),
        (_program(input_map="(d1, d0)"), "identity"),
        (
            _program(iterators="#linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>"),
            "parallel",
        ),
        (_program(scalar='"math.sin"(%acc) : (f32) -> f32'), "input block argument"),
        (_program(yielded="%acc"), "directly yield"),
        (_program(scalar='"math.sin"(%input) <{fastmath = #arith.fastmath<reassoc>}> : (f32) -> f32'), "fast-math"),
        (_program(scalar='"math.cos"(%input) {unsafe = true} : (f32) -> f32'), "unknown attribute"),
        (
            _program(
                scalar_body='%value = "math.sin"(%input) : (f32) -> f32\n'
                '        %extra = "arith.negf"(%value) : (f32) -> f32'
            ),
            "exactly one",
        ),
        (_program(generic_attributes="{unknown = true}"), "unrecognized"),
        (_program(scalar='"math.tanh"(%input) : (f32) -> f32'), "registered unary"),
        (_program(scalar='"math.powf"(%input, %input) : (f32, f32) -> f32'), "registered unary"),
        (
            _program(
                scalar_body='%literal = "arith.constant"() <{value = 2.000000e+00 : f32}> : () -> f32\n'
                '        %value = "math.powf"(%input, %literal) '
                "<{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32"
            ),
            "exactly one",
        ),
    ],
)
def test_unproved_sources_are_refused(text, reason):
    with pytest.raises(InvalidLinalgPattern, match=reason):
        recognize_static_f32_math_body(_generic(text))


def test_source_wrapper_binds_read_bytes_and_ordinals(tmp_path):
    path = tmp_path / "neutral.mlir"
    path.write_text(_program())
    ordinal = next(i for i, op in enumerate(mq.walk(mq.parse(path.read_text()))) if mq.op_name(op) == "linalg.generic")
    screened = screen_static_f32_math_source(path, (ordinal,))
    assert screened.raw_sha256 == sha256(path.read_bytes()).hexdigest()
    assert screened.normalized_sha256
    assert screened.ordinals == ((ordinal, recognize_static_f32_math_body(_generic(path.read_text()))),)
    path.write_text(_program(shape="3x2"))
    assert screen_static_f32_math_source(path, (ordinal,)).raw_sha256 != screened.raw_sha256
    for bad in ((True,), (-1,), (ordinal, ordinal), (10000,)):
        with pytest.raises(InvalidLinalgPattern):
            screen_static_f32_math_source(path, bad)


def test_provenance_labels_cannot_turn_an_unproved_body_into_math():
    text = _program(
        scalar='"math.tanh"(%input) : (f32) -> f32',
        generic_attributes='{prov.op = "sin", prov.aten = "aten.sin.default"}',
    )
    with pytest.raises(InvalidLinalgPattern, match="registered unary"):
        recognize_static_f32_math_body(_generic(text))


def _selected_math(kind: str):
    return {
        "host": {
            "package_sha256": "a" * 64,
            "capability_spec_sha256": "b" * 64,
            "dtype_strategy": "int8_w8a8",
            "capability_spec": {
                "schema": "merlin.host_capabilities.v1",
                "status": "reviewed",
                "compiler": {"package_sha256": "a" * 64, "dtype_strategy": "int8_w8a8"},
                "operations": [
                    {
                        "id": "neutral_math_source",
                        "ops": ["linalg.generic"],
                        "placement": "host",
                        "signature": {
                            "ordered_operand_dtypes": ["f32", "f32"],
                            "ordered_result_dtypes": ["f32"],
                            "ranks": [2],
                        },
                        "source_body": {"schema": STATIC_F32_MATH_SOURCE_BODY_SCHEMA, "operation": f"math.{kind}"},
                    }
                ],
                "evidence": {"scope": "neutral declaration only"},
            },
        }
    }


@pytest.mark.parametrize("kind", ("sin", "cos"))
def test_typed_host_screen_requires_closed_declared_body_and_all_occurrences(kind):
    selected = _selected_math(kind)
    declaration = selected["host"]["capability_spec"]["operations"][0]["source_body"]
    assert validate_static_f32_math_source_body(declaration) == declaration
    assert static_f32_math_ordered_types(declaration) == ("f32", "f32", "f32")
    validate_host_capabilities(selected["host"]["capability_spec"])
    source = _generic(_program(scalar=f'"math.{kind}"(%input) : (f32) -> f32'))
    row = {"mlir_operation": "linalg.generic", "count": 2}
    observed = {
        "family": "elementwise_map",
        "ordered_operand_dtypes": ["f32", "f32"],
        "ordered_result_dtypes": ["f32"],
        "rank": 2,
    }
    assert admit_host_operation(selected, row, observed)["status"] == "unknown"
    accepted = admit_host_operation(
        selected,
        row,
        observed,
        source_operations=(source, _generic(_program(scalar=f'"math.{kind}"(%input) : (f32) -> f32'))),
    )
    assert accepted["status"] == "admitted" and accepted["reviewed"] is True
    assert accepted["source_body_proof"]["schema"] == STATIC_F32_MATH_SOURCE_BODY_SCHEMA
    assert len(accepted["source_body_proof"]["patterns"]) == 2
    for operations, signature in (
        ((source,), observed),
        ((source, source), observed),
        ((source, _generic(_program(scalar='"math.cos"(%input) : (f32) -> f32'))), observed)
        if kind == "sin"
        else ((source, _generic(_program(scalar='"math.sin"(%input) : (f32) -> f32'))), observed),
        ((source, _generic(_program(scalar=f'"math.{kind}"(%input) : (f32) -> f32'))), {**observed, "rank": 3}),
    ):
        assert admit_host_operation(selected, row, signature, source_operations=operations)["status"] == "unsupported"


def test_math_declaration_refuses_extra_fields_pow_and_wrong_schema():
    selected = _selected_math("sin")
    declaration = selected["host"]["capability_spec"]["operations"][0]
    for invalid in (
        {**declaration["source_body"], "predicate": "none"},
        {**declaration["source_body"], "operation": "math.powf"},
        {**declaration["source_body"], "operation": []},
        {**declaration["source_body"], "schema": "merlin.static_boolean_body.v1"},
    ):
        declaration["source_body"] = invalid
        with pytest.raises(ValueError, match="source_body"):
            validate_host_capabilities(selected["host"]["capability_spec"])
