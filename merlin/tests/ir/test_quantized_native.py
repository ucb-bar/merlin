# ruff: noqa: E501 -- MLIR fixture operations keep one source line
"""General quantized fallback: one entry rounding and exact rational readout."""

import json
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from mlir_fixtures import forward, generic, native_library, pointwise

import merlin.targetgen.core_aten_admission as admission
import merlin.targetgen.core_aten_components as components
import merlin.targetgen.core_aten_numeric as numeric
import merlin.targetgen.core_aten_spec_int8 as spec
import merlin.targetgen.quantized_native as fallback
from merlin.common.yaml import load_yaml
from merlin.llvmlower.floating_components import rewrite_floating_components
from merlin.llvmlower.quantized_native import finite_source, parameters, quantize_entries
from merlin.targetgen.core_aten_batch_grade import grade_core_aten_batch
from merlin.targetgen.core_aten_numeric import prepare_spec_int8
from merlin.targetgen.exact_arithmetic import ExactTensor, quantize
from merlin.targetgen.quantized_native import prepare
from merlin.targetgen.quantized_native_reference import (
    Gaussian,
    rational_inverse_real_transform,
    rational_real_transform,
)

SPEC = Path(__file__).resolve().parents[3] / "examples/gemmini/target/software-spec.yaml"


PREPARE = dict(target="test", facts={}, software_spec=None, host_capabilities=None)


def contract():
    return parameters(load_yaml(SPEC))


def test_declared_calibration_independent_of_payload():
    p = contract()
    assert Fraction(p["scale_numerator"], p["scale_denominator"]) == Fraction(1, 64)
    assert p["calibration_domain"]["evaluation_inputs"] == "never_read"
    assert p["nonfinite"] == "byte_storage_zero_nan_saturating_infinity"
    s = load_yaml(SPEC)
    s["numerical_semantics"]["calibration_domain"]["evaluation_inputs"] = "read"
    with pytest.raises(ValueError):
        parameters(s)


def test_quantized_grading_checks_oracle_bytes_and_retains_eager_diagnostic():
    reference = np.array([Fraction(1, 64)], np.float32).tobytes()
    record = dict(
        overload="fixture",
        status="bundled",
        numeric_policy="spec_int8",
        output_indices=[0],
        output_abi=[dict(dtype="f32", shape=[1])],
        case=dict(expected=dict(kind="tensor", dtype="float32", shape=[1], values=[0.016])),
        numeric_admission=dict(
            quantized_contract=contract(),
            integer_reference=dict(
                status="prepared", engine="independent_source", expected_output_bytes={"0": reference.hex()}
            ),
        ),
    )
    positive = grade_core_aten_batch(dict(cases=[record]), [reference])["cases"]["fixture"]
    assert positive["status"] == "pass" and positive["precision_class"] == "b"
    negative = grade_core_aten_batch(dict(cases=[record]), [np.array([0.016], np.float32).tobytes()])["cases"][
        "fixture"
    ]
    assert negative["status"] == "mismatch"
    assert negative["integer_reference"]["status"] == "mismatch"


def test_full_entry_domain_and_posts_unchanged(tmp_path):
    n = 520
    t = f"tensor<{n}xf32>"
    copy = generic([("%x", t)], t, "", yields="%a")
    module = forward(f"%x:{t}", f"({t},{t})", copy, ret="%r,%x", attrs="{llvm.emit_c_interface}")
    quantize_entries(module, contract())
    module.verify()
    function = next(op for op in module.walk() if op.name == "func.func")
    ret = function.body.block.last_op
    assert ret.operands[1] is function.body.block.args[0]
    run = native_library(module, tmp_path)
    x = (np.arange(n, dtype=np.float32) - 260) / 128
    x[:8] = [np.nan, np.inf, -np.inf, -0.0, np.finfo(np.float32).max, np.finfo(np.float32).tiny, 0.5 / 64, 1.5 / 64]
    y = np.zeros(n, np.float32)
    post = np.zeros(n, np.float32)
    run(x, y, post)

    expected = np.asarray(
        (quantize(x, Fraction(1, 64), nonfinite=contract()["nonfinite"]) * Fraction(1, 64)).tolist(), np.float32
    )
    assert y.tobytes() == expected.tobytes()
    assert post.tobytes() == x.tobytes()


def test_gaussian_reference_is_rational():
    a, b = Gaussian(Fraction(1, 3), Fraction(7, 11)), Gaussian(2, -3)
    p = a * b
    assert p.real == Fraction(85, 33) and p.imag == Fraction(3, 11)
    q = p / b
    assert q.real == a.real and q.imag == a.imag


def test_rational_transform_oracle_has_no_trigonometric_approximation():
    x = ExactTensor(np.array([[Fraction(i, 3) for i in range(4)]], object), "f32")
    result = rational_real_transform([x, [1], 0, True], lambda v: v.values, ExactTensor).values
    assert [(v.real, v.imag) for v in result.flat] == [(2, 0), (Fraction(-2, 3), Fraction(2, 3)), (Fraction(-2, 3), 0)]
    bad = ExactTensor(np.array([Fraction(1)] * 3, object), "f32")
    with pytest.raises(ValueError, match="non-rational root"):
        rational_real_transform([bad, [0], 0, True], lambda v: v.values, ExactTensor)

    half = ExactTensor(np.array([Gaussian(1), Gaussian(2, 3), Gaussian(4)], object), "f32")
    inverse = rational_inverse_real_transform([half, [0], 0, 4], lambda v: v.values, ExactTensor)
    assert list(inverse.values) == [9, -9, 1, 3]


@pytest.fixture
def declared_spec(monkeypatch):
    software = load_yaml(SPEC)
    monkeypatch.setattr(spec, "select_spec", lambda *a: dict(spec=software))


def test_runtime_parallel_products_do_not_gain_native_admission(tmp_path, declared_spec):
    t = "tensor<2xf32>"
    (tmp_path / "model.mlir").write_text(str(pointwise([t, t], t, "%v=arith.mulf %a,%b : f32")))
    original = (tmp_path / "model.mlir").read_bytes()
    report = dict(cases=[dict(status="bundled", input_abi=[dict(dtype="f32")] * 2, output_abi=[dict(dtype="f32")])])
    assert prepare(tmp_path, report, **PREPARE) is None
    assert (tmp_path / "model.mlir").read_bytes() == original


def test_reduction_dependent_parallel_product_is_still_a_composition(tmp_path, declared_spec):
    t, one, maps = "tensor<2xf32>", "tensor<1xf32>", ["affine_map<(d0)->(d0)>", "affine_map<(d0)->(0)>"]
    seed = f"%z=arith.constant 0.0 : f32\n%seed=tensor.splat %z : {one}\n"
    add = "%s=arith.addf %a,%acc : f32"
    total = generic(
        [("%x", t)], one, add, yields="%s", name="%sum", iterators=["reduction"], maps=maps, out="%seed", acc="%acc"
    )
    product = generic([("%x", t), ("%sum", one)], t, "%v=arith.mulf %a,%b : f32", maps=maps + maps[:1])
    (tmp_path / "model.mlir").write_text(str(forward(f"%x:{t}", t, seed + total + "\n" + product)))
    report = dict(cases=[dict(status="bundled", input_abi=[dict(dtype="f32")], output_abi=[dict(dtype="f32")])])
    assert prepare(tmp_path, report, **PREPARE) is None
    assert "parallel runtime product" in (tmp_path / "quantized-native-refusal.json").read_text()


def test_complex_components_with_frozen_coefficients_reach_numeric_proof(tmp_path, monkeypatch, declared_spec):
    (tmp_path / "model.mlir").write_text("""module {
      func.func @forward(%x:tensor<1x2xf32>) -> tensor<1x1xf32> {
        %w=arith.constant dense<1.0> : tensor<2x1xf32>
        %z=arith.constant 0.0 : f32
        %seed=tensor.splat %z : tensor<1x1xf32>
        %r=linalg.matmul ins(%x,%w:tensor<1x2xf32>,tensor<2x1xf32>) outs(%seed:tensor<1x1xf32>) -> tensor<1x1xf32>
        return %r : tensor<1x1xf32>
    }}""")
    called = []

    def prove(*args, **kw):
        called.append(True)
        return dict(status="admitted", exact_dtype_sites=1, quantized_contract=kw["quantized_contract"])

    def reference(directory, mode):
        (directory / "quantized-native-reference.json").write_text(
            json.dumps(dict(expected_output_bytes={"0": "00000000"}))
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(components, "prepare_integer_components", prove)
    monkeypatch.setattr(numeric, "_source_reference", reference)
    report = dict(
        output_count=1,
        cases=[dict(status="bundled", input_abi=[dict(dtype="complex<f32>")], output_abi=[dict(dtype="f32")])],
    )
    plan = prepare(tmp_path, report, **PREPARE)
    assert called and plan["native_admission"]["status"] == "device"


def test_native_exact_success_does_not_try_quantization(tmp_path, monkeypatch):
    expected = dict(native_admission=dict(status="device"))
    monkeypatch.setattr(admission, "prepare", lambda *args, **kw: expected)

    def forbidden(*args, **kw):
        raise AssertionError("native exact path changed")

    monkeypatch.setattr(fallback, "prepare", forbidden)
    assert prepare_spec_int8(tmp_path, {}) is expected


@pytest.mark.parametrize("oracle_failure", [False, True])
def test_payload_changes_cannot_choose_host_placement(tmp_path, monkeypatch, declared_spec, oracle_failure):
    prepared_sources = []

    def component_plan(directory, report, selection, provider, package, module, **kw):
        prepared_sources.append(str(module))
        return dict(status="admitted", exact_dtype_sites=1, quantized_contract=kw["quantized_contract"])

    def independent_reference(directory, mode):
        if oracle_failure:
            return SimpleNamespace(returncode=1, stderr="oracle qualification failed")
        request = json.loads((directory / "quantized-native-request.json").read_text())
        value = request["records"][0]["case"]["values"][0]
        (directory / "quantized-native-reference.json").write_text(
            json.dumps(dict(expected_output_bytes={"0": np.array([value], np.float32).tobytes().hex()}))
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(components, "prepare_integer_components", component_plan)
    monkeypatch.setattr(numeric, "_source_reference", independent_reference)
    plans = []
    for ordinal, values in enumerate(([0.0], [float("inf")], [float("nan")])):
        directory = tmp_path / str(ordinal)
        directory.mkdir()
        # Live floating arithmetic: only the declared quantized model can discharge it.
        source = str(
            pointwise(["tensor<1xf32>"], "tensor<1xf32>", "%k=arith.constant 0.5 : f32\n%v=arith.addf %a,%k : f32")
        )
        (directory / "model.mlir").write_text(source)
        abi = dict(input_abi=[dict(dtype="f32")], output_abi=[dict(dtype="f32")])
        report = dict(output_count=1, cases=[dict(status="bundled", **abi, case=dict(values=values))])
        if oracle_failure:
            with pytest.raises(ArithmeticError, match="qualification failed"):
                prepare(directory, report, **PREPARE)
            assert (directory / "model.mlir").read_text() == source
        else:
            plan = prepare(directory, report, **PREPARE)
            assert plan["native_admission"]["status"] == "device"
            plans.append({k: v for k, v in plan.items() if k != "integer_reference"})
    assert len(set(prepared_sources)) == 1
    if plans:
        assert plans[0] == plans[1] == plans[2]


def test_nan_word_overlay_is_removed_but_live_nan_is_refused():
    def fixture(yielded):
        body = """%nan=arith.constant 0x7FC00000 : f32
          %isnan=arith.cmpf uno,%a,%a : f32
          %v=arith.select %isnan,%nan,%a : f32"""
        return pointwise(["tensor<1xf32>"], "tensor<1xf32>", body, yields=yielded)

    module = fixture("%v")
    finite_source(module)
    module.verify()
    assert not any(op.name == "arith.cmpf" for op in module.walk())
    with pytest.raises(ValueError, match="nonfinite literal"):
        finite_source(fixture("%nan"))


def test_rational_divisors_survive_layouts_until_exported_quantize(tmp_path):
    t, square = "tensor<256xf32>", "tensor<16x16xf32>"
    prelude = f"%three=arith.constant 3.0 : f32\n%divisor=tensor.splat %three : {t}\n"
    quotient = generic([("%x", t), ("%divisor", t)], t, "%v=arith.divf %a,%b : f32", name="%q")
    view = f"%shape=arith.constant dense<[16,16]> : tensor<2xi64>\n%view=tensor.reshape %q(%shape) : ({t},tensor<2xi64>) -> {square}"
    total = generic([("%view", square), ("%view", square)], square, "%v=arith.addf %a,%b : f32")
    module = forward(f"%x:{t}", square, f"{prelude}{quotient}\n{view}\n{total}", attrs="{llvm.emit_c_interface}")
    p = contract()
    grids = quantize_entries(module, p)
    bounds, proofs = rewrite_floating_components(
        module, [None], known_grids=grids, allow_reassociation=True, quantized_contract=p
    )
    module.verify()
    assert any(row["denominator"] == 3 for row in proofs)
    # A rational parameter is an actual native weight significand. A device
    # identity followed by a host division would not discharge this contract.
    producer = next(
        op
        for op in module.walk()
        if op.name == "linalg.generic" and any(it.data.value == "reduction" for it in op.get_iterator_types())
    )
    weights = []
    value = producer.inputs[1]
    while value.owner.name == "tensor.insert":
        product = value.owner.operands[0].owner
        assert product.name == "arith.muli"
        weights.append(product.operands[1].owner.value.value.data)
        value = value.owner.operands[1]
    assert weights and set(weights) == {64}
    x = np.arange(-128, 128, dtype=np.float32) / 64
    y = np.zeros((16, 16), np.float32)
    native_library(module, tmp_path)(x, y)
    expected = np.asarray(
        [min(127, max(-128, round(Fraction(2 * i, 3)))) / 64 for i in range(-128, 128)], np.float32
    ).reshape(16, 16)
    assert y.tobytes() == expected.tobytes()


def test_storage_obligation_refuses_quantization_before_any_payload_or_provider_access(tmp_path, monkeypatch):
    # A storage copy: its IR has no floating arithmetic, so eager bits are obligatory.
    (tmp_path / "model.mlir").write_text(
        "module { func.func @forward(%x: tensor<2xf32>) -> tensor<2xf32> { return %x : tensor<2xf32> } }"
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("a storage obligation must not enter quantized preparation")

    monkeypatch.setattr(spec, "select_spec", forbidden)
    for value in (0.0, float("inf"), float("nan")):
        report = dict(cases=[dict(status="bundled", case=dict(schema={}, arguments=dict(values=[value]), expected={}))])
        assert prepare(tmp_path, report, **PREPARE) is None
        assert (
            json.loads((tmp_path / "quantized-native-refusal.json").read_text())["code"]
            == "precision_class_requires_native_eager_bits"
        )
