"""Quantized activation has real byte arithmetic and fully checked IO boundaries."""

import pytest
from mlir_fixtures import generic

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.targetgen import floating_contract, native_byte_activation, operator_review, quantized_literal_reference
from merlin.targetgen.native_quantized_activation import prepare


def source(dtype, *, extra=False):
    t = f"tensor<4x{dtype}>"
    relu = generic([("%x", t)], t, f"""%zero=arith.constant 0.0 : {dtype}
      %p=arith.cmpf olt,%a,%zero : {dtype}
      %v=arith.select %p,%zero,%a : {dtype}""")
    effect = '"fixture.effect"() : () -> ()' if extra else ""
    return parse_mlir_text(
        f"module {{func.func @forward(%x:{t}) -> ({t},{t}) {{\n{relu}\n{effect}\nreturn %r,%x : {t},{t} }} }}"
    )


def setup(monkeypatch, *, precision="quantized_exact", maximum=1):
    calls = []
    if precision != "quantized_exact":
        # An IR whose floating work keeps eager bits (identity/movement) cannot be quantized.
        monkeypatch.setattr(floating_contract, "requires_quantized_model", lambda m: dict(quantized=False, ops=[]))
    monkeypatch.setattr(operator_review, "review_record", lambda r: dict(operator_accepted=True))
    monkeypatch.setattr(quantized_literal_reference, "qualify", lambda d, r, p, c: {**p, "contract": c})

    def compile(d, r, s, p, pkg, module, root, data, binding):
        module.verify()
        from merlin.targetgen.native_unit_policy import construction_evidence

        assert construction_evidence(module)["status"] == "eligible"
        assert str(data.type.element_type) == "i8"
        assert root.inputs[0] is data and data.owner.op_name.data == "quant_ext.quantize_per_tensor"
        assert binding["op_ordinal"] == list(module.walk()).index(data.owner)
        calls.append(module)
        return dict(exact_dtype_sites=1)

    monkeypatch.setattr(native_byte_activation, "compile_activation", compile)
    selection = dict(
        spec=dict(
            numerical_semantics=dict(
                quantized_native_fallback=dict(admission="after_native_exact_failure"),
                calibration_domain=dict(
                    minimum=-maximum,
                    maximum=maximum,
                    evaluation_inputs="never_read",
                    scale_policy="enclosing_power_of_two",
                ),
                quantization_nonfinite="byte_storage_zero_nan_saturating_infinity",
            )
        )
    )
    report = dict(cases=[dict(status="bundled", case=dict(schema=1, arguments={}, expected={}))])
    return calls, selection, report


@pytest.mark.parametrize("dtype", ["f16", "bf16", "f32"])
def test_all_declared_codes_have_a_real_byte_activation_and_exact_storage_boundary(tmp_path, monkeypatch, dtype):
    calls, selection, report = setup(monkeypatch)
    plan = prepare(tmp_path, report, selection, None, None, source(dtype))
    assert plan is not None and len(calls) == 1
    assert plan["native_semantic_proof"]["storage_narrowing"] == "all_256_codes_exact"
    ops = list(calls[0].walk())
    assert sum(getattr(getattr(op, "op_name", None), "data", "") == "quant_ext.quantize_per_tensor" for op in ops) == 1
    assert any(op.name == "arith.cmpi" for op in ops)
    assert any(getattr(getattr(op, "op_name", None), "data", "") == "quant_ext.dequantize_per_tensor" for op in ops)


@pytest.mark.parametrize("options", [dict(precision="bit_exact_vs_eager"), dict(maximum=1e8)])
def test_quantization_cannot_waive_eager_bits_or_an_unrepresentable_output_domain(tmp_path, monkeypatch, options):
    calls, selection, report = setup(monkeypatch, **options)
    assert prepare(tmp_path, report, selection, None, None, source("f16")) is None
    assert calls == []


def test_an_extra_effect_is_not_erased_by_source_specialization(tmp_path, monkeypatch):
    calls, selection, report = setup(monkeypatch)
    assert prepare(tmp_path, report, selection, None, None, source("f16", extra=True)) is None
    assert calls == []
