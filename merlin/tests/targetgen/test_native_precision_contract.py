"""Native eager proofs cannot silently substitute for declared quantized bits."""

import copy

import pytest
from mlir_fixtures import pointwise

from merlin.targetgen.native_precision_contract import proves_quantized_model, requires_quantized_model


def software():
    return dict(
        numerical_semantics=dict(
            quantized_native_fallback=dict(admission="after_native_exact_failure"),
            calibration_domain=dict(
                minimum=-1, maximum=1, evaluation_inputs="never_read", scale_policy="enclosing_power_of_two"
            ),
            quantization_nonfinite="byte_storage_zero_nan_saturating_infinity",
        )
    )


def elementwise(source, output, body):
    ops, _, yielded = body.rpartition("linalg.yield ")
    return pointwise([f"tensor<4x{source}>"], f"tensor<4x{output}>", ops, yields=yielded.split(" :")[0])


@pytest.mark.parametrize(
    "source,output,body,wanted",
    [
        # Floating arithmetic on floating storage: the declared quantized model.
        ("f32", "f32", "%k=arith.constant 0.5 : f32\n%v=arith.addf %a,%k : f32\nlinalg.yield %v : f32", True),
        ("f64", "f64", "%k=arith.constant 2.0 : f64\n%v=arith.mulf %a,%k : f64\nlinalg.yield %v : f64", True),
        (
            "bf16",
            "bf16",
            "%w=arith.extf %a : bf16 to f32\n%k=arith.constant 0.5 : f32\n%s=arith.addf %w,%k : f32\n"
            "%v=arith.truncf %s : f32 to bf16\nlinalg.yield %v : bf16",
            True,
        ),
        (
            "f16",
            "f16",
            "%z=arith.constant 0.0 : f16\n%p=arith.cmpf olt,%a,%z : f16\n%v=arith.select %p,%z,%a : f16\n"
            "linalg.yield %v : f16",
            True,
        ),
        ("f32", "f64", "%v=arith.extf %a : f32 to f64\nlinalg.yield %v : f64", True),
        # Movement, identities, NaN re-encoding and integer storage keep eager bits.
        ("f32", "f32", "linalg.yield %a : f32", False),
        # x * 1 and x / 1 are exact identities (ruling "exact identities are bit-exact"); any other
        # literal scale or offset, including x + (-0.0), is an affine map.
        ("f32", "f32", "%k=arith.constant 1.0 : f32\n%v=arith.mulf %a,%k : f32\nlinalg.yield %v : f32", False),
        ("bf16", "bf16", "%k=arith.constant 1.0 : bf16\n%v=arith.divf %a,%k : bf16\nlinalg.yield %v : bf16", False),
        ("f32", "f32", "%k=arith.constant -0.0 : f32\n%v=arith.addf %a,%k : f32\nlinalg.yield %v : f32", True),
        # A select between x and an exact IEEE identity of x is x for every condition.
        (
            "f32",
            "f32",
            "%z=arith.constant 0.0 : f32\n%k=arith.constant 1.0 : f32\n%p=arith.cmpf ogt,%a,%z : f32\n"
            "%m=arith.mulf %a,%k : f32\n%v=arith.select %p,%a,%m : f32\nlinalg.yield %v : f32",
            False,
        ),
        (
            "f16",
            "f16",
            "%w=arith.extf %a : f16 to f32\n%v=arith.truncf %w : f32 to f16\nlinalg.yield %v : f16",
            False,
        ),
        (
            "f32",
            "f32",
            "%q=arith.constant 0x7FC00000 : f32\n%n=arith.cmpf uno,%a,%a : f32\n%v=arith.select %n,%q,%a : f32\n"
            "linalg.yield %v : f32",
            False,
        ),
        ("i64", "f32", "%c=arith.sitofp %a : i64 to f32\n%k=arith.constant 0.5 : f32\n%v=arith.addf %c,%k : f32\nlinalg.yield %v : f32", False),
        ("i32", "i32", "%k=arith.constant 3 : i32\n%v=arith.addi %a,%k : i32\nlinalg.yield %v : i32", False),
    ],
)
def test_ir_floating_work_controls_the_declared_model(source, output, body, wanted):
    assert requires_quantized_model(elementwise(source, output, body)) is wanted


def test_dead_floating_arithmetic_does_not_change_the_model():
    body = "%w=arith.extf %a : f32 to f64\n%k=arith.constant 1.0 : f64\n%t=arith.addf %w,%k : f64\nlinalg.yield %a : f32"
    assert requires_quantized_model(elementwise("f32", "f32", body)) is False


def test_byte_model_requires_matching_parameters_and_independent_reference():
    from merlin.llvmlower.quantized_native import parameters

    spec = software()
    plan = dict(
        quantized_contract=parameters(spec),
        integer_reference=dict(engine="independent_source_quantized_exact_rational"),
    )
    assert proves_quantized_model(plan, spec)
    for change in (dict(scale_numerator=3), dict(zero_point=1)):
        bad = copy.deepcopy(plan)
        bad["quantized_contract"].update(change)
        assert not proves_quantized_model(bad, spec)
    for engine in (None, "eager", "native_typed_reference"):
        bad = copy.deepcopy(plan)
        bad["integer_reference"]["engine"] = engine
        assert not proves_quantized_model(bad, spec)


def test_legacy_model_requires_declared_calibration_and_independent_rational_engine():
    spec = software()
    recipe = copy.deepcopy(spec["numerical_semantics"])
    plan = dict(capture_recipe=recipe, integer_reference=dict(engine="independent_pt2e_exact_rational"))
    assert proves_quantized_model(plan, spec)
    bad = copy.deepcopy(plan)
    bad["capture_recipe"]["calibration_domain"]["minimum"] = -2
    assert not proves_quantized_model(bad, spec)
    bad = copy.deepcopy(plan)
    bad["capture_recipe"]["calibration_domain"]["evaluation_inputs"] = "observed"
    assert not proves_quantized_model(bad, spec)
    assert not proves_quantized_model(dict(exact_dtype_sites=10), spec)


IDENTITY = "module { func.func @forward(%x: tensor<2xf32>) -> tensor<2xf32> { return %x : tensor<2xf32> } }"


def quantized_admission(tmp_path, monkeypatch, accepted):
    """A spec file for an identity source whose IR is declared quantized and review ``accepted``."""
    import yaml

    from merlin.targetgen import native_precision_contract, operator_review

    (tmp_path / "model.mlir").write_text(IDENTITY)
    spec = software()
    spec["numerical_semantics"]["native_unit_contract"] = dict(excluded_constructions=[])
    path = tmp_path / "spec.yaml"
    path.write_text(yaml.safe_dump(spec))
    monkeypatch.setattr(native_precision_contract, "requires_quantized_model", lambda _: True)
    monkeypatch.setattr(operator_review, "review_record", lambda _: dict(operator_accepted=accepted))
    return path


def test_wrong_model_rolls_back_source_abi_and_retained_record_identity(tmp_path, monkeypatch):
    from merlin.targetgen.core_aten_admission import prepare

    original = IDENTITY
    path = quantized_admission(tmp_path, monkeypatch, True)
    (tmp_path / "inputs.npz").write_bytes(b"original input storage")
    record = dict(status="bundled", input_abi=[dict(dtype="f32", shape=[2])])
    report = dict(cases=[record], output_count=1)

    def candidate(directory, report, **kwargs):
        (directory / "model.mlir").write_text(original + "\n// prepared eager candidate")
        (directory / "inputs.npz").write_bytes(b"changed storage")
        report["cases"][0]["input_abi"] = [dict(dtype="i8", shape=[8])]
        report["output_count"] = 99
        return dict(selection=dict(identities={}), exact_dtype_sites=1)

    plan = prepare(tmp_path, report, candidate, software_spec=path)
    assert plan["refusal_code"] == "declared_quantized_precision_requires_quantized_model"
    assert (tmp_path / "model.mlir").read_text() == original
    assert (tmp_path / "inputs.npz").read_bytes() == b"original input storage"
    assert report["output_count"] == 1
    assert report["cases"][0] is record
    assert record["input_abi"] == [dict(dtype="f32", shape=[2])]


def test_first_native_refusal_keeps_its_reason(tmp_path, monkeypatch):
    # A later precision check does not overwrite the reason of an earlier refusal.
    from merlin.targetgen.core_aten_admission import prepare

    path = quantized_admission(tmp_path, monkeypatch, False)
    report = dict(cases=[dict(status="bundled", input_abi=[dict(dtype="f32", shape=[2])])], output_count=1)
    plan = prepare(tmp_path, report, lambda *a, **k: dict(selection=dict(identities={}), exact_dtype_sites=1),
                   software_spec=path)
    assert plan["refusal_code"] == "native_unit_operator_acceptance_required"


@pytest.mark.parametrize("error,refused", [(ValueError, True), (TypeError, False)])
def test_only_a_fold_refusal_keeps_the_raw_arithmetic(monkeypatch, error, refused):
    import merlin.targetgen.core_aten_numeric as numeric

    def fold(module):
        raise error("fold")

    monkeypatch.setattr(numeric, "normalize_closed_identities", fold)
    module = elementwise("f32", "f32", "%k=arith.constant 0.5 : f32\n%v=arith.addf %a,%k : f32\nlinalg.yield %v : f32")
    if refused:
        assert requires_quantized_model(module)
    else:
        with pytest.raises(TypeError):
            requires_quantized_model(module)
