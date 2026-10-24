"""Refusal and semantic gates for declared and native integer execution."""

import os
import subprocess

import numpy as np
import pytest
from targetgen_fixtures import measured_routing


def run_torch(script):
    result = subprocess.run(
        [os.environ["MERLIN_M2M_PYTHON"], "-c", script], capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_native_source_oracle_preserves_wrapping_output_and_raw_accumulator():
    run_torch(r'''
import numpy as np
from xdsl.parser import Parser
from merlin.frontends.linalg_mlir import make_context
from merlin.llvmlower.modular_contractions import widen_module
from merlin.targetgen.core_aten_w8a8_reference import evaluate
source = """builtin.module {
func.func @forward(%a: tensor<1x2xi8>, %b: tensor<2x1xi8>) -> tensor<1x1xi8> {
  %c = arith.constant 0 : i8
  %e = tensor.empty() : tensor<1x1xi8>
  %z = linalg.fill ins(%c : i8) outs(%e : tensor<1x1xi8>) -> tensor<1x1xi8>
  %r = linalg.matmul ins(%a, %b : tensor<1x2xi8>, tensor<2x1xi8>)
    outs(%z : tensor<1x1xi8>) -> tensor<1x1xi8>
  func.return %r : tensor<1x1xi8>
}}"""
module = Parser(make_context(), source).parse_module()
assert widen_module(module, [('i8','i8','i32')]) == 1
from merlin.kernels.shapes import observe_contractions
from xdsl.dialects.builtin import FunctionType
function = next(op for op in module.walk() if op.name == 'func.func')
returned = next(op for op in module.walk() if op.name == 'func.return')
acc = observe_contractions(module)[0][0].results[0]
returned.operands = [*returned.operands, acc]
function.properties['function_type'] = FunctionType.from_lists(
    [a.type for a in function.body.block.args], [v.type for v in returned.operands])
arrays = [np.array([[127, -128]], np.int8), np.array([[127], [-128]], np.int8)]
values, _ = evaluate(module, arrays, [], quant_min=-128, quant_max=127, eps=2**-12)
assert values[0].item() == 1
assert values[1].item() == 32513
''')


def test_static_calibration_refuses_nonfinite_inputs_and_intermediates():
    run_torch(r"""
import torch
from merlin.targetgen._recipe_quantizer import apply_recipe, CalibrationRefused
recipe = {
    'schema':'quant_recipe_v1', 'status':'derived', 'families':['contraction'],
    'weight':dict(dtype='int8',granularity='tensor',symmetric=True,quant_min=-127,quant_max=127),
    'activation':dict(dtype='int8',granularity='tensor',symmetric=True,quant_min=-128,quant_max=127,mode='static'),
}
class Model(torch.nn.Module):
    def forward(self,a,b,c):
        return (a @ b) @ c
for value in [float('nan'), float('inf'), 3e38]:
    inputs = (torch.full((2,2), value), torch.full((2,2), 2.), torch.ones(2,2))
    try:
        apply_recipe(Model(),recipe,example_inputs=inputs)
    except CalibrationRefused as exc:
        assert exc.code == 'nonfinite_calibration'
    else:
        raise AssertionError('nonfinite calibration must be typed refusal')
""")


def test_spec_policy_observation_changes_values_only_after_exact_gate():
    run_torch(r"""
import copy
from merlin.targetgen.core_aten_eval import _validate_observation
original = dict(kind='tensor',dtype='float32',shape=[1],stride=[1],storage_offset=0,requires_grad=False,values=[1.])
integer = {**original,'values':[.9]}
case = dict(expected=original,post_arguments={},comparison='torch_close',
            comparison_parameters=dict(rtol=1e-5,atol=1e-6),mutated_arguments=[],output_input_aliases=[])
observation = dict(status='executed',execution_kind='simulator',output=integer,
                   post_arguments={},mutated_arguments=[],output_input_aliases=[])
try:
    _validate_observation(case,observation)
except AssertionError:
    pass
else:
    raise AssertionError('fp32 comparison should detect approximation')
observation['numeric_reference'] = dict(policy='spec_int8',admission_status='admitted',
    gate=dict(status='pass',results=[dict(status='pass')]),expected=integer,post_arguments={})
_validate_observation(case,observation)
for field,value in [('stride',[2]),('storage_offset',1),('requires_grad',True)]:
    changed = copy.deepcopy(observation)
    changed['numeric_reference']['expected'][field] = value
    try:
        _validate_observation(case,changed)
    except AssertionError as exc:
        assert 'metadata' in str(exc)
    else:
        raise AssertionError('numeric policy must retain eager metadata')
observation['numeric_reference']['gate']['status'] = 'mismatch'
try:
    _validate_observation(case,observation)
except AssertionError as exc:
    assert 'exact integer-reference gate' in str(exc)
else:
    raise AssertionError('failed reference must not alter eager values')
""")


def measured_spec_int8(overload, dtype, expected, raw, admission):
    """A measured device spec_int8 case whose independent integer reference is ``raw``."""
    reference = dict(status="prepared", engine="independent", expected_output_bytes={"0": raw.hex()})
    return dict(
        overload=overload,
        status="bundled",
        numeric_policy="spec_int8",
        case=dict(expected=expected),
        output_indices=[0],
        output_abi=[dict(dtype=dtype, shape=[1])],
        numeric_admission=dict(status="admitted", **admission, integer_reference=reference),
        routing=measured_routing(overload),
    )


@pytest.mark.parametrize(
    "dtype,narrowing,qualification",
    [
        ("i8", True, "device path conforms; host int8 narrowing not declared"),
        ("i32", False, "conforming"),
    ],
)
def test_native_integer_grade_qualifies_only_exact_measured_device(monkeypatch, dtype, narrowing, qualification):
    from merlin.targetgen.core_aten_batch_grade import grade_core_aten_batch

    monkeypatch.setattr("merlin.targetgen.core_aten_device.verify_execution_evidence", lambda _: True)
    raw = np.array([7], np.int8 if dtype == "i8" else np.int32).tobytes()
    expected = dict(kind="tensor", dtype="int8" if narrowing else "int32", shape=[1], values=[7])
    admission = dict(native_contractions=1, host_int8_narrowing=narrowing)
    record = measured_spec_int8("aten._int_mm.default", dtype, expected, raw, admission)
    result = grade_core_aten_batch(dict(cases=[record]), [raw])
    verdict = result["cases"][record["overload"]]
    assert verdict["numeric_qualification"] == qualification
    assert result["spec_int8_passed_count"] == 1
    result = grade_core_aten_batch(dict(cases=[record]), [bytes(len(raw))])
    assert result["spec_int8_passed_count"] == 0
    assert result["cases"][record["overload"]]["integer_reference"]["status"] == "mismatch"


@pytest.mark.parametrize(
    "code",
    [
        "no_eligible_recipe_operations",
        "nonfinite_calibration",
        "complex_calibration",
        "integerization_golden_disagreement",
    ],
)
def test_structured_numeric_refusal_roundtrip_and_unknown_code_fails_closed(tmp_path, code):
    from merlin.targetgen.capture_refusals import NumericCaptureRefused, read_refusal, write_refusal

    write_refusal(tmp_path, code=code, reason="proven refusal")
    refused = read_refusal(tmp_path / "refusal.json")
    assert isinstance(refused, NumericCaptureRefused)
    assert refused.code == code
    write_refusal(tmp_path, code="unexpected_runtime_failure", reason="not a policy refusal")
    assert read_refusal(tmp_path / "refusal.json") is None


def test_finite_calibration_refuses_internal_histogram_range_overflow():
    run_torch(r"""
import torch
import torchao.quantization.pt2e.quantize_pt2e as pt2e
from torchao.quantization.pt2e import HistogramObserver, MinMaxObserver
from torchao.quantization.pt2e.observer import ObserverBase
from merlin.targetgen._recipe_quantizer import apply_recipe, CalibrationRefused, require_finite_observer_state

maximum = torch.finfo(torch.float32).max
source = torch.tensor([[-maximum, -1.], [0., maximum]])
assert torch.isfinite(source).all()
observer = HistogramObserver(dtype=torch.int8, qscheme=torch.per_tensor_symmetric,
                             quant_min=-128, quant_max=127)
observer(source)
assert all(torch.isfinite(buffer).all() for buffer in observer.buffers())
assert not torch.isfinite(observer.max_val - observer.min_val)
minimum_maximum = MinMaxObserver(dtype=torch.int8, qscheme=torch.per_tensor_symmetric,
                               quant_min=-128, quant_max=127)
minimum_maximum(source)
assert torch.isfinite(minimum_maximum.calculate_qparams()[0]).all()
require_finite_observer_state(torch.nn.Sequential(minimum_maximum), ObserverBase, HistogramObserver,
                              calibration_sample=1)

recipe = {
    "schema":"quant_recipe_v1", "status":"derived", "families":["contraction"],
    "weight":dict(dtype="int8",granularity="tensor",symmetric=True,quant_min=-127,quant_max=127),
    "activation":dict(dtype="int8",granularity="tensor",symmetric=True,quant_min=-128,quant_max=127,mode="static"),
}
class Model(torch.nn.Module):
    def forward(self, a, b):
        return a @ b

def forbidden(*args, **kwargs):
    raise AssertionError("invalid histogram state must refuse before convert_pt2e")
pt2e.convert_pt2e = forbidden
try:
    apply_recipe(Model(), recipe, example_inputs=(source, torch.full((2, 2), .125)))
except CalibrationRefused as exc:
    assert exc.code == "nonfinite_calibration"
    assert exc.details["observer_path"].startswith("activation_post_process_")
    assert exc.details["observer_buffer"] == "max_val-min_val"
    assert exc.details["calibration_sample"] == 1
    assert exc.details["nonfinite_count"] == 1
else:
    raise AssertionError("finite endpoints with overflowing histogram range must refuse")
""")


def test_standalone_worker_refusal_preserves_details_and_fallback(tmp_path):
    import sys
    from pathlib import Path

    from merlin.targetgen import capture_refusals

    script = r"""
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from capture_refusals import read_refusal, write_exception_refusal
out = Path(sys.argv[2])
refusal = RuntimeError("actual observer range overflow")
refusal.code = "nonfinite_calibration"
refusal.details = {"observer_path": "activation_post_process_0", "calibration_sample": 2}
write_exception_refusal(out, refusal, fallback_code="no_eligible_recipe_operations")
receipt = read_refusal(out / "refusal.json")
assert receipt.code == refusal.code
assert str(receipt) == str(refusal)
assert receipt.details["observer_path"] == refusal.details["observer_path"]
assert receipt.details["calibration_sample"] == 2
write_exception_refusal(out, RuntimeError("no eligible contraction"),
                        fallback_code="no_eligible_recipe_operations")
receipt = read_refusal(out / "refusal.json")
assert receipt.code == "no_eligible_recipe_operations"
assert str(receipt) == "no eligible contraction"
"""
    subprocess.run(
        [sys.executable, "-I", "-S", "-c", script, str(Path(capture_refusals.__file__).parent), str(tmp_path)],
        check=True,
    )


def test_contraction_free_quantized_execution_keeps_its_exact_reference_policy(monkeypatch):
    from merlin.targetgen.core_aten_batch_grade import grade_core_aten_batch

    monkeypatch.setattr("merlin.targetgen.core_aten_device.verify_execution_evidence", lambda _: True)
    raw = np.array([7.0], np.float64).tobytes()
    expected = dict(kind="tensor", dtype="float64", shape=[1], values=[7.0])
    admission = dict(quantized_contractions=0, quantized_non_contraction_sites=1)
    record = measured_spec_int8("aten._to_copy.default", "f64", expected, raw, admission)
    result = grade_core_aten_batch(dict(cases=[record]), [raw])
    assert result["cases"][record["overload"]]["numeric_execution_policy"] == "spec_int8"
    assert result["spec_int8_passed_count"] == 1
    changed = grade_core_aten_batch(dict(cases=[record]), [np.array([7.000000000000001], np.float64).tobytes()])
    assert changed["spec_int8_passed_count"] == 0
