"""Compare conditional W8A8 execution with the TorchAO versions named by the EL4 guide.

These CPU primitive/linear observations do not identify selected capture environments,
approve independent policy review or prove a concrete IEEE model instance.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest
from test_lean_w8a8_body import exported
from test_lean_w8a8_dequant_width import contract as dequant_contract
from test_lean_w8a8_dequant_width import source_program as dequant_program
from test_lean_w8a8_dynamic_policy import compare_native, mixed_reference, qualify, reused_dynamic_source
from test_lean_w8a8_module import source
from test_lean_w8a8_native_recipe import dynamic_matmul_reference, quantize_rows, samples_for_shape
from test_lean_w8a8_source_program import execute
from test_lean_w8a8_source_quantize import contract as quant_contract
from test_lean_w8a8_source_quantize import request_for, samples_for, source_program

from merlin.common.paths import repo_root
from merlin.llvmlower.passes_quant_int import lower_contraction_int8
from merlin.verify.lean import differential as diff

pytestmark = pytest.mark.usefixtures("lean_oracle")


@pytest.fixture(params=["0.15.0", "0.16.0"])
def torchao_reference(request):
    variable = f"MERLIN_TORCHAO_{request.param.replace('.', '_')}_PYTHON"
    selected = os.environ.get(variable)
    if not selected or not Path(selected).is_file():
        if os.environ.get("MERLIN_LEAN_REQUIRED"):
            pytest.fail(f"{variable} must name the pinned CPU TorchAO reference environment")
        pytest.skip(f"set {variable} to enable versioned TorchAO qualification")
    return Path(selected).absolute(), request.param


def invoke(cases, reference):
    python, version = reference
    environment = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    command = [str(python), "-B", str(repo_root() / "merlin/tests/ir/lean_torchao_oracle.py"), version]
    return subprocess.run(
        command, input=json.dumps(cases), text=True, capture_output=True, check=False, env=environment
    )


def oracle(cases, work, reference):
    _, version = reference
    proc = invoke(cases, reference)
    (work / "torchao-stderr.log").write_text(proc.stderr)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    answer = json.loads(proc.stdout)
    assert answer["scope"] == "versioned_cpu_torchao_fidelity_only"
    assert answer["torch"] == "2.10.0+cpu" and answer["torchao"] == version and answer["numpy"] == "2.4.6"
    assert answer["default_dynamic_config_version"] == 1
    assert answer["gpu_extensions_used"] is False and answer["whole_pass_verified"] is False
    (work / "torchao-results.json").write_text(json.dumps(answer, indent=2))
    return answer["results"]


@pytest.mark.parametrize("shape", [(2, 3, 4), (3, 17, 5), (1, 8, 2)])
def test_qualified_dynamic_output_against_actual_torchao(
    tmp_path, lean_tools, fidelity_tools, torchao_reference, shape
):
    import numpy as np

    before = source(*shape)
    after = before.clone()
    assert lower_contraction_int8(after) == 1
    checked, documents = qualify(before, after, tmp_path, lean_tools)
    assert checked["conditional_initialized_policy_refinement"]
    samples = samples_for_shape(*shape)

    def words(value):
        return [str(int(v)) for v in value.view(np.uint32).flat]

    cases = [
        {"kind": "dynamic_contraction", "shape": shape, "left_bits": words(a), "right_bits": words(b)}
        for a, b in samples
    ]
    actual = oracle(cases, tmp_path, torchao_reference)
    for result, (left, right) in zip(actual, samples, strict=True):
        assert result["route"] == "configured_quantization_and_scaled_integer_matmul"
        assert result["linear_dispatch_accepted"] is False
        qleft, left_scale = quantize_rows(left)
        qright, right_scale = quantize_rows(right.T)
        assert result["left_bytes"] == qleft.reshape(-1).tolist()
        assert result["right_bytes"] == qright.T.reshape(-1).tolist()
        assert result["left_scale_bits"] == words(left_scale)
        assert result["right_scale_bits"] == words(right_scale)
        assert result["output_bits"] == words(dynamic_matmul_reference(left, right))
    arguments = [[words(a), words(b)] for a, b in samples]
    compare_native(documents[1], arguments, [case["output_bits"] for case in actual], tmp_path, fidelity_tools)


def test_captured_reuse_uses_actual_torchao_integer_linear_dispatch(tmp_path, lean_tools, torchao_reference):
    import numpy as np

    before = reused_dynamic_source(False)
    after = before.clone()
    assert lower_contraction_int8(after) == 1
    checked, documents = qualify(before, after, tmp_path, lean_tools, reach="captured")
    assert checked["conditional_initialized_policy_refinement"]

    def words(value):
        return [str(int(v)) for v in value.view(np.uint32).flat]

    samples = samples_for_shape(2, 3, 4)
    cases = [
        {
            "kind": "calibrated_reuse",
            "shape": [2, 3, 4],
            "left_bits": words(a),
            "right_bits": words(b),
            "weight_scale_bits": int(np.float32(0.2).view(np.uint32)),
        }
        for a, b in samples
    ]
    actual = oracle(cases, tmp_path, torchao_reference)
    for result, (left, right) in zip(actual, samples, strict=True):
        assert result["route"] == "calibrated_torchao_integer_linear_dispatch"
        assert result["linear_dispatch_accepted"] is True
        assert result["output_bits"] == words(mixed_reference(left, right, False, calibrated=True))
    # The actual artifact retains source Q/DQ operations. Execute its declared additive
    # interpretation; the native frozen-MLIR runner cannot interpret those source operations.
    observed = execute(documents[1], [[words(a), words(b)] for a, b in samples], tmp_path)
    compare_values(observed, [case["output_bits"] for case in actual])


@pytest.mark.parametrize("width", [32, 64])
@pytest.mark.parametrize("shape", [(), (2, 8), (1, 3, 5)])
def test_source_quantization_and_dequantization_against_actual_torchao(
    tmp_path, lean_tools, fidelity_tools, torchao_reference, width, shape
):
    cases, arguments = samples_for(width, shape)
    # An odd zero point at a rounding tie distinguishes rounding before/after its addition.
    import numpy as np

    ties = np.resize(np.array([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5, -0.0, 0.0], dtype=np.float32), shape)
    bits = [int(v) for v in ties.reshape(-1).view(np.uint32)]
    scale = int(np.float32(1).view(np.uint32))
    cases.append({"zero_width": width, "shape": shape, "input_bits": bits, "scale_bits": scale, "zero": 17})
    arguments.append([[str(v) for v in bits], [str(scale)], ["17"]])
    observed = oracle([{"kind": "affine_per_tensor", **case} for case in cases], tmp_path, torchao_reference)
    quantized = [case["quantized_bytes"] for case in observed]
    dequantized = [case["dequantized_bits"] for case in observed]
    proc, lean = quant_contract(request_for(width, shape, arguments), tmp_path)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    compare_values(lean, quantized)
    native(source_program(width, shape), arguments, quantized, tmp_path / "quantized", lean_tools, fidelity_tools)
    dequant_args = [[q, sample[1], sample[2]] for q, sample in zip(quantized, arguments, strict=True)]
    proc, lean = dequant_contract(
        {"zero_width": width, "kind": "per_tensor", "shape": shape, "axis": 0, "samples": dequant_args}, tmp_path
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    compare_values(lean, dequantized)
    native(
        dequant_program(width, "per_tensor", shape, 0, reference=True),
        dequant_args,
        dequantized,
        tmp_path / "dequantized",
        lean_tools,
        fidelity_tools,
    )


def compare_values(lean, expected):
    assert all(sample["status"] == "ok" for sample in lean)
    assert [[value["bits"] for value in sample["values"]] for sample in lean] == [[sample] for sample in expected]


def native(program, arguments, expected, work, tools, fidelity_tools):
    work.mkdir()
    document = exported(program, work, tools)
    inputs, outputs = diff.signature(json.loads(document.read_text()), "forward")
    result = diff.execute_mlir(
        document.with_suffix(".mlir"),
        arguments,
        work,
        fidelity_tools,
        inputs=inputs,
        outputs=outputs,
        function="forward",
    )
    assert result["results"] == expected


@pytest.mark.parametrize(
    "kind",
    [
        "empty",
        "overflow_bound",
        "nan",
        "zero_scale",
        "wrong_width",
        "calibrated_zero_scale",
        "calibrated_nan_conversion",
    ],
)
def test_torchao_oracle_refuses_invalid_native_observations(tmp_path, torchao_reference, kind):
    import numpy as np

    if kind in {"empty", "overflow_bound"}:
        case = {"kind": "dynamic_contraction", "shape": [1, 0 if kind == "empty" else 131072, 1]}
        reason = "nonempty shapes inside the i32 bound"
    elif kind.startswith("calibrated"):
        one = int(np.float32(1).view(np.uint32))
        scale = np.float32(0.0 if kind == "calibrated_zero_scale" else 1e-45)
        case = {
            "kind": "calibrated_reuse",
            "shape": [1, 1, 1],
            "left_bits": [one],
            "right_bits": [0],
            "weight_scale_bits": int(scale.view(np.uint32)),
        }
        reason = "finite positive scale" if kind == "calibrated_zero_scale" else "undefined float-to-byte"
    else:
        value = np.float32(float("nan") if kind == "nan" else 1.0)
        scale = np.float32(0.0 if kind == "zero_scale" else 1.0)
        case = {
            "kind": "affine_per_tensor",
            "shape": [1],
            "input_bits": [int(value.view(np.uint32))],
            "scale_bits": int(scale.view(np.uint32)),
            "zero": 0,
            "zero_width": 16 if kind == "wrong_width" else 64,
        }
        reason = {
            "nan": "undefined float-to-byte",
            "zero_scale": "finite positive scale",
            "wrong_width": "i32/i64 zero points",
        }[kind]
    proc = invoke([case], torchao_reference)
    assert proc.returncode != 0 and reason in proc.stderr, proc.stdout + proc.stderr
    assert not (tmp_path / "native-results.json").exists()
