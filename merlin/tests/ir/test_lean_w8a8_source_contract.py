"""Validate actual captured-source syntax and checked candidate runtime invocations.

The adapter remains uninstalled. Contract recognition does not establish whole-model support;
runtime instance checks do not prove a condition for every possible model input.
"""

from __future__ import annotations

import json
import subprocess

import pytest
from test_lean_w8a8_body import exported
from test_lean_w8a8_dequant_width import samples as dequant_samples
from test_lean_w8a8_source_quantize import samples_for, word

from merlin.common.paths import repo_root
from merlin.frontends.linalg_mlir import parse_mlir_text

pytestmark = pytest.mark.usefixtures("lean_oracle")


def source(kind, width=64, shape=(2, 8), *, metadata=None, scale_shape=()):
    dims = "".join(f"{n}x" for n in shape)
    parameters = "".join(f"{n}x" for n in scale_shape)
    input_type = f"tensor<{dims}{'f32' if kind == 'quantize' else 'i8'}>"
    result_type = f"tensor<{dims}{'i8' if kind == 'quantize' else 'f32'}>"
    scale_type, zero_type = f"tensor<{parameters}f32>", f"tensor<i{width}>"
    if metadata is None:
        metadata = "quant_min = -128 : i64, quant_max = 127 : i64"
        if kind == "quantize":
            metadata += ', output_dtype = "int8"'
    return parse_mlir_text(f"""func.func @forward(%x: {input_type}, %s: {scale_type}, %z: {zero_type})
        -> {result_type} {{
      %r = "quant_ext.{kind}_per_tensor"(%x, %s, %z) <{{{metadata}}}>
        : ({input_type}, {scale_type}, {zero_type}) -> {result_type}
      return %r : {result_type}
    }}""")


def inspect(document, work):
    proc = subprocess.run(
        [
            "lake",
            "env",
            "lean",
            "--run",
            str(repo_root() / "merlin/tests/fixtures/lean_w8a8_source_contract.lean"),
            str(document),
        ],
        cwd=repo_root() / "merlin/lean",
        text=True,
        capture_output=True,
        check=False,
    )
    assert proc.returncode in (0, 1), proc.stdout + proc.stderr
    report = json.loads(proc.stdout)
    assert report["scope"] == "source_contract_eligibility_only"
    assert report["whole_pass_verified"] is False
    (work / "source-contract-report.json").write_text(json.dumps(report, indent=2))
    return report


def execute(document, samples, work):
    request = work / "source-contract-request.json"
    request.write_text(
        json.dumps(
            {
                "contract": "w8a8_source_contract",
                "export": document.read_text(),
                "function": "forward",
                "samples": samples,
            }
        )
    )
    proc = subprocess.run(
        ["lake", "env", "lean", "--run", "Differential.lean", str(request.absolute())],
        cwd=repo_root() / "merlin/lean",
        text=True,
        capture_output=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    answer = json.loads(proc.stdout)
    assert answer["scope"] == "source_contract_instances_only"
    assert answer["whole_pass_verified"] is False
    (work / "source-contract-instances.json").write_text(json.dumps(answer, indent=2))
    return answer


@pytest.mark.parametrize("kind", ["quantize", "dequantize"])
@pytest.mark.parametrize("width", [32, 64])
@pytest.mark.parametrize("shape", [(), (2, 8), (0, 3)])
def test_actual_source_contract_and_runtime_instances(tmp_path, lean_tools, kind, width, shape):
    document = exported(source(kind, width, shape), tmp_path, lean_tools, generic=True, allow_unregistered=True)
    report = inspect(document, tmp_path)
    assert not report["refusals"] and len(report["sites"]) == 1
    assert report["sites"][0]["zero_width"] == width
    assert report["sites"][0]["shape"] == list(shape)
    _, arguments = samples_for(width, shape) if kind == "quantize" else dequant_samples(width, "per_tensor", shape, 0)
    answer = execute(document, arguments, tmp_path)
    assert answer["source_contract_matched"] is True
    assert answer["instance_checks"] == [True] * len(arguments)
    assert all(s["status"] == "ok" and all(None not in v["bits"] for v in s["values"]) for s in answer["results"])


@pytest.mark.parametrize("lower", [-128, -127])
def test_dequantization_range_is_metadata_not_a_storage_bound(tmp_path, lean_tools, lower):
    module = source("dequantize", metadata=f"quant_min = {lower} : i64, quant_max = 127 : i64")
    document = exported(module, tmp_path, lean_tools, generic=True, allow_unregistered=True)
    report = inspect(document, tmp_path)
    assert len(report["sites"]) == 1 and not report["refusals"]


@pytest.mark.parametrize(
    "kind,metadata",
    [
        ("quantize", 'quant_min = -127 : i64, quant_max = 127 : i64, output_dtype = "int8"'),
        ("quantize", 'quant_min = -128 : i64, quant_max = 128 : i64, output_dtype = "int8"'),
        ("quantize", 'quant_min = -128 : i64, quant_max = 127 : i64, output_dtype = "uint8"'),
        ("quantize", "quant_min = -128 : i64, quant_max = 127 : i64"),
        ("dequantize", "quant_min = -127 : i64"),
        ("dequantize", 'quant_min = -128 : i64, quant_max = 127 : i64, input_dtype = "uint8"'),
    ],
)
def test_unsupported_source_metadata_is_refused(tmp_path, lean_tools, kind, metadata):
    document = exported(source(kind, metadata=metadata), tmp_path, lean_tools, generic=True, allow_unregistered=True)
    report = inspect(document, tmp_path)
    assert not report["sites"] and len(report["refusals"]) == 1
    answer = execute(document, [], tmp_path)
    assert answer["source_contract_matched"] is False and answer["instance_checks"] == []


@pytest.mark.parametrize("change", ["unknown_attribute", "duplicate_key", "wrong_ssa_type", "unknown_ssa"])
def test_structured_source_corruptions_are_refused(tmp_path, lean_tools, change):
    # These corruptions exercise the decoder/recognizer boundary directly. They are not
    # represented as compiler-verified mutation witnesses.
    document = exported(source("quantize"), tmp_path, lean_tools, generic=True, allow_unregistered=True)
    data = json.loads(document.read_text())
    block = data["module"]["ops"][0]["regions"][0]["blocks"][0]
    op = block["ops"][0]
    if change == "unknown_attribute":
        op["attrs"]["approximation"] = {"kind": "string", "value": "anything"}
    elif change == "duplicate_key":
        op["attrs"]["quant_min"] = op["props"]["quant_min"]
    elif change == "wrong_ssa_type":
        block["args"][2]["type"]["element"]["width"] = 16
    else:
        op["operands"][0] = 999999
    document.write_text(json.dumps(data))
    report = inspect(document, tmp_path)
    assert not report["sites"] and len(report["refusals"]) == 1


def test_one_invalid_runtime_cell_cannot_pass_a_recognized_operation(tmp_path, lean_tools):
    document = exported(source("quantize", shape=(2,)), tmp_path, lean_tools, generic=True, allow_unregistered=True)
    arguments = [[[str(word(1)), str(word(float("nan")))], [str(word(1))], ["0"]]]
    answer = execute(document, arguments, tmp_path)
    assert answer["source_contract_matched"] is True
    assert answer["instance_checks"] == [False]
    assert answer["results"][0]["values"][0]["bits"][1] is None
    assert not (tmp_path / "native-results.json").exists()


def test_initialized_dequantization_is_distinct_from_finite_data(tmp_path, lean_tools):
    document = exported(source("dequantize", shape=(2,)), tmp_path, lean_tools, generic=True, allow_unregistered=True)
    arguments = [[["0", "128"], [str(word(float("nan")))], ["0"]]]
    answer = execute(document, arguments, tmp_path)
    assert answer["source_contract_matched"] is True and answer["instance_checks"] == [True]
    assert None not in answer["results"][0]["values"][0]["bits"]


def test_combined_checker_refuses_invalid_runtime_tensors(lean_oracle):
    proc = subprocess.run(
        [
            str(lean_oracle),
            "env",
            "lean",
            "--run",
            str(repo_root() / "merlin/tests/fixtures/lean_w8a8_source_instances.lean"),
        ],
        cwd=repo_root() / "merlin/lean",
        text=True,
        capture_output=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert proc.stdout.strip() == "source instance guards passed"
