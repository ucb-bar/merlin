"""Qualify numerical conditions through complete source functions and actual SSA writes.

Acceptance is universal on the explicit finite-input/model domain. It neither proves the
primitive floating laws for the host interpreter nor approves the provisional recipe.
"""

from __future__ import annotations

import json
import subprocess

import pytest
from test_lean_w8a8_body import exported
from test_lean_w8a8_source_facts import source as parameters
from test_lean_w8a8_source_program import execute, source
from test_lean_w8a8_source_quantize import samples_for, source_program, word

from merlin.common.paths import repo_root
from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.verify.lean import differential as diff

pytestmark = pytest.mark.usefixtures("lean_oracle")


def qualify(document, work):
    proc = subprocess.run(
        [
            "lake",
            "env",
            "lean",
            "--run",
            str(repo_root() / "merlin/tests/fixtures/lean_w8a8_finite_defined.lean"),
            str(document),
        ],
        cwd=repo_root() / "merlin/lean",
        text=True,
        capture_output=True,
        check=False,
    )
    assert proc.returncode in (0, 1), proc.stdout + proc.stderr
    result = json.loads(proc.stdout)
    assert result["scope"] == "provisional_finite_input_initialized_functions"
    assert result["input_domain"] == "typed_initialized_inputs_with_finite_f32_values"
    assert result["floating_domain"] == "byte_extension_of_base_satisfying_explicit_primitive_laws"
    assert result["synthetic_model_domain_inhabited"] is True
    assert result["primitive_model_laws_verified"] is False
    assert result["whole_pass_verified"] is False and result["policy_reviewed"] is False
    assert result["theorem"] == "MerlinVerify.Check.PolicyW8A8FiniteDefined.check_sound"
    assert (proc.returncode == 0) == result["conditional_source_initialized"]
    (work / "finite-defined-report.json").write_text(json.dumps(result, indent=2))
    return result


@pytest.mark.parametrize("width", [32, 64])
@pytest.mark.parametrize("shape", [(), (2, 8), (0, 3)])
@pytest.mark.parametrize("dequantize", [False, True])
def test_complete_source_quantizer_has_conditional_initialized_execution(
    tmp_path, lean_tools, width, shape, dequantize
):
    document = exported(
        source(width, shape, dequantize=dequantize), tmp_path, lean_tools, generic=True, allow_unregistered=True
    )
    result = qualify(document, tmp_path)
    assert result["conditional_source_initialized"]
    assert result["functions"] == [
        {"function": "forward", "conditional_source_initialized": True, "unconditional_source_initialized": False}
    ]


@pytest.mark.parametrize("width", [32, 64])
@pytest.mark.parametrize("shape", [(), (2, 8), (0, 3)])
def test_qualified_source_function_agrees_with_native_mlir(tmp_path, lean_tools, fidelity_tools, width, shape):
    document = exported(source(width, shape), tmp_path, lean_tools, generic=True, allow_unregistered=True)
    assert qualify(document, tmp_path)["conditional_source_initialized"]
    _, arguments = samples_for(width, shape)
    actual = execute(document, [[sample[0]] for sample in arguments], tmp_path)
    native_work = tmp_path / "native"
    native_work.mkdir()
    native_document = exported(source_program(width, shape), native_work, lean_tools)
    inputs, outputs = diff.signature(json.loads(native_document.read_text()), "forward")
    native = diff.execute_mlir(
        native_work / "actual.mlir",
        [[sample[0], [str(word(0.1))], ["17"]] for sample in arguments],
        native_work,
        fidelity_tools,
        inputs=inputs,
        outputs=outputs,
        function="forward",
    )
    assert all(sample["status"] == "ok" for sample in actual)
    assert [sample["values"][0]["bits"] for sample in actual] == native["results"]


@pytest.mark.parametrize("scale", ["0.0", "-0.1", "0x00000001", "0x007FFFFF", "0x7F800000", "0x7FC00001"])
def test_bad_or_subnormal_scale_cannot_acquire_a_normal_fact(tmp_path, lean_tools, scale):
    document = exported(source(64, (2,), scale=scale), tmp_path, lean_tools, generic=True, allow_unregistered=True)
    assert not qualify(document, tmp_path)["conditional_source_initialized"]


@pytest.mark.parametrize("width", [32, 64])
@pytest.mark.parametrize("scale", ["0x00800000", "0x7F7FFFFF"])
def test_both_normal_scale_boundaries_are_supported(tmp_path, lean_tools, width, scale):
    document = exported(source(width, (2,), scale=scale), tmp_path, lean_tools, generic=True, allow_unregistered=True)
    assert qualify(document, tmp_path)["conditional_source_initialized"]


@pytest.mark.parametrize(
    "bits,expected", [("00000000", True), ("bf800000", True), ("7f800000", False), ("7fc00001", False)]
)
def test_constant_data_class_is_derived_from_its_actual_bits(tmp_path, lean_tools, bits, expected):
    document = exported(source(64, (2,)), tmp_path, lean_tools, generic=True, allow_unregistered=True)
    data = json.loads(document.read_text())
    ops = data["module"]["ops"][0]["regions"][0]["blocks"][0]["ops"]
    scalar = json.loads(json.dumps(ops[0]))
    scalar["results"][0]["id"] = 900001
    table = scalar["props"] if "value" in scalar["props"] else scalar["attrs"]
    table["value"]["bits"] = bits
    quantizer = next(op for op in ops if op["name"] == "quant_ext.quantize_per_tensor")
    tensor_ty = data["module"]["ops"][0]["regions"][0]["blocks"][0]["args"][0]["type"]
    tensor = {
        "name": "tensor.splat",
        "results": [{"id": 900002, "type": tensor_ty}],
        "operands": [900001],
        "attrs": {},
        "props": {},
        "regions": [],
        "successors": [],
    }
    quantizer["operands"][0] = 900002
    ops[:0] = [scalar, tensor]
    document.write_text(json.dumps(data))
    assert qualify(document, tmp_path)["conditional_source_initialized"] == expected


def test_finite_entry_domain_excludes_the_nan_counterexample(tmp_path, lean_tools):
    document = exported(source(64, (1,)), tmp_path, lean_tools, generic=True, allow_unregistered=True)
    assert qualify(document, tmp_path)["conditional_source_initialized"]
    actual = execute(document, [[[str(word(float("nan")))]]], tmp_path)
    assert actual[0]["status"] == "ok" and actual[0]["values"][0]["bits"] == [None]


def test_empty_function_roster_is_not_qualification(tmp_path, lean_tools):
    document = exported(parse_mlir_text("module {}"), tmp_path, lean_tools, generic=True, allow_unregistered=True)
    assert not qualify(document, tmp_path)["conditional_source_initialized"]


@pytest.mark.parametrize("unknown", ["argument", "producer"])
def test_scale_type_or_entry_finiteness_does_not_prove_normality(tmp_path, lean_tools, unknown):
    document = exported(parameters(unknown=unknown), tmp_path, lean_tools, generic=True, allow_unregistered=True)
    assert not qualify(document, tmp_path)["conditional_source_initialized"]


@pytest.mark.parametrize("target", ["scalar", "tensor", "data"])
def test_actual_ssa_writes_replace_numerical_facts(tmp_path, lean_tools, target):
    document = exported(source(64, (2,)), tmp_path, lean_tools, generic=True, allow_unregistered=True)
    data = json.loads(document.read_text())
    block = data["module"]["ops"][0]["regions"][0]["blocks"][0]
    ops = block["ops"]
    if target == "scalar":
        replacement = json.loads(json.dumps(ops[0]))
        table = replacement["props"] if "value" in replacement["props"] else replacement["attrs"]
        table["value"]["bits"] = "7fc00001"
        ops.insert(1, replacement)
    else:
        original = ops[1]["results"][0] if target == "tensor" else block["args"][0]
        ops.insert(
            2,
            {
                "name": "tensor.empty",
                "results": [original],
                "operands": [],
                "attrs": {},
                "props": {},
                "regions": [],
                "successors": [],
            },
        )
    document.write_text(json.dumps(data))
    assert not qualify(document, tmp_path)["conditional_source_initialized"]


def test_initialized_arithmetic_does_not_supply_a_numerical_class(tmp_path, lean_tools):
    document = exported(source(64, (2,)), tmp_path, lean_tools, generic=True, allow_unregistered=True)
    data = json.loads(document.read_text())
    ops = data["module"]["ops"][0]["regions"][0]["blocks"][0]["ops"]
    constant = ops[0]["results"][0]
    changed = {**constant, "id": 900001}
    ops[1]["operands"] = [changed["id"]]
    ops.insert(
        1,
        {
            "name": "arith.mulf",
            "results": [changed],
            "operands": [constant["id"]] * 2,
            "attrs": {},
            "props": {},
            "regions": [],
            "successors": [],
        },
    )
    document.write_text(json.dumps(data))
    assert not qualify(document, tmp_path)["conditional_source_initialized"]


@pytest.mark.parametrize("width", [32, 64])
def test_internal_call_does_not_inherit_a_callee_certificate(tmp_path, lean_tools, width):
    document = exported(source(width, (2,), wrapper=True), tmp_path, lean_tools, generic=True, allow_unregistered=True)
    result = qualify(document, tmp_path)
    assert not result["conditional_source_initialized"]
    assert {entry["function"]: entry["conditional_source_initialized"] for entry in result["functions"]} == {
        "kernel": True,
        "forward": False,
    }


@pytest.mark.parametrize("change", ["metadata", "regions"])
def test_source_contract_is_rechecked_in_the_function(tmp_path, lean_tools, change):
    document = exported(source(64, (2,)), tmp_path, lean_tools, generic=True, allow_unregistered=True)
    data = json.loads(document.read_text())
    operations = data["module"]["ops"][0]["regions"][0]["blocks"][0]["ops"]
    quantizer = next(op for op in operations if op["name"] == "quant_ext.quantize_per_tensor")
    if change == "metadata":
        quantizer["attrs"]["unknown_numerical_mode"] = {"kind": "unit"}
    else:
        quantizer["regions"] = [{"blocks": []}]
    document.write_text(json.dumps(data))
    assert not qualify(document, tmp_path)["conditional_source_initialized"]
