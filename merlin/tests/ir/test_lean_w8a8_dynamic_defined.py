"""Qualify the complete emitted dynamic sequence under explicit numerical model laws."""

from __future__ import annotations

import json
import subprocess

import pytest
from test_lean_w8a8_body import exported
from test_lean_w8a8_module import source
from test_lean_w8a8_scalar import parts

from merlin.common.paths import repo_root
from merlin.llvmlower.passes_quant_int import lower_contraction_int8

pytestmark = pytest.mark.usefixtures("lean_oracle")


def qualify(document, work):
    proc = subprocess.run(
        [
            "lake",
            "env",
            "lean",
            "--run",
            str(repo_root() / "merlin/tests/fixtures/lean_w8a8_dynamic_defined.lean"),
            str(document),
        ],
        cwd=repo_root() / "merlin/lean",
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode in (0, 1), proc.stdout + proc.stderr
    report = json.loads(proc.stdout)
    assert report["scope"] == "provisional_dynamic_w8a8_initialized_functions"
    assert report["input_domain"] == "typed_initialized_inputs_with_finite_f32_values"
    assert report["floating_domain"] == "byte_extension_with_explicit_source_and_dynamic_primitive_laws"
    assert report["additional_dynamic_primitive_laws"] == 3
    assert report["primitive_model_laws_verified"] is False
    assert report["whole_pass_verified"] is False and report["policy_reviewed"] is False
    assert report["synthetic_model_domain_inhabited"] is True
    assert report["theorem"] == "MerlinVerify.Check.PolicyW8A8DynamicDefined.check_sound"
    assert (proc.returncode == 0) == report["conditional_initialized"]
    (work / "dynamic-defined-report.json").write_text(json.dumps(report, indent=2))
    return report


@pytest.mark.parametrize("shape", [(2, 3, 4), (3, 3, 3), (1, 0, 2), (0, 3, 2), (2, 3, 0)])
def test_complete_generated_dynamic_w8a8_function_is_initialized(tmp_path, lean_tools, shape):
    module = source(*shape)
    assert lower_contraction_int8(module) == 1
    document = exported(module, tmp_path, lean_tools, generic=True, allow_unregistered=True)
    report = qualify(document, tmp_path)
    assert report["conditional_initialized"]
    assert not report["functions"][0]["source_laws_only_initialized"]


@pytest.mark.parametrize(
    "change", ["denominator", "floor", "reciprocal", "rounding", "upper_bound", "lower_bound", "old_destination"]
)
def test_actual_numeric_contract_corruptions_are_refused(tmp_path, lean_tools, change):
    from xdsl.dialects.builtin import FloatAttr, f32

    module = source(3, 3, 3)
    assert lower_contraction_int8(module) == 1
    stages = parts(module)
    scale = next(op for kind, _, op in stages if kind == "scale")
    reciprocal = next(op for kind, _, op in stages if kind == "reciprocal")
    quantize = next(op for kind, _, op in stages if kind == "quantize")
    scale_ops = list(scale.body.block.ops)
    quantize_ops = list(quantize.body.block.ops)
    if change == "denominator":
        scale_ops[0].operands[1].owner.properties["value"] = FloatAttr(126.0, f32)
    elif change == "floor":
        scale_ops[2].properties["value"] = FloatAttr(0.0, f32)
    elif change == "reciprocal":
        reciprocal.body.block.first_op.properties["value"] = FloatAttr(2.0, f32)
    elif change == "rounding":
        quantize_ops[1].name = "math.round"
    elif change in ("upper_bound", "lower_bound"):
        position, value = (2, 128.0) if change == "upper_bound" else (3, -128.0)
        quantize_ops[position].operands[1].owner.properties["value"] = FloatAttr(value, f32)
    else:
        quantize.body.block.last_op.operands = [quantize.body.block.args[-1]]
    document = exported(module, tmp_path, lean_tools, generic=True, allow_unregistered=True)
    assert not qualify(document, tmp_path)["conditional_initialized"]


@pytest.mark.parametrize(
    "change", ["literal_shadow", "literal_nan", "missing_region", "metadata", "wrong_argument_type"]
)
def test_actual_binding_and_region_protocol_cannot_forge_numeric_evidence(tmp_path, lean_tools, change):
    module = source()
    assert lower_contraction_int8(module) == 1
    document = exported(module, tmp_path, lean_tools, generic=True, allow_unregistered=True)
    data = json.loads(document.read_text())
    ops = data["module"]["ops"][0]["regions"][0]["blocks"][0]["ops"]
    quantize = next(
        op
        for op in ops
        if op["name"] == "linalg.generic"
        and any(inner["name"] == "arith.fptosi" for inner in op["regions"][0]["blocks"][0]["ops"])
    )
    body = quantize["regions"][0]["blocks"][0]
    if change.startswith("literal_"):
        upper_id = body["ops"][2]["operands"][1]
        constant = next(op for op in ops if any(result["id"] == upper_id for result in op["results"]))
        if change == "literal_nan":
            table = constant["props"] if "value" in constant["props"] else constant["attrs"]
            table["value"]["bits"] = "7fc00001"
        else:
            # The protocol is adversarial: a new write must invalidate the earlier literal.
            ops.insert(
                ops.index(constant) + 1,
                {
                    "name": "arith.mulf",
                    "results": constant["results"],
                    "operands": [upper_id] * 2,
                    "attrs": {},
                    "props": {},
                    "regions": [],
                    "successors": [],
                },
            )
    elif change == "missing_region":
        quantize["regions"] = []
    elif change == "metadata":
        quantize["attrs"]["unknown_numerical_mode"] = {"kind": "unit"}
    else:
        body["args"][0]["type"] = {"kind": "int", "width": 32}
    document.write_text(json.dumps(data))
    assert not qualify(document, tmp_path)["conditional_initialized"]
