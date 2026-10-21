"""Qualify universal typed support and strict policy refinement under the draft semantics.

The claims permit poison. Successful samples, calibrated parameters and source support never
stand in for initialized execution or the separate whole-function numerical domain.
"""

from __future__ import annotations

import json
import subprocess

import pytest
from test_lean_w8a8_body import exported
from test_lean_w8a8_module import calibrated_source
from test_lean_w8a8_module import source as contraction
from test_lean_w8a8_source_facts import source as parameter_source
from test_lean_w8a8_source_program import source

from merlin.common.paths import repo_root
from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.passes_quant_int import lower_contraction_int8

pytestmark = pytest.mark.usefixtures("lean_oracle")


def support(document, work):
    proc = subprocess.run(
        [
            "lake",
            "env",
            "lean",
            "--run",
            str(repo_root() / "merlin/tests/fixtures/lean_w8a8_source_support.lean"),
            str(document),
        ],
        cwd=repo_root() / "merlin/lean",
        text=True,
        capture_output=True,
        check=False,
    )
    assert proc.returncode in (0, 1), proc.stdout + proc.stderr
    report = json.loads(proc.stdout)
    assert report["scope"] == "provisional_source_typed_support"
    assert report["domain"] == "all_declared_typed_inputs_poison_permitted"
    assert report["whole_pass_verified"] is False and report["source_defined"] is False
    assert report["theorem"] == "MerlinVerify.Check.PolicyW8A8Support.checkFunction_sound"
    assert (proc.returncode == 0) == report["supported"]
    (work / "source-support-report.json").write_text(json.dumps(report, indent=2))
    return report


def policy(before, after, work, tools, scope="all"):
    documents = []
    for name, program in [("before", before), ("after", after)]:
        directory = work / name
        directory.mkdir()
        documents.append(exported(program, directory, tools, generic=True, allow_unregistered=True))
    proc = subprocess.run(
        [
            "lake",
            "env",
            "lean",
            "--run",
            str(repo_root() / "merlin/tests/fixtures/lean_w8a8_supported_policy.lean"),
            str(documents[0]),
            str(documents[1]),
            scope,
        ],
        cwd=repo_root() / "merlin/lean",
        text=True,
        capture_output=True,
        check=False,
    )
    assert proc.returncode in (0, 1), proc.stdout + proc.stderr
    report = json.loads(proc.stdout)
    assert report["scope"] == "provisional_typed_policy_refinement"
    assert report["whole_pass_verified"] is False and report["policy_reviewed"] is False
    assert report["initialized_execution_verified"] is False
    assert report["domain"] == "all_declared_typed_inputs_poison_permitted"
    assert report["theorem"] == "MerlinVerify.Check.PolicyW8A8Supported.check_sound"
    assert (proc.returncode == 0) == report["policy_refinement"]
    (work / "supported-policy-report.json").write_text(json.dumps(report, indent=2))
    return report


@pytest.mark.parametrize("width", [32, 64])
@pytest.mark.parametrize("shape", [(), (2, 8), (0, 3)])
def test_actual_quantize_and_dequantize_function_support(tmp_path, lean_tools, width, shape):
    document = exported(
        source(width, shape, dequantize=True), tmp_path, lean_tools, generic=True, allow_unregistered=True
    )
    assert support(document, tmp_path)["supported"]


@pytest.mark.parametrize("scale", ["0.0", "-0.1", "0x7F800000", "0x7FC00001"])
def test_invalid_calibration_cannot_pass_universal_support(tmp_path, lean_tools, scale):
    document = exported(source(64, (1,), scale=scale), tmp_path, lean_tools, generic=True, allow_unregistered=True)
    assert not support(document, tmp_path)["supported"]


@pytest.mark.parametrize("unknown", ["argument", "producer"])
def test_parameter_declarations_do_not_supply_runtime_facts(tmp_path, lean_tools, unknown):
    document = exported(parameter_source(unknown=unknown), tmp_path, lean_tools, generic=True, allow_unregistered=True)
    assert not support(document, tmp_path)["supported"]


@pytest.mark.parametrize("overwrite", ["scalar", "tensor"])
def test_source_support_invalidates_ssa_parameter_overwrites(tmp_path, lean_tools, overwrite):
    # Direct malformed SSA tests; these duplicate definitions are not compiler-valid inputs.
    document = exported(parameter_source(), tmp_path, lean_tools, generic=True, allow_unregistered=True)
    data = json.loads(document.read_text())
    operations = data["module"]["ops"][0]["regions"][0]["blocks"][0]["ops"]
    original = operations[0 if overwrite == "scalar" else 1]["results"][0]
    operations.insert(
        1 if overwrite == "scalar" else 2,
        {
            "name": "arith.addf" if overwrite == "scalar" else "tensor.empty",
            "results": [original],
            "operands": [original["id"], original["id"]] if overwrite == "scalar" else [],
            "attrs": {},
            "props": {},
            "regions": [],
            "successors": [],
        },
    )
    document.write_text(json.dumps(data))
    assert not support(document, tmp_path)["supported"]


def test_support_does_not_claim_poison_free_conversion(tmp_path, lean_tools):
    from test_lean_w8a8_source_program import execute
    from test_lean_w8a8_source_quantize import word

    document = exported(
        source(64, (1,), scale="0x00000001"), tmp_path, lean_tools, generic=True, allow_unregistered=True
    )
    assert support(document, tmp_path)["supported"]
    # All parameters are initialized/admissible, but 0 * overflowing reciprocal is NaN.
    actual = execute(document, [[[str(word(0))]]], tmp_path)
    assert actual == [{"status": "ok", "values": [{"shape": [1], "bits": [None]}]}]


@pytest.mark.parametrize("shape", [(2, 3, 4), (1, 0, 2), (0, 3, 2)])
def test_dynamic_pass_has_supported_strict_policy_refinement(tmp_path, lean_tools, shape):
    before = contraction(*shape)
    after = before.clone()
    assert lower_contraction_int8(after) == 1
    assert policy(before, after, tmp_path, lean_tools)["policy_refinement"]


def test_matching_unsupported_source_cannot_supply_nonvacuity(tmp_path, lean_tools):
    before = parse_mlir_text("""func.func @forward(%x: tensor<2xf32>) -> tensor<2xf32> {
      %r = "frontend.unsupported"(%x) : (tensor<2xf32>) -> tensor<2xf32>
      return %r : tensor<2xf32>
    }""")
    assert not policy(before, before.clone(), tmp_path, lean_tools, scope="captured")["policy_refinement"]


def test_captured_metadata_contract_cannot_be_silently_widened(tmp_path, lean_tools):
    before = calibrated_source()
    after = before.clone()
    assert lower_contraction_int8(after) == 1
    # This fixture's TypeAttr input_dtype is outside the explicit provisional adapter. A raw
    # policy match alone must not turn that missing source support into qualification.
    assert not policy(before, after, tmp_path, lean_tools, scope="captured")["policy_refinement"]


def test_missing_selected_transformation_refuses(tmp_path, lean_tools):
    before = contraction()
    assert not policy(before, before.clone(), tmp_path, lean_tools)["policy_refinement"]


def test_empty_function_roster_cannot_claim_policy_nonvacuity(tmp_path, lean_tools):
    before = parse_mlir_text("module {}")
    assert not policy(before, before.clone(), tmp_path, lean_tools, scope="captured")["policy_refinement"]


def test_internal_call_support_stays_explicitly_unsupported(tmp_path, lean_tools):
    document = exported(source(64, (2, 8), wrapper=True), tmp_path, lean_tools, generic=True, allow_unregistered=True)
    report = support(document, tmp_path)
    assert not report["supported"]
    assert {item["function"]: item["supported"] for item in report["functions"]} == {"kernel": True, "forward": False}


def test_support_permits_ub_and_never_claims_success(tmp_path, lean_tools):
    from test_lean_w8a8_source_program import execute

    program = parse_mlir_text("""func.func @forward(%x: tensor<1xf32>) -> tensor<1xf32> {
      %false = arith.constant false
      cf.assert %false, "precondition"
      return %x : tensor<1xf32>
    }""")
    document = exported(program, tmp_path, lean_tools)
    assert support(document, tmp_path)["supported"]
    assert execute(document, [[["0"]]], tmp_path) == [{"status": "ub"}]
