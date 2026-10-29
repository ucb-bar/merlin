"""Finite-data quantizer conditions are sufficient only under explicit primitive model laws.

The runtime-instance guard remains separate and checks actual arithmetic. Native and PyTorch
comparisons are finite fidelity evidence; they do not discharge the quantified model laws.
"""

from __future__ import annotations

import json
import subprocess

import pytest
from test_lean_w8a8_body import exported
from test_lean_w8a8_source_contract import inspect
from test_lean_w8a8_source_facts import source
from test_lean_w8a8_source_quantize import pytorch, request_for, samples_for, source_program, word

from merlin.common.paths import repo_root
from merlin.verify.lean import differential as diff

pytestmark = pytest.mark.usefixtures("lean_oracle")


def report(width, shape, samples, work):
    request = work / "finite-domain-request.json"
    request.write_text(json.dumps({"contract": "w8a8_source_quantize", **request_for(width, shape, samples)}))
    proc = subprocess.run(
        ["lake", "env", "lean", "--run", "Differential.lean", str(request)],
        cwd=repo_root() / "merlin/lean",
        text=True,
        capture_output=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    result = json.loads(proc.stdout)
    assert result["whole_pass_verified"] is False
    assert result["primitive_model_laws_verified"] is False
    assert result["finite_precondition_theorem"] == "MerlinVerify.Check.PolicyW8A8FiniteDomain.check_sound"
    assert result["domain_scope"] == "candidate_runtime_instances_only"
    (work / "finite-domain-report.json").write_text(json.dumps(result, indent=2))
    return result


@pytest.mark.parametrize("width", [32, 64])
@pytest.mark.parametrize("shape", [(), (2, 8), (0, 3)])
def test_finite_preconditions_against_pytorch_and_native_mlir(tmp_path, lean_tools, fidelity_tools, width, shape):
    cases, arguments = samples_for(width, shape)
    result = report(width, shape, arguments, tmp_path)
    assert result["finite_preconditions"] == [True] * 7 + [False]
    assert result["domain_checks"] == [True] * 8
    actual = [sample["values"][0]["bits"] for sample in result["results"]]
    assert actual == pytorch(cases, tmp_path)
    document = exported(source_program(width, shape), tmp_path, lean_tools)
    inputs, outputs = diff.signature(json.loads(document.read_text()), "forward")
    native = diff.execute_mlir(
        tmp_path / "actual.mlir",
        arguments,
        tmp_path,
        fidelity_tools,
        inputs=inputs,
        outputs=outputs,
        function="forward",
    )
    assert actual == native["results"]


@pytest.mark.parametrize("width", [32, 64])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_finite_guard_is_sufficient_and_conservative(tmp_path, width, value):
    result = report(width, (1,), [[[str(word(value))], [str(word(1))], ["0"]]], tmp_path)
    assert result["finite_preconditions"] == [False]
    # Infinite data can clip to initialized bytes in the source recipe; NaN cannot. Refusing
    # an input class here must not falsely claim that every rejected instance is undefined.
    assert result["domain_checks"] == [value == value]


@pytest.mark.parametrize("width", [32, 64])
def test_tiny_scale_with_zero_data_keeps_the_poison_counterexample(tmp_path, width):
    result = report(width, (1,), [[["0"], ["1"], ["0"]]], tmp_path)
    assert result["finite_preconditions"] == [False] and result["domain_checks"] == [False]
    assert result["results"][0]["values"][0]["bits"] == [None]


@pytest.mark.parametrize(
    "scale,expected", [("0.1", True), ("0x00800000", True), ("0x007FFFFF", False), ("0x00000001", False)]
)
def test_normal_scale_precondition_uses_actual_ssa_producers(tmp_path, lean_tools, scale, expected):
    document = exported(source(scale=scale), tmp_path, lean_tools, generic=True, allow_unregistered=True)
    result = inspect(document, tmp_path)
    assert result["normal_scale_theorem"] == "MerlinVerify.Check.PolicyW8A8FiniteParameters.check_sound"
    assert len(result["sites"]) == 1 and not result["refusals"]
    assert result["sites"][0]["calibration_parameters_checked"]
    assert result["sites"][0]["normal_scale_checked"] == expected


@pytest.mark.parametrize("unknown", ["argument", "producer"])
def test_unknown_scale_does_not_acquire_a_numerical_precondition(tmp_path, lean_tools, unknown):
    document = exported(source(unknown=unknown), tmp_path, lean_tools, generic=True, allow_unregistered=True)
    result = inspect(document, tmp_path)
    assert not result["sites"][0]["normal_scale_checked"]
