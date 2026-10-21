"""Universal initialized execution must be proved separately from typed policy support.

The new source dequantization rule allows actual runtime parameters of either integer width.
Poison data, unsupported quantizers and unsupported calls cannot acquire definedness by matching.
"""

from __future__ import annotations

import json
import subprocess

import pytest
from test_lean_w8a8_body import exported
from test_lean_w8a8_dequant_width import samples, source_program
from test_lean_w8a8_module import source as contraction
from test_lean_w8a8_source_program import execute
from test_lean_w8a8_source_program import source as quantize
from test_lean_w8a8_source_support import support

from merlin.common.paths import repo_root
from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.passes_quant_int import lower_contraction_int8
from merlin.verify.lean import differential as diff

pytestmark = pytest.mark.usefixtures("lean_oracle")


def qualify(before, after, work, scope="captured"):
    proc = subprocess.run(
        [
            "lake",
            "env",
            "lean",
            "--run",
            str(repo_root() / "merlin/tests/fixtures/lean_w8a8_defined_policy.lean"),
            str(before),
            str(after),
            scope,
        ],
        cwd=repo_root() / "merlin/lean",
        text=True,
        capture_output=True,
        check=False,
    )
    assert proc.returncode in (0, 1), proc.stdout + proc.stderr
    report = json.loads(proc.stdout)
    assert report["scope"] == "provisional_initialized_policy_refinement"
    assert report["domain"] == "all_declared_typed_initialized_inputs"
    assert report["theorem"] == "MerlinVerify.Check.PolicyW8A8Qualified.check_sound"
    assert report["whole_pass_verified"] is False and report["policy_reviewed"] is False
    assert (proc.returncode == 0) == report["initialized_policy_refinement"]
    (work / "defined-policy-report.json").write_text(json.dumps(report, indent=2))
    return report


@pytest.mark.parametrize("width", [32, 64])
@pytest.mark.parametrize("shape", [(), (2, 8), (0, 3)])
def test_runtime_parameter_dequantization_is_universally_initialized(tmp_path, lean_tools, width, shape):
    program = source_program(width, "per_tensor", shape, 0, reference=False)
    document = exported(program, tmp_path, lean_tools, generic=True, allow_unregistered=True)
    assert support(document, tmp_path)["supported"]
    report = qualify(document, document, tmp_path)
    assert report["initialized_policy_refinement"]
    assert report["functions"] == [{"function": "forward", "source_initialized": True}]


@pytest.mark.parametrize("width", [32, 64])
@pytest.mark.parametrize("shape", [(), (2, 8), (0, 3)])
def test_whole_dequantization_function_matches_native_mlir(tmp_path, lean_tools, fidelity_tools, width, shape):
    source = tmp_path / "source"
    source.mkdir()
    document = exported(
        source_program(width, "per_tensor", shape, 0, reference=False),
        source,
        lean_tools,
        generic=True,
        allow_unregistered=True,
    )
    _, arguments = samples(width, "per_tensor", shape, 0)
    actual = execute(document, arguments, source)
    reference = tmp_path / "reference"
    reference.mkdir()
    standard = exported(source_program(width, "per_tensor", shape, 0, reference=True), reference, lean_tools)
    inputs, outputs = diff.signature(json.loads(standard.read_text()), "forward")
    native = diff.execute_mlir(
        reference / "actual.mlir",
        arguments,
        reference,
        fidelity_tools,
        inputs=inputs,
        outputs=outputs,
        function="forward",
    )
    assert [result["values"][0]["bits"] for result in actual] == native["results"]


@pytest.mark.parametrize("scale", ["0.0", "-0.1", "0x7F800000", "0x7FC00001"])
def test_initialized_float_values_are_distinct_from_finiteness(tmp_path, lean_tools, scale):
    program = parse_mlir_text(f"""func.func @forward(%q: tensor<2xi8>) -> tensor<2xf32> {{
      %sv = arith.constant {scale} : f32
      %s = tensor.splat %sv : tensor<f32>
      %zv = arith.constant 17 : i64
      %z = tensor.splat %zv : tensor<i64>
      %r = "quant_ext.dequantize_per_tensor"(%q, %s, %z)
        : (tensor<2xi8>, tensor<f32>, tensor<i64>) -> tensor<2xf32>
      return %r : tensor<2xf32>
    }}""")
    document = exported(program, tmp_path, lean_tools, generic=True, allow_unregistered=True)
    assert qualify(document, document, tmp_path)["initialized_policy_refinement"]
    result = execute(document, [[["1", "255"]]], tmp_path)
    assert result[0]["status"] == "ok" and all(bit is not None for bit in result[0]["values"][0]["bits"])


@pytest.mark.parametrize("width", [32, 64])
def test_uninitialized_dequantization_data_cannot_acquire_definedness(tmp_path, lean_tools, width):
    program = source_program(width, "per_tensor", (2,), 0, reference=False)
    document = exported(program, tmp_path, lean_tools, generic=True, allow_unregistered=True)
    data = json.loads(document.read_text())
    block = data["module"]["ops"][0]["regions"][0]["blocks"][0]
    q = {**block["args"][0], "id": 999999}
    block["ops"][0]["operands"][0] = q["id"]
    block["ops"].insert(
        0,
        {
            "name": "tensor.empty",
            "results": [q],
            "operands": [],
            "attrs": {},
            "props": {},
            "regions": [],
            "successors": [],
        },
    )
    document.write_text(json.dumps(data))
    assert support(document, tmp_path)["supported"]
    assert not qualify(document, document, tmp_path)["initialized_policy_refinement"]


@pytest.mark.parametrize("mutation", ["metadata", "storage", "width", "scale_shape", "wiring", "target_metadata"])
def test_actual_contract_and_policy_mutations_refuse(tmp_path, lean_tools, mutation):
    document = exported(
        source_program(64, "per_tensor", (2,), 0, reference=False),
        tmp_path,
        lean_tools,
        generic=True,
        allow_unregistered=True,
    )
    original = json.loads(document.read_text())
    target = tmp_path / "mutated.json"
    changed = json.loads(document.read_text())
    block = changed["module"]["ops"][0]["regions"][0]["blocks"][0]
    op = block["ops"][0]
    if mutation in {"metadata", "target_metadata"}:
        op["attrs"]["unreviewed_numeric_mode"] = {"kind": "string", "value": "changed"}
    elif mutation == "storage":
        block["args"][0]["type"]["element"] = {"kind": "int", "width": 16}
    elif mutation == "width":
        block["args"][2]["type"]["element"] = {"kind": "int", "width": 16}
    elif mutation == "scale_shape":
        block["args"][1]["type"]["shape"] = [1]
    else:
        op["operands"][1] = op["operands"][2]
    target.write_text(json.dumps(changed))
    if mutation == "target_metadata":
        assert qualify(document, document, tmp_path)["initialized_policy_refinement"]
        assert not qualify(document, target, tmp_path)["initialized_policy_refinement"]
    else:
        assert original != changed
        assert not qualify(target, target, tmp_path)["initialized_policy_refinement"]


def test_typed_policy_support_cannot_replace_dynamic_quantizer_domain(tmp_path, lean_tools):
    from test_lean_w8a8_source_support import policy

    before = contraction()
    after = before.clone()
    assert lower_contraction_int8(after) == 1
    assert policy(before, after, tmp_path, lean_tools)["policy_refinement"]
    assert not qualify(tmp_path / "before/actual.json", tmp_path / "after/actual.json", tmp_path, "all")[
        "initialized_policy_refinement"
    ]


@pytest.mark.parametrize("kind", ["quantizer", "assertion", "internal_call", "empty_roster"])
def test_missing_totality_proof_never_passes_as_initialized(tmp_path, lean_tools, kind):
    if kind in {"quantizer", "internal_call"}:
        program = quantize(64, (1,), scale="0x00000001", wrapper=kind == "internal_call")
    elif kind == "assertion":
        program = parse_mlir_text("""func.func @forward(%x: tensor<1xf32>) -> tensor<1xf32> {
          %false = arith.constant false
          cf.assert %false, "precondition"
          return %x : tensor<1xf32>
        }""")
    else:
        program = parse_mlir_text("module {}")
    document = exported(program, tmp_path, lean_tools, generic=True, allow_unregistered=True)
    assert not qualify(document, document, tmp_path)["initialized_policy_refinement"]
