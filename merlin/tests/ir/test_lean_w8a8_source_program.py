"""Exercise provisional Q/DQ semantics through the actual program evaluator.

Native comparisons use separate standard-MLIR reference programs. These finite executions
qualify integration, not independent policy approval or whole-program input-domain proofs.
"""

from __future__ import annotations

import json
import subprocess

import pytest
from test_lean_w8a8_body import exported
from test_lean_w8a8_source_quantize import samples_for, source_program, word

from merlin.common.paths import repo_root
from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.verify.lean import differential as diff

pytestmark = pytest.mark.usefixtures("lean_oracle")


def source(width, shape, *, scale="0.1", zero=17, wrapper=False, dequantize=False):
    dims = "".join(f"{n}x" for n in shape)
    input_type, byte_type = f"tensor<{dims}f32>", f"tensor<{dims}i8>"
    output_type = input_type if dequantize else byte_type
    function = "kernel" if wrapper else "forward"
    tail = f"return %q : {byte_type}"
    if dequantize:
        tail = f"""%dq = "quant_ext.dequantize_per_tensor"(%q, %s, %z)
          <{{quant_min = -127 : i64, quant_max = 127 : i64}}>
          : ({byte_type}, tensor<f32>, tensor<i{width}>) -> {input_type}
          return %dq : {input_type}"""
    text = f"""func.func @{function}(%x: {input_type}) -> {output_type} {{
      %sv = arith.constant {scale} : f32
      %s = tensor.splat %sv : tensor<f32>
      %zv = arith.constant {zero} : i{width}
      %z = tensor.splat %zv : tensor<i{width}>
      %intervening = arith.addi %zv, %zv : i{width}
      %q = "quant_ext.quantize_per_tensor"(%x, %s, %z)
        <{{quant_min = -128 : i64, quant_max = 127 : i64, output_dtype = "int8"}}>
        : ({input_type}, tensor<f32>, tensor<i{width}>) -> {byte_type}
      {tail}
    }}"""
    if wrapper:
        text += f"""func.func @forward(%x: {input_type}) -> {output_type} {{
          %r = func.call @kernel(%x) : ({input_type}) -> {output_type}
          return %r : {output_type}
        }}"""
    return parse_mlir_text("module {" + text + "}")


def execute(document, samples, work):
    request = work / "source-program-request.json"
    request.write_text(
        json.dumps(
            {
                "contract": "w8a8_source_program",
                "export": document.read_text(),
                "function": "forward",
                "index_bits": 64,
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
    result = json.loads(proc.stdout)
    assert result["scope"] == "source_program_candidate_instances_only"
    assert result["whole_pass_verified"] is False
    (work / "source-program-results.json").write_text(json.dumps(result, indent=2))
    return result["results"]


def prefix_report(document, work):
    proc = subprocess.run(
        [
            "lake",
            "env",
            "lean",
            "--run",
            str(repo_root() / "merlin/tests/fixtures/lean_w8a8_source_prefix.lean"),
            str(document),
        ],
        cwd=repo_root() / "merlin/lean",
        text=True,
        capture_output=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    report = json.loads(proc.stdout)
    assert report["scope"] == "source_prefix_calibration_only"
    assert report["whole_pass_verified"] is False
    assert all(report["segments"]) == all(site["calibration_parameters_checked"] for site in report["sites"])
    (work / "source-prefix-report.json").write_text(json.dumps(report, indent=2))
    return report


@pytest.mark.parametrize("width", [32, 64])
@pytest.mark.parametrize("shape", [(), (2, 8), (0, 3)])
def test_complete_constant_prefix_agrees_with_native(tmp_path, lean_tools, fidelity_tools, width, shape):
    document = exported(source(width, shape), tmp_path, lean_tools, generic=True, allow_unregistered=True)
    sites = prefix_report(document, tmp_path)["sites"]
    assert len(sites) == 1 and sites[0]["calibration_parameters_checked"]
    _, arguments = samples_for(width, shape)
    samples = [[sample[0]] for sample in arguments]
    result = execute(document, samples, tmp_path)
    assert all(sample["status"] == "ok" for sample in result)
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
    assert [sample["values"][0]["bits"] for sample in result] == native["results"]


@pytest.mark.parametrize("width", [32, 64])
def test_extension_is_used_through_internal_calls(tmp_path, lean_tools, width):
    document = exported(
        source(width, (2, 8), wrapper=True), tmp_path, lean_tools, generic=True, allow_unregistered=True
    )
    _, arguments = samples_for(width, (2, 8))
    samples = [[sample[0]] for sample in arguments]
    called = execute(document, samples, tmp_path)
    direct_work = tmp_path / "direct"
    direct_work.mkdir()
    direct_document = exported(source(width, (2, 8)), direct_work, lean_tools, generic=True, allow_unregistered=True)
    assert called == execute(direct_document, samples, direct_work)


@pytest.mark.parametrize("width", [32, 64])
def test_quantize_dequantize_uses_actual_produced_bytes(tmp_path, lean_tools, width):
    import numpy as np

    document = exported(
        source(width, (2, 8), dequantize=True), tmp_path, lean_tools, generic=True, allow_unregistered=True
    )
    assert all(site["calibration_parameters_checked"] for site in prefix_report(document, tmp_path)["sites"])
    _, arguments = samples_for(width, (2, 8))
    samples = [[sample[0]] for sample in arguments]
    result = execute(document, samples, tmp_path)
    quant_work = tmp_path / "quantize"
    quant_work.mkdir()
    quant_document = exported(source(width, (2, 8)), quant_work, lean_tools, generic=True, allow_unregistered=True)
    quantized = execute(quant_document, samples, quant_work)
    for actual, quant in zip(result, quantized, strict=True):
        q = np.array([int(v) for v in quant["values"][0]["bits"]], dtype=np.uint8).view(np.int8)
        expected = ((q.astype(np.float32) - np.float32(17)) * np.float32(0.1)).view(np.uint32)
        assert actual["status"] == "ok"
        assert actual["values"][0]["bits"] == [str(v) for v in expected]


@pytest.mark.parametrize("scale", ["0.0", "-0.1", "0x7F800000"])
def test_invalid_calibration_refuses_complete_program(tmp_path, lean_tools, scale):
    document = exported(source(64, (1,), scale=scale), tmp_path, lean_tools, generic=True, allow_unregistered=True)
    assert not prefix_report(document, tmp_path)["sites"][0]["calibration_parameters_checked"]
    assert execute(document, [[[str(word(1))]]], tmp_path)[0]["status"] == "stuck"


@pytest.mark.parametrize("change", ["metadata", "parameter_type", "unknown_producer", "scalar_write", "tensor_write"])
def test_malformed_or_unsupported_program_never_reuses_stale_facts(tmp_path, lean_tools, change):
    # Metadata can be compiler-valid; SSA overwrites and declaration/value mismatches here are
    # direct malformed-JSON tests, not compiler-valid source witnesses.
    document = exported(source(64, (1,)), tmp_path, lean_tools, generic=True, allow_unregistered=True)
    data = json.loads(document.read_text())
    operations = data["module"]["ops"][0]["regions"][0]["blocks"][0]["ops"]
    quant = operations[-2]
    if change == "metadata":
        quant["props"]["unexpected_semantics"] = {"kind": "string", "value": "unsupported"}
    elif change == "parameter_type":
        operations[1]["results"][0]["type"] = {
            "kind": "tensor",
            "shape": [1],
            "element": {"kind": "float", "format": "f32"},
            "encoding": None,
        }
    elif change == "unknown_producer":
        operations[1]["name"] = "frontend.unknown"
    else:
        original = operations[0 if change == "scalar_write" else 1]["results"][0]
        operations.insert(
            1 if change == "scalar_write" else 2,
            {
                "name": "frontend.unknown",
                "results": [original],
                "operands": [],
                "attrs": {},
                "props": {},
                "regions": [],
                "successors": [],
            },
        )
    document.write_text(json.dumps(data))
    assert not prefix_report(document, tmp_path)["sites"][0]["calibration_parameters_checked"]
    assert execute(document, [[[str(word(1))]]], tmp_path)[0]["status"] == "stuck"


@pytest.mark.parametrize("value,scale", [(float("nan"), "0.1"), (0.0, "0x00000001")])
def test_initialized_input_does_not_hide_poison_conversion(tmp_path, lean_tools, value, scale):
    document = exported(source(64, (1,), scale=scale), tmp_path, lean_tools, generic=True, allow_unregistered=True)
    assert prefix_report(document, tmp_path)["sites"][0]["calibration_parameters_checked"]
    result = execute(document, [[[str(word(value))]]], tmp_path)
    assert result == [{"status": "ok", "values": [{"shape": [1], "bits": [None]}]}]


@pytest.mark.parametrize("calibrated", [False, True])
def test_raw_pass_output_still_requires_the_ssa_renaming_bridge(tmp_path, lean_tools, calibrated):
    from test_lean_w8a8_module import calibrated_source, check_module
    from test_lean_w8a8_module import source as contraction

    from merlin.llvmlower.passes_quant_int import lower_contraction_int8

    before = calibrated_source() if calibrated else contraction()
    after = before.clone()
    assert lower_contraction_int8(after) == 1
    report = check_module(before, after, tmp_path, lean_tools, scope="captured" if calibrated else "all")
    assert report["conditional_match"] and not report["exact_match"], report
    assert report["exact_match_theorem"] == "MerlinVerify.Check.PolicyExact.check_sound"
    assert report["policy_reviewed"] is False


@pytest.mark.parametrize("width", [32, 64])
def test_unchanged_captured_source_matches_under_any_semantics(tmp_path, lean_tools, width):
    from test_lean_w8a8_module import check_module

    from merlin.llvmlower.passes_quant_int import lower_contraction_int8

    # This exercises real Q syntax with no selected contraction. It establishes identity-path
    # non-vacuity, not coverage of a changed W8A8 site or source definedness.
    before = source(width, (2, 8))
    after = before.clone()
    assert lower_contraction_int8(after) == 0
    report = check_module(before, after, tmp_path, lean_tools, scope="captured")
    assert report["exact_match"], report


def test_exact_policy_match_checks_module_attributes(tmp_path, lean_tools):
    from test_lean_w8a8_module import check_module
    from xdsl.dialects.builtin import StringAttr

    before = source(64, (2, 8))
    after = before.clone()
    after.attributes["test.changed_semantics"] = StringAttr("unmodelled")
    report = check_module(before, after, tmp_path, lean_tools, scope="captured")
    assert report["expanded"] and not report["exact_match"], report
