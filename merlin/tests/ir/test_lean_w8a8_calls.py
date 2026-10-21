"""Qualify numerical call contracts against actual bodies, signatures and module fuel.

No caller hint, callee name or typed result supplies a missing numerical fact. Cycles and
external declarations remain unsupported, and floating-model laws stay explicit premises.
"""

from __future__ import annotations

import json
import subprocess

import pytest
from test_lean_w8a8_body import exported
from test_lean_w8a8_finite_policy import qualify as policy
from test_lean_w8a8_finite_policy import recipe_values, reused_reference
from test_lean_w8a8_finite_policy import source as policy_source
from test_lean_w8a8_source_program import execute, source
from test_lean_w8a8_source_quantize import samples_for, source_program, word

from merlin.common.paths import repo_root
from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.passes_quant_int import lower_contraction_int8
from merlin.verify.lean import differential as diff

pytestmark = pytest.mark.usefixtures("lean_oracle")


def function_name(op):
    return op.properties["sym_name"].data


def with_bridges(module, input_types, output_type, depth):
    from xdsl.dialects.builtin import SymbolRefAttr

    assert depth >= 1
    forward = next(op for op in module.walk() if op.name == "func.func" and function_name(op) == "forward")
    call = next(op for op in forward.walk() if op.name == "func.call")
    previous = "kernel"
    arguments = ", ".join(f"%x{i}: {ty}" for i, ty in enumerate(input_types))
    operands = ", ".join(f"%x{i}" for i in range(len(input_types)))
    signature = ", ".join(input_types)
    for index in range(1, depth):
        name = f"bridge_{index}"
        parsed = parse_mlir_text(f"""func.func @{name}({arguments}) -> {output_type} {{
          %r = func.call @{previous}({operands}) : ({signature}) -> {output_type}
          return %r : {output_type}
        }}""")
        module.body.block.add_op(next(iter(parsed.body.block.ops)).clone())
        previous = name
    call.properties["callee"] = SymbolRefAttr(previous)
    return module


def quantizer(width=64, shape=(2, 8), *, depth=1, dequantize=False):
    module = source(width, shape, wrapper=True, dequantize=dequantize)
    dims = "".join(f"{size}x" for size in shape)
    ty = f"tensor<{dims}f32>"
    output = ty if dequantize else f"tensor<{dims}i8>"
    return with_bridges(module, [ty], output, depth)


def qualify(document, work):
    proc = subprocess.run(
        [
            "lake",
            "env",
            "lean",
            "--run",
            str(repo_root() / "merlin/tests/fixtures/lean_w8a8_called.lean"),
            str(document),
        ],
        cwd=repo_root() / "merlin/lean",
        text=True,
        capture_output=True,
        check=False,
    )
    assert proc.returncode in (0, 1), proc.stdout + proc.stderr
    result = json.loads(proc.stdout)
    assert result["scope"] == "provisional_finite_input_initialized_internal_calls"
    assert result["input_domain"] == "typed_initialized_inputs_with_finite_f32_values"
    assert result["floating_domain"] == "byte_extension_of_base_satisfying_explicit_primitive_laws"
    assert result["primitive_model_laws_verified"] is False
    assert result["whole_pass_verified"] is False and result["policy_reviewed"] is False
    assert result["theorem"] == "MerlinVerify.Check.PolicyW8A8Called.check_sound"
    assert (proc.returncode == 0) == result["conditional_source_initialized"]
    (work / "called-report.json").write_text(json.dumps(result, indent=2))
    return result


@pytest.mark.parametrize("width", [32, 64])
@pytest.mark.parametrize("shape", [(), (2, 8), (0, 3)])
@pytest.mark.parametrize("depth", [1, 3])
@pytest.mark.parametrize("dequantize", [False, True])
def test_quantizer_call_chain_has_initialized_execution(tmp_path, lean_tools, width, shape, depth, dequantize):
    document = exported(
        quantizer(width, shape, depth=depth, dequantize=dequantize),
        tmp_path,
        lean_tools,
        generic=True,
        allow_unregistered=True,
    )
    report = qualify(document, tmp_path)
    assert report["conditional_source_initialized"]
    assert len(report["functions"]) == depth + 1
    assert report["module_fuel"] == depth + 2


@pytest.mark.parametrize("width", [32, 64])
@pytest.mark.parametrize("depth", [1, 4])
def test_checked_call_chain_agrees_with_native_mlir(tmp_path, lean_tools, fidelity_tools, width, depth):
    document = exported(quantizer(width, depth=depth), tmp_path, lean_tools, generic=True, allow_unregistered=True)
    assert qualify(document, tmp_path)["conditional_source_initialized"]
    _, arguments = samples_for(width, (2, 8))
    actual = execute(document, [[sample[0]] for sample in arguments], tmp_path)
    native_work = tmp_path / "native"
    native_work.mkdir()
    native_document = exported(source_program(width, (2, 8)), native_work, lean_tools)
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
    assert [sample["values"][0]["bits"] for sample in actual] == native["results"]


def numerical_source(*, data="identity", scale="0.1"):
    data_body = (
        "return %x : tensor<2xf32>"
        if data == "identity"
        else """
      %nan = arith.constant 0x7FC00001 : f32
      %r = tensor.splat %nan : tensor<2xf32>
      return %r : tensor<2xf32>"""
    )
    return parse_mlir_text(f"""module {{
      func.func @data(%x: tensor<2xf32>) -> tensor<2xf32> {{ {data_body} }}
      func.func @scale() -> f32 {{
        %r = arith.constant {scale} : f32
        return %r : f32
      }}
      func.func @forward(%x: tensor<2xf32>) -> tensor<2xi8> {{
        %d = func.call @data(%x) : (tensor<2xf32>) -> tensor<2xf32>
        %sv = func.call @scale() : () -> f32
        %s = tensor.splat %sv : tensor<f32>
        %zv = arith.constant 0 : i64
        %z = tensor.splat %zv : tensor<i64>
        %q = "quant_ext.quantize_per_tensor"(%d, %s, %z)
          <{{quant_min = -128 : i64, quant_max = 127 : i64, output_dtype = "int8"}}>
          : (tensor<2xf32>, tensor<f32>, tensor<i64>) -> tensor<2xi8>
        return %q : tensor<2xi8>
      }}
    }}""")


def test_callee_preserves_finite_and_produces_normal_facts(tmp_path, lean_tools):
    document = exported(numerical_source(), tmp_path, lean_tools, generic=True, allow_unregistered=True)
    report = qualify(document, tmp_path)
    assert report["conditional_source_initialized"]
    facts = {entry["function"]: entry["output_classes"] for entry in report["functions"]}
    assert facts["data"] == ["MerlinVerify.Check.FiniteFacts.Class.finite"]
    assert facts["scale"] == ["MerlinVerify.Check.FiniteFacts.Class.normal"]


@pytest.mark.parametrize("scalar", [False, True])
def test_normal_caller_facts_satisfy_a_finite_callee_input(tmp_path, lean_tools, scalar):
    ty = "f32" if scalar else "tensor<2xf32>"
    producer = (
        "%x = arith.constant 0.1 : f32"
        if scalar
        else """
      %s = arith.constant 0.1 : f32
      %x = tensor.splat %s : tensor<2xf32>"""
    )
    program = parse_mlir_text(f"""module {{
      func.func @identity(%x: {ty}) -> {ty} {{ return %x : {ty} }}
      func.func @forward() -> {ty} {{
        {producer}
        %r = func.call @identity(%x) : ({ty}) -> {ty}
        return %r : {ty}
      }}
    }}""")
    document = exported(program, tmp_path, lean_tools, generic=True, allow_unregistered=True)
    report = qualify(document, tmp_path)
    assert report["conditional_source_initialized"]
    assert all(
        entry["output_classes"] == ["MerlinVerify.Check.FiniteFacts.Class.finite"] for entry in report["functions"]
    )


@pytest.mark.parametrize("data,scale", [("nan", "0.1"), ("identity", "-0.1"), ("identity", "0x7FC00001")])
def test_initialized_callee_does_not_supply_missing_numerical_facts(tmp_path, lean_tools, data, scale):
    document = exported(
        numerical_source(data=data, scale=scale), tmp_path, lean_tools, generic=True, allow_unregistered=True
    )
    report = qualify(document, tmp_path)
    assert not report["conditional_source_initialized"]
    assert {entry["function"]: entry["initialized"] for entry in report["functions"]} == {
        "data": True,
        "scale": True,
        "forward": False,
    }


@pytest.mark.parametrize("kind", ["direct", "mutual", "external"])
def test_recursive_and_external_calls_cannot_bootstrap_contracts(tmp_path, lean_tools, kind):
    if kind == "external":
        helpers, target = "func.func private @unknown(tensor<2xf32>) -> tensor<2xf32>", "unknown"
    elif kind == "mutual":
        helpers, target = (
            """func.func @other(%x: tensor<2xf32>) -> tensor<2xf32> {
          %r = func.call @forward(%x) : (tensor<2xf32>) -> tensor<2xf32>
          return %r : tensor<2xf32>
        }""",
            "other",
        )
    else:
        helpers, target = "", "forward"
    program = parse_mlir_text(f"""module {{ {helpers}
      func.func @forward(%x: tensor<2xf32>) -> tensor<2xf32> {{
        %r = func.call @{target}(%x) : (tensor<2xf32>) -> tensor<2xf32>
        return %r : tensor<2xf32>
      }}
    }}""")
    document = exported(program, tmp_path, lean_tools, generic=True, allow_unregistered=True)
    assert not qualify(document, tmp_path)["conditional_source_initialized"]


@pytest.mark.parametrize("failure", ["poison", "ub", "nonfinite_argument"])
def test_successful_types_and_call_hints_cannot_supply_a_missing_contract(tmp_path, lean_tools, failure):
    if failure == "nonfinite_argument":
        helpers = "func.func @kernel(%x: tensor<2xf32>) -> tensor<2xf32> { return %x : tensor<2xf32> }"
        prefix = "%nan = arith.constant 0x7FC00001 : f32\n%x = tensor.splat %nan : tensor<2xf32>"
        call = "%r = func.call @kernel(%x) : (tensor<2xf32>) -> tensor<2xf32>"
    else:
        body = "%e = tensor.empty() : tensor<2xf32>\nreturn %e : tensor<2xf32>"
        if failure == "ub":
            body = """%false = arith.constant false
              cf.assert %false, "not a source precondition"
              %zero = arith.constant 0.0 : f32
              %e = tensor.splat %zero : tensor<2xf32>
              return %e : tensor<2xf32>"""
        helpers = f"func.func @kernel() -> tensor<2xf32> {{ {body} }}"
        prefix, call = "", "%r = func.call @kernel() : () -> tensor<2xf32>"
    program = parse_mlir_text(f"""module {{ {helpers}
      func.func @forward() -> tensor<2xf32> {{ {prefix}
        {call}
        return %r : tensor<2xf32>
      }}
    }}""")
    document = exported(program, tmp_path, lean_tools, generic=True, allow_unregistered=True)
    data = json.loads(document.read_text())
    for function in data["module"]["ops"]:
        function["attrs"]["prov.verified"] = {"kind": "unit"}
    document.write_text(json.dumps(data))
    assert not qualify(document, tmp_path)["conditional_source_initialized"]


@pytest.mark.parametrize("change", ["missing", "result_type", "metadata", "nested_symbol", "regions"])
def test_actual_call_signature_and_metadata_are_checked(tmp_path, lean_tools, change):
    document = exported(quantizer(), tmp_path, lean_tools, generic=True, allow_unregistered=True)
    data = json.loads(document.read_text())
    forward = next(
        op
        for op in data["module"]["ops"]
        if op["props"].get("sym_name", op["attrs"].get("sym_name", {})).get("value") == "forward"
    )
    call = forward["regions"][0]["blocks"][0]["ops"][0]
    table = call["props"] if "callee" in call["props"] else call["attrs"]
    if change == "missing":
        table["callee"]["root"] = "missing"
    elif change == "result_type":
        call["results"][0]["type"]["element"] = {"kind": "float", "format": "f32"}
    elif change == "metadata":
        call["attrs"]["unreviewed_call_mode"] = {"kind": "unit"}
    elif change == "nested_symbol":
        table["callee"]["nested"] = ["child"]
    else:
        call["regions"] = [{"blocks": []}]
    document.write_text(json.dumps(data))
    assert not qualify(document, tmp_path)["conditional_source_initialized"]


def test_callee_definition_order_does_not_choose_its_contract(tmp_path, lean_tools):
    document = exported(quantizer(depth=3), tmp_path, lean_tools, generic=True, allow_unregistered=True)
    data = json.loads(document.read_text())
    data["module"]["ops"].reverse()
    document.write_text(json.dumps(data))
    assert qualify(document, tmp_path)["conditional_source_initialized"]


@pytest.mark.parametrize("width", [32, 64])
def test_real_w8a8_rewrite_inside_a_callee_has_policy_refinement(tmp_path, lean_tools, width):
    import numpy as np
    from xdsl.dialects.builtin import StringAttr

    before = policy_source(width=width)
    kernel = next(op for op in before.walk() if op.name == "func.func")
    kernel.properties["sym_name"] = StringAttr("kernel")
    wrapper = parse_mlir_text("""func.func @forward(%a: tensor<2x3xf32>, %b: tensor<3x4xf32>) -> tensor<2x4xf32> {
      %r = func.call @kernel(%a, %b) : (tensor<2x3xf32>, tensor<3x4xf32>) -> tensor<2x4xf32>
      return %r : tensor<2x4xf32>
    }""")
    before.body.block.add_op(next(iter(wrapper.body.block.ops)).clone())
    with_bridges(before, ["tensor<2x3xf32>", "tensor<3x4xf32>"], "tensor<2x4xf32>", 3)
    after = before.clone()
    assert lower_contraction_int8(after) == 1
    result, documents = policy(before, after, tmp_path, lean_tools)
    assert result["conditional_initialized_policy_refinement"]
    assert len(result["functions"]) == 4
    left, right = recipe_values()
    expected = reused_reference(left, right)
    actual = execute(documents[1], [[[str(word(x)) for x in values.flat] for values in (left, right)]], tmp_path)
    assert actual[0]["status"] == "ok"
    assert actual[0]["values"][0]["bits"] == [str(x) for x in expected.view(np.uint32).flat]
