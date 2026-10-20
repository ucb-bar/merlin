"""Ordinary host legalization, independent arithmetic reconstruction and mutations."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest
from xdsl.dialects import arith, builtin, cf, func, math
from xdsl.interpreter import Interpreter, InterpreterFunctions, impl, register_impls
from xdsl.interpreters.arith import ArithFunctions
from xdsl.interpreters.linalg import LinalgFunctions
from xdsl.ir import Block, Region

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.automatic_host import optimize_host_quantization, prepare_automatic_host
from merlin.llvmlower.optional_passes import Selection, applied
from merlin.llvmlower.quant_round_check import verify_quantization_rewrite
from merlin.transforms.certificates import ir_text


def _source(width=8, dtype="f32", *, divisor="1.0"):
    return f"""builtin.module {{
      func.func @forward(%a: tensor<3x5x{dtype}>) -> tensor<3x5xi{width}> {{
        %e = tensor.empty() : tensor<3x5xi{width}>
        %r = linalg.generic {{indexing_maps = [affine_map<(d0,d1)->(d0,d1)>,
          affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]}}
          ins(%a:tensor<3x5x{dtype}>) outs(%e:tensor<3x5xi{width}>) {{
        ^bb0(%x:{dtype},%o:i{width}):
          %s = arith.constant {divisor} : {dtype}
          %hi = arith.constant 127.0 : {dtype}
          %lo = arith.constant -127.0 : {dtype}
          %d = arith.divf %x,%s : {dtype}
          %round = math.roundeven %d : {dtype}
          %h = arith.minimumf %round,%hi : {dtype}
          %l = arith.maximumf %h,%lo : {dtype}
          %q = arith.fptosi %l : {dtype} to i{width}
          linalg.yield %q : i{width}
        }} -> tensor<3x5xi{width}>
        func.return %r : tensor<3x5xi{width}>
      }}
    }}"""


@register_impls
class _ScalarConversions(InterpreterFunctions):
    @impl(arith.SelectOp)
    def select(self, interpreter, op, args):
        return (args[1] if args[0] else args[2],)

    @impl(arith.DivfOp)
    def divide(self, interpreter, op, args):
        return (float(np.float32(args[0]) / np.float32(args[1])),)

    @impl(arith.FPToSIOp)
    def truncate(self, interpreter, op, args):
        return (int(args[0]),)

    @impl(arith.SIToFPOp)
    def widen(self, interpreter, op, args):
        return (float(np.float32(args[0])),)

    @impl(arith.NegfOp)
    def negate(self, interpreter, op, args):
        return (-args[0],)

    @impl(math.RoundEvenOp)
    def round_even(self, interpreter, op, args):
        return (float(np.rint(np.float32(args[0]))),)


def _scalar_outputs(module, values):
    body = next(op for op in module.walk() if op.name == "linalg.generic").regions[0]
    interpreter = Interpreter(module)
    for functions in (ArithFunctions(), LinalgFunctions(), _ScalarConversions()):
        interpreter.register_implementations(functions)
    return np.array([interpreter.run_ssacfg_region(body, (float(value), 0))[0] for value in values])


@pytest.mark.parametrize("width", [8, 16, 32])
def test_realized_arithmetic_executes_ties_tails_and_saturation(width):
    source = parse_mlir_text(_source(width))
    original = ir_text(source)
    rewritten, report = optimize_host_quantization(source, numerical_policy="frozen-value-policy")
    values = np.array(
        [-np.inf, -10000, -127.5, -126.5, -2.5, -1.5, -0.5, -0.0, 0.0, 0.5, 1.5, 2.5, 126.5, 127.5, np.inf],
        dtype=np.float32,
    )
    expected = np.clip(np.rint(values), -127, 127).astype(np.int32)
    np.testing.assert_array_equal(_scalar_outputs(source, values), expected)
    np.testing.assert_array_equal(_scalar_outputs(rewritten, values), expected)
    assert ir_text(source) == original
    assert report["rewrites"] == 1 and report["round_sites_after"] == 0
    assert report["checks"][-1]["name"] == "arithmetic_conformance"


@pytest.mark.parametrize("fault", ["tie", "clamp", "sign", "output", "effect"])
def test_checker_reconstructs_actual_output_and_rejects_corruption(fault):
    source = parse_mlir_text(_source())
    rewritten, _ = optimize_host_quantization(source, numerical_policy="frozen-value-policy")
    if fault == "tie":
        op = next(op for op in rewritten.walk() if op.name == "arith.cmpf" and op.predicate.value.data == 1)
        op.properties["predicate"] = builtin.IntegerAttr(3, builtin.i64)
    elif fault == "clamp":
        op = next(op for op in rewritten.walk() if op.name == "arith.minimumf")
        op.operands = (op.operands[0], op.operands[0])
    elif fault == "sign":
        op = next(op for op in rewritten.walk() if op.name == "arith.select")
        op.operands = (op.operands[0], op.operands[2], op.operands[1])
    elif fault == "output":
        op = next(op for op in rewritten.walk() if op.name == "linalg.yield")
        op.operands = (op.parent.args[1],)
    else:
        op = next(op for op in rewritten.walk() if op.name == "linalg.yield")
        op.parent.insert_op_before(builtin.UnregisteredOp.with_name("opaque.observe").create(), op)
    with pytest.raises(ValueError, match="arithmetic|unmodeled"):
        verify_quantization_rewrite(source, rewritten)


def test_zero_scale_and_code_budget_keep_legal_source_route():
    for source, budget in ((_source(divisor="0.0"), 4096), (_source(), 0)):
        module = parse_mlir_text(source)
        rewritten, report = optimize_host_quantization(module, numerical_policy="frozen", max_added_operations=budget)
        assert ir_text(rewritten) == ir_text(module)
        assert report["rewrites"] == 0
    assert report["refused_code_budget"] == 1


def test_unknown_effects_remain_barriers():
    source = _source().replace("%round =", '"opaque.observe"() : () -> ()\n          %round =')
    module = parse_mlir_text(source)
    rewritten, report = optimize_host_quantization(module, numerical_policy="frozen")
    assert ir_text(rewritten) == ir_text(module)
    assert report["refused_effect_bodies"] == 1


def test_ordinary_preparation_selects_pass_after_placement_without_extra_flags(tmp_path):
    from merlin.runtime.backends.zephyr_model import prepare_for_lowering

    source = tmp_path / "model.mlir"
    source.write_text(_source())
    work = tmp_path / "build"
    work.mkdir()
    original = source.read_bytes()
    result, _features = prepare_for_lowering(
        source, work, int8_compute=True, blocking=False, features=frozenset({"respect_captured_quantization_scope"})
    )
    assert source.read_bytes() == original
    actual = parse_mlir_text(result.read_text())
    assert not any(op.name == "math.roundeven" for op in actual.walk())
    report = json.loads((work / "automatic_host/report.json").read_text())
    assert report["rewrites"] == 1


def test_missing_policy_and_explicit_ablation_are_inert(tmp_path):
    source = tmp_path / "model.mlir"
    source.write_text(_source())
    policy = tmp_path / "policy.json"
    assert prepare_automatic_host(source, tmp_path / "missing", policy_path=policy) == source
    policy.write_text(json.dumps({"schema": "merlin.quantization_policy.v1", "selection": "captured_scope"}))
    with applied(Selection.parse("-automatic-host-numerics")):
        assert prepare_automatic_host(source, tmp_path / "disabled", policy_path=policy) == source
    assert not (tmp_path / "disabled").exists()


def test_buffer_generics_keep_their_original_route_and_memory_effects():
    source = _source().replace("tensor<", "memref<")
    source = source.replace(
        "func.func @forward(%a: memref<3x5xf32>) -> memref<3x5xi8>",
        "func.func @forward(%a: memref<3x5xf32>, %e: memref<3x5xi8>)",
    ).replace("%e = tensor.empty() : memref<3x5xi8>", "")
    source = source.replace("%r = linalg.generic", "linalg.generic").replace("} -> memref<3x5xi8>", "}")
    source = source.replace("func.return %r : memref<3x5xi8>", "func.return")
    module = parse_mlir_text(source)
    module.verify()
    rewritten, report = optimize_host_quantization(module, numerical_policy="frozen")
    assert ir_text(rewritten) == ir_text(module)
    assert report["refused_effect_bodies"] == 1 and report["rewrites"] == 0


def test_unknown_tensor_encoding_keeps_the_original_route():
    source = _source().replace("tensor<3x5xi8>", 'tensor<3x5xi8, "opaque.layout">')
    module = parse_mlir_text(source)
    module.verify()
    rewritten, report = optimize_host_quantization(module, numerical_policy="frozen")
    assert ir_text(rewritten) == ir_text(module)
    assert report["refused_effect_bodies"] == 1 and report["rewrites"] == 0


def test_cfg_is_a_conservative_barrier_with_a_verified_unchanged_route():
    module = parse_mlir_text(_source())
    entry, exit = Block(), Block()
    entry.add_op(cf.BranchOp(exit))
    exit.add_op(func.ReturnOp())
    module.body.block.add_op(func.FuncOp("control_flow", ((), ()), Region((entry, exit))))
    rewritten, report = optimize_host_quantization(module, numerical_policy="frozen")
    assert ir_text(rewritten) == ir_text(module)
    assert report["refused_control_flow"] == 1 and report["rewrites"] == 0
    verify_quantization_rewrite(module, module.clone())


def test_documented_environment_switch_and_explicit_pass_override(tmp_path, monkeypatch):
    source = tmp_path / "model.mlir"
    source.write_text(_source())
    policy = tmp_path / "policy.json"
    policy.write_text(
        json.dumps(
            {
                "schema": "merlin.quantization_policy.v1",
                "selection": "captured_scope",
                "prepared_mlir_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            }
        )
    )
    monkeypatch.setenv("MERLIN_AUTOMATIC_HOST_NUMERICS", "0")
    assert prepare_automatic_host(source, tmp_path / "disabled", policy_path=policy) == source
    assert not (tmp_path / "disabled").exists()
    with applied(Selection.parse("automatic-host-numerics")):
        output = prepare_automatic_host(source, tmp_path / "enabled", policy_path=policy)
    assert output != source
    report = json.loads((tmp_path / "enabled/report.json").read_text())
    assert "pipeline.py" in report["library_sources_sha256"]


@pytest.mark.parametrize("source_hash", [None, "0" * 64])
def test_missing_or_stale_policy_binding_preserves_legal_preparation_with_diagnostic(tmp_path, source_hash):
    source = tmp_path / "model.mlir"
    source.write_text(_source())
    policy = tmp_path / "policy.json"
    policy.write_text(
        json.dumps(
            {
                "schema": "merlin.quantization_policy.v1",
                "selection": "captured_scope",
                "prepared_mlir_sha256": source_hash,
            }
        )
    )
    output = prepare_automatic_host(source, tmp_path / "declined", policy_path=policy)
    assert output == source and source.read_text() == _source()
    report = json.loads((tmp_path / "declined/report.json").read_text())
    assert report["rewrites"] == 0 and report["refused_policy_binding"] == 1
    assert report["source_file_sha256"] == report["output_file_sha256"]
