"""A torchAO integer source is preserved and graded against its own framework output."""

# ruff: noqa: E501 -- the MLIR fixture retains the exact generic operation syntax.

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from merlin.compile.source_quantization import inspect_source_w8a8

_MODEL = """builtin.module attributes {prov.quantization = "int8_dyn_act_int8_weight"} {
  func.func @forward(%a: tensor<4x19xi8>, %b: tensor<19x8xi8>) -> tensor<4x8xi32> {
    %zero = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %init = "tensor.splat"(%zero) : (i32) -> tensor<4x8xi32>
    %result = "linalg.generic"(%a, %b, %init) <{indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%lhs: i8, %rhs: i8, %acc: i32):
      %lhs32 = "arith.extsi"(%lhs) : (i8) -> i32
      %rhs32 = "arith.extsi"(%rhs) : (i8) -> i32
      %product = "arith.muli"(%lhs32, %rhs32) : (i32, i32) -> i32
      %sum = "arith.addi"(%acc, %product) : (i32, i32) -> i32
      "linalg.yield"(%sum) : (i32) -> ()
    }) {prov.op = "int_matmul", prov.aten = "aten._int_mm.default"} : (tensor<4x19xi8>, tensor<19x8xi8>, tensor<4x8xi32>) -> tensor<4x8xi32>
    func.return %result : tensor<4x8xi32>
  }
}"""


def _bundle(tmp_path, source=_MODEL):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "model.mlir").write_text(source, encoding="utf-8")
    return bundle


def test_source_contract_counts_actual_integer_macs(tmp_path):
    contract = inspect_source_w8a8(_bundle(tmp_path))
    assert contract["source_scheme"] == "int8_dyn_act_int8_weight"
    assert contract["matrix_operand_format"] == "signed_int8"
    assert contract["matrix_accumulator_format"] == "int32"
    assert contract["integer_matmuls"] == 1
    assert contract["integer_valid_macs"] == 4 * 8 * 19
    assert contract["quantization_passes"] == "preserve_source"
    assert contract["reference_file"] == "golden.npy"


@pytest.mark.parametrize(
    "source, reason",
    [
        (_MODEL.replace("int8_dyn_act_int8_weight", "int8_weight_only"), "prov.quantization"),
        (_MODEL.replace('"arith.muli"(%lhs32, %rhs32)', '"arith.addi"(%lhs32, %rhs32)'), "malformed"),
    ],
)
def test_source_contract_rejects_wrong_scheme_or_arithmetic(tmp_path, source, reason):
    with pytest.raises(ValueError, match=reason):
        inspect_source_w8a8(_bundle(tmp_path, source))


def test_source_w8a8_host_uses_framework_golden_without_second_quantization(tmp_path, monkeypatch):
    from merlin import compile_cli
    from merlin.mining import registry
    from merlin.runtime import dispatch_runtime

    bundle = _bundle(tmp_path)
    for name in ("weights.safetensors", "weights.safetensors.manifest.json", "inputs.npz", "input_order.json"):
        (bundle / name).write_bytes(b"fixture")
    golden = np.array([1.0, -2.0, 3.0, 4.0], dtype=np.float32)
    np.save(bundle / "golden.npy", golden)
    monkeypatch.setattr(registry, "load_rvv_package", lambda _path: SimpleNamespace(is_int8=True, backend="rvv"))
    monkeypatch.setattr(compile_cli, "host_lane_identity", lambda _path: {"dtype_strategy": "int8_w8a8"})
    seen = {}

    def run_model(_bundle, _work, *, int8_compute, **_kwargs):
        seen["int8_compute"] = int8_compute
        return {"output": golden, "host_kernels_ran": 1, "host_contractions_ran": 1}

    monkeypatch.setattr(dispatch_runtime, "run_model", run_model)
    result = compile_cli.compile_rvv(
        "fixture",
        "int8_w8a8",
        run="host",
        verify=True,
        package="fixture-package",
        auto_capture=False,
        timeout=10,
        capture_bundle=bundle,
    )
    assert seen == {"int8_compute": False}
    assert result["status"] == "verified"
    assert result["verify"]["tier_ok"] == "w8a8"
    assert "fp32_cos" not in result["verify"]
    assert result["reference"]["kind"] == "torchao_w8a8_framework"


def test_cli_accepts_explicit_integer_source_for_matrix_route(tmp_path):
    from merlin.compile import command
    from merlin.compile_cli import _RUNS, _RVV_DTYPES

    parser = command.parser(_RVV_DTYPES, _RUNS, "k1", ("k1", "zephyr", "spike", "verilator"))
    args = parser.parse_args(
        [
            "--workload",
            "fixture",
            "--dtype",
            "int8_w8a8",
            "--target",
            "rvv",
            "--capture-bundle",
            str(tmp_path),
            "--run",
            "spike",
            "--board",
            "board",
            "--matrix-unit",
            "unit",
            "--matrix-config",
            "config",
            "--matrix-support-target",
            "support",
        ]
    )
    command.validate(parser, args)
    assert args.capture_bundle == str(tmp_path)
