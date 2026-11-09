"""A torchAO integer source is preserved and graded against its own framework output."""

# ruff: noqa: E501 -- the MLIR fixture retains the exact generic operation syntax.

from __future__ import annotations

import hashlib
import json
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
    }) {prov.op = "int_matmul", prov.aten = "aten._int_mm.default", prov.int_mm_lhs_dtype = "int8", prov.int_mm_rhs_dtype = "int8"} : (tensor<4x19xi8>, tensor<19x8xi8>, tensor<4x8xi32>) -> tensor<4x8xi32>
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
    assert contract["activation_quantization"] == "explicit_source_graph"
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
        (_MODEL.replace("value = 0 : i32", "value = 1 : i32"), "without a proven zero initializer"),
    ],
)
def test_source_contract_rejects_wrong_scheme_or_arithmetic(tmp_path, source, reason):
    with pytest.raises(ValueError, match=reason):
        inspect_source_w8a8(_bundle(tmp_path, source))


def test_source_contract_rejects_opaque_affine_with_missing_numerical_parameters(tmp_path):
    source = _MODEL.replace(
        "  func.func @forward",
        "  func.func private @torchao_quantize_affine_default(tensor<4x19xf32>, tensor<4xf32>) -> tensor<4x19xi8>\n"
        "  func.func @forward",
    ).replace(
        "    func.return %result",
        "    %x = tensor.empty() : tensor<4x19xf32>\n"
        "    %s = tensor.empty() : tensor<4xf32>\n"
        "    %unused = func.call @torchao_quantize_affine_default(%x, %s) : "
        "(tensor<4x19xf32>, tensor<4xf32>) -> tensor<4x19xi8>\n"
        "    func.return %result",
    )
    with pytest.raises(ValueError, match="opaque TorchAO affine calls"):
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


def test_source_w8a8_host_cache_is_strict_and_preserves_source_reference(tmp_path, monkeypatch):
    from merlin import compile_cli
    from merlin.compile import source_quantization
    from merlin.mining import registry
    from merlin.runtime import dispatch_runtime

    bundle = _bundle(tmp_path)
    for name in ("weights.safetensors", "weights.safetensors.manifest.json", "inputs.npz", "input_order.json"):
        (bundle / name).write_bytes(b"fixture")
    golden = np.array([1.0, -2.0, 3.0, 4.0], dtype=np.float32)
    np.save(bundle / "golden.npy", golden)
    cache = tmp_path / "cache"
    cache.mkdir()
    contract = source_quantization.inspect_source_w8a8(bundle)
    contract["capture_receipt"]["status"] = "verified_materialized"
    monkeypatch.setattr(source_quantization, "inspect_source_w8a8", lambda _path: contract)
    monkeypatch.setattr(registry, "load_rvv_package", lambda _path: SimpleNamespace(is_int8=True, backend="rvv"))
    seen = {}
    evidence = {
        "schema": "merlin.host_cache.v1",
        "mode": "cache_only",
        "cache_dir": str(cache),
        "identity": {"compiler_sha256": "a" * 64},
        "loaded": [{"key": "a" * 16, "path": str(cache / "model_host.so"), "sha256": "b" * 64, "bytes": 1}],
        "loaded_unique": 1,
        "miss_count": 0,
        "verified_after_execution": True,
    }

    def run_model(_bundle, _work, **kwargs):
        seen.update(kwargs)
        return {"output": golden, "host_kernels_ran": 1, "host_contractions_ran": 1, "host_cache": evidence}

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
        host_cache_dir=cache,
    )
    assert seen["cache_dir"] == cache
    assert seen["cache_only"] is True
    assert seen["int8_compute"] is False
    assert result["status"] == "verified"
    assert result["host_cache"] == evidence
    assert result["reference"]["kind"] == "torchao_w8a8_framework"
    assert result["reference"]["sha256"] == hashlib.sha256((bundle / "golden.npy").read_bytes()).hexdigest()
    assert result["source_quantization"]["source_mlir_sha256"] == contract["source_mlir_sha256"]
    assert result["source_quantization"]["capture_receipt"]["status"] == "verified_materialized"
    assert result["verify"]["tier_ok"] == "w8a8"

    def unverified_cache(_bundle, _work, **_kwargs):
        return {"output": golden, "host_cache": {**evidence, "verified_after_execution": False}}

    monkeypatch.setattr(dispatch_runtime, "run_model", unverified_cache)
    with pytest.raises(ValueError, match="exact loaded-kernel provenance"):
        compile_cli.compile_rvv(
            "fixture",
            "int8_w8a8",
            run="host",
            verify=True,
            package="fixture-package",
            auto_capture=False,
            timeout=10,
            capture_bundle=bundle,
            host_cache_dir=cache,
        )

    def missing_kernel(_bundle, _work, **_kwargs):
        raise FileNotFoundError("host kernel cache misses 1 required kernel")

    monkeypatch.setattr(dispatch_runtime, "run_model", missing_kernel)
    with pytest.raises(FileNotFoundError, match="cache misses 1"):
        compile_cli.compile_rvv(
            "fixture",
            "int8_w8a8",
            run="host",
            verify=True,
            package="fixture-package",
            auto_capture=False,
            timeout=10,
            capture_bundle=bundle,
            host_cache_dir=cache,
        )


@pytest.mark.parametrize("extra", [["--run", "spike"], ["--no-verify"], ["--dtype", "fp32"]])
def test_cli_host_cache_rejects_non_source_host_routes(tmp_path, extra):
    from merlin.compile import command
    from merlin.compile_cli import _RUNS, _RVV_DTYPES

    cache = tmp_path / "cache"
    cache.mkdir()
    parser = command.parser(_RVV_DTYPES, _RUNS, "k1", ("k1", "zephyr", "spike", "verilator"))
    args = parser.parse_args(
        [
            "--workload",
            "fixture",
            "--target",
            "rvv",
            "--dtype",
            "int8_w8a8",
            "--run",
            "host",
            "--capture-bundle",
            str(tmp_path),
            "--host-cache-dir",
            str(cache),
            *extra,
        ]
    )
    with pytest.raises(SystemExit):
        command.validate(parser, args)


def test_cli_host_cache_rejects_symlink_root(tmp_path):
    from merlin.compile import command
    from merlin.compile_cli import _RUNS, _RVV_DTYPES

    cache = tmp_path / "cache"
    cache.mkdir()
    link = tmp_path / "cache-link"
    link.symlink_to(cache, target_is_directory=True)
    parser = command.parser(_RVV_DTYPES, _RUNS, "k1", ("k1", "zephyr", "spike", "verilator"))
    args = parser.parse_args(
        [
            "--workload",
            "fixture",
            "--dtype",
            "int8_w8a8",
            "--run",
            "host",
            "--capture-bundle",
            str(tmp_path),
            "--host-cache-dir",
            str(link),
        ]
    )
    with pytest.raises(SystemExit):
        command.validate(parser, args)


def test_cli_host_cache_miss_is_structured_error_json(tmp_path, monkeypatch, capsys):
    from merlin import compile_cli
    from merlin.runtime import dispatch_runtime

    cache = tmp_path / "cache"
    cache.mkdir()

    class HostCacheMiss(Exception):
        def __init__(self):
            super().__init__("host cache miss_count=1 before execution")
            self.cache_dir = str(cache)
            self.identity = {"sha256": "a" * 64}
            self.miss_count = 1
            self.missing = [{"key": "b" * 16, "symbol": "forward_kernel_0"}]

    monkeypatch.setattr(dispatch_runtime, "HostCacheMiss", HostCacheMiss, raising=False)

    def miss(_args, _run, _selection):
        raise HostCacheMiss()

    monkeypatch.setattr(compile_cli, "_dispatch", miss)
    code = compile_cli.main(
        [
            "--workload",
            "fixture",
            "--target",
            "rvv",
            "--dtype",
            "int8_w8a8",
            "--run",
            "host",
            "--capture-bundle",
            str(tmp_path),
            "--host-cache-dir",
            str(cache),
            "--json",
        ]
    )
    assert code == 1
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "error"
    assert report["host_cache"]["miss_count"] == 1
    assert report["host_cache"]["loaded"] == []


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
