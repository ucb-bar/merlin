"""Pinned source CPU LayerNorm rewrite, refusals, and generated host arithmetic."""

import hashlib
import importlib
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from merlin.common.paths import repo_root
from merlin.llvmlower.abi import HostModel
from merlin.llvmlower.codegen import mlir_runtime_c
from merlin.llvmlower.lower import lower_model_file
from merlin.llvmlower.passes_xdsl import preprocess_text
from merlin.llvmlower.toolchain import clang
from merlin.llvmlower.torchao_cpu_layer_norm import TORCH_SOURCE_COMMIT

SOURCE_OUTPUT_SHA256 = "3f59889aad278b3eceb3cfee37905db798a19af3de4e95c1e2cd4ea281f4285d"


def _source() -> str:
    return (repo_root() / "merlin/tests/ir/torch216_ln256_source.mlir").read_text()


def _select(monkeypatch):
    monkeypatch.setenv("MERLIN_TORCHAO_CPU_AVX2_LAYER_NORM", "1")
    monkeypatch.setenv("MERLIN_TORCHAO_CPU_SOURCE_COMMIT", TORCH_SOURCE_COMMIT)
    monkeypatch.setenv("MERLIN_TORCHAO_CPU_CAPABILITY", "AVX2")


def test_default_keeps_captured_arithmetic(monkeypatch):
    monkeypatch.delenv("MERLIN_TORCHAO_CPU_AVX2_LAYER_NORM", raising=False)
    monkeypatch.delenv("MERLIN_PASSES", raising=False)
    output, stats = preprocess_text(_source())
    assert "torchao_cpu_avx2_layer_norms" not in stats
    assert "merlin.torchao_cpu_layer_norm" not in output
    assert output.count("linalg.reduce") == 2


def test_rewrites_complete_region_and_ignores_region_name(monkeypatch):
    _select(monkeypatch)
    output, stats = preprocess_text(_source().replace("layer_norm_0", "independent_region"))
    assert stats["torchao_cpu_avx2_layer_norms"] == 1
    assert "merlin.torchao_cpu_layer_norm" in output
    assert "linalg.reduce" not in output
    again, second_stats = preprocess_text(output)
    assert second_stats["torchao_cpu_avx2_layer_norms"] == 0
    assert again == output


@pytest.mark.parametrize(
    ("variable", "value", "message"),
    [
        ("MERLIN_TORCHAO_CPU_SOURCE_COMMIT", "wrong", "exact captured"),
        ("MERLIN_TORCHAO_CPU_CAPABILITY", "AVX512", "AVX2 dispatch"),
    ],
)
def test_refuses_wrong_source(monkeypatch, variable, value, message):
    _select(monkeypatch)
    monkeypatch.setenv(variable, value)
    with pytest.raises(ValueError, match=message):
        preprocess_text(_source())


def test_refuses_wrong_quantization(monkeypatch):
    _select(monkeypatch)
    with pytest.raises(ValueError, match="W8A8 capture"):
        preprocess_text(
            _source().replace('prov.quantization = "int8_dyn_act_int8_weight"', 'prov.quantization = "fp32"')
        )


def test_refuses_conflicting_capture_metadata(monkeypatch):
    _select(monkeypatch)
    source = _source().replace(
        'prov.quantization = "int8_dyn_act_int8_weight"',
        'prov.quantization = "int8_dyn_act_int8_weight", prov.torch_source_commit = "wrong"',
    )
    with pytest.raises(ValueError, match="conflicting captured"):
        preprocess_text(source)


def test_refuses_wrong_width_and_malformed_region(monkeypatch):
    _select(monkeypatch)
    narrow = _source().replace("1x196x256xf32", "1x196x128xf32")
    with pytest.raises(ValueError, match="width-256"):
        preprocess_text(narrow)
    half = _source().replace("1x196x256xf32", "1x196x256xbf16")
    with pytest.raises(ValueError, match="f32"):
        preprocess_text(half)
    source = _source()
    head, tail = source.rsplit("arith.addf", 1)
    with pytest.raises(ValueError, match="affine output"):
        preprocess_text(head + "arith.subf" + tail)
    with pytest.raises(ValueError, match="operation provenance"):
        preprocess_text(source.replace('prov.op = "layer_norm"', 'prov.op = "other"', 1))


def _compiled_model(tmp_path, monkeypatch):
    _select(monkeypatch)
    source, stats = preprocess_text(_source())
    assert stats["torchao_cpu_avx2_layer_norms"] == 1
    path = tmp_path / "source.mlir"
    path.write_text(source)
    lowered = lower_model_file(path, tmp_path / "lower", targets=(), textual=True)
    compiler = clang()
    if not compiler.is_file():
        pytest.skip("configured LLVM clang is unavailable")
    obj, library = tmp_path / "layer_norm.o", tmp_path / "layer_norm.so"
    subprocess.run([str(compiler), "-O2", "-fPIC", "-c", str(lowered.ll_path), "-o", str(obj)], check=True)
    subprocess.run(["cc", "-shared", "-fPIC", str(obj), str(mlir_runtime_c()), "-lm", "-o", str(library)], check=True)
    return HostModel.load(str(library))


def _run(model, x, gamma, beta):
    output = np.zeros_like(x)
    model([(v.ctypes.data, v.shape) for v in (x, gamma, beta, output)])
    return output


def test_generated_host_matches_pinned_source_words(tmp_path, monkeypatch):
    model = _compiled_model(tmp_path, monkeypatch)
    x = np.random.default_rng(12).standard_normal((1, 196, 256)).astype(np.float32)
    gamma = np.random.default_rng(5).standard_normal(256).astype(np.float32)
    beta = np.random.default_rng(6).standard_normal(256).astype(np.float32)
    actual = _run(model, x, gamma, beta)
    assert hashlib.sha256(actual.tobytes()).hexdigest() == SOURCE_OUTPUT_SHA256


def test_generated_host_matches_trained_and_holdout_torchao(tmp_path, monkeypatch):
    paths = {
        key: os.environ.get(key)
        for key in (
            "MERLIN_SPECTFORMER_TRAINED_TAP",
            "MERLIN_SPECTFORMER_HOLDOUT_TAP",
            "MERLIN_SPECTFORMER_MODEL2MLIR",
            "MERLIN_SPECTFORMER_DIR",
            "MERLIN_SPECTFORMER_CKPT",
        )
    }
    if not all(paths.values()):
        pytest.skip("same-input SpectFormer source tap paths not configured")
    torch = pytest.importorskip("torch")
    if torch.version.git_version != TORCH_SOURCE_COMMIT or torch.backends.cpu.get_cpu_capability() != "AVX2":
        pytest.skip("captured Torch CPU source wheel unavailable")
    sys.path[:0] = [
        paths["MERLIN_SPECTFORMER_MODEL2MLIR"],
        str(Path(paths["MERLIN_SPECTFORMER_MODEL2MLIR"]) / "workloads/spectformer"),
    ]
    monkeypatch.setenv("SPECTFORMER_DIR", paths["MERLIN_SPECTFORMER_DIR"])
    monkeypatch.setenv("SPECTFORMER_CKPT", paths["MERLIN_SPECTFORMER_CKPT"])
    get_model_and_inputs = importlib.import_module("loader").get_model_and_inputs
    pipeline = importlib.import_module("m2m.capture.torchao_pipeline")
    torch.manual_seed(0)
    np.random.seed(0)
    network, inputs = get_model_and_inputs()
    network = pipeline.apply_quantization(
        network.eval(),
        pipeline.QuantizationConfig(scheme="int8_dyn_act_int8_weight"),
        example_inputs=tuple(inputs),
    )
    norm = network.blocks[0].norm2
    gamma = np.ascontiguousarray(norm.weight.detach().numpy())
    beta = np.ascontiguousarray(norm.bias.detach().numpy())
    model = _compiled_model(tmp_path, monkeypatch)
    for key in ("MERLIN_SPECTFORMER_TRAINED_TAP", "MERLIN_SPECTFORMER_HOLDOUT_TAP"):
        x = np.ascontiguousarray(np.load(paths[key])["view_2"])
        with torch.no_grad():
            expected = norm(torch.from_numpy(x)).numpy()
        actual = _run(model, x, gamma, beta)
        np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
