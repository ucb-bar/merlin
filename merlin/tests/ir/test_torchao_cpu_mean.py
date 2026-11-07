"""Emitted source CPU mean order through the opt-in xDSL rewrite."""

import subprocess

import numpy as np
import pytest
from xdsl.dialects import arith, builtin, func, tensor
from xdsl.dialects.builtin import DenseArrayBase, FloatAttr, StringAttr, TensorType, f32, i64
from xdsl.dialects.linalg import ops as linalg
from xdsl.ir import Block, Region

from merlin.llvmlower.abi import HostModel
from merlin.llvmlower.codegen import mlir_runtime_c
from merlin.llvmlower.lower import lower_model_file
from merlin.llvmlower.passes_xdsl import preprocess_text, preprocess_text_with_transform_map
from merlin.llvmlower.toolchain import clang
from merlin.llvmlower.torchao_cpu_mean import TORCH_SOURCE_COMMIT
from merlin.xdsl_dialects._common import text


def _source_mean_sum(rows=3):
    input_type, output_type = TensorType(f32, [rows, 2048]), TensorType(f32, [rows])
    body = Block(arg_types=[input_type])
    zero = arith.ConstantOp(FloatAttr(0.0, f32))
    init = tensor.SplatOp(zero.result, [], output_type)
    reduce_body = Block(arg_types=[f32, f32])
    add = arith.AddfOp(reduce_body.args[0], reduce_body.args[1])
    reduce_body.add_ops([add, linalg.YieldOp(add.result)])
    reduce = linalg.ReduceOp(body.args[0], init.result, DenseArrayBase.from_list(i64, [1]), Region(reduce_body))
    reduce.attributes["prov.aten"] = StringAttr("aten.mean.dim")
    body.add_ops([zero, init, reduce, func.ReturnOp(reduce.result[0])])
    module = builtin.ModuleOp(
        [func.FuncOp("forward", ([input_type], [output_type]), Region(body))],
        attributes={"prov.quantization": StringAttr("int8_dyn_act_int8_weight")},
    )
    module.verify()
    return module


def _rows():
    count = 2048
    indices = np.arange(3 * count, dtype=np.int64).reshape(3, count)
    positive = ((indices[0] * 1664525 + 1013904223) % 32749).astype(np.float32) / np.float32(8192)
    signed = ((indices[1] * 22695477 + 1) % 65521).astype(np.float32) / np.float32(4096) - np.float32(8)
    cancel = np.where(indices[2] % 2 == 0, np.float32(1e4), np.float32(-1e4))
    cancel += ((indices[2] * 1103515245 + 12345) % 97).astype(np.float32) / np.float32(64)
    return np.stack((positive, signed, cancel))


def test_host_dispatch_preserves_mean_source_contract(monkeypatch):
    from merlin.llvmlower.kernel_backend import extract_kernel
    from merlin.runtime.dispatch_runtime import _normalize_model_module
    from merlin.xdsl_dialects.lowering.outline import outline_dispatches

    monkeypatch.setenv("MERLIN_TORCHAO_CPU_AVX2_MEAN", "1")
    monkeypatch.setenv("MERLIN_TORCHAO_CPU_SOURCE_COMMIT", TORCH_SOURCE_COMMIT)
    source = _source_mean_sum()
    _normalize_model_module(source, int8_compute=False, quant_passes=None, prequant_gather=False)
    assert "merlin.torchao_cpu_mean" in text(source)
    outlined = outline_dispatches(source)
    assert outlined.module.attributes["prov.quantization"] == source.attributes["prov.quantization"]
    for dispatch in outlined.dispatches:
        kernel = extract_kernel(outlined.module, dispatch.symbol)
        assert kernel.attributes["prov.quantization"] == source.attributes["prov.quantization"]
        _, stats = preprocess_text(text(kernel))
        assert stats["torchao_cpu_avx2_means"] == 0


@pytest.mark.parametrize("optimization", ["-O0", "-O2"])
def test_emitted_avx2_source_order_matches_independent_torch_words(tmp_path, monkeypatch, optimization):
    monkeypatch.setenv("MERLIN_TORCHAO_CPU_AVX2_MEAN", "1")
    monkeypatch.setenv("MERLIN_TORCHAO_CPU_SOURCE_COMMIT", TORCH_SOURCE_COMMIT)
    lowered_text, stats = preprocess_text(text(_source_mean_sum()))
    assert stats["torchao_cpu_avx2_means"] == 1
    assert lowered_text.count("merlin.torchao_cpu_mean") == 1
    source = tmp_path / "source.mlir"
    source.write_text(lowered_text)
    lowered = lower_model_file(source, tmp_path / "lower", targets=(), textual=True)
    obj, lib = tmp_path / "mean.o", tmp_path / "mean.so"
    subprocess.run([str(clang()), optimization, "-fPIC", "-c", str(lowered.ll_path), "-o", str(obj)], check=True)
    if optimization == "-O2":
        subprocess.run(
            [
                str(clang()),
                "-O2",
                "--target=riscv64-unknown-elf",
                "-march=rv64gcv",
                "-mabi=lp64d",
                "-c",
                str(lowered.ll_path),
                "-o",
                str(tmp_path / "mean_rv64.o"),
            ],
            check=True,
        )
    subprocess.run(["cc", "-shared", "-fPIC", str(obj), str(mlir_runtime_c()), "-lm", "-o", str(lib)], check=True)
    inputs = _rows()
    output = np.zeros(3, dtype=np.float32)
    HostModel.load(str(lib))([(x.ctypes.data, x.shape) for x in (inputs, output)])
    # PyTorch 2.16.0.dev20261104 AVX2 torch.from_numpy(inputs).sum(-1),
    # independently frozen from the captured wheel.
    np.testing.assert_array_equal(output.view(np.uint32), (1165998554, 3247528064, 1153425408))


def test_source_commit_must_match(tmp_path, monkeypatch):
    monkeypatch.setenv("MERLIN_TORCHAO_CPU_AVX2_MEAN", "1")
    monkeypatch.setenv("MERLIN_TORCHAO_CPU_SOURCE_COMMIT", "different")
    with pytest.raises(ValueError, match="exact captured"):
        preprocess_text(text(_source_mean_sum()))


def test_source_operation_accounting_is_preserved(monkeypatch):
    monkeypatch.setenv("MERLIN_TORCHAO_CPU_AVX2_MEAN", "1")
    monkeypatch.setenv("MERLIN_TORCHAO_CPU_SOURCE_COMMIT", TORCH_SOURCE_COMMIT)
    lowered_text, stats, source_map = preprocess_text_with_transform_map(text(_source_mean_sum()))
    assert stats["torchao_cpu_avx2_means"] == 1
    assert lowered_text.count("merlin.torchao_cpu_mean") == 1
    assert source_map["operations"]
