"""Source-fidelity admission for the captured SpectFormer patch convolution."""

from __future__ import annotations

from pathlib import Path

import pytest

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.torchao_cpu_patch_conv import TORCH_SOURCE_COMMIT, rewrite_torch216_avx2_patch_conv

SOURCE = Path(__file__).parent / "data" / "torchao_patch_conv.mlir"


def source():
    return SOURCE.read_text()


def test_exact_patch_conv_rewrites_bias_and_hwc_fma():
    module = parse_mlir_text(source())
    assert rewrite_torch216_avx2_patch_conv(module, source_commit=TORCH_SOURCE_COMMIT, cpu_capability="AVX2") == 1
    module.verify()
    ops = list(module.walk())
    assert sum(op.name == "linalg.matmul" for op in ops) == 0
    assert sum(op.name == "math.fma" for op in ops) == 1
    assert sum("merlin.torchao_cpu_patch_conv" in op.attributes for op in ops) == 1
    assert all(op.name != "arith.addf" for op in ops)
    assert sum(op.name == "linalg.generic" for op in ops) == 3  # gather, bias seed, fused contraction


@pytest.mark.parametrize("commit,capability", [("wrong", "AVX2"), (TORCH_SOURCE_COMMIT, "AVX512")])
def test_patch_conv_rejects_unpinned_torch_cpu(commit, capability):
    with pytest.raises(ValueError, match="exact Torch 2.16"):
        rewrite_torch216_avx2_patch_conv(parse_mlir_text(source()), source_commit=commit, cpu_capability=capability)


def test_patch_conv_rejects_changed_gather_layout():
    changed = source().replace("((d4 * 16) + d1)", "((d4 * 16) + d2)", 1)
    with pytest.raises(ValueError, match="im2col gather"):
        rewrite_torch216_avx2_patch_conv(
            parse_mlir_text(changed), source_commit=TORCH_SOURCE_COMMIT, cpu_capability="AVX2"
        )


def test_patch_conv_rejects_changed_source_scheme():
    changed = source().replace("int8_dyn_act_int8_weight", "other", 1)
    with pytest.raises(ValueError, match="TorchAO W8A8"):
        rewrite_torch216_avx2_patch_conv(
            parse_mlir_text(changed), source_commit=TORCH_SOURCE_COMMIT, cpu_capability="AVX2"
        )


def test_default_preprocessing_keeps_source_arithmetic(monkeypatch):
    from merlin.llvmlower.passes_xdsl import preprocess_text

    monkeypatch.delenv("MERLIN_TORCHAO_CPU_AVX2_PATCH_CONV", raising=False)
    out, stats = preprocess_text(source())
    assert "torchao_cpu_avx2_patch_convs" not in stats
    assert "linalg.matmul" in out
    assert "math.fma" not in out


def test_selected_rewrite_keeps_source_transform_map(monkeypatch):
    from merlin.llvmlower.passes_xdsl import preprocess_text_with_transform_map

    monkeypatch.setenv("MERLIN_TORCHAO_CPU_AVX2_PATCH_CONV", "1")
    monkeypatch.setenv("MERLIN_TORCHAO_CPU_SOURCE_COMMIT", TORCH_SOURCE_COMMIT)
    monkeypatch.setenv("MERLIN_TORCHAO_CPU_CAPABILITY", "AVX2")
    out, stats, receipt = preprocess_text_with_transform_map(source())
    assert stats["torchao_cpu_avx2_patch_convs"] == 1
    assert receipt["schema"] == "source_transform_map_v1"
    assert "math.fma" in out


def test_opt_in_lowering_selects_llvm_fma(monkeypatch, tmp_path):
    from merlin.llvmlower import toolchain
    from merlin.llvmlower.lower import lower_model

    if not toolchain.available():
        pytest.skip("m2m/clang toolchain unavailable")
    monkeypatch.setenv("MERLIN_TORCHAO_CPU_AVX2_PATCH_CONV", "1")
    monkeypatch.setenv("MERLIN_TORCHAO_CPU_SOURCE_COMMIT", TORCH_SOURCE_COMMIT)
    monkeypatch.setenv("MERLIN_TORCHAO_CPU_CAPABILITY", "AVX2")
    result = lower_model(source(), tmp_path, targets=("riscv",))
    assert result.stats["torchao_cpu_avx2_patch_convs"] == 1
    assert "@llvm.fma.f32" in result.ll_path.read_text()
    assert result.riscv_obj.is_file()
