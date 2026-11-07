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
    assert rewrite_torch216_avx2_patch_conv(module, source_commit=TORCH_SOURCE_COMMIT, cpu_capability="AVX2") == 0


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


def test_normalization_recipe_requires_patch_source_and_cpu():
    from merlin.runtime.dispatch_transform_audit import _validate_normalization_recipe

    base = {
        "int8_compute": False,
        "quant_passes": None,
        "prequant_gather": False,
        "selection_policy": "all",
        "torchao_cpu_avx2_patch_conv": True,
        "torch_source_commit": TORCH_SOURCE_COMMIT,
        "torch_cpu_capability": "AVX2",
    }
    assert _validate_normalization_recipe(base)["torchao_cpu_avx2_patch_conv"] is True
    for key in ("torch_source_commit", "torch_cpu_capability"):
        with pytest.raises(ValueError, match="invalid options"):
            _validate_normalization_recipe({**base, key: ""})


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


@pytest.mark.parametrize(
    "passes",
    [
        "torchao-cpu-avx2-patch-conv",
        "torchao-cpu-avx2-layer-norm,torchao-cpu-avx2-patch-conv",
    ],
)
def test_run_model_rewrites_before_outline_and_standalone_kernels_skip(monkeypatch, tmp_path, passes):
    """Exercise run_model's actual normalizer, outline, audit replay, and kernel seam."""
    from merlin.llvmlower.kernel_backend import extract_kernel
    from merlin.llvmlower.passes_xdsl import preprocess_text
    from merlin.runtime import dispatch_runtime as runtime
    from merlin.xdsl_dialects._common import text

    monkeypatch.setenv("MERLIN_PASSES", passes)
    monkeypatch.setenv("MERLIN_TORCHAO_CPU_SOURCE_COMMIT", TORCH_SOURCE_COMMIT)
    monkeypatch.setenv("MERLIN_TORCHAO_CPU_CAPABILITY", "AVX2")
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "model.mlir").write_text(source())

    observed = {}
    real_audit = runtime.record_model_transform_audit

    def observe_audit(model_source, workdir, normalized, outlined, **kwargs):
        observed["normalized"] = normalized
        observed["outlined"] = outlined
        observed["recipe"] = kwargs["normalization_recipe"]
        return real_audit(model_source, workdir, normalized, outlined, **kwargs)

    class ReachedBinding(Exception):
        pass

    def stop_at_binding(_model_dir):
        raise ReachedBinding

    monkeypatch.setattr(runtime, "record_model_transform_audit", observe_audit)
    monkeypatch.setattr(runtime, "resolve_forward_args", stop_at_binding)
    with pytest.raises(ReachedBinding):
        runtime.run_model(model_dir, tmp_path / "work", transform_audit=True)

    assert observed["recipe"]["torchao_cpu_avx2_patch_conv"] is True
    assert observed["recipe"]["torchao_cpu_avx2_layer_norm"] is ("torchao-cpu-avx2-layer-norm" in passes)
    assert observed["recipe"]["torch_source_commit"] == TORCH_SOURCE_COMMIT
    assert observed["recipe"]["torch_cpu_capability"] == "AVX2"
    normalized = observed["normalized"]
    ops = list(normalized.walk())
    assert sum(op.name == "math.fma" for op in ops) == 1
    assert not any(op.name == "linalg.matmul" for op in ops)
    assert not any(op.name == "arith.addf" for op in ops)
    outlined = observed["outlined"]
    assert outlined.dispatches
    for dispatch in outlined.dispatches:
        kernel = extract_kernel(outlined.module, dispatch.symbol)
        _, stats = preprocess_text(text(kernel))
        assert stats["torchao_cpu_avx2_patch_convs"] == 0


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
