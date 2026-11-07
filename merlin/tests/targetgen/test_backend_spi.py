"""Executable backend loading and scalar independence."""

import subprocess
import sys
from types import SimpleNamespace

import pytest

from merlin.llvmlower import backend_spi


def test_vector_backend_requires_one_compatible_owner(monkeypatch):
    from merlin.targetgen import plugins

    methods = {
        name: lambda *args, **kw: "passes"
        for name in (
            "build_pipeline",
            "baseline_schedule_text",
            "parallel_transform_schedule",
            "vec_pre_schedule",
            "normalize_features",
            "feature_names",
            "apply_schedule",
            "apply_pipeline",
            "apply_cflags",
            "vector_cflags",
            "schedule_path",
            "prepare_build_bundle",
            "spike_cflags",
            "spike_harness_cflags",
            "accum_microkernel_marker",
            "accum_microkernel_runner_source",
            "accum_microkernel_rewrite_source",
            "vec_noncontraction_shape",
            "tag_vector_generics",
            "vectorize_amax_reduction",
            "default_vec_rank_shape",
            "vec_rank_bounds",
            "im2col_pack_api",
            "perop_blocks_api",
            "perop_features_api",
            "weight_panel_api",
            "lmul_group_api",
            "lmul_group_feature",
            "microkernel_policy_api",
            "blocking_cost_api",
            "unclaimed_op_classes",
        )
    }
    good = SimpleNamespace(API_VERSION=1, **methods)
    good.bind = lambda _features: good
    monkeypatch.setattr(plugins, "load_declared_chain", lambda target, key: (("rvv", good),))
    assert backend_spi.load_vector_codegen("kodiak") is good
    monkeypatch.setattr(plugins, "load_declared_chain", lambda target, key: ())
    with pytest.raises(backend_spi.BackendAPIError, match="expected one"):
        backend_spi.load_vector_codegen("kodiak")
    bad = SimpleNamespace(API_VERSION=2, **methods)
    monkeypatch.setattr(plugins, "load_declared_chain", lambda target, key: (("rvv", bad),))
    with pytest.raises(backend_spi.BackendAPIError, match="requires 1"):
        backend_spi.load_vector_codegen("kodiak")


def test_chip_features_bind_only_to_the_selected_backend(monkeypatch):
    from merlin.llvmlower.feature_registry import ImprFeature
    from merlin.targetgen import plugins

    feature = ImprFeature("chip_matrix_route", "PASS", "Test chip operation route")
    bound: list[tuple[ImprFeature, ...]] = []
    methods = {
        name: lambda *args, **kwargs: None
        for name in (
            "build_pipeline",
            "baseline_schedule_text",
            "parallel_transform_schedule",
            "vec_pre_schedule",
            "normalize_features",
            "feature_names",
            "apply_schedule",
            "apply_pipeline",
            "apply_cflags",
            "vector_cflags",
            "schedule_path",
            "prepare_build_bundle",
            "spike_cflags",
            "spike_harness_cflags",
            "accum_microkernel_marker",
            "accum_microkernel_runner_source",
            "accum_microkernel_rewrite_source",
            "vec_noncontraction_shape",
            "tag_vector_generics",
            "vectorize_amax_reduction",
            "default_vec_rank_shape",
            "vec_rank_bounds",
            "im2col_pack_api",
            "perop_blocks_api",
            "perop_features_api",
            "weight_panel_api",
            "lmul_group_api",
            "lmul_group_feature",
            "microkernel_policy_api",
            "blocking_cost_api",
            "unclaimed_op_classes",
        )
    }
    backend = SimpleNamespace(API_VERSION=1, **methods)

    def bind(features):
        bound.append(features)
        return backend

    backend.bind = bind
    extension = SimpleNamespace(API_VERSION=1, feature_definitions=lambda: (feature,))

    def declared(target, key):
        if key == "rvv_codegen":
            return (("rvv", backend),)
        return (("chip", extension),) if target == "chip" else ()

    monkeypatch.setattr(plugins, "load_declared_chain", declared)
    assert backend_spi.load_vector_codegen("chip") is backend
    assert backend_spi.load_vector_codegen("rvv") is backend
    assert bound == [(feature,), ()]


def test_scalar_lowering_never_resolves_vector_provider(monkeypatch, tmp_path):
    from merlin.llvmlower import pipeline

    def forbidden(_target):
        raise AssertionError("scalar lowering resolved RVV codegen")

    monkeypatch.setattr(backend_spi, "load_vector_codegen", forbidden)
    source = "module { func.func @main() { return } }"
    lowered = pipeline.lower_to_llvm_ir(source, workdir=tmp_path, vectorize=False)
    assert "define" in lowered


def test_scalar_runtime_flags_never_resolve_vector_provider(monkeypatch):
    from merlin.runtime.backends import spike_model, zephyr_model

    def forbidden(_target):
        raise AssertionError("scalar runtime resolved RVV codegen")

    monkeypatch.setattr(backend_spi, "load_vector_codegen", forbidden)
    assert zephyr_model._cflags("scalar")[0] == "-march=rv64gc"
    assert "-march=rv64gc" in spike_model._harness_cflags(["-march=rv64gc"], backend="scalar")


def test_native_pipeline_listing_is_scalar_only_without_vector_provider(monkeypatch):
    from merlin.compile.debug import native_pipelines

    monkeypatch.setenv("MERLIN_TARGET_PATH", "")
    assert set(native_pipelines()) == {"serial", "openmp"}


def test_vector_lowering_uses_provider_feature_without_old_registry(tmp_path):
    source = """
from pathlib import Path
import sys
from merlin.llvmlower.pipeline import lower_to_llvm_ir
out = lower_to_llvm_ir(
    'module { func.func @main() { return } }',
    workdir=Path(sys.argv[1]),
    vectorize=True,
    features=frozenset({'intrinsic_microkernel'}),
)
assert 'define void @main' in out
assert 'merlin.llvmlower.impr_features' not in sys.modules
"""
    result = subprocess.run([sys.executable, "-c", source, str(tmp_path)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_vector_preparation_uses_provider_without_old_registry(tmp_path):
    source = """
from pathlib import Path
import sys
from merlin.runtime.backends.zephyr_model import prepare_for_lowering
work = Path(sys.argv[1])
src = work / 'model.mlir'
src.write_text('module { func.func @main() { return } }')
prepared, _features = prepare_for_lowering(
    src, work, backend='rvv', blocking=False,
    features=frozenset({'fuse_quantize_round_convert'}),
)
assert prepared.is_file()
assert 'merlin.llvmlower.impr_features' not in sys.modules
"""
    result = subprocess.run([sys.executable, "-c", source, str(tmp_path)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_chain_loads_base_code_from_base_root(monkeypatch, tmp_path):
    from merlin.targetgen import plugins, provider_compose
    from merlin.targetgen.support_pins import load_pins

    base = tmp_path / "base"
    chip = tmp_path / "chip"
    for root in (base, chip):
        (root / "contracts").mkdir(parents=True)
    pins = load_pins()
    assert pins.base is not None
    base_commit = pins.base.commit
    (tmp_path / "SOURCE.yaml").write_text(f"path: base\nsource:\n  commit: {base_commit}\n")
    (base / "provider.yaml").write_text(
        "schema: merlin.provider.v1\nid: rvv-support\ntarget: rvv\nrole: support\n"
        "contract: contracts/target_contract.yaml\n"
    )
    (base / "contracts/target_contract.yaml").write_text("name: rvv\nplugin:\n  rvv_codegen: backend.py\n")
    (base / "backend.py").write_text("API_VERSION = 1\ndef build_pipeline(*a, **kw): return 'base'\n")
    (chip / "provider.yaml").write_text(
        "schema: merlin.provider.v2\nid: chip-support\ntarget: chip\nrole: support\n"
        f"extends:\n  target: rvv\n  commit: {base_commit}\n"
    )
    (chip / "contracts/overlay.yaml").write_text("name: chip\n")
    (chip / "contracts/target_contract.yaml").write_text("name: chip\n")
    monkeypatch.setattr(plugins, "resolve_support", lambda target: SimpleNamespace(base=chip))
    monkeypatch.setattr(provider_compose, "_candidate_roots", lambda: [base])
    namespace = "merlin._oot_targets.rvv"
    saved = {
        name: module for name, module in sys.modules.items() if name == namespace or name.startswith(namespace + ".")
    }
    for name in saved:
        sys.modules.pop(name)
    try:
        entries = plugins.load_declared_chain("chip", "rvv_codegen")
        assert [owner for owner, _ in entries] == ["rvv"]
        assert entries[0][1].build_pipeline(None) == "base"
    finally:
        for name in tuple(sys.modules):
            if name == namespace or name.startswith(namespace + "."):
                sys.modules.pop(name)
        sys.modules.update(saved)
