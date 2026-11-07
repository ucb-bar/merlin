"""Executable backend loading and scalar independence."""

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
        )
    }
    good = SimpleNamespace(API_VERSION=1, **methods)
    monkeypatch.setattr(plugins, "load_declared_chain", lambda target, key: (("rvv", good),))
    assert backend_spi.load_vector_codegen("kodiak") is good
    monkeypatch.setattr(plugins, "load_declared_chain", lambda target, key: ())
    with pytest.raises(backend_spi.BackendAPIError, match="expected one"):
        backend_spi.load_vector_codegen("kodiak")
    bad = SimpleNamespace(API_VERSION=2, **methods)
    monkeypatch.setattr(plugins, "load_declared_chain", lambda target, key: (("rvv", bad),))
    with pytest.raises(backend_spi.BackendAPIError, match="requires 1"):
        backend_spi.load_vector_codegen("kodiak")


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
    entries = plugins.load_declared_chain("chip", "rvv_codegen")
    assert [owner for owner, _ in entries] == ["rvv"]
    assert entries[0][1].build_pipeline(None) == "base"
