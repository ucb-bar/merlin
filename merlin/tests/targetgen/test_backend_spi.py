"""Executable backend loading and scalar independence."""

from types import SimpleNamespace

import pytest

from merlin.llvmlower import backend_spi


def test_vector_backend_requires_one_compatible_owner(monkeypatch):
    from merlin.targetgen import plugins

    good = SimpleNamespace(API_VERSION=1, build_pipeline=lambda *args, **kw: "passes")
    monkeypatch.setattr(plugins, "load_declared_chain", lambda target, key: (("rvv", good),))
    assert backend_spi.load_vector_codegen("kodiak") is good
    monkeypatch.setattr(plugins, "load_declared_chain", lambda target, key: ())
    with pytest.raises(backend_spi.BackendAPIError, match="expected one"):
        backend_spi.load_vector_codegen("kodiak")
    bad = SimpleNamespace(API_VERSION=2, build_pipeline=good.build_pipeline)
    monkeypatch.setattr(plugins, "load_declared_chain", lambda target, key: (("rvv", bad),))
    with pytest.raises(backend_spi.BackendAPIError, match="requires 1"):
        backend_spi.load_vector_codegen("kodiak")


def test_scalar_lowering_never_resolves_vector_provider(monkeypatch, tmp_path):
    from merlin.llvmlower import pipeline

    def forbidden(_target):
        raise AssertionError("scalar lowering resolved RVV codegen")

    monkeypatch.setattr(backend_spi, "optional_vector_codegen", forbidden)
    source = "module { func.func @main() { return } }"
    lowered = pipeline.lower_to_llvm_ir(source, workdir=tmp_path, vectorize=False)
    assert "define" in lowered


def test_chain_loads_base_code_from_base_root(monkeypatch, tmp_path):
    from merlin.targetgen import plugins, provider_compose

    base = tmp_path / "base"
    chip = tmp_path / "chip"
    for root in (base, chip):
        (root / "contracts").mkdir(parents=True)
    base_commit = "a80eea3eb749e5fd4cb822779a08d0d8ca09e5d5"
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
