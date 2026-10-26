"""Sealed CPU dispatch follows the native proof declaration, never the host."""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from merlin.targetgen import native_ieee_certificate
from merlin_experiments.capture_execution import sealed_m2m


@pytest.mark.parametrize("host_value", [None, "avx512"])
def test_guest_and_bwrap_record_cpu_dispatch_without_host_passthrough(tmp_path, monkeypatch, host_value):
    monkeypatch.delenv("ATEN_CPU_CAPABILITY", raising=False)
    if host_value is not None:
        monkeypatch.setenv("ATEN_CPU_CAPABILITY", host_value)
    else:
        assert "ATEN_CPU_CAPABILITY" not in os.environ
    certificate = json.loads(Path(native_ieee_certificate.__file__).with_suffix(".json").read_bytes())
    expected = certificate["host"]["capability"].lower()
    assert sealed_m2m._guest_env(tmp_path, replayable_logs=False)["ATEN_CPU_CAPABILITY"] == expected
    argv = sealed_m2m._sandbox_argv(tmp_path / "bwrap", tmp_path, tmp_path, tmp_path, ("python",), tmp_path)
    settings = [argv[i + 1 : i + 3] for i, arg in enumerate(argv) if arg == "--setenv"]
    assert ["ATEN_CPU_CAPABILITY", expected] in settings


def test_policy_digest_binds_the_certificate_cpu_dispatch(tmp_path, monkeypatch):
    declaration = tmp_path / "certificate.json"
    monkeypatch.setattr(native_ieee_certificate, "__file__", str(declaration.with_suffix(".py")))
    declaration.write_text(json.dumps({"host": {"capability": "AVX2"}}))
    original = sealed_m2m._policy(("python",), tmp_path)
    declaration.write_text(json.dumps({"host": {"capability": "AVX512"}}))
    assert sealed_m2m._policy(("python",), tmp_path) != original


@pytest.mark.parametrize("value", [None, "avx2", "avx512"])
def test_loader_cannot_replace_the_proof_cpu_dispatch(tmp_path, value):
    selected = {"ATEN_CPU_CAPABILITY": value}
    with pytest.raises(sealed_m2m.SealedM2MError, match="ATEN_CPU_CAPABILITY"):
        sealed_m2m._guest_env(tmp_path, replayable_logs=False, loader_env=selected)
    with pytest.raises(sealed_m2m.SealedM2MError, match="unsafe name"):
        sealed_m2m._declared_loader_env("", selected)


@pytest.mark.parametrize(
    "dispatch",
    [
        {"capability": "AVX512", "status": "observed"},
        {"capability": None, "status": "unavailable"},
        {"capability": 2, "status": "observed"},
        {"capability": "AVX2", "status": "unavailable"},
        {},
        None,
    ],
)
def test_materialized_capture_refuses_unverified_cpu_dispatch(tmp_path, monkeypatch, dispatch):
    monkeypatch.setattr(
        sealed_m2m,
        "_validation_functions",
        lambda: SimpleNamespace(_materialized=lambda *_, **__: {"status": "verified_materialized"}),
    )
    (tmp_path / "meta.json").write_text(json.dumps({"determinism": {"cpu_dispatch": dispatch}}))
    with pytest.raises(sealed_m2m.SealedM2MError, match="CPU dispatch"):
        sealed_m2m._materialized(tmp_path, tmp_path, tmp_path)


@pytest.mark.parametrize("capability", ["AVX2", "avx2", "AvX2"])
def test_materialized_capture_accepts_declared_dispatch_case_insensitively(tmp_path, monkeypatch, capability):
    monkeypatch.setattr(
        sealed_m2m,
        "_validation_functions",
        lambda: SimpleNamespace(_materialized=lambda *_, **__: {"status": "verified_materialized"}),
    )
    (tmp_path / "meta.json").write_text(
        json.dumps({"determinism": {"cpu_dispatch": {"capability": capability, "status": "observed"}}})
    )
    assert sealed_m2m._materialized(tmp_path, tmp_path, tmp_path)["status"] == "verified_materialized"
