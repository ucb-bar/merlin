"""A spike model declared by its build INPUTS (``built_from``): portable across toolchains, still fail-closed."""

from __future__ import annotations

import hashlib
import json

import pytest

from merlin.targetgen import spike_extension as SX


def _build(tmp_path, *, source="s" * 64, header="h" * 64, selftest_ok=True, describe=None):
    so = tmp_path / "libmodel.so"
    so.write_bytes(b"a locally compiled model")
    digest = hashlib.sha256(so.read_bytes()).hexdigest()
    manifest = {
        "sha256": describe or digest,
        "source_sha256": source,
        "header_sha256": header,
        "selftest": {"ok": selftest_ok},
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    return so, digest


def _resolve(monkeypatch, so, **built_from):
    block = {
        "extension_name": "unit_model",
        "extlib": str(so),
        "built_from": {"source_sha256": "s" * 64, "header_sha256": "h" * 64, **built_from},
    }
    monkeypatch.setattr(SX, "declared_extension", lambda target: block)
    return SX.resolve("t", default_library_dir="/fallback", default_extension_name="fallback")


def test_a_build_from_the_declared_inputs_is_accepted(monkeypatch, tmp_path):
    so, digest = _build(tmp_path)
    ext = _resolve(monkeypatch, so)
    assert ext.declared and ext.sha256 == digest and ext.extension_name == "unit_model"


@pytest.mark.parametrize(
    "kwargs, reason",
    [
        ({"source": "x" * 64}, "built from source_sha256"),
        ({"selftest_ok": False}, "self-test"),
        ({"describe": "0" * 64}, "describes a different file"),
    ],
)
def test_anything_else_is_refused(monkeypatch, tmp_path, kwargs, reason):
    so, _ = _build(tmp_path, **kwargs)
    with pytest.raises(SX.SpikeExtensionError, match=reason):
        _resolve(monkeypatch, so)


def test_a_missing_manifest_is_refused(monkeypatch, tmp_path):
    so, _ = _build(tmp_path)
    (tmp_path / "manifest.json").unlink()
    with pytest.raises(SX.SpikeExtensionError, match="no readable build manifest"):
        _resolve(monkeypatch, so)


def test_an_exact_digest_and_built_from_cannot_both_be_declared(monkeypatch, tmp_path):
    so, digest = _build(tmp_path)
    block = {"extension_name": "m", "extlib": str(so), "sha256": digest, "built_from": {"source_sha256": "s"}}
    monkeypatch.setattr(SX, "declared_extension", lambda target: block)
    with pytest.raises(SX.SpikeExtensionError, match="pick one"):
        SX.resolve("t", default_library_dir="/f", default_extension_name="f")
