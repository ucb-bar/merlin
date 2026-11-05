"""Exact bounded source preparation; no synthetic capture or admission pass."""

import ast
import hashlib
import json

import pytest
from merlin_experiments.phase0 import development_sealed_batch as B
from test_development_capture_intake import fixture

LIMITS = dict(max_work=10000, max_elements=10000, max_seed_bytes=100000, max_contractions=1)


def test_preparation_preserves_all_source_and_reference_bytes_and_original_refusals(tmp_path, monkeypatch):
    root, digest, supplement = fixture(tmp_path)
    original = (root / "manifest.json").read_bytes()
    output = tmp_path / "prepared"
    result = B.prepare(
        root, manifest_sha256=digest, ordinals=(0,), output=output, limits=LIMITS, supplements=(supplement,)
    )
    selected = result["members"][0]
    manifest = json.loads(original)
    assert selected["request_sha256"] == manifest["members"][0]["request_sha256"]
    assert (output / "workload/member_0.py").read_bytes() == (root / manifest["members"][0]["loader"]).read_bytes()
    assert selected["reference"]["kind"] == "supplement_complete_npy"
    assert result["capture_intake"]["members"][1]["original_status"] == "source_refused"
    assert len(result["capture_intake"]["members"]) == 2
    assert not result["source_sealed"] and not result["writer_admitted"]
    assert (root / "manifest.json").read_bytes() == original
    ast.parse((output / "workload/loader.py").read_text())


@pytest.mark.parametrize("ordinals", [(), (0, 0), (True,), (1,), (2,)])
def test_ineligible_or_ambiguous_members_refuse_before_materializing_workload(tmp_path, ordinals):
    root, digest, supplement = fixture(tmp_path)
    output = tmp_path / "prepared"
    with pytest.raises(ValueError):
        B.prepare(
            root, manifest_sha256=digest, ordinals=ordinals, output=output, limits=LIMITS, supplements=(supplement,)
        )
    assert not output.exists()


def test_original_capture_cost_and_loader_drift_cannot_be_silently_admitted(tmp_path):
    root, digest, supplement = fixture(tmp_path)
    output = tmp_path / "prepared"
    with pytest.raises(ValueError, match="cost or numerical"):
        B.prepare(
            root,
            manifest_sha256=digest,
            ordinals=(0,),
            output=output,
            limits=dict(LIMITS, max_work=1),
            supplements=(supplement,),
        )
    member = json.loads((root / "manifest.json").read_bytes())["members"][0]
    loader = root / member["loader"]
    loader.chmod(0o600)
    loader.write_text("changed source")
    with pytest.raises(ValueError):
        B.prepare(root, manifest_sha256=digest, ordinals=(0,), output=output, limits=LIMITS, supplements=(supplement,))
    assert not output.exists()


def test_host_only_declaration_cannot_be_staged_as_accelerator_development(tmp_path, monkeypatch):
    root, digest, supplement = fixture(tmp_path)
    original = B.INTAKE.references

    def host_only(*args, **kwargs):
        manifest, references = original(*args, **kwargs)
        manifest["members"][0]["description"].update(contractions=0, scalar_products=0)
        return manifest, references

    monkeypatch.setattr(B.INTAKE, "references", host_only)
    with pytest.raises(ValueError, match="nonzero accelerator work"):
        B.prepare(
            root,
            manifest_sha256=digest,
            ordinals=(0,),
            output=tmp_path / "prepared",
            limits=LIMITS,
            supplements=(supplement,),
        )
    assert not (tmp_path / "prepared").exists()


def _runtime_selection(tmp_path, *, present, version=2):
    entry = tmp_path / "runtime-store"
    if present:
        entry.mkdir()
    selection = tmp_path / "runtime.json"
    selection.write_text(
        json.dumps(
            dict(
                schema=f"merlin.sealed_m2m_readonly_runtime.v{version}",
                entry=str(entry),
                source_cas_identity="0" * 64,
                effective_runtime_identity="1" * 64,
            )
        )
    )
    return selection, hashlib.sha256(selection.read_bytes()).hexdigest()


@pytest.mark.parametrize("present", [False, True])
@pytest.mark.parametrize("version", [2, 3])
def test_storage_preflight_never_upgrades_a_directory_to_a_runtime_grant(tmp_path, present, version):
    selection, digest = _runtime_selection(tmp_path, present=present, version=version)
    result = B.runtime_preflight(selection, sha256=digest)
    assert result["status"] == ("ORIGINAL_OWNER_REQUIRED" if present else "REFUSED_MISSING_SELECTED_RUNTIME")
    assert result["runtime_qualified"] is False
    assert result["required_next"].startswith("original capture_selection")
    selection.write_text("changed declaration")
    with pytest.raises(ValueError, match="bytes changed"):
        B.runtime_preflight(selection, sha256=digest)


def test_oversized_runtime_declaration_refuses_before_reading_bytes(tmp_path, monkeypatch):
    from pathlib import Path

    selection = tmp_path / "runtime.json"
    with selection.open("wb") as stream:
        stream.truncate((1 << 20) + 1)
    monkeypatch.setattr(Path, "read_bytes", lambda *_: pytest.fail("oversized selection must not be read"))
    with pytest.raises(ValueError, match="exceeds"):
        B.runtime_preflight(selection, sha256="0" * 64)


def test_status_command_reads_retained_stage_records_without_any_new_preparation(tmp_path, monkeypatch, capsys):
    (tmp_path / "preparation.json").write_text('{"source_sealed":false,"writer_admitted":false}')
    (tmp_path / "runtime-preflight.json").write_text('{"status":"REFUSED_MISSING_SELECTED_RUNTIME"}')
    monkeypatch.setattr(B, "prepare", lambda *_a, **_kw: pytest.fail("status must not prepare or execute"))
    assert B.main(["--status", str(tmp_path)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["recorded_preparation_only"] is True
    assert result["preparation.json"]["writer_admitted"] is False
