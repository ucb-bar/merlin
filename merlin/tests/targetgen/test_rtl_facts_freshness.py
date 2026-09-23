"""CIRCT facts cache identity must bind feature claims to exact extractor and RTL bytes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from merlin.targetgen.rtl import circt_introspect, facts


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record(fir: Path) -> dict:
    digest = _sha256(fir)
    extractor = Path(circt_introspect.__file__).resolve()
    return {
        "generator": {
            "name": "merlin.targetgen.rtl.circt_introspect",
            "version": circt_introspect.GENERATOR_VERSION,
        },
        "inputs": {
            "extractor_sha256": _sha256(extractor),
            "fir_sha256": digest,
        },
        "facts": {
            "interfaces": [
                {
                    "name": "elaborated_rtl_features",
                    "status": "derived",
                    "features": {"max_pool": True},
                    "source": str(fir),
                    "source_sha256": digest,
                }
            ]
        },
    }


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("MERLIN_REPO_ROOT", str(tmp_path / "repo"))
    monkeypatch.setenv("MERLIN_OUT_ROOT", str(tmp_path / "out"))
    monkeypatch.delenv("MERLIN_TARGET_PATH", raising=False)
    monkeypatch.delenv("MERLIN_RTL_FACTS", raising=False)
    monkeypatch.setattr(facts, "facts_alias", lambda target: target)
    monkeypatch.setattr(facts, "_committed_facts_path", lambda target: None)
    monkeypatch.setattr(facts, "_written_by_another_family", lambda *args: False)
    facts.clear_resolution_cache()
    yield
    facts.clear_resolution_cache()


def test_discovery_rejects_changed_fir_even_after_resolution_memo(tmp_path: Path, isolated) -> None:
    fir = tmp_path / "design.fir"
    fir.write_text("module StoreController :\n", encoding="utf-8")
    artifact = facts.rtl_facts_path("synthetic")
    artifact.parent.mkdir(parents=True)
    record = _record(fir)
    artifact.write_text(json.dumps(record), encoding="utf-8")
    assert facts.find_facts("synthetic") == artifact

    fir.write_text("module StoreController :\n  skip\n", encoding="utf-8")
    assert facts.find_facts("synthetic") is None

    fir.write_text("module StoreController :\n", encoding="utf-8")
    record["inputs"]["extractor_sha256"] = "0" * 64
    artifact.write_text(json.dumps(record), encoding="utf-8")
    assert facts.find_facts("synthetic") is None


def test_ensure_facts_regenerates_a_stale_default_cache(tmp_path: Path, monkeypatch, isolated) -> None:
    fir = tmp_path / "design.fir"
    fir.write_text("module StoreController :\n", encoding="utf-8")
    artifact = tmp_path / "facts.json"
    stale = _record(fir)
    stale["inputs"]["extractor_sha256"] = "0" * 64
    artifact.write_text(json.dumps(stale), encoding="utf-8")
    regenerated: list[str] = []

    monkeypatch.delenv("MERLIN_RTL_FACTS", raising=False)
    monkeypatch.setattr(facts, "rtl_facts_path", lambda target, explicit=None: artifact)
    monkeypatch.setattr(facts, "_committed_facts_path", lambda target: None)
    monkeypatch.setattr(facts, "_warn_if_degraded", lambda target: None)

    def regenerate(path: Path, target: str) -> None:
        regenerated.append(target)
        path.write_text(json.dumps(_record(fir)), encoding="utf-8")

    monkeypatch.setattr(facts, "_dump_facts_for_kind", regenerate)
    assert facts.ensure_facts("synthetic") == artifact
    assert regenerated == ["synthetic"]


@pytest.mark.parametrize("pin", ["fir_sha256", "feature", "extractor_sha256"])
def test_declared_pin_mismatch_is_unavailable_without_extraction(tmp_path, isolated, pin):
    fir = tmp_path / "design.fir"
    fir.write_text("module Test :\n")
    artifact = facts.rtl_facts_path("synthetic")
    artifact.parent.mkdir(parents=True)
    record = _record(fir)
    if pin == "feature":
        record["facts"]["interfaces"][0]["source_sha256"] = "0" * 64
    else:
        record["inputs"][pin] = "0" * 64
    artifact.write_text(json.dumps(record))
    assert facts.find_facts("synthetic") is None


def test_missing_source_invalidates_populated_memo(tmp_path, isolated):
    fir = tmp_path / "design.fir"
    fir.write_text("module Test :\n")
    artifact = facts.rtl_facts_path("synthetic")
    artifact.parent.mkdir(parents=True)
    artifact.write_text(json.dumps(_record(fir)))
    assert facts.find_facts("synthetic") == artifact
    fir.unlink()
    assert facts.find_facts("synthetic") is None


def test_declared_extraction_layout_and_reader_are_cache_inputs(tmp_path, monkeypatch, isolated):
    fir = tmp_path / "design.fir"
    fir.write_text("module Test :\n")
    contract = tmp_path / "target_contract.yaml"
    contract.write_text("rtl_extraction: {}\n")
    monkeypatch.setattr(facts, "target_contract_path", lambda target: contract)
    artifact = facts.rtl_facts_path("synthetic")
    artifact.parent.mkdir(parents=True)
    record = _record(fir)
    record["inputs"].update(
        target="synthetic",
        extraction_contract_sha256=_sha256(contract),
        extraction_reader_sha256=_sha256(Path(circt_introspect.extraction_contract.__file__)),
    )
    artifact.write_text(json.dumps(record))
    assert facts.find_facts("synthetic") == artifact

    contract.write_text("rtl_extraction: {changed: true}\n")
    assert facts.find_facts("synthetic") is None
    contract.write_text("rtl_extraction: {}\n")
    record["inputs"]["extraction_reader_sha256"] = "0" * 64
    artifact.write_text(json.dumps(record))
    assert facts.find_facts("synthetic") is None


def test_legacy_without_pins_is_read_without_invented_provenance(tmp_path, isolated):
    artifact = facts.rtl_facts_path("synthetic")
    artifact.parent.mkdir(parents=True)
    record = {"facts": {"target": "synthetic", "memories": []}}
    artifact.write_text(json.dumps(record))
    assert facts.find_facts("synthetic") == artifact
    assert facts.load_facts("synthetic") == record
