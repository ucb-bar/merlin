"""CIRCT facts cache identity must bind feature claims to exact extractor and RTL bytes."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

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
        "facts": {"interfaces": [{
            "name": "elaborated_rtl_features",
            "status": "derived",
            "features": {"max_pool": True},
            "source": str(fir),
            "source_sha256": digest,
        }]},
    }


def test_circt_facts_identity_rejects_changed_fir_and_old_feature_schema(tmp_path: Path) -> None:
    fir = tmp_path / "design.fir"
    fir.write_text("module StoreController :\n", encoding="utf-8")
    artifact = tmp_path / "facts.json"
    record = _record(fir)
    artifact.write_text(json.dumps(record), encoding="utf-8")
    assert facts._circt_facts_current(artifact)

    fir.write_text("module StoreController :\n  skip\n", encoding="utf-8")
    assert not facts._circt_facts_current(artifact)

    fir.write_text("module StoreController :\n", encoding="utf-8")
    record["facts"]["interfaces"][0]["name"] = "elaborated_config_features"
    artifact.write_text(json.dumps(record), encoding="utf-8")
    assert not facts._circt_facts_current(artifact)


def test_ensure_facts_regenerates_a_stale_default_cache(tmp_path: Path, monkeypatch) -> None:
    fir = tmp_path / "design.fir"
    fir.write_text("module StoreController :\n", encoding="utf-8")
    artifact = tmp_path / "facts.json"
    stale = _record(fir)
    stale["generator"]["version"] = "obsolete"
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
