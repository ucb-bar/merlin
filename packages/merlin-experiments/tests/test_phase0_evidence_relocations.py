"""Relocation preserves exact recorded bytes without rewriting facts or old custody."""

import hashlib
import json
from pathlib import Path

import pytest
from merlin_experiments.phase0 import evidence
from merlin_experiments.phase0.evidence_rtl_sources import observe_sources


def _fixture(tmp_path):
    original, replacement = tmp_path / "old.py", tmp_path / "frozen.py"
    original.write_text("changed source\n")
    replacement.write_text("original source\n")
    digest = hashlib.sha256(replacement.read_bytes()).hexdigest()
    facts = {"inputs": {"reader_sources": [{"path": str(original), "sha256": digest}]}}
    relocation = {str(original): {"path": str(replacement), "sha256": digest}}
    observed = []

    def observe(path, role, **kwargs):
        observed.append(Path(path))
        return Path(path).read_bytes()

    return facts, relocation, observed, observe


def test_explicit_relocation_keeps_facts_and_records_actual_source_custody(tmp_path):
    facts, relocation, observed, observe = _fixture(tmp_path)
    before = json.dumps(facts, sort_keys=True)
    diagnostics = []
    receipt = observe_sources(facts, observe, diagnostics, relocations=relocation, facts_sha256="f" * 64)
    original, row = next(iter(relocation.items()))
    assert observed == [Path(row["path"])] and diagnostics == []
    assert receipt["sources"] == [{"original_path": original, "replacement_path": row["path"], "sha256": row["sha256"]}]
    assert receipt["facts_sha256"] == "f" * 64 and json.dumps(facts, sort_keys=True) == before


def test_default_keeps_original_source_contradiction(tmp_path):
    facts, _, observed, observe = _fixture(tmp_path)
    diagnostics = []
    assert observe_sources(facts, observe, diagnostics) is None
    assert observed == [Path(facts["inputs"]["reader_sources"][0]["path"])]
    assert diagnostics[0]["status"] == "contradiction"


@pytest.mark.parametrize(
    "change", ["wrong_bytes", "wrong_digest", "missing_digest", "unused", "relative", "symlink", "extra"]
)
def test_relocation_refuses_unbound_or_changed_source(tmp_path, change):
    facts, relocation, _, observe = _fixture(tmp_path)
    key, row = next(iter(relocation.items()))
    if change == "wrong_bytes":
        Path(row["path"]).write_text("other bytes")
    elif change == "wrong_digest":
        row["sha256"] = "0" * 64
    elif change == "missing_digest":
        facts["inputs"]["reader_sources"][0].pop("sha256")
    elif change == "unused":
        relocation[str(tmp_path / "unknown.py")] = dict(row)
    elif change == "relative":
        row["path"] = "relative.py"
    elif change == "symlink":
        link = tmp_path / "alias.py"
        link.symlink_to(row["path"])
        row["path"] = str(link)
    else:
        row["waive"] = True
    with pytest.raises(ValueError):
        observe_sources(facts, observe, [], relocations=relocation, facts_sha256="f" * 64)


def test_ordinary_selection_exports_relocation_and_replays_without_original_reads(tmp_path, monkeypatch):
    # Use the ordinary evidence-selection fixture; it refuses any live extraction.
    from test_phase0_evidence import _selection

    _, raw, _ = _selection(monkeypatch, tmp_path)
    facts, relocation, _, _ = _fixture(tmp_path)
    document = json.loads(raw.read_bytes())
    document["inputs"] = facts["inputs"]
    raw.write_text(json.dumps(document))
    original_bytes = raw.read_bytes()
    selected = evidence.select_evidence("fixture", facts_path=raw, rtl_source_relocations=relocation)
    assert not any(d["component"] == "rtl-source" for d in selected.diagnostics)
    root = tmp_path / "export"
    manifest = evidence.export_evidence(selected, root)
    assert "hardware/source-relocations.json" in manifest["artifacts"]
    assert raw.read_bytes() == original_bytes
    row = next(iter(relocation.values()))
    Path(row["path"]).write_text("changed after immutable selection")
    loaded = evidence.load_exported_evidence(root)
    assert loaded.rtl_source_relocations == selected.rtl_source_relocations
    source = next(s for s in loaded.source_snapshots if str(s.path) == row["path"])
    assert source.sha256 == row["sha256"]
