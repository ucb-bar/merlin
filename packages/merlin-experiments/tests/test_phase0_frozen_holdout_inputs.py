from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from merlin.targetgen import composition_corpus as CC
from merlin_experiments.phase0 import freeze
from merlin_experiments.phase0.generation_paths import _require_distinct_corpus_destinations

LAYER = {
    "operation": {"op": "matmul", "attributes": {"lhs": "A", "weight": "W"}},
    "inputs": [{"name": "A", "shape": [4, 8]}, {"name": "W", "shape": [8, 16]}],
}


def _checkout(tmp_path):
    corpus = tmp_path / "checkout/merlin/contract/capsules"
    (corpus / "_model_layers/layer0").mkdir(parents=True)
    (corpus / "_model_layers/layer0/capsule.yaml").write_text(yaml.safe_dump(LAYER))
    (corpus / "_model_layers/layer0/weights.bin").write_bytes(b"w")
    (corpus / "_perf/p0").mkdir(parents=True)
    (corpus / "_perf/p0/capsule.yaml").write_text("x: 1\n")
    (corpus / "layers/other").mkdir(parents=True)
    (corpus / "layers/other/capsule.yaml").write_text("x: 2\n")
    (corpus / "MANIFEST.yaml").write_text(yaml.safe_dump({"generated": ["_model_layers/layer0", "layers/other"]}))
    return tmp_path / "checkout", corpus


def test_held_out_inputs_name_exactly_what_held_out_shapes_reads(tmp_path):
    _, corpus = _checkout(tmp_path)
    assert CC.held_out_inputs(corpus / "MANIFEST.yaml") == ["MANIFEST.yaml", "_model_layers/layer0/capsule.yaml"]


def test_the_frozen_snapshot_keeps_only_the_held_out_inputs_of_the_corpus(tmp_path):
    root, corpus = _checkout(tmp_path)
    excluded = freeze._held_out_capsule_exclusions(root)
    base = "merlin/contract/capsules"
    assert f"{base}/MANIFEST.yaml" not in excluded
    assert f"{base}/_model_layers" not in excluded and f"{base}/_model_layers/layer0" not in excluded
    assert f"{base}/_model_layers/layer0/capsule.yaml" not in excluded
    assert {f"{base}/_model_layers/layer0/weights.bin", f"{base}/_perf", f"{base}/_perf/p0/capsule.yaml",
            f"{base}/layers", f"{base}/layers/other/capsule.yaml"} <= excluded


def test_held_out_shapes_are_unchanged_when_read_from_the_pruned_snapshot(tmp_path):
    import shutil

    root, corpus = _checkout(tmp_path)
    excluded = freeze._held_out_capsule_exclusions(root)
    snapshot = tmp_path / "snapshot"
    shutil.copytree(root, snapshot, ignore=lambda directory, names: [
        name for name in names if Path(directory, name).relative_to(root).as_posix() in excluded
    ])
    assert CC.held_out_shapes(snapshot / "merlin/contract/capsules/MANIFEST.yaml") == CC.held_out_shapes(
        corpus / "MANIFEST.yaml"
    ) == frozenset({(4, 8, 16)})


def test_a_checkout_without_a_manifest_still_excludes_the_whole_corpus(tmp_path):
    (tmp_path / "merlin/contract/capsules/x").mkdir(parents=True)
    assert freeze._held_out_capsule_exclusions(tmp_path) == {"merlin/contract/capsules"}


@pytest.fixture
def no_registry(monkeypatch):
    monkeypatch.setattr("merlin.targetgen.corpora.capsule_corpus_roots", lambda owner_root=None: [])
    monkeypatch.setattr("merlin.common.paths.checkout_root", lambda: None)


def test_a_corpus_inside_the_runs_frozen_snapshot_is_not_a_retained_destination(tmp_path, no_registry):
    evidence = tmp_path / "evidence"
    frozen_corpus = evidence / "private/source/merlin/contract/capsules"
    frozen_corpus.mkdir(parents=True)
    (frozen_corpus / "MANIFEST.yaml").write_text("generated: []\n")
    te = SimpleNamespace(capsule_corpus=str(frozen_corpus))
    _require_distinct_corpus_destinations(te, output_root=tmp_path / "out", evidence_root=evidence)


def test_a_retained_corpus_still_blocks_an_overlapping_destination(tmp_path, no_registry):
    retained = tmp_path / "retained/capsules"
    retained.mkdir(parents=True)
    te = SimpleNamespace(capsule_corpus=str(retained))
    with pytest.raises(ValueError, match="overlaps source capsule corpus"):
        _require_distinct_corpus_destinations(te, output_root=retained / "new", evidence_root=tmp_path / "evidence")
    with pytest.raises(ValueError, match="overlaps source capsule corpus"):
        _require_distinct_corpus_destinations(te, output_root=tmp_path / "out", evidence_root=tmp_path / "retained")
