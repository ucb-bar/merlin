"""Operator public tests never rewrite completed automatic generation receipts."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from merlin_experiments.cli import main
from merlin_experiments.corpus import public_supplement as S
from merlin_experiments.corpus import release
from merlin_experiments.runner import fingerprint
from merlin_experiments.spec import SpecError
from test_corpus_release import _member, _seal, release_fixture  # noqa: F401, F811


def selection(tmp_path):
    root = tmp_path / "requested-inputs"
    _member(root, "user_once", "requested", "public")
    provenance = tmp_path / "operator-materialization.json"
    provenance.write_text('{"origin":"operator_requested","generation_membership_claim":false}\n')
    path = root / "public-supplement.json"
    document = {
        "schema": S.SCHEMA,
        "target": "fixture-device",
        "origin": "operator_requested",
        "members": [{"member": "user_once/requested", "sha256": fingerprint(root / "user_once/requested")}],
        "provenance": {"path": str(provenance), "sha256": fingerprint(provenance)},
    }
    path.write_text(json.dumps(document))
    return path, document


def test_ordinary_prepare_and_seal_bind_separate_operator_origin(release_fixture, tmp_path, capsys):  # noqa: F811
    f = release_fixture
    selected, document = selection(tmp_path)
    assert main(["run", str(f["definition"]), "--phase", "0", "--run-dir", str(f["run"])]) == 0
    capsys.readouterr()
    _, attempt, original = release.source_run(f["run"])
    before = (fingerprint(original), fingerprint(f["run"] / "resolved-plan.json"), dict(attempt))
    assert (
        main(
            [
                "corpus",
                "prepare",
                "--historical-overlay",
                str(f["run"]),
                "--output",
                str(f["release"]),
                "--public-supplement",
                str(selected),
            ]
        )
        == 0
    )
    report = json.loads(capsys.readouterr().out)
    assert report["counts"]["public_source"] == report["counts"]["public_admitted"] == 3
    assert report["counts"]["hidden_source"] == report["counts"]["hidden_admitted"] == 1
    prepared = json.loads((f["release"] / "private/preparation.json").read_bytes())
    record = prepared["assembly"]["public_supplement"]
    assert record["selection"]["sha256"] == fingerprint(selected)
    assert record["provenance"] == document["provenance"]
    corpus = f["release"] / "payload/corpus"
    manifest = yaml.safe_load((corpus / "MANIFEST.yaml").read_bytes())
    assert manifest["generated"] == ["isa/generated_member"]
    assert manifest["operator_requested"] == ["user_once/requested"]
    assert fingerprint(corpus / "user_once/requested") == document["members"][0]["sha256"]
    _seal(f, report, capsys)
    _, after_attempt, after = release.source_run(f["run"])
    assert (fingerprint(after), fingerprint(f["run"] / "resolved-plan.json"), dict(after_attempt)) == before


@pytest.mark.parametrize("damage", ["target", "origin", "hidden", "unlisted", "bytes", "provenance"])
def test_selection_refuses_foreign_unlisted_or_changed_inputs(tmp_path, damage):
    path, document = selection(tmp_path)
    if damage in {"target", "origin"}:
        document[damage] = "foreign"
    elif damage == "hidden":
        document["members"][0]["member"] = "hidden/requested"
    elif damage == "unlisted":
        _member(path.parent, "user_once", "omitted", "public")
    elif damage == "bytes":
        (path.parent / "user_once/requested/capsule.interface.mlir").write_text("module { // changed\n}\n")
    else:
        Path(document["provenance"]["path"]).write_text("changed\n")
    path.write_text(json.dumps(document))
    with pytest.raises((SpecError, ValueError)):
        S.read(path, target="fixture-device")


def test_verify_reopens_all_source_and_staged_bytes(tmp_path):
    path, _ = selection(tmp_path)
    record = S.read(path, target="fixture-device")
    import shutil

    staged = tmp_path / "staged"
    shutil.copytree(path.parent, staged)
    S.verify(record, staged)
    (staged / "user_once/requested/golden.yaml").write_text("outputs: {changed: true}\n")
    with pytest.raises(SpecError, match="staged public supplement"):
        S.verify(record, staged)


def test_stage_refuses_existing_capsule_identity_without_replacement(tmp_path):
    path, _ = selection(tmp_path)
    destination = tmp_path / "corpus"
    _member(destination, "isa", "requested", "public")
    (destination / "MANIFEST.yaml").write_text("generated: [isa/requested]\n")
    te = SimpleNamespace(target="fixture-device", workload_spec=None, model_layer_shapes=None)
    with pytest.raises(SpecError, match="cannot replace"):
        S.stage(path, destination, te=te)


def test_public_supplement_cannot_launder_a_private_hardlink(tmp_path):
    import os

    path, document = selection(tmp_path)
    private = tmp_path / "private"
    private.mkdir()
    answer = private / "answer"
    answer.write_text("private answer\n")
    public_alias = path.parent / "user_once/requested/golden.yaml"
    public_alias.unlink()
    os.link(answer, public_alias)
    document["members"][0]["sha256"] = fingerprint(public_alias.parent)
    path.write_text(json.dumps(document))
    destination = tmp_path / "corpus"
    _member(destination, "isa", "base", "public")
    (destination / "MANIFEST.yaml").write_text("generated: [isa/base]\n")
    te = SimpleNamespace(target="fixture-device", workload_spec=None, model_layer_shapes=None)
    with pytest.raises(RuntimeError, match="host-only|private"):
        S.stage(path, destination, te=te, private_roots=[private])


def test_operator_selection_does_not_change_phase2_or_generated_membership(tmp_path):
    from merlin_experiments.corpus.phase_selection import generate_phase_selections

    path, _ = selection(tmp_path)
    destination = tmp_path / "corpus"
    _member(destination, "isa", "base", "public")
    _member(destination, "_perf", "existing_perf", "public")
    generated = ["isa/base", "_perf/existing_perf"]
    phases = {"fixture-device": generate_phase_selections(generated, performance_category="_perf")}
    (destination / "MANIFEST.yaml").write_text(yaml.safe_dump({"generated": generated, "phase_corpora": phases}))
    te = SimpleNamespace(target="fixture-device", workload_spec=None, model_layer_shapes=None)
    S.stage(path, destination, te=te)
    manifest = yaml.safe_load((destination / "MANIFEST.yaml").read_bytes())
    assert manifest["phase_corpora"] == phases
    assert manifest["generated"] == generated
    assert manifest["operator_requested"] == ["user_once/requested"]
