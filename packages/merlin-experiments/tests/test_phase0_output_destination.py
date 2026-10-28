"""Generation cannot implicitly overwrite the descriptor's source corpus."""

from types import SimpleNamespace

import pytest
from merlin_experiments.phase0 import __main__ as cli
from merlin_experiments.phase0 import generation


def test_api_requires_destination_before_loading_or_mutating_inputs(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("input resolution must not start without an explicit output")

    monkeypatch.setattr(generation, "_descriptor_for", forbidden)
    monkeypatch.setattr(generation, "_ensure_contract_on_path", forbidden)
    with pytest.raises(ValueError, match="explicit output_root"):
        generation.generate_target("fixture")


def test_cli_requires_destination_before_generation(monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        pytest.fail("generation must not start without an explicit output")

    monkeypatch.setattr(cli, "generate_target", forbidden)
    with pytest.raises(SystemExit) as raised:
        cli.main(["--target", "fixture"])
    assert raised.value.code == 2
    assert "--output-root" in capsys.readouterr().err


def test_cli_forwards_explicit_artifact_destination(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(cli, "generate_target", lambda target, **options: calls.append((target, options)) or [])
    profiles, descriptor, output = tmp_path / "recipes", tmp_path / "target.yaml", tmp_path / "artifacts"
    assert (
        cli.main(
            [
                "--target",
                "fixture",
                "--profiles-root",
                str(profiles),
                "--descriptor",
                str(descriptor),
                "--output-root",
                str(output),
            ]
        )
        == 0
    )
    assert calls == [("fixture", {"profiles_root": profiles, "descriptor": descriptor, "output_root": output})]
    assert not output.exists()


def test_cli_refuses_implicit_recipe_discovery_before_generation(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "generate_target", lambda *a, **k: pytest.fail("implicit input reached generator"))
    with pytest.raises(SystemExit) as raised:
        cli.main(["--target", "fixture", "--output-root", str(tmp_path / "output")])
    assert raised.value.code == 2
    assert not (tmp_path / "output").exists()


def test_api_refuses_implicit_inputs_before_descriptor_environment_setup(monkeypatch, tmp_path):
    monkeypatch.setattr(generation, "_descriptor_for", lambda *a: pytest.fail("descriptor discovery ran"))
    monkeypatch.setattr(generation, "_ensure_contract_on_path", lambda *a: pytest.fail("provider setup ran"))
    with pytest.raises(ValueError, match="explicit recipe inputs"):
        generation.generate_target("fixture", output_root=tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_generation_refuses_source_corpus_before_loading_profile(monkeypatch, tmp_path):
    source = tmp_path / "source" / "isa"
    source.mkdir(parents=True)
    (source / "capsule.yaml").write_text("historical input\n")
    descriptor = tmp_path / "target.yaml"
    descriptor.write_text("target: fixture\n")
    monkeypatch.setattr(generation, "_ensure_contract_on_path", lambda *args: None)
    monkeypatch.setattr(generation, "load_target_experiment", lambda *args: SimpleNamespace(capsule_corpus=source))
    monkeypatch.setattr(generation, "load_profile", lambda *args, **kwargs: pytest.fail("source corpus was read"))
    with pytest.raises(ValueError, match="output_root .* overlaps source capsule corpus"):
        generation.generate_target(
            "fixture", descriptor=descriptor, output_root=source.parent, profiles_root=tmp_path / "profiles"
        )
    assert (source / "capsule.yaml").read_text() == "historical input\n"
    assert not (source.parent / "MANIFEST.yaml").exists()


def test_evidence_output_rejects_alias_to_source_corpus(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(source, target_is_directory=True)
    te = SimpleNamespace(capsule_corpus=source)
    with pytest.raises(ValueError, match="evidence_root .* overlaps source capsule corpus"):
        generation._require_distinct_corpus_destinations(
            te, output_root=tmp_path / "output", evidence_root=alias / "evidence"
        )
    generation._require_distinct_corpus_destinations(
        te, output_root=tmp_path / "output", evidence_root=tmp_path / "evidence"
    )


def test_registered_corpus_is_never_an_output_even_for_another_descriptor(tmp_path, monkeypatch):
    from merlin.targetgen import corpora

    retained = tmp_path / "retained"
    retained.mkdir()
    monkeypatch.setattr(corpora, "capsule_corpus_roots", lambda **_kwargs: [retained])
    te = SimpleNamespace(capsule_corpus=tmp_path / "unrelated" / "isa")
    with pytest.raises(ValueError, match="output_root .* overlaps source capsule corpus"):
        generation._require_distinct_corpus_destinations(te, output_root=retained / "isa", evidence_root=None)


def test_external_workspace_without_registry_still_protects_source(tmp_path, monkeypatch):
    monkeypatch.setenv("MERLIN_REPO_ROOT", str(tmp_path / "external-workspace"))
    source = tmp_path / "external" / "isa"
    te = SimpleNamespace(capsule_corpus=source)
    generation._require_distinct_corpus_destinations(te, output_root=tmp_path / "run" / "capsules", evidence_root=None)
    with pytest.raises(ValueError, match="output_root .* overlaps source capsule corpus"):
        generation._require_distinct_corpus_destinations(te, output_root=source, evidence_root=None)


def test_malformed_legacy_registry_is_not_treated_as_absent(tmp_path, monkeypatch):
    from merlin.targetgen import corpora

    def malformed(**_kwargs):
        raise ValueError("malformed")

    monkeypatch.setattr(corpora, "capsule_corpus_roots", malformed)
    with pytest.raises(ValueError, match="malformed"):
        generation._require_distinct_corpus_destinations(
            SimpleNamespace(capsule_corpus=tmp_path / "source"), output_root=tmp_path / "run", evidence_root=None
        )


def test_missing_selected_registry_is_not_treated_as_empty(tmp_path, monkeypatch):
    from merlin.common import paths

    monkeypatch.setattr(paths, "checkout_root", lambda: tmp_path / "missing-owner")
    with pytest.raises(FileNotFoundError, match="corpora.yaml"):
        generation._require_distinct_corpus_destinations(
            SimpleNamespace(capsule_corpus=tmp_path / "source"), output_root=tmp_path / "run", evidence_root=None
        )


def test_a_frozen_snapshots_corpus_copy_is_not_a_protected_source(tmp_path):
    evidence = tmp_path / "run" / "phase0"
    snapshot_corpus = evidence / "private" / "source" / "merlin" / "contract" / "capsules" / "isa"
    frozen_target = SimpleNamespace(capsule_corpus=snapshot_corpus)
    generation._require_distinct_corpus_destinations(
        frozen_target, output_root=evidence / "capsules", evidence_root=evidence
    )
    # The snapshot keeps the held-out inputs of the corpus; that private copy is still not a retained source.
    snapshot_corpus.mkdir(parents=True)
    generation._require_distinct_corpus_destinations(
        frozen_target, output_root=evidence / "capsules", evidence_root=evidence
    )
    retained = tmp_path / "retained" / "capsules"
    retained.mkdir(parents=True)
    with pytest.raises(ValueError, match="evidence_root .* overlaps source capsule corpus"):
        generation._require_distinct_corpus_destinations(
            SimpleNamespace(capsule_corpus=retained), output_root=tmp_path / "out", evidence_root=tmp_path / "retained"
        )
