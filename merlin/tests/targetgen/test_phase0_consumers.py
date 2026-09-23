"""Research callers consume the authoritative Phase 0 owners, never the CLI script."""

from types import SimpleNamespace

import pytest
from merlin_experiments.phase0 import declarations, profiles, writer

from merlin.targetgen import capsule_runner, corpus_spec, group_capsules, store_probe, target_experiment


def _target(monkeypatch, tmp_path):
    target = SimpleNamespace(target="fixture", sim_via="synthetic", capsule_corpus=tmp_path)
    binding = object()
    selected = SimpleNamespace(
        target="fixture",
        profile="authored-profile",
        descriptor=tmp_path / "target.yaml",
        profile_inputs=lambda: {
            "recipe": tmp_path / "public.yaml",
            "performance_template": tmp_path / "performance.yaml",
            "hidden_profile": tmp_path / "private.yaml",
        },
    )
    monkeypatch.setattr(declarations, "for_target", lambda target: selected)
    monkeypatch.setattr(target_experiment, "load_target_experiment", lambda path: target)

    def load(profile, **kwargs):
        assert profile == "authored-profile"
        assert kwargs == {"include_holdouts": False, **selected.profile_inputs()}
        return {"datapath": {"subnormal_operand_flush": True}}

    monkeypatch.setattr(profiles, "load_profile", load)

    def derive(experiment, datapath):
        assert experiment is target
        assert datapath == {"subnormal_operand_flush": True}
        return binding

    monkeypatch.setattr(corpus_spec, "derive_binding", derive)
    return binding


def test_group_writer_uses_canonical_profile_and_writer_owners(monkeypatch, tmp_path):
    binding = _target(monkeypatch, tmp_path)
    calls = []

    def write(entry, actual_binding, root):
        assert actual_binding is binding
        calls.append(entry["name"])
        if entry["name"] == "refused":
            raise ValueError("synthetic refusal")
        return root / entry["name"]

    monkeypatch.setattr(writer, "_write_capsule", write)
    report = group_capsules.write(
        "fixture",
        {
            "entries": [
                {"name": "accepted", "entry": {"name": "accepted"}},
                {"name": "refused", "entry": {"name": "refused"}},
            ]
        },
        tmp_path,
    )
    assert calls == ["accepted", "refused"]
    assert report == {
        "built": {"accepted": str(tmp_path / "accepted")},
        "refused_by_generator": {"refused": "ValueError: synthetic refusal"},
    }


def test_store_probe_uses_shared_writer_without_real_execution(monkeypatch, tmp_path):
    binding = _target(monkeypatch, tmp_path)
    package = tmp_path / "package"
    package.mkdir()
    (package / "manifest.yaml").write_text("{}")
    monkeypatch.setattr(store_probe, "ladder", lambda *args, **kwargs: [(2, 3, 4)])
    monkeypatch.setattr(capsule_runner, "oracle_adapters", lambda *args: {"L2": object()})
    calls = []

    def write(entry, actual_binding, root):
        assert actual_binding is binding
        assert (entry["M"], entry["K"], entry["N"]) == (2, 3, 4)
        calls.append(entry["name"])
        root.mkdir(parents=True)
        (root / "capsule.yaml").write_text("name: synthetic\n")
        return root

    monkeypatch.setattr(writer, "_write_capsule", write)
    monkeypatch.setattr(capsule_runner, "run_capsule", lambda *args, **kwargs: {"tiers": {"L2": {"status": "pass"}}})
    result = store_probe.probe("fixture", dtype="i8", edge=1, package_dir=package, workroot=tmp_path / "work")
    assert calls == ["PROBE_store_m2k3n4"]
    assert result.largest_ran == store_probe.working_set_elements(2, 3, 4)
    assert len(result.points) == 1 and result.points[0].ran


def test_explicit_declaration_prevents_ambient_selection(monkeypatch, tmp_path):
    _target(monkeypatch, tmp_path)
    selected = declarations.for_target("fixture")
    monkeypatch.setattr(declarations, "for_target", lambda *args: pytest.fail("explicit declaration rediscovered"))
    assert group_capsules.write("fixture", {"entries": []}, tmp_path, declaration=selected) == {
        "built": {},
        "refused_by_generator": {},
    }
    package = tmp_path / "package"
    package.mkdir()
    (package / "manifest.yaml").write_text("{}")
    monkeypatch.setattr(capsule_runner, "oracle_adapters", lambda *args: {})
    result = store_probe.probe("fixture", dtype="f32", edge=1, package_dir=package, declaration=selected)
    assert "resolves no oracle adapter" in result.unavailable


def test_ambiguous_selection_refuses_without_guessing(monkeypatch, tmp_path):
    from merlin_experiments.spec import SpecError

    def ambiguous(target):
        raise SpecError("requires one declaration; found 2")

    monkeypatch.setattr(declarations, "for_target", ambiguous)
    monkeypatch.setattr(writer, "_write_capsule", lambda *args: pytest.fail("ambiguous declaration wrote output"))
    with pytest.raises(SpecError, match="found 2"):
        group_capsules.write("fixture", {"entries": []}, tmp_path)
    package = tmp_path / "package"
    package.mkdir()
    (package / "manifest.yaml").write_text("{}")
    result = store_probe.probe("fixture", dtype="f32", edge=1, package_dir=package)
    assert "found 2" in result.unavailable
    assert not result.points


def test_incompatible_descriptor_is_not_used(monkeypatch, tmp_path):
    _target(monkeypatch, tmp_path)
    monkeypatch.setattr(target_experiment, "load_target_experiment", lambda *args: SimpleNamespace(target="other"))
    with pytest.raises(ValueError, match="does not match"):
        group_capsules.write("fixture", {"entries": []}, tmp_path)


def test_group_cli_selects_explicit_definition_before_output(monkeypatch, tmp_path):
    _target(monkeypatch, tmp_path)
    selected = declarations.for_target("fixture")
    seen = []

    def select(path):
        seen.append(path)
        return selected

    monkeypatch.setattr(declarations, "from_definition", select)
    monkeypatch.setattr(declarations, "for_target", lambda *args: pytest.fail("explicit definition ignored"))
    from merlin.common import mlir_query

    monkeypatch.setattr(mlir_query, "parse", lambda *args: object())
    monkeypatch.setattr(
        group_capsules,
        "entries",
        lambda *args, **kwargs: {"entries": [], "stated": 0, "accelerator_groups": 0, "distinct": 0, "unstated": {}},
    )
    definition = tmp_path / "experiment.yaml"
    assert (
        group_capsules.main(
            [
                "--target",
                "fixture",
                "--definition",
                str(definition),
                "--capture",
                "fixture.mlir",
                "--out",
                str(tmp_path / "out"),
            ]
        )
        == 0
    )
    assert seen == [definition]
