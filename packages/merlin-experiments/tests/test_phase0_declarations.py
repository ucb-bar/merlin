"""Declaration routing uses definition paths, never recipe naming conventions."""

from pathlib import Path

import pytest
import yaml
from merlin_experiments.phase0 import declarations as D
from merlin_experiments.spec import SpecError


def authored_inputs(monkeypatch):
    """Use the qualifier's committed YAML copies without importing checkout code."""
    from merlin.common.paths import repo_root

    retained = Path(__file__).with_name("source-inputs")
    root = retained if retained.is_dir() else repo_root()
    monkeypatch.setenv("MERLIN_REPO_ROOT", str(root))
    return root / "examples/gemmini/experiment.yaml", root / "examples/gemmini/target/descriptor.yaml"


def definition(tmp_path, name="one", **overrides):
    config = dict(
        descriptor="inputs/device.yaml",
        recipe="inputs/arbitrary.recipe.yaml",
        performance_template="templates/performance.yaml",
        profile="corpus-alias",
        hidden_profile="private/withheld.yaml",
    )
    config.update(overrides)
    path = tmp_path / f"{name}.yaml"
    path.write_text(
        yaml.safe_dump(
            dict(
                schema_version=1,
                id=name,
                target="runtime-device",
                phases={0: dict(adapter="capsule_derivation", config=config)},
            )
        )
    )
    return path


def catalog(tmp_path, *paths):
    path = tmp_path / "catalog.yaml"
    path.write_text(yaml.safe_dump(dict(schema_version=1, experiments={p.stem: p.name for p in paths})))
    return path


def test_paths_and_optional_absence_without_private_reads(tmp_path, monkeypatch):
    path = definition(tmp_path)
    original = Path.read_text

    def read(self, *args, **kwargs):
        assert "withheld" not in str(self)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read)
    monkeypatch.chdir(tmp_path.parent)
    declaration = D.from_definition(path)
    assert declaration.recipe == tmp_path / "inputs/arbitrary.recipe.yaml"
    assert declaration.descriptor == tmp_path / "inputs/device.yaml"
    assert declaration.hidden_profile == tmp_path / "private/withheld.yaml"
    assert declaration.synth_profile is None
    assert set(declaration.profile_inputs()) == {
        "recipe",
        "performance_template",
        "conformance_spec",
        "synth_profile",
        "smt_profile",
        "hidden_profile",
    }


def test_target_and_profile_selection_and_ambiguity(tmp_path):
    first = definition(tmp_path)
    source = catalog(tmp_path, first)
    assert D.for_target("runtime-device", catalog_path=source).id == "one"
    assert D.for_target("corpus-alias", catalog_path=source).id == "one"
    with pytest.raises(SpecError, match="found 0"):
        D.for_target("missing", catalog_path=source)
    second = definition(tmp_path, "two", profile="different")
    source = catalog(tmp_path, first, second)
    with pytest.raises(SpecError, match="found 2"):
        D.for_target("runtime-device", catalog_path=source)


def test_missing_explicit_recipe_refused(tmp_path):
    path = definition(tmp_path)
    document = yaml.safe_load(path.read_text())
    config = document["phases"][0]["config"]
    for name in ("recipe", "performance_template", "hidden_profile"):
        config.pop(name)
    path.write_text(yaml.safe_dump(document))
    with pytest.raises(SpecError, match="recipe"):
        D.from_definition(path)


def test_templates_skipped(tmp_path):
    path = definition(tmp_path)
    document = yaml.safe_load(path.read_text())
    document["kind"] = "template"
    path.write_text(yaml.safe_dump(document))
    assert D.all_declarations(catalog_path=catalog(tmp_path, path)) == ()


def _performance_captures(descriptor, tmp_path):
    """The descriptor's declared Phase 2 form-scale roster, one distinct capture path per label."""
    from merlin.targetgen.target_experiment import load_target_experiment

    declared = load_target_experiment(descriptor).workload_spec.get("performance_applications") or []
    return {label: tmp_path / "performance" / label / "model.mlir" for label in declared}


def test_requirement_derivation_selects_authored_capability_contract(tmp_path, monkeypatch):
    from merlin_experiments.phase0 import requirements

    from merlin.common.paths import repo_root
    from merlin.targetgen.target_experiment import load_target_experiment

    definition, descriptor = authored_inputs(monkeypatch)
    roster = load_target_experiment(descriptor).workload_spec["applications"]

    class ContractObserved(Exception):
        pass

    def observe(target, **kwargs):
        assert target == "gemmini"
        assert kwargs["capability_contract_path"] == (
            repo_root() / "examples/gemmini/target/contracts/target_contract.yaml"
        )
        raise ContractObserved

    monkeypatch.setattr(requirements, "select_evidence", observe)
    captures = {label: tmp_path / label / "model.mlir" for label in roster}
    with pytest.raises(ContractObserved):
        requirements.derive(
            definition,
            captures,
            rtl_facts=tmp_path / "facts.json",
            output_root=tmp_path / "derived",
            performance_captures=_performance_captures(descriptor, tmp_path),
        )


def test_requirement_derivation_observes_the_preselected_capture_python(tmp_path, monkeypatch):
    from merlin_experiments.phase0 import capture_execution_attestation, capture_selection, requirements

    from merlin.targetgen.target_experiment import load_target_experiment

    definition, descriptor = authored_inputs(monkeypatch)
    roster = load_target_experiment(descriptor).workload_spec["applications"]
    interpreter = tmp_path / "capture-venv" / "bin" / "python"
    monkeypatch.setattr(capture_selection, "verify", lambda *args, **kwargs: {"status": "verified_preselected_replay"})
    monkeypatch.setattr(capture_execution_attestation, "attest_sealed_m2m", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        capture_selection,
        "load",
        lambda *args, **kwargs: {
            "run_dir": str(tmp_path / "selected-run"),
            "plan": {"venv": str(interpreter.parent.parent), "loader_sha256": "a" * 64},
        },
    )

    class PythonObserved(Exception):
        pass

    def observe(target, **kwargs):
        assert kwargs["capture_python"] == interpreter
        raise PythonObserved

    monkeypatch.setattr(requirements, "select_evidence", observe)
    captures = {label: tmp_path / label / "model.mlir" for label in roster}
    selections = {label: (tmp_path / label / "capture-selection.json", "a" * 64) for label in roster}
    with pytest.raises(PythonObserved):
        requirements.derive(
            definition,
            captures,
            rtl_facts=tmp_path / "facts.json",
            output_root=tmp_path / "derived",
            capture_preselections=selections,
            performance_captures=_performance_captures(descriptor, tmp_path),
        )


def test_performance_capture_selections_cover_the_roster_and_are_replay_verified(tmp_path, monkeypatch):
    from merlin_experiments.phase0 import capture_execution_attestation, capture_selection, requirements

    from merlin.targetgen.target_experiment import load_target_experiment

    definition, descriptor = authored_inputs(monkeypatch)
    roster = load_target_experiment(descriptor).workload_spec["applications"]
    captures = {label: tmp_path / label / "model.mlir" for label in roster}
    performance = _performance_captures(descriptor, tmp_path)
    assert performance, "the target declares a performance-scale roster"
    selections = {label: (tmp_path / "performance" / label / "selection.json", "b" * 64) for label in performance}
    first = sorted(selections)[0]
    with pytest.raises(ValueError, match="entire declared performance roster"):
        requirements.derive(
            definition,
            captures,
            rtl_facts=tmp_path / "facts.json",
            output_root=tmp_path / "derived",
            performance_captures=performance,
            performance_preselections={first: selections[first]},
        )
    verified, attested = [], []

    def verify(path, *, expected_sha256, model_path):
        verified.append((path, expected_sha256, model_path))
        return {"status": "verified_preselected_replay"}

    monkeypatch.setattr(capture_selection, "verify", verify)
    monkeypatch.setattr(
        capture_execution_attestation,
        "attest_sealed_m2m",
        lambda evidence, *, selection_path, model_path: attested.append(model_path) or {},
    )

    class Selected(Exception):
        pass

    def observe(target, **kwargs):
        raise Selected

    monkeypatch.setattr(requirements, "select_evidence", observe)
    with pytest.raises(Selected):
        requirements.derive(
            definition,
            captures,
            rtl_facts=tmp_path / "facts.json",
            output_root=tmp_path / "derived",
            performance_captures=performance,
            performance_preselections=selections,
        )
    assert sorted(verified) == sorted((path, sha, performance[label]) for label, (path, sha) in selections.items())
    assert sorted(attested) == sorted(performance.values())
