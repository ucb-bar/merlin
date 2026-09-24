"""Explicit Phase 0 inputs bind file membership using the runner's existing plan pins.

No generator is launched. A synthetic installed source identity isolates these
input-boundary checks from concurrent source edits and optional hardware tools.
"""

import hashlib
import json
from copy import deepcopy

import pytest
import yaml
from merlin_experiments import SpecError, adapters, load_spec, runner
from merlin_experiments.phase0.profiles import synthesis_input_identity


@pytest.fixture
def authored(tmp_path, monkeypatch):
    monkeypatch.setenv("MERLIN_REPO_ROOT", str(tmp_path))
    monkeypatch.setenv("MERLIN_OUT_ROOT", str(tmp_path / "out"))
    entrypoint = tmp_path / "installed" / "merlin_experiments" / "phase0" / "__main__.py"
    entrypoint.parent.mkdir(parents=True)
    entrypoint.write_text("# synthetic installed identity; never executed\n")
    monkeypatch.setattr(adapters, "phase0_sources", lambda: (entrypoint, (entrypoint,)))
    monkeypatch.setattr(
        runner, "_phase0_source_inputs", lambda: (entrypoint, {"phase0:source:__main__.py": str(entrypoint)})
    )
    definitions = tmp_path / "authored"
    definitions.mkdir()
    (definitions / "target.yaml").write_text("target: fixture\n")
    (definitions / "recipe.yaml").write_text("capsules: []\n")
    (definitions / "performance.yaml").write_text("sweeps: []\n")
    (definitions / "conformance.yaml").write_text(
        "application_demands:\n  status: not_declared\n  coverage_status: not_applicable\n"
    )
    (definitions / "synth.yaml").write_text(
        yaml.safe_dump(
            {
                "provenance": {
                    "selected_inputs": synthesis_input_identity(
                        conformance_spec=definitions / "conformance.yaml",
                        recipe=definitions / "recipe.yaml",
                        descriptor=definitions / "target.yaml",
                    )
                },
                "capsules": [],
            }
        )
    )
    config = {
        "descriptor": "target.yaml",
        "recipe": "recipe.yaml",
        "performance_template": "performance.yaml",
        "conformance_spec": "conformance.yaml",
        "synth_profile": "synth.yaml",
        "smt_profile": "absent-smt.yaml",
        "hidden_profile": "absent-hidden.yaml",
    }
    definition = definitions / "experiment.yaml"

    def make(overrides=None):
        selected = dict(config) if overrides is None else overrides
        definition.write_text(
            yaml.safe_dump(
                {
                    "schema_version": 1,
                    "id": "fixture",
                    "target": "fixture",
                    "phases": {"0": {"adapter": "capsule_derivation", "config": selected}},
                }
            )
        )
        return runner.resolve_plan(load_spec(definition), run_dir=tmp_path / "run")

    return make, definitions, config


def freeze(plan):
    report = runner.preflight(plan)
    assert report["configuration_ready"], report
    return {**plan, "inputs": report["inputs"]}


@pytest.mark.parametrize("changed", ["recipe", "conformance", "descriptor"])
def test_preflight_rejects_stale_selected_synthesis(authored, changed):
    make, root, _ = authored
    name = {"recipe": "recipe.yaml", "conformance": "conformance.yaml", "descriptor": "target.yaml"}[changed]
    with (root / name).open("a") as stream:
        stream.write(
            "\nworkload_spec: {operators: [matmul]}\n" if changed == "descriptor" else "\n# reviewed input changed\n"
        )
    report = runner.preflight(make())
    assert not report["configuration_ready"]
    assert any("stale selected synthesis" in error for error in report["errors"])


def test_preflight_labels_digestless_selection_unverified(authored):
    make, root, _ = authored
    (root / "synth.yaml").write_text("provenance: {}\ncapsules: []\n")
    report = runner.preflight(make())
    assert report["phase0_synthesis"]["0"]["status"] == "unverified_legacy"
    assert not report["configuration_ready"]


def test_detailed_application_inventory_is_frozen_and_verified(authored):
    make, root, _ = authored
    detailed = {"status": "not_declared", "coverage_status": "not_applicable", "applications": {}, "n_operations": 0}
    digest = hashlib.sha256(json.dumps(detailed, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    (root / "inventory.json").write_text(json.dumps(detailed, indent=2) + "\n")
    (root / "conformance.yaml").write_text(
        yaml.safe_dump(
            {
                "application_demands": {
                    "status": "not_declared",
                    "coverage_status": "not_applicable",
                    "sidecar": "inventory.json",
                    "full_inventory_sha256": digest,
                }
            }
        )
    )
    profile = yaml.safe_load((root / "synth.yaml").read_text())
    profile["provenance"]["selected_inputs"] = synthesis_input_identity(
        conformance_spec=root / "conformance.yaml", recipe=root / "recipe.yaml", descriptor=root / "target.yaml"
    )
    (root / "synth.yaml").write_text(yaml.safe_dump(profile))
    frozen = freeze(make())
    assert frozen["phase0_operator_inputs"]["phase0:operator:application_demands_sidecar"]["present"]
    (root / "inventory.json").write_text(json.dumps(detailed, sort_keys=True) + "\n")
    with pytest.raises(SpecError, match="frozen input changed"):
        runner._verify_inputs(frozen)
    (root / "inventory.json").write_text(json.dumps({**detailed, "status": "incomplete"}) + "\n")
    assert not runner.preflight(make())["configuration_ready"]


def test_explicit_plan_preserves_absent_flags_and_no_directory_discovery(authored):
    make, root, _ = authored
    plan = make()
    argv = plan["phases"]["0"]["argv"]
    assert "--profiles-root" not in argv
    assert argv[argv.index("--hidden-profile") + 1] == str(root / "absent-hidden.yaml")
    assert plan["phase0_operator_inputs"]["phase0:operator:hidden_profile"] == {
        "path": str(root / "absent-hidden.yaml"),
        "present": False,
    }
    assert plan["phase0_operator_inputs"]["phase0:operator:smt_profile"] == {
        "path": str(root / "absent-smt.yaml"),
        "present": False,
    }
    assert "phase0:profiles" not in plan["input_paths"]
    runner._verify_inputs(freeze(plan))


@pytest.mark.parametrize("change", ["appear", "disappear", "bytes", "recipe", "performance"])
def test_frozen_plan_refuses_membership_and_byte_changes(authored, change):
    make, root, _ = authored
    frozen = freeze(make())
    if change == "appear":
        (root / "absent-hidden.yaml").write_text("capsules: []\n")
    elif change == "disappear":
        (root / "synth.yaml").unlink()
    else:
        name = {"bytes": "synth", "recipe": "recipe", "performance": "performance"}[change]
        (root / (name + ".yaml")).write_text("changed: true\n")
    with pytest.raises(SpecError, match="membership|frozen input changed"):
        runner._verify_inputs(frozen)


@pytest.mark.parametrize(
    "name", ["recipe", "performance_template", "conformance_spec", "synth_profile", "smt_profile", "hidden_profile"]
)
def test_command_cannot_redirect_or_duplicate_declared_input(authored, name):
    make, _, _ = authored
    frozen = freeze(make())
    argv = frozen["phases"]["0"]["argv"]
    flag = "--" + name.replace("_", "-")
    argv[argv.index(flag) + 1] += ".other"
    with pytest.raises(SpecError, match="differs from"):
        runner._verify_inputs(frozen)
    argv[argv.index(flag) + 1] = frozen["phases"]["0"]["inputs"][name]
    argv.extend([flag, argv[argv.index(flag) + 1]])
    with pytest.raises(SpecError, match="one explicit"):
        runner._verify_inputs(frozen)


def test_missing_membership_metadata_cannot_upgrade_new_explicit_plan(authored):
    make, _, _ = authored
    frozen = freeze(make())
    del frozen["phase0_operator_inputs"]
    with pytest.raises(SpecError, match="membership"):
        runner._verify_inputs(frozen)


@pytest.mark.parametrize(
    "flag",
    [
        "recipe",
        "performance-template",
        "conformance-spec",
        "synth-profile",
        "smt-profile",
        "hidden-profile",
        "profiles-root",
    ],
)
def test_equals_form_cannot_override_frozen_explicit_argument(authored, flag):
    make, _, _ = authored
    frozen = freeze(make())
    frozen["phases"]["0"]["argv"].append(f"--{flag}=/other/path")
    with pytest.raises(SpecError, match="frozen explicit|cannot use"):
        runner._verify_inputs(frozen)


def test_absent_sidecar_cannot_be_created_inside_mutable_run(authored):
    make, root, config = authored
    config["hidden_profile"] = str(root.parent / "run" / "future-sidecar.yaml")
    with pytest.raises(SpecError, match="overlaps frozen input"):
        make(config)


@pytest.mark.parametrize("change", ["rename", "remove", "add"])
def test_absent_declaration_identity_is_distinct_from_omission(authored, change):
    make, root, config = authored
    if change == "add":
        del config["hidden_profile"]
    frozen = freeze(make(config))
    command = frozen["phases"]["0"]
    argv = command["argv"]
    if change == "rename":
        command["inputs"]["hidden_profile"] = str(root / "other-absent.yaml")
        argv[argv.index("--hidden-profile") + 1] = command["inputs"]["hidden_profile"]
    elif change == "remove":
        del command["inputs"]["hidden_profile"]
        index = argv.index("--hidden-profile")
        del argv[index : index + 2]
    else:
        command["inputs"]["hidden_profile"] = str(root / "new-absent.yaml")
        argv.extend(["--hidden-profile", command["inputs"]["hidden_profile"]])
    with pytest.raises(SpecError, match="membership"):
        runner._verify_inputs(frozen)


def test_historical_profiles_root_keeps_original_identity_rules(authored):
    make, root, _ = authored
    profiles = root / "profiles"
    profiles.mkdir()
    (profiles / "fixture.yaml").write_text("capsules: []\n")
    plan = make({"descriptor": "target.yaml", "profiles_root": "profiles"})
    assert "phase0_operator_inputs" not in plan
    frozen = freeze(plan)
    original = deepcopy(frozen)
    runner._verify_inputs(frozen)
    assert frozen == original
    (profiles / "fixture.hidden.yaml").write_text("capsules: []\n")
    with pytest.raises(SpecError, match="frozen input changed"):
        runner._verify_inputs(frozen)


@pytest.mark.parametrize(
    "config",
    [
        {"recipe": "r"},
        {"performance_template": "p"},
        {"hidden_profile": "h"},
        {"recipe": "r", "performance_template": "p", "profiles_root": "profiles"},
        {"recipe": "r", "performance_template": "p", "comparison_manifest": True},
    ],
)
def test_invalid_mixed_or_partial_declarations_refuse(config):
    with pytest.raises(SpecError):
        adapters.ADAPTERS["capsule_derivation"].validate({"descriptor": "target.yaml", **config})
