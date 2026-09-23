"""Real integer derivation and reviewed preparation using shared source fixtures."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import yaml
from merlin_experiments.runner import fingerprint


@pytest.fixture
def phase0_handoff(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "reviewed_corpus_fixture", Path(__file__).with_name("reviewed_corpus_fixtures.py")
    )
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    return helper.build_phase0_handoff(tmp_path)


def _derive(fixture):
    baseline = fingerprint(fixture["baseline"])
    result = fixture["cli"]("run", fixture["definition"], "--phase", "0", "--run-dir", fixture["run"])
    assert result.returncode == 0, result.stderr + result.stdout
    assert fingerprint(fixture["baseline"]) == baseline
    plan = json.loads((fixture["run"] / "resolved-plan.json").read_text())
    command = plan["phases"]["0"]
    assert command["argv"][1:3] == ["-m", "merlin_experiments.phase0"]
    assert Path(command["entrypoint"]).is_relative_to(fixture["installed"])
    source_pins = {name for name in plan["inputs"] if name.startswith("phase0:source:")}
    sources = (fixture["installed"] / "merlin_experiments/phase0").glob("*.py")
    assert source_pins == {"phase0:source:" + path.name for path in sources}
    assert {name for name in plan["inputs"] if name.startswith("phase0:startup:")} == {
        "phase0:startup:package",
        "phase0:startup:spec",
        "phase0:startup:specir_integration",
        "phase0:startup:provenance_binding",
    }
    config = yaml.safe_load(fixture["definition"].read_text())["phases"]["0"]["config"]
    if "profiles_root" in config:
        assert plan["inputs"]["phase0:profiles"]["path"] == str(fixture["profiles"])
    else:
        assert "phase0:profiles" not in plan["inputs"]
        assert plan["inputs"]["phase0:recipe"]["path"] == config["recipe"]
    record = json.loads((fixture["run"] / "orchestration.json").read_text())
    assert record["attempts"][-1]["output_sha256"] == fingerprint(command["engine_output"])
    golden = yaml.safe_load((Path(command["engine_output"]) / "isa/generated_member/golden.yaml").read_text())
    assert golden["golden_source"] == "merlin_tensor_int"
    assert golden["outputs"] and any(value for rows in golden["outputs"].values() for row in rows for value in row)
    return plan


@pytest.mark.parametrize("input_mode", ["legacy-directory", "explicit-files"])
def test_real_installed_shaped_derivation_receipt_and_prepare(phase0_handoff, input_mode):
    fixture = phase0_handoff
    if input_mode == "explicit-files":
        document = yaml.safe_load(fixture["definition"].read_text())
        config = document["phases"]["0"]["config"]
        del config["profiles_root"]
        config.update(
            recipe=str(fixture["profiles"] / "fixture-device.yaml"),
            performance_template=str(fixture["profiles"] / "_perf.yaml"),
            synth_profile=str(fixture["profiles"] / "absent-synth.yaml"),
        )
        fixture["definition"].write_text(yaml.safe_dump(document))
    shadow = fixture["workspace"] / "merlin_experiments"
    shadow.mkdir()
    (shadow / "__init__.py").write_text("raise RuntimeError('cwd package must not replace pinned implementation')\n")
    _derive(fixture)
    assert not (fixture["workspace"] / "merlin/contract/capsules/generate_corpus.py").exists()
    assert not (fixture["workspace"] / "merlin/experiments/capsule_bench/harness/selfcheck_shim.py").exists()
    result = fixture["cli"]("corpus", "prepare", fixture["run"], "--output", fixture["release"])
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["state"] == "awaiting_operator_review"
    assert report["counts"]["public_admitted"] == 2
    assert report["counts"]["hidden_admitted"] == 1
    assert "private_member_identity" not in result.stdout + result.stderr
    assert not (fixture["release"] / "private/seal.json").exists()
    emitted = fixture["run"] / "phase0/capsules/isa/generated_member/golden.yaml"
    staged = fixture["release"] / "payload/corpus/isa/generated_member/golden.yaml"
    assert emitted.read_bytes() == staged.read_bytes()


@pytest.mark.parametrize(
    "tamper",
    [
        "helper",
        "new_helper",
        "profile",
        "hidden_sidecar",
        "removed_sidecar",
        "output",
        "forged_provenance",
        "package_startup",
        "spec_startup",
        "specir_integration",
        "provenance_binding",
    ],
)
def test_real_phase0_handoff_refuses_changed_closure(phase0_handoff, tamper):
    fixture = phase0_handoff
    if tamper == "removed_sidecar":
        (fixture["profiles"] / "fixture-device.hidden.yaml").write_text("capsules: []\n")
    _derive(fixture)
    if tamper in {"helper", "new_helper"}:
        name = "writer.py" if tamper == "helper" else "unexpected.py"
        path = fixture["installed"] / "merlin_experiments/phase0" / name
        path.write_text((path.read_text() if path.exists() else "") + "\n# source membership/byte drift\n")
    elif tamper in {"profile", "hidden_sidecar"}:
        name = "fixture-device.yaml" if tamper == "profile" else "fixture-device.hidden.yaml"
        path = fixture["profiles"] / name
        path.write_text((path.read_text() if path.exists() else "capsules: []\n") + "\n# input drift\n")
    elif tamper == "specir_integration":
        path = fixture["installed"] / "merlin/integrations/specir.py"
        path.write_text(path.read_text() + "\n# integration source drift\n")
    elif tamper in {"package_startup", "spec_startup", "provenance_binding"}:
        relative = {
            "package_startup": "__init__.py",
            "spec_startup": "spec.py",
            "provenance_binding": "resources/legacy_entrypoints.json",
        }[tamper]
        path = fixture["installed"] / "merlin_experiments" / relative
        path.write_text(path.read_text() + "\n")
    elif tamper == "removed_sidecar":
        (fixture["profiles"] / "fixture-device.hidden.yaml").unlink()
    elif tamper == "output":
        (fixture["run"] / "phase0/capsules/isa/generated_member/golden.yaml").write_text("outputs: {}\n")
    else:
        manifest = fixture["run"] / "phase0/capsules/MANIFEST.yaml"
        document = yaml.safe_load(manifest.read_text())
        document["generated_by"] = "arbitrary-citation.py"
        manifest.write_text(yaml.safe_dump(document))
        # Even an updated output receipt cannot make an arbitrary provenance string
        # a supported binding. This is not a signature or hostile-host security claim.
        record_path = fixture["run"] / "orchestration.json"
        record = json.loads(record_path.read_text())
        record["attempts"][-1]["output_sha256"] = fingerprint(manifest.parent)
        record_path.write_text(json.dumps(record))
    result = fixture["cli"]("corpus", "prepare", fixture["run"], "--output", fixture["release"])
    assert result.returncode == 2
    assert not fixture["release"].exists()
    assert "private_member_identity" not in result.stdout + result.stderr


def test_installed_orchestration_requires_explicit_profile_input(phase0_handoff):
    fixture = phase0_handoff
    document = yaml.safe_load(fixture["definition"].read_text())
    del document["phases"]["0"]["config"]["profiles_root"]
    fixture["definition"].write_text(yaml.safe_dump(document))
    result = fixture["cli"]("preflight", fixture["definition"], "--phase", "0")
    assert result.returncode == 2
    assert "explicit profiles_root input" in result.stderr
    assert not fixture["run"].exists()


def test_real_hidden_integer_derivation_is_counted_not_publicly_named(phase0_handoff):
    fixture = phase0_handoff
    public = yaml.safe_load((fixture["profiles"] / "fixture-device.yaml").read_text())
    hidden_name = "derived_private_identity"
    hidden = dict(public["capsules"][0], name=hidden_name, cat="hidden", label="hidden")
    (fixture["profiles"] / "fixture-device.hidden.yaml").write_text(yaml.safe_dump({"capsules": [hidden]}))
    descriptor = fixture["workspace"] / "source-experiment/target_experiment.yaml"
    document = yaml.safe_load(descriptor.read_text())
    document["grading"]["hidden_capability_admission"] = {"source_capsules": 2, "admitted_capsules": 2}
    descriptor.write_text(yaml.safe_dump(document))
    _derive(fixture)
    manifest = (fixture["run"] / "phase0/capsules/MANIFEST.yaml").read_text()
    assert hidden_name not in manifest
    assert yaml.safe_load(manifest)["held_out"]["n_generated"] == 1
    result = fixture["cli"]("corpus", "prepare", fixture["run"], "--output", fixture["release"])
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["counts"]["hidden_admitted"] == 2
    assert hidden_name not in result.stdout + result.stderr
    emitted = fixture["run"] / "phase0/capsules/hidden" / hidden_name / "golden.yaml"
    staged = fixture["release"] / "payload/corpus/hidden" / hidden_name / "golden.yaml"
    assert staged.read_bytes() == emitted.read_bytes()


@pytest.mark.parametrize("tamper", ["entrypoint", "module", "argv", "import_path", "missing_pin"])
def test_module_provenance_requires_actual_resolved_frozen_implementation(phase0_handoff, tamper):
    fixture = phase0_handoff
    plan = _derive(fixture)
    command = plan["phases"]["0"]
    if tamper == "entrypoint":
        command["entrypoint"] = str(fixture["installed"] / "merlin_experiments/phase0/writer.py")
    elif tamper == "module":
        command["module"] = "arbitrary.generator"
    elif tamper == "argv":
        command["argv"][2] = "arbitrary.generator"
    elif tamper == "import_path":
        command["env"]["PYTHONSAFEPATH"] = ""
    else:
        del plan["inputs"]["phase0:source:writer.py"]
    # A self-consistent plan hash does not turn an unsupported binding into one
    # of the narrowly trusted module→historical-citation relationships.
    plan_path = fixture["run"] / "resolved-plan.json"
    plan_path.write_text(json.dumps(plan))
    record_path = fixture["run"] / "orchestration.json"
    record = json.loads(record_path.read_text())
    record["plan_sha256"] = fingerprint(plan_path)
    record_path.write_text(json.dumps(record))
    result = fixture["cli"]("corpus", "prepare", fixture["run"], "--output", fixture["release"])
    assert result.returncode == 2
    assert not fixture["release"].exists()
