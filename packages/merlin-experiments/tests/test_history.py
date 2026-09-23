"""Stored orchestration discovery uses existing receipts and never creates output."""

import hashlib
import json
import os

from merlin_experiments.history import runs


def record(root, target, experiment, name):
    directory = root / target / experiment / name
    directory.mkdir(parents=True)
    plan = {"target": target, "experiment": experiment, "evidence_authority": "fixture evidence only"}
    payload = json.dumps(plan).encode()
    (directory / "resolved-plan.json").write_bytes(payload)
    (directory / "orchestration.json").write_text(
        json.dumps({"plan_sha256": hashlib.sha256(payload).hexdigest(), "state": "completed", "attempts": []})
    )
    return directory


def test_missing_root_is_read_only(tmp_path):
    root = tmp_path / "missing"
    assert runs(root=root)["runs"] == []
    assert not root.exists()


def test_cli_lists_receipts_without_catalog(tmp_path, capsys):
    from merlin_experiments.cli import main

    record(tmp_path, "alpha", "functional", "run")
    assert main(["runs", "--root", str(tmp_path), "--target", "alpha"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["scope"] == "phase_orchestrations"
    assert result["runs"][0]["experiment"] == "functional"


def test_exact_filters_and_stable_cross_target_discovery(tmp_path):
    b = record(tmp_path, "beta", "functional", "run2")
    a = record(tmp_path, "alpha", "functional", "run1")
    record(tmp_path, "alpha", "performance", "run3")
    before = {p: p.read_bytes() for p in tmp_path.rglob("*.json")}
    result = runs(root=tmp_path, experiment="functional")
    assert [r["run_dir"] for r in result["runs"]] == [str(a), str(b)]
    assert result["problems"] == []
    assert len(runs(root=tmp_path, target="alpha")["runs"]) == 2
    assert runs(root=tmp_path, target="*")["runs"] == []
    assert {p: p.read_bytes() for p in tmp_path.rglob("*.json")} == before


def test_corrupt_incomplete_and_misplaced_records_are_visible(tmp_path):
    damaged = record(tmp_path, "alpha", "functional", "damaged")
    (damaged / "resolved-plan.json").write_text("{}")
    incomplete = record(tmp_path, "alpha", "functional", "incomplete")
    (incomplete / "orchestration.json").unlink()
    misplaced = record(tmp_path, "alpha", "functional", "misplaced")
    misplaced.rename(tmp_path / "alpha/functional/renamed")
    # A moved run name remains valid; target/experiment identities may not change.
    (tmp_path / "other").mkdir()
    (tmp_path / "alpha").rename(tmp_path / "other/alpha")
    result = runs(root=tmp_path / "other")
    assert len(result["runs"]) == 1
    assert len(result["problems"]) == 2
    (tmp_path / "other/alpha/functional").rename(tmp_path / "other/alpha/different")
    result = runs(root=tmp_path / "other")
    assert result["runs"] == []
    assert any("record identity differs" in row["error"] for row in result["problems"])


def test_aliases_and_native_runs_are_not_traversed(tmp_path):
    root = tmp_path / "runs"
    original = record(root, "alpha", "functional", "run")
    (root / "alias").symlink_to(root / "alpha", target_is_directory=True)
    (original.parent / "alias").symlink_to(original, target_is_directory=True)
    native = root / "alpha/native/run"
    native.mkdir(parents=True)
    (native / "run_record.json").write_text("{}")
    result = runs(root=root)
    assert len(result["runs"]) == 1
    assert result["problems"] == []


def test_discovery_does_not_read_special_files(tmp_path):
    directory = tmp_path / "alpha/functional/pipe"
    directory.mkdir(parents=True)
    os.mkfifo(directory / "resolved-plan.json")
    result = runs(root=tmp_path)
    assert result["runs"] == []
    assert "regular files" in result["problems"][0]["error"]


def test_cli_navigates_frozen_phase_handoffs_without_reading_live_inputs(tmp_path, capsys):
    from merlin_experiments.cli import main

    directory = tmp_path / "out" / "runs" / "alpha" / "functional" / "run"
    directory.mkdir(parents=True)
    descriptor = str(tmp_path / "private/descriptor.yaml")
    seal = str(tmp_path / "review/private/seal.json")
    plan = {
        "target": "alpha",
        "experiment": "functional",
        "definition": str(tmp_path / "experiment.yaml"),
        "frozen_at": "2026-09-23T00:00:00+00:00",
        "evidence_authority": "native evidence only",
        "inputs": {
            "definition": {"path": str(tmp_path / "experiment.yaml"), "sha256": "a" * 64},
            "phase0:operator:recipe": {"path": str(tmp_path / "recipe.yaml"), "sha256": "b" * 64},
            "phase1:descriptor": {"path": descriptor, "sha256": "c" * 64},
            "phase1:corpus_seal": {"path": seal, "sha256": "d" * 64},
        },
        "phase0_operator_inputs": {
            "phase0:operator:hidden_profile": {
                "path": str(tmp_path / "secret/hidden.yaml"),
                "present": True,
            }
        },
        "spec": {
            "phases": {
                "2": {
                    "config": {
                        "functional_run_id": "functional-1",
                        "functional_submission_sha256": "e" * 64,
                    }
                }
            }
        },
        "phases": {
            "0": {
                "adapter": "capsule_derivation",
                "inputs": {"recipe": str(tmp_path / "recipe.yaml")},
                "engine_output": str(directory / "phase0/capsules"),
            },
            "1": {
                "adapter": "capsule_bench",
                "inputs": {"descriptor": descriptor, "corpus_seal": seal},
                "engine_output": str(directory / "native-functional"),
            },
            "2": {"adapter": "measured_claims", "inputs": {}, "engine_output": str(directory / "phase2")},
        },
    }
    payload = json.dumps(plan).encode()
    (directory / "resolved-plan.json").write_bytes(payload)
    (directory / "orchestration.json").write_text(
        json.dumps(
            {
                "plan_sha256": hashlib.sha256(payload).hexdigest(),
                "state": "execution_succeeded",
                "attempts": [{"phase": "0", "state": "execution_succeeded", "output_sha256": "f" * 64}],
            }
        )
    )
    assert main(["lineage", str(directory)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["phases"]["0"]["handoff"]["generated_capsules_sha256"] == "f" * 64
    assert result["phases"]["1"]["handoff"]["selected_corpus_release"] == str(tmp_path / "review")
    assert result["phases"]["2"]["handoff"]["functional_submission_sha256"] == "e" * 64
    assert result["phases"]["0"]["handoff"]["private_profile_present_at_freeze"] is True
    assert "secret/hidden.yaml" not in json.dumps(result)
    assert result["qualification"] == "not_assessed_by_lineage"
    assert result["definition"]["identity"] == "frozen_input"
    assert not (tmp_path / "experiment.yaml").exists()
    assert main(["runs", "--root", str(tmp_path / "out" / "runs")]) == 0
    assert len(json.loads(capsys.readouterr().out)["runs"]) == 1
    (directory / "resolved-plan.json").write_bytes(payload + b" ")
    assert main(["lineage", str(directory)]) == 2
    assert "plan changed" in capsys.readouterr().err
