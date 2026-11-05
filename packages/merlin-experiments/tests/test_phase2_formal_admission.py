"""Normal performance search reuses full qualification, never a capsule-only pass."""

import json
from types import SimpleNamespace

import pytest
import yaml
from merlin_experiments.phase1.feedback import certification, private_full_model_execution
from merlin_experiments.phase2 import functional_inputs as FI
from merlin_experiments.phase2.contracts import StageGateError, sha256_file


@pytest.fixture
def admission(tmp_path):
    programs = {"network_a": ["encode", "decode"], "network_b": ["model"]}
    executions = {
        model: {program: {"engine": "spike", "timeout_s": 1, "host_execution": {"atol": 0, "rtol": 0}}
                for program in stages}
        for model, stages in programs.items()
    }
    descriptor = tmp_path / "descriptor.yaml"
    descriptor.write_text(yaml.safe_dump({
        "target": "synthetic",
        "phase1_gates": {
            "private_full_models": {"required": True, "models": list(programs), "programs": programs},
            "whole_model": {"private_full_models": {"required": True, "programs": executions}},
        },
    }, sort_keys=False))
    target = SimpleNamespace(path=descriptor, descriptor_sha256=sha256_file(descriptor), target="synthetic")
    run = tmp_path / "run"
    run.mkdir()
    manifest = {"submission_sha256": "a" * 64, "completion": {"formal_grade_complete": True}}
    environment = {"task_scope": {"target": target.target},
                   "public_corpus_input": {"descriptor_sha256": target.descriptor_sha256}}
    (run / "run_manifest.yaml").write_text(yaml.safe_dump(manifest))
    (run / "environment.yaml").write_text(yaml.safe_dump(environment))
    return SimpleNamespace(functional=SimpleNamespace(run_dir=run, digest="a" * 64), target=target,
                           manifest=manifest, environment=environment, programs=programs)


def test_reopens_original_model_and_stage_obligations_without_execution(admission, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("admission must not execute a model")

    monkeypatch.setattr(private_full_model_execution, "run", forbidden)
    observed = []

    def original_grade(returncode, run, **kwargs):
        assert returncode == 0 and run == admission.functional.run_dir
        assert kwargs["required_models"] == tuple(admission.programs)
        assert kwargs["required_programs"] == {k: tuple(v) for k, v in admission.programs.items()}
        assert set(kwargs["execution_gate"]["programs"]) == set(admission.programs)
        assert kwargs["hidden_capsule_mode"] == "rtl"
        observed.append(kwargs)
        return {"complete": True, "failures": []}

    monkeypatch.setattr(certification, "_official_grade_result", original_grade)
    FI.require_formal_qualification(admission.functional, admission.target)
    assert len(observed) == 1


def test_capsule_success_without_formal_completion_refuses(admission):
    admission.manifest["completion"]["formal_grade_complete"] = False
    (admission.functional.run_dir / "run_manifest.yaml").write_text(yaml.safe_dump(admission.manifest))
    with pytest.raises(StageGateError, match="formal qualification"):
        FI.require_formal_qualification(admission.functional, admission.target)


def test_claimed_success_without_actual_private_evidence_refuses(admission):
    # Exercise the real formal reader: an asserted completion flag has neither
    # linked model evidence nor numerical execution and cannot authorize search.
    with pytest.raises(StageGateError, match="private_full_model_build_gate_incomplete"):
        FI.require_formal_qualification(admission.functional, admission.target)


def test_deferred_program_is_not_a_complete_functional_baseline(admission):
    document = yaml.safe_load(admission.target.path.read_bytes())
    document["phase1_gates"]["whole_model"]["private_full_models"]["programs"]["network_a"]["decode"] = {
        "deferred": "external engine unavailable"
    }
    admission.target.path.write_text(yaml.safe_dump(document, sort_keys=False))
    admission.target.descriptor_sha256 = sha256_file(admission.target.path)
    admission.environment["public_corpus_input"]["descriptor_sha256"] = admission.target.descriptor_sha256
    (admission.functional.run_dir / "environment.yaml").write_text(yaml.safe_dump(admission.environment))
    with pytest.raises(StageGateError, match="every required program"):
        FI.require_formal_qualification(admission.functional, admission.target)


@pytest.mark.parametrize("mutation", ["descriptor", "manifest", "environment", "candidate"])
def test_qualification_cannot_change_during_admission(admission, monkeypatch, mutation):
    def grade(*args, **kwargs):
        if mutation == "descriptor":
            with admission.target.path.open("a") as stream:
                stream.write("\n# changed\n")
        else:
            field = "environment" if mutation == "environment" else "manifest"
            document = getattr(admission, field)
            document["submission_sha256" if mutation == "candidate" else "changed"] = "b" * 64
            name = "environment.yaml" if field == "environment" else "run_manifest.yaml"
            (admission.functional.run_dir / name).write_text(json.dumps(document))
        return {"complete": True, "failures": []}

    monkeypatch.setattr(certification, "_official_grade_result", grade)
    with pytest.raises(StageGateError, match="changed during admission"):
        FI.require_formal_qualification(admission.functional, admission.target)


@pytest.mark.parametrize("required", [None, False])
def test_build_only_qualification_does_not_admit_normal_search(admission, required):
    document = yaml.safe_load(admission.target.path.read_bytes())
    if required is None:
        del document["phase1_gates"]["whole_model"]
    else:
        document["phase1_gates"]["whole_model"]["private_full_models"]["required"] = False
    admission.target.path.write_text(yaml.safe_dump(document, sort_keys=False))
    admission.target.descriptor_sha256 = sha256_file(admission.target.path)
    admission.environment["public_corpus_input"]["descriptor_sha256"] = admission.target.descriptor_sha256
    (admission.functional.run_dir / "environment.yaml").write_text(yaml.safe_dump(admission.environment))
    with pytest.raises(StageGateError, match="build and execution obligations"):
        FI.require_formal_qualification(admission.functional, admission.target)
