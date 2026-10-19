"""New selected criterion requires canonical static and numerical producers together."""

import json
import sys
import time
from copy import deepcopy
from types import ModuleType

import pytest
import yaml
from merlin_experiments.phase1.feedback import private_full_model_execution as PFX
from merlin_experiments.phase1.feedback import private_numerical_selection as Q

from merlin.compile.model_execution_inputs import file_sha256, strict_tree_sha256


@pytest.fixture
def selected(tmp_path, monkeypatch):
    pins = {}
    for name in ("models", "reference_manifest", "reference_sums"):
        path = tmp_path / name
        path.write_text("original sealed fixture bytes")
        pins[name] = {"path": str(path), "sha256": file_sha256(path)}
    config = tmp_path / "config.yaml"
    config.write_text("{}")
    root, run = tmp_path / "static", tmp_path / "native-run"
    root.mkdir()
    run.mkdir()
    (run / "descriptor").write_text("original selected descriptor")
    spec = tmp_path / "private.yaml"
    choice = {
        "schema": Q.SCHEMA,
        "mode": Q.MODE,
        "config": str(config),
        "static_root": str(root),
        **pins,
        "comparison": "quantized-reference",
        "output_scope": "all_declared",
        "float_policy": deepcopy(Q.POLICY),
        "kernel_exact": True,
        "deadline_unix": time.time() + 60,
    }
    spec.write_text(yaml.safe_dump({"numerical_qualification": choice}))
    submission = tmp_path / "compiler"
    submission.mkdir()
    (submission / "compiler.py").write_text("original fresh candidate")
    candidate = strict_tree_sha256(submission)["sha256"]
    static = {
        "target": "fixture",
        "passed": True,
        "candidate_tree_sha256": candidate,
        "private_spec_sha256": file_sha256(spec),
        "required_programs": {"model": ["prefill", "decode"]},
    }
    proof = {
        "schema": "merlin.phase1.selected_quantized_full_chain.v1",
        "mode": Q.MODE,
        "passed": True,
        "candidate_tree_sha256": candidate,
        "private_spec_sha256": file_sha256(spec),
        "models": {"model": {"steps": 33, "arrays": 99}},
    }
    numerical = ModuleType("merlin_experiments.phase1.feedback.private_chain_receipt")
    seen = []

    def verify(path, **kwargs):
        seen.append((path, kwargs))
        return deepcopy(proof)

    numerical.verify = verify
    chains = ModuleType("merlin_experiments.phase1.feedback.private_model_chain")
    chains.join = lambda *a, **k: deepcopy(static)
    monkeypatch.setitem(sys.modules, numerical.__name__, numerical)
    monkeypatch.setitem(sys.modules, chains.__name__, chains)
    from merlin_experiments.phase1 import feedback

    monkeypatch.setattr(feedback, "private_chain_receipt", numerical, raising=False)
    monkeypatch.setattr(feedback, "private_model_chain", chains, raising=False)
    (root / "static-index.json").write_text(
        json.dumps(
            {
                "schema": "merlin.phase1.native_static_chain_index.v1",
                "run_dir": str(run),
                "records": {"model": "actual producer"},
            }
        )
    )
    (root / "full-chain-qualification.json").write_text("{}")
    return spec, choice, submission, static, proof, seen, run


def test_absent_selection_preserves_legacy_mode(tmp_path):
    path = tmp_path / "legacy.yaml"
    path.write_text("models: []")
    assert Q.read(path) is None


@pytest.mark.parametrize("fault", ["mode", "missing", "changed_reference", "deadline", "finite"])
def test_selected_criterion_refuses_missing_or_changed_premises(selected, fault):
    spec, choice, *_ = selected
    if fault == "changed_reference":
        from pathlib import Path

        Path(choice["models"]["path"]).write_text("changed")
    elif fault == "missing":
        choice.pop("kernel_exact")
    elif fault == "deadline":
        choice["deadline_unix"] = float("nan")
    elif fault == "finite":
        choice["float_policy"]["finite_required"] = 1
    else:
        choice["mode"] = "host_execution_passed"
    spec.write_text(yaml.safe_dump({"numerical_qualification": choice}))
    with pytest.raises(ValueError):
        Q.read(spec)


def test_intake_and_completion_reopen_both_producers_without_cpu_twin_claim(selected):
    spec, _, submission, static, _, seen, run = selected
    built, numerical = Q.consume(
        run_dir=run, descriptor=run / "descriptor", submission=submission, private_spec_path=spec
    )
    assert built == static and numerical["host_execution"] == {"selected": False, "status": "not_selected"}
    assert PFX.complete(
        numerical,
        {"programs": static["required_programs"]},
        static=built,
        candidate_sha256=static["candidate_tree_sha256"],
    )
    assert len(seen) == 2


def test_no_pass_flag_survives_changed_canonical_proof_or_static_record(selected):
    spec, _, submission, static, proof, _, run = selected
    built, numerical = Q.consume(
        run_dir=run, descriptor=run / "descriptor", submission=submission, private_spec_path=spec
    )
    proof["passed"] = False
    assert not Q.complete(
        numerical,
        static=built,
        gate={"programs": static["required_programs"]},
        candidate_sha256=static["candidate_tree_sha256"],
    )
    assert not Q.complete(numerical, static=None, gate={}, candidate_sha256=static["candidate_tree_sha256"])


def test_completion_reopens_the_original_static_issuer(selected, monkeypatch):
    spec, _, submission, static, _, _, run = selected
    built, numerical = Q.consume(
        run_dir=run, descriptor=run / "descriptor", submission=submission, private_spec_path=spec
    )
    from merlin_experiments.phase1.feedback import private_model_chain

    def missing(*args, **kwargs):
        raise ValueError("original static issuance changed")

    monkeypatch.setattr(private_model_chain, "join", missing)
    assert not Q.complete(
        numerical,
        static=built,
        gate={"programs": static["required_programs"]},
        candidate_sha256=static["candidate_tree_sha256"],
    )


def test_missing_evidence_has_bounded_wait(tmp_path):
    with pytest.raises(ValueError, match="deadline expired"):
        Q._wait(tmp_path / "missing", time.time() - 1)
