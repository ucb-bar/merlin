"""Original-path verifier replay cannot change authority, inputs, or evidence."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from merlin_experiments.phase1.feedback import capsule_qualification_handoff as H
from merlin_experiments.phase1.feedback import capsule_qualification_replay as R
from test_capsule_certification import selected as selected
from test_capsule_qualification_handoff import external_fixture
from test_capsule_qualification_handoff import qualified as qualified


def guarded_source(qualified, monkeypatch, tmp_path):
    _, _, record = qualified
    frozen = tmp_path / "snapshot-verifier.py"
    frozen.write_bytes(Path(H.__file__).read_bytes())
    monkeypatch.setattr(H, "__file__", str(frozen))
    # These tests exercise the fixed replay operation; real transport is checked
    # separately using the existing sealed-Python integration fixture.
    monkeypatch.setattr(R.FP, "_ACTIVE_CONTEXT", {"seal": {"path": "fixture", "sha256": "a" * 64}})
    return record


def test_original_checked_source_runs_unchanged_verifier_with_original_path(qualified, monkeypatch, tmp_path):
    s, _, _ = qualified
    record = guarded_source(qualified, monkeypatch, tmp_path)
    assert R._verify_original(record["receipt"], s.descriptor) == record
    assert record["verifier_source"]["path"] != H.__file__


@pytest.mark.parametrize("fault", ["unsealed", "code", "observation", "dependency"])
def test_replay_refuses_nonidentical_or_unobserved_producer(qualified, monkeypatch, tmp_path, fault):
    s, _, _ = qualified
    record = guarded_source(qualified, monkeypatch, tmp_path)
    if fault == "unsealed":
        monkeypatch.setattr(R.FP, "_ACTIVE_CONTEXT", None)
    elif fault == "code":
        Path(H.__file__).write_text("raise AssertionError('different source')\n")
    elif fault == "observation":
        source = record["verification"]
        path = Path(source["path"])
        path.write_bytes(path.read_bytes() + b" ")
    else:
        other = tmp_path / "different-dependency.py"
        other.write_text("# changed dependency\n")
        monkeypatch.setattr(H.G, "__file__", str(other))
    with pytest.raises(ValueError):
        R._verify_original(record["receipt"], s.descriptor)


def test_delegation_uses_fixed_guarded_child_and_rejects_changed_result(qualified, monkeypatch, tmp_path):
    s, _, _ = qualified
    record = guarded_source(qualified, monkeypatch, tmp_path)
    commands = []
    monkeypatch.setattr(R.FP, "inherited_python_command", lambda argv: ["guarded", *argv])

    def execute(command, **kwargs):
        commands.append(command)
        assert kwargs == {"timeout": 180, "text": True}
        assert command[2:4] == ["-m", R.__name__]
        return SimpleNamespace(returncode=0, stdout=json.dumps(record), stderr="")

    monkeypatch.setattr(R.owned_process, "run", execute)
    assert R.verify(record["receipt"], descriptor=s.descriptor) == record
    assert len(commands) == 1
    monkeypatch.setattr(
        R.owned_process,
        "run",
        lambda *a, **k: SimpleNamespace(returncode=0, stdout=json.dumps({**record, "authoring_converged": True})),
    )
    with pytest.raises(ValueError, match="different qualification evidence"):
        R.verify(record["receipt"], descriptor=s.descriptor)


def test_failed_fresh_verifier_never_falls_back_to_saved_pass(qualified, monkeypatch, tmp_path):
    s, _, _ = qualified
    record = guarded_source(qualified, monkeypatch, tmp_path)
    monkeypatch.setattr(R.FP, "inherited_python_command", lambda argv: argv)
    monkeypatch.setattr(
        R.owned_process,
        "run",
        lambda *a, **k: SimpleNamespace(returncode=1, stdout=json.dumps(record), stderr="actual refusal"),
    )
    with pytest.raises(ValueError, match="actual refusal"):
        R.verify(record["receipt"], descriptor=s.descriptor)


def test_selected_original_verifier_reopens_new_run_aliases_and_candidate(qualified, monkeypatch, tmp_path):
    s, run, original = external_fixture(qualified, monkeypatch, tmp_path)
    record = guarded_source(qualified, monkeypatch, tmp_path)

    def execute(pointer, *, descriptor, selection):
        assert selection["context"] == {
            "target": s.context.target,
            "descriptor": str(s.context.descriptor),
            "repo": str(s.context.repo),
        }
        return R._verify_original(pointer, descriptor, selection)

    monkeypatch.setattr(R, "_delegate", execute)
    args = dict(
        context=s.context,
        candidate=run / "submission",
        capsules=s.public,
        hidden_capsules=s.hidden,
        contract=s.contract,
        facts=None,
    )
    assert R.selected(run, **args) == original == record
    (run / "submission/compiler.py").write_text("# different candidate\n")
    with pytest.raises(ValueError, match="compiler differs"):
        R.selected(run, **args)


def test_new_run_mutation_during_original_selection_is_refused(qualified, monkeypatch, tmp_path):
    s, run, record = external_fixture(qualified, monkeypatch, tmp_path)
    guarded_source(qualified, monkeypatch, tmp_path)

    def execute(*args, **kwargs):
        (run / "environment.yaml").write_text("changed: true\n")
        return record

    monkeypatch.setattr(R, "_delegate", execute)
    with pytest.raises(ValueError, match="selection changed"):
        R.selected(
            run,
            context=s.context,
            candidate=run / "submission",
            capsules=s.public,
            hidden_capsules=s.hidden,
            contract=s.contract,
            facts=None,
        )


def test_absent_selection_preserves_ordinary_unimported_path(monkeypatch, tmp_path):
    monkeypatch.delenv(H.ENV, raising=False)
    assert (
        R.selected(
            tmp_path, context=None, candidate=None, capsules=None, hidden_capsules=None, contract=None, facts=None
        )
        is None
    )
