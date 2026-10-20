"""Separate historical qualification sources never replace the current queue."""

import contextvars
import json
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
from merlin_experiments import source_snapshot as SS
from merlin_experiments.phase1.feedback import capsule_qualification_replay as R
from merlin_experiments.phase1.feedback import capsule_qualification_snapshot as Q
from test_capsule_certification import selected as selected
from test_capsule_qualification_handoff import qualified as qualified
from test_capsule_qualification_replay import guarded_source


def source_snapshot(tmp_path, *, mutate=None):
    source = tmp_path / "selected-source"
    for module in Q._critical_modules():
        relative = module.__name__.replace(".", "/") + ".py"
        path = source / "code" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(Path(module.__file__).read_bytes())
    if mutate is not None:
        mutate(source)
    root = tmp_path / "verification-snapshot"
    seal = SS.create(
        source,
        root,
        output_root=tmp_path / "future-output",
        source_roots=("code",),
        python_roots=("code",),
        legacy_roots=(),
        root_files=(),
        runtime_links=False,
    )
    return root, Q._pin(seal)["sha256"]


@pytest.fixture
def selected_snapshot(qualified, monkeypatch, tmp_path):
    s, _, _ = qualified
    record = guarded_source(qualified, monkeypatch, tmp_path)
    root, digest = source_snapshot(tmp_path)
    return s, record, root, digest


def enter(row):
    _, record, root, digest = row
    return R.qualification_verification_snapshot(root, seal_sha256=digest, qualification=record["receipt"])


def test_separate_snapshot_runs_only_fixed_child_and_rechecks_result(selected_snapshot, monkeypatch):
    s, record, root, _ = selected_snapshot
    calls = []

    def guarded(snapshot, argv, **kwargs):
        calls.append((snapshot, argv, kwargs))
        assert snapshot == root
        assert argv[:3] == [sys.executable, "-m", R.__name__]
        assert json.loads(argv[3])["receipt"] == record["receipt"]
        return ["separate-frozen-child", *argv]

    monkeypatch.setattr(R.FP, "python_command", guarded)
    monkeypatch.setattr(
        R.FP, "inherited_python_command", lambda *_: pytest.fail("current queue must not verify old profile")
    )
    monkeypatch.setattr(
        R.owned_process, "run", lambda *a, **k: SimpleNamespace(returncode=0, stdout=json.dumps(record), stderr="")
    )
    with enter(selected_snapshot):
        assert R.verify(record["receipt"], descriptor=s.descriptor) == record
        assert R.verify(record["receipt"], descriptor=s.descriptor) == record
    assert len(calls) == 2  # Each use executes; no saved verdict replaces the child.
    assert Q._ACTIVE.get() is None


@pytest.mark.parametrize("fault", ["receipt", "command", "expired", "seal", "unsealed"])
def test_foreign_or_expired_selection_refuses_before_execution(selected_snapshot, monkeypatch, fault):
    _, record, _, _ = selected_snapshot
    with enter(selected_snapshot):
        copied = contextvars.copy_context()
        if fault == "receipt":
            with pytest.raises(ValueError, match="different receipt"):
                Q.selected_command({**record["receipt"], "sha256": "f" * 64}, [])
        elif fault == "command":
            with pytest.raises(ValueError, match="fixed original replay entry"):
                Q.selected_command(record["receipt"], [sys.executable, "-c", "raise SystemExit"])
        elif fault == "unsealed":
            monkeypatch.setattr(Q.FP, "_ACTIVE_CONTEXT", None)
            with pytest.raises(ValueError, match="sealed host owner"):
                Q.selected_command(record["receipt"], [])
            monkeypatch.setattr(Q.FP, "_ACTIVE_CONTEXT", {"seal": {"path": "fixture", "sha256": "a" * 64}})
        elif fault == "seal":
            selection = Q._ACTIVE.get()
            Q._ISSUED[selection]["seal_sha256"] = "f" * 64
            with pytest.raises(ValueError, match="seal changed"):
                Q.selected_command(record["receipt"], [])
            Q._ISSUED[selection]["seal_sha256"] = selected_snapshot[3]
    if fault == "expired":
        with pytest.raises(ValueError, match="live host selection"):
            copied.run(Q.revalidate_selected, record["receipt"])


@pytest.mark.parametrize("module", [module.__name__ for module in Q._critical_modules()])
def test_resealed_foreign_child_or_verifier_is_not_selected(qualified, monkeypatch, tmp_path, module):
    record = guarded_source(qualified, monkeypatch, tmp_path)

    def mutate(source):
        path = source / "code" / (module.replace(".", "/") + ".py")
        path.write_bytes(path.read_bytes() + b"\n# different selected source\n")

    root, digest = source_snapshot(tmp_path, mutate=mutate)
    with pytest.raises(ValueError, match="differs from its host"):
        with R.qualification_verification_snapshot(root, seal_sha256=digest, qualification=record["receipt"]):
            pytest.fail("foreign source entered")
    assert Q._ACTIVE.get() is None


def test_mid_call_snapshot_mutation_refuses_saved_success(selected_snapshot, monkeypatch):
    s, record, root, _ = selected_snapshot
    monkeypatch.setattr(R.FP, "python_command", lambda snapshot, argv, **k: argv)
    path = root / "code/merlin_experiments/phase1/feedback/capsule_qualification_replay.py"

    def execute(*args, **kwargs):
        path.chmod(0o600)
        path.write_bytes(path.read_bytes() + b"\n# changed during child\n")
        return SimpleNamespace(returncode=0, stdout=json.dumps(record), stderr="")

    monkeypatch.setattr(R.owned_process, "run", execute)
    with pytest.raises((ValueError, SS.SnapshotError)):
        with enter(selected_snapshot):
            R.verify(record["receipt"], descriptor=s.descriptor)
    assert Q._ACTIVE.get() is None


def test_child_refusal_does_not_fall_back_and_scope_unwinds(selected_snapshot, monkeypatch):
    s, record, _, _ = selected_snapshot
    monkeypatch.setattr(R.FP, "python_command", lambda snapshot, argv, **k: argv)
    monkeypatch.setattr(
        R.owned_process,
        "run",
        lambda *a, **k: SimpleNamespace(returncode=1, stdout=json.dumps(record), stderr="fresh refusal"),
    )
    with pytest.raises(ValueError, match="fresh refusal"):
        with enter(selected_snapshot):
            R.verify(record["receipt"], descriptor=s.descriptor)
    assert Q._ACTIVE.get() is None


def test_nested_selection_refuses_without_replacing_owner(selected_snapshot):
    with enter(selected_snapshot):
        original = Q._ACTIVE.get()
        with pytest.raises(ValueError, match="cannot be nested"):
            with enter(selected_snapshot):
                pytest.fail("nested scope entered")
        assert Q._ACTIVE.get() is original


@pytest.mark.parametrize("changed", ["environment", "bundle_manifest", "marker"])
def test_original_payload_scope_reopens_all_selection_metadata(tmp_path, monkeypatch, changed):
    import yaml

    workspace = tmp_path / "workspace"
    snapshot = tmp_path / "bundle_inputs"
    snapshot.mkdir()
    marker = snapshot / "snapshot.json"
    marker.write_text(json.dumps({"repo": str(tmp_path / "original-repo")}))
    env = tmp_path / "environment.yaml"
    env.write_text(
        yaml.safe_dump(
            {
                "workspace_path": str(workspace),
                "bundle_input_snapshot": {"path": str(snapshot), "manifest_sha256": Q._pin(marker)["sha256"]},
            }
        )
    )
    bundle = tmp_path / "input_bundle_manifest.yaml"
    bundle.write_text("bundle_id: original\n")
    record = {"environment": Q._pin(env), "bundle_manifest": Q._pin(bundle)}
    calls = []

    @contextmanager
    def original_scope(ws, selected, *, repo):
        calls.append((ws, selected, repo))
        yield

    monkeypatch.setattr(Q.BW, "verified_snapshot_scope", original_scope)
    with pytest.raises(ValueError, match="inputs changed"):
        with Q.original_payload_scope(record):
            path = {"environment": env, "bundle_manifest": bundle, "marker": marker}[changed]
            path.write_bytes(path.read_bytes() + b" ")
    assert calls == [(workspace, {"bundle_id": "original"}, tmp_path / "original-repo")]


def test_cleanup_exception_survives_simultaneous_snapshot_drift(selected_snapshot, monkeypatch):
    s, record, root, _ = selected_snapshot
    monkeypatch.setattr(R.FP, "python_command", lambda snapshot, argv, **k: argv)

    def execute(*args, **kwargs):
        path = root / "code/merlin_experiments/phase1/feedback/capsule_qualification_replay.py"
        path.chmod(0o600)
        path.write_bytes(path.read_bytes() + b"\n# drift\n")
        raise R.owned_process.CleanupUnknown("unreaped original child")

    monkeypatch.setattr(R.owned_process, "run", execute)
    with pytest.raises(R.owned_process.CleanupUnknown, match="unreaped") as caught:
        with enter(selected_snapshot):
            R.verify(record["receipt"], descriptor=s.descriptor)
    assert caught.value.__notes__
    assert Q._ACTIVE.get() is None
