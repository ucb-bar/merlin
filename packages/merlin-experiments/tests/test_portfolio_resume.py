"""Dispatch bindings only: no verifier subprocess or checkpoint qualification."""

import copy
import json
import socket
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from merlin_experiments import source_snapshot as SS
from merlin_experiments.phase2 import contracts as C
from merlin_experiments.phase2 import host_policy as HP
from merlin_experiments.phase2 import portfolio_resume as R

MODULE = "merlin_experiments.phase2.portfolio_resume"


@pytest.fixture(autouse=True)
def no_execution(monkeypatch):
    def refuse(*args, **kwargs):
        pytest.fail("resume dispatch tests cannot launch processes or listeners")

    monkeypatch.setattr(subprocess, "Popen", refuse)
    monkeypatch.setattr(socket.socket, "bind", refuse)


@pytest.fixture
def case(tmp_path, monkeypatch):
    snapshot = tmp_path / "snapshot"
    source = snapshot / "python/merlin_experiments/phase2/portfolio_resume.py"
    controller = snapshot / "controller.py"
    contract = snapshot / "contract"
    shared = snapshot / "python/merlin"
    shared.mkdir(parents=True)
    source.parent.mkdir(parents=True)
    source.write_text("# sealed verifier fixture, never executed\n")
    controller.write_text("# sealed controller identity\n")
    identities = {"python/" + MODULE: str(source), "controller/global": str(controller)}
    for relative in HP.RESOURCE_FILES:
        path = contract / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n")
        identities["resource/contract/" + relative] = str(path)
    shared_source = shared / "synthetic.py"
    shared_source.write_text("# compiler shared source\n")
    files = {str(path.relative_to(snapshot)): C.sha256_file(path) for path in snapshot.rglob("*") if path.is_file()}
    receipt = {
        "schema": SS.SCHEMA,
        "files": files,
        "python_roots": ["python"],
        "legacy_roots": [],
        "legacy_names": [],
        "internal_aliases": {},
        "external_links": {},
        "selected_provider": None,
        "directories": sorted(str(path.relative_to(snapshot)) for path in snapshot.rglob("*") if path.is_dir()),
    }
    monkeypatch.setattr(SS, "verify", lambda root: copy.deepcopy(receipt))
    checkpoint = tmp_path / "stage/global_iterations/checkpoint.json"
    checkpoint.parent.mkdir(parents=True)
    document = {
        "source_snapshot": str(snapshot),
        "source_snapshot_files_sha256": C.document_sha256(files),
        "compiler_dependencies": {"shared_source_root": str(shared)},
        "host_verification_policy": {
            "schema": HP.SCHEMA,
            "identities": identities,
            "sources": {path: C.sha256_file(Path(path)) for path in identities.values()},
        },
    }
    dispatch = R.installed_dispatch(
        snapshot=snapshot,
        controller_source=controller,
        contract_root=contract,
        compiler_shared_source_root=shared,
    )
    launch = {"source_snapshot": str(snapshot), "checkpoint_verifier": dispatch}
    launch_path = checkpoint.parent.parent / "launch.json"
    checkpoint.write_text(json.dumps(document))
    launch_path.write_text(json.dumps(launch))
    return SimpleNamespace(
        root=tmp_path,
        snapshot=snapshot,
        source=source,
        controller=controller,
        contract=contract,
        shared=shared,
        receipt=receipt,
        checkpoint=checkpoint,
        document=document,
        launch=launch,
        launch_path=launch_path,
    )


def save(case):
    case.checkpoint.write_text(json.dumps(case.document))
    case.launch_path.write_text(json.dumps(case.launch))


def test_installed_dispatch_is_bound_to_checkpoint_owners(case):
    dispatch = case.launch["checkpoint_verifier"]
    assert dispatch == {
        "schema": "merlin.portfolio-checkpoint-dispatch.v1",
        "kind": "installed",
        "module": MODULE,
        "source": str(case.source),
        "controller_source": str(case.controller),
        "contract_root": str(case.contract),
        "compiler_shared_source_root": str(case.shared),
    }
    resolved = R.resolve(case.checkpoint)
    assert resolved.snapshot == case.snapshot
    assert resolved.document == case.document
    assert resolved.argv == (sys.executable, "-m", MODULE, "--checkpoint", str(case.checkpoint))
    assert resolved.environment["PYTHONPATH"] == str(case.snapshot / "python")


@pytest.mark.parametrize("field", ["controller_source", "contract_root", "compiler_shared_source_root"])
def test_other_launch_selection_refuses_even_with_identical_controller_bytes(case, field):
    replacement = case.snapshot / "other" / field
    if field == "controller_source":
        replacement.parent.mkdir()
        replacement.write_bytes(case.controller.read_bytes())
    else:
        replacement.mkdir(parents=True)
        if field == "contract_root":
            for relative in HP.RESOURCE_FILES:
                path = replacement / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes((case.contract / relative).read_bytes())
    case.launch["checkpoint_verifier"][field] = str(replacement)
    save(case)
    with pytest.raises(ValueError):
        R.resolve(case.checkpoint)


@pytest.mark.parametrize("mutation", ["missing", "kind", "module", "source", "snapshot", "membership"])
def test_missing_or_changed_installed_dispatch_never_falls_back_to_native(case, mutation):
    if mutation == "missing":
        case.launch.pop("checkpoint_verifier")
    elif mutation in {"kind", "module"}:
        case.launch["checkpoint_verifier"][mutation] = "unknown"
    elif mutation == "source":
        case.launch["checkpoint_verifier"]["source"] = str(case.controller)
    elif mutation == "snapshot":
        case.document["source_snapshot"] = str(case.root / "other-snapshot")
    else:
        case.document["source_snapshot_files_sha256"] = "0" * 64
    save(case)
    with pytest.raises(ValueError):
        R.resolve(case.checkpoint)


@pytest.mark.parametrize("field", ["source", "controller_source", "contract_root", "compiler_shared_source_root"])
def test_alias_and_escape_selections_refuse(case, field):
    original = Path(case.launch["checkpoint_verifier"][field])
    alias = case.snapshot / ("alias-" + field)
    alias.symlink_to(original, target_is_directory=original.is_dir())
    case.launch["checkpoint_verifier"][field] = str(alias)
    save(case)
    with pytest.raises(ValueError):
        R.resolve(case.checkpoint)


@pytest.mark.parametrize("selection", ["source", "controller", "resource"])
def test_selected_owner_bytes_must_match_checkpoint_and_snapshot(case, selection):
    path = {
        "source": case.source,
        "controller": case.controller,
        "resource": case.contract / HP.RESOURCE_FILES[0],
    }[selection]
    path.write_text("# changed selected bytes\n")
    with pytest.raises(ValueError):
        R.resolve(case.checkpoint)


def test_duplicate_sealed_module_owners_refuse(case):
    relative = "other_python/" + MODULE.replace(".", "/") + ".py"
    source = case.snapshot / relative
    source.parent.mkdir(parents=True)
    source.write_bytes(case.source.read_bytes())
    case.receipt["python_roots"].append("other_python")
    case.receipt["files"][relative] = C.sha256_file(source)
    case.document["source_snapshot_files_sha256"] = C.document_sha256(case.receipt["files"])
    save(case)
    with pytest.raises(ValueError, match="unique"):
        R.resolve(case.checkpoint)


def test_historical_native_layout_is_explicit_and_preserves_archived_import(case):
    case.launch.pop("checkpoint_verifier")
    save(case)
    layout = R.NativeVerifierLayout(
        controller_relative="controller.py",
        python_roots=("python",),
        module="archived_controller",
    )
    resolved = R.resolve(case.checkpoint, native_layout=layout)
    assert resolved.argv[:2] == (sys.executable, "-c")
    assert "import archived_controller as G" in resolved.argv[2]
    assert "G.consume_global_candidate" in resolved.argv[2]
    assert resolved.argv[-1] == str(case.checkpoint)
    assert resolved.document == case.document
    case.controller.write_text("# altered historical verifier\n")
    with pytest.raises(ValueError, match="changed"):
        R.resolve(case.checkpoint, native_layout=layout)


def test_malformed_installed_record_cannot_use_explicit_native_fallback(case):
    case.launch["checkpoint_verifier"]["module"] = "untrusted_module"
    save(case)
    layout = R.NativeVerifierLayout(
        controller_relative="controller.py",
        python_roots=("python",),
        module="archived_controller",
    )
    with pytest.raises(ValueError):
        R.resolve(case.checkpoint, native_layout=layout)


@pytest.mark.parametrize("returncode", [0, 7])
def test_verification_transport_preserves_retained_record_bytes(case, monkeypatch, returncode):
    before = {path: path.read_bytes() for path in (case.checkpoint, case.launch_path)}
    captured = []

    def frozen_command(snapshot, argv, *, verifier_source):
        assert snapshot == case.snapshot
        assert argv == (sys.executable, "-m", MODULE, "--checkpoint", str(case.checkpoint))
        assert verifier_source == Path(SS.__file__)
        return ["synthetic-frozen-transport", *argv]

    def transport(argv, **kwargs):
        assert argv[0] == "synthetic-frozen-transport"
        assert kwargs["timeout"] == 30
        assert kwargs["env"]["MERLIN_REPO_ROOT"] == str(case.snapshot)
        captured.append(argv)
        return SimpleNamespace(returncode=returncode, stderr="synthetic verifier refusal")

    monkeypatch.setattr(R.frozen_python, "python_command", frozen_command)
    monkeypatch.setattr(R.subprocess, "run", transport)
    if returncode:
        with pytest.raises(ValueError, match="synthetic verifier refusal"):
            R.verify(case.checkpoint)
    else:
        assert R.verify(case.checkpoint) == case.document
    assert len(captured) == 1
    assert {path: path.read_bytes() for path in before} == before
