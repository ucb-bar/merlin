"""Coarse pre-capture ordering and byte-binding checks; no runtime snapshot."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
from merlin_experiments import cli
from merlin_experiments.capture_execution import sealed_m2m
from merlin_experiments.capture_execution.sealed_static import _file_digest
from merlin_experiments.phase0 import capture_selection as selected
from merlin_experiments.phase0.requirements import capture_selection_specs, quantization_policy_specs


def _fixture(tmp_path, monkeypatch):
    library = tmp_path / "libfixture.so"
    library.write_bytes(b"selected system library")
    bwrap = tmp_path / "bwrap"
    bwrap.write_bytes(b"selected bubblewrap")
    roots = {name: tmp_path / name for name in ("m2m", "workload", "merlin", "schemas", "venv", "base")}
    for path in roots.values():
        path.mkdir()
    worker = roots["merlin"] / "targetgen/worker.py"
    worker.parent.mkdir()
    worker.write_bytes(b"selected worker")
    plan = {
        "schema": sealed_m2m.SCHEMA,
        "status": "plan_only",
        "m2m_root": str(roots["m2m"]),
        "workload_root": str(roots["workload"]),
        "merlin_root": str(roots["merlin"]),
        "schemas_root": str(roots["schemas"]),
        "venv": str(roots["venv"]),
        "base": str(roots["base"]),
        "worker": str(worker),
        "dtype": "fp32",
        "recipe": None,
        "system_libs": [str(library)],
        "max_snapshot_bytes": 15_000_000_000,
    }
    monkeypatch.setattr(sealed_m2m, "prepare_plan", lambda **_: dict(plan))
    monkeypatch.setattr(selected, "_bwrap_binary", lambda *_: bwrap)
    run, destination = tmp_path / "fresh-run", tmp_path / "selection"
    arguments = {
        "m2m_root": roots["m2m"],
        "workload_root": roots["workload"],
        "worker": worker,
        "venv": roots["venv"],
        "schemas_root": roots["schemas"],
        "run_dir": run,
        "output_dir": destination,
    }
    return plan, library, bwrap, run, destination, arguments


def test_preselection_precedes_fresh_issue_and_refuses_tampered_bytes(tmp_path, monkeypatch):
    plan, library, bwrap, run, destination, arguments = _fixture(tmp_path, monkeypatch)
    run.mkdir()
    with pytest.raises(ValueError, match="fresh"):
        selected.select(**arguments)
    run.rmdir()
    with pytest.raises(ValueError, match="checkpoint"):
        selected.select(**arguments, checkpoint=library)
    identity = selected.select(**arguments)
    path = Path(identity["path"])
    assert identity["phase0_admission"] == "not_granted"
    assert path.stat().st_mode & 0o077 == destination.stat().st_mode & 0o077 == 0
    document = selected.load(path, expected_sha256=identity["sha256"])
    assert document["checkpoint"] == {"kind": "none"}
    assert document["system_libraries"][0]["sha256"] == _file_digest(library)
    assert document["bwrap"]["sha256"] == _file_digest(bwrap)
    assert capture_selection_specs([f"workload={path}@{identity['sha256']}"]) == {
        "workload": (path, identity["sha256"])
    }
    calls = []

    def fake_issue(_plan, destination, **kwargs):
        calls.append((_plan, destination, kwargs))
        destination.mkdir()
        return destination / "sealed_m2m_pending.json"

    monkeypatch.setattr(sealed_m2m, "issue", fake_issue)
    monkeypatch.setattr(sealed_m2m, "prepare_plan", lambda **_: {**plan, "worker_sha256": "changed"})
    with pytest.raises(ValueError, match="source, runtime"):
        selected.issue(path, expected_sha256=identity["sha256"])
    assert calls == [] and not run.exists()
    monkeypatch.setattr(sealed_m2m, "prepare_plan", lambda **_: dict(plan))
    path.chmod(0o600)
    path.write_bytes(path.read_bytes() + b" ")
    path.chmod(0o400)
    with pytest.raises(ValueError, match="bytes differ"):
        selected.issue(path, expected_sha256=identity["sha256"])
    assert calls == [] and not run.exists()
    path.chmod(0o600)
    path.write_bytes(selected._json(document) + b"\n")
    path.chmod(0o400)
    selected.issue(path, expected_sha256=identity["sha256"])
    assert calls[0][2]["capture_selection_sha256"] == identity["sha256"]
    assert calls[0][2]["selected_system_libraries"] == document["system_libraries"]
    assert calls[0][2]["selected_bwrap_sha256"] == document["bwrap"]["sha256"]
    with pytest.raises(ValueError, match="already exists"):
        selected.issue(path, expected_sha256=identity["sha256"])


def test_independent_replay_binds_selection_and_keeps_admission_closed(tmp_path, monkeypatch):
    plan, library, bwrap, run, _, arguments = _fixture(tmp_path, monkeypatch)
    identity = selected.select(**arguments)
    path = Path(identity["path"])
    manifest = selected.load(path, expected_sha256=identity["sha256"])
    capture = run / "capture"
    capture.mkdir(parents=True)
    run.chmod(0o700)
    (capture / "model.mlir").write_bytes(b"module {}\n")
    (capture / "capture_receipt.json").write_bytes(b"{}\n")
    guest_library = run / "snapshots/guest-root" / library.relative_to("/")
    guest_library.parent.mkdir(parents=True)
    shutil.copy2(library, guest_library)
    pending = run / "sealed_m2m_pending.json"
    pending.write_text(
        json.dumps(
            {
                "schema": sealed_m2m.SCHEMA,
                "capture_selection_sha256": identity["sha256"],
                "plan": plan,
                "policy_sha256": manifest["sandbox_policy_sha256"],
                "bwrap_sha256": manifest["bwrap"]["sha256"],
                "issuer_sha256": manifest["issuer_source_sha256"],
            }
        )
    )
    monkeypatch.setattr(
        sealed_m2m,
        "replay_verify",
        lambda *_args, **_kwargs: {
            "status": "verified_sandbox_replay",
            "sealed_source_closure_replayed": True,
            "receipt_sha256": _file_digest(pending),
        },
    )
    result = selected.verify(path, expected_sha256=identity["sha256"], model_path=capture / "model.mlir")
    assert result["status"] == "verified_preselected_replay"
    assert result["phase0_admission"] == "not_granted"
    assert result["source_closure_verified"] is False
    guest_library.write_bytes(b"changed system library")
    with pytest.raises(ValueError, match="system library bytes"):
        selected.verify(path, expected_sha256=identity["sha256"], model_path=capture / "model.mlir")
    shutil.copy2(library, guest_library)
    doc = json.loads(pending.read_bytes())
    doc["capture_selection_sha256"] = "0" * 64
    pending.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="does not bind"):
        selected.verify(path, expected_sha256=identity["sha256"], model_path=capture / "model.mlir")


def test_derivation_cli_forwards_complete_preselection_without_upgrading_legacy(tmp_path, monkeypatch, capsys):
    from merlin_experiments.phase0 import requirements

    definition = tmp_path / "experiment.yaml"
    definition.write_text("schema_version: 1\n")
    model = tmp_path / "model.mlir"
    model.write_text("module {}\n")
    manifest = tmp_path / "capture-selection.json"
    manifest.write_text("{}\n")
    facts = tmp_path / "facts.json"
    facts.write_text("{}\n")
    received = []
    monkeypatch.setattr(
        requirements,
        "derive",
        lambda *args, **kwargs: (
            received.append((args, kwargs)) or {"status": "diagnostic", "phase0_admission": "not_granted"}
        ),
    )
    argv = [
        "corpus",
        "derive",
        str(definition),
        "--application-capture",
        f"iteration={model}",
        "--rtl-facts",
        str(facts),
        "--output",
        str(tmp_path / "derived"),
    ]
    assert cli.main(argv) == 0
    assert received[-1][1]["capture_preselections"] == {}
    assert received[-1][1]["quantization_policies"] == {}
    assert "not_granted" in capsys.readouterr().out
    digest = "a" * 64
    assert cli.main([*argv, "--application-capture-selection", f"iteration={manifest}@{digest}"]) == 0
    assert received[-1][1]["capture_preselections"] == {"iteration": (manifest, digest)}
    assert cli.main([*argv, "--application-quant-policy", f"iteration={manifest}@{digest}"]) == 0
    assert received[-1][1]["quantization_policies"] == {"iteration": (manifest, digest)}
    assert quantization_policy_specs([f"iteration={manifest}@{digest}"]) == {"iteration": (manifest, digest)}

    monkeypatch.undo()
    monkeypatch.setattr(requirements, "from_definition", lambda _: SimpleNamespace(descriptor=definition))
    monkeypatch.setattr(
        requirements, "load_target_experiment", lambda _: SimpleNamespace(workload_spec={"applications": ["iteration"]})
    )
    with pytest.raises(ValueError, match="entire declared iteration roster"):
        requirements.derive(
            definition,
            {"iteration": model},
            rtl_facts=facts,
            output_root=tmp_path / "never-written",
            capture_preselections={"other": (manifest, digest)},
        )
    assert not (tmp_path / "never-written").exists()


def test_installed_capture_cli_selects_schemas_from_its_worker_package(tmp_path, monkeypatch, capsys):
    from merlin.common import paths

    package = tmp_path / "site-packages/merlin"
    bundled = package / "_data/schemas"
    bundled.mkdir(parents=True)
    checkout_schemas = tmp_path / "checkout/merlin/schemas"
    checkout_schemas.mkdir(parents=True)
    monkeypatch.delenv("MERLIN_SCHEMAS_DIR", raising=False)
    monkeypatch.setattr(paths, "module_source_path", lambda _: package / "__init__.py")
    monkeypatch.setattr(paths, "schemas_dir", lambda: checkout_schemas)
    selected_arguments = []
    monkeypatch.setattr(selected, "select", lambda **kwargs: selected_arguments.append(kwargs) or {"sha256": "a" * 64})

    assert (
        cli.main(
            [
                "corpus",
                "capture",
                "select",
                "--m2m-root",
                str(tmp_path / "m2m"),
                "--workload-root",
                str(tmp_path / "workload"),
                "--venv",
                str(tmp_path / "venv"),
                "--run-dir",
                str(tmp_path / "run"),
                "--output",
                str(tmp_path / "selection"),
            ]
        )
        == 0
    )
    assert selected_arguments[0]["worker"] == package / "targetgen/_m2m_capture_worker.py"
    assert selected_arguments[0]["schemas_root"] == bundled
    assert '"sha256"' in capsys.readouterr().out


def test_selected_v2_timeout_is_bound_into_the_selection_and_reaches_issue(tmp_path, monkeypatch):
    plan, _library, _bwrap, run, destination, arguments = _fixture(tmp_path, monkeypatch)
    prepared = []

    def prepare(**kwargs):
        prepared.append(kwargs)
        timeout = kwargs.get("execution_timeout_seconds")
        return {**plan, **({"execution_timeout_seconds": timeout} if timeout is not None else {})}

    monkeypatch.setattr(sealed_m2m, "prepare_plan", prepare)
    default = selected.select(**arguments)
    default_doc = selected.load(Path(default["path"]), expected_sha256=default["sha256"])
    assert "execution_timeout_seconds" not in prepared[-1] or prepared[-1]["execution_timeout_seconds"] is None
    assert "execution_timeout_seconds" not in default_doc["plan"]
    timed_arguments = {**arguments, "run_dir": tmp_path / "timed-run", "output_dir": tmp_path / "timed-selection"}
    timed = selected.select(**timed_arguments, execution_timeout_seconds=900)
    timed_doc = selected.load(Path(timed["path"]), expected_sha256=timed["sha256"])
    assert timed_doc["schema"] == selected.SCHEMA
    assert timed_doc["plan"]["execution_timeout_seconds"] == 900
    assert timed["sha256"] != default["sha256"]
    # The sandbox policy the selection pins is the 900 s policy, not the historical 120 s one.
    command = sealed_m2m._command_v2(tmp_path / "timed-run/capture", dtype="fp32", recipe=False, options=None)
    assert timed_doc["sandbox_policy_sha256"] == sealed_m2m._policy(
        command, tmp_path / "timed-run/capture", replayable_logs=True, timeout_seconds=900
    )
    assert default_doc["sandbox_policy_sha256"] == sealed_m2m._policy(
        sealed_m2m._command_v2(run / "capture", dtype="fp32", recipe=False, options=None),
        run / "capture",
        replayable_logs=True,
    )
    # Tampering with the recorded timeout cannot keep the independently supplied identity.
    path = Path(timed["path"])
    forged = {**timed_doc, "plan": {**timed_doc["plan"], "execution_timeout_seconds": 14_400}}
    forged["plan_sha256"] = selected._digest(selected._json(forged["plan"]))
    path.chmod(0o600)
    path.write_bytes(selected._json(forged) + b"\n")
    path.chmod(0o400)
    with pytest.raises(ValueError, match="pre-execution identity"):
        selected.load(path, expected_sha256=timed["sha256"])
    path.chmod(0o600)
    path.write_bytes(selected._json(timed_doc) + b"\n")
    path.chmod(0o400)
    # Issue re-prepares with the selected timeout and passes the bound plan to the issuer.
    issued = []
    monkeypatch.setattr(
        sealed_m2m, "issue", lambda plan_, destination_, **kwargs: issued.append(plan_) or destination_ / "x"
    )
    selected.issue(path, expected_sha256=timed["sha256"])
    assert prepared[-1]["execution_timeout_seconds"] == 900
    assert issued[0]["execution_timeout_seconds"] == 900
