from __future__ import annotations

import json
import subprocess
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from merlin.compare import paper_k1_orchestrator as orchestrator


def _controller_root(tmp_path: Path) -> Path:
    root = tmp_path / "controller"
    compare = root / "merlin" / "python" / "merlin" / "compare"
    compare.mkdir(parents=True)
    (compare / "paper_measurement_controller.py").write_text("# controller\n")
    (compare / "paper_k1_orchestrator.py").write_text("# orchestrator\n")
    (compare / "paper_model_abi_runner.c").write_text("/* runner */\n")
    (compare / "paper_k1_board_probe.c").write_text("/* probe */\n")
    schemas = root / "merlin" / "schemas"
    schemas.mkdir(parents=True)
    (schemas / "paper.yaml").write_text("type: object\n")
    return root


def _contract(tmp_path: Path, *, run_id: str = "run-0", model: str = "gemma2_2b") -> Path:
    root = tmp_path / run_id
    root.mkdir()
    cell = {"model": model, "backend": "merlin_frozen",
            "precision": "w8a8", "core_count": 4}
    result_identity = {
        "timestamp": "20260831T000000Z", "git_sha": "deadbee", "study_label": "study",
        "target": "k1", "model": model, "checkpoint": "checkpoint", "fidelity": "full",
        "backend": "merlin_frozen", "runtime": "merlin", "precision": "w8a8",
        "quantization": "static_w8a8", "core_count": 4,
    }
    tool = root / "build-tool"
    header = bytearray(64)
    header[:6] = b"\x7fELF\x02\x01"
    header[18:20] = (243).to_bytes(2, "little")
    tool.write_bytes(header)
    tool.chmod(0o755)
    contract = {
        "target": "k1", "run_id": run_id, "study_sha256": "a" * 64,
        "cell": cell, "timeout_seconds": 30,
        "result_identity": result_identity, "artifact_sha256": "9" * 64,
        "session": {"kind": "image_stream"},
        "frozen_provenance": {"study_sha256": "a" * 64,
                              "compiler_policy_sha256": "8" * 64},
        "build": {"tool": {"path": "build-tool",
                            "sha256": orchestrator._sha_file(tool)}},
    }
    path = root / "measurement_contract.yaml"
    path.write_text(yaml.safe_dump(contract), encoding="utf-8")
    (root / "frozen-resource.bin").write_bytes(b"digest-bound contract input")
    return path


def _fake_contract(path: Path):
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    return dict(value), dict(value["cell"])


def _plan(tmp_path: Path, monkeypatch) -> tuple[Path, dict]:
    monkeypatch.setattr(orchestrator, "_contract", _fake_contract)
    path = orchestrator.create_matrix_plan(
        [_contract(tmp_path)], tmp_path / "matrix.json",
        controller_root=_controller_root(tmp_path))
    return path, json.loads(path.read_text(encoding="utf-8"))


def test_plan_binds_complete_contract_and_controller_trees(tmp_path: Path, monkeypatch) -> None:
    plan_path, plan = _plan(tmp_path, monkeypatch)

    loaded = orchestrator._load_plan(plan_path)

    assert loaded["status"] == "frozen"
    assert loaded["study_sha256"] == "a" * 64
    assert loaded["cells"][0]["contract_path"] == "measurement_contract.yaml"
    assert loaded["cells"][0]["tree_files"] == 3
    assert loaded["controller"]["tree_files"] == 5
    assert len(loaded["matrix_sha256"]) == 64
    resource = Path(plan["cells"][0]["contract_root"]) / "frozen-resource.bin"
    resource.write_bytes(b"post-plan edit")
    with pytest.raises(ValueError, match="contract tree changed"):
        orchestrator._load_plan(plan_path)


def test_systemd_command_serializes_board_and_runs_controller_on_remote_host(
        tmp_path: Path) -> None:
    commands: list[list[str]] = []

    def runner(argv, **_kwargs):
        commands.append(list(argv))
        return subprocess.CompletedProcess(argv, 0, "terminal\n", "")

    transport = orchestrator.K1SSHSystemdTransport(
        orchestrator.SSHConfig(host="192.0.2.8"), runner=runner)

    evidence = transport.run_cell(
        remote_controller="/cache/controller", remote_contract="/cache/contract",
        remote_run_root="/runs/cell", contract_tree_sha256="b" * 64,
        unit_name="merlin-paper-matrix-000", timeout_seconds=30)

    command = commands[0][-1]
    assert "/usr/bin/systemd-run" in command
    assert "--wait" in command and "--collect" in command
    assert "/usr/bin/flock -w 150 /run/lock/merlin-paper-k1.lock" in command
    assert "merlin.compare.paper_k1_orchestrator remote-cell" in command
    assert "PYTHONPATH=/cache/controller/repo/merlin/python" in command
    assert evidence["unit"] == "merlin-paper-matrix-000"


def test_ssh_stage_names_cache_by_tree_and_verifies_uploaded_archive_digest(
        tmp_path: Path) -> None:
    commands: list[list[str]] = []

    def runner(argv, **_kwargs):
        commands.append(list(argv))
        return subprocess.CompletedProcess(argv, 0, "", "")

    archive = tmp_path / "contract.tar"
    archive.write_bytes(b"frozen archive")
    archive_sha = orchestrator._sha_file(archive)
    tree_sha = "7" * 64
    transport = orchestrator.K1SSHSystemdTransport(
        orchestrator.SSHConfig(host="192.0.2.8"), runner=runner)

    remote = transport.stage(
        archive, archive_sha256=archive_sha, tree_sha256=tree_sha, kind="contract")

    assert remote == f"/var/lib/merlin-paper-k1/cache/contract/{tree_sha}"
    assert any(command[0] == "scp" and str(archive) in command for command in commands)
    verification = commands[-1][-1]
    assert "/usr/bin/sha256sum" in verification
    assert archive_sha in verification and tree_sha in verification


def _environment_receipt(matrix_sha256: str, controller_tree_sha256: str,
                         runtime_requirements_sha256: str,
                         required_core_count: int) -> bytes:
    file_row = {"path": "/identity", "resolved_path": "/identity",
                "sha256": "1" * 64, "size": 1}
    modules = {
        name: {**file_row, "name": name, "import_name": imported}
        for name, imported in orchestrator._REMOTE_MODULES.items()
    }
    tools = {
        name: {**file_row, "path": path, "resolved_path": path}
        for name, path in orchestrator._REMOTE_TOOLS.items()
    }
    frequencies = [{"core_id": core, "governor": "performance",
                    "current_khz": 1600000, "max_khz": 1600000}
                   for core in range(required_core_count)]
    document = {
        "schema_version": 1, "kind": "paper_k1_environment_preflight_v1",
        "status": "ready", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "matrix_sha256": matrix_sha256,
        "controller_tree_sha256": controller_tree_sha256,
        "runtime_requirements_sha256": runtime_requirements_sha256,
        "machine": "riscv64", "python": {
            **file_row, "path": "/usr/bin/python3", "resolved_path": "/usr/bin/python3",
            "version": "3.12", "implementation": "CPython",
        },
        "modules": modules, "tools": tools,
        "runtime": {
            "required_core_count": required_core_count,
            "available_affinity": list(range(required_core_count)),
            "procfs_task_state": True, "systemd_manager": True,
            "systemd_version_sha256": "2" * 64, "openssl_ed25519": True,
            "board_probe_source_sha256": "3" * 64,
            "board_probe_executable_sha256": "4" * 64,
            "board_probe": {"kind": "merlin_board_probe_v1", "vlen_source": "csr",
                            "governor": "performance"},
            "core_frequencies": frequencies,
            "mapped_libraries": [{"path": "/lib/ld-linux-riscv64-lp64d.so.1",
                                  "sha256": "5" * 64, "size": 1}],
        },
    }
    return (json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n").encode()


class _FakeTransport:
    def __init__(self, *, board_terminal: bool = False, corrupt: bool = False):
        self.board_terminal = board_terminal
        self.corrupt = corrupt
        self.stages: list[str] = []
        self.events: list[str] = []
        self.preflight_calls = 0
        self.run_calls = 0
        self.retrieve_calls = 0

    def stage(self, _archive: Path, *, archive_sha256: str, tree_sha256: str,
              kind: str) -> str:
        assert len(archive_sha256) == len(tree_sha256) == 64
        self.stages.append(kind)
        self.events.append(f"stage:{kind}")
        return f"/cache/{kind}/{tree_sha256}"

    def environment_preflight(self, *, remote_controller: str, remote_output: str,
                              matrix_sha256: str, controller_tree_sha256: str,
                              runtime_requirements_sha256: str,
                              required_core_count: int) -> bytes:
        assert remote_controller.startswith("/cache/controller/")
        assert remote_output.endswith("/environment-preflight.json")
        self.preflight_calls += 1
        self.events.append("environment-preflight")
        return _environment_receipt(
            matrix_sha256, controller_tree_sha256, runtime_requirements_sha256,
            required_core_count)

    def terminal_exists(self, _remote_run_root: str) -> bool:
        self.events.append("terminal-exists")
        return self.board_terminal

    def run_cell(self, **_kwargs):
        self.run_calls += 1
        self.board_terminal = True
        return {"unit": "synthetic", "started_monotonic_ns": 1,
                "ended_monotonic_ns": 2, "stdout_tail": "", "stderr_tail": ""}

    def retrieve(self, _remote_run_root: str, destination: Path) -> str:
        self.retrieve_calls += 1
        if self.corrupt:
            destination.write_bytes(b"not a tar")
            return orchestrator._sha_file(destination)
        source = destination.parent / "fake-remote"
        (source / "output").mkdir(parents=True)
        (source / ".paper-controller-issuance-v1").mkdir()
        (source / "terminal.json").write_text("{}\n")
        (source / "result.yaml").write_text("run_id: run-0\n")
        (source / "output" / "receipt.yaml").write_text("receipt\n")
        (source / ".paper-controller-issuance-v1" / "entry").write_text("issuance\n")
        with tarfile.open(destination, "w") as stream:
            for name in ("terminal.json", "result.yaml", "output",
                         ".paper-controller-issuance-v1"):
                stream.add(source / name, arcname=name)
        return orchestrator._sha_file(destination)


def _verified(_transport_root: Path, planned, *, retrieval_sha256: str, final_dir: Path):
    result = {
        "run_id": planned["run_id"], "measurement_receipt": {
            "path": str(final_dir / "transport/output/receipt.yaml"), "sha256": "c" * 64,
        },
    }
    return {
        "terminal": {}, "localized_result": result, "issuance_fingerprint": "d" * 64,
        "receipt_sha256": "c" * 64, "remote_result_sha256": "e" * 64,
        "retrieval_archive_sha256": retrieval_sha256,
    }


def _resume_state(cell_dir: Path, _planned):
    return json.loads((cell_dir / "terminal-state.json").read_text(encoding="utf-8"))


def test_matrix_is_sequential_atomic_and_resume_never_contacts_board_for_terminal_cell(
        tmp_path: Path, monkeypatch) -> None:
    plan_path, _ = _plan(tmp_path, monkeypatch)
    monkeypatch.setattr(orchestrator, "_validate_retrieved", _verified)
    monkeypatch.setattr(orchestrator, "_validate_local_terminal", _resume_state)
    first = _FakeTransport()
    output = orchestrator.run_matrix(
        plan_path, transport=first, output_dir=tmp_path / "run")

    assert first.stages == ["controller", "contract"]
    assert first.preflight_calls == 1
    assert first.events.index("environment-preflight") < first.events.index("terminal-exists")
    assert first.run_calls == 1 and first.retrieve_calls == 1
    notary = yaml.safe_load((output / "issuance-notary.yaml").read_text(encoding="utf-8"))
    assert notary == {
        "schema_version": 1, "kind": "paper_external_issuance_notary_v1",
        "study_sha256": "a" * 64, "fingerprints": {"run-0": "d" * 64},
    }
    terminal_dirs = list((output / "cells").iterdir())
    assert len(terminal_dirs) == 1
    assert (terminal_dirs[0] / "terminal-state.json").is_file()

    resumed = _FakeTransport()
    orchestrator.run_matrix(plan_path, transport=resumed, output_dir=output, resume=True)
    assert resumed.stages == []
    assert resumed.preflight_calls == 0
    assert resumed.run_calls == resumed.retrieve_calls == 0


def test_host_recovers_board_terminal_cell_without_rerunning_measurement(
        tmp_path: Path, monkeypatch) -> None:
    plan_path, _ = _plan(tmp_path, monkeypatch)
    monkeypatch.setattr(orchestrator, "_validate_retrieved", _verified)
    transport = _FakeTransport(board_terminal=True)

    orchestrator.run_matrix(plan_path, transport=transport, output_dir=tmp_path / "run")

    assert transport.preflight_calls == 1
    assert transport.stages == ["controller"]
    assert transport.run_calls == 0
    assert transport.retrieve_calls == 1


def test_corrupt_retrieval_never_becomes_a_terminal_cell_or_final_notary(
        tmp_path: Path, monkeypatch) -> None:
    plan_path, _ = _plan(tmp_path, monkeypatch)
    transport = _FakeTransport(corrupt=True)
    output = tmp_path / "run"

    with pytest.raises(tarfile.ReadError):
        orchestrator.run_matrix(plan_path, transport=transport, output_dir=output)

    assert not (output / "cells").exists()
    assert not (output / "issuance-notary.yaml").exists()
    state = json.loads((output / "matrix-state.json").read_text(encoding="utf-8"))
    assert state["status"] == "incomplete" and state["terminal_cells"] == 0


def test_plan_rejects_host_cross_compiler_for_board_local_rebuild(
        tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(orchestrator, "_contract", _fake_contract)
    contract = _contract(tmp_path)
    tool = contract.parent / "build-tool"
    header = bytearray(tool.read_bytes())
    header[18:20] = (62).to_bytes(2, "little")  # EM_X86_64
    tool.write_bytes(header)
    value = yaml.safe_load(contract.read_text(encoding="utf-8"))
    value["build"]["tool"]["sha256"] = orchestrator._sha_file(tool)
    contract.write_text(yaml.safe_dump(value), encoding="utf-8")

    with pytest.raises(ValueError, match="cross-compiler cannot execute"):
        orchestrator.create_matrix_plan(
            [contract], tmp_path / "plan.json", controller_root=_controller_root(tmp_path))


class _BadEnvironmentTransport(_FakeTransport):
    def environment_preflight(self, **kwargs) -> bytes:
        self.preflight_calls += 1
        self.events.append("environment-preflight")
        value = json.loads(super().environment_preflight(**kwargs))
        value["modules"].pop("aet")
        return (json.dumps(value) + "\n").encode()


def test_invalid_remote_dependency_receipt_blocks_before_any_cell_contact(
        tmp_path: Path, monkeypatch) -> None:
    plan_path, _ = _plan(tmp_path, monkeypatch)
    transport = _BadEnvironmentTransport()
    output = tmp_path / "run"

    with pytest.raises(ValueError, match="module roster"):
        orchestrator.run_matrix(plan_path, transport=transport, output_dir=output)

    assert transport.run_calls == transport.retrieve_calls == 0
    assert "terminal-exists" not in transport.events
    failure = json.loads((output / "environment-preflight-failure.json").read_text())
    assert failure["status"] == "blocked_before_first_cell"
    assert not (output / "environment-preflight.json").exists()


def test_prepare_materializes_frozen_order_with_one_aet_parent_and_never_executes(
        tmp_path: Path, monkeypatch) -> None:
    from merlin.compare import paper, paper_contract_registry, study

    models = [SimpleNamespace(name="m0"), SimpleNamespace(name="m1")]
    backend = SimpleNamespace(name="b0")
    cells = [SimpleNamespace(model=model, backend=backend, precision="fp32", core_count=1,
                             key=f"{model.name}/b0/fp32/1c") for model in models]
    preflight = SimpleNamespace(ready=True, errors=(), blockers=())
    spec = SimpleNamespace(
        target="k1", label="frozen-test", source_path=None,
        preflight=lambda: preflight, matrix=lambda: tuple(cells),
        sha256=lambda: "a" * 64,
    )
    source = tmp_path / "study.frozen.yaml"
    source.write_text("status: frozen\n", encoding="utf-8")
    aet_root = tmp_path / "aet" / "parent"
    aet_root.mkdir(parents=True)
    (aet_root / "run_record.json").write_text("{}\n")
    handle = SimpleNamespace(
        run_id="aet-parent", run_dir=aet_root, timestamp="20260831T000000Z",
        git_sha="deadbee")
    finished = []
    monkeypatch.setattr(paper.PaperStudySpec, "from_yaml", staticmethod(lambda _path: spec))
    monkeypatch.setattr(study, "execution_matrix", lambda _spec: tuple(cells))
    monkeypatch.setattr(study, "_base_result", lambda *_args: {})
    monkeypatch.setattr(orchestrator, "start_run", lambda **_kwargs: handle)
    monkeypatch.setattr(
        orchestrator, "finish_run", lambda _handle, status, summary: finished.append((status, summary)))
    monkeypatch.setattr(orchestrator, "_contract", _fake_contract)

    def build(_spec, cell, *, run_id, timestamp, git_sha, staging_dir, base_result):
        assert timestamp == handle.timestamp and git_sha == handle.git_sha and base_result == {}
        staging_dir.mkdir(parents=True)
        tool = staging_dir / "build-tool"
        header = bytearray(64)
        header[:6] = b"\x7fELF\x02\x01"
        header[18:20] = (243).to_bytes(2, "little")
        tool.write_bytes(header)
        tool.chmod(0o755)
        contract = {
            "target": "k1", "run_id": run_id, "study_sha256": "a" * 64,
            "cell": {"model": cell.model.name, "backend": cell.backend.name,
                     "precision": cell.precision, "core_count": cell.core_count},
            "timeout_seconds": 30,
            "build": {"tool": {"path": "build-tool",
                                "sha256": orchestrator._sha_file(tool)}},
        }
        path = staging_dir / "measurement_contract.yaml"
        path.write_text(yaml.safe_dump(contract), encoding="utf-8")
        return path

    monkeypatch.setattr(paper_contract_registry, "build_registered_contract", build)
    monkeypatch.setattr(
        orchestrator, "produce_receipt",
        lambda *_args, **_kwargs: pytest.fail("prepare must not execute a measurement"))
    output = orchestrator.prepare_contract_matrix(source, output_dir=tmp_path / "prepared")

    plan = json.loads((output / "k1-matrix-plan.json").read_text())
    assert [row["run_id"] for row in plan["cells"]] == [
        "aet-parent__cell000", "aet-parent__cell001"]
    assert [row["cell"]["model"] for row in plan["cells"]] == ["m0", "m1"]
    assert all((Path(row["contract_root"]) / "measurement_contract.yaml").is_file()
               for row in plan["cells"])
    prepared = yaml.safe_load((output / "prepared-matrix.yaml").read_text())
    assert prepared["aet_parent"]["run_id"] == "aet-parent"
    assert prepared["matrix_plan"]["matrix_sha256"] == plan["matrix_sha256"]
    assert finished == [("ok", {"prepared": True, "n_cells": 2,
                                 "matrix_sha256": plan["matrix_sha256"]})]


def _finalization_fixture(tmp_path: Path, monkeypatch):
    from merlin.compare import paper, paper_attribution, paper_report, study

    monkeypatch.setattr(orchestrator, "_contract", _fake_contract)
    contracts = [
        _contract(tmp_path, run_id="run-0", model="model-0"),
        _contract(tmp_path, run_id="run-1", model="model-1"),
    ]
    plan_path = orchestrator.create_matrix_plan(
        contracts, tmp_path / "matrix.json", controller_root=_controller_root(tmp_path))
    plan = json.loads(plan_path.read_text())
    run = tmp_path / "run"
    run.mkdir()
    (run / "matrix-plan.json").write_bytes(plan_path.read_bytes())
    requirements_sha = orchestrator._canonical_sha(plan["runtime_requirements"])
    environment = _environment_receipt(
        plan["matrix_sha256"], plan["controller"]["tree_sha256"], requirements_sha,
        plan["runtime_requirements"]["required_core_count"])
    (run / "environment-preflight.json").write_bytes(environment)
    fingerprints = {"run-0": "d" * 64, "run-1": "e" * 64}
    notary = {"schema_version": 1, "kind": "paper_external_issuance_notary_v1",
              "study_sha256": "a" * 64, "fingerprints": fingerprints}
    (run / "issuance-notary.yaml").write_text(yaml.safe_dump(notary), encoding="utf-8")
    for planned in plan["cells"]:
        root = Path(planned["contract_root"])
        contract = yaml.safe_load((root / "measurement_contract.yaml").read_text())
        cell_dir = run / "cells" / orchestrator._cell_label(planned)
        (cell_dir / "transport").mkdir(parents=True)
        result = {
            **contract["result_identity"], "run_id": contract["run_id"],
            "artifact_sha256": contract["artifact_sha256"], "session": contract["session"],
            "provenance": dict(contract["frozen_provenance"]),
        }
        (cell_dir / "result.yaml").write_text(yaml.safe_dump(result), encoding="utf-8")
        state = {"issuance_fingerprint": fingerprints[contract["run_id"]]}
        (cell_dir / "terminal-state.json").write_text(json.dumps(state) + "\n")
    matrix_state = {
        "schema_version": 1, "status": "complete", "matrix_sha256": plan["matrix_sha256"],
        "terminal_cells": 2, "expected_cells": 2,
        "environment_preflight_sha256": orchestrator._sha_file(
            run / "environment-preflight.json"),
        "issuance_notary_sha256": orchestrator._sha_file(run / "issuance-notary.yaml"),
    }
    (run / "matrix-state.json").write_text(json.dumps(matrix_state) + "\n")
    study_path = tmp_path / "study.frozen.yaml"
    study_path.write_text("status: frozen\n")
    ordered_cells = [SimpleNamespace(
        model=SimpleNamespace(name=row["cell"]["model"]),
        backend=SimpleNamespace(name=row["cell"]["backend"]),
        precision=row["cell"]["precision"], core_count=row["cell"]["core_count"])
        for row in plan["cells"]]
    spec = SimpleNamespace(status="frozen", target="k1", sha256=lambda: "a" * 64)
    monkeypatch.setattr(paper.PaperStudySpec, "from_yaml", staticmethod(lambda _path: spec))
    monkeypatch.setattr(study, "execution_matrix", lambda _spec: tuple(ordered_cells))
    monkeypatch.setattr(orchestrator, "validate_paper_result", lambda _result: None)
    monkeypatch.setattr(
        orchestrator, "_validate_local_terminal",
        lambda cell_dir, _planned: json.loads((cell_dir / "terminal-state.json").read_text()))
    monkeypatch.setattr(paper_attribution, "attach_causal_attribution", lambda _spec, _results: None)
    sealed_calls = []

    def seal(_spec, results, *, trusted_issuance_fingerprints):
        sealed_calls.append(([result["run_id"] for result in results],
                             dict(trusted_issuance_fingerprints)))
        return {"schema_version": 3, "study_sha256": "a" * 64, "results": results,
                "measurement_roots": [],
                "content_seal": {"seal_sha256": "f" * 64}}

    monkeypatch.setattr(paper_report, "seal_results_document", seal)
    monkeypatch.setattr(
        paper_report, "build_paper_report",
        lambda _spec, document, *, trusted_issuance_fingerprints: {
            "accepted": document["content_seal"]["seal_sha256"],
            "fingerprints": dict(trusted_issuance_fingerprints),
        })
    return plan_path, plan, run, study_path, sealed_calls


def test_finalize_seals_exact_plan_order_and_publishes_results_last(
        tmp_path: Path, monkeypatch) -> None:
    plan_path, plan, run, study_path, sealed_calls = _finalization_fixture(tmp_path, monkeypatch)

    output = orchestrator.finalize_matrix(plan_path, run, study_path)

    document = yaml.safe_load(output.read_text())
    assert [result["run_id"] for result in document["results"]] == [
        row["run_id"] for row in plan["cells"]]
    assert sealed_calls == [(["run-0", "run-1"], {"run-0": "d" * 64, "run-1": "e" * 64})]
    finalization = json.loads((run / "results-finalization.json").read_text())
    assert finalization["status"] == "complete"
    assert finalization["results_sha256"] == orchestrator._sha_file(output)
    assert len(finalization["terminal_cells"]) == 2


@pytest.mark.parametrize("mutation", ["missing", "extra", "result_identity"])
def test_finalize_rejects_partial_extra_or_identity_tampered_cells(
        tmp_path: Path, monkeypatch, mutation: str) -> None:
    plan_path, plan, run, study_path, sealed_calls = _finalization_fixture(tmp_path, monkeypatch)
    cells = run / "cells"
    if mutation == "missing":
        missing = cells / orchestrator._cell_label(plan["cells"][1])
        missing.rename(run / "missing-cell")
    elif mutation == "extra":
        (cells / "extra-cell").mkdir()
    else:
        path = cells / orchestrator._cell_label(plan["cells"][0]) / "result.yaml"
        value = yaml.safe_load(path.read_text())
        value["model"] = "tampered"
        path.write_text(yaml.safe_dump(value))

    with pytest.raises(ValueError, match="roster|identity"):
        orchestrator.finalize_matrix(plan_path, run, study_path)

    assert sealed_calls == []
    assert not (run / "results.yaml").exists()


def test_finalize_rejects_notary_or_prerequisite_tampering_before_seal(
        tmp_path: Path, monkeypatch) -> None:
    plan_path, _plan_value, run, study_path, sealed_calls = _finalization_fixture(
        tmp_path, monkeypatch)
    notary = yaml.safe_load((run / "issuance-notary.yaml").read_text())
    notary["fingerprints"]["run-0"] = "0" * 64
    (run / "issuance-notary.yaml").write_text(yaml.safe_dump(notary))

    with pytest.raises(ValueError, match="notary changed"):
        orchestrator.finalize_matrix(plan_path, run, study_path)

    assert sealed_calls == []


def test_finalize_rejects_plan_order_that_differs_from_frozen_study(
        tmp_path: Path, monkeypatch) -> None:
    from merlin.compare import study

    plan_path, plan, run, study_path, sealed_calls = _finalization_fixture(tmp_path, monkeypatch)
    reverse = [SimpleNamespace(
        model=SimpleNamespace(name=row["cell"]["model"]),
        backend=SimpleNamespace(name=row["cell"]["backend"]),
        precision=row["cell"]["precision"], core_count=row["cell"]["core_count"])
        for row in reversed(plan["cells"])]
    monkeypatch.setattr(study, "execution_matrix", lambda _spec: tuple(reverse))

    with pytest.raises(ValueError, match="order/membership"):
        orchestrator.finalize_matrix(plan_path, run, study_path)

    assert sealed_calls == []


def test_remote_payload_refuses_non_riscv_host(tmp_path: Path, monkeypatch) -> None:
    contract = _contract(tmp_path)
    monkeypatch.setattr(orchestrator, "_contract", _fake_contract)
    monkeypatch.setattr(orchestrator.platform, "machine", lambda: "x86_64")

    with pytest.raises(ValueError, match="locally on a RISC-V K1"):
        orchestrator.run_remote_cell(
            contract, tmp_path / "remote-run", contract_tree_sha256="f" * 64)

    assert not (tmp_path / "remote-run").exists()


def test_remote_environment_preflight_refuses_non_riscv_before_writing_receipt(
        tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(orchestrator.platform, "machine", lambda: "x86_64")
    output = tmp_path / "environment-preflight.json"

    with pytest.raises(ValueError, match="must execute on RISC-V"):
        orchestrator.create_remote_environment_receipt(
            output, matrix_sha256="a" * 64, controller_tree_sha256="b" * 64,
            runtime_requirements_sha256="c" * 64, required_core_count=8,
            expected_python="/usr/bin/python3")

    assert not output.exists()
