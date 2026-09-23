"""Installed host installation inputs, without simulator or process execution."""

import argparse
import hashlib
import json
import os
import py_compile
import socket
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from merlin_experiments.phase2 import fast_evaluation_installation as F


@pytest.fixture(autouse=True)
def no_execution(monkeypatch):
    def refuse(*args, **kwargs):
        pytest.fail("fast installation tests cannot launch processes or listeners")

    monkeypatch.setattr(subprocess, "Popen", refuse)
    monkeypatch.setattr(socket.socket, "bind", refuse)


def test_exact_file_rejects_changed_bytes_and_symlinks(tmp_path):
    path = tmp_path / "input.json"
    path.write_bytes(b"{}")
    digest = hashlib.sha256(b"{}").hexdigest()
    assert F._exact_file(path, digest, label="fixture") == path
    alias = tmp_path / "alias.json"
    alias.symlink_to(path)
    with pytest.raises(ValueError, match="linked"):
        F._exact_file(alias, digest, label="fixture")
    path.write_bytes(b"[]")
    with pytest.raises(ValueError, match="differs"):
        F._exact_file(path, digest, label="fixture")


@pytest.mark.parametrize("partial", [False, True])
def test_cli_exact_only_is_distinct_from_incomplete_accuracy_installation(partial):
    args = SimpleNamespace(
        fast_evaluation_calibration=None,
        fast_evaluation_calibration_sha256=None,
        fast_evaluation_quality_observer=None,
        fast_evaluation_quality_observer_sha256=None,
        fast_evaluation_quality_observer_symbol="observe" if partial else None,
        fast_evaluation_classification_member_sha256=None,
        fast_evaluation_held_out_corpus=[],
        fast_evaluation_maximum_model_seconds=10,
    )
    parser = argparse.ArgumentParser()
    if partial:
        with pytest.raises(SystemExit):
            F.validate_cli(args, parser)
    else:
        assert F.validate_cli(args, parser) is False


def _sha(value: str | bytes) -> str:
    raw = value if isinstance(value, bytes) else value.encode()
    return hashlib.sha256(raw).hexdigest()


def _calibration(target_sha256: str) -> dict:
    receipts = [_sha(f"receipt-{index}") for index in range(4)]
    return {
        "schema": "phase2_host_analytical_calibration_v1",
        "target_sha256": target_sha256,
        "evidence_sha256s": receipts,
        "composition": {"operator": "sum", "eta": 0.0, "provenance_sha256": receipts[0]},
        "movement_balance": {
            "schema": "merlin_movement_balance_v1",
            "status": "derived",
            "peak_bytes_per_cycle": 4.0,
            "base_latency_cycles": 1.0,
            "domain_bytes": [8, 128],
            "n_distinct_sizes": 4,
            "residual_cycles": [0.0, 0.0, 0.0, 0.0],
            "provenance_sha256": receipts[1],
        },
        "accelerator_compute_roles": ["execute"],
        "risk_score": 0.1,
        "features": [
            {
                "id": "compute",
                "pointer": "/activity/compute",
                "resource": "arithmetic",
                "kind": "compute",
                "cycles_per_unit": {"lo": 1.0, "hi": 1.0},
                "floor_cycles_per_unit": 0.5,
                "effects": ["tiling"],
                "provenance_sha256": receipts[2],
            },
            {
                "id": "movement",
                "pointer": "/activity/movement",
                "resource": "transfer",
                "kind": "movement",
                "physical_bytes_per_unit": 8.0,
                "commands_per_unit": 1,
                "floor_cycles_per_unit": 1.0,
                "effects": ["movement"],
                "provenance_sha256": receipts[3],
            },
            {
                "id": "encoding",
                "pointer": "/activity/encoding",
                "resource": "transfer",
                "kind": "encoding",
                "cycles_per_unit": {"lo": 1.0, "hi": 1.0},
                "physical_bytes_per_unit": 8.0,
                "commands_per_unit": 1,
                "transitions_per_unit": 1,
                "floor_cycles_per_unit": 1.0,
                "effects": ["encoding"],
                "provenance_sha256": receipts[3],
            },
        ],
    }


def _four_sentinels():
    return tuple(SimpleNamespace(capsule_sha256=_sha(f"member-{index}")) for index in range(4))


def _configured_inputs(tmp_path: Path, target_sha256: str):
    calibration = tmp_path / "calibration.json"
    calibration.write_text(json.dumps(_calibration(target_sha256)), encoding="utf-8")
    adapter = tmp_path / "quality_observer.py"
    adapter.write_text(
        "MERLIN_HOST_QUALITY_OBSERVER_CONTRACT = "
        "'host_reference_only_no_target_or_model_simulator_v1'\n"
        "def observe_quality(**kwargs):\n"
        "    return {'corpus_path': str(kwargs['corpus_path'])}\n",
        encoding="utf-8",
    )
    corpora = []
    for index, sentinel in enumerate(_four_sentinels()):
        corpus = tmp_path / f"held-out-{index}.json"
        corpus.write_text(json.dumps({"sample": index}), encoding="utf-8")
        corpora.append([sentinel.capsule_sha256, str(corpus), _sha(corpus.read_bytes())])
    return calibration, adapter, corpora


def test_absent_evaluator_inputs_write_exact_only_receipt(tmp_path):
    sentinels = _four_sentinels()

    installation, receipt = F.prepare(
        sentinels=sentinels,
        target_sha256=_sha("target"),
        stage_root=tmp_path,
        configured=False,
        calibration=None,
        calibration_sha256=None,
        quality_observer=None,
        quality_observer_sha256=None,
        quality_observer_symbol=None,
        classification_member_sha256=None,
        corpora=[],
        maximum_model_seconds=60.0,
    )

    assert installation.experiment_kwargs() == {}
    assert receipt["status"] == "exact_only_fallback"
    assert receipt["fallback"]["approximation_allowed"] is False
    assert receipt["portfolio_member_sha256s"] == [sentinel.capsule_sha256 for sentinel in sentinels]
    persisted = json.loads((tmp_path / "fast_evaluation_installation.json").read_text())
    assert persisted["fallback"] == receipt["fallback"]
    assert persisted["execution"].endswith("no_complete_model_or_layer_simulation")


def test_complete_pinned_inputs_install_direct_experiment_kwargs(tmp_path):
    target_sha256 = _sha("target")
    sentinels = _four_sentinels()
    calibration, adapter, corpora = _configured_inputs(tmp_path, target_sha256)

    installation, receipt = F.prepare(
        sentinels=sentinels,
        target_sha256=target_sha256,
        stage_root=tmp_path,
        configured=True,
        calibration=calibration,
        calibration_sha256=_sha(calibration.read_bytes()),
        quality_observer=adapter,
        quality_observer_sha256=_sha(adapter.read_bytes()),
        quality_observer_symbol="observe_quality",
        classification_member_sha256=sentinels[0].capsule_sha256,
        corpora=corpora,
        maximum_model_seconds=12.0,
    )

    assert receipt["status"] == "installed"
    assert receipt["fallback"] is None
    assert receipt["provider_binding"]["target_sha256"] == target_sha256
    assert receipt["provider_binding"]["maximum_model_seconds"] == 12.0
    assert set(installation.experiment_kwargs()) == {
        "fast_evaluation_provider",
        "fast_evaluation_policy",
        "quality_budgets",
        "fast_evaluation_provider_binding",
    }
    assert set(receipt["input_bindings"]["held_out_corpora"]) == {sentinel.capsule_sha256 for sentinel in sentinels}


def test_host_observer_rechecks_corpus_bytes_on_every_call(tmp_path):
    target_sha256 = _sha("target")
    sentinels = _four_sentinels()
    _calibration_path, adapter, corpora = _configured_inputs(tmp_path, target_sha256)
    records = {member: {"path": path, "sha256": digest} for member, path, digest in corpora}
    observer = F._load_host_quality_observer(adapter, _sha(adapter.read_bytes()), "observe_quality", records)

    assert observer(sentinel=sentinels[0])["corpus_path"] == corpora[0][1]
    Path(corpora[0][1]).write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="differs from its exact SHA-256 pin"):
        observer(sentinel=sentinels[0])


def test_pinned_observer_executes_current_source_not_valid_stale_bytecode(tmp_path):
    path = tmp_path / "observer.py"
    prefix = (
        "MERLIN_HOST_QUALITY_OBSERVER_CONTRACT = "
        "'host_reference_only_no_target_or_model_simulator_v1'\n"
        "def observe_quality(**kwargs):\n"
    )
    original = prefix + "    return 'old'\n"
    selected = prefix + "    return 'new'\n"
    assert len(original.encode()) == len(selected.encode())
    path.write_text(original)
    stamp = path.stat()
    bytecode = py_compile.compile(
        str(path),
        doraise=True,
        invalidation_mode=py_compile.PycInvalidationMode.TIMESTAMP,
    )
    assert Path(bytecode).is_file()
    path.write_text(selected)
    os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    assert path.stat().st_size == stamp.st_size
    assert path.stat().st_mtime_ns == stamp.st_mtime_ns
    corpus = tmp_path / "corpus.json"
    corpus.write_text("{}")
    member = _sha("member")
    observer = F._load_host_quality_observer(
        path,
        _sha(selected),
        "observe_quality",
        {member: {"path": str(corpus), "sha256": _sha(corpus.read_bytes())}},
    )
    assert observer(sentinel=SimpleNamespace(capsule_sha256=member)) == "new"


def test_observer_source_change_between_admission_and_read_refuses(tmp_path, monkeypatch):
    path = tmp_path / "observer.py"
    original = (
        "MERLIN_HOST_QUALITY_OBSERVER_CONTRACT = "
        "'host_reference_only_no_target_or_model_simulator_v1'\n"
        "def observe_quality(**kwargs):\n"
        "    return 'old'\n"
    )
    path.write_text(original)
    admit = F._exact_file

    def changed_after_admission(candidate, digest, *, label):
        result = admit(candidate, digest, label=label)
        result.write_text(original.replace("'old'", "'new'"))
        return result

    monkeypatch.setattr(F, "_exact_file", changed_after_admission)
    with pytest.raises(ValueError, match="changed|differ|SHA|hash"):
        F._load_host_quality_observer(path, _sha(original), "observe_quality", {})


def test_worker_forwards_all_fast_evaluation_pins(tmp_path):
    calibration = (tmp_path / "calibration.json").resolve()
    observer = (tmp_path / "observer.py").resolve()
    corpus = (tmp_path / "corpus.json").resolve()
    args = F.worker_arguments(
        calibration,
        "a" * 64,
        observer,
        "b" * 64,
        "observe_quality",
        "c" * 64,
        [["c" * 64, str(corpus), "d" * 64]],
        17.5,
    )

    assert args == (
        "--fast-evaluation-maximum-model-seconds",
        "17.5",
        "--fast-evaluation-calibration",
        str(calibration),
        "--fast-evaluation-calibration-sha256",
        "a" * 64,
        "--fast-evaluation-quality-observer",
        str(observer),
        "--fast-evaluation-quality-observer-sha256",
        "b" * 64,
        "--fast-evaluation-quality-observer-symbol",
        "observe_quality",
        "--fast-evaluation-classification-member-sha256",
        "c" * 64,
        "--fast-evaluation-held-out-corpus",
        "c" * 64,
        str(corpus),
        "d" * 64,
    )


def test_partial_fast_evaluation_configuration_is_not_silently_downgraded():
    parser = argparse.ArgumentParser()
    args = SimpleNamespace(
        fast_evaluation_calibration=Path("calibration.json"),
        fast_evaluation_calibration_sha256=None,
        fast_evaluation_quality_observer=None,
        fast_evaluation_quality_observer_sha256=None,
        fast_evaluation_quality_observer_symbol=None,
        fast_evaluation_classification_member_sha256=None,
        fast_evaluation_held_out_corpus=[],
        fast_evaluation_maximum_model_seconds=60.0,
    )

    with pytest.raises(SystemExit) as caught:
        F.validate_cli(args, parser)
    assert caught.value.code == 2
