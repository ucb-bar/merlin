"""Production launcher wiring for the four-model host analytical evaluator."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from merlin.common.paths import repo_root


SCRIPTS = repo_root() / "merlin/experiments/gemmini_perf_bench/scripts"
sys.path.insert(0, str(SCRIPTS))
L = importlib.import_module("launch_global_agent_experiment")


def _sha(value: str | bytes) -> str:
    raw = value if isinstance(value, bytes) else value.encode()
    return hashlib.sha256(raw).hexdigest()


def _calibration(target_sha256: str) -> dict:
    receipts = [_sha(f"receipt-{index}") for index in range(4)]
    return {
        "schema": "phase2_host_analytical_calibration_v1",
        "target_sha256": target_sha256,
        "evidence_sha256s": receipts,
        "composition": {
            "operator": "sum", "eta": 0.0, "provenance_sha256": receipts[0]},
        "movement_balance": {
            "schema": "merlin_movement_balance_v1", "status": "derived",
            "peak_bytes_per_cycle": 4.0, "base_latency_cycles": 1.0,
            "domain_bytes": [8, 128], "n_distinct_sizes": 4,
            "residual_cycles": [0.0, 0.0, 0.0, 0.0],
            "provenance_sha256": receipts[1],
        },
        "accelerator_compute_roles": ["execute"],
        "risk_score": 0.1,
        "features": [
            {
                "id": "compute", "pointer": "/activity/compute", "resource": "arithmetic",
                "kind": "compute", "cycles_per_unit": {"lo": 1.0, "hi": 1.0},
                "floor_cycles_per_unit": 0.5, "effects": ["tiling"],
                "provenance_sha256": receipts[2],
            },
            {
                "id": "movement", "pointer": "/activity/movement", "resource": "transfer",
                "kind": "movement", "physical_bytes_per_unit": 8.0,
                "commands_per_unit": 1, "floor_cycles_per_unit": 1.0,
                "effects": ["movement"], "provenance_sha256": receipts[3],
            },
            {
                "id": "encoding", "pointer": "/activity/encoding", "resource": "transfer",
                "kind": "encoding", "cycles_per_unit": {"lo": 1.0, "hi": 1.0},
                "physical_bytes_per_unit": 8.0, "commands_per_unit": 1,
                "transitions_per_unit": 1, "floor_cycles_per_unit": 1.0,
                "effects": ["encoding"], "provenance_sha256": receipts[3],
            },
        ],
    }


def _four_sentinels():
    return tuple(SimpleNamespace(capsule_sha256=_sha(f"member-{index}"))
                 for index in range(4))


def _configured_inputs(tmp_path: Path, target_sha256: str):
    calibration = tmp_path / "calibration.json"
    calibration.write_text(json.dumps(_calibration(target_sha256)), encoding="utf-8")
    adapter = tmp_path / "quality_observer.py"
    adapter.write_text(
        "MERLIN_HOST_QUALITY_OBSERVER_CONTRACT = "
        "'host_reference_only_no_target_or_model_simulator_v1'\n"
        "def observe_quality(**kwargs):\n"
        "    return {'corpus_path': str(kwargs['corpus_path'])}\n",
        encoding="utf-8")
    corpora = []
    for index, sentinel in enumerate(_four_sentinels()):
        corpus = tmp_path / f"held-out-{index}.json"
        corpus.write_text(json.dumps({"sample": index}), encoding="utf-8")
        corpora.append([sentinel.capsule_sha256, str(corpus), _sha(corpus.read_bytes())])
    return calibration, adapter, corpora


def test_absent_evaluator_inputs_write_exact_only_receipt(tmp_path):
    sentinels = _four_sentinels()

    installation, receipt = L._prepare_fast_evaluator_installation(
        sentinels=sentinels, target_sha256=_sha("target"), stage_root=tmp_path,
        configured=False, calibration=None, calibration_sha256=None,
        quality_observer=None, quality_observer_sha256=None,
        quality_observer_symbol=None, classification_member_sha256=None,
        corpora=[], maximum_model_seconds=60.0)

    assert installation.experiment_kwargs() == {}
    assert receipt["status"] == "exact_only_fallback"
    assert receipt["fallback"]["approximation_allowed"] is False
    assert receipt["portfolio_member_sha256s"] == [
        sentinel.capsule_sha256 for sentinel in sentinels]
    persisted = json.loads((tmp_path / "fast_evaluation_installation.json").read_text())
    assert persisted["fallback"] == receipt["fallback"]
    assert persisted["execution"].endswith("no_complete_model_or_layer_simulation")


def test_complete_pinned_inputs_install_direct_experiment_kwargs(tmp_path):
    target_sha256 = _sha("target")
    sentinels = _four_sentinels()
    calibration, adapter, corpora = _configured_inputs(tmp_path, target_sha256)

    installation, receipt = L._prepare_fast_evaluator_installation(
        sentinels=sentinels, target_sha256=target_sha256, stage_root=tmp_path,
        configured=True, calibration=calibration,
        calibration_sha256=_sha(calibration.read_bytes()),
        quality_observer=adapter, quality_observer_sha256=_sha(adapter.read_bytes()),
        quality_observer_symbol="observe_quality",
        classification_member_sha256=sentinels[0].capsule_sha256,
        corpora=corpora, maximum_model_seconds=12.0)

    assert receipt["status"] == "installed"
    assert receipt["fallback"] is None
    assert receipt["provider_binding"]["target_sha256"] == target_sha256
    assert receipt["provider_binding"]["maximum_model_seconds"] == 12.0
    assert set(installation.experiment_kwargs()) == {
        "fast_evaluation_provider", "fast_evaluation_policy", "quality_budgets",
        "fast_evaluation_provider_binding",
    }
    assert set(receipt["input_bindings"]["held_out_corpora"]) == {
        sentinel.capsule_sha256 for sentinel in sentinels}


def test_host_observer_rechecks_corpus_bytes_on_every_call(tmp_path):
    target_sha256 = _sha("target")
    sentinels = _four_sentinels()
    _calibration_path, adapter, corpora = _configured_inputs(tmp_path, target_sha256)
    records = {member: {"path": path, "sha256": digest}
               for member, path, digest in corpora}
    observer = L._load_host_quality_observer(
        adapter, _sha(adapter.read_bytes()), "observe_quality", records)

    assert observer(sentinel=sentinels[0])["corpus_path"] == corpora[0][1]
    Path(corpora[0][1]).write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="differs from its exact SHA-256 pin"):
        observer(sentinel=sentinels[0])


def test_worker_forwards_all_fast_evaluation_pins(tmp_path):
    calibration = (tmp_path / "calibration.json").resolve()
    observer = (tmp_path / "observer.py").resolve()
    corpus = (tmp_path / "corpus.json").resolve()
    args = L._fast_evaluation_worker_arguments(
        calibration, "a" * 64, observer, "b" * 64, "observe_quality", "c" * 64,
        [["c" * 64, str(corpus), "d" * 64]], 17.5)

    assert args == (
        "--fast-evaluation-maximum-model-seconds", "17.5",
        "--fast-evaluation-calibration", str(calibration),
        "--fast-evaluation-calibration-sha256", "a" * 64,
        "--fast-evaluation-quality-observer", str(observer),
        "--fast-evaluation-quality-observer-sha256", "b" * 64,
        "--fast-evaluation-quality-observer-symbol", "observe_quality",
        "--fast-evaluation-classification-member-sha256", "c" * 64,
        "--fast-evaluation-held-out-corpus", "c" * 64, str(corpus), "d" * 64,
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
        L._validate_fast_evaluation_cli(args, parser)
    assert caught.value.code == 2
