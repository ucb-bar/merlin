"""The provisional Gemmini differential admits only pinned, RTL-attributed measurements."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml

from merlin.benchharness import hash_tree
from merlin.common.paths import repo_root


def _load_runner():
    scripts = repo_root() / "merlin/experiments/gemmini_perf_bench/scripts"
    sys.path.insert(0, str(scripts))
    spec = importlib.util.spec_from_file_location(
        "_provisional_gemmini_perf_differential", scripts / "run_provisional_differential.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


PD = _load_runner()


def _package(root: Path, package_id: str) -> tuple[Path, str]:
    root.mkdir(parents=True)
    (root / "manifest.yaml").write_text(yaml.safe_dump({
        "artifact_type": "mlir_oot_target_backend",
        "target": "gemmini",
        "package_id": package_id,
        "language": "python",
        "authoring": {"mode": "test_fixture"},
        "integrity_exempt": False,
    }), encoding="utf-8")
    (root / "tool.py").write_text("print('fixture')\n", encoding="utf-8")
    return root, hash_tree(root)["sha256"]


def _corpus(root: Path) -> Path:
    root.mkdir(parents=True)
    for kernel_id in ("k1", "k2"):
        (root / kernel_id).mkdir()
        (root / kernel_id / "capsule.yaml").write_text("id: fixture\n", encoding="utf-8")
    (root / "kernel_corpus.yaml").write_text(yaml.safe_dump({
        "golden_kernels": [{"id": "k1"}, {"id": "k2"}],
    }), encoding="utf-8")
    return root


def test_kernel_selection_is_explicit_ordered_and_unique(tmp_path: Path) -> None:
    corpus = _corpus(tmp_path / "kernels")
    assert [row["id"] for row in PD._selected_corpus("k2,k1", corpus)] == ["k2", "k1"]
    for invalid, pattern in (("", "explicit"), ("all", "explicit"),
                             ("k1,k1", "duplicate"), ("unknown", "unknown")):
        with pytest.raises(PD.ProvisionalExperimentError, match=pattern):
            PD._selected_corpus(invalid, corpus)


def test_package_identity_is_exact_and_captures_declared_provenance(tmp_path: Path) -> None:
    package, digest = _package(tmp_path / "package", "pinned-baseline")
    (package / "__pycache__").mkdir()
    (package / "__pycache__" / "tool.pyc").write_bytes(b"unhashed transient bytecode")
    record = PD.inspect_package(package, digest.upper(), "baseline")
    assert record["source_tree_sha256"] == digest
    assert record["declared_provenance"]["package_id"] == "pinned-baseline"
    assert len(record["manifest_sha256"]) == 64
    assert "__pycache__" in record["excluded_source_paths"]
    snapshot = tmp_path / "snapshot"
    assert PD._materialize_hashed_tree(package, snapshot, "baseline") == digest
    assert not (snapshot / "__pycache__").exists()
    with pytest.raises(PD.ProvisionalExperimentError, match="digest mismatch"):
        PD.inspect_package(package, "0" * 64, "baseline")

    (package / "link").symlink_to(package / "tool.py")
    with pytest.raises(PD.ProvisionalExperimentError, match="live symlink"):
        PD.inspect_package(package, digest, "baseline")


def test_spike_cycles_are_rejected_and_l3_requires_rtl_cycle_authority() -> None:
    spike = PD._normalize_spike_tier({
        "status": "pass", "cycles": 1234,
        "derived_from_rtl": False, "cycle_accurate": False,
    })
    assert spike["correct"] is True
    assert spike["cycles"] is None
    assert spike["raw_cycle_value_rejected"] is True
    assert spike["cycles_admitted_as_performance_evidence"] is False

    valid = PD._normalize_verilator_tier({
        "status": "pass", "cycles": 91,
        "derived_from_rtl": True, "cycle_accurate": True,
    })
    assert valid["cycles"] == 91
    assert valid["cycles_admitted_as_performance_evidence"] is True

    for bad in (
        {"status": "pass", "cycles": 0, "derived_from_rtl": True, "cycle_accurate": True},
        {"status": "pass", "cycles": True, "derived_from_rtl": True, "cycle_accurate": True},
        {"status": "pass", "cycles": 91, "derived_from_rtl": False, "cycle_accurate": True},
        {"status": "pass", "cycles": 91, "derived_from_rtl": True, "cycle_accurate": False},
    ):
        rejected = PD._normalize_verilator_tier(bad)
        assert rejected["cycles"] is None
        assert rejected["cycles_admitted_as_performance_evidence"] is False
        assert rejected["refusal"]


def test_run_is_serial_low_priority_and_writes_honest_reports(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    baseline, baseline_sha = _package(tmp_path / "baseline", "base")
    candidate, candidate_sha = _package(tmp_path / "candidate", "candidate")
    corpus = _corpus(tmp_path / "kernels")
    calls: list[tuple[str, str, int, tuple[str, ...]]] = []
    nice_calls: list[int] = []

    monkeypatch.setattr(PD.CR, "load_capsule", lambda path, contract: {"id": Path(path).name})
    monkeypatch.setattr(PD.CR, "default_adapters", lambda: {"L2": object(), "L3": object()})

    def fake_run(capsule, package, **kwargs):
        lane = Path(package).name
        calls.append((str(capsule["id"]), lane, kwargs["workers"],
                      tuple(capsule["required_oracle_tiers"])))
        cycles = 100 if lane == "baseline" else 80
        return {
            "status": "pass",
            "tiers": {
                "L2": {"status": "pass", "cycles": 999,
                       "derived_from_rtl": False, "cycle_accurate": False},
                "L3": {"status": "pass", "cycles": cycles,
                       "derived_from_rtl": True, "cycle_accurate": True,
                       "evidence": "rtl_verilator_console.log"},
            },
        }

    monkeypatch.setattr(PD.CR, "run_capsule", fake_run)
    monkeypatch.setattr(PD.os, "nice", lambda increment: nice_calls.append(increment) or 10)
    code, run_dir, report = PD.run_experiment(PD.ExperimentConfig(
        baseline_package=baseline,
        baseline_sha256=baseline_sha,
        candidate_package=candidate,
        candidate_sha256=candidate_sha,
        kernels="k1,k2",
        run_id="provisional_test",
        timeout_s=7,
        nice_increment=10,
        output_root=tmp_path / "runs",
        kernels_root=corpus,
    ))

    assert code == 0 and report["status"] == "GO"
    assert nice_calls == [10]
    assert calls == [
        ("k1", "baseline", 1, ("L0", "L1", "L2", "L3")),
        ("k1", "candidate", 1, ("L0", "L1", "L2", "L3")),
        ("k2", "baseline", 1, ("L0", "L1", "L2", "L3")),
        ("k2", "candidate", 1, ("L0", "L1", "L2", "L3")),
    ]
    assert all(cell["spike"]["cycles"] is None for cell in report["cells"])
    assert all(pair["candidate_speedup_vs_baseline"] == 1.25 for pair in report["pairs"])
    persisted = json.loads((run_dir / "provisional_differential.json").read_text())
    assert persisted["packages"]["baseline"]["snapshot_tree_sha256_after"] == baseline_sha
    assert "999" not in json.dumps(persisted), "raw Spike cycles must not escape into the summary"
    markdown = (run_dir / "provisional_differential.md").read_text()
    assert "not** claim the missing full agentic Phase-P contract" in markdown
    assert "Spike is correctness-only" in markdown


def test_invalid_l3_evidence_makes_the_experiment_no_go(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    baseline, baseline_sha = _package(tmp_path / "baseline", "base")
    candidate, candidate_sha = _package(tmp_path / "candidate", "candidate")
    corpus = _corpus(tmp_path / "kernels")
    monkeypatch.setattr(PD.CR, "load_capsule", lambda path, contract: {"id": Path(path).name})
    monkeypatch.setattr(PD.CR, "default_adapters", lambda: {"L2": object(), "L3": object()})
    monkeypatch.setattr(PD.os, "nice", lambda increment: 10)
    monkeypatch.setattr(PD.CR, "run_capsule", lambda *args, **kwargs: {
        "status": "pass",
        "tiers": {
            "L2": {"status": "pass", "cycles": 12,
                   "derived_from_rtl": False, "cycle_accurate": False},
            "L3": {"status": "pass", "cycles": 50,
                   "derived_from_rtl": True, "cycle_accurate": False},
        },
    })
    code, _run_dir, report = PD.run_experiment(PD.ExperimentConfig(
        baseline, baseline_sha, candidate, candidate_sha, "k1", "invalid_l3",
        output_root=tmp_path / "runs", kernels_root=corpus))
    assert code == 2 and report["status"] == "NO_GO"
    assert report["pairs"][0]["candidate_speedup_vs_baseline"] is None
    assert all(cell["verilator"]["cycles"] is None for cell in report["cells"])
