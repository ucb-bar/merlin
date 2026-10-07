"""INDEX.yaml follows the events it lists: a phase-1 freeze and a phase-2 ``best`` move regenerate it.

A refresh is wrapped so it can never fail the event it follows, which is exactly why a broken one would
look like an idle one; these tests read the written index back instead of trusting the call.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from merlin_experiments.phase1.feedback import freeze as phase1_freeze
from merlin_experiments.phase2.whole_model_measured import ledger as L

from merlin.common import oot_repo as O
from merlin.common.paths import phase_runs_root
from merlin.common.yaml import load_yaml
from merlin.targetgen import target_index as TI

TARGET = "toy"


@pytest.fixture()
def out_root(tmp_path, monkeypatch):
    root = tmp_path / "out"
    for sub in ("runs", "artifacts", "build"):
        (root / sub).mkdir(parents=True)
    monkeypatch.setenv("MERLIN_OUT_ROOT", str(root))
    return root


def _package(root: Path, body: str = "x = 1\n") -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest.yaml").write_text("target: toy\n", encoding="utf-8")
    (root / "transforms.py").write_text(body, encoding="utf-8")
    return root


def test_refresh_writes_only_for_a_canonical_phase_run(out_root, tmp_path):
    inside = phase_runs_root(TARGET, 2) / "20260929T120000Z_whole_model_measured_abc1234"
    inside.mkdir(parents=True)
    assert TI.refresh_for_run(inside, 2) == TI.index_path(TARGET)
    assert load_yaml(TI.index_path(TARGET))["target"] == TARGET
    TI.index_path(TARGET).unlink()
    assert TI.refresh_for_run(tmp_path / "elsewhere" / "run", 2) is None
    assert TI.refresh_for_run(inside, 1) is None  # a phase-2 run is not under the phase-1 root
    assert not TI.index_path(TARGET).exists()


def test_refresh_never_raises(out_root, monkeypatch, capsys):
    run = phase_runs_root(TARGET, 1) / "run"
    run.mkdir(parents=True)

    def broken(target):
        raise OSError("disk full")

    monkeypatch.setattr(TI, "write_index", broken)
    assert TI.refresh_for_run(run, 1) is None
    assert "INDEX.yaml not refreshed" in capsys.readouterr().err


def test_a_phase1_freeze_lists_the_frozen_compiler(out_root, tmp_path):
    run_dir = phase_runs_root(TARGET, 1) / "fixture-arm-run"
    _package(run_dir / "submission")
    (run_dir / "oot").mkdir()
    record = phase1_freeze.freeze(run_dir, repo=tmp_path)
    index = load_yaml(TI.index_path(TARGET))
    assert [row["frozen_commit"] for row in index["phase1_frozen"]] == [record["oot"]["frozen_commit"]]
    assert index["phase1_frozen"][0]["run"].endswith("fixture-arm-run")


def test_a_phase2_best_move_lists_the_run_best(out_root, tmp_path):
    run_dir = phase_runs_root(TARGET, 2) / "20260929T120000Z_whole_model_measured_abc1234"
    repo = O.init(run_dir / "oot")
    package = _package(tmp_path / "pkg", "x = 2\n")
    digest = O.package_digest(package)
    ledger = L.OotLedger(repo, run_id=run_dir.name, records=run_dir / L.ITERATIONS, clock=lambda: "20260929T130000Z")
    ledger.record(package, digest, label="candidate")
    screen = SimpleNamespace(
        measurement_for=lambda d: {"timing_status": "MEASURED", "objective_cycles": 10},
        attributable=lambda d: True,
    )
    objective = SimpleNamespace(
        screen=screen, certifier=None, summary=lambda: {"best": {"package_sha256": digest}}, _confirmed=lambda d: True
    )
    assert not TI.index_path(TARGET).exists()
    events = ledger.sync(objective)
    assert [e["kind"] for e in events] == ["measured", "best"]
    rows = load_yaml(TI.index_path(TARGET))["phase2_best"]
    assert [row["package_digest"] for row in rows] == [digest]
    assert rows[0]["best_commit"] == O.tags(repo)[O.BEST_TAG]
