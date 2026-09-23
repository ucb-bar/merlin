"""Performance readers see canonical and historical products without rewriting either."""

from __future__ import annotations

import importlib.util
import json
import sys

import pytest

from merlin.common import artifacts
from merlin.common.paths import repo_root


@pytest.fixture
def readers(tmp_path, monkeypatch):
    modules = {}
    before = list(sys.path)
    for name in ("cycle_sweep", "headline"):
        path = repo_root() / "merlin/experiments/performance_contract" / (name + ".py")
        spec = importlib.util.spec_from_file_location("product_reader_" + name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        modules[name] = module
    sys.path[:] = before
    monkeypatch.setenv("MERLIN_OUT_ROOT", str(tmp_path / "out"))
    return modules


def _record(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content))
    return path


@pytest.mark.parametrize("newest", ["canonical", "legacy"])
def test_latest_sweep_orders_both_homes_by_product_name(readers, newest):
    canonical = artifacts.product_root("perf-ledger") / "fixture/v2"
    legacy = artifacts.artifacts_dir() / "perf-ledger/fixture/v2"
    roots = {"canonical": canonical, "legacy": legacy}
    records = []
    for label, root in roots.items():
        token = "20260922" if label == newest else "20260921"
        records.append(_record(root / f"perf-ledger_fixture_v2_{token}_abc/cycle_sweep.json", {"from": label}))
    before = {path: path.read_bytes() for path in records}
    assert readers["cycle_sweep"].latest_sweep("fixture") == {"from": newest}
    assert {path: path.read_bytes() for path in records} == before
    assert not canonical.is_symlink() and not legacy.is_symlink()


def _workload(cycles):
    return {"shape": {"m": 2, "k": 3, "n": 4}, "result": {"ran": True, "halted": True, "cycles": cycles}}


def test_prior_runs_merge_historical_and_new_products_in_original_order(readers):
    old = _record(
        artifacts.artifacts_dir() / "perf-workload/fixture/v1/product_20260921/workload_record.json", _workload(21)
    )
    new = _record(
        artifacts.product_root("perf-workload") / "fixture/v1/product_20260922/workload_record.json", _workload(22)
    )
    before = {path: path.read_bytes() for path in (old, new)}
    rows = readers["headline"].prior_runs("fixture")
    assert [row["cycles"] for row in rows] == [21, 22]
    assert [row["product"] for row in rows] == [str(old.parent), str(new.parent)]
    assert {path: path.read_bytes() for path in (old, new)} == before


@pytest.mark.parametrize(
    "topic,filename,reader",
    [
        ("perf-ledger", "cycle_sweep.json", "cycle_sweep"),
        ("perf-workload", "workload_record.json", "headline"),
    ],
)
def test_organized_legacy_symlink_does_not_duplicate_products(readers, topic, filename, reader):
    canonical = artifacts.product_root(topic)
    body = {"from": "canonical"} if reader == "cycle_sweep" else _workload(21)
    _record(canonical / "fixture/v2/product_20260921" / filename, body)
    old = artifacts.artifacts_dir() / topic
    old.symlink_to(canonical, target_is_directory=True)
    result = (
        readers[reader].latest_sweep("fixture") if reader == "cycle_sweep" else readers[reader].prior_runs("fixture")
    )
    assert result == body if reader == "cycle_sweep" else len(result) == 1
    assert old.is_symlink()
