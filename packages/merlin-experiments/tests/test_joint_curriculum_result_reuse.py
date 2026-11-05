"""A recorded curriculum pass is reused only from the exact bytes it was recorded from.

``capsule_result.json`` is grade output that later stages of the same grade rewrite in place. Losing
a read lease on it, or any change to it, must only drop that pass's reuse (the case is graded again)
and must never fail the grade or let the old verdict stand for new bytes.
"""

import json

import pytest
from merlin_experiments.phase1.feedback import joint_curriculum as C

from merlin.common import read_lease_digest as L


def _policy():
    return {"correctness_curriculum": {"original_capsules": [f"case{i:02}" for i in range(6)]}}


def _records(tmp_path):
    records = []
    for i in range(6):
        name = f"case{i:02}"
        root = tmp_path / name
        root.mkdir()
        capsule = {"name": name, "label": "public", "inputs": [{"shape": [i + 1]}], "operation": {"op": "matmul"}}
        (root / "capsule.yaml").write_text(json.dumps(capsule))
        (root / "input.bin").write_bytes(bytes([i]))
        records.append(("matmul", i + 1, name, root))
    return records


def _recorded_pass(tmp_path):
    records = _records(tmp_path)
    runs = tmp_path / "run"
    progress = C.Progress(tmp_path / "progress", identity={"compiler": "same"}, runs_root=runs)
    progress.select(records, policy=_policy(), hint=[])
    receipt = runs / "runs" / "suite" / "case05" / "capsule_result.json"
    receipt.parent.mkdir(parents=True)
    row = {
        "capsule": "case05",
        "status": "pass",
        "compiler_source": {"sha256": "a" * 64},
        "numeric": {"status": "pass"},
        "tiers": {tier: {"status": "pass", "engine": "spike"} for tier in ("L0", "L1", "L2")},
    }
    receipt.write_text(json.dumps(row))
    progress.completed({"capsule": "case05"})
    assert "case05" in progress.report()["exact_correctness_reused"]
    return records, progress, receipt


def test_unchanged_result_is_reused(tmp_path, capsys):
    records, progress, _ = _recorded_pass(tmp_path)
    chosen, report = progress.select(records, policy=_policy(), hint=[])
    assert "case05" in report["exact_correctness_reused"] and all(r[2] != "case05" for r in chosen)
    assert "not reused" not in capsys.readouterr().out


def test_lost_read_lease_drops_reuse_not_the_grade(tmp_path, monkeypatch, capsys):
    records, progress, receipt = _recorded_pass(tmp_path)

    def custody_lost(path):
        raise L.CustodyChanged("original snapshot read lease custody changed")

    monkeypatch.setattr(C, "_result", custody_lost)
    chosen, report = progress.select(records, policy=_policy(), hint=[])  # nothing leaves select
    assert "case05" not in report["exact_correctness_reused"] and "case05" in report["pending"]
    assert any(r[2] == "case05" for r in chosen)  # graded again by the ordinary native path
    out = capsys.readouterr().out
    assert out.count("saved pass for case05 not reused") == 1
    assert "CustodyChanged" in out and str(receipt.resolve()) in out


def test_rewritten_result_is_never_reused_with_its_old_verdict(tmp_path, capsys):
    _, progress, receipt = _recorded_pass(tmp_path)
    row = json.loads(receipt.read_text())
    row["numeric"]["status"] = "fail"
    receipt.write_text(json.dumps(row))
    for _ in range(2):
        report = progress.report()
        assert "case05" not in report["exact_correctness_reused"] and "case05" in report["pending"]
    assert capsys.readouterr().out.count("recorded result bytes changed") == 1  # one line per change


def test_in_place_rewrite_of_a_leased_result_keeps_identical_bytes_reusable(tmp_path, monkeypatch):
    """The live failure: a grade output in the lease cache, rewritten by the suite, then re-pinned."""
    L.close()
    monkeypatch.setenv("MERLIN_SNAPSHOT_LEASE_CACHE", "1")
    monkeypatch.setenv("MERLIN_SNAPSHOT_LEASE_MIN_BYTES", "0")
    try:
        _, progress, receipt = _recorded_pass(tmp_path)
        if L.digest(receipt.resolve(), minimum_bytes=0) is None or L._CACHE is None or not L._CACHE.entries:
            pytest.skip("Linux read leases unavailable for the selected temporary filesystem")
        receipt.write_bytes(receipt.read_bytes())  # in-place rewrite breaks the lease, bytes unchanged
        assert "case05" in progress.report()["exact_correctness_reused"]
        receipt.write_bytes(receipt.read_bytes() + b"\n")
        assert "case05" not in progress.report()["exact_correctness_reused"]
    finally:
        L.close()
