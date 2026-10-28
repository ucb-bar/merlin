"""The operator tier floor leaves exempted capsules on their declared ladder and everything else unchanged."""

from __future__ import annotations

import threading

import pytest

from merlin.targetgen import tier_policy as TP


@pytest.fixture(autouse=True)
def _floor(monkeypatch):
    monkeypatch.setenv(TP.TIER_FLOOR_ENV, "L3")


def test_unexempted_scored_capsule_is_raised_as_before():
    assert TP.required_tiers({"required_oracle_tiers": ["L0", "L1", "L2"]}) == ["L0", "L1", "L2", "L3"]


def test_a_capsule_declaring_l3_is_unchanged():
    capsule = {"required_oracle_tiers": ["L0", "L1", "L2", "L3"]}
    assert TP.required_tiers(capsule) == ["L0", "L1", "L2", "L3"]


def test_unset_floor_returns_the_declared_value_itself(monkeypatch):
    monkeypatch.delenv(TP.TIER_FLOOR_ENV)
    declared = ["L0", "L1", "L2"]
    assert TP.required_tiers({"required_oracle_tiers": declared}) is declared


def test_declared_exemption_field_keeps_the_declared_ladder():
    capsule = {"required_oracle_tiers": ["L0", "L1", "L2"], TP.FLOOR_EXEMPT_FIELD: True}
    assert TP.required_tiers(capsule) == ["L0", "L1", "L2"]
    # Only a literal True exempts: a stray truthy string must not silently lower the bar.
    assert TP.required_tiers(dict(capsule, **{TP.FLOOR_EXEMPT_FIELD: "yes"})) == ["L0", "L1", "L2", "L3"]


def test_scoped_directory_exemption_is_exact_and_released(tmp_path):
    inside, outside = tmp_path / "a", tmp_path / "b"
    inside.mkdir(), outside.mkdir()
    cap = lambda d: {"required_oracle_tiers": ["L0", "L1", "L2"], "__dir__": str(d)}  # noqa: E731
    with TP.floor_exempt([inside]):
        assert TP.required_tiers(cap(inside)) == ["L0", "L1", "L2"]
        assert TP.required_tiers(cap(outside)) == ["L0", "L1", "L2", "L3"]
        with TP.floor_exempt([inside]):  # re-entrant
            pass
        assert TP.required_tiers(cap(inside)) == ["L0", "L1", "L2"]
    assert TP.required_tiers(cap(inside)) == ["L0", "L1", "L2", "L3"]


def test_exemption_is_visible_to_worker_threads(tmp_path):
    seen = []
    capsule = {"required_oracle_tiers": ["L0", "L1", "L2"], "__dir__": str(tmp_path)}
    with TP.floor_exempt([tmp_path]):
        worker = threading.Thread(target=lambda: seen.append(TP.required_tiers(capsule)))
        worker.start()
        worker.join()
    assert seen == [["L0", "L1", "L2"]]


def test_exemption_released_when_the_block_raises(tmp_path):
    capsule = {"required_oracle_tiers": ["L0"], "__dir__": str(tmp_path)}
    with pytest.raises(RuntimeError), TP.floor_exempt([tmp_path]):
        raise RuntimeError("grade failed")
    assert TP.required_tiers(capsule) == ["L0", "L1", "L2", "L3"]
