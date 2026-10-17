"""Phase 1 entrypoints lease the one shared CPU pool at their class, and caps default to its size."""

from __future__ import annotations

import pytest
from merlin_experiments.phase1.brokers import selfcheck as B
from merlin_experiments.phase1.brokers import simjob

from merlin.common import cpu_pool as C


@pytest.fixture
def pool(monkeypatch, tmp_path):
    def select(size):
        monkeypatch.setenv("MERLIN_COMPUTE_CPUS", ",".join(map(str, range(size))))
        monkeypatch.setenv("MERLIN_COMPUTE_CPU_SLOTS", str(tmp_path / "slots"))

    for key in ("MERLIN_COMPUTE_CPUS", "MERLIN_COMPUTE_CPU_SLOTS", "MERLIN_SELFCHECK_BROKER_JOBS",
                "MERLIN_SELFCHECK_ORDINARY_JOBS", C.PRIORITY_ENV):
        monkeypatch.delenv(key, raising=False)
    return select


def test_selfcheck_and_simjob_concurrency_default_to_the_pool(pool, monkeypatch):
    assert (B._dispatch_capacity(), B._ordinary_capacity(), simjob._default_max_jobs()) == (1, 1, 4)  # legacy
    pool(18)
    assert (B._dispatch_capacity(), B._ordinary_capacity(), simjob._default_max_jobs()) == (2, 18, 18)
    monkeypatch.setenv("MERLIN_SELFCHECK_BROKER_JOBS", "1")
    monkeypatch.setenv("MERLIN_SELFCHECK_ORDINARY_JOBS", "2")
    assert (B._dispatch_capacity(), B._ordinary_capacity()) == (1, 2)  # explicit values still win
    for bad in ("0", "19", "-1", "2.0", ""):
        monkeypatch.setenv("MERLIN_SELFCHECK_ORDINARY_JOBS", bad)
        with pytest.raises(ValueError, match="MERLIN_SELFCHECK_ORDINARY_JOBS"):
            B._ordinary_capacity()
