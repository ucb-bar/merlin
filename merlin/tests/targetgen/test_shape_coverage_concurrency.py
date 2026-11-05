"""Concurrent lowering-coverage corners change wall time, never a per-corner outcome.

Each corner is an independent emit invocation of the same immutable submission. Running them on
threads must give byte-for-byte the summary the serial loop gives, keep the caller's execution
context in every corner, and hold one shared compute slot per in-flight corner.
"""

from __future__ import annotations

import contextvars
import os
import random
import threading
import time

import pytest

from merlin.targetgen import lowering_coverage as LC

_SCOPE = contextvars.ContextVar("fixture_execution_scope", default=None)


def _binding(monkeypatch, tile=16):
    monkeypatch.setattr(
        LC,
        "_binding",
        lambda t: type(
            "B",
            (),
            {
                "operand_dtype": "int8",
                "accum_dtype": "int32",
                "mlir_dtype": staticmethod(lambda tok: {"int8": "i8"}.get(tok, "i32")),
            },
        )(),
    )
    monkeypatch.setattr(LC, "tile_edge", lambda t: tile)


def _mixed_outcomes(seen):
    """A backend that lowers most shapes, declines the M tail, emits nothing for K at two tiles and
    errors on the sub-tile case, with jittered latency so completion order differs from plan order."""

    def fake(
        package, *, target, m, k, n, operand_mlir, accum_mlir, contract=None, timeout=300, additional_forbidden=()
    ):
        seen.append((m, k, n, _SCOPE.get(), threading.current_thread().name))
        time.sleep(random.uniform(0, 0.03))
        if (m, k, n) == (17, 16, 16):
            return "declined", "no M remainder loop", 0
        if (m, k, n) == (16, 32, 16):
            return "empty", "emitted no commands", 0
        if (m, k, n) == (15, 15, 15):
            return "error", "RuntimeError: emitter crashed", 0
        return "lowered", None, m + k + n

    return fake


def _without_timing(result):
    return {key: value for key, value in result.items() if key != "timing"}


@pytest.mark.parametrize("workers", [2, 4, 8, 16])
def test_concurrent_sweep_matches_serial_per_corner(monkeypatch, workers):
    _binding(monkeypatch)
    seen = []
    monkeypatch.setattr(LC, "probe_shape", _mixed_outcomes(seen))
    serial = LC.sweep("pkg", target="fixture")
    concurrent = LC.sweep("pkg", target="fixture", workers=workers)
    assert _without_timing(concurrent) == _without_timing(serial)
    assert [c["corner"] for c in concurrent["corners"]] == [*LC.CORNERS, *LC.TAIL_CORNERS]
    assert concurrent["tail_cases_uncovered"] == ["m_tail", "sub_tile"]
    assert concurrent["multi_tile_axes_uncovered"] == ["k"]
    assert concurrent["all_covered"] is False
    assert concurrent["timing"]["workers"] == workers
    assert set(concurrent["timing"]["corner_wall_s"]) == {c["corner"] for c in concurrent["corners"]}
    assert len(seen) == 2 * len(serial["corners"])


def test_every_concurrent_corner_keeps_the_callers_context(monkeypatch):
    _binding(monkeypatch)
    seen = []
    monkeypatch.setattr(LC, "probe_shape", _mixed_outcomes(seen))
    token = _SCOPE.set("caller-owned executor")
    try:
        LC.sweep("pkg", target="fixture", workers=4)
    finally:
        _SCOPE.reset(token)
    assert {scope for *_, scope, _thread in seen} == {"caller-owned executor"}
    assert all(name.startswith("lowcov") for *_, name in seen)


def test_each_concurrent_corner_holds_one_shared_compute_slot(monkeypatch, tmp_path):
    if not hasattr(os, "sched_getaffinity"):
        pytest.skip("Linux affinity required")
    from merlin.common.cpu_pool import current_thread_owns_cpu_lease

    cpus = sorted(os.sched_getaffinity(0))[:2]
    monkeypatch.setenv("MERLIN_COMPUTE_CPUS", ",".join(map(str, cpus)))
    monkeypatch.setenv("MERLIN_COMPUTE_CPU_SLOTS", str(tmp_path / "slots"))
    monkeypatch.delenv("MERLIN_PREPARATION_CPUS", raising=False)
    _binding(monkeypatch)
    lock, active, peak, leased = threading.Lock(), [0], [0], []

    def fake(package, *, target, m, k, n, **kwargs):
        with lock:
            active[0] += 1
            peak[0] = max(peak[0], active[0])
        try:
            leased.append((current_thread_owns_cpu_lease(), len(os.sched_getaffinity(0))))
            time.sleep(0.02)
            return "lowered", None, m + k + n
        finally:
            with lock:
                active[0] -= 1

    monkeypatch.setattr(LC, "probe_shape", fake)
    result = LC.sweep("pkg", target="fixture", workers=8)
    assert result["all_covered"] is True
    assert leased and all(owned and width == 1 for owned, width in leased)
    assert peak[0] <= len(cpus), "never more corners in flight than leased compute slots"


def test_a_refused_lease_is_an_error_outcome_never_coverage(monkeypatch, tmp_path):
    _binding(monkeypatch)
    monkeypatch.setenv("MERLIN_COMPUTE_CPUS", "not-a-cpu")
    monkeypatch.setenv("MERLIN_COMPUTE_CPU_SLOTS", str(tmp_path / "slots"))
    monkeypatch.setattr(LC, "probe_shape", lambda *a, **k: pytest.fail("a probe ran without its lease"))
    result = LC.sweep("pkg", target="fixture", workers=4)
    assert {c["outcome"] for c in result["corners"]} == {"error"}
    assert result["baseline_tile_lowered"] is False
    assert result["all_covered"] is False


def test_invalid_worker_count_refuses_before_probing(monkeypatch):
    _binding(monkeypatch)
    monkeypatch.setattr(LC, "probe_shape", lambda *a, **k: pytest.fail("probed with an invalid worker count"))
    for workers in (0, -1, 2.0, True):
        with pytest.raises(ValueError, match="positive integer"):
            LC.sweep("pkg", target="fixture", workers=workers)


def test_plan_identity_names_every_probe_input(monkeypatch):
    _binding(monkeypatch)
    plan = LC.sweep_plan("fixture")
    assert [name for name, _ in plan["probes"]] == [*LC.CORNERS, *LC.TAIL_CORNERS]
    assert dict(plan["probes"])["m_tail"] == (17, 16, 16)
    same = LC.plan_identity(LC.sweep_plan("fixture"))
    assert LC.plan_identity(plan) == same
    _binding(monkeypatch, tile=32)
    assert LC.plan_identity(LC.sweep_plan("fixture")) != same, "a different derived tile is a different plan"
    assert LC.plan_identity(LC.sweep_plan("other")) != LC.plan_identity(LC.sweep_plan("fixture"))
