"""How wide a grade fans out — and, on a shared host, how it declines to.

The bounds fail in different directions and so are tested in both. A sizing function that always
returned 1 would satisfy "does not overload the host"; one that always returned the core count would
satisfy "uses the machine". Every test here pins a bound against its opposite.
"""
from __future__ import annotations

import pytest

from merlin.targetgen import capsule_grade as CG

CPUS = 48
GB = 1024 ** 3

#: The real reader, captured before the autouse fixture stubs it out for the sizing tests.
_REAL_CORES_IN_USE = CG._cores_in_use


@pytest.fixture(autouse=True)
def host(monkeypatch):
    """A 48-core host with ample memory and nothing else running, unless a test says otherwise."""
    monkeypatch.delenv("MERLIN_GRADE_WORKERS", raising=False)
    monkeypatch.delenv("MERLIN_GRADE_WORKER_BYTES", raising=False)
    monkeypatch.setattr(CG.os, "sched_getaffinity", lambda pid: set(range(CPUS)))
    monkeypatch.setattr(CG, "_available_memory_bytes", lambda: 120 * GB)
    monkeypatch.setattr(CG, "_cores_in_use", lambda: 0.0)
    return monkeypatch


def test_an_idle_host_is_actually_used(host):
    """The paired direction for every backing-off test below: idle means wide."""
    assert CG.default_grade_workers() == int(CPUS * CG.GRADE_HOST_SHARE)


def test_a_busy_host_narrows_the_grade(host):
    """The point of the whole exercise: someone else is using the machine."""
    host.setattr(CG, "_cores_in_use", lambda: 39.1)
    # Truncated, not rounded: 8.9 free cores buys 8 workers, never 9.
    assert CG.default_grade_workers() == int(CPUS - 39.1) == 8


def test_backing_off_scales_with_how_busy_the_host_is(host):
    got = []
    for busy in (0.0, 12.0, 24.0, 36.0):
        host.setattr(CG, "_cores_in_use", lambda b=busy: b)
        got.append(CG.default_grade_workers())
    assert got == sorted(got, reverse=True), f"not monotonic in host load: {got}"
    assert got[0] > got[-1], "load made no difference at all"


def test_a_saturated_host_still_makes_progress(host):
    """The load bound is a courtesy, and a courtesy must not be able to halt the experiment."""
    host.setattr(CG, "_cores_in_use", lambda: float(CPUS) + 20)
    assert CG.default_grade_workers() == CG.GRADE_MIN_WORKERS


def test_an_unreadable_load_falls_back_to_the_static_bounds(host):
    """UNKNOWN is not "busy". A host that does not publish load must not be treated as saturated --
    that would silently serialize every grade on any platform without /proc."""
    host.setattr(CG, "_cores_in_use", lambda: None)
    assert CG.default_grade_workers() == int(CPUS * CG.GRADE_HOST_SHARE)


def test_the_memory_bound_has_no_courtesy_floor(host):
    """Running out of memory is not a matter of manners: the floor that protects progress against the
    LOAD bound must not lift the grade above what the host can actually hold."""
    host.setattr(CG, "_available_memory_bytes", lambda: CG.GRADE_WORKER_BYTES)
    assert CG.default_grade_workers() == 1 < CG.GRADE_MIN_WORKERS


def test_memory_pressure_from_another_tenant_narrows_the_grade(host):
    host.setattr(CG, "_available_memory_bytes", lambda: 2 * GB)
    assert CG.default_grade_workers() == 2 * GB // CG.GRADE_WORKER_BYTES


def test_an_operator_who_names_a_number_gets_it(host):
    """Even on a host this would otherwise refuse to load up."""
    host.setattr(CG, "_cores_in_use", lambda: float(CPUS))
    host.setenv("MERLIN_GRADE_WORKERS", "24")
    assert CG.default_grade_workers() == 24


def test_never_more_workers_than_capsules(host):
    assert CG.default_grade_workers(3) == 3
    assert CG.default_grade_workers(1) == 1


def test_a_small_host_leaves_cores_for_the_driver(host):
    host.setattr(CG.os, "sched_getaffinity", lambda pid: {0, 1, 2, 3})
    assert 1 <= CG.default_grade_workers() <= 2


# --------------------------------------------------------------------------------------------
# reading the host
# --------------------------------------------------------------------------------------------

def _loadavg(monkeypatch, text: str):
    """Point the reader at a synthetic /proc/loadavg, and un-stub it from the autouse fixture."""
    import merlin.targetgen.capsule_grade as mod
    monkeypatch.setattr(mod, "_cores_in_use", _REAL_CORES_IN_USE)

    class _P:
        def __init__(self, *a): pass
        def read_text(self, *a, **k): return text
    monkeypatch.setattr(mod, "Path", _P)


def test_cores_in_use_prefers_whichever_reading_is_higher(monkeypatch):
    """Smoothed load survives a dip but lags a burst; nr_running catches the burst but is one noisy
    sample. Taking the larger is what makes the pair safe in both directions."""
    _loadavg(monkeypatch, "37.94 32.65 25.42 15/3988 4103694")
    assert _REAL_CORES_IN_USE() == pytest.approx(37.94)        # average dominates
    _loadavg(monkeypatch, "0.50 0.40 0.30 31/3988 4103694")
    assert _REAL_CORES_IN_USE() == pytest.approx(30.0)         # instantaneous burst dominates


def test_cores_in_use_discounts_this_process(monkeypatch):
    """A grade must not back off on account of its own runnable task."""
    _loadavg(monkeypatch, "0.00 0.00 0.00 1/100 42")
    assert _REAL_CORES_IN_USE() == pytest.approx(0.0)


def test_cores_in_use_is_none_when_it_cannot_be_read(monkeypatch):
    _loadavg(monkeypatch, "not a loadavg line")
    assert _REAL_CORES_IN_USE() is None

    import merlin.targetgen.capsule_grade as mod

    class _Boom:
        def __init__(self, *a): pass
        def read_text(self, *a, **k): raise OSError("no /proc here")
    monkeypatch.setattr(mod, "Path", _Boom)
    assert _REAL_CORES_IN_USE() is None
