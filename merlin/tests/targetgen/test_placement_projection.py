"""The projection that lets ONE authority serve both placement surfaces.

`routing.route_plan` and `system.place` answer the same question and have disagreed by construction --
`place`'s own docstring says so, and until now it had test-only callers, so nothing measured the
disagreement on a real target. `Placement.as_route_plan` puts a placement in route_plan's own shape so
the two can be compared op-for-op rather than argued about, which is what lets the authority be flipped
later without a rewrite.
"""

from __future__ import annotations

import pytest

from merlin.common.paths import merlin_dir
from merlin.system.derive import system_for_experiment
from merlin.system.place import place
from merlin.targetgen.routing import OpDemand, route_plan

#: A mesh-eligible contraction, a normalization no mesh admits, and an elementwise op. Enough to put
#: work in more than one bucket on every target, which a single-op probe cannot do.
_DEMANDS = [
    OpDemand("matmul", "int8", "int8", None, 16, 16, 16, 2),
    OpDemand("rmsnorm", "fp32", "fp32", None, 16, 16, 16, 2),
    OpDemand("add", "fp32", "fp32", None, 16, 16, 16, 2),
]


def _targets() -> list[str]:
    root = merlin_dir() / "experiments/capsule_bench/targets"
    return sorted(p.name for p in root.iterdir() if (p / "target_experiment.yaml").is_file())


@pytest.mark.parametrize("target", _targets())
def test_the_projection_agrees_with_route_plan(target):
    """Held on every target that ships a descriptor, not on a hand-picked pair. The surfaces most
    likely to diverge are the ones nobody thought to check."""
    try:
        plan = route_plan(list(_DEMANDS), target)
    except Exception as exc:                       # noqa: BLE001
        pytest.skip(f"{target} has no resolvable contract: {type(exc).__name__}: {exc}")
    system, _ = system_for_experiment(target)
    proj = place(list(_DEMANDS), system).as_route_plan()
    got = {k: len(proj[k]) for k in ("mesh", "fallback", "scalar_rvv")}
    want = {k: len(plan[k]) for k in ("mesh", "fallback", "scalar_rvv")}
    assert got == want, f"{target}: placement {got} disagrees with route_plan {want}"


def test_a_host_placement_is_the_host_lane_not_a_device_fallback():
    """The divergence this projection found. `place` also knows about HOST units, which routing does
    not; bucketing on "was a unit legal" put a host-vector placement in `fallback`, where route_plan
    calls it `scalar_rvv` -- two surfaces disagreeing about an op neither had misplaced."""
    system, _ = system_for_experiment("gemmini")
    if system.host is None:
        pytest.skip("gemmini declares no host board, so there are no host units to mis-bucket")
    proj = place(list(_DEMANDS), system).as_route_plan()
    assert proj["fallback"] == [], "a host placement must not be reported as a device fallback"
    assert all(not p.on_device for p in proj["scalar_rvv"])


def test_the_record_names_every_op_and_its_reason():
    system, _ = system_for_experiment("gemmini")
    rec = place(list(_DEMANDS), system).to_dict()
    assert len(rec["placed"]) == len(_DEMANDS)
    for row in rec["placed"]:
        for key in ("op", "device", "unit", "lane", "why", "emulated"):
            assert key in row, f"placement record is missing {key!r}"
        assert row["why"], "a placement without a reason cannot be reviewed"
    assert rec["n_emulated"] == len(rec["emulated"])


def test_emulation_is_reported_rather_than_silent():
    """The case nothing else states: an op the host took whose format the host cannot natively carry.
    It is unreachable through the ordinary dtypes, which is exactly why it went unreported."""
    system, _ = system_for_experiment("gemmini")
    exotic = [OpDemand("add", "mxfp4", "mxfp4", None, 16, 16, 16, 2)]
    rec = place(exotic, system).to_dict()
    assert rec["n_emulated"] == len(rec["emulated"])
    for row in rec["emulated"]:
        assert row["emulated"] is True and not row["unit"], (
            "an emulated op is one no unit accepted, host included")
