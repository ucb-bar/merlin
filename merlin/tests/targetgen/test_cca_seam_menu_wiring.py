"""The arm-3/arm-4 CCA seam menu must ENUMERATE the RTL-derived compiler levers for any target — not just
RVV. The seam menu the prompt surfaces is `cca_contract.check_bijection(target)` +
`action_catalog.escalation_ladder(axis, target)`; these must lazily derive+register the target's spatial
levers (rtl_backend.register) so a systolic target gets a real "which section to modify + next lever"
answer, while the RVV reference and the RTL-facts moat are untouched.
"""
from __future__ import annotations

import pytest

from merlin.kernels import cca_contract as CC
from merlin.kernels import action_catalog as AC


def _has_rtl(target: str) -> bool:
    from merlin.targetgen.rtl import mlc_bridge
    return mlc_bridge.mlc_available()[0] and mlc_bridge.discovered_dim(target) is not None


def test_seam_menu_lazily_populates_derived_levers_fresh_process():
    if not _has_rtl("gemmini"):
        pytest.skip("mlc/RTL facts unavailable")
    # NO manual register() — the lazy ensure_backend hook must trigger it on first seam-menu use.
    lev = CC.leverable_axes("gemmini")
    assert {"spatial.dataflow", "spatial.accumulator_resident"} <= lev
    assert CC.routed_axes("gemmini") >= {"spatial.dataflow", "spatial.accumulator_resident"}
    lad = AC.escalation_ladder("spatial.dataflow", "gemmini")
    assert len(lad) == 1 and lad[0]["action_class"] == "HEURISTIC"
    assert lad[0]["target_seam"] == "rtl_codegen:spatial.dataflow"
    b = CC.check_bijection("gemmini")
    assert b.orphan_fields == [] and b.orphan_routes == [] and b.ladder_errors == []


def test_family_axis_not_leverable_without_hw_backing():
    """atlas has a mesh but NO discovered accumulator memory, so spatial.accumulator_resident must NOT be
    leverable for atlas (no phantom orphan) — even though it shares the `spatial` family with dataflow."""
    if not _has_rtl("atlas"):
        pytest.skip("mlc/RTL facts unavailable")
    lev = CC.leverable_axes("atlas")
    assert "spatial.dataflow" in lev
    assert "spatial.accumulator_resident" not in lev          # RTL has no accumulator memory
    b = CC.check_bijection("atlas")
    assert "spatial.accumulator_resident" not in b.orphan_fields
    assert b.orphan_fields == [] and b.orphan_routes == []


def test_rvv_reference_unchanged():
    # the in-tree reference backend's routes never go through the deriver
    assert len(CC.leverable_axes("rvv")) >= 10
    lad = AC.escalation_ladder("compute.accumulator_resident", "rvv")
    assert [r["action_class"] for r in lad] == ["PASS", "CODEGEN"]


def test_no_rtl_access_degrades_to_empty_not_crash(monkeypatch):
    """A backend resolved with no mlc/RTL access (the non-CIRCT arm) yields an empty menu, never an
    exception — so the moat (RTL levers = CIRCT arm only) self-gates without breaking the seam menu."""
    from merlin.targetgen import rtl_backend as RB
    monkeypatch.setattr(RB, "target_profile",
                        lambda t: RB.TargetProfile(target=t, legal_opcodes=None, memory_map=None, dim=None))
    probe = "noRTL_probe_target"
    assert CC.leverable_axes(probe) == set()
    assert AC.escalation_ladder("spatial.dataflow", probe) == []
    b = CC.check_bijection(probe)
    assert b.orphan_fields == [] and b.orphan_routes == []


def test_tooling_readiness_both_targets_zero_generation():
    """The zero-generation readiness gate: every tool an arm advertises must produce real output for a
    target (no agent run). Verifies atlas AND gemmini are tooling-ready before any spend."""
    import sys
    from merlin.common.paths import merlin_dir
    h = str(merlin_dir() / "experiments/capsule_bench/harness")
    if h not in sys.path:
        sys.path.insert(0, h)
    import importlib
    TR = importlib.import_module("tooling_readiness")
    for tgt in ("atlas", "gemmini"):
        if not _has_rtl(tgt):
            pytest.skip(f"mlc/RTL facts unavailable for {tgt}")
        rep = TR.readiness(tgt)
        failed = [c["check"] for c in rep["checks"] if not c["ok"]]
        assert rep["ok"], f"{tgt} not tooling-ready: {failed}"
