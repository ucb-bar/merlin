"""M3 payoff: the static data-movement REUSE advisory in rtl_checks.

mlc's Model-3 reuse math (predict_dma_volume + spills, no arc/cycles) gives the resident footprint,
fit, and refetch factor for a declared matmul. The advisory check compares that static budget to the
observed MVIN count and flags gross operand re-streaming — a data-movement inefficiency spike is blind
to and that otherwise needs a perf sim. It is INFO severity: pure feedback, it must never move the
verdict (which rides the correctness checks). It fails closed (skipped) when mlc is unavailable.
"""
from __future__ import annotations

import yaml
import pytest

from merlin.targetgen import rtl_checks as RC
from merlin.targetgen import rtl_check_runner as RR, rtl_check_compiler as CC
from merlin.targetgen.rtl import mlc_bridge
from merlin.targetgen.rtl.facts import load_facts

import json

_FACTS = CC._facts_to_rc(load_facts("gemmini"))


def _matmul_capsule():
    for _name, p in sorted(RR._capsule_index().items()):
        c = yaml.safe_load(p.read_text())
        if RC._declared_op(c) in ("matmul", "matmul_resident") and RC._declared_mkn(c):
            return c
    return None


def _clean_single_tile_trace(n_mvin: int) -> dict:
    """An otherwise error-clean 16x16 (single-tile) matmul RoCC sequence with n_mvin operand loads:
    braced by FENCE, configs before use, PRELOAD before COMPUTE, exactly one MVOUT (1 tile)."""
    seq = [("FENCE", None), ("CONFIG_EX", 0), ("CONFIG_LD", 0), ("CONFIG_ST", 0)]
    seq += [("MVIN", 2)] * n_mvin
    seq += [("PRELOAD", 6), ("COMPUTE_PRELOADED", 4), ("MVOUT", 3), ("FENCE", None)]
    return {"instructions": [{"index": i, "class": c, "funct": f} for i, (c, f) in enumerate(seq)]}


def _dm(rep):
    return next(c for c in rep.checks if c.id == "T0.data_movement_reuse")


@pytest.mark.skipif(mlc_bridge.matmul_reuse_prediction(16, 16, 16, dim=16, capacity_bytes=262144) is None,
                    reason="mlc Model-3 reuse model unavailable")
def test_reuse_within_budget_passes_and_verdict_ok():
    cap = _matmul_capsule()
    assert cap is not None
    M, N, K = RC._declared_mkn(cap)
    pred = mlc_bridge.matmul_reuse_prediction(M, N, K, dim=RC._mesh(_FACTS)[0],
                                              capacity_bytes=_FACTS["scratchpad_bytes"],
                                              elem_bytes=RC._elem_bytes(cap))
    ideal = pred["footprint_tiles"] * pred["refetch"]
    rep = RC.screen(_clean_single_tile_trace(ideal), cap, _FACTS)
    assert _dm(rep).severity == "info" and _dm(rep).status == "pass"
    assert rep.verdict == "ok"                              # otherwise-clean trace, nothing gates


@pytest.mark.skipif(mlc_bridge.matmul_reuse_prediction(16, 16, 16, dim=16, capacity_bytes=262144) is None,
                    reason="mlc Model-3 reuse model unavailable")
def test_gross_restream_flagged_but_never_gates_verdict():
    cap = _matmul_capsule()
    M, N, K = RC._declared_mkn(cap)
    pred = mlc_bridge.matmul_reuse_prediction(M, N, K, dim=RC._mesh(_FACTS)[0],
                                              capacity_bytes=_FACTS["scratchpad_bytes"],
                                              elem_bytes=RC._elem_bytes(cap))
    ideal = max(pred["footprint_tiles"] * pred["refetch"], 1)
    rep = RC.screen(_clean_single_tile_trace(6 * ideal), cap, _FACTS)   # 6x -> well past the 3x slack
    dm = _dm(rep)
    assert dm.severity == "info" and dm.status == "fail"    # flagged as feedback
    assert dm.ratio and dm.ratio >= 3
    # ...yet the verdict is unchanged: an info fail contributes to neither the error nor warn count.
    assert rep.n_error == 0 and rep.verdict == "ok"


def test_fails_closed_to_skipped_without_mlc(monkeypatch):
    """When the mlc reuse model is unavailable the advisory is skipped honestly (never a fabricated pass)."""
    monkeypatch.setattr(mlc_bridge, "matmul_reuse_prediction", lambda *a, **k: None)
    cap = _matmul_capsule()
    rep = RC.screen(_clean_single_tile_trace(2), cap, _FACTS)
    dm = _dm(rep)
    assert dm.status == "skipped" and "unavailable" in dm.message
