"""The cycle-accurate cert budget scales with the capsule's SIZE, not one flat number.

Measured defect this pins (a harness defect, not a submission defect): the per-capsule L3 budget was a
single scalar — a target-confirmed T_obs, else a flat 2400 s. ``GM0_deep_k_fits_single_i8`` emits 2310
instructions and completed its RTL cert in 1818 s; ``GM1_deep_k_spills_i8`` emits 3084 (1.335x) and
projects ~2400 s, so it was recorded a ``tool_crash`` roughly 1% over a hard wall. Its numerics passed.
A capsule failed for being large is a verdict about the harness that gets counted against the agent.

The budget is now `fixed + rate * instructions` least-squares fitted to the cert-tier walls the run
itself observed on PASSING capsules, priced at the largest promoted capsule and given the same generous
2x margin the measured-T_obs path already applies — floored at the old bound (it can only grow) and
capped so it still bounds a hang. These tests pin the derivation AND its fail-closed behavior.
"""
from __future__ import annotations

import importlib.util
import json
import sys

import pytest

from merlin.common.paths import merlin_dir

HARNESS = merlin_dir() / "experiments/capsule_bench/harness"


def _mod(name: str):
    if str(HARNESS) not in sys.path:
        sys.path.insert(0, str(HARNESS))
    spec = importlib.util.spec_from_file_location(name, HARNESS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as e:  # noqa: BLE001 — harness deps absent in this env
        pytest.skip(f"{name} not importable here: {type(e).__name__}: {e}")
    return mod


def _run_tree(root, obs: dict[str, tuple[int, float | None]], tier: str = "L3"):
    """A minimal ``_qa_work`` tree: per capsule an instruction trace of ``n`` ops and, when a wall is
    given, a PASSING cert-tier record that cost it."""
    for name, (n, wall) in obs.items():
        d = root / "_qa_work" / "runs_00" / "runs" / "t-capsule-bench" / name
        (d / "generated").mkdir(parents=True, exist_ok=True)
        (d / "generated" / "instruction_trace.json").write_text(
            json.dumps({"instructions": [{"index": i} for i in range(n)]}))
        if wall is not None:
            (d / "capsule_result.json").write_text(json.dumps(
                {"capsule": name,
                 "tiers": {tier: {"status": "pass", "timing": {"adapter_wall_s": wall}}}}))
    return root


# --- the cost model is fitted from the run's own measurements -------------------------------------

def test_cost_model_is_fitted_from_the_runs_own_passing_walls(tmp_path):
    """The MEASURED pair (7 instr -> 112 s, 2310 instr -> 1819 s) must recover the affine cost of this
    corpus: ~0.73 s per emitted instruction over a ~140 s fixed build/boot term."""
    loop = _mod("run_baseline_qa_loop")
    root = _run_tree(tmp_path, {"tiny": (7, 111.9), "big": (2310, 1819.1)})
    obs = loop._l3_cost_observations(root, "L3")
    assert sorted(obs) == [(7, 111.9), (2310, 1819.1)]
    rate, fixed, basis = loop._l3_cost_fit(obs)
    assert 0.6 < rate < 0.9 and 90 < fixed < 190 and "least squares" in basis
    # ... and it prices the capsule that was failed for being large ABOVE the old flat wall
    assert fixed + rate * 3084 > 2350


def test_a_truncated_tier_is_not_fitted(tmp_path):
    """A failed/crashed tier's wall is bounded by the budget it was given; fitting a cost model to a
    truncated measurement is how a timeout becomes self-confirming. Only PASSING walls are used."""
    loop = _mod("run_baseline_qa_loop")
    root = _run_tree(tmp_path, {"ok": (100, 200.0)})
    d = root / "_qa_work" / "runs_00" / "runs" / "t-capsule-bench" / "timedout"
    (d / "generated").mkdir(parents=True)
    (d / "generated" / "instruction_trace.json").write_text(
        json.dumps({"instructions": [{"index": i} for i in range(3084)]}))
    (d / "capsule_result.json").write_text(json.dumps(
        {"capsule": "timedout",
         "tiers": {"L3": {"status": "fail", "reason": "tool_crash: timeout",
                          "timing": {"adapter_wall_s": 2400.0}}}}))
    assert loop._l3_cost_observations(root, "L3") == [(100, 200.0)]
    # its SIZE is still known — the capsule is priced, it just does not calibrate the model
    assert loop._l3_instruction_counts(root)["timedout"] == 3084


# --- the budget itself ----------------------------------------------------------------------------

def test_a_large_capsule_is_not_failed_for_being_large(tmp_path):
    """The budget is priced at the LARGEST promoted capsule, so a 3084-instruction capsule gets more
    than the 2310-instruction one's measured 1818 s cost."""
    loop = _mod("run_baseline_qa_loop")
    root = _run_tree(tmp_path, {"tiny": (7, 111.9), "gm0": (2310, 1819.1), "gm1": (3084, None)})
    flat = loop._verilator_per_capsule_timeout()
    budget = loop._verilator_l3_budget(root, ["tiny", "gm0", "gm1"], "L3")
    assert budget >= flat, "the budget can only ever grow above the unscaled bound"
    assert budget > 2400, "GM1's projected ~2400s cost must fit inside the budget"
    rec = json.loads((root / "l3_timeout_derivation.json").read_text())[-1]
    assert rec["largest_capsule"] == "gm1" and rec["largest_instructions"] == 3084
    assert rec["budget_s"] == budget and rec["cost_model"]["s_per_instruction"] > 0
    assert "least squares" in rec["basis"]


def test_the_budget_still_bounds_a_hang(tmp_path):
    """A fit extrapolated far past its observations must not switch the hang bound off."""
    loop = _mod("run_baseline_qa_loop")
    root = _run_tree(tmp_path, {"tiny": (7, 111.9), "big": (2310, 1819.1),
                                "monster": (10_000_000, None)})
    flat = loop._verilator_per_capsule_timeout()
    budget = loop._verilator_l3_budget(root, ["tiny", "big", "monster"], "L3")
    assert budget == flat * loop.L3_BUDGET_MAX_MULT
    assert json.loads((root / "l3_timeout_derivation.json").read_text())[-1]["capped"] is True


def test_no_observations_falls_back_to_the_unscaled_bound_and_says_so(tmp_path):
    """FAIL CLOSED to today's behavior: with no passing cert-tier wall there is no cost model, so the
    unscaled bound is returned AND the reason is recorded — never a rate invented to fill the gap."""
    loop = _mod("run_baseline_qa_loop")
    root = _run_tree(tmp_path, {"a": (100, None), "b": (3084, None)})
    flat = loop._verilator_per_capsule_timeout()
    assert loop._verilator_l3_budget(root, ["a", "b"], "L3") == flat
    rec = json.loads((root / "l3_timeout_derivation.json").read_text())[-1]
    assert rec["basis"].startswith("NOT SCALED") and rec["budget_s"] == flat
    assert "cost_model" not in rec


def test_an_unsized_capsule_is_recorded_not_assumed_small(tmp_path):
    """A promoted capsule with no emitted trace cannot be priced. It is excluded from the pricing and
    NAMED in the record; the floor still covers it."""
    loop = _mod("run_baseline_qa_loop")
    root = _run_tree(tmp_path, {"tiny": (7, 111.9), "big": (2310, 1819.1)})
    loop._verilator_l3_budget(root, ["tiny", "big", "never_emitted"], "L3")
    rec = json.loads((root / "l3_timeout_derivation.json").read_text())[-1]
    assert rec["unsized"] == ["never_emitted"] and rec["n_sized"] == 2 and rec["n_capsules"] == 3
