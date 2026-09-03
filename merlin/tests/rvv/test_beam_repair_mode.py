"""An incorrect SEED must not end the search.

The beam took `parents = [seed] if seed.gate_ok else []`, so a model whose BASELINE is numerically
wrong produced "0 forks, best=seed, gate_ok=False" and no lever was ever tried. That is neither
hypothetical nor rare, and the levers that would fix it are already in the space: deepjscc went from
cos 0.9176 to BIT-EXACT purely by switching to per-op register blocking. Measured on lstmnetvit int8,
the frozen seed reports w8a8_rel 0.250 (25% off) at cos 0.985 -- a search that stops there has thrown
away the one question worth asking about that model.

Repair mode keeps the search running with CORRECTNESS as the objective. Nothing can be credited a win
from an incorrect baseline: no speedup is computed at all, and rank_results sorts correctness first.
"""
import os

from merlin.mining import load_rvv_package
from merlin.mining.beam import _correctness_residual, run_beam

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
HAND_V0 = os.path.join(ROOT, "out/artifacts/targets", "rvv", "hand_v0")

_DIVS = ["lmul_class: expert='m4' vs ours='m2'",
         "fma_form: expert='vf' vs ours=None",
         "vl_strategy: expert='vsetvl_loop' vs ours='vsetivli_fixed'"]


# --------------------------------------------------------------------------- the residual

def test_the_residual_is_the_WORST_reported_relative_error():
    """Max, not mean: an aggregate can look fine while individual elements are far out -- a kernel
    measured 1209% off per element at a passing cos."""
    r = _correctness_residual({"correctness": {"w8a8_rel": 0.25, "fp32_rel": 0.258,
                                               "fp32_max_rel": 0.272, "gate_ok": False}})
    assert r == 0.272


def test_an_unmeasured_residual_is_UNKNOWN_not_zero():
    """None must not sort ahead of a candidate measured at 3%. Reporting 0.0 for "nobody measured it"
    is the failure this repo keeps hitting."""
    assert _correctness_residual({"correctness": {"gate_ok": True}}) is None
    assert _correctness_residual({}) is None
    # a bool must not be read as a number
    assert _correctness_residual({"correctness": {"rel_ok": True}}) is None


# --------------------------------------------------------------------------- the search

def _certify(*, gate_ok, residual, package_dir=None, **_):
    """Mock certify with a chosen correctness outcome."""
    return {"correctness": {"gate_ok": gate_ok, "fp32_rel": residual},
            "measurement": [{"target": "k1", "cycle_accurate": False,
                             "cycles": 1000, "wall_ns": 900}],
            "structural_match": 0.5, "divergences": _DIVS}


def test_an_incorrect_seed_no_longer_ends_the_run(tmp_path):
    """The defect itself: forks must be minted and measured."""
    def cert(**kw):
        return _certify(gate_ok=False, residual=0.25, **kw)

    out = run_beam(HAND_V0, model_dir=tmp_path / "wl", curated_text="", op_key={"op": "gemm"},
                   runs_root=tmp_path / "runs", out_root=tmp_path / "gen",
                   width=2, depth=1, top_k=1, timestamp="t", certify_fn=cert)
    assert len(out["nodes"]) > 1, "an incorrect seed still produced no forks"
    assert out["repair_mode"] is True
    assert out["seed_correctness_residual"] == 0.25


def test_no_speedup_is_credited_against_an_incorrect_baseline(tmp_path):
    """A ratio against a seed that computes the wrong answer is a speedup over a program that does
    not work. It must not exist, not merely be flagged."""
    def cert(**kw):
        return _certify(gate_ok=False, residual=0.25, **kw)

    out = run_beam(HAND_V0, model_dir=tmp_path / "wl", curated_text="", op_key={"op": "gemm"},
                   runs_root=tmp_path / "runs", out_root=tmp_path / "gen",
                   width=2, depth=1, top_k=1, timestamp="t", certify_fn=cert)
    assert all(n.get("speedup") is None for n in out["nodes"]), \
        [(n["run_id"], n.get("speedup")) for n in out["nodes"]]


def test_a_correct_seed_still_reports_speedups_and_is_not_in_repair_mode(tmp_path):
    """The normal path must be untouched."""
    def cert(**kw):
        return _certify(gate_ok=True, residual=0.0, **kw)

    out = run_beam(HAND_V0, model_dir=tmp_path / "wl", curated_text="", op_key={"op": "gemm"},
                   runs_root=tmp_path / "runs", out_root=tmp_path / "gen",
                   width=2, depth=1, top_k=1, timestamp="t", certify_fn=cert)
    assert out["repair_mode"] is False
    assert any(n.get("speedup") is not None for n in out["nodes"])


def test_a_fork_that_RESTORES_correctness_wins_over_the_incorrect_seed(tmp_path):
    """The outcome the mode exists for. rank_results sorts correctness first, so the repairing lever
    becomes `best` without any speed comparison being involved."""
    seen = {"n": 0}

    def cert(**kw):
        seen["n"] += 1
        # the seed is wrong; the first fork repairs it
        if seen["n"] == 1:
            return _certify(gate_ok=False, residual=0.25, **kw)
        return _certify(gate_ok=True, residual=0.0, **kw)

    out = run_beam(HAND_V0, model_dir=tmp_path / "wl", curated_text="", op_key={"op": "gemm"},
                   runs_root=tmp_path / "runs", out_root=tmp_path / "gen",
                   width=2, depth=1, top_k=1, timestamp="t", certify_fn=cert)
    best = out["best"]
    assert best is not None and best["gate_ok"] is True
    assert best["run_id"] != "hand_v0__beam", "the repaired fork must outrank the broken seed"


def test_the_search_climbs_toward_correctness_across_generations(tmp_path):
    """With nothing yet correct, the candidates that got CLOSER than their parent are carried forward,
    so depth buys progress instead of stopping after one generation."""
    calls = {"n": 0}

    def cert(**kw):
        calls["n"] += 1
        # seed 0.25, then steadily closer -- never correct, so only the residual can drive
        residual = {1: 0.25}.get(calls["n"], max(0.01, 0.25 - 0.02 * calls["n"]))
        return _certify(gate_ok=False, residual=residual, **kw)

    out = run_beam(HAND_V0, model_dir=tmp_path / "wl", curated_text="", op_key={"op": "gemm"},
                   runs_root=tmp_path / "runs", out_root=tmp_path / "gen",
                   width=2, depth=3, top_k=1, timestamp="t", certify_fn=cert)
    depths = {n.get("depth") for n in out["nodes"]}
    assert max(depths) > 1, f"the repair search stopped after one generation: depths={sorted(depths)}"
    best_resid = min(n["correctness_residual"] for n in out["nodes"]
                     if n.get("correctness_residual") is not None)
    assert best_resid < 0.25, "no candidate got closer than the seed"


def test_a_candidate_that_does_not_improve_the_residual_is_not_carried_forward(tmp_path):
    """Not progress, so not a parent. Otherwise the search wanders on an unchanged objective."""
    def cert(**kw):
        return _certify(gate_ok=False, residual=0.25, **kw)   # never improves

    out = run_beam(HAND_V0, model_dir=tmp_path / "wl", curated_text="", op_key={"op": "gemm"},
                   runs_root=tmp_path / "runs", out_root=tmp_path / "gen",
                   width=2, depth=3, top_k=1, timestamp="t", certify_fn=cert)
    assert max(n.get("depth") or 0 for n in out["nodes"]) == 1, \
        "a search making no progress on the residual must stop, not keep spending board time"
