"""Beam-search orchestration (mock certify, fast/deterministic): propose -> mint -> certify ->
rank -> top-k -> beam_tree, plus deferred lever-2/3 work-item capture and the gap-router."""
import os

from merlin.rvvgen.beam import run_beam
from merlin.rvvgen import load_rvv_package
from merlin.kernels.rvv_knobs import propose_forks

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
HAND_V0 = os.path.join(ROOT, "out/artifacts/targets", "rvv", "hand_v0")

_DIVS = ["lmul_class: expert='m4' vs ours='m2'",
         "fma_form: expert='vf' vs ours=None",
         "vl_strategy: expert='vsetvl_loop' vs ours='vsetivli_fixed'"]


def test_cca_divergences_lifts_ours_from_objdump_and_diffs_vs_expert(tmp_path):
    # the beam's CCA-mode divergence flow: lift OUR emitted CCA from a run's objdump.txt, diff vs the
    # expert CCA -> the CCA Divergences that drive the CCA-native proposer (whose forks get audited).
    from pathlib import Path

    from merlin.common.paths import merlin_dir
    from merlin.kernels import cca
    from merlin.rvvgen.beam import _cca_divergences

    gen = tmp_path / "run1" / "generated"
    gen.mkdir(parents=True)
    # ours = the baseline (mul_add); expert = fused_fma -> a contraction_form divergence must surface
    (gen / "objdump.txt").write_text(
        (merlin_dir() / "tests" / "data" / "cca_asm" / "ours_baseline_matmul.objdump").read_text())
    expert = cca.CCA(op="matmul", backend=["rvv"],
                     compute=cca.ComputeFacet(op="matmul", contraction_form="fused_fma"))
    divs = _cca_divergences(tmp_path / "run1", expert, {"op": "matmul"})
    axes = {d.axis for d in divs}
    assert "compute.contraction_form" in axes
    # no objdump -> empty (never crashes)
    assert _cca_divergences(tmp_path / "nope", expert, {"op": "matmul"}) == []


def _mock_certify(*, package_dir, model_dir, runs_root, run_id, targets, baseline_run_dir):
    """Wider N tile/vector -> higher structural_match (so the beam should climb toward it)."""
    pkg = load_rvv_package(package_dir)
    n = pkg.op_match[0]["vector"][-2] if pkg.op_match else 8
    return {"correctness": {"gate_ok": True},
            "measurement": [{"target": "spike", "cycle_accurate": False, "cycles": 4_000_000 // n}],
            "structural_match": min(0.95, 0.45 + 0.02 * n),
            "divergences": _DIVS}


def test_gap_router_splits_forkable_and_deferred():
    pkg = load_rvv_package(HAND_V0)
    props = propose_forks(_DIVS, pkg.knobs)
    forkable = [p for p in props if p.forkable]
    deferred = [p for p in props if not p.forkable]
    assert any(p.targets == "lmul_class" and "op_match" in p.overrides for p in forkable)
    # the fused-vfmacc recovery + vl-loop are honestly DEFERRED (need lever-2/3, not a knob)
    levers = {p.targets: p.lever for p in deferred}
    assert levers.get("fma_form") == "llvm_requirement"
    assert levers.get("vl_strategy") == "llvm_requirement"


def test_beam_climbs_and_writes_tree(tmp_path):
    out = run_beam(HAND_V0, model_dir=tmp_path / "wl", curated_text="", op_key={"op": "gemm"},
                   runs_root=tmp_path / "runs", out_root=tmp_path / "gen",
                   width=2, depth=2, top_k=1, timestamp="t", certify_fn=_mock_certify)
    # seed (N=8) -> wider-N forks should win
    seed = next(n for n in out["nodes"] if n["lever"] == "seed")
    assert out["best"]["structural_match"] >= seed["structural_match"]
    # forks were minted on disk with lineage
    forks = list((tmp_path / "gen" / "rvv").glob("rvv_tuned_v*_d*_*"))
    assert forks, "no fork packages minted"
    fk = load_rvv_package(forks[0])
    assert fk.manifest["lineage"]["parent_run_id"] in {n["run_id"] for n in out["nodes"]}
    # deferred lever-2/3 work-items (fma fusion, vl-loop) recorded, not silently dropped
    deferred_targets = {d["targets"] for d in out["deferred"]}
    assert {"fma_form", "vl_strategy"} <= deferred_targets
    assert os.path.isfile(out["tree_path"])


def test_beam_stops_when_no_correct_parent(tmp_path):
    def all_fail(**kw):
        return {"correctness": {"gate_ok": False}, "measurement": [], "divergences": _DIVS}
    out = run_beam(HAND_V0, model_dir=tmp_path / "wl", curated_text="", op_key={"op": "gemm"},
                   runs_root=tmp_path / "runs", out_root=tmp_path / "gen",
                   width=2, depth=3, top_k=1, timestamp="t", certify_fn=all_fail)
    # seed fails the gate -> no parents -> no forks minted
    assert out["best"] is None or not out["best"]["gate_ok"]
    assert not list((tmp_path / "gen" / "rvv").glob("rvv_tuned_*"))


def test_rank_results_prefers_real_k1_speedup_and_fails_closed_on_incorrectness():
    """Ranking driver: real K1 speedup (measured silicon) beats the structural_match proxy; a fork
    that broke numerics never outranks a correct one no matter how 'fast'."""
    from merlin.rvvgen.sweep import rank_results
    nodes = [
        {"run_id": "seed", "gate_ok": True, "speedup": 1.0, "structural_match": 0.9},
        {"run_id": "fast_correct", "gate_ok": True, "speedup": 1.4, "structural_match": 0.5},
        {"run_id": "high_sm_slow", "gate_ok": True, "speedup": 0.8, "structural_match": 0.99},
        {"run_id": "fast_but_broken", "gate_ok": False, "speedup": 3.0, "structural_match": 0.99},
    ]
    ranked = [n["run_id"] for n in rank_results(nodes)]
    assert ranked[0] == "fast_correct"          # real speedup drives (beats the higher structural_match)
    assert ranked[-1] == "fast_but_broken"      # broke numerics -> last, no speed credit (real-vs-fake)


def test_rank_results_falls_back_to_structural_match_without_k1():
    """No k1 run (speedup=None everywhere) -> ranking falls back to the structural_match proxy."""
    from merlin.rvvgen.sweep import rank_results
    nodes = [
        {"run_id": "a", "gate_ok": True, "speedup": None, "structural_match": 0.6},
        {"run_id": "b", "gate_ok": True, "speedup": None, "structural_match": 0.9},
    ]
    assert [n["run_id"] for n in rank_results(nodes)][0] == "b"
