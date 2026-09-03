"""Beam-search orchestration (mock certify, fast/deterministic): propose -> mint -> certify ->
rank -> top-k -> beam_tree, plus deferred lever-2/3 work-item capture and the gap-router."""
import os

from merlin.mining.beam import run_beam
from merlin.mining import load_rvv_package
from merlin.kernels.knobs import propose_forks

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
    from merlin.mining.beam import _cca_divergences

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


def test_an_incorrect_seed_is_explored_but_never_credited(tmp_path):
    """This test used to assert that an incorrect seed minted NO forks, which made the defect the
    contract: a model whose BASELINE is numerically wrong could never have any lever tried, even
    though the fix is often a lever already in the space (deepjscc went from cos 0.9176 to BIT-EXACT
    purely by switching to per-op register blocking). Measured on lstmnetvit int8, the frozen seed
    reports w8a8_rel 0.250 at cos 0.985, and the beam answered "0 forks, best=seed, gate_ok=False".

    The property it was really protecting is kept and asserted below: nothing is credited a win from
    an incorrect baseline. No speedup is computed at all -- a ratio against a seed that computes the
    wrong answer is a speedup over a program that does not work -- and `best` never comes back
    passing. What changes is that the search now runs, with correctness as the objective.

    This certify reports NO relative error, so there is no residual to climb. That must stop the
    search after one generation rather than spend the remaining depth: an UNKNOWN residual is not
    progress, and unknown must never be read as 0.
    """
    def all_fail(**kw):
        return {"correctness": {"gate_ok": False}, "measurement": [], "divergences": _DIVS}
    out = run_beam(HAND_V0, model_dir=tmp_path / "wl", curated_text="", op_key={"op": "gemm"},
                   runs_root=tmp_path / "runs", out_root=tmp_path / "gen",
                   width=2, depth=3, top_k=1, timestamp="t", certify_fn=all_fail)
    assert out["repair_mode"] is True
    assert out["best"] is None or not out["best"]["gate_ok"]
    # no win from a broken baseline, on either axis
    assert all(n.get("speedup") is None for n in out["nodes"])
    # explored, so the levers were at least tried
    assert len(out["nodes"]) > 1, "an incorrect seed must still get one generation of levers"
    assert list((tmp_path / "gen" / "rvv").glob("rvv_tuned_*"))
    # but stopped, because no candidate reported a residual to improve on
    assert max(n.get("depth") or 0 for n in out["nodes"]) == 1, (
        "with no residual signal the search must stop, not spend the remaining depth")


def test_rank_results_prefers_real_k1_speedup_and_fails_closed_on_incorrectness():
    """Ranking driver: real K1 speedup (measured silicon) beats the structural_match proxy; a fork
    that broke numerics never outranks a correct one no matter how 'fast'."""
    from merlin.mining.sweep import rank_results
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
    from merlin.mining.sweep import rank_results
    nodes = [
        {"run_id": "a", "gate_ok": True, "speedup": None, "structural_match": 0.6},
        {"run_id": "b", "gate_ok": True, "speedup": None, "structural_match": 0.9},
    ]
    assert [n["run_id"] for n in rank_results(nodes)][0] == "b"


def test_instrumented_beam_emits_aet_parent_and_child_runs(tmp_path, monkeypatch):
    """BB3: run_instrumented_beam opens an aet PARENT run holding beam_tree.yaml and emits one CHILD
    aet run per fork with a metrics/summary_metrics.json (aet compare reads these). Uses a mock
    certify that writes an objdump (so CCA divergences surface -> forks) + a K1 wall that improves
    with a wider N tile, so a fork earns a real speedup."""
    import json
    from pathlib import Path

    from merlin.common.paths import merlin_dir
    from merlin.mining.beam_cli import run_instrumented_beam

    monkeypatch.setenv("MERLIN_OUT_ROOT", str(tmp_path / "out"))
    ours_objd = (merlin_dir() / "tests" / "data" / "cca_asm" / "ours_baseline_matmul.objdump").read_text()
    expert_objd = merlin_dir() / "tests" / "data" / "cca_asm" / "xnnpack_f32_gemm_rvv.objdump"

    def mock_certify(*, package_dir, model_dir, runs_root, run_id, targets, baseline_run_dir):
        gen = Path(runs_root) / run_id / "generated"
        gen.mkdir(parents=True, exist_ok=True)
        (gen / "objdump.txt").write_text(ours_objd)          # -> ours CCA lifts -> divergences
        pkg = load_rvv_package(package_dir)
        n = pkg.op_match[0]["vector"][-2] if pkg.op_match else 8
        return {"correctness": {"gate_ok": True},
                "measurement": [{"target": "k1", "cycle_accurate": False,
                                 "cycles": 4_000_000 // n, "wall_ns": 900_000 // n}]}

    res = run_instrumented_beam(
        seed_pkg=HAND_V0, model_dir=tmp_path / "wl", expert_objdump=expert_objd,
        op="matmul", targets=("k1",), width=2, depth=1, top_k=1, certify_fn=mock_certify,
        expert_wall_ns=500_000)   # the XNNPACK target wall (P5): forks report attainment vs it

    # P5: the best fork reports attainment_vs_expert = xnn_wall / fork_wall (the real XNNPACK scoreboard)
    best = res.get("best") or {}
    assert isinstance(best.get("attainment_vs_expert"), float)
    parent_dir = Path(res["parent_run_dir"])
    assert (parent_dir / "beam_tree.yaml").is_file()          # full per-step record in the parent run
    assert (parent_dir / "run_record.json").is_file()
    # a wider-N fork was minted and earned a REAL K1 speedup over the frozen seed
    assert len(res["nodes"]) > 1, "no forks minted (CCA divergences did not surface)"
    assert (res.get("best") or {}).get("speedup", 0) > 1.0

    # one CHILD aet run per fork, each with a metrics/summary_metrics.json carrying the headline metrics
    runs_root = tmp_path / "out" / "runs" / "rvv" / "beam"
    summaries = list(runs_root.rglob("metrics/summary_metrics.json"))
    assert summaries, "no child summary_metrics.json emitted"
    payloads = [json.loads(p.read_text()) for p in summaries]
    fork_payloads = [p for p in payloads if p.get("role") != "beam_parent" and "speedup" in p]
    assert any(p.get("speedup") and p["speedup"] > 1.0 and p["gate_ok"] for p in fork_payloads)


def test_escalation_routes_unmet_promise_to_next_stronger_class():
    """BB2: when a fork leaves a residual (didn't achieve its promised facet), _escalations routes the
    next-stronger class for the unmet axis — turning the beam into a knob→…→CODEGEN escalation engine."""
    from merlin.mining.beam import _escalations
    from merlin.kernels.action_catalog import route
    from merlin.kernels.cca_compare import Divergence
    from merlin.kernels import cca

    action = route(Divergence("compute.accumulator_resident", True, False, "rvv"))
    assert action.action_class == "PASS"
    # the fork's emitted asm still shows resident=False -> the PASS promise was unmet.
    achieved = cca.CCA(op="matmul", backend=["rvv"],
                       compute=cca.ComputeFacet(accumulator_resident=False))
    esc = _escalations(action, achieved, {"op_match": [{"op": "matmul", "tile": [4, 8, 1], "vector": [4, 8, 1]}]})
    classes = [getattr(e.action, "action_class", None) for e in esc]
    assert "CODEGEN" in classes                       # escalated PASS -> CODEGEN (the microkernel emitter)
    # no residual -> no escalation.
    ok = cca.CCA(op="matmul", backend=["rvv"], compute=cca.ComputeFacet(accumulator_resident=True))
    assert _escalations(action, ok, {}) == []
