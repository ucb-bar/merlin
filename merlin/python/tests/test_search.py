"""Search layer: candidate/evaluator/grid/evolutionary/MAP-Elites over compilation strategies."""
from merlin.common import paths
from merlin.common.yaml import load_yaml
from merlin.design_pressure.pressure_vector import compute_rpv
from merlin.design_pressure.workloads.vla_action_chunk_decode import build_region
from merlin.dse.strategy import strategy_from_passes
from merlin.search import evolutionary, grid, map_elites
from merlin.search.archive import archive_rows
from merlin.search.candidate import make_candidate, seed_candidates
from merlin.search.evaluator import make_evaluator
from merlin.search.mutations import mutate, neighbours
from merlin.search.reports import build_report

BENCH = paths.merlin_dir() / "benchmarks" / "semantic_memory"


def _vla_evaluator():
    rpv = compute_rpv(build_region(H=16, reuse_count=16, epilogue=True, K=256))
    return make_evaluator([("vla", rpv)])


def test_single_region_ranks_resident_above_baseline():
    ev = _vla_evaluator()
    rows = grid.grid_search_strategies(seed_candidates(), ev)
    order = [r["strategy"] for r in rows]
    assert order.index("resident_sw_visible") < order.index("opaque_baseline")
    assert order.index("resident_commit_sw_visible") < order.index("hardware_managed_reuse")


def test_illegal_feature_everywhere_scores_junk():
    """A resident strategy is junk on a no-reuse-only workload set (legal nowhere)."""
    no_reuse = compute_rpv(load_yaml(str(BENCH / "no_reuse_matmul.yaml")))
    ev = make_evaluator([("no_reuse", no_reuse)])
    bad = make_candidate(strategy_from_passes(["hoist-pack", "make-resident"]))
    sc = ev.evaluate(bad)
    assert sc.correctness == 0.0 and sc.coverage == 0.0


def test_baseline_is_correct_everywhere():
    no_reuse = compute_rpv(load_yaml(str(BENCH / "no_reuse_matmul.yaml")))
    ev = make_evaluator([("no_reuse", no_reuse)])
    base = next(c for c in seed_candidates() if c.artifact["variant_class"] == "baseline")
    sc = ev.evaluate(base)
    assert sc.correctness == 1.0 and sc.coverage == 1.0


def test_mutation_changes_features():
    import random
    parent = next(c for c in seed_candidates() if c.artifact["variant_class"] == "baseline")
    child = mutate(parent, random.Random(0))
    assert child.id != parent.id
    assert parent.id in child.lineage
    assert len(neighbours(parent)) == 5


def test_evolutionary_search_improves_or_holds():
    ev = _vla_evaluator()
    res = evolutionary.evolutionary_search(seed_candidates(), ev, generations=5, population=5, seed=1)
    assert res["best"].score.correctness == 1.0
    assert res["best"].score.total >= res["history"][0]["best_total"] - 1e-9


def test_map_elites_builds_portfolio(tmp_path):
    ev = _vla_evaluator()
    me = map_elites.map_elites_search(seed_candidates(), ev, iterations=30, seed=2,
                                      workload_regime="decode_like")
    assert me["occupied_cells"] >= 3            # a portfolio, not one winner
    # resident_object family is represented in the portfolio
    mems = {r["memory_abstraction"] for r in archive_rows(me["archive"])}
    assert "resident_object" in mems
    arts = build_report(me["archive"], title="vla", out_dir=tmp_path)
    assert (tmp_path / "regime_map.csv").is_file()
    assert (tmp_path / "decision_report.md").is_file()
