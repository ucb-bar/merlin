"""Cost-model invariants that hold independent of absolute calibration."""
from merlin.design_pressure.pressure_vector import compute_rpv
from merlin.design_pressure.synthesize import FEATURE_ACCUMULATOR, FEATURE_RESIDENT
from merlin.design_pressure.workloads.vla_action_chunk_decode import build_region
from merlin.dse.cost_model import evaluate_cost
from merlin.dse.exploitability import exploitability, row_for
from merlin.dse.hardware_space import default_cost_model
from merlin.dse.variants import build_variants, contract_plans

CM = default_cost_model()


def _rpv(H, epilogue=True):
    return compute_rpv(build_region(H=H, reuse_count=H, epilogue=epilogue, K=256, M=1, N=256))


def test_variant_cycle_ordering():
    """oracle <= software_visible <= baseline for the residency feature (reuse>1)."""
    rpv = _rpv(8)
    plans = build_variants(rpv, FEATURE_RESIDENT)
    cyc = {k: evaluate_cost(rpv, p, CM)["cycles"] for k, p in plans.items()}
    assert cyc["oracle"] <= cyc["software_visible"] <= cyc["baseline"]
    assert cyc["software_visible"] <= cyc["hardware_managed"] <= cyc["baseline"]


def test_exploitability_bounds_and_zero_at_reuse_one():
    assert exploitability(0.98, 1.0) == 0.0       # no oracle benefit -> 0
    assert 0.0 <= exploitability(2.3, 2.6) <= 1.0
    r1 = row_for(_rpv(1), FEATURE_RESIDENT, 1, CM)
    assert r1["exploitability"] == 0.0


def test_exploitability_rises_with_reuse():
    e1 = row_for(_rpv(1), FEATURE_RESIDENT, 1, CM)["exploitability"]
    e4 = row_for(_rpv(4), FEATURE_RESIDENT, 4, CM)["exploitability"]
    assert e1 == 0.0 and e4 > 0.5          # benefit becomes exploitable as reuse grows


def test_interface_is_not_free():
    """At reuse=1 the resident/commit contracts are NOT better than opaque (setup unamortised)."""
    rpv = _rpv(1)
    lat = {c: evaluate_cost(rpv, p, CM)["cycles"] for c, p in contract_plans(rpv).items()}
    assert lat["I0"] <= lat["I2"]
    assert lat["I0"] <= lat["I3"]


def test_phase_transition_best_changes_category():
    """The best contract moves I0 -> I2 -> I3 as the action horizon grows (epilogue on)."""
    def best(H):
        rpv = _rpv(H, epilogue=True)
        lat = {c: evaluate_cost(rpv, p, CM)["cycles"] for c, p in contract_plans(rpv).items()}
        return min(lat, key=lambda c: lat[c])
    assert best(1) == "I0"
    assert best(4) == "I2"
    assert best(32) == "I3"


def test_accumulator_commit_not_worth_it_without_epilogue():
    """Without an epilogue, exposing accumulator-commit (I3) is pure overhead vs I2."""
    rpv = _rpv(16, epilogue=False)
    lat = {c: evaluate_cost(rpv, p, CM)["cycles"] for c, p in contract_plans(rpv).items()}
    assert lat["I2"] <= lat["I3"]
    assert min(lat, key=lambda c: lat[c]) == "I2"
