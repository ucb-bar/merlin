"""Strategy substrate: strategies reproduce contract costs and round-trip through the schema."""
from merlin.common import schemas
from merlin.design_pressure.pressure_vector import compute_rpv
from merlin.design_pressure.workloads.vla_action_chunk_decode import build_region
from merlin.dse.cost_model import evaluate_cost
from merlin.dse.hardware_space import default_cost_model
from merlin.dse.strategy import (Strategy, behavior_descriptors, default_strategies,
                                 evaluate_strategy, from_dict, load_strategies,
                                 strategy_from_passes)
from merlin.dse.variants import contract_plans


def _rpv():
    return compute_rpv(build_region(H=16, reuse_count=16, epilogue=True, K=256))


def test_strategies_reproduce_contract_costs():
    rpv, cm = _rpv(), default_cost_model()
    contracts = {k: evaluate_cost(rpv, p, cm)["cycles"] for k, p in contract_plans(rpv).items()}
    by_id = {s.id: evaluate_strategy(s, rpv, cm)["cycles"] for s in default_strategies()}
    assert by_id["opaque_baseline"] == contracts["I0"]
    assert by_id["resident_sw_visible"] == contracts["I2"]                 # resident == I2
    assert by_id["resident_commit_sw_visible"] == contracts["I3"]          # resident+commit == I3


def test_strategy_id_is_stable_and_content_addressed():
    s = default_strategies()[2]
    assert s.hash() == s.hash()
    assert s.hash() != default_strategies()[3].hash()


def test_strategy_schema_roundtrip():
    s = default_strategies()[3]
    d = s.to_dict()
    assert schemas.validate(d, "compilation_strategy") == []
    assert from_dict(d).lowering_pipeline == s.lowering_pipeline


def test_strategy_from_passes_derives_features_and_class():
    s = strategy_from_passes(["hoist-pack", "make-resident", "defer-commit"])
    assert set(s.interface_features) == {"resident_packed_tensor", "accumulator_commit"}
    assert s.variant_class == "software_visible"
    assert behavior_descriptors(s)["memory_abstraction"] == "resident_object"


def test_load_strategies_default_and_from_dicts():
    assert len(load_strategies()) == 5
    loaded = load_strategies([default_strategies()[0].to_dict()])
    assert isinstance(loaded[0], Strategy)
