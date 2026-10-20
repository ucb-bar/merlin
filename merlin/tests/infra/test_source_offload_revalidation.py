"""One live before/after evidence check per source freshness boundary.

Synthetic ownership only: these cases measure no runtime or hardware costs.
"""

import pytest
from test_source_dense_lowering import run
from test_source_offload_cost import cost_case
from xdsl.dialects import builtin

from merlin.llvmlower.source_model import bind_source_lowerings, source_provider_builder


def registered(config):
    _, builder = bind_source_lowerings((config,), None, None, None)
    return builder.__self__


def test_live_owner_is_checked_once_before_and_after_each_freshness_boundary(tmp_path):
    config, _, _, _, _, calls = cost_case(tmp_path)
    before = len(calls)
    owner = registered(config)
    assert len(calls) - before == 2, "registration must check the original live owner before and after"
    for _ in range(2):
        before = len(calls)
        owner.revalidate()
        assert len(calls) - before == 2, "one boundary must not repeat complete evidence I/O"


def test_same_boundary_routing_reuses_the_already_verified_comparison(tmp_path):
    config, module, _, _, _, calls = cost_case(tmp_path, bounds=(90, 110))
    owner = registered(config)
    before = len(calls)
    regions, receipt = owner._place_dense(module, config, ())
    assert regions == () and receipt["actual_route"] == "original"
    assert len(calls) - before == 2, "routing must not reopen the same evidence without an intervening boundary"


def test_registration_never_uses_a_previously_verified_but_expired_owner(tmp_path):
    config, _, _, _, evidence, _ = cost_case(tmp_path)
    evidence["current"] = False
    with pytest.raises(ValueError, match="measurement owner"):
        registered(config)


def test_next_freshness_boundary_reopens_owner_instead_of_caching_success(tmp_path):
    config, _, _, _, evidence, _ = cost_case(tmp_path)
    owner = registered(config)
    owner.revalidate()
    evidence["current"] = False
    with pytest.raises(ValueError, match="measurement owner"):
        owner.revalidate()


def test_empty_provider_fallback_still_reopens_original_owner_at_final_selection(tmp_path):
    config, module, _, _, evidence, calls = cost_case(tmp_path, ownership=False, unknown=True)
    _, builder = run(tmp_path, config, module)
    before = len(calls)
    assert source_provider_builder(builder) is None
    assert len(calls) - before == 2
    evidence["current"] = False
    with pytest.raises(ValueError, match="measurement owner"):
        source_provider_builder(builder)


@pytest.mark.parametrize("mutation", ["budget", "storage", "source", "callback", "plan", "policy"])
def test_live_evidence_callback_cannot_change_outer_selection_within_the_same_boundary(tmp_path, mutation):
    config, _, _, decision, _, calls = cost_case(tmp_path)
    original_check = decision._check
    armed = False

    def original_owner():
        record = original_check()
        if armed:
            if mutation == "budget":
                object.__setattr__(config, "static_budget", config.static_budget + 1)
            elif mutation == "storage":
                object.__setattr__(config.storage.lhs, "capacity", config.storage.lhs.capacity + 8)
            elif mutation == "source":
                config.offload_plan.source.module.attributes["prov.changed"] = builtin.UnitAttr()
            elif mutation == "callback":
                object.__setattr__(config, "prepare", lambda *args: pytest.fail("mutated provider must not run"))
            elif mutation == "plan":
                object.__setattr__(config, "offload_plan", None)
            else:
                object.__setattr__(config.selected_contracts, "numerical_policy", "changed policy")
        return record

    object.__setattr__(decision, "_check", original_owner)
    owner = registered(config)
    armed = True
    before = len(calls)
    with pytest.raises(ValueError, match="changed|stale|differs|replan"):
        owner.revalidate()
    assert len(calls) - before == 2


def test_registration_rechecks_outer_controls_after_the_original_owner_returns(tmp_path):
    config, _, _, decision, _, _ = cost_case(tmp_path)
    original_check = decision._check

    def original_owner():
        record = original_check()
        object.__setattr__(config, "borrow_contiguous", not config.borrow_contiguous)
        return record

    object.__setattr__(decision, "_check", original_owner)
    with pytest.raises(ValueError, match="changed|stale|differs|replan"):
        registered(config)
