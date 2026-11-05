"""The dense panel epoch recipe: each dropped drain is backed by an ordering fact and every route verifies.

A synthetic ``command_ordering`` record (the convolution epoch recipe's: three queues, a load queue
that may complete writes out of order) drives the recipe on small decode, prefill and ragged GEMMs.
With every fact present one drain per output block remains; a missing fact keeps the drains its flag
covers, and the recipe's ordering model reproduces the race a dropped drain would leave.
"""

from __future__ import annotations

import copy

import pytest
from author_skill_fixtures import load_script
from test_conv_epoch_recipe import facts_file, ordering_file, ordering_record

SKILL = "heavy-pass-recipes"
SHAPES = ((1, 40, 70), (8, 40, 36), (7, 23, 13))  # edge 4: decode, prefill rows, ragged tails on every axis


@pytest.fixture(scope="module")
def recipe():
    return load_script(SKILL, "dense_epoch_recipe")


def test_epochs_keep_one_drain_per_output_block_and_every_route_verifies(recipe, tmp_path):
    out = recipe.recipe(facts_file(tmp_path), SHAPES, ordering=ordering_file(tmp_path, ordering_record()))
    assert out["epoch_law"] == "admitted"
    flags = {k: v for k, v in out["law_flags"].items() if not k.startswith("_")}
    assert set(flags) == set(recipe.FLAGS)
    assert all(row["status"] in ("fact", "program") and row["basis"] for row in flags.values())
    assert flags["complete_device_drain"]["status"] == "program"
    assert "out_of_order" in out["law_flags"]["_not_relied_on"]["basis"][0]
    for case in out["cases"]:
        before, after = case["routes"]["drained"], case["routes"]["epochs"]
        assert after["result"] == "DensePanelEpochResult"
        assert after["data_commands"] == before["data_commands"] and after["effects"] == before["effects"]
        assert after["drains"] == case["output_blocks"] < before["drains"]
        assert after["library"]["unknown_premises"] == []
        for row in (before, after):
            assert row["numerics"]["source_product_match"] is True
            assert row["order_check"]["hazard_pairs"] == 0 and row["order_check"]["verdict"] == "clear"
            assert row["read_then_write"]["hazard"] == 0
        assert after["numerics"]["pending_peak"] > before["numerics"]["pending_peak"]


def test_a_missing_fact_keeps_the_drains_its_flag_covers(recipe, tmp_path):
    facts = facts_file(tmp_path)
    shape = (SHAPES[0],)
    record = ordering_record()
    record["cross_queue"] = [
        c for c in record["cross_queue"] if (c["older"], c["younger"]) != ("PRELOAD.c_addr", "MVOUT.acc_addr")
    ]
    out = recipe.recipe(facts, shape, ordering=ordering_file(tmp_path, record, "no-readout-wait.json"))
    assert out["law_flags"]["accumulator_result_visibility"]["status"] == "UNKNOWN"
    case = out["cases"][0]
    # the K stream merges; the drain before each output block's readout stays
    assert case["routes"]["epochs"]["drains"] == 2 * case["output_blocks"] < case["routes"]["drained"]["drains"]

    record = ordering_record()
    for c in record["cross_queue"]:  # a load no longer waits for the compute that read its rows
        if (c["older"], c["younger"]) == ("COMPUTE_PRELOADED.a_spad", "MVIN.spad_addr"):
            c.update(overlap="issue", tracked=False)
    out = recipe.recipe(facts, shape, ordering=ordering_file(tmp_path, record, "no-war.json"))
    assert out["law_flags"]["local_write_after_read"]["status"] == "UNKNOWN"
    case = out["cases"][0]
    assert case["routes"]["epochs"]["drains"] == case["routes"]["drained"]["drains"]

    out = recipe.recipe(facts, shape)  # no ordering facts at all: nothing to back a flag
    assert (
        out["law_flags"] is None
        and out["cases"][0]["comparison"]["drains_after"] == (out["cases"][0]["comparison"]["drains_before"])
    )


def test_the_ordering_model_shows_the_race_a_dropped_drain_would_leave(recipe, tmp_path):
    support = recipe._load("recipe_support")
    conv = recipe._load("conv_epoch_recipe")
    caps = support.capacities(facts_file(tmp_path))
    record = ordering_record()
    flags = recipe.law_flags(record, {}, conv.roles(record))
    p, s, r0 = support.dense_view(SHAPES[0], caps)
    a = support.adapter()
    resources, block = recipe.two_window(p, s, r0, 8)
    from merlin.compile.scheduling.dense_panel_pipeline import DensePanelOverlapAdmission

    admission = DensePanelOverlapAdmission(
        a.configuration_identity,
        a.numerical_policy_identity,
        "test",
        4096,
        4096,
        2,
        2,
        max_issued_readouts=block.m * block.n,
        pending_readouts=1,
        readout_configuration="ordered_snapshot_before_store_issue",
    )
    flags["finite_queue_progress"] = {"status": "fact", "basis": ["test"]}
    merged = recipe.schedule(p, s, resources, a, block, admission, recipe.law(a, flags, "test"), "merged")
    masks = {"scratchpad": caps["operand"]["rows"] - 1, "accumulator": caps["accumulator"]["rows"] - 1}
    roles = conv.roles(record)
    good = recipe.run_route(merged.kernel, a, p, s, record, roles, masks, numerics=True, check_order=False)
    assert good["numerics"]["source_product_match"] and not good["problems"]
    broken = copy.deepcopy(record)
    for c in broken["cross_queue"]:
        if c["older"] in ("COMPUTE_PRELOADED.a_spad", "PRELOAD.weight_spad") and c["younger"] == "MVIN.spad_addr":
            c.update(overlap="issue", tracked=False)
    bad = recipe.run_route(merged.kernel, a, p, s, broken, roles, masks, numerics=True, check_order=True)
    assert not bad["numerics"]["source_product_match"]
    assert bad["read_then_write"]["hazard"] > 0 and bad["problems"]


def test_count_only_skips_numerics_but_still_checks_order(recipe, tmp_path):
    out = recipe.recipe(
        facts_file(tmp_path), (SHAPES[0],), ordering=ordering_file(tmp_path, ordering_record()), numerics_budget=10
    )
    for row in out["cases"][0]["routes"].values():
        assert "skipped" in row["numerics"] and row["order_check"]["verdict"] == "clear"
