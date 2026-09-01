"""A tier selection that drops a MANDATORY tier can never score a pass — so it must not be possible.

Measured before the fix: `--tiers L3` on a radiance submission reported ``pass: false`` with
``{L2: unavailable, L3: pass}``, while the same bytes passed under ``--tiers L2,L3``. The unreached
mandatory tier is scored ``NOT_RUN_IS_NOT_PASS``, which is right in itself, so the row reads as a backend
failure while the oracle that ran says ``pass``. Across two runs and six repeats ~25000 L3-only
self-checks produced not one pass while L2 passed normally, and every one of them was charged to the
agent.

These tests are target-agnostic: the capsule set and adapter maps are synthetic, and the tier names are
whatever the fixtures declare, so nothing here encodes a fact about any shipped target.
"""
from __future__ import annotations

import sys

import pytest
import yaml

from merlin.common.paths import merlin_dir

sys.path.insert(0, str(merlin_dir() / "experiments/capsule_bench/harness"))

import agent_selfcheck as SC  # noqa: E402


def _capsule(root, name, required):
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "capsule.yaml").write_text(yaml.safe_dump({"name": name, "required_oracle_tiers": required}))
    return d


def test_mandatory_tiers_are_read_from_the_capsules_that_will_be_graded(tmp_path):
    _capsule(tmp_path, "A_op", ["TA", "TB"])
    _capsule(tmp_path, "B_op", ["TB", "TC"])
    assert SC.mandatory_tiers(tmp_path) == {"TA", "TB", "TC"}


def test_a_capsule_subset_narrows_the_mandatory_set(tmp_path):
    _capsule(tmp_path, "A_op", ["TA"])
    _capsule(tmp_path, "B_op", ["TZ"])
    assert SC.mandatory_tiers(tmp_path, {"A_op"}) == {"TA"}
    assert "TZ" not in SC.mandatory_tiers(tmp_path, {"A_op"})


def test_a_capsule_declaring_nothing_demands_nothing(tmp_path):
    d = tmp_path / "C_op"
    d.mkdir()
    (d / "capsule.yaml").write_text(yaml.safe_dump({"name": "C_op"}))
    assert SC.mandatory_tiers(tmp_path) == set()


def test_an_unreadable_capsule_states_no_requirement_rather_than_raising(tmp_path):
    d = tmp_path / "D_op"
    d.mkdir()
    (d / "capsule.yaml").write_text("{{ not: valid: yaml")
    assert SC.mandatory_tiers(tmp_path) == set()


def test_a_dropped_mandatory_tier_is_restored():
    full = {"TL": object(), "TC": object()}
    picked = {"TC": full["TC"]}                      # the caller asked for the cert tier alone
    out, added = SC.restore_mandatory_tiers(picked, full, {"TL"})
    assert added == ["TL"]
    assert set(out) == {"TL", "TC"}, "the requested tier must still run beside the restored one"


def test_restoring_never_invents_a_tier_the_endpoint_cannot_reach():
    full = {"TC": object()}
    out, added = SC.restore_mandatory_tiers({"TC": full["TC"]}, full, {"TL"})
    assert added == [] and set(out) == {"TC"}, "an unreachable mandatory tier cannot be conjured"


def test_a_selection_that_already_covers_its_mandatory_tiers_is_left_alone():
    full = {"TL": object(), "TC": object()}
    picked = dict(full)
    out, added = SC.restore_mandatory_tiers(picked, full, {"TL"})
    assert added == [] and out is picked


def test_the_default_ladder_is_not_widened_when_nothing_is_mandatory():
    full = {"TL": object(), "TC": object()}
    out, added = SC.restore_mandatory_tiers({"TL": full["TL"]}, full, set())
    assert added == [] and set(out) == {"TL"}


@pytest.mark.parametrize("requested", ["TC", "TC,TL", ""])
def test_no_reachable_mandatory_tier_survives_selection_missing(requested):
    """Whatever is asked for, the resulting ladder covers every reachable mandatory tier."""
    full = {"TL": object(), "TC": object()}
    default = {"TL": full["TL"]}
    picked, err = SC.select_tiers(full, default, requested)
    assert err is None
    picked, _added = SC.restore_mandatory_tiers(picked, full, {"TL"})
    assert "TL" in picked


def test_promotion_asks_for_the_loop_tier_beside_the_cert_tier():
    """`promote` requesting the cert tier alone is what made a promoted cert unable to record a pass."""
    src = (merlin_dir() / "experiments/capsule_bench/harness/tier_promote.py").read_text()
    assert '"tiers": f"{loop_tier},{cert_tier}"' in src, \
        "a promoted cert job must request the loop tier too, or its mandatory tier goes unreached"
    assert '"tiers": cert_tier,' not in src, "the cert-tier-only request must not come back"


def test_the_cert_child_keeps_its_output():
    """Thousands of certs failed with no diagnostic because the child's output went to DEVNULL."""
    src = (merlin_dir() / "experiments/capsule_bench/harness/simjob_broker.py").read_text()
    assert "stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL" not in src
    assert "simlog_" in src, "the cert child's output must land somewhere an operator can read"
