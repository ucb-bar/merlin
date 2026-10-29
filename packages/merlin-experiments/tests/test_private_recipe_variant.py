"""Reviewed numerical variants still bind exact parent and selected recipe bytes."""

import json
from copy import deepcopy
from hashlib import sha256

import pytest
from merlin_experiments.phase1.feedback import private_recipe_variant as V

from merlin.targetgen.quant_recipe import digest


def fixture():
    base = {"schema": "quant_recipe_v1", "target": "fixture", "status": "derived", "weight": {"granularity": "tensor"}}
    base["recipe_sha256"] = digest(base)
    selected = deepcopy(base)
    selected["weight"]["granularity"] = "channel"
    selected["framework_capture_policy"] = {"calibration_count": 1024}
    selected["recipe_sha256"] = digest(selected)
    parent, raw = (json.dumps(d, sort_keys=True).encode() for d in (base, selected))
    review = {
        "schema": V.SCHEMA,
        "status": "reviewed",
        "reviewed_by": "operator",
        "reason": "Approved host per-channel requantization and disjoint calibration stream",
        "base_file_sha256": sha256(parent).hexdigest(),
        "base_recipe_sha256": base["recipe_sha256"],
        "selected_file_sha256": sha256(raw).hexdigest(),
        "selected_recipe_sha256": selected["recipe_sha256"],
        "changes": V.changes(base, selected),
    }
    return review, parent, raw


def test_review_binds_parent_and_selected_and_complete_changes():
    review, parent, selected = fixture()
    proof = V.verify(review, parent=parent, selected=selected, target="fixture")
    assert proof["base_recipe_sha256"] == review["base_recipe_sha256"]
    assert len(proof["review_sha256"]) == 64


@pytest.mark.parametrize(
    "mutation",
    [
        "unreviewed",
        "no_author",
        "wrong_parent",
        "wrong_selected",
        "omitted_change",
        "changed_value",
        "wrong_target",
        "bad_self_digest",
    ],
)
def test_unreviewed_or_changed_recipe_refuses(mutation):
    review, parent, selected = fixture()
    target = "fixture"
    if mutation == "unreviewed":
        review["status"] = "proposed"
    elif mutation == "no_author":
        review["reviewed_by"] = ""
    elif mutation == "wrong_parent":
        parent += b"\n"
    elif mutation == "wrong_selected":
        selected += b"\n"
    elif mutation == "omitted_change":
        review["changes"].pop()
    elif mutation == "changed_value":
        review["changes"][0]["after"] = {"calibration_count": 1}
    elif mutation == "wrong_target":
        target = "other"
    else:
        body = json.loads(selected)
        body["weight"]["granularity"] = "block"
        selected = json.dumps(body, sort_keys=True).encode()
        review["selected_file_sha256"] = sha256(selected).hexdigest()
    with pytest.raises(ValueError):
        V.verify(review, parent=parent, selected=selected, target=target)


def test_diff_distinguishes_absence_null_and_json_types():
    assert V.changes({}, {"a": None}) == [{"path": ["a"], "before_present": False, "after": None}]
    assert V.changes({"a": None}, {}) == [{"path": ["a"], "before": None, "after_present": False}]
    assert V.changes({"a": 1}, {"a": True}) == [{"path": ["a"], "before": 1, "after": True}]
