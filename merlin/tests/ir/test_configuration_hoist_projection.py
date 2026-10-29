"""SDK-free phase1 tree/admission controls, not Lean outcomes or approval."""

import copy
import json

import pytest

from merlin.common.paths import repo_root
from merlin.verify.configuration_hoist_bridge import Unsupported, project

ROOT = repo_root() / "out/artifacts/verification/merlin-passes/loop-scope-20261028/NEXT-HOIST/Q"


@pytest.fixture
def actual():
    return json.loads((ROOT / "actual-baseline-v1/ACTUAL.json").read_text())


def loop(record, which="before"):
    return next(item["fields"] for item in record[which]["fields"]["body"]["tuple"] if item["class"].endswith(".Loop"))


def test_complete_actual_new_producer_and_static_registry(actual):
    result = project(actual)
    assert result["move"]["extent"] == 3
    assert result["move"]["earlier"] == [0] and result["move"]["later"] == [4]
    assert result["move"]["setup"] == 1 and result["move"]["body"] == [2, 3]
    assert result["before"]["atoms"] == result["after"]["atoms"]
    assert len(result["before"]["commands"]) == 3 and len(result["after"]["commands"]) == 4
    assert result["fields"] == ["configuration:0", "configuration:1"]
    assert not result["authority"] and result["kernel_outcome"] == "UNOBSERVED"


@pytest.mark.parametrize("side", ["before", "after"])
@pytest.mark.parametrize("field", ["iter_args", "yields", "results"])
def test_all_completion_fields_both_actual_endpoints_explicitly_refused(actual, side, field):
    loop(actual, side)[field]["tuple"].append("unsupported-token")
    with pytest.raises(Unsupported, match="completion-token"):
        project(actual)


@pytest.mark.parametrize("extent", [0, -1, True, 1_000_001])
def test_empty_bad_or_unbounded_control_not_silently_hoisted(actual, extent):
    loop(actual)["extent"] = extent
    with pytest.raises(Unsupported, match="tripcount"):
        project(actual)


@pytest.mark.parametrize("key", ["unit", "produces", "awaits", "stages"])
def test_call_async_annotations_refused(actual, key):
    call = loop(actual)["body"]["tuple"][1]["fields"]
    call[key] = "device" if key in {"unit", "produces"} else {"tuple": ["token"]}
    with pytest.raises(Unsupported, match="asynchronous"):
        project(actual)


def test_iv_dependent_setter_refused(actual):
    setter = loop(actual)["body"]["tuple"][0]["fields"]
    setter["args"]["tuple"][1]["tuple"][1] = {"class": "merlin.sched.ir.expr.Var", "fields": {"name": "i"}}
    with pytest.raises(Unsupported, match="IV-dependent"):
        project(actual)


def test_even_identical_setup_body_write_is_outside_initial_domain(actual):
    body = loop(actual)["body"]["tuple"]
    body.append(copy.deepcopy(body[0]))
    with pytest.raises(Unsupported, match="ALL body write"):
        project(actual)


def test_unknown_body_not_a_totality_or_frame_claim(actual):
    loop(actual)["body"]["tuple"][1]["fields"]["instr"] = "unknown"
    with pytest.raises(Unsupported, match="explicit ordered"):
        project(actual)


def test_all_outer_metadata_and_operand_changes_remain_visible(actual):
    original = project(actual)
    actual["after"]["fields"]["attrs"]["tuple"].append({"tuple": ["changed", "value"]})
    changed = project(actual)
    assert changed["before"]["frame"] != changed["after"]["frame"]
    assert changed != original  # actual semantic rejection belongs to Lean, not this inventory


def test_changed_body_order_not_normalized_away(actual):
    # Change a retained call's constant even though its Python class/type matches.
    body = loop(actual, "after")["body"]["tuple"]
    body[0]["fields"]["args"]["tuple"][1]["tuple"][1]["fields"]["value"] = 1
    changed = project(actual)
    assert changed["before"]["atoms"] != changed["after"]["atoms"]


def test_old_producer_cannot_supply_selected_boundary(actual):
    actual["revision"] = "4499eebb959c0f92cefd92cc6d80646ba66474ed"
    with pytest.raises(Unsupported, match="no4499fallback"):
        project(actual)
