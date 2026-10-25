"""Hostile request types cannot weaken scoped numerical obligations."""

import copy

import pytest

from merlin.verify.lean import ordered_evidence, ordered_policy


def contract():
    return [
        {
            "function": "forward",
            "exponent_caps": [127],
            "source_required_bounds": [{"value": 4, "upper": 142}],
            "reference_required_bounds": [{"value": 4, "upper": 150}],
        }
    ]


def roster():
    return {
        "forward": [{"id": 0, "type": {"kind": "tensor", "shape": [2], "element": {"kind": "float", "format": "f32"}}}]
    }


@pytest.mark.parametrize(
    "fault",
    [
        "boolean_cap",
        "boolean_id",
        "boolean_bound",
        "duplicate",
        "legacy_shared",
        "no_demands",
        "out_of_range",
        "wrong_function",
    ],
)
def test_malformed_scoped_contracts_are_rejected(fault):
    raw = contract()
    entry = raw[0]
    if fault == "boolean_cap":
        entry["exponent_caps"][0] = True
    elif fault == "boolean_id":
        entry["source_required_bounds"][0]["value"] = True
    elif fault == "boolean_bound":
        entry["reference_required_bounds"][0]["upper"] = True
    elif fault == "duplicate":
        entry["source_required_bounds"] *= 2
    elif fault == "legacy_shared":
        entry["required_bounds"] = entry.pop("source_required_bounds")
        del entry["reference_required_bounds"]
    elif fault == "no_demands":
        entry["source_required_bounds"] = entry["reference_required_bounds"] = []
    elif fault == "out_of_range":
        entry["exponent_caps"][0] = 255
    else:
        entry["function"] = "unrelated"
    with pytest.raises(ValueError):
        ordered_evidence.contracts(raw, roster())


def test_source_and_reference_caps_are_separate_requests():
    raw = contract()
    expected = copy.deepcopy(raw)
    actual = ordered_evidence.contracts(raw, roster())["forward"]
    assert actual == expected[0]
    assert actual["source_required_bounds"] != actual["reference_required_bounds"]
    raw[0]["reference_required_bounds"] = []
    assert ordered_evidence.contracts(raw, roster())["forward"]["reference_required_bounds"] == []


def test_existing_receipt_is_preserved_before_tools_are_invoked(tmp_path):
    destination = tmp_path / "receipt"
    destination.mkdir()
    existing = destination / "receipt.json"
    existing.write_text("retained checkpoint")
    with pytest.raises(ValueError, match="new destination"):
        ordered_policy.check_w8a8_ordered_draft(
            tmp_path,
            destination,
            input_contracts=existing,
            exponent_contracts=tmp_path / "caps.json",
            m2m_python=tmp_path / "unused",
        )
    with pytest.raises(ValueError, match="new destination"):
        ordered_policy.replay(existing, destination, m2m_python=tmp_path / "unused")
    assert existing.read_text() == "retained checkpoint"
