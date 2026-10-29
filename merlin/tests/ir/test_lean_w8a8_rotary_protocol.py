"""Adversarial protocol checks derived from an actual qualified pass attempt."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_lean_w8a8_rotary import evidence, invoke, prepare

from merlin.verify.lean import rotary_evidence, rotary_policy


@pytest.fixture(scope="module")
def baseline(tmp_path_factory, fidelity_tools):
    work = tmp_path_factory.mktemp("rotary-protocol-baseline")
    tools = SimpleNamespace(m2m_python=fidelity_tools.compiler_python)
    prepared = prepare(work, tools)
    report = evidence(invoke(prepared, work / "receipt", tools))
    assert report["requested_prefix_checks_passed"]
    before = json.loads(Path(report["program_exports"][0]["path"]).read_text())
    raw = json.loads(prepared[2].read_text())
    entries = rotary_evidence.contracts(raw, rotary_policy._roster(before))
    return report["evidence"], entries, before, raw


@pytest.mark.parametrize(
    "mutation",
    [
        "promote",
        "erase_premise",
        "wrong_domain",
        "bool_prefix",
        "missing_function",
        "uninhabited",
        "unbound_ssa",
        "missing_demand",
    ],
)
def test_protocol_cannot_invent_review_or_bound_evidence(baseline, mutation):
    original, entries, before, _ = baseline
    changed = copy.deepcopy(original)
    function = changed["functions"][0]
    if mutation == "promote":
        changed["whole_pass_verified"] = True
    elif mutation == "erase_premise":
        changed["arithmetic_agreement_assumed"] = False
    elif mutation == "wrong_domain":
        function["exponent_caps"][0] = 249
    elif mutation == "bool_prefix":
        function["source_prefix"]["checked"] = 1
    elif mutation == "missing_function":
        changed["functions"] = []
    elif mutation == "uninhabited":
        function["domain_inhabited"] = False
    elif mutation == "unbound_ssa":
        function["source_prefix"]["tracked_facts"][0]["id"] = 10**9
    else:
        function["source_prefix"]["required_bounds"].pop()
    with pytest.raises(ValueError):
        rotary_evidence.validate(changed, entries=entries, before=before, index_bits=64, scope="captured")


@pytest.mark.parametrize(
    "mutation",
    [
        "bool_cap",
        "overflow_cap",
        "integer_cap",
        "short_arguments",
        "duplicate_function",
        "duplicate_demand",
        "empty_demands",
        "bool_id",
    ],
)
def test_input_contracts_cannot_silently_change_scope(baseline, mutation):
    _, _, before, raw = baseline
    changed = copy.deepcopy(raw)
    entry = changed[0]
    if mutation == "bool_cap":
        entry["exponent_caps"][0] = True
    elif mutation == "overflow_cap":
        entry["exponent_caps"][0] = 255
    elif mutation == "integer_cap":
        before = copy.deepcopy(before)
        before["module"]["ops"][0]["regions"][0]["blocks"][0]["args"][0]["type"]["element"] = {
            "kind": "int",
            "width": 64,
        }
    elif mutation == "short_arguments":
        entry["exponent_caps"].pop()
    elif mutation == "duplicate_function":
        changed.append(copy.deepcopy(entry))
    elif mutation == "duplicate_demand":
        entry["required_bounds"].append(copy.deepcopy(entry["required_bounds"][0]))
    elif mutation == "empty_demands":
        entry["required_bounds"] = []
    else:
        entry["required_bounds"][0]["value"] = False
    with pytest.raises(ValueError):
        rotary_evidence.contracts(changed, rotary_policy._roster(before))
