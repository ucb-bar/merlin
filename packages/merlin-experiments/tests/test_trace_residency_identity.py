"""An unresolved pointer is not evidence that two loads read the same source."""
from copy import deepcopy

import pytest

from merlin.targetgen import trace_check as TC


def load(source, destination=32):
    return {"class": "MVIN", "decoded": {"dram": source, "spad_addr": destination, "rows": 1, "cols": 16}}


def known(index=0):
    return {"kind": "argbase", "raw": None, "arg_index": index, "offset": 0}


UNKNOWN = {"kind": "unknown", "raw": None, "arg_index": None, "offset": None}


def test_unresolved_loads_cannot_establish_equal_sources():
    assert TC.residency_findings({"instructions": [load(UNKNOWN), load(deepcopy(UNKNOWN))]}) == []


@pytest.mark.parametrize("destination", [32, None])
def test_unresolved_write_invalidates_previous_content(destination):
    instructions = [load(known()), load(UNKNOWN, destination), load(known())]
    assert TC.residency_findings({"instructions": instructions}) == []


def test_known_reload_remains_a_violation():
    findings = TC.residency_findings({"instructions": [load(known()), load(known())]})
    assert len(findings) == 1 and "redundant load" in findings[0]


def test_distinct_known_sources_are_not_reloads():
    assert TC.residency_findings({"instructions": [load(known()), load(known(1))]}) == []


def test_readonly_global_reload_is_identified_without_inventing_an_address():
    source = {"kind": "global", "symbol": "padding", "offset": 8, "raw": None}
    assert "redundant load" in TC.residency_findings({"instructions": [load(source), load(source)]})[0]
    other = {**source, "symbol": "other"}
    assert TC.residency_findings({"instructions": [load(source), load(other)]}) == []
    assert TC._operand_identity({**source, "symbol": None}) is None


def test_unknown_config_never_proves_stale_mode():
    rows = [{"class": "CONFIG_EX", "rs1": UNKNOWN, "rs2": 0} for _ in range(2)]
    assert TC._stale_mode_config_findings(rows) == []


def test_known_stale_mode_remains_a_violation():
    rows = [{"class": "CONFIG_EX", "rs1": 4, "rs2": 0} for _ in range(2)]
    assert "re-issue" in TC._stale_mode_config_findings(rows)[0]


def test_redundant_configuration_is_advisory_but_actual_reload_still_refuses(monkeypatch):
    config = {"class": "CONFIG_EX", "rs1": 4, "rs2": 0}
    trace = {"instructions": [{"class": "FENCE"}, config, load(known()),
                              {"class": "MVOUT"}, config, {"class": "MVOUT"}, {"class": "FENCE"}]}
    monkeypatch.setattr(TC, "_bound_trace_view", lambda *_: (trace, {}, [], True))
    result = TC.check(trace, {"modes": {"resident_reuse": True}}, target="fixture")
    assert TC.requirements_complete(result)
    assert result["execution_config_advisory"]["required"] is False
    assert "re-issue" in result["execution_config_advisory"]["findings"][0]
    trace["instructions"].insert(-2, load(known()))
    result = TC.check(trace, {"modes": {"resident_reuse": True}}, target="fixture")
    assert not TC.requirements_complete(result)
    assert any("redundant load" in reason for reason in result["violations"])


def test_unresolved_residency_is_unavailable_and_not_complete(monkeypatch):
    trace = {"instructions": [{"class": "FENCE"}, load(UNKNOWN), load(UNKNOWN),
                              {"class": "MVOUT"}, {"class": "MVOUT"}, {"class": "FENCE"}]}
    monkeypatch.setattr(TC, "_bound_trace_view", lambda *_: (trace, {}, [], True))
    result = TC.check(trace, {"modes": {"resident_reuse": True}}, target="fixture")
    assert result["status"] == "pass"  # unresolved identities are missing evidence, not a violation
    assert result["trace_requirement_status"] == "unavailable_fields"
    assert result["unavailable_requirements"] == ["resident_reuse"]
    assert not TC.requirements_complete(result)
    assert not any("redundant load" in text for text in result["violations"])
