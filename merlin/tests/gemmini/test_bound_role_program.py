"""A bound selection that declares the in-repo role program gets the in-repo semantics.

The support-selected route evaluates the contract's ``encoding``, ``rocc_operand_roles`` role program
and ``rtl_checks`` protocol with the generic RoCC semantics. A data-bound contract carrying the same
declarations beside its instruction roster must derive identical ISA constants, readout bits, decode
and hardware checks over the same facts, and must refuse when its declarations disagree.
"""

from __future__ import annotations

import copy
import json

import pytest
import yaml

from merlin.common.paths import merlin_dir, repo_root
from merlin.targetgen import rtl_checks_generic as RC
from merlin.targetgen.rocc import semantics as S

TARGET = "gemmini"
CONTRACT = repo_root() / "examples" / TARGET / "target" / "contracts" / "target_contract.yaml"
FACTS = merlin_dir() / "tests" / "data" / "timing_probes" / "gemmini_facts.json"


def _support_contract():
    return yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))


def _facts():
    return json.loads(FACTS.read_text(encoding="utf-8"))


def _bound_contract(support):
    """The support contract's declarations in the bound shape, round-tripped through JSON like a prepared file."""
    roles = support["rocc_operand_roles"]
    classes = support["encoding"]["semantic_class"]
    roster = [{"funct": int(code), "class": label, "operands": {}} for code, label in classes.items()]
    bound = {
        "name": TARGET,
        "compute_units": copy.deepcopy(support["compute_units"]),
        "encoding": copy.deepcopy(support["encoding"]),
        "rocc_operand_roles": {
            "version": 1,
            "word_bits": 64,
            "instructions": roster,
            **{key: copy.deepcopy(roles[key]) for key in S.ROLE_PROGRAM_KEYS & set(roles)},
        },
        "rtl_checks": copy.deepcopy(support["rtl_checks"]),
    }
    return json.loads(json.dumps(bound))


def _operands(isa):
    """Constant operand pairs that exercise every declared selector code and an all-ones operand."""
    selector = isa["CONFIG_SELECTOR"]
    codes = [code << selector["offset"] for code in range(1 << selector["width"])]
    values = sorted({0, 1, 16, (1 << 31) | 5, (1 << 64) - 1, *codes})
    return [({"kind": "const", "raw": a}, {"kind": "const", "raw": b}) for a in values for b in values[:4]]


def test_bound_role_program_derives_the_support_isa_constants_and_readout_bits():
    support, facts = _support_contract(), _facts()
    bound = _bound_contract(support)
    selected = S.bind(target=TARGET, contract=bound, facts=facts)
    expected = S.derive_isa_constants(support, facts)
    assert selected.isa_constants(TARGET) == expected
    assert {"F1", "C_ACC", "ACC_I8", "ACC_ACCUM", "FULL_C_BIT", "RETAIN_SENTINEL"} <= set(expected)
    assert expected["OPERAND_FIELDS"] and expected["CONFIG_SUBTYPE"]
    readout = S.derived_readout_bits(32, contract=bound)
    assert readout and readout == S.derived_readout_bits(32, contract=support)


def test_bound_role_program_decodes_and_encodes_like_the_support_semantics():
    support, facts = _support_contract(), _facts()
    selected = S.bind(target=TARGET, contract=_bound_contract(support), facts=facts)
    isa = selected.isa_constants(TARGET)
    reference = S.derive_isa_constants(support, facts)
    for funct in sorted({*isa["FUNCT_CLASS"], 99}):
        for rs1, rs2 in _operands(isa):
            assert selected.decode_instruction(funct, rs1, rs2, isa) == S.decode_instruction(funct, rs1, rs2, reference)
    for name in sorted({*isa["FUNCT_CLASS"].values(), *isa["CONFIG_SUBTYPE"].values()}):
        sub = next((code for code, label in isa["CONFIG_SUBTYPE"].items() if label == name), 0)
        rs1 = sub << isa["CONFIG_SELECTOR"]["offset"]
        assert selected.instruction_funct(name, rs1, isa) == S.instruction_funct(name, rs1, reference)


def _trace(semantics, isa):
    """A decoded instruction roster: every classified code once, bracketed by fences."""
    instructions = [{"index": 0, "class": "FENCE", "funct": None, "decoded": {}}]
    for index, funct in enumerate(sorted(isa["FUNCT_CLASS"]), start=1):
        rs1, rs2 = {"kind": "const", "raw": 0}, {"kind": "const", "raw": 16}
        klass, decoded = semantics.decode_instruction(funct, rs1, rs2, isa)
        instructions.append({"index": index, "class": klass, "funct": funct, "decoded": decoded})
    instructions.append({"index": len(instructions), "class": "FENCE", "funct": None, "decoded": {}})
    return {"instructions": instructions, "abi": {}}


def _select(monkeypatch, tmp_path, contract, facts):
    path = tmp_path / f"contract-{len(list(tmp_path.glob('contract-*')))}.json"
    path.write_text(json.dumps(contract), encoding="utf-8")
    facts_path = tmp_path / "facts.json"
    facts_path.write_text(json.dumps(facts), encoding="utf-8")
    monkeypatch.setenv("MERLIN_TARGET_CONTRACT", str(path))
    monkeypatch.setenv("MERLIN_RTL_FACTS", str(facts_path))
    monkeypatch.setenv("MERLIN_TARGET_PATH", "")


def test_bound_protocol_checks_report_what_the_support_protocol_reports(monkeypatch, tmp_path):
    support, facts = _support_contract(), _facts()
    bound = _bound_contract(support)
    selected = S.bind(target=TARGET, contract=bound, facts=facts)
    assert isinstance(selected.rtl_checks, RC.BoundProtocolChecks)
    assert RC.protocol_from_contract(bound, TARGET) == RC.protocol_from_contract(support, TARGET)
    trace = _trace(selected, selected.isa_constants(TARGET))

    _select(monkeypatch, tmp_path, support, facts)
    expected = RC.screen(trace, None, None, target=TARGET)
    _select(monkeypatch, tmp_path, bound, facts)
    actual = selected.rtl_checks.screen(trace, None, None, target=TARGET)
    assert [check.to_dict() for check in actual.checks] == [check.to_dict() for check in expected.checks]
    assert [check.id for check in actual.checks] == [rule["id"] for rule in support["rtl_checks"]["rules"]]
    assert actual.rtl_facts == expected.rtl_facts


def test_bound_protocol_checks_refuse_a_selection_they_were_not_bound_to(monkeypatch, tmp_path):
    support, facts = _support_contract(), _facts()
    selected = S.bind(target=TARGET, contract=_bound_contract(support), facts=facts)
    trace = _trace(selected, selected.isa_constants(TARGET))
    _select(monkeypatch, tmp_path, support, facts)
    with pytest.raises(RC.RtlChecksUnavailable, match="differ from the bound"):
        selected.rtl_checks.screen(trace, None, None, target=TARGET)
    changed = copy.deepcopy(facts)
    changed["fixture"] = "changed"
    _select(monkeypatch, tmp_path, _bound_contract(support), changed)
    with pytest.raises(RC.RtlChecksUnavailable, match="differ from the bound"):
        selected.rtl_checks.load_default_facts(TARGET)


@pytest.mark.parametrize("mutation", ["relabel", "unroster", "protocol", "rule_list", "orphan_protocol", "foreign_key"])
def test_inconsistent_role_program_declarations_refuse(mutation):
    support, facts = _support_contract(), _facts()
    bound = _bound_contract(support)
    roles = bound["rocc_operand_roles"]
    if mutation == "relabel":
        roles["instructions"][0]["class"] = "RELABELED"
    elif mutation == "unroster":
        roles["instructions"].pop()
    elif mutation == "protocol":
        bound["rocc_operand_protocol"] = {"version": 1}
    elif mutation == "rule_list":
        bound["rtl_checks"] = ["decode_clean"]
    elif mutation == "orphan_protocol":
        for key in S.ROLE_PROGRAM_KEYS:
            roles.pop(key, None)
    else:
        roles["schedule_hint"] = {}
    with pytest.raises(ValueError):
        S.bind(target=TARGET, contract=bound, facts=facts)


def test_bound_roster_without_a_role_program_keeps_class_only_semantics():
    support, facts = _support_contract(), _facts()
    bound = _bound_contract(support)
    for key in S.ROLE_PROGRAM_KEYS:
        bound["rocc_operand_roles"].pop(key, None)
    bound["rtl_checks"] = ["decode_clean"]
    selected = S.bind(target=TARGET, contract=bound, facts=facts)
    assert set(selected.isa_constants(TARGET)) == {"CUSTOM_OPCODE", "FUNCT3", "FUNCT_CLASS"}
    assert isinstance(selected.rtl_checks, RC.BoundChecks)
    assert S.derived_readout_bits(32, contract=bound) == {}
