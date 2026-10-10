"""Synthetic fact selections exercise data binding without native target code."""

import copy
from types import SimpleNamespace

import pytest

from merlin.targetgen.rocc.semantics import bind


def selection():
    contract = {
        "name": "synthetic_array",
        "rocc_operand_roles": {
            "version": 1,
            "word_bits": 64,
            "instructions": [
                {"funct": 17, "class": "TRANSFER", "operands": {"rs2": {"bundle": "Extent"}}},
                {"funct": 23, "class": "COMPUTE", "operands": {}},
            ],
        },
        "rtl_checks": ["decode_clean", "legal_funct"],
    }
    facts = {
        "facts": {
            "interfaces": [
                {
                    "name": "funct_decode_table",
                    "legal_funct": [17, 23],
                    "custom_opcode": 91,
                    "funct3": 6,
                    "scope": "complete_rocc_funct7",
                    "complete_isa": True,
                },
                {
                    "name": "register_bundle_layouts",
                    "bundles": {
                        "Extent": {
                            "width": 64,
                            "fields": {
                                "rows": {"offset": 11, "width": 5},
                                "address": {"offset": 0, "width": 11},
                            },
                        }
                    },
                },
            ]
        }
    }
    return contract, facts


def bound(contract=None, facts=None):
    c, f = selection()
    return bind(target=c["name"], contract=c if contract is None else contract, facts=f if facts is None else facts)


def test_layout_decode_assembly_and_detached_selection():
    contract, facts = selection()
    owner = bound(contract, facts)
    isa = owner.isa_constants(contract["name"])
    operand = {"raw": (7 << 11) | 9, "kind": "const", "arg_index": None, "offset": 0}
    assert owner.decode_instruction(17, {"raw": None}, operand, isa) == ("TRANSFER", {"rs2.rows": 7, "rs2.address": 9})
    assert owner.instruction_funct("COMPUTE", 0, isa) == 23
    assert owner.decode_instruction(18, {}, {}, isa) == ("UNKNOWN", {})
    assert owner.decode_instruction(17, {}, {"raw": None}, isa)[1] == {"rs2.rows": None, "rs2.address": None}
    facts["facts"]["interfaces"][0]["custom_opcode"] = 3
    contract["rocc_operand_roles"]["instructions"][0]["class"] = "CHANGED"
    assert owner.isa_constants("synthetic_array")["CUSTOM_OPCODE"] == 91
    isa["CUSTOM_OPCODE"] = 3
    with pytest.raises(ValueError, match="selected ISA"):
        owner.decode_instruction(17, {}, {}, isa)
    with pytest.raises(ValueError, match="different target"):
        owner.isa_constants("another_array")


@pytest.mark.parametrize(
    "mutation", ["overlap", "unknown_width", "missing_bundle", "duplicate_interface", "unobserved_code"]
)
def test_unsupported_or_ambiguous_layouts_refuse(mutation):
    contract, facts = selection()
    fields = facts["facts"]["interfaces"][1]["bundles"]["Extent"]["fields"]
    if mutation == "overlap":
        fields["rows"]["offset"] = 0
    elif mutation == "unknown_width":
        fields["rows"]["width"] = None
        fields["rows"]["slot_width"] = 5
    elif mutation == "missing_bundle":
        contract["rocc_operand_roles"]["instructions"][0]["operands"]["rs2"]["bundle"] = "Absent"
    elif mutation == "duplicate_interface":
        facts["facts"]["interfaces"].append(copy.deepcopy(facts["facts"]["interfaces"][0]))
    else:
        contract["rocc_operand_roles"]["instructions"][0]["funct"] = 19
    with pytest.raises(ValueError):
        bound(contract, facts)


def test_complete_hardware_table_and_unknown_trace_controls():
    checks = bound().rtl_checks
    trace = {"instructions": [{"class": "TRANSFER", "funct": 17}, {"class": "FENCE", "funct": None}]}
    assert checks.screen(trace, target="synthetic_array").verdict == "ok"
    assert checks.screen({"instructions": []}, target="synthetic_array").verdict == "reject"
    changed = copy.deepcopy(trace)
    changed["instructions"][0]["class"] = "UNKNOWN"
    assert checks.screen(changed, target="synthetic_array").verdict == "reject"
    changed["instructions"][0] = {"class": "FENCE", "funct": 99}
    assert checks.screen(changed, target="synthetic_array").verdict == "reject"
    facts = checks.load_default_facts("synthetic_array")
    assertions = checks.compile_trace_checks(facts, {}, "TRACE")
    assert "TRACE: RTL_CHECK legal_funct pass" in assertions
    assert "RTL_CHECK legal_funct pass" in checks.render_trace(trace, facts)
    facts["facts"]["interfaces"][0]["custom_opcode"] = 4
    with pytest.raises(ValueError, match="original facts"):
        checks.render_trace(trace, facts)
    contract, facts = selection()
    facts["facts"]["interfaces"][0]["complete_isa"] = False
    contract["rtl_checks"] = ["legal_funct"]
    report = bound(contract, facts).rtl_checks.screen(trace, target=contract["name"])
    assert report.verdict == "vacuous" and report.n_skipped == 1


def test_ordinary_assembler_decoder_uses_bound_fields(monkeypatch):
    from merlin.runtime.backends import base
    from merlin.targetgen.rocc import asm, decode

    owner = bound()
    monkeypatch.setattr(base, "get_backend", lambda target: SimpleNamespace(rocc_semantics=owner))
    source = asm.assemble_program("synthetic_array", [("TRANSFER", 0, (7 << 11) | 9)], kernel_symbol="entry")
    trace = decode.decode_text(source, target="synthetic_array")
    (instruction,) = trace["instructions"]
    assert instruction["class"] == "TRANSFER"
    assert instruction["decoded"] == {"rs2.rows": 7, "rs2.address": 9}
    assert owner.rtl_checks.screen(trace, target="synthetic_array").verdict == "ok"


@pytest.mark.parametrize("fallback", [False, True])
def test_signed_llvm_constants_keep_the_register_bits(monkeypatch, fallback):
    from merlin.runtime.backends import base
    from merlin.targetgen.rocc import asm, decode

    owner = bound()
    monkeypatch.setattr(base, "get_backend", lambda target: SimpleNamespace(rocc_semantics=owner))
    source = asm.assemble_program("synthetic_array", [("TRANSFER", -1, -1)], kernel_symbol="entry")
    if fallback:
        source = "not a module\n" + source
    trace = decode.decode_text(source, target="synthetic_array")
    (instruction,) = trace["instructions"]
    assert instruction["decoded"] == {"rs2.rows": 31, "rs2.address": 2047}
    with pytest.raises(ValueError, match="register width"):
        owner.decode_instruction(17, {}, {"raw": -(1 << 64)}, owner.isa_constants("synthetic_array"))


@pytest.mark.parametrize(
    "rule", ["preferred_dataflow", "tile_coverage", "reuse", "movement_balance", "preload_before_compute"]
)
def test_strategy_or_unimplemented_protocol_rules_refuse(rule):
    contract, facts = selection()
    contract["rtl_checks"].append(rule)
    with pytest.raises(ValueError, match="unsupported hardware check"):
        bound(contract, facts)
