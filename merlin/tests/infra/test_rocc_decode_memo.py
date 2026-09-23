"""Cache RoCC syntax, not provider selection, facts, or interpreted traces."""

from types import SimpleNamespace

import pytest

from merlin.runtime.backends import base
from merlin.targetgen.rocc import decode as RD

MODULE = """module {
  llvm.func @kernel() {
    %a = llvm.mlir.constant(80 : i64) : i64
    %b = llvm.mlir.constant(37 : i64) : i64
    llvm.inline_asm has_side_effects ".insn r 0x2b, 3, 9, x0, $0, $1", "r,r" %a, %b : (i64, i64) -> ()
    llvm.return
  }
}"""


@pytest.fixture(autouse=True)
def clean():
    RD._PARSE_MEMO.clear()
    yield
    RD._PARSE_MEMO.clear()


@pytest.fixture
def stub(monkeypatch):
    state = SimpleNamespace(parses=[], interpretations=[], offset=0)

    def provider(target):
        def facts(requested):
            assert requested == target
            return {"CUSTOM_OPCODE": 0x2B, "FUNCT3": 3, "OFFSET": state.offset}

        def interpret(funct, rs1, rs2, isa):
            state.interpretations.append(target)
            return target.upper(), {"value": rs1["raw"] + isa["OFFSET"], "rhs": dict(rs2)}

        return SimpleNamespace(
            rocc_semantics=SimpleNamespace(
                isa_constants=facts,
                decode_instruction=interpret,
                instruction_funct=lambda name, rs1, isa: 9,
            )
        )

    monkeypatch.setattr(base, "get_backend", provider)
    parse = RD._parse_module

    def counted_parse(text):
        state.parses.append(text)
        return parse(text)

    monkeypatch.setattr(RD, "_parse_module", counted_parse)
    return state


def test_same_text_parsed_once_but_interpreted_each_time(stub):
    first = RD.decode_text(MODULE, target="first")
    second = RD.decode_text(MODULE, target="first")
    assert first == second
    assert len(stub.parses) == 1
    assert stub.interpretations == ["first", "first"]


def test_source_label_is_per_caller(stub):
    first = RD.decode_text(MODULE, source="alpha.mlir", target="first")
    second = RD.decode_text(MODULE, source="beta.mlir", target="first")
    assert first["source"] == "alpha.mlir"
    assert second["source"] == "beta.mlir"
    assert len(stub.parses) == 1


def test_different_text_is_parsed_again(stub):
    RD.decode_text(MODULE, target="first")
    RD.decode_text(MODULE + "\n// edited\n", target="first")
    assert len(stub.parses) == 2


def test_changed_facts_reinterpret_without_reparsing(stub):
    first = RD.decode_text(MODULE, target="first")
    stub.offset = 11
    second = RD.decode_text(MODULE, target="first")
    assert first["instructions"][0]["decoded"]["value"] == 80
    assert second["instructions"][0]["decoded"]["value"] == 91
    assert len(stub.parses) == 1


def test_different_providers_share_syntax_not_interpretation(stub):
    first = RD.decode_text(MODULE, target="first")
    second = RD.decode_text(MODULE, target="second")
    assert first["instructions"][0]["class"] == "FIRST"
    assert second["instructions"][0]["class"] == "SECOND"
    assert stub.interpretations == ["first", "second"]
    assert len(stub.parses) == 1


def test_cached_syntax_does_not_bypass_provider_refusal(stub, monkeypatch):
    RD.decode_text(MODULE, target="first")

    def refuse(target):
        raise base.PluginOwnershipError("provider changed")

    monkeypatch.setattr(base, "get_backend", refuse)
    with pytest.raises(base.PluginOwnershipError, match="provider changed"):
        RD.decode_text(MODULE, target="first")
    assert len(stub.parses) == 1


def test_malformed_text_caches_parse_failure_but_repeats_fallback(stub):
    text = "not a module\n" + MODULE
    first = RD.decode_text(text, target="first")
    stub.offset = 7
    second = RD.decode_text(text, target="second")
    assert RD._PARSE_MEMO[text] is None
    assert len(stub.parses) == 1
    assert first["instructions"][0]["class"] == "FIRST"
    assert second["instructions"][0]["class"] == "SECOND"
    assert second["instructions"][0]["decoded"]["value"] == 87


def test_callers_cannot_mutate_later_nested_results(stub):
    first = RD.decode_text(MODULE, target="first")
    first["instructions"][0]["decoded"]["rhs"]["raw"] = -1
    first["summary"]["class_histogram"]["FIRST"] = 999
    second = RD.decode_text(MODULE, target="first")
    assert second["instructions"][0]["decoded"]["rhs"]["raw"] == 37
    assert second["summary"]["class_histogram"] == {"FIRST": 1}
    second["instructions"].clear()
    third = RD.decode_text(MODULE, target="first")
    assert len(third["instructions"]) == 1
    assert len(stub.parses) == 1


def test_parse_cache_bounded_and_evicted_input_reparsed(stub, monkeypatch):
    monkeypatch.setattr(RD, "_PARSE_MEMO_MAX", 4)
    for index in range(12):
        RD.decode_text(MODULE + f"\n// {index}\n", target="first")
    assert len(RD._PARSE_MEMO) == 4
    oldest = MODULE + "\n// 0\n"
    assert oldest not in RD._PARSE_MEMO
    RD.decode_text(oldest, target="first")
    assert len(stub.parses) == 13
    assert len(RD._PARSE_MEMO) == 4
