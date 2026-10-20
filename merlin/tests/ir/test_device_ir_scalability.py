"""Static cost measurement never counts quoted/comment text or qualifies execution."""

import hashlib
from types import SimpleNamespace

import pytest

from merlin.llvmlower import device_ir_scalability as S
from merlin.llvmlower.device_build import freeze_device_ir_budget

PROGRAM = b"""@msg = private constant [1 x i8] c"call asm define ; not instructions"
; define void @fake() { call asm }
declare void @callee()
define void @kernel() {
entry:
  call void asm sideeffect "call asm \\22escaped\\22", ""()
  br label %exit
exit:
  call void @callee()
  ret void
}
"""


def test_actual_counts_ignore_comments_strings_and_keep_code_vs_module_bytes(tmp_path):
    path = tmp_path / "device.ll"
    path.write_bytes(PROGRAM)
    analysis = S.analyze_device_ir(path)
    assert analysis["sha256"] == hashlib.sha256(PROGRAM).hexdigest()
    assert (analysis["functions"], analysis["basic_blocks"], analysis["call_sites"], analysis["inline_asm_sites"]) == (
        1,
        2,
        2,
        1,
    )
    assert analysis["lines"] == PROGRAM.count(b"\n")
    assert analysis["function_code_bytes"] + analysis["module_bytes"] == len(PROGRAM)
    assert analysis["global_definition_bytes"] == len(PROGRAM.splitlines(keepends=True)[0])
    assert analysis["quoted_bytes"] > 0 and analysis["lexical_complete"]


def test_long_constant_line_and_cross_chunk_quotes_are_bounded_and_not_code(tmp_path):
    path = tmp_path / "device.ll"
    # More than two scanner chunks in ONE initializer line, with misleading words.
    prefix = b'@data = constant [1 x i8] c"'
    literal = b"call asm define " * 150000
    path.write_bytes(
        prefix + literal + b'"\n' + PROGRAM.split(b"define void @kernel()", 1)[1].join([b"define void @kernel()", b""])
    )
    analysis = S.analyze_device_ir(path)
    assert analysis["global_definition_bytes"] == len(prefix) + len(literal) + 2
    assert analysis["functions"] == 1 and analysis["call_sites"] == 2 and analysis["inline_asm_sites"] == 1
    assert analysis["function_code_bytes"] < 512


@pytest.mark.parametrize("word", [b"call", b"callbr", b"invoke", b"asm"])
def test_exact_keywords_split_at_chunk_boundary_keep_identifier_and_quote_exclusions(tmp_path, word):
    header = b"define void @kernel() {\nentry:\n"
    call = b"call" if word == b"asm" else word
    instruction = call + b' void asm sideeffect "call asm", ""()\n'
    offset = instruction.index(word)
    padding = b" " * (1024 * 1024 - 2 - len(header) - offset)
    source = header + padding + instruction + b"""  %callback = add i64 0, 0
  %invoke = add i64 0, 0
  %asm = add i64 0, 0
  %x = ptrtoint ptr @asm to i64
asm:
  ; call invoke asm
  ret void
}
"""
    path = tmp_path / "split.ll"
    path.write_bytes(source)
    analysis = S.analyze_device_ir(path)
    assert analysis["sha256"] == hashlib.sha256(source).hexdigest()
    assert analysis["call_sites"] == 1 and analysis["inline_asm_sites"] == 1
    assert analysis["functions"] == 1 and analysis["basic_blocks"] == 2


@pytest.mark.parametrize("identifier", [b"%callback", b"@asm", b"%invoker", b"callbr_suffix"])
def test_split_identifiers_do_not_become_instruction_keywords(tmp_path, identifier):
    header = b"define void @kernel() {\n"
    padding = b" " * (1024 * 1024 - 3 - len(header))
    source = header + padding + identifier + b" = add i64 0, 0\n ret void\n}\n"
    path = tmp_path / "identifier.ll"
    path.write_bytes(source)
    analysis = S.analyze_device_ir(path)
    assert analysis["call_sites"] == 0 and analysis["inline_asm_sites"] == 0


@pytest.mark.parametrize("field", ["max_bytes", "max_code_bytes", "max_inline_asm_sites", "max_lines"])
def test_explicit_limits_refuse_expansion_while_default_is_report_only(tmp_path, field):
    path = tmp_path / "device.ll"
    path.write_bytes(PROGRAM + b'define void @second() {\n call void asm "x", ""()\n ret void\n}\n')
    analysis = S.analyze_device_ir(path)
    assert S.DeviceIRBudget(**{field: 1}).evaluate(analysis)["status"] == "accepted"
    decision = S.DeviceIRBudget(mode="enforce", **{field: 1}).evaluate(analysis)
    assert decision["status"] == "refused" and decision["exceeded"][0]["limit"] == 1
    assert decision["reason"] == "device_ir_static_expansion_budget"


def test_early_total_byte_refusal_never_reads_or_hashes_payload(tmp_path, monkeypatch):
    path = tmp_path / "large.ll"
    with path.open("wb") as stream:
        stream.truncate(16 * 1024 * 1024)
    monkeypatch.setattr(S, "analyze_device_ir", lambda _path: pytest.fail("early refusal read payload"))
    result = S.inspect_device_ir(path, S.DeviceIRBudget(mode="enforce", max_bytes=1024))
    assert result["decision"]["status"] == "refused"
    assert result["analysis"]["sha256"] is None and not result["analysis"]["lexical_complete"]


def test_policy_is_copied_before_callbacks_and_never_selected_from_environment(monkeypatch):
    monkeypatch.setenv("MERLIN_DEVICE_IR_BUDGET", "disabled")
    selected = S.DeviceIRBudget(mode="enforce", max_inline_asm_sites=5)
    frozen = freeze_device_ir_budget(SimpleNamespace(device_ir_budget=selected))
    object.__setattr__(selected, "max_inline_asm_sites", 1000)
    assert frozen.record()["max_inline_asm_sites"] == 5
    assert freeze_device_ir_budget(None).mode == "report_only"
    with pytest.raises(ValueError):
        S.DeviceIRBudget(mode="enforce")
    with pytest.raises(ValueError):
        S.DeviceIRBudget(max_bytes=True)
    with pytest.raises(ValueError):
        freeze_device_ir_budget(SimpleNamespace(device_ir_budget={"max_bytes": 1}))
