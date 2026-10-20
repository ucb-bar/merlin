"""Command integer casts must not hide CPU payload work or require convolution metadata."""

import pytest
from xdsl.context import Context
from xdsl.dialects import builtin, llvm
from xdsl.parser import Parser

from merlin.targetgen import offload_census
from merlin.targetgen.rocc import decode
from merlin.verify import host_compute_audit as HA

SOURCE = """module {
  llvm.func @entry(%input: !llvm.ptr, %output: !llvm.ptr) {
    %one = llvm.mlir.constant(1 : i64) : i64
    %small = llvm.trunc %one : i64 to i32
    %wide = llvm.zext %small : i32 to i64
    %signed = llvm.sext %small : i32 to i64
    %base = llvm.ptrtoint %input : !llvm.ptr to i64
    %address = llvm.add %base, %wide : i64
    "llvm.inline_asm"(%address, %signed) <{asm_string = ".insn r 0x7b, 0x3, 0x2, x0, $0, $1",
      constraints = "r,r,~{memory}", has_side_effects}> : (i64, i64) -> ()
    llvm.return
  }
}"""


def _judge(source, monkeypatch):
    monkeypatch.setattr(decode, "isa_constants", lambda target: {"CUSTOM_OPCODE": 123, "FUNCT_CLASS": {2: "MVIN"}})
    context = Context(allow_unregistered=True)
    for dialect in (builtin.Builtin, llvm.LLVM):
        context.load_dialect(dialect)
    module = Parser(context, source).parse_module()
    return HA.audit(module, [HA.GroupSite(0, "accelerator", "entry", 1, 1)], target="fixture")["groups"][0]


def test_integer_casts_are_proved_command_work(monkeypatch):
    row = _judge(SOURCE, monkeypatch)
    assert row["verdict"] == HA.CLEAN, row
    assert row["host_command_orchestration_conversions"] == 3
    assert row["host_value_arithmetic"] == 0
    assert row["host_total_arithmetic"] >= 3


@pytest.mark.parametrize(
    "extra",
    [
        "%value = llvm.load %input : !llvm.ptr -> i64\n llvm.store %value, %output : i64, !llvm.ptr",
        '%float = "llvm.uitofp"(%one) : (i64) -> f32',
    ],
)
def test_real_memory_or_float_work_still_refuses(monkeypatch, extra):
    row = _judge(SOURCE.replace("llvm.return", extra + "\n llvm.return"), monkeypatch)
    assert row["verdict"] != HA.CLEAN, row
    assert "host_command_orchestration_conversions" not in row


def test_target_is_forwarded_without_convolution_preparation(monkeypatch):
    monkeypatch.setattr("merlin.targetgen.input_preparation.source_convolution", lambda *args, **kwargs: None)
    observed = []
    monkeypatch.setattr(offload_census, "host_compute_row", lambda *args, **kwargs: observed.append(kwargs))
    offload_census._source_host_compute("case", {}, {}, "", "executed", target="fixture")
    assert observed == [{"input_preparation": None, "target": "fixture"}]
