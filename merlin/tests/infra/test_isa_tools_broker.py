"""The ISA-dev-tools broker routes asm/disasm/lint requests to the derived tools and stays oracle-free. The
underlying tools are unit-tested in tests/targetgen; here we lock the broker's request routing + error
handling with a synthetic model (no model venv, no llvm-mc), so the assisted-arm wiring is covered.
"""
from __future__ import annotations

import importlib.util

import pytest

from merlin.common.paths import merlin_dir
from merlin.targetgen.isa_model import IsaModel
from merlin.targetgen import isa_asm as A


def _load_broker():
    p = merlin_dir() / "experiments" / "capsule_bench" / "harness" / "isa_tools_broker.py"
    spec = importlib.util.spec_from_file_location("isa_tools_broker", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _sig(opcode: int, fields: dict) -> tuple[int, int]:
    var = 0
    for bits in fields.values():
        for b in bits:
            if isinstance(b, int) and b >= 0:
                var |= (1 << b)
    mask = (~var) & 0xFFFFFFFF
    return mask, opcode & mask


def _model() -> IsaModel:
    mm_f = {"rd": [7, 8, 9, 10, 11], "rs1": [15, 16, 17, 18, 19]}
    mm_mask, mm_val = _sig(0x2B, mm_f)
    hl_mask, hl_val = _sig(0x73, {})
    by = {"MatMul": {"class": "MatMul", "role": "matmul", "fixed_mask": mm_mask, "fixed_value": mm_val,
                     "fields": mm_f},
          "Halt": {"class": "Halt", "role": "scalar", "fixed_mask": hl_mask, "fixed_value": hl_val,
                   "fields": {}}}
    return IsaModel(target="fake", by_mnemonic=by, roles={"matmul": ["MatMul"]}, halt_mnemonics=("Halt",),
                    halt_signatures=((hl_mask, hl_val),))   # termination detection is by DERIVED signature


@pytest.fixture()
def broker(monkeypatch):
    BR = _load_broker()
    m = _model()
    # Hermetic: pin the endpoint+target to the fake model's (self-hosted external_backend), so routing does
    # not depend on the ambient MERLIN_TARGET_EXPERIMENT (whose default gemmini endpoint would send asm to
    # the RoCC path and ignore the monkeypatched model).
    monkeypatch.setattr(BR, "_endpoint_and_target", lambda: ("external_backend", "fake"))
    monkeypatch.setattr(BR, "_model", lambda: m)
    monkeypatch.setattr(BR, "_assemble", lambda text: A.assemble_text(m, text))  # bypass llvm-mc
    return BR


def test_asm_returns_words(broker):
    out = broker._handle({"cmd": "asm", "text": "MatMul rd=1, rs1=1\nHalt\n"})
    assert out["n"] == 2 and out["word_lines"].startswith(".word 0x") and "error" not in out


def test_asm_refuses_unknown_op(broker):
    out = broker._handle({"cmd": "asm", "text": "NOPE\n"})
    assert "unknown instruction" in out["error"]


def test_disasm_decodes_own_words(broker):
    out = broker._handle({"cmd": "disasm", "kernel_s": "MatMul rd=5, rs1=3\nHalt\n"})
    assert [r["mnemonic"] for r in out["records"]] == ["MatMul", "Halt"]
    assert out["records"][0]["operands"] == {"rd": 5, "rs1": 3}


def test_lint_flags_missing_halt_and_reports_coverage(broker):
    out = broker._handle({"cmd": "lint", "kernel_s": "MatMul rd=1, rs1=1\n", "op": "matmul"})
    assert any(f["rule"] == "no_halt" for f in out["findings"])       # never halts -> flagged
    assert "MatMul" in out["coverage"]["present"]


def test_unknown_cmd_and_empty_model(broker, monkeypatch):
    assert "unknown cmd" in broker._handle({"cmd": "bogus"})["error"]
    monkeypatch.setattr(broker, "_model", lambda: IsaModel(target="bare"))
    assert "no derived ISA model" in broker._handle({"cmd": "asm", "text": "x"})["error"]
