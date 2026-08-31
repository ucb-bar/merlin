"""The ISA-dev-tools broker routes asm/disasm/lint requests to the derived tools and stays oracle-free. The
underlying tools are unit-tested in tests/targetgen; here we lock the broker's request routing + error
handling with a synthetic model (no model venv, no llvm-mc), so the assisted-arm wiring is covered.
"""
from __future__ import annotations

import importlib.util
import sys
from dataclasses import replace

import pytest

from merlin.common.paths import merlin_dir
from merlin.targetgen.isa_model import IsaModel
from merlin.targetgen import isa_asm as A


def _load_broker():
    """Load the broker (a harness script, not an installed module) by path.

    It must be registered in ``sys.modules`` before ``exec_module``: the broker declares a dataclass
    under ``from __future__ import annotations``, and dataclasses resolves those string annotations
    by looking its class's module up there. Skipping the registration — the easy half of the
    importlib recipe — makes construction fail with a bare ``AttributeError: 'NoneType'``."""
    p = merlin_dir() / "experiments" / "capsule_bench" / "harness" / "isa_tools_broker.py"
    spec = importlib.util.spec_from_file_location("isa_tools_broker", p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
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
    # `halt_signatures` (the terminator's DERIVED decode signature), not just the mnemonic: the lint
    # detects termination by signature so a terminator and a barrier sharing one coarse class cannot
    # be confused. A model carrying only `halt_mnemonics` is one the terminator was never derived
    # for, and lint then fails closed with `halt_unknown` — see the test below.
    return IsaModel(target="fake", by_mnemonic=by, roles={"matmul": ["MatMul"]},
                    halt_mnemonics=("Halt",), halt_signatures=((hl_mask, hl_val),))


@pytest.fixture()
def broker():
    """The broker plus a context naming the fixed-format (IsaModel) arm.

    `_handle` dispatches on its context's endpoint. Resolved from the ambient run, that endpoint is
    whatever `target_experiment.yaml` happens to say, so a RoCC target in the environment routes
    every request to `_rocc_handle` and the synthetic model below is never consulted — the tests then
    measure the host's configuration instead of the routing they claim to lock. Passing the context
    explicitly is what makes these cases hermetic. `assemble` bypasses llvm-mc.
    """
    BR = _load_broker()
    m = _model()
    ctx = BR.BrokerCtx(endpoint="fixed_format", target="fake",
                       model=lambda: m, assemble=lambda text: A.assemble_text(m, text))
    return BR, ctx


def test_rocc_endpoint_is_routed_to_the_rocc_tools():
    """The other arm of that dispatch: a RoCC target's canonical artifact is `llvm.inline_asm` MLIR,
    not a `.word` kernel, so its requests must not reach the IsaModel tools."""
    BR = _load_broker()
    assert BR.is_rocc_endpoint(BR.ROCC_ENDPOINT) is True
    assert BR.is_rocc_endpoint("fixed_format") is False
    assert BR.is_rocc_endpoint(None) is False


def test_rocc_endpoint_executes_the_live_package_imports_end_to_end():
    """Catch stale broker import aliases before an agent spends a round discovering them."""
    BR = _load_broker()
    ctx = BR.BrokerCtx(endpoint=BR.ROCC_ENDPOINT, target="gemmini")
    assembled = BR._handle({"cmd": "asm", "text": "CONFIG_EX 0 0\nFENCE\n"}, ctx)
    assert assembled["n"] == 2 and "error" not in assembled
    linted = BR._handle({"cmd": "lint", "mlir": assembled["mlir"]}, ctx)
    assert linted["n"] == 2
    assert linted["n_unknown"] == 0


def test_asm_returns_words(broker):
    BR, ctx = broker
    out = BR._handle({"cmd": "asm", "text": "MatMul rd=1, rs1=1\nHalt\n"}, ctx)
    assert out["n"] == 2 and out["word_lines"].startswith(".word 0x") and "error" not in out


def test_asm_refuses_unknown_op(broker):
    BR, ctx = broker
    out = BR._handle({"cmd": "asm", "text": "NOPE\n"}, ctx)
    assert "unknown instruction" in out["error"]


def test_disasm_decodes_own_words(broker):
    BR, ctx = broker
    out = BR._handle({"cmd": "disasm", "kernel_s": "MatMul rd=5, rs1=3\nHalt\n"}, ctx)
    assert [r["mnemonic"] for r in out["records"]] == ["MatMul", "Halt"]
    assert out["records"][0]["operands"] == {"rd": 5, "rs1": 3}


def test_lint_flags_missing_halt_and_reports_coverage(broker):
    BR, ctx = broker
    out = BR._handle({"cmd": "lint", "kernel_s": "MatMul rd=1, rs1=1\n", "op": "matmul"}, ctx)
    assert any(f["rule"] == "no_halt" for f in out["findings"])       # never halts -> flagged
    assert "MatMul" in out["coverage"]["present"]


def test_lint_fails_closed_when_the_terminator_was_never_derived(broker):
    """A model carrying only `halt_mnemonics` has no derived terminator signature, so termination
    cannot be checked. Lint must say so (`halt_unknown`, INFO) rather than assert a `no_halt` error
    it has no basis for — the derive-or-report-UNKNOWN contract."""
    BR, ctx = broker
    m = replace(_model(), halt_signatures=())
    out = BR._handle({"cmd": "lint", "kernel_s": "MatMul rd=1, rs1=1\n", "op": "matmul"},
                     replace(ctx, model=lambda: m))
    rules = {f["rule"]: f["severity"] for f in out["findings"]}
    assert rules.get("halt_unknown") == "info"
    assert "no_halt" not in rules


def test_unknown_cmd_and_empty_model(broker):
    BR, ctx = broker
    assert "unknown cmd" in BR._handle({"cmd": "bogus"}, ctx)["error"]
    empty = replace(ctx, model=lambda: IsaModel(target="bare"))
    assert "no derived ISA model" in BR._handle({"cmd": "asm", "text": "x"}, empty)["error"]
