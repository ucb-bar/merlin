"""Real Q/DQ compiler gates; source-model acceptance is separate from approval."""

import json
from pathlib import Path

import pytest
from test_lean_dequant_contract import PER_CHANNEL, SMALL, wrap
from test_lean_quantize_contract import source

from merlin.llvmlower import lower, passes_xdsl
from merlin.verify import compiler_qdq as qdq
from merlin.verify import pass_validation
from merlin.verify.speculative_gate import GateFailure, SpeculativeGate


def program(kind):
    if kind == "quantize":
        return source()
    if kind == "tensor":
        return wrap([SMALL])
    return PER_CHANNEL.replace("ZERO", "i32").replace("RANGE", "quant_min = -128 : i64, quant_max = 127 : i64")


@pytest.mark.parametrize(
    "kind,textual,all_checks",
    [
        pytest.param("quantize", False, True, id="xdsl-all"),
        pytest.param("tensor", False, False, id="xdsl-tensor"),
        pytest.param("channel", True, False, id="textual-channel"),
    ],
)
def test_actual_compiler_qdq_gate_matches_ungated_llvm(tmp_path, pass_gate_tools, kind, textual, all_checks):
    raw = program(kind)
    baseline = lower.lower_model(raw, tmp_path / "baseline", targets=(), textual=textual)
    gate = SpeculativeGate(tmp_path / "gate", job_timeout=900)
    checks = pass_validation.DEFAULT_CHECKS if all_checks else frozenset({qdq.SELECTOR})
    validation = pass_validation.PassValidation(gate, pass_gate_tools, checks)
    checked = lower.lower_model(raw, tmp_path / "checked", targets=(), textual=textual, pass_validation=validation)
    assert checked.ll_path.read_bytes() == baseline.ll_path.read_bytes()
    assert gate.verdicts and all(row["verdict"] == "accepted" for row in gate.verdicts)
    qdq_rows = [row for row in gate.verdicts if row["check"] == qdq.CHECK]
    assert len(qdq_rows) == 1
    if all_checks:
        assert validation.passes == frozenset(pass_validation.CHECKS)
    else:
        assert len(gate.verdicts) == 1
    detail = qdq_rows[0]["detail"]
    assert detail["source_model_status"] == "proposed" and detail["source_spec_approved"] is False
    assert detail["source_correspondence"] == {"status": "not_requested", "original_program_bound": False}
    assert detail["native_fidelity"] is False and detail["framework_equivalence"] is False
    assert detail["source_sites"] == [qdq.SOURCE_NAMES[("quantize", "tensor", "channel").index(kind)]]
    if kind == "quantize":
        assert detail["strengthenings"]["definedness"] == "abstracted"
    for path in detail["chains"].values():
        assert json.loads(Path(path).read_text())["refinement_verdict"] == "verified"


@pytest.mark.parametrize("textual", ([False, True])[:1])
def test_actual_corrupted_qdq_output_stops_compilation(tmp_path, pass_gate_tools, monkeypatch, textual):
    from test_lean_quantize_contract import _wrong_multiply

    if textual:
        actual = passes_xdsl.preprocess_text_textual

        def corrupt(raw):
            after, stats = actual(raw)
            assert '"arith.mulf"' in after
            return after.replace('"arith.mulf"', '"arith.addf"', 1), stats

        monkeypatch.setattr(passes_xdsl, "preprocess_text_textual", corrupt)
    else:
        actual = passes_xdsl.lower_quant_ext

        def corrupt(module, **kwargs):
            count = actual(module, **kwargs)
            _wrong_multiply(module)
            return count

        monkeypatch.setattr(passes_xdsl, "lower_quant_ext", corrupt)
    gate = SpeculativeGate(tmp_path / "gate", job_timeout=900)
    validation = pass_validation.PassValidation(gate, pass_gate_tools, frozenset({qdq.SELECTOR}))
    with pytest.raises(GateFailure):
        lower.lower_model(
            program("quantize"), tmp_path / "checked", targets=(), textual=textual, pass_validation=validation
        )
    assert gate.verdicts[0]["verdict"] == "not_established"
    assert not any(path.suffix in {".ll", ".o", ".so"} for path in (tmp_path / "checked").rglob("*"))
