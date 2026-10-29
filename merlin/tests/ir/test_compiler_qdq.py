"""Real Q/DQ compiler gates; source-model acceptance is separate from approval."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_lean_dequant_contract import PER_CHANNEL, SMALL, wrap
from test_lean_quantize_contract import source

from merlin.llvmlower import lower, passes_xdsl
from merlin.verify import compiler_qdq as qdq
from merlin.verify import pass_validation
from merlin.verify.gate_errors import GateRejected, GateUnsupported
from merlin.verify.speculative_gate import GateFailure, SpeculativeGate


def program(kind):
    if kind == "quantize":
        return source()
    if kind == "tensor":
        return wrap([SMALL])
    return PER_CHANNEL.replace("ZERO", "i32").replace("RANGE", "quant_min = -128 : i64, quant_max = 127 : i64")


@pytest.mark.parametrize("kind", ["quantize", "tensor", "channel"])
@pytest.mark.parametrize("textual", [False, True])
def test_actual_compiler_qdq_gate_matches_ungated_llvm(tmp_path, pass_gate_tools, kind, textual):
    raw = program(kind)
    baseline = lower.lower_model(raw, tmp_path / "baseline", targets=(), textual=textual)
    gate = SpeculativeGate(tmp_path / "gate", job_timeout=900)
    validation = pass_validation.PassValidation(gate, pass_gate_tools, frozenset({qdq.SELECTOR}))
    checked = lower.lower_model(raw, tmp_path / "checked", targets=(), textual=textual, pass_validation=validation)
    assert checked.ll_path.read_bytes() == baseline.ll_path.read_bytes()
    assert len(gate.verdicts) == 1 and gate.verdicts[0]["verdict"] == "accepted"
    detail = gate.verdicts[0]["detail"]
    assert detail["source_model_status"] == "proposed" and detail["source_spec_approved"] is False
    assert detail["source_correspondence"] == {"status": "not_requested", "original_program_bound": False}
    assert detail["native_fidelity"] is False and detail["framework_equivalence"] is False
    assert detail["source_sites"] == [qdq.SOURCE_NAMES[("quantize", "tensor", "channel").index(kind)]]
    if kind == "quantize":
        assert detail["strengthenings"]["definedness"] == "abstracted"
    for path in detail["chains"].values():
        assert json.loads(Path(path).read_text())["refinement_verdict"] == "verified"


@pytest.mark.parametrize("textual", [False, True])
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


def test_textual_and_nested_xdsl_capture_one_complete_actual_boundary(tmp_path):
    jobs = []
    validation = SimpleNamespace(
        passes={qdq.SELECTOR},
        qdq_source=None,
        index_bits=64,
        tools=SimpleNamespace(sdk_bin="/unused", compiler_python="/unused"),
        gate=SimpleNamespace(_worker_path=[], _aliases={}, submit=lambda name, **job: jobs.append((name, job))),
    )
    raw = program("quantize")
    token = pass_validation._ACTIVE.set(validation)
    try:
        after, _ = qdq.observe_textual(raw, passes_xdsl.preprocess_text_textual)
    finally:
        pass_validation._ACTIVE.reset(token)
    assert len(jobs) == 1 and jobs[0][0] == qdq.CHECK
    assert jobs[0][1]["before"] == raw and jobs[0][1]["after"] == after


def test_qdq_stays_outside_all():
    assert qdq.SELECTOR in pass_validation.CHECKS and qdq.SELECTOR not in pass_validation.DEFAULT_CHECKS


def test_source_selection_requires_all_explicit_tools(monkeypatch, tmp_path):
    monkeypatch.setenv(qdq.ENV_CAPTURE, str(tmp_path))
    for name in ("MERLIN_QDQ_CAPTURE_PYTHON", "MERLIN_QDQ_PRODUCER", "MERLIN_QDQ_TORCH_WHEEL"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(ValueError, match="explicit"):
        qdq.SourceCapture.from_environment()


@pytest.mark.parametrize("bits", [True, 32])
def test_other_index_domains_refuse_before_tools(bits):
    with pytest.raises(GateUnsupported, match="index64"):
        qdq.check({"params": {"sdk_bin": "/unused", "compiler_python": "/unused", "index_bits": bits}})


def test_saved_matched_status_cannot_bind_another_actual_input(tmp_path, monkeypatch):
    from merlin.common import owned_process

    capture = tmp_path / "capture"
    capture.mkdir()
    for name in ("original.pt2", "qdq-frontend.json", "qdq-producer.json"):
        (capture / name).write_text("placeholder, never executed")
    (capture / "qdq-source-check.json").write_text('{"status":"matched"}')
    (capture / "qdq-source.mlir").write_text(program("quantize"))
    monkeypatch.setattr(owned_process, "run", lambda *a, **kw: pytest.fail("mismatched input started source replay"))
    selection = qdq.SourceCapture(
        capture, tmp_path / "unused-python", tmp_path / "producer", tmp_path / "wheel"
    ).describe()
    with pytest.raises(GateRejected, match="actual Q/DQ boundary input differs"):
        qdq._capture(selection, program("quantize").replace("quant_max = 127", "quant_max = 126"), tmp_path)


@pytest.mark.parametrize("detail", [None, {}, {"status": "matched", "source_spec_approved": True}])
def test_pending_or_approved_status_cannot_replace_a_qdq_result(tmp_path, monkeypatch, detail):
    from merlin.verify import gate_worker

    (tmp_path / "request.json").write_text(json.dumps({"check": qdq.CHECK}))
    monkeypatch.setattr(gate_worker, "resolve", lambda name: lambda request: detail)
    assert gate_worker.main(["worker", str(tmp_path)]) == 0
    assert json.loads((tmp_path / "verdict.json").read_text())["verdict"] == "not_established"
