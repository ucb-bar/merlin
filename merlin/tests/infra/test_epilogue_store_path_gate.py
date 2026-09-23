"""The store-path gate is wired, phased, and the phase is read from the declaration.

Three things a gate can be wrong about independently: whether it RUNS, whether it BLOCKS, and
whether the phase it blocks at was declared or assumed. Each gets a mutation here. The comparison
itself is :mod:`merlin.tests.infra.test_epilogue_oracle`; what is under test below is the wiring,
which is where this repo's gates have actually failed -- silently, by never firing.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from merlin.perf import epilogue_oracle as EO
from merlin.perf import gate_phase
from merlin.targetgen import capsule_grade as CG
from merlin.targetgen import epilogue_store_path as SP

_TARGET = "toy_npu"  # in-tree, no hardware; the gate takes its target as a parameter


def _gap_report():
    admitted = [EO.Admission(index=0, op="matmul", stages=("bias_add", "acc_scale"), extents=None)]
    asked = EO.asks([{"opcode": "COMMIT", "attributes": {"epilogue": ["bias_add"]}}])
    return EO.compare(asked, admitted, licensed=("bias_add", "acc_scale"))


def _corpus(tmp_path: Path):
    """A model row with the two artifacts the gate reads, staged where the grade stages them."""
    capsule_dir = tmp_path / "capsules" / "M0"
    capsule_dir.mkdir(parents=True)
    (capsule_dir / "capsule.interface.mlir").write_text("builtin.module { }", encoding="utf-8")
    runs = tmp_path / "runs"
    (runs / "M0" / "generated").mkdir(parents=True)
    (runs / "M0" / "generated" / "command_buffer.json").write_text(
        json.dumps({"abi_version": "0.1", "target": _TARGET, "commands": []}), encoding="utf-8"
    )
    caps = [{"name": "M0", "kind": "model", "__dir__": str(capsule_dir)}]
    results = [{"capsule": "M0", "kind": "model", "status": "pass"}]
    return results, caps, runs


def test_the_gate_records_both_numbers_on_the_row_it_judged(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(SP, "assess", lambda *_a, **_k: _gap_report())
    results, caps, runs = _corpus(tmp_path)
    judged = SP.apply_gate(results, caps, runs, target=_TARGET, phase=gate_phase.PHASE_REPORT)
    record = results[0]["epilogue_store_path"]
    assert record["status"] == EO.VERDICT_GAP
    assert record["epilogue_share_on_store_path"] == 0.5
    assert record["n_silent_fallbacks"] == 1
    assert [row["capsule"] for row in judged] == ["M0"]


def test_at_report_a_decided_gap_is_recorded_and_blocks_nothing(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(SP, "assess", lambda *_a, **_k: _gap_report())
    results, caps, runs = _corpus(tmp_path)
    SP.apply_gate(results, caps, runs, target=_TARGET, phase=gate_phase.PHASE_REPORT)
    assert results[0]["status"] == "pass" and "failure" not in results[0]


def test_at_fail_the_same_gap_fails_the_row(monkeypatch, tmp_path) -> None:
    """The mutation that proves the gate CAN fire. A rule that cannot fire measures nothing."""
    monkeypatch.setattr(SP, "assess", lambda *_a, **_k: _gap_report())
    results, caps, runs = _corpus(tmp_path)
    SP.apply_gate(results, caps, runs, target=_TARGET, phase=gate_phase.PHASE_FAIL)
    assert results[0]["status"] == "fail"
    assert results[0]["failure"]["plane"] == SP.PLANE
    assert results[0]["failure"]["category"] == SP.CATEGORY


def test_an_undecided_verdict_never_blocks_at_either_phase(monkeypatch, tmp_path) -> None:
    """`incomplete` is a status, orthogonal to phase, and it is never a pass and never a block."""
    monkeypatch.setattr(SP, "assess", lambda *_a, **_k: EO.incomplete("nothing derived", (), (), {}))
    for phase in gate_phase.PHASES:
        results, caps, runs = _corpus(tmp_path / phase)
        SP.apply_gate(results, caps, runs, target=_TARGET, phase=phase)
        assert results[0]["status"] == "pass", phase
        assert results[0]["epilogue_store_path"]["status"] == EO.VERDICT_INCOMPLETE
        assert results[0]["epilogue_store_path"]["epilogue_share_on_store_path"] is None


def test_a_run_that_left_no_artifact_is_undecided_rather_than_clean(tmp_path) -> None:
    results, caps, runs = _corpus(tmp_path)
    (runs / "M0" / "generated" / "command_buffer.json").unlink()
    SP.apply_gate(results, caps, runs, target=_TARGET, phase=gate_phase.PHASE_FAIL)
    assert results[0]["epilogue_store_path"]["status"] == EO.VERDICT_INCOMPLETE
    assert "command buffer" in results[0]["epilogue_store_path"]["why"]
    assert results[0]["status"] == "pass"


def test_a_row_that_is_not_a_model_is_not_judged(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(SP, "assess", lambda *_a, **_k: _gap_report())
    results, caps, runs = _corpus(tmp_path)
    results[0]["kind"] = "op"
    assert SP.apply_gate(results, caps, runs, target=_TARGET, phase=gate_phase.PHASE_FAIL) == []
    assert "epilogue_store_path" not in results[0]


# --- the phase is declared, not assumed ---------------------------------------------------------
def test_the_gate_is_declared_and_its_phase_comes_from_the_declaration() -> None:
    assert SP.GATE in gate_phase.declared_gates()
    assert gate_phase.configured_phase(SP.GATE) in gate_phase.PHASES


def test_flipping_the_declaration_changes_what_blocks(monkeypatch, tmp_path) -> None:
    """The declaration is READ, not decorative: the same verdict blocks or not as the FILE says.

    Written as a real file under a stand-in ``merlin/`` rather than by patching the accessor, so a
    declaration that is parsed and then ignored fails here.
    """
    from merlin.common import paths

    monkeypatch.setattr(SP, "assess", lambda *_a, **_k: _gap_report())
    contract = tmp_path / "merlin" / "contract"
    contract.mkdir(parents=True)
    monkeypatch.setattr(paths, "merlin_dir", lambda: tmp_path / "merlin")
    for phase, expected in ((gate_phase.PHASE_REPORT, "pass"), (gate_phase.PHASE_FAIL, "fail")):
        (contract / "gate_phases.yaml").write_text(f"gates:\n  {SP.GATE}: {phase}\n", encoding="utf-8")
        gate_phase._declared.cache_clear()
        results, caps, runs = _corpus(tmp_path / phase)
        SP.apply_gate(results, caps, runs, target=_TARGET)  # phase resolved from the declaration
        assert results[0]["status"] == expected, phase
    gate_phase._declared.cache_clear()


# --- the aggregate, and the headline it reaches -------------------------------------------------
def test_a_corpus_that_decided_nothing_reports_no_share_rather_than_zero() -> None:
    rows = [
        {
            "capsule": "M0",
            "status": EO.VERDICT_INCOMPLETE,
            "epilogue_share_on_store_path": None,
            "n_silent_fallbacks": 0,
        }
    ]
    aggregate = SP.score_rows(rows)
    assert aggregate == {
        "n_model_rows": 1,
        "n_decided": 0,
        "n_incomplete": 1,
        "share": None,
        "silent_fallbacks": 0,
        "by_capsule": {"M0": EO.VERDICT_INCOMPLETE},
    }


@pytest.mark.parametrize(
    ("aggregate", "expected"),
    [
        (
            {"n_model_rows": 1, "n_incomplete": 1, "share": None, "silent_fallbacks": 0},
            "store-path readout not measured on 1 model capsule(s)",
        ),
        (
            {"n_model_rows": 1, "n_incomplete": 0, "share": 0.0, "silent_fallbacks": 54},
            "store-path readout 0% of what the target applies, 54 site(s) silently on the host",
        ),
        (
            {"n_model_rows": 1, "n_incomplete": 0, "share": 1.0, "silent_fallbacks": 0},
            "store-path readout 100% of what the target applies",
        ),
    ],
)
def test_the_number_reaches_the_string_people_quote(aggregate, expected) -> None:
    headline = CG._headline({"n_passed": 1, "n_capsules": 1, "epilogue_store_path": aggregate})
    assert expected in headline


def test_a_grade_with_no_model_rows_says_nothing_about_the_store_path() -> None:
    headline = CG._headline({"n_passed": 1, "n_capsules": 1, "epilogue_store_path": SP.score_rows([])})
    assert "store-path" not in headline
