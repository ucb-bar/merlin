"""Installed canonical original C64 owner qualification, never huge native loops.

This file intentionally has no scratch overlay. Promote only with the reviewed
source bundle and run sequentially on the coordinated CPU29 lane.
"""

import copy
import json
from dataclasses import replace
from pathlib import Path

import pytest
from configuration_elision_qualification_cases import QUALIFICATION
from configuration_hoist_native_consumer import compare
from test_compiler_configuration_hoist import actual
from test_compiler_configuration_hoist_multi import actual as multi_actual

from merlin.compile.scheduling.configuration_c_consumer import ConfigurationCConsumer
from merlin.compile.scheduling.configuration_loops import hoist_loop_configuration
from merlin.sched.isa import InstrDef, InstructionSet, Operand
from merlin.verify import compiler_configuration_hoist as gate
from merlin.verify import compiler_configuration_hoist_intake as intake
from merlin.verify import pass_validation
from merlin.verify.speculative_gate import GateFailure, SpeculativeGate

MAX64 = (1 << 63) - 1


def native_iset():
    operands = tuple(Operand(key, "int") for key in ("slot", "a", "b"))
    return InstructionSet(
        "configuration_fixture",
        {
            "pair": InstrDef("pair", operands, lambda args: "pair(" + ",".join(args) + ");"),
            "pulse": InstrDef("pulse", operands, lambda args: "pulse(" + ",".join(args) + ");"),
            "work": InstrDef("work", operands, lambda args: "if (work(" + ",".join(args) + ")) return;"),
        },
        {},
        "/* synchronous */",
    )


def checked_public(work, tools, *, extent):
    chosen, original, _ = actual(work, extent=extent)
    consumer = ConfigurationCConsumer(index_bits=64)
    worker = SpeculativeGate(work / "worker", job_timeout=300)
    validation = pass_validation.PassValidation(worker, tools, frozenset({gate.SELECTOR}))
    result = consumer.hoist(
        original,
        chosen.selected_contracts.configuration_adapter(QUALIFICATION),
        selected_contracts=chosen.selected_contracts,
        pass_validation=validation,
    )
    assert len(worker.verdicts) == 1 and worker.verdicts[0]["verdict"] == "accepted"
    detail = worker.verdicts[0]["detail"]
    assert detail["schema"] == "merlin.compiler-configuration-hoist.gate.v2"
    assert detail["c_consumer"] == consumer.record()
    report = json.loads(Path(detail["receipt"]).read_text())
    assert report["schema"] == "merlin.compiler-configuration-hoist.checked.v2"
    assert report["source_request"]["c_consumer"] == consumer.record()
    assert report["candidate"]["all_index_model_premise_retained"] is True
    assert report["hardware_model_approved"] is False
    return consumer, chosen, original, result, detail, report


def test_actual_original_c64_gate_then_controlled_small_native(tmp_path, pass_gate_tools):
    _, _, original, result, detail, report = checked_public(tmp_path, pass_gate_tools, extent=3)
    assert detail["domain"] == intake.DOMAIN
    text = result.emit_c(native_iset(), symbol="actual_kernel")
    (tmp_path / "controlled.c").write_text(text)
    assert "for (int64_t i" in text
    rows = compare(original.text(), result.kernel.text(), tmp_path / "native")
    # The actual compiled final source contains this exact controlled emission.
    # This finite independent state machine is not instruction/hardware approval.
    assert text in (tmp_path / "native/delivered/kernel.c").read_text()
    assert len(rows) == 15
    (tmp_path / "observations.json").write_text(json.dumps(rows, indent=2) + "\n")
    with pytest.raises(ValueError, match="no original live"):
        copy.copy(result).emit_c(native_iset(), symbol="actual_kernel")
    assert detail["native_fidelity"] is False


@pytest.mark.parametrize("extent", [1_000_001, MAX64])
def test_huge_count_is_canonical_symbolic_text_and_fresh_controlled_replay_only(tmp_path, pass_gate_tools, extent):
    consumer, _, original, result, detail, report = checked_public(tmp_path, pass_gate_tools, extent=extent)
    assert detail["domain"] == intake.WIDE_DOMAIN
    assert len(report["proof_sources"]) == 11
    text = result.emit_c(native_iset(), symbol="actual_kernel")
    assert str(extent) in text and "for (int64_t i" in text
    (tmp_path / "huge-controlled-text-only.c").write_text(text)
    if extent == 1_000_001:
        return
    replayed = consumer.replay(
        detail["receipt"],
        lean=pass_gate_tools.sdk_bin / "lean",
        destination=tmp_path / "fresh-replay",
        timeout=300,
    )
    assert replayed.kernel.text() == result.kernel.text()
    assert replayed.emit_c(native_iset(), symbol="actual_kernel") == text
    assert original.body[1].extent == MAX64
    # No compilation, loading, interpretation or execution of these huge loops.


def test_huge_changed_count_is_genuinely_refused_by_canonical_worker(tmp_path, pass_gate_tools):
    chosen, original, result = actual(tmp_path, extent=MAX64)
    consumer = ConfigurationCConsumer(index_bits=64)
    loop = result.kernel.body[2]
    final = replace(
        result.kernel,
        body=(*result.kernel.body[:2], replace(loop, extent=MAX64 - 1), *result.kernel.body[3:]),
    )
    worker = SpeculativeGate(tmp_path / "worker", job_timeout=300)
    params = gate.job_params(chosen, sdk_bin=pass_gate_tools.sdk_bin, c_consumer=consumer)
    validation = pass_validation.PassValidation(worker, pass_gate_tools, frozenset({gate.SELECTOR}))
    with pytest.raises(GateFailure):
        with pass_validation.validating(validation):
            worker.submit(gate.CHECK, before=original.text(), after=final.text(), params=params)
            worker.join()
    assert worker.verdicts[0]["verdict"] == "not_established"


def test_default_public_multi_profile_keeps_legacy_worker_schema(tmp_path, pass_gate_tools):
    chosen, original, baseline, adapter = multi_actual(tmp_path, "both")
    worker = SpeculativeGate(tmp_path / "worker", job_timeout=300)
    validation = pass_validation.PassValidation(worker, pass_gate_tools, frozenset({gate.SELECTOR}))
    result = hoist_loop_configuration(
        original, adapter, selected_contracts=chosen.selected_contracts, pass_validation=validation
    )
    assert result.kernel.text() == baseline.kernel.text()
    detail = worker.verdicts[0]["detail"]
    assert detail["schema"] == "merlin.compiler-configuration-hoist.gate.v1"
    assert detail["domain"] == intake.MULTI_DOMAIN and "c_consumer" not in detail
    report = json.loads(Path(detail["receipt"]).read_text())
    assert report["schema"] == "merlin.compiler-configuration-hoist.checked.v1"
    assert report["source_request"]["schema"] == "merlin.compiler-configuration.selection.v1"
