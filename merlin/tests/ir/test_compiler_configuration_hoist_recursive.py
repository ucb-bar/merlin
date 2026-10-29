"""Canonical recursive Kernel proof/owner integration; no instruction-model approval."""

import copy
import json
from dataclasses import replace

import pytest
from configuration_hoist_multi_native_consumer import compare
from nested_configuration_hoist_inputs import actual
from test_compiler_configuration_hoist import call
from test_compiler_configuration_hoist_wide import native_iset

from merlin.compile.scheduling.configuration_c_consumer import ConfigurationCConsumer
from merlin.compile.scheduling.configuration_loops import hoist_loop_configuration
from merlin.verify import compiler_configuration_hoist as gate
from merlin.verify import compiler_configuration_hoist_intake as intake
from merlin.verify import pass_validation
from merlin.verify.configuration_hoist_bridge import Unsupported
from merlin.verify.speculative_gate import GateFailure, SpeculativeGate


@pytest.mark.parametrize("case,moved", [("inner", 1), ("outer-and-inner", 2), ("indexed", 1), ("overlap", 3)])
def test_recursive_actual_full_endpoints_and_all_environment_model(tmp_path, case, moved):
    chosen, source, result, _ = actual(tmp_path, case)
    candidate = intake.prepare(source.text(), result.kernel.text(), selection=chosen)
    intake.revalidate(candidate)
    assert candidate.domain == intake.NESTED_DOMAIN
    description = candidate.describe()
    assert description["all_environment_model_premise_retained"] is True
    assert description["max_loop_depth"] == 2 and description["actual_loop_count"] == 2
    assert description["moved_setter_count"] == moved
    projection = json.loads(candidate.projection_json)
    assert projection["actual_final_join"] is True
    assert projection["before"]["atoms"] == projection["after"]["atoms"]
    assert "otherLoop" not in candidate.query
    assert "PartialHoist.Recursive.Models fields literal before.atoms run" in candidate.query
    assert len(json.loads(candidate.proof_sources_json)) == 11
    assert candidate.roots[-1] == "ActualConfigurationHoist.complete_kernel"


@pytest.mark.parametrize(
    "domain", [intake.DOMAIN, intake.INDEXED_DOMAIN, intake.PREFIX_DOMAIN, intake.MULTI_DOMAIN, intake.WIDE_DOMAIN]
)
def test_recursive_domain_cannot_be_relabelled_as_old_profile(tmp_path, domain):
    chosen, source, result, _ = actual(tmp_path)
    candidate = intake.prepare(source.text(), result.kernel.text(), selection=chosen)
    with pytest.raises((ValueError, gate.Unavailable)):
        gate.ready(replace(candidate, domain=domain))


def test_nested_wide_count_requires_original_C64_consumer(tmp_path):
    chosen, source, _, adapter = actual(tmp_path)
    source = replace(source, body=(replace(source.body[0], extent=(1 << 63) - 1),))
    consumer = ConfigurationCConsumer(index_bits=64)
    consumer.admit(source)
    result = hoist_loop_configuration(source, adapter)
    with pytest.raises(Unsupported):
        intake.prepare(source.text(), result.kernel.text(), selection=chosen)
    candidate = intake.prepare(source.text(), result.kernel.text(), selection=chosen, c_consumer=consumer)
    assert candidate.domain == intake.NESTED_DOMAIN
    assert json.loads(candidate.request_json)["c_consumer"] == consumer.record()
    assert str((1 << 63) - 1) in candidate.query
    # Symbolic/text only: this test launches no SDK, C compiler or enormous loop.


@pytest.mark.parametrize(
    "case,explicit,consumer_bound",
    [("inner", False, True), ("outer-and-inner", True, False), ("indexed", False, False)],
)
def test_public_recursive_producer_worker_live_delivery_and_fresh_replay(
    tmp_path, pass_gate_tools, case, explicit, consumer_bound
):
    chosen, source, expected, adapter = actual(tmp_path, case)
    consumer = ConfigurationCConsumer(index_bits=64) if consumer_bound else None
    producer = hoist_loop_configuration if consumer is None else consumer.hoist
    worker = SpeculativeGate(tmp_path / "worker", job_timeout=300)
    validation = pass_validation.PassValidation(worker, pass_gate_tools, frozenset({gate.SELECTOR}))
    if explicit:
        result = producer(source, adapter, selected_contracts=chosen.selected_contracts, pass_validation=validation)
    else:
        with pass_validation.validating(validation):
            result = producer(source, adapter, selected_contracts=chosen.selected_contracts)
    assert result.kernel.text() == expected.kernel.text()
    assert len(worker.verdicts) == 1 and worker.verdicts[0]["verdict"] == "accepted"
    detail = worker.verdicts[0]["detail"]
    assert detail["domain"] == intake.NESTED_DOMAIN and detail["native_fidelity"] is False
    assert detail["hardware_model_approved"] is False
    assert detail["compiler_enforcement"] is True and detail["independently_qualified"] is False
    assert detail["schema"] == (
        "merlin.compiler-configuration-hoist.gate.v2"
        if consumer_bound
        else "merlin.compiler-configuration-hoist.gate.v1"
    )
    if consumer is not None:
        assert detail["c_consumer"] == consumer.record()
        emitted = result.emit_c(native_iset(), symbol="actual_kernel")
        rows = compare(source.text(), result.kernel.text(), tmp_path / "native")
        assert len(rows) == 27
        assert emitted in (tmp_path / "native/delivered/kernel.c").read_text()
        (tmp_path / "finite-native-observations.json").write_text(json.dumps(rows, indent=2) + "\n")
        fresh = consumer.replay(
            detail["receipt"], lean=pass_gate_tools.sdk_bin / "lean", destination=tmp_path / "fresh-replay", timeout=300
        )
        assert fresh.kernel.text() == result.kernel.text()
        assert fresh.emit_c(native_iset(), symbol="actual_kernel") == emitted
        with pytest.raises(ValueError, match="no original live"):
            copy.copy(fresh).emit_c(native_iset(), symbol="actual_kernel")


@pytest.mark.parametrize("mutation", ["operand", "frame"])
def test_actual_recursive_target_mutants_are_genuine_closed_check_refusals(tmp_path, pass_gate_tools, mutation):
    chosen, source, result, _ = actual(tmp_path)
    if mutation == "frame":
        final = replace(result.kernel, attrs=(("owner", "changed"),))
    else:
        outer = result.kernel.body[0]
        inner = outer.body[1]
        work = replace(inner.body[0], args=call("work", a=1).args)
        final = replace(
            result.kernel,
            body=(replace(outer, body=(outer.body[0], replace(inner, body=(work, *inner.body[1:])), *outer.body[2:])),),
        )
    candidate = intake.prepare(source.text(), final.text(), selection=chosen)
    assert candidate.domain == intake.NESTED_DOMAIN
    intake.revalidate(candidate)
    worker = SpeculativeGate(tmp_path / "worker", job_timeout=300)
    validation = pass_validation.PassValidation(worker, pass_gate_tools, frozenset({gate.SELECTOR}))
    with pytest.raises(GateFailure):
        with pass_validation.validating(validation):
            worker.submit(
                gate.CHECK,
                before=source.text(),
                after=final.text(),
                params=gate.job_params(chosen, sdk_bin=pass_gate_tools.sdk_bin),
            )
            worker.join()
    assert len(worker.verdicts) == 1 and worker.verdicts[0]["verdict"] == "not_established"
