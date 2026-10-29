"""Original explicit C64 consumer ownership, admission and legacy API controls.

Bounded host/C-text checks only; no theorem runner or loop execution.
"""

import copy
import json
import pickle
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError, replace

import pytest
from configuration_elision_qualification_cases import QUALIFICATION
from test_compiler_configuration_hoist import actual

from merlin.compile.scheduling.configuration_c_consumer import ConfigurationCConsumer, ConfigurationCResult
from merlin.sched.codegen import emit_c_function
from merlin.sched.ir import Const, Var, select
from merlin.sched.isa import InstrDef, InstructionSet, Operand
from merlin.verify import compiler_configuration_hoist as gate
from merlin.verify import compiler_configuration_hoist_intake as intake

MAX64 = (1 << 63) - 1


def iset():
    operands = tuple(Operand(key, "int") for key in ("slot", "a", "b"))
    return InstructionSet(
        "configuration_fixture",
        {
            name: InstrDef(name, operands, lambda args, name=name: name + "(" + ",".join(args) + ");")
            for name in ("pair", "pulse", "work")
        },
        {},
        "/* synchronous */",
    )


def issued(work, extent=3):
    chosen, original, _ = actual(work, extent=extent)
    owner = ConfigurationCConsumer(index_bits=64)
    result = owner.hoist(
        original,
        chosen.selected_contracts.configuration_adapter(QUALIFICATION),
        selected_contracts=chosen.selected_contracts,
    )
    return owner, chosen, original, result


@pytest.mark.parametrize("width", [None, 32, True, 64.0, "64", 128])
def test_only_explicit_exact_index64_is_selectable(width):
    with pytest.raises(ValueError, match="explicitly selected"):
        ConfigurationCConsumer(index_bits=width)


def test_live_actual_producer_result_uses_captured_builtin_width(tmp_path):
    owner, chosen, original, result = issued(tmp_path)
    assert result.index_bits == 64
    assert result.kernel.text() != original.text()
    assert result.emit_c(iset(), symbol="kernel") == emit_c_function(
        result.kernel, iset(), symbol="kernel", index_bits=64
    )
    assert "for (int64_t i" in result.emit_c(iset(), symbol="kernel")
    with pytest.raises(TypeError):
        result.emit_c(iset(), symbol="kernel", index_bits=32)
    with pytest.raises(FrozenInstanceError):
        owner.index_bits = 32
    # A diagnostic report supplies no authority; editing it changes no input,
    # output, selected contract, width or emitted program.
    result.report["status"] = "invented"
    assert result.index_bits == 64


@pytest.mark.parametrize("kind", ["copy", "deepcopy", "constructed", "replace"])
def test_equal_or_copied_carriers_have_no_original_owner(tmp_path, kind):
    _, _, _, result = issued(tmp_path)
    foreign = {
        "copy": lambda: copy.copy(result),
        "deepcopy": lambda: copy.deepcopy(result),
        "constructed": lambda: ConfigurationCResult(result.kernel, result.report),
        "replace": lambda: replace(result),
    }[kind]()
    with pytest.raises(ValueError, match="no original live"):
        foreign.emit_c(iset(), symbol="kernel")


def test_serialization_and_foreign_thread_are_not_original_operations(tmp_path):
    _, _, _, result = issued(tmp_path)
    with pytest.raises(TypeError, match="cannot be serialized"):
        pickle.dumps(result)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(result.emit_c, iset(), symbol="kernel")
        with pytest.raises(ValueError, match="host/thread"):
            future.result()


def test_frozen_source_and_output_are_checked_before_emission(tmp_path):
    _, _, original, result = issued(tmp_path)
    object.__setattr__(original, "attrs", (("changed", "source"),))
    with pytest.raises(ValueError, match="changed before emission"):
        result.emit_c(iset(), symbol="kernel")


def test_mutated_actual_output_is_not_deliverable(tmp_path):
    _, _, _, result = issued(tmp_path)
    object.__setattr__(result.kernel, "attrs", (("changed", "output"),))
    with pytest.raises(ValueError, match="changed before emission"):
        result.emit_c(iset(), symbol="kernel")


def test_selected_facts_and_contracts_stay_bound_through_emission(tmp_path):
    _, chosen, _, result = issued(tmp_path)
    chosen.selected_contracts.software_path.write_text("{}")
    with pytest.raises(ValueError, match="specification changed"):
        result.emit_c(iset(), symbol="kernel")


def test_emitter_function_rebinding_is_not_original_code(tmp_path, monkeypatch):
    from merlin.sched import codegen

    _, _, _, result = issued(tmp_path)
    original = codegen.emit_c_function
    monkeypatch.setattr(codegen, "emit_c_function", lambda *a, **k: original(*a, **k))
    with pytest.raises(ValueError, match="sources/code changed"):
        result.emit_c(iset(), symbol="kernel")


@pytest.mark.parametrize("extent", [1_000_001, (1 << 31), MAX64])
def test_wide_extent_is_only_symbolic_producer_and_c_text(tmp_path, extent):
    owner, chosen, original, result = issued(tmp_path, extent)
    text = result.emit_c(iset(), symbol="kernel")
    assert str(extent) in text and "int64_t" in text
    with pytest.raises(ValueError, match="bounded indexed loop"):
        intake.prepare(original.text(), result.kernel.text(), selection=chosen)
    candidate = intake.prepare(original.text(), result.kernel.text(), selection=chosen, c_consumer=owner)
    intake.revalidate(candidate)
    assert candidate.domain == intake.WIDE_DOMAIN
    assert len(json.loads(candidate.proof_sources_json)) == 11
    assert candidate.describe()["schema"].endswith("candidate.v2")
    assert candidate.describe()["all_index_model_premise_retained"] is True
    assert "Focus.check" in candidate.query


@pytest.mark.parametrize("extent", [(1 << 63), True, 3.0])
def test_actual_domain_and_exact_count_admission_refuse_before_emission(tmp_path, extent):
    chosen, original, _ = actual(tmp_path)
    original = replace(original, body=(original.body[0], replace(original.body[1], extent=extent), *original.body[2:]))
    with pytest.raises(ValueError):
        ConfigurationCConsumer(index_bits=64).admit(original)


@pytest.mark.parametrize(
    "expression",
    [Var("i") * (1 << 64), select("i", 1 << 64, Const(1), Var("i")), select("i", 99, Const(1 << 64), Var("i"))],
)
def test_dead_arm_factors_and_comparison_literals_follow_actual_renderer(tmp_path, expression):
    chosen, original, _ = actual(tmp_path, extent=1)
    loop = original.body[1]
    work = replace(loop.body[1], args=tuple((k, expression if k == "a" else v) for k, v in loop.body[1].args))
    original = replace(original, body=(original.body[0], replace(loop, body=(loop.body[0], work)), *original.body[2:]))
    with pytest.raises(ValueError, match="literal exceeds"):
        ConfigurationCConsumer(index_bits=64).admit(original)
    with pytest.raises(ValueError, match="literal exceeds"):
        emit_c_function(original, iset(), symbol="kernel", index_bits=64)


def test_old_and_versioned_request_job_shapes_are_distinct(tmp_path):
    owner, chosen, original, result = issued(tmp_path)
    old = gate.job_params(chosen, sdk_bin="never-executed")
    assert set(old) == {"schema", "sdk_bin", "selection"} and old["schema"].endswith("job.v1")
    new = gate.job_params(chosen, sdk_bin="never-executed", c_consumer=owner)
    assert set(new) == {"schema", "sdk_bin", "selection", "c_consumer"} and new["schema"].endswith("job.v2")
    legacy = intake.prepare(original.text(), result.kernel.text(), selection=chosen)
    bound = intake.prepare(original.text(), result.kernel.text(), selection=chosen, c_consumer=owner)
    assert legacy.domain == bound.domain == intake.DOMAIN
    assert legacy.describe()["schema"].endswith("candidate.v1")
    assert "c_consumer" not in legacy.describe()
    assert set(json.loads(bound.request_json)) == {"schema", "selection", "c_consumer"}
    assert intake.reopen_request(json.loads(legacy.request_json))[1] is None
    assert intake.reopen_request(json.loads(bound.request_json))[1].record() == owner.record()


@pytest.mark.parametrize("mutation", ["width", "extra", "source", "schema"])
def test_versioned_consumer_request_cannot_change_domain_or_sources(tmp_path, mutation):
    owner, chosen, original, result = issued(tmp_path)
    record = owner.record()
    if mutation == "width":
        record["index_bits"] = 32
    elif mutation == "extra":
        record["approval"] = True
    elif mutation == "source":
        record["sources"][0]["sha256"] = "0" * 64
    else:
        record["schema"] = "invented"
    with pytest.raises(ValueError):
        ConfigurationCConsumer.reopen(record)


def test_source_and_final_domains_are_both_checked(tmp_path):
    owner, chosen, original, result = issued(tmp_path)
    loop = result.kernel.body[2]
    final = replace(
        result.kernel, body=(*result.kernel.body[:2], replace(loop, extent=1 << 63), *result.kernel.body[3:])
    )
    with pytest.raises(ValueError, match="signed 64-bit"):
        intake.prepare(original.text(), final.text(), selection=chosen, c_consumer=owner)


def test_consumer_does_not_infer_contract_or_accept_substitution(tmp_path):
    owner, chosen, original, result = issued(tmp_path)
    adapter = chosen.selected_contracts.configuration_adapter(QUALIFICATION)
    with pytest.raises(ValueError, match="typed selected ISA"):
        owner.hoist(original, adapter)
    with pytest.raises(ValueError, match="cannot be substituted"):
        owner.hoist(original, adapter, c_consumer=ConfigurationCConsumer(index_bits=64))


def test_uncontrolled_builtin_default_emission_is_unchanged(tmp_path):
    _, _, _, result = issued(tmp_path)
    assert "for (int i" in emit_c_function(result.kernel, iset(), symbol="kernel")
    assert "for (int32_t i" in emit_c_function(result.kernel, iset(), symbol="kernel", index_bits=32)


def test_consumer_guard_also_runs_after_actual_target_rendering(tmp_path):
    _, _, original, result = issued(tmp_path)
    target = iset()
    definition = target.instrs["work"]

    # The target callback is an external implementation premise, but it cannot
    # mutate the actual frozen inputs and still return selected C output.
    def mutate(args):
        object.__setattr__(original, "attrs", (("during", "render"),))
        return "work(" + ",".join(args) + ");"

    target = replace(target, instrs={**target.instrs, "work": replace(definition, render_c=mutate)})
    with pytest.raises(ValueError, match="changed before emission"):
        result.emit_c(target, symbol="kernel")


@pytest.mark.parametrize(
    "schema", ["merlin.compiler-configuration-hoist.job.v1", "merlin.compiler-configuration-hoist.job.v2"]
)
def test_forged_transport_fields_are_rejected_before_any_proof(tmp_path, monkeypatch, schema):
    owner, chosen, original, result = issued(tmp_path)
    params = gate.job_params(chosen, sdk_bin="never-executed", c_consumer=owner)
    params["schema"] = schema
    if schema.endswith("v2"):
        params["c_consumer"]["index_bits"] = 32
    monkeypatch.setattr(gate, "check_candidate", lambda *a, **k: pytest.fail("malformed job reached proof runner"))
    from merlin.verify.gate_errors import GateUnsupported

    with pytest.raises(GateUnsupported):
        gate.check({"params": params})


@pytest.mark.parametrize("schema", ["invented", "merlin.compiler-configuration-hoist.checked.v1"])
def test_legacy_or_invented_report_cannot_issue_controlled_replay(tmp_path, monkeypatch, schema):
    owner, _, _, _ = issued(tmp_path)
    report = tmp_path / "saved.json"
    report.write_text(json.dumps({"schema": schema, "buffers": []}))
    monkeypatch.setattr(gate, "check_candidate", lambda *a, **k: pytest.fail("malformed replay reached proof runner"))
    with pytest.raises(gate.Unavailable, match="descriptive replay record"):
        owner.replay(report, lean="never-executed", destination=tmp_path / "proof")


def test_wide_profile_is_not_just_a_caller_domain_label(tmp_path, monkeypatch):
    owner, chosen, original, result = issued(tmp_path, 1_000_001)
    candidate = intake.prepare(original.text(), result.kernel.text(), selection=chosen, c_consumer=owner)
    forged = replace(candidate, request_json=json.dumps(intake.request(chosen)))
    monkeypatch.setattr(gate.runner, "prove", lambda *a, **k: pytest.fail("forged profile reached proof runner"))
    with pytest.raises((ValueError, gate.Unavailable)):
        gate.check_candidate(forged, lean="never-executed", destination=tmp_path / "proof")


def test_actual_multiple_wide_scopes_keep_complete_focus_join(tmp_path):
    owner, chosen, original, result = issued(tmp_path, MAX64)
    second = replace(original.body[1], var="j", extent=1_000_001)
    original = replace(original, body=(*original.body[:2], second, *original.body[2:]))
    adapter = chosen.selected_contracts.configuration_adapter(QUALIFICATION)
    result = owner.hoist(original, adapter, selected_contracts=chosen.selected_contracts)
    candidate = intake.prepare(original.text(), result.kernel.text(), selection=chosen, c_consumer=owner)
    projection = json.loads(candidate.projection_json)
    assert candidate.domain == intake.WIDE_DOMAIN
    assert projection["actual_loop_count"] == projection["focused_loop_count"] == 2
    assert projection["actual_final_join"] is True
    assert projection["before"]["atoms"] == projection["after"]["atoms"]


def test_actual_emitter_render_alias_is_bound_too(tmp_path, monkeypatch):
    from merlin.sched import codegen

    _, _, _, result = issued(tmp_path)
    original = codegen.render
    monkeypatch.setattr(codegen, "render", lambda *a, **k: original(*a, **k))
    with pytest.raises(ValueError, match="sources/code changed"):
        result.emit_c(iset(), symbol="kernel")


@pytest.mark.parametrize("profile", ["static", "indexed", "prefix", "multi"])
def test_legacy_profiles_and_math_queries_remain_unchanged_under_optional_consumer(tmp_path, profile):
    if profile == "static":
        chosen, original, result = actual(tmp_path)
        expected, count = intake.DOMAIN, 9
    elif profile == "indexed":
        from test_compiler_configuration_hoist_indexed import actual as indexed_actual

        chosen, original, result = indexed_actual(tmp_path, Var("i") * 4 + 1)
        expected, count = intake.INDEXED_DOMAIN, 9
    elif profile == "prefix":
        from test_compiler_configuration_hoist_prefix import actual as prefix_actual

        chosen, original, result, *_ = prefix_actual(tmp_path, "overwritten-final")
        expected, count = intake.PREFIX_DOMAIN, 10
    else:
        from test_compiler_configuration_hoist_multi import actual as multi_actual

        chosen, original, result, _ = multi_actual(tmp_path, "both")
        expected, count = intake.MULTI_DOMAIN, 11
    legacy = intake.prepare(original.text(), result.kernel.text(), selection=chosen)
    bound = intake.prepare(
        original.text(), result.kernel.text(), selection=chosen, c_consumer=ConfigurationCConsumer(index_bits=64)
    )
    assert legacy.domain == bound.domain == expected
    assert len(json.loads(legacy.proof_sources_json)) == len(json.loads(bound.proof_sources_json)) == count
    assert legacy.query == bound.query and legacy.projection_json == bound.projection_json
    assert legacy.roots == bound.roots
    assert json.loads(legacy.request_json)["schema"] == "merlin.compiler-configuration.selection.v1"
    assert legacy.describe()["schema"] == "merlin.compiler-configuration-hoist.candidate.v1"
    intake.revalidate(legacy)
    intake.revalidate(bound)
