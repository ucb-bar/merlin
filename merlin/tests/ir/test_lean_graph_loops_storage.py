"""Actual loop/effect binding, complete storage admission and valid hazard refusals."""

from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path

import pytest

from merlin.sched.ir import Const, Var, concretize, instances, parse_kernel, select
from merlin.verify.lean_graph_loops import LeanSelection
from merlin.verify.lean_graph_loops_storage import ProducerSelection, storage_obligation, verify_storage_files


def flattened(kernel):
    return replace(
        kernel,
        body=tuple(
            replace(call, args=tuple((name, Const(value)) for name, value in concretize(call, environment).items()))
            for call, environment in instances(kernel)
        ),
    )


@pytest.fixture(scope="module")
def contexts():
    selected = os.environ.get("MERLIN_GRAPH_LOOP_STORAGE_SELECTION")
    producer = os.environ.get("MERLIN_GRAPH_LOOP_PRODUCER_SELECTION")
    capture = os.environ.get("MERLIN_GRAPH_LOOP_CAPTURE")
    if not all((selected, producer, capture)):
        pytest.skip("explicit kernel/producer/capture selection unavailable; no storage result inferred")
    lean = json.loads(Path(selected).read_bytes())
    for key in ("executable", "library", "build_record"):
        lean[key] = Path(lean[key])
    reference = json.loads(Path(producer).read_bytes())
    reference["source"] = Path(reference["source"])
    for key in ("source_files", "initialized_external"):
        reference[key] = tuple(reference[key])
    return LeanSelection(**lean), ProducerSelection(**reference), Path(capture)


def check(tmp_path, contexts, before, *, name="proof", after=None, producer=None, effects_path=None, bits=64):
    selection, original_producer, _ = contexts
    source, result = tmp_path / (name + ".before.kernel"), tmp_path / (name + ".after.kernel")
    source.write_text(before.text())
    result.write_text((flattened(before) if after is None else after).text())
    return verify_storage_files(
        source,
        result,
        selection=selection,
        producer=producer or original_producer,
        output=tmp_path / name,
        index_bits=bits,
        max_dynamic_calls=256,
        effects_path=effects_path,
    )


def argument(call, name, value):
    assert name in dict(call.args)
    return replace(call, args=tuple((key, value if key == name else old) for key, old in call.args))


def fixture_kernel(contexts):
    return parse_kernel((contexts[2] / "before.kernel").read_bytes().decode())


def test_actual_storage_and_reused_drained_tokens_are_kernel_checked(contexts, tmp_path):
    kernel = fixture_kernel(contexts)
    result = check(tmp_path, contexts, kernel, effects_path=contexts[2] / "effects.json")
    assert result["lean_storage_checked"] and result["complete_boundary_checked"]
    assert len(result["transitive_axioms"]) == 21
    assert result["source_admission"]["report"]["iterations_checked"] == 3
    assert result["initialized_external_premise"] == ["input"]
    assert not any(
        result[key]
        for key in (
            "pipeline_receipt_issued",
            "numerical_gates_discharged",
            "adapter_independently_qualified",
            "external_output_coverage_proved",
            "renderer_or_hardware_proved",
        )
    )


@pytest.mark.parametrize("mutation", ["missing_wait", "missing_final_drain", "wrong_wait"])
def test_actual_completion_mutations_refuse(contexts, tmp_path, mutation):
    kernel = fixture_kernel(contexts)
    loop = kernel.body[0]
    body = list(loop.body)
    if mutation == "missing_wait":
        del body[2]
    elif mutation == "missing_final_drain":
        body.pop()
    else:
        body[2] = argument(body[2], "token", Const(1))
    before = replace(kernel, body=(replace(loop, body=tuple(body)),))
    result = check(tmp_path, contexts, before)
    assert not result["source_admission"]["accepted"]
    assert result["returncode"] != 0 and not result["lean_storage_checked"]
    assert not result["complete_boundary_checked"]


def test_late_overwrite_after_good_prefix_is_not_a_prefix_certificate(contexts, tmp_path):
    kernel = fixture_kernel(contexts)
    loop = kernel.body[0]
    body = list(loop.body)
    body[1] = argument(body[1], "slot", select(Var(loop.var), 6, 0, 1))
    prefix = replace(kernel, body=(replace(loop, extent=6, body=tuple(body)),))
    full = replace(kernel, body=(replace(loop, extent=7, body=tuple(body)),))
    producer = contexts[1]
    good = check(tmp_path, contexts, prefix, name="good_prefix", producer=replace(producer, iterations=6))
    assert good["lean_storage_checked"] and good["complete_boundary_checked"]
    assert flattened(prefix).body == flattened(full).body[:60]
    bad = check(tmp_path, contexts, full, name="late_overwrite", producer=replace(producer, iterations=7))
    assert bad["returncode"] != 0 and not bad["lean_storage_checked"]
    assert "conflicts" in bad["source_admission"]["error"]


def test_last_tail_external_address_is_bounded_for_the_complete_domain(contexts, tmp_path):
    kernel = fixture_kernel(contexts)
    loop = kernel.body[0]
    body = list(loop.body)
    index = Var(loop.var)
    body[7] = argument(body[7], "host", select(index, 6, 14, index * 2 + 1))
    before = replace(kernel, body=(replace(loop, extent=7, body=tuple(body)),))
    result = check(tmp_path, contexts, before, producer=replace(contexts[1], iterations=7), bits=32)
    assert result["returncode"] != 0 and not result["lean_storage_checked"]
    assert "window" in result["source_admission"]["error"]


def test_saved_effect_annotation_cannot_replace_fresh_actual_projection(contexts, tmp_path):
    effects = json.loads((contexts[2] / "effects.json").read_bytes())
    effects["blocks"][-1][0]["accesses"][1]["start"] = 1
    saved = tmp_path / "stale-effects.json"
    saved.write_text(json.dumps(effects))
    with pytest.raises(ValueError, match="saved effects/declarations differ"):
        check(tmp_path, contexts, fixture_kernel(contexts), effects_path=saved)


def test_changed_source_adapter_identity_and_selected_source_refuse(contexts, tmp_path):
    selected = contexts[1]
    identities = dict(selected.adapter_identities, qualification="different-qualification")
    with pytest.raises(ValueError, match="fresh adapter/source identity"):
        check(tmp_path, contexts, fixture_kernel(contexts), producer=replace(selected, adapter_identities=identities))
    sources = [dict(row) for row in selected.source_files]
    sources[0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="selected producer source changed"):
        replace(selected, source_files=tuple(sources)).dependencies()


def test_nonmaterialized_actual_trace_cannot_reuse_effects(contexts, tmp_path):
    kernel = fixture_kernel(contexts)
    actual = flattened(kernel)
    last = argument(actual.body[-1], "token", Const(999))
    with pytest.raises(ValueError, match="actual selected adapter projection refused"):
        check(tmp_path, contexts, kernel, after=replace(actual, body=actual.body[:-1] + (last,)))


def test_private_initialization_cannot_be_assumed_from_external_input(contexts, tmp_path):
    selected = replace(contexts[1], initialized_external=("local",))
    with pytest.raises(ValueError, match="initialized external windows must be distinct"):
        check(tmp_path, contexts, fixture_kernel(contexts), producer=selected)


def test_storage_projection_rejects_partial_and_unknown_effect_domains():
    with pytest.raises(ValueError, match="private ownership"):
        storage_obligation({}, {"private_ownership": False}, initialized_external=("input",))


def test_changed_actual_effect_file_during_kernel_check_refuses(contexts, tmp_path, monkeypatch):
    from merlin.verify import lean_graph_loops_storage as implementation

    saved = tmp_path / "effects.json"
    saved.write_bytes((contexts[2] / "effects.json").read_bytes())
    original = implementation.subprocess.run

    def write_after_kernel(command, *args, **kwargs):
        result = original(command, *args, **kwargs)
        if str(command[0]) == str(contexts[0].executable):
            saved.write_bytes(saved.read_bytes() + b" ")
        return result

    monkeypatch.setattr(implementation.subprocess, "run", write_after_kernel)
    result = check(tmp_path, contexts, fixture_kernel(contexts), effects_path=saved)
    assert result["returncode"] == 0 and not result["lean_storage_checked"]
    assert "changed during checking" in result["refusal"]


def test_independent_seven_period_actual_capture_checks_signed32_bounds(contexts, tmp_path):
    capture = os.environ.get("MERLIN_GRAPH_LOOP_SEVEN_CAPTURE")
    if capture is None:
        pytest.skip("independently captured seven-period producer unavailable")
    capture = Path(capture)
    result = verify_storage_files(
        capture / "before.kernel",
        capture / "after.kernel",
        selection=contexts[0],
        producer=replace(contexts[1], iterations=7),
        output=tmp_path / "actual_seven",
        index_bits=32,
        max_dynamic_calls=70,
        effects_path=capture / "effects.json",
    )
    assert result["lean_storage_checked"] and result["complete_boundary_checked"]
    assert result["source_admission"]["report"]["dynamic_calls_checked"] == 70
    assert "GraphLoopStorageObligation.index_bounds" in result["transitive_axioms"]


@pytest.mark.parametrize("product", ["Obligation.lean", "fresh-effects.json"])
def test_intervening_query_or_projection_writer_invalidates_kernel_success(contexts, tmp_path, monkeypatch, product):
    from merlin.verify import lean_graph_loops_storage as implementation

    original = implementation.subprocess.run

    def write_after_kernel(command, *args, **kwargs):
        result = original(command, *args, **kwargs)
        if str(command[0]) == str(contexts[0].executable):
            path = tmp_path / "proof" / product
            path.write_bytes(path.read_bytes() + b" ")
        return result

    monkeypatch.setattr(implementation.subprocess, "run", write_after_kernel)
    result = check(tmp_path, contexts, fixture_kernel(contexts))
    assert result["returncode"] == 0 and not result["lean_storage_checked"]
    assert "changed during checking" in result["refusal"]


def test_selected_source_and_adapter_maps_cannot_change_by_alias(contexts):
    selected = contexts[1]
    source_rows = [dict(row) for row in selected.source_files]
    identities = dict(selected.adapter_identities)
    independent = replace(selected, source_files=tuple(source_rows), adapter_identities=identities)
    source_rows[0]["sha256"] = "0" * 64
    identities["qualification"] = "changed-after-selection"
    assert independent.source_files[0]["sha256"] == selected.source_files[0]["sha256"]
    assert independent.adapter_identities == selected.adapter_identities
    with pytest.raises(TypeError):
        independent.adapter_identities["qualification"] = "changed"
    with pytest.raises(TypeError):
        independent.source_files[0]["sha256"] = "0" * 64
    assert independent.dependencies() == selected.dependencies()
