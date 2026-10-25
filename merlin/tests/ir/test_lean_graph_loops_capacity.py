"""Separately selected capacity mode over the unchanged delivered v1 boundary."""

from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path

import pytest

from merlin.sched.ir import Const, Var, concretize, instances, parse_kernel, select
from merlin.verify.lean_graph_loops import LeanSelection
from merlin.verify.lean_graph_loops_capacity import (
    CAPACITY_ROOTS,
    STORAGE_ROOTS,
    CapacitySelection,
    LegacySeal,
    admission_digest,
    verify_capacity_files,
)
from merlin.verify.lean_graph_loops_storage import ProducerSelection


@pytest.fixture(scope="module")
def context():
    selected = os.environ.get("MERLIN_GRAPH_LOOP_CAPACITY_SELECTION")
    captured = os.environ.get("MERLIN_GRAPH_LOOP_CAPTURE")
    if not selected or not captured:
        pytest.skip("explicit capacity and actual producer context unavailable")
    data = json.loads(Path(selected).read_bytes())
    data["legacy"]["manifest"] = Path(data["legacy"]["manifest"])
    data["legacy"] = LegacySeal(**data["legacy"])
    for name in ("base_proof", "proof"):
        for key in ("executable", "library", "build_record"):
            data[name][key] = Path(data[name][key])
        data[name] = LeanSelection(**data[name])
    data["producer"]["source"] = Path(data["producer"]["source"])
    data["producer"] = ProducerSelection(**data["producer"])
    return CapacitySelection(**data), Path(captured)


def flattened(kernel):
    return replace(
        kernel,
        body=tuple(
            replace(call, args=tuple((key, Const(value)) for key, value in concretize(call, env).items()))
            for call, env in instances(kernel)
        ),
    )


def argument(call, name, value):
    return replace(call, args=tuple((key, value if key == name else old) for key, old in call.args))


def check(tmp_path, context, *, kernel=None, selection=None, effects_path=None):
    selected, capture = context
    source, target = capture / "before.kernel", capture / "after.kernel"
    if kernel is not None:
        source, target = tmp_path / "before.kernel", tmp_path / "after.kernel"
        source.write_text(kernel.text())
        target.write_text(flattened(kernel).text())
    return verify_capacity_files(
        source,
        target,
        selection=selection or selected,
        output=tmp_path / "proof",
        index_bits=64,
        max_dynamic_calls=256,
        effects_path=effects_path,
    )


def test_actual_complete_capacity_and_legacy_identity(context, tmp_path):
    selected, capture = context
    before = selected.legacy.dependencies()
    result = check(tmp_path, context, effects_path=capture / "effects.json")
    assert result["lean_capacity_checked"] and result["complete_boundary_checked"]
    assert result["selected_admission"]["max_pending"] == 2
    assert len(result["transitive_axioms"]) == len(STORAGE_ROOTS + CAPACITY_ROOTS)
    assert not result["pipeline_receipt_issued"] and not result["numerical_gates_discharged"]
    assert before == selected.legacy.dependencies()


def test_actual_seven_period_signed32_case(context, tmp_path):
    selected, _capture = context
    capture = os.environ.get("MERLIN_GRAPH_LOOP_SEVEN_CAPTURE")
    if capture is None:
        pytest.skip("independent seven-period capture unavailable")
    capture = Path(capture)
    result = verify_capacity_files(
        capture / "before.kernel",
        capture / "after.kernel",
        selection=replace(selected, producer=replace(selected.producer, iterations=7)),
        output=tmp_path / "seven",
        index_bits=32,
        max_dynamic_calls=70,
        effects_path=capture / "effects.json",
    )
    assert result["lean_capacity_checked"] and result["complete_boundary_checked"]


def test_three_pending_operations_pass_v1_memory_but_exceed_actual_capacity(context, tmp_path):
    selected, capture = context
    kernel = parse_kernel((capture / "before.kernel").read_bytes().decode())
    loop = kernel.body[0]
    index = Var(loop.var)
    third = argument(argument(loop.body[4], "token", Const(4)), "host", select(index, 2, 0, index * 2 + 2))
    final_wait = argument(loop.body[-1], "token", Const(4))
    body = loop.body[:8] + (third,) + loop.body[8:] + (final_wait,)
    kernel = replace(kernel, body=(replace(loop, body=body),))
    result = check(tmp_path, context, kernel=kernel)
    underlying = json.loads((tmp_path / "proof/v1-check/CHECK.json").read_bytes())
    assert underlying["lean_storage_checked"]
    assert not underlying["source_admission"]["accepted"]
    assert "pending-work bound" in underlying["source_admission"]["error"]
    assert result["returncode"] != 0 and not result["lean_capacity_checked"]
    assert not result["complete_boundary_checked"]
    assert selected.legacy.dependencies()


@pytest.mark.parametrize("field", ["max_pending", "concurrent_pairs", "qualification_identity"])
def test_entire_original_admission_is_bound_not_only_the_limit(context, tmp_path, field):
    selected, capture = context
    policy = json.loads((capture / "effects.json").read_bytes())["admission"]
    if field == "max_pending":
        policy[field] = 3
    elif field == "concurrent_pairs":
        policy[field] = policy[field][:-1]
    else:
        policy[field] = "different-selected-assumption"
    with pytest.raises(ValueError, match="originally selected assumptions"):
        check(tmp_path, context, selection=replace(selected, admission_sha256=admission_digest(policy)))


def test_saved_admission_cannot_promote_a_larger_limit(context, tmp_path):
    _selected, capture = context
    effects = json.loads((capture / "effects.json").read_bytes())
    effects["admission"]["max_pending"] = 3
    saved = tmp_path / "altered.json"
    saved.write_text(json.dumps(effects))
    with pytest.raises(ValueError, match="saved admission differs"):
        check(tmp_path, context, effects_path=saved)


@pytest.mark.parametrize("field", ["legacy", "tool", "proof"])
def test_changed_original_source_runtime_or_proof_selection_refuses(context, tmp_path, field):
    selected, _capture = context
    if field == "legacy":
        wrong = replace(selected, legacy=replace(selected.legacy, sha256="0" * 64))
    elif field == "tool":
        wrong = replace(selected, proof=replace(selected.proof, sha256="0" * 64))
    else:
        wrong = replace(selected, proof=replace(selected.proof, build_sha256="0" * 64))
    with pytest.raises(ValueError):
        check(tmp_path, context, selection=wrong)


def test_missing_final_drain_remains_refused_by_v1(context, tmp_path):
    _selected, capture = context
    kernel = parse_kernel((capture / "before.kernel").read_bytes().decode())
    loop = kernel.body[0]
    result = check(tmp_path, context, kernel=replace(kernel, body=(replace(loop, body=loop.body[:-1]),)))
    assert not result["lean_capacity_checked"] and not result["complete_boundary_checked"]
    assert "v1" in result["refusal"]


def test_intervening_capacity_query_writer_refuses(context, tmp_path, monkeypatch):
    from merlin.verify import lean_graph_loops_capacity as implementation

    original = implementation.subprocess.run
    target = tmp_path / "proof/Obligation.lean"

    def write_after_capacity(command, *args, **kwargs):
        result = original(command, *args, **kwargs)
        if str(command[-1]) == str(target.resolve()):
            target.write_bytes(target.read_bytes() + b" ")
        return result

    monkeypatch.setattr(implementation.subprocess, "run", write_after_capacity)
    result = check(tmp_path, context)
    assert result["returncode"] == 0 and not result["lean_capacity_checked"]
    assert "changed during checking" in result["refusal"]
