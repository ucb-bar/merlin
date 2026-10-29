"""Complete actual Kernel correspondence and valid changed-IR refusals."""

from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path

import pytest

from merlin.sched.ir import NULL, Call, ConcretePtr, Const, Kernel, Loop, Ptr, Var, concretize, instances, select
from merlin.verify.lean_graph_loops import LeanSelection, project_pair, verify_files


def flattened(kernel):
    calls = []
    for call, environment in instances(kernel):
        values = concretize(call, environment)
        args = []
        for name, _ in call.args:
            value = values[name]
            if type(value) is ConcretePtr:
                value = Ptr(value.tensor, Const(value.offset))
            elif type(value) is int:
                value = Const(value)
            elif value is None:
                value = NULL
            args.append((name, value))
        calls.append(replace(call, args=tuple(args)))
    return replace(kernel, body=tuple(calls))


def loop_kernel(extent=3):
    index = Var("iteration")
    return Kernel(
        "complete_calls",
        (),
        (
            Loop(
                "iteration",
                extent,
                (
                    Call("copy", (("offset", index * 2), ("size", Const(2)))),
                    Call("observe", (("offset", index * 2 + 1),)),
                ),
            ),
        ),
    )


@pytest.fixture(scope="module")
def selection():
    selected = os.environ.get("MERLIN_GRAPH_LOOP_SELECTION")
    if selected is None:
        pytest.skip("explicit graph Lean selection unavailable; no kernel result inferred")
    data = json.loads(Path(selected).read_text())
    for key in ("executable", "library", "build_record"):
        data[key] = Path(data[key])
    return LeanSelection(**data)


def check(tmp_path, selection, before, after, *, name="proof", bits=64):
    source, result = tmp_path / (name + ".before.kernel"), tmp_path / (name + ".after.kernel")
    source.write_text(before.text())
    result.write_text(after.text())
    return verify_files(
        source, result, selection=selection, output=tmp_path / name, index_bits=bits, max_dynamic_calls=1024
    )


def test_complete_projection_ignores_candidate_count_hints():
    before = loop_kernel(7)
    result = project_pair(before, flattened(before), index_bits=64, max_dynamic_calls=14)
    assert result["dynamic_calls"] == 14 and len(result["target"]) == 14
    with pytest.raises(ValueError, match="no prefix checked"):
        project_pair(before, flattened(before), index_bits=64, max_dynamic_calls=13)


@pytest.mark.parametrize("mutation", ["missing", "duplicated", "renamed_boundary", "unknown_attributes", "nested"])
def test_incomplete_or_unsupported_boundaries_refuse(mutation):
    before = loop_kernel()
    after = flattened(before)
    if mutation == "missing":
        after = replace(after, body=after.body[:-1])
    elif mutation == "duplicated":
        after = replace(after, body=after.body + after.body[-1:])
    elif mutation == "renamed_boundary":
        after = replace(after, name="different")
    elif mutation == "unknown_attributes":
        before = replace(before, attrs=(("unmodeled", "true"),))
    else:
        before = replace(before, body=(replace(before.body[0], body=before.body),))
    with pytest.raises(ValueError):
        project_pair(before, after, index_bits=64, max_dynamic_calls=1024)


def test_unprojected_execution_fields_are_not_silently_erased():
    before = loop_kernel()
    first, second = before.body[0].body
    before = replace(before, body=(replace(before.body[0], body=(replace(first, unit="separate-agent"), second)),))
    with pytest.raises(ValueError, match="separately projected"):
        project_pair(before, flattened(before), index_bits=64, max_dynamic_calls=1024)


def test_empty_domain_cannot_issue_an_empty_success():
    before = loop_kernel(0)
    with pytest.raises(ValueError, match="structurally valid"):
        project_pair(before, replace(before, body=()), index_bits=64, max_dynamic_calls=1024)


def test_actual_production_materialization_is_kernel_checked(selection, tmp_path):
    capture = os.environ.get("MERLIN_GRAPH_LOOP_CAPTURE")
    if capture is None:
        pytest.skip("actual pinned producer capture unavailable")
    capture = Path(capture)
    result = verify_files(
        capture / "before.kernel",
        capture / "after.kernel",
        selection=selection,
        output=tmp_path / "actual",
        index_bits=64,
        max_dynamic_calls=128,
    )
    assert result["lean_kernel_relation_checked"]
    assert result["projection"]["dynamic_calls"] == 30
    assert result["transitive_axioms"] and not result["pipeline_receipt_issued"]
    assert result["python_projection_trusted"] and not result["numerical_gates_discharged"]


def test_late_wrong_address_passes_good_prefix_but_fails_complete_relation(selection, tmp_path):
    before = loop_kernel(9)
    correct = flattened(before)
    last = correct.body[-1]
    changed = replace(last, args=(("offset", Const(999)),))
    after = replace(correct, body=correct.body[:-1] + (changed,))
    good_prefix = loop_kernel(8)
    assert flattened(good_prefix).body == after.body[:16]
    assert check(tmp_path, selection, good_prefix, flattened(good_prefix), name="prefix")[
        "lean_kernel_relation_checked"
    ]
    result = check(tmp_path, selection, before, after, name="complete")
    assert not result["lean_kernel_relation_checked"] and result["returncode"] != 0


def test_tail_operand_is_recomputed_and_wrong_last_tail_refuses(selection, tmp_path):
    index = Var("tile")
    before = Kernel(
        "tail", (), (Loop("tile", 3, (Call("copy", (("offset", index * 4), ("size", select(index, 2, 1, 4)))),)),)
    )
    correct = flattened(before)
    assert check(tmp_path, selection, before, correct, name="tail")["lean_kernel_relation_checked"]
    wrong = replace(correct.body[-1], args=(("offset", Const(8)), ("size", Const(4))))
    result = check(tmp_path, selection, before, replace(correct, body=correct.body[:-1] + (wrong,)), name="bad_tail")
    assert not result["lean_kernel_relation_checked"]


def test_index_intermediate_overflow_is_not_an_integer_mathematics_pass(selection, tmp_path):
    index = Var("iteration")
    before = Kernel("overflow", (), (Loop("iteration", 2, (Call("copy", (("offset", index + (2**31 - 1)),)),)),))
    result = check(tmp_path, selection, before, flattened(before), name="overflow", bits=32)
    assert not result["lean_kernel_relation_checked"]


def test_changed_selected_semantics_and_import_path_refuse(selection, tmp_path):
    altered = tmp_path / "build.json"
    altered.write_bytes(selection.build_record.read_bytes() + b" ")
    with pytest.raises(ValueError, match="dependency record changed"):
        replace(selection, build_record=altered).dependencies()
    with pytest.raises(ValueError, match="import search path"):
        replace(selection, library=tmp_path / "foreign-library").dependencies()
    with pytest.raises(ValueError, match="executable changed"):
        replace(selection, sha256="0" * 64).dependencies()


def test_parse_and_digest_share_the_same_input_buffer(selection, tmp_path, monkeypatch):
    before, after = tmp_path / "before.kernel", tmp_path / "after.kernel"
    kernel = loop_kernel()
    before.write_text(kernel.text())
    after.write_text(flattened(kernel).text())
    original = Path.read_text

    def forbid_second_read(path, *args, **kwargs):
        if path in (before, after):
            pytest.fail("actual IR parsed from a second, separately mutable read")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", forbid_second_read)
    result = verify_files(
        before, after, selection=selection, output=tmp_path / "same-buffer", index_bits=64, max_dynamic_calls=128
    )
    assert result["lean_kernel_relation_checked"]


def test_intervening_original_ir_writer_invalidates_success(selection, tmp_path, monkeypatch):
    from merlin.verify import lean_graph_loops as implementation

    before, after = tmp_path / "before.kernel", tmp_path / "after.kernel"
    kernel = loop_kernel()
    before.write_text(kernel.text())
    after.write_text(flattened(kernel).text())
    original = implementation.subprocess.run

    def write_after_proof(*args, **kwargs):
        result = original(*args, **kwargs)
        after.write_text(after.read_text().replace("observe", "changed"))
        return result

    monkeypatch.setattr(implementation.subprocess, "run", write_after_proof)
    result = verify_files(
        before, after, selection=selection, output=tmp_path / "intervening", index_bits=64, max_dynamic_calls=128
    )
    assert result["returncode"] == 0 and not result["lean_kernel_relation_checked"]
    assert "changed during checking" in result["refusal"]


def test_intervening_query_writer_invalidates_kernel_success(selection, tmp_path, monkeypatch):
    from merlin.verify import lean_graph_loops as implementation

    original = implementation.subprocess.run

    def write_after_proof(command, *args, **kwargs):
        result = original(command, *args, **kwargs)
        query = Path(command[-1])
        query.write_bytes(query.read_bytes() + b" ")
        return result

    monkeypatch.setattr(implementation.subprocess, "run", write_after_proof)
    before = loop_kernel()
    result = check(tmp_path, selection, before, flattened(before))
    assert result["returncode"] == 0 and not result["lean_kernel_relation_checked"]
    assert "changed during checking" in result["refusal"]


@pytest.mark.parametrize(
    "report",
    [
        "'claim' depends on axioms: [propext",
        "'claim' does not depend on any axioms but this is not a complete report",
        "'claim' depends on axioms: [Lean.ofReduceBool]",
        "'claim' depends on axioms: [sorryAx]",
    ],
)
def test_incomplete_or_unapproved_axiom_report_refuses(report):
    from merlin.verify.lean_graph_loops import _audit

    with pytest.raises(ValueError):
        _audit(report, ("claim",))
