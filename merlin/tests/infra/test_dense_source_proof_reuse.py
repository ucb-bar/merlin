"""Reuse exact projected programs; mutated semantics still need a new proof."""

from collections import OrderedDict
from dataclasses import replace

import pytest
from test_compact_dense import ADAPTER, fixture, mutate

from merlin.compile.scheduling import compact_verification as checks
from merlin.compile.scheduling.block_schedule import Load
from merlin.compile.scheduling.compact_dense import emit_compact_dense
from merlin.sched.ir import Call, Loop
from merlin.transforms.dense_planning import dense_blocks


@pytest.fixture
def proven(monkeypatch):
    monkeypatch.setattr(checks, "_SOURCE_PROOFS", OrderedDict())
    p, r, s = fixture(shape=(4, 6, 4), tile=(2, 2, 2))
    b = next(b for b in dense_blocks(p, r) if (b.m, b.n) == (2, 2))
    kernel = emit_compact_dense(p, b, s, ADAPTER, name="proof", dtypes=("i8", "i8", "i32"))
    report = checks.verify_compact_dense(kernel, p, r, s, ADAPTER, static_budget=128, timeout_ms=5000)
    assert report["solver_queries"] > 0
    assert len(checks._SOURCE_PROOFS) == 1
    return kernel, p, r, s, report


def no_solver(_name):
    raise RuntimeError("a changed program must not borrow the previous proof")


def test_unit_completion_wrappers_reuse_exact_proof_without_solver(proven, monkeypatch):
    import z3

    kernel, p, r, s, original = proven
    monkeypatch.setattr(z3, "Tactic", no_solver)
    # _project substitutes all unit-loop indices before considering reuse.
    wrapped = replace(kernel, name="new completion grouping", body=(Loop("completed_group", 1, kernel.body),))
    hit = checks.verify_compact_dense(wrapped, p, r, s, ADAPTER, static_budget=128, timeout_ms=1)
    assert hit["source_proof_reused"] is True
    assert hit["solver_queries"] == 0
    assert hit["reused_solver_queries"] == original["solver_queries"]
    assert hit["kernel_sha256"] == wrapped.digest()
    assert hit["requested_load_bytes"] == original["requested_load_bytes"]
    # Callers cannot poison the retained successful report.
    backing = next(iter(original["padded_high_water_bytes"]))
    original["padded_high_water_bytes"][backing] = 10**9
    again = checks.verify_compact_dense(kernel, p, r, s, ADAPTER, static_budget=128, timeout_ms=1)
    assert again["padded_high_water_bytes"][backing] != 10**9


@pytest.mark.parametrize("change", ["operand", "storage", "resource", "decoder", "domain"])
def test_changed_source_effects_and_context_do_not_reuse(proven, monkeypatch, change):
    import z3

    kernel, p, r, s, _ = proven
    adapter = ADAPTER
    if change == "operand":
        kernel = mutate(kernel, "lhs_load", "dst", 1)
    elif change == "storage":
        s = replace(s, lhs=replace(s.lhs, base=s.lhs.base + 1))
    elif change == "resource":
        r = replace(r, operand_bytes=r.operand_bytes + 1)
    elif change == "decoder":
        def project(call):
            op = ADAPTER.project(call)
            return replace(op, row=op.row + 1) if type(op) is Load else op

        adapter = replace(adapter, project=project)
    else:
        adapter = replace(adapter, numerical_policy_identity="another numerical policy")
    monkeypatch.setattr(z3, "Tactic", no_solver)
    with pytest.raises((RuntimeError, ValueError)):
        checks.verify_compact_dense(kernel, p, r, s, adapter, static_budget=128, timeout_ms=5000)
    assert len(checks._SOURCE_PROOFS) == 1


def test_current_budgets_and_effect_annotations_are_checked_before_reuse(proven):
    kernel, p, r, s, _ = proven
    with pytest.raises(ValueError, match="static budget"):
        checks.verify_compact_dense(kernel, p, r, s, ADAPTER, static_budget=0)
    with pytest.raises(ValueError, match="budgets"):
        checks.verify_compact_dense(kernel, p, r, s, ADAPTER, static_budget=128, timeout_ms=0)
    def annotated(body):
        return tuple(replace(stmt, body=annotated(stmt.body)) if type(stmt) is Loop
                     else replace(stmt, unit="another_engine") if type(stmt) is Call else stmt for stmt in body)

    with pytest.raises(ValueError, match="unmodeled execution"):
        checks.verify_compact_dense(replace(kernel, body=annotated(kernel.body)), p, r, s, ADAPTER, static_budget=128)


def test_failed_checks_do_not_create_cache_entries(monkeypatch):
    import z3

    monkeypatch.setattr(checks, "_SOURCE_PROOFS", OrderedDict())
    p, r, s = fixture(shape=(4, 6, 4), tile=(2, 2, 2))
    b = next(b for b in dense_blocks(p, r) if (b.m, b.n) == (2, 2))
    kernel = emit_compact_dense(p, b, s, ADAPTER, name="no proof", dtypes=("i8", "i8", "i32"))
    monkeypatch.setattr(z3, "Tactic", no_solver)
    with pytest.raises(RuntimeError):
        checks.verify_compact_dense(kernel, p, r, s, ADAPTER, static_budget=128)
    assert not checks._SOURCE_PROOFS
