"""Fail closed on incomplete canonical results, without making reports proof authority.

These are worker protocol tests. The canonical handlers run with stubbed proof
engines; compiler/Lean integration and deliberate wrong-output tests are separate.
"""

from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import pytest

from merlin.verify import gate_checks, gate_worker

CHECKS = ("tensor_refinement", "compiler_loop_refinement", "compiler_buffer_refinement")


def _worker(tmp_path, monkeypatch, name, detail):
    (tmp_path / "request.json").write_text(json.dumps({"check": name}))
    monkeypatch.setattr(gate_worker, "resolve", lambda name: lambda request: detail)
    assert gate_worker.main(["worker", str(tmp_path)]) == 0
    return json.loads((tmp_path / "verdict.json").read_text())


def _canonical_result(tmp_path, monkeypatch, name, *, universal="abstracted", defined=False):
    """Exercise actual result construction, including completion validation calls."""
    from merlin.verify import buffer_reuse_gate, buffer_reuse_intake, compiler_loop_intake, loop_transform_gate
    from merlin.verify.lean import check

    before, after = tmp_path / "before.mlir", tmp_path / "after.mlir"
    before.write_text("module {}")
    after.write_text("module {} // after")
    sdk = tmp_path / "sdk"
    sdk.mkdir()
    for tool in ("lake", "lean", "python"):
        (sdk / tool).touch()
    job = {
        "directory": str(tmp_path),
        "before": str(before),
        "after": str(after),
        "params": {
            "sdk_bin": str(sdk),
            "compiler_python": str(sdk / "python"),
            "index_bits": 64,
            "intermediates": [],
            "kinds": [compiler_loop_intake.UNROLL],
            "factor": 2,
            "tile": 3,
            "allocator": "total_private_unobservable",
        },
    }
    axioms = ["propext", "Classical.choice", "Quot.sound"]
    if name == "tensor_refinement":
        monkeypatch.setattr(check, "lean_exe", lambda: sdk / "lean")
        monkeypatch.setattr(
            check, "check_typed_stage", lambda *a, **kw: {"verdict": "verified", "functions": ["f"], "axioms": axioms}
        )
        monkeypatch.setattr(check, "check_stage", lambda *a, **kw: {"verdict": universal})
        monkeypatch.setattr(check, "check_support", lambda *a, **kw: {"verdict": "verified"})
        monkeypatch.setattr(
            check, "check_definedness", lambda *a, **kw: {"verdict": "verified" if defined else "abstracted"}
        )
        monkeypatch.setattr(
            check,
            "compose_chain",
            lambda *a, **kw: {"refinement_verdict": "verified", "verdict": "verified" if defined else "abstracted"},
        )
        return gate_checks.tensor_refinement(job)
    report = tmp_path / "REPORT.json"
    report.write_text(
        json.dumps(
            {
                "relation": loop_transform_gate.RELATION,
                "premises": loop_transform_gate.PREMISES,
                "audit": {"axioms_used": axioms},
            }
        )
    )
    completion = SimpleNamespace(report=str(report))
    prepared = SimpleNamespace(relation=buffer_reuse_intake.RELATION)
    validations = []
    if name == "compiler_loop_refinement":
        monkeypatch.setattr(compiler_loop_intake, "prepare", lambda *a, **kw: prepared)
        monkeypatch.setattr(loop_transform_gate, "check", lambda *a, **kw: completion)
        monkeypatch.setattr(loop_transform_gate, "validate_completion", lambda *a: validations.append(a))
        detail = gate_checks.compiler_loop_refinement(job)
    else:
        job["params"]["kinds"] = ["shape_view_normalization", "buffer_reuse", compiler_loop_intake.TILE]
        monkeypatch.setattr(buffer_reuse_intake, "prepare_chain", lambda *a, **kw: prepared)
        monkeypatch.setattr(
            buffer_reuse_gate, "check", lambda *a, **kw: SimpleNamespace(status="checked", completion=completion)
        )
        monkeypatch.setattr(buffer_reuse_gate, "validate_completion", lambda *a: validations.append(a))
        detail = gate_checks.compiler_buffer_refinement(job)
    assert validations == [(completion, prepared)]
    return detail


@pytest.mark.parametrize("name", CHECKS)
@pytest.mark.parametrize("detail", [None, {}, {"status": "pending"}])
def test_incomplete_handler_results_never_become_worker_acceptance(tmp_path, monkeypatch, name, detail):
    result = _worker(tmp_path, monkeypatch, name, detail)
    assert result["verdict"] == "not_established"
    assert result["detail"]["error_type"] == "GateRejected"


@pytest.mark.parametrize("name", CHECKS)
def test_canonical_handler_result_is_accepted_but_each_missing_component_refuses(tmp_path, monkeypatch, name):
    detail = _canonical_result(tmp_path, monkeypatch, name)
    assert _worker(tmp_path, monkeypatch, name, detail)["verdict"] == "accepted"
    for key in detail:
        partial = copy.deepcopy(detail)
        del partial[key]
        result = _worker(tmp_path, monkeypatch, name, partial)
        assert result["verdict"] == "not_established", key


@pytest.mark.parametrize("universal", ["verified", "abstracted", "unmeasured"])
@pytest.mark.parametrize("defined", [False, True])
def test_optional_tensor_strengthenings_remain_optional(tmp_path, monkeypatch, universal, defined):
    detail = _canonical_result(tmp_path, monkeypatch, "tensor_refinement", universal=universal, defined=defined)
    assert _worker(tmp_path, monkeypatch, "tensor_refinement", detail)["verdict"] == "accepted"
    assert ("definedness" in detail["established"]) is defined
    assert ("universal_boundary" in detail["established"]) is (universal == "verified")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("chains", {}),
        ("chains", {"another_function": "chain.json"}),
        ("chains", {"f": ""}),
        ("functions", ["f", "f"]),
        ("established", ["definedness"]),
        ("strengthenings", {"universal_boundary": "abstracted", "definedness": "pending"}),
        (
            "claim",
            "strict refinement on inputs of the declared types; defined on typed, nonpoison inputs (no UB or poison)",
        ),
        ("axioms", ["sorryAx"]),
    ],
)
def test_tensor_coverage_and_optional_claims_cannot_be_dropped_or_promoted(tmp_path, monkeypatch, field, value):
    detail = _canonical_result(tmp_path, monkeypatch, "tensor_refinement")
    detail[field] = value
    assert _worker(tmp_path, monkeypatch, "tensor_refinement", detail)["verdict"] == "not_established"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("claim", "native execution equivalence"),
        ("premises", "unconditional"),
        ("boundaries", {}),
        ("boundaries", {"other_pass": "theorem"}),
        ("boundaries", {"ordered_loop_unrolling": ""}),
        ("boundaries", {"ordered_loop_unrolling": "Merlin.GraphGateProfileAdjacent.complete3"}),
        ("native_fidelity", True),
        ("general_tiling", True),
        ("axioms", ["Lean.trustCompiler"]),
    ],
)
def test_loop_result_cannot_drop_coverage_or_promote_its_model_claim(tmp_path, monkeypatch, field, value):
    detail = _canonical_result(tmp_path, monkeypatch, "compiler_loop_refinement")
    detail[field] = value
    assert _worker(tmp_path, monkeypatch, "compiler_loop_refinement", detail)["verdict"] == "not_established"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("claim", "all defined entries equal returned values/errors/terminal world"),
        ("premises", {}),
        ("premises", {"allocator": "may_fail"}),
        ("covered_chain", []),
        ("covered_chain", ["unknown_pass"]),
        ("covered_chain", ["ordered_loop_segmentation", "buffer_reuse"]),
        ("compiler_enforcement", False),
        ("native_allocator_fidelity", True),
        ("full_error_or_world_equivalence", True),
        ("axioms", ["sorryAx"]),
    ],
)
def test_buffer_result_preserves_allocator_and_possible_success_boundary(tmp_path, monkeypatch, field, value):
    detail = _canonical_result(tmp_path, monkeypatch, "compiler_buffer_refinement")
    detail[field] = value
    assert _worker(tmp_path, monkeypatch, "compiler_buffer_refinement", detail)["verdict"] == "not_established"


@pytest.mark.parametrize("kinds", [["shape_view_normalization"], ["buffer_reuse"], ["ordered_loop_segmentation"]])
def test_buffer_chain_allows_other_stages_to_be_identity(tmp_path, monkeypatch, kinds):
    detail = _canonical_result(tmp_path, monkeypatch, "compiler_buffer_refinement")
    detail["covered_chain"] = kinds
    assert _worker(tmp_path, monkeypatch, "compiler_buffer_refinement", detail)["verdict"] == "accepted"
