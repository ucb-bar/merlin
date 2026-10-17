"""Phase 2 scores accelerator work; Merlin-supplied host compute is excluded from every score."""

from __future__ import annotations

import hashlib

import pytest
from merlin_experiments.phase2 import metric_scope as S


def _declaration(tmp_path, placements, decisions):
    program = tmp_path / "model.mlir"
    program.write_text("module {}\n")
    return {
        "kind": "model",
        "linalg_mlir": "model.mlir",
        "software_screen": {
            "status": "admitted",
            "program_sha256": hashlib.sha256(program.read_bytes()).hexdigest(),
            "placements": placements,
            "decisions": decisions,
        },
    }


def _row(lane, op, count=1):
    return {"status": "admitted", "placement": lane, "mlir_operation": op, "count": count}


def test_the_scope_is_the_frozen_split_and_names_what_is_excluded(tmp_path):
    declaration = _declaration(
        tmp_path,
        {"accelerator": 1, "host": 1},
        [_row("accelerator", "linalg.matmul", 2), _row("host", "linalg.softmax")],
    )
    scope = S.bound_scope(declaration, tmp_path)
    assert scope["kind"] == S.MERLIN_HOST_SPLIT
    assert scope["scored_operations"] == {"linalg.matmul": 2}
    assert scope["excluded_operations"] == {"linalg.softmax": 1}
    assert scope["candidate_whole_program_admissible"] is False
    assert "im2col" in scope["scored"] and "Merlin-supplied" in scope["excluded"]


def test_an_unbound_placement_has_no_scope(tmp_path):
    declaration = _declaration(tmp_path, {"accelerator": 1}, [_row("accelerator", "linalg.matmul")])
    (tmp_path / "model.mlir").write_text("module { changed }\n")
    with pytest.raises(S.MetricScopeRefusal, match="other program bytes"):
        S.bound_scope(declaration, tmp_path)
    with pytest.raises(S.MetricScopeRefusal):
        S.declared_scope({"kind": "model"})


def test_a_split_programs_absolute_count_is_never_promoted_to_a_score():
    split = S.declared_scope({"software_screen": {"placements": {"accelerator": 2, "host": 1}}})
    assert S.scored_cycles(split, 1000)["status"] == "unknown"
    assert S.scored_cycles(split, 1000, merlin_host_cycles=300)["cycles"] == 700
    with pytest.raises(S.MetricScopeRefusal):
        S.scored_cycles(split, 1000, merlin_host_cycles=1000)
    only = S.declared_scope({"software_screen": {"placements": {"accelerator": 2}}})
    assert S.scored_cycles(only, 1000) == {
        "status": "scored",
        "cycles": 1000,
        "basis": "complete call; no Merlin host stage",
    }


def test_a_paired_delta_needs_byte_identical_merlin_host_code():
    split = S.declared_scope({"software_screen": {"placements": {"accelerator": 2, "host": 1}}})
    same = S.paired_delta(
        split, candidate_cycles=900, baseline_cycles=1000, candidate_host_sha256="a" * 64, baseline_host_sha256="a" * 64
    )
    assert same == {
        "status": "scored",
        "delta_cycles": -100,
        "basis": "Merlin host code is byte-identical in both arms, so its stage cancels",
    }
    for other in ("b" * 64, None):
        refused = S.paired_delta(
            split,
            candidate_cycles=900,
            baseline_cycles=1000,
            candidate_host_sha256="a" * 64,
            baseline_host_sha256=other,
        )
        assert refused["status"] == "refused" and refused["delta_cycles"] is None
