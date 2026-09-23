"""A training member that publishes no reusable artifacts must not abort the campaign.

Widening the portfolio was impossible: an empty artifact set is tolerated for the OBJECTIVE (the
bundle is simply not written) but raised for a training member, so adding any model whose emission
does not publish a full reusable artifact set killed the run for every member including the
objective. The models with the most optimization room are exactly the ones whose emission is
hardest, which made the check self-defeating.

The member's ANALYSIS is still carried, because a host-lane cost census is what identifies where a
model's cost is and it does not depend on the artifacts being reusable for probe preparation.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from merlin_experiments.phase2.portfolio_analysis import PortfolioAnalysis

from merlin.common.digest import sha256_bytes


def _artifacts(tag: str) -> dict[str, object]:
    lowered = f"{tag}:lowered_text"
    command = json.dumps({"member": tag})
    return {
        "lowered_text": lowered,
        "decoded_trace": [],
        "command_buffer": json.loads(command),
        "command_buffer_text": command,
        "candidate_sha256": "cand",
        "candidate_lowered_sha256": sha256_bytes(lowered.encode()),
        "candidate_command_buffer_sha256": sha256_bytes(command.encode()),
        "task_instruction_evidence": {},
    }


def _analysis(capsule: str) -> dict[str, object]:
    artifacts = _artifacts("primary" if capsule == "objective" else capsule)
    return {
        "candidate_sha256": "cand",
        "workload": {"capsule_sha256": capsule},
        "diagnostics": {},
        "emission": {key: artifacts[key] for key in ("candidate_lowered_sha256", "candidate_command_buffer_sha256")},
    }


def _stub(tmp_path: Path, capsules: tuple[str, ...]):
    """A minimal object exposing exactly what `persist_static_analysis_bundle` reads."""
    stub = SimpleNamespace(
        session=SimpleNamespace(
            journal=SimpleNamespace(output=tmp_path),
            inputs=SimpleNamespace(
                portfolio_sentinels=tuple(SimpleNamespace(capsule_sha256=c) for c in capsules),
                portfolio_identity_sha256="portfolio",
            ),
        ),
        written={},
    )
    return stub


def _persist(stub, *, portfolio_artifacts, capsules):
    record = {
        "candidate_sha256": "cand",
        "iteration": 0,
        "analysis": _analysis(capsules[0]),
        "portfolio": {"members": [{"analysis": _analysis(c)} for c in capsules]},
        "cross_run_static_analysis_binding": {"binding": "x"},
    }
    result = PortfolioAnalysis.persist_static_analysis_bundle(
        stub, record, _artifacts("primary"), portfolio_artifacts=portfolio_artifacts
    )
    if result is not None:
        path = Path(result["path"])
        assert path.stat().st_mode & 0o222 == 0
        assert sha256_bytes(path.read_bytes()) == result["sha256"]
        stub.written["bundle"] = json.loads(path.read_bytes())
    return result


def test_a_member_without_artifacts_is_recorded_and_the_bundle_is_written(tmp_path) -> None:
    capsules = ("objective", "trainee")
    stub = _stub(tmp_path, capsules)
    result = _persist(stub, capsules=capsules, portfolio_artifacts={"objective": _artifacts("primary")})
    assert result is not None, "the bundle must still be written"
    rows = stub.written["bundle"]["portfolio_member_artifacts"]
    assert [row["capsule_sha256"] for row in rows] == list(capsules)
    assert rows[0]["status"] == "reusable"
    assert rows[1]["artifacts"] is None
    assert rows[1]["status"] == "member_published_no_reusable_artifacts"


def test_the_members_analysis_survives_even_with_no_artifacts(tmp_path) -> None:
    """The cost census is the reason to widen the portfolio at all."""
    capsules = ("objective", "trainee")
    stub = _stub(tmp_path, capsules)
    _persist(stub, capsules=capsules, portfolio_artifacts={"objective": _artifacts("primary")})
    analyses = stub.written["bundle"]["member_analyses"]
    assert len(analyses) == 2
    assert analyses[1]["workload"]["capsule_sha256"] == "trainee"


def test_a_member_WITH_artifacts_is_still_marked_reusable(tmp_path) -> None:
    capsules = ("objective", "trainee")
    stub = _stub(tmp_path, capsules)
    _persist(
        stub,
        capsules=capsules,
        portfolio_artifacts={"objective": _artifacts("primary"), "trainee": _artifacts("trainee")},
    )
    rows = stub.written["bundle"]["portfolio_member_artifacts"]
    assert all(row["status"] == "reusable" for row in rows)
    assert rows[1]["artifacts"]["lowered_text"] == "trainee:lowered_text"


def test_the_objectives_own_missing_artifacts_are_still_fatal(tmp_path) -> None:
    """Tolerating the OBJECTIVE would leave a bundle that looks complete and binds to nothing."""
    capsules = ("objective", "trainee")
    stub = _stub(tmp_path, capsules)
    with pytest.raises(ValueError) as caught:
        # An explicit None for the objective defeats the setdefault that would supply it.
        _persist(stub, capsules=capsules, portfolio_artifacts={"objective": None, "trainee": _artifacts("trainee")})
    assert "objective" in str(caught.value)


def test_several_members_may_each_lack_artifacts(tmp_path) -> None:
    capsules = ("objective", "a", "b", "c")
    stub = _stub(tmp_path, capsules)
    _persist(stub, capsules=capsules, portfolio_artifacts={"objective": _artifacts("primary")})
    rows = stub.written["bundle"]["portfolio_member_artifacts"]
    assert len(rows) == 4
    assert [row["artifacts"] is None for row in rows] == [False, True, True, True]
