"""Publication preserves chronological evidence without executing native engines."""

import copy
import json

import pytest
from merlin_experiments.phase2 import contracts
from merlin_experiments.phase2.revision_journal import RevisionJournal


def record(iteration):
    return {"iteration": iteration, "probe_receipts": [], "analysis": {"status": "synthetic"}}


def test_fresh_and_reverted_publication_preserve_identity_and_immutable_evidence(tmp_path):
    journal = RevisionJournal(tmp_path)
    first = record(0)
    baseline = {"lowered_text": "baseline"}
    primary = {"lowered_text": "first", "baseline_artifacts": baseline}
    portfolio = {"model": {"lowered_text": "first"}}
    journal.publish_analysis(first, primary, portfolio)
    initial_path = tmp_path / "iteration_0000.json"
    initial_bytes = initial_path.read_bytes()
    initial_hash = contracts.sha256_file(initial_path)
    assert journal.record_sha256[0] == initial_hash
    assert initial_bytes == contracts.canonical_json(first)
    assert journal.artifacts is journal.iteration_artifacts[0] is primary
    assert journal.portfolio_artifacts is journal.iteration_portfolio_artifacts[0] is portfolio
    assert journal.baseline_artifacts is baseline
    assert "baseline_artifacts" not in primary
    assert initial_path.stat().st_mode & 0o222 == 0

    first["probe_receipts"].append({"path": "synthetic-probe.json", "sha256": "a" * 64})
    first["decision_feedback"] = {"status": "synthetic"}
    sandboxes = {"candidate": {"command_prefix": ["synthetic"], "package_path": "source"}}
    journal.compiler_sandboxes[0] = sandboxes
    journal.compiler_sandbox_sha256[0] = sandbox_hash = contracts.document_sha256(sandboxes)
    second_primary = {"lowered_text": "second"}
    second_portfolio = {"model": {"lowered_text": "second"}}
    journal.publish_analysis(record(1), second_primary, second_portfolio)
    reused_primary = copy.deepcopy(primary)
    journal.publish_reuse(record(2), reused_primary, source_iteration=0)

    assert [row["iteration"] for row in journal.iterations] == [0, 1, 2]
    assert journal.artifacts is journal.iteration_artifacts[2] is reused_primary
    assert journal.previous_artifacts is second_primary
    assert journal.previous_portfolio_artifacts is second_portfolio
    assert journal.portfolio_artifacts == portfolio
    assert journal.portfolio_artifacts is not portfolio
    assert journal.compiler_sandboxes[2] == sandboxes
    assert journal.compiler_sandboxes[2] is not sandboxes
    assert journal.compiler_sandbox_sha256[2] == sandbox_hash
    assert journal.iterations[2]["probe_receipts"] == []
    assert initial_path.read_bytes() == initial_bytes
    assert journal.record_sha256[0] == contracts.sha256_file(initial_path) == initial_hash
    assert json.loads(initial_bytes)["probe_receipts"] == []


@pytest.mark.parametrize("iteration", [0, 2, -1, True, "1", None])
@pytest.mark.parametrize("publication", ["analysis", "reuse", "imported_seed"])
def test_invalid_publication_does_not_advance_state_or_replace_record(tmp_path, iteration, publication):
    journal = RevisionJournal(tmp_path)
    journal.publish_analysis(record(0), {"lowered_text": "original"}, {})
    before = copy.deepcopy(vars(journal))
    original = (tmp_path / "iteration_0000.json").read_bytes()
    with pytest.raises(ValueError):
        if publication == "analysis":
            journal.publish_analysis(record(iteration), {"lowered_text": "invalid"}, {})
        elif publication == "reuse":
            journal.publish_reuse(record(iteration), {"lowered_text": "invalid"}, source_iteration=0)
        else:
            journal.publish_imported_seed(record(iteration), {}, {}, reconstructed_sandboxes=None, sandbox_digest=None)
    assert vars(journal) == before
    assert (tmp_path / "iteration_0000.json").read_bytes() == original
    assert list(tmp_path.iterdir()) == [tmp_path / "iteration_0000.json"]


def test_failed_record_write_does_not_publish_state(tmp_path):
    journal = RevisionJournal(tmp_path)
    path = tmp_path / "iteration_0000.json"
    path.write_text("preexisting evidence")
    before = copy.deepcopy(vars(journal))
    with pytest.raises(FileExistsError):
        journal.publish_analysis(record(0), {"baseline_artifacts": {"original": True}}, {})
    assert vars(journal) == before
    assert path.read_text() == "preexisting evidence"


def test_imported_seed_preserves_reconstructed_sandbox_binding(tmp_path):
    journal = RevisionJournal(tmp_path)
    primary, portfolio = {"lowered_text": "imported"}, {"model": {"lowered_text": "imported"}}
    sandboxes = {"candidate": {"package_path": "reconstructed"}, "baseline": {"package_path": "baseline"}}
    digest = contracts.document_sha256(sandboxes)
    journal.publish_imported_seed(
        record(0), primary, portfolio, reconstructed_sandboxes=sandboxes, sandbox_digest=digest
    )
    assert journal.artifacts is primary
    assert journal.portfolio_artifacts is portfolio
    assert journal.compiler_sandboxes[0] == sandboxes
    assert journal.compiler_sandbox_sha256[0] == digest
    journal.publish_analysis(record(1), {"lowered_text": "next"}, {})
    assert journal.previous_artifacts is primary
    assert journal.previous_portfolio_artifacts is portfolio
    assert journal.compiler_sandbox_sha256[0] == digest
