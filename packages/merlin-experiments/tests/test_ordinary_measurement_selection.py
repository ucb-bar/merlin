"""Original retained normal evaluator selection with synthetic external engines."""

import pytest
from merlin_experiments.phase2 import ordinary_measurement_selection as SELECT
from merlin_experiments.phase2 import ordinary_search_policy as OS
from merlin_experiments.phase2 import search_coordinator as SC
from merlin_experiments.phase2 import search_inputs as INPUTS
from merlin_experiments.phase2.contracts import StageGateError
from test_ordinary_fast_feedback import extend_corpus
from test_ordinary_search_policy import ordinary_inputs as ordinary_inputs
from test_ordinary_search_policy import revisions


def test_explicit_public_selection_keeps_unselected_obligations_and_cannot_rank(ordinary_inputs, tmp_path):
    extend_corpus(ordinary_inputs, tmp_path)
    policy = OS.prepare_ordinary_search_policy(**ordinary_inputs.arguments)
    candidate, baseline = revisions(policy, tmp_path)
    members = tuple(sorted(policy.validate_search_inputs()))[:1]
    result = policy.evaluate_retained(
        candidate,
        baseline,
        work_root=tmp_path / "selected",
        call_index=0,
        timeout_s=20,
        kind=SELECT.KIND,
        selected_members=members,
    )
    document = INPUTS.SearchInputs(policy).validate_result(result, candidate, SELECT.KIND)
    SELECT.validate_feedback(document, members=members)
    assert document["stopping"]["status"] == "undeterminable"
    assert document["summary"]["recoverable"]["status"] == "unavailable"
    assert {key for key in members} == {(row["family"], row["capsule"]) for row in document["cells"] if row["measured"]}
    for row in document["cells"]:
        if (row["family"], row["capsule"]) not in members:
            assert row["baseline_gsim_cycles"] is None and row["candidate_gsim_cycles"] is None
    assert {spec.family for spec, *_ in ordinary_inputs.executions} <= {key[0] for key in members}
    ranking = SC.rank_measured_contenders(
        [
            {
                "state": "completed",
                "kind": SELECT.KIND,
                "source_sha256": candidate.source_sha256,
                "job_id": "synthetic-selected",
                "revision_id": candidate.revision_id,
                "result": document,
            }
        ],
        roster=tuple(sorted(policy.validate_search_inputs())),
        corpus_sha256=policy.search_corpus.capsules_sha256,
        configuration_sha256=policy.feedback_evaluator.certificate.sha256,
        timing_kind=INPUTS.ORDINARY_KIND,
    )
    assert ranking["ranked"] == []


def test_selective_scope_cannot_smuggle_private_or_duplicate_members(ordinary_inputs, tmp_path):
    extend_corpus(ordinary_inputs, tmp_path)
    policy = OS.prepare_ordinary_search_policy(**ordinary_inputs.arguments)
    first = tuple(sorted(policy.validate_search_inputs()))[0]
    assert SELECT.parse_members(policy, "/".join(first)) == (first,)
    with pytest.raises(StageGateError, match="repeats"):
        SELECT.parse_members(policy, "/".join(first) + "," + "/".join(first))
    with pytest.raises(StageGateError, match="unknown"):
        SELECT.parse_members(policy, "private/unavailable")
