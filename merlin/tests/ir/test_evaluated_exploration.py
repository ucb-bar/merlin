import hashlib
from dataclasses import replace

import pytest
from test_pass_exploration import emission, facts, families, source

from merlin.transforms.evaluated_exploration import (
    CompletedObservation,
    ExplorationIdentity,
    exploration_context_identity,
    explore_evaluated,
)
from merlin.transforms.pipeline_state import PipelineContext

CONTEXT = PipelineContext(hardware="selected-public-facts", numerical_policy="source-exact",
                          resources="bounded-views", execution="synchronous", library="reviewed")


def identity():
    return ExplorationIdentity(*("1" * 64 for _ in range(9)), exploration_context_identity(CONTEXT),
                               "xdsl", "modeled_roi")


def observe(program, request, **changes):
    return replace(CompletedObservation(request, hashlib.sha256(program.data).hexdigest(), "2" * 64,
        "pass", "complete", True, 200 if b"arith.addi" in program.data else 100), **changes)


def run(evaluate, **options):
    return explore_evaluated(source(), families(), facts, output="program", emission=emission,
                            evaluate=evaluate, identity=identity(), context=CONTEXT, **options)


def test_real_registered_transform_dedup_and_exact_evaluated_product_selected():
    evaluated = []
    def evaluate(program, receipt, request):
        evaluated.append(program)
        return observe(program, request)
    result = run(evaluate)
    assert len(result.candidates) == 4 and len(evaluated) == 2
    assert sum(candidate.duplicate_of is not None for candidate in result.candidates) == 2
    assert result.winner.emission is evaluated[1]  # same actual object; no winner recompilation
    assert b"arith.addi" not in result.winner.emission.data
    assert result.ranking["ranked"][0][1] == (100,)
    assert result.ranking["provisional"]


@pytest.mark.parametrize("changes", [{"cycles": None}, {"complete_roi": False},
    {"timing_status": "unknown"}, {"numerical_status": "fail"}])
def test_unknown_partial_or_numerical_failure_never_ranks(changes):
    result = run(lambda p, r, i: observe(p, i, **changes))
    assert result.winner is None
    assert len(result.ranking["unranked_programs"]) == 2


@pytest.mark.parametrize("field", ["source_sha256", "abi_sha256", "inputs_sha256", "reference_sha256",
                                  "provider_sha256", "scope_sha256", "roi_sha256"])
def test_mismatched_source_or_execution_domain_refuses(field):
    with pytest.raises(ValueError, match="exact source/program/provider/ROI"):
        run(lambda p, r, i: observe(p, replace(i, **{field: "3" * 64})))


def test_pending_wrong_program_and_wrong_context_have_no_count_fallback():
    assert run(lambda *args: None).winner is None
    with pytest.raises(ValueError, match="exact source/program/provider/ROI"):
        run(lambda p, r, i: observe(p, i, program_sha256="0" * 64))
    with pytest.raises(ValueError, match="context/format"):
        explore_evaluated(source(), families(), facts, output="program", emission=emission,
            evaluate=lambda *args: pytest.fail("wrong context must refuse before evaluation"),
            identity=replace(identity(), context_sha256="0" * 64), context=CONTEXT)
    assert len(run(lambda *args: None, max_candidates=1).candidates) == 1


@pytest.mark.parametrize('changes,status', [
    ({'numerical_status': 'fail'}, 'numerical_fail'),
    ({'numerical_status': 'unknown'}, 'numerical_unknown'),
    ({'timing_status': 'unknown'}, 'timing_unknown'),
    ({'timing_status': 'failed'}, 'timing_failed'),
    ({'complete_roi': False}, 'timing_partial_roi'),
    ({'cycles': None}, 'timing_cycles_missing'),
])
def test_actual_unranked_observation_has_actionable_coverage_not_a_score(changes, status):
    result = run(lambda p, r, i: observe(p, i, **changes))
    assert all(row['status'] == status for row in result.outcomes)
    assert all(not row['evaluation']['eligible'] for row in result.outcomes)
    assert result.summary()['performance_qualification'] is False and result.winner is None


def test_explicit_incumbent_distinguishes_no_change_dedup_and_improvement():
    baseline = run(lambda *args: None, max_candidates=1).candidates[0].emission
    incumbent = observe(baseline, identity())
    calls = []
    def evaluate(program, receipt, domain):
        calls.append(program)
        return observe(program, domain)
    result = run(evaluate, incumbent=incumbent)
    assert len(calls) == 1  # incumbent's exact artifact and permutation duplicates are not re-evaluated
    assert [row['status'] for row in result.outcomes].count('same_as_incumbent') == 2
    assert [row['status'] for row in result.outcomes].count('cycle_improvement') == 2
    assert sum(row['duplicate_of'] is not None for row in result.outcomes) == 2
    unchanged = next(row for row in result.outcomes if row['same_as_incumbent'])
    assert unchanged['transformation']['branch_ir'] == 'unchanged'
    improved = next(row for row in result.outcomes if row['status'] == 'cycle_improvement')
    assert improved['transformation']['branch_ir'] == 'changed'
    assert improved['comparison']['delta_cycles'] == -100
    assert result.ranking['selection'] == 'candidate_improvement' and result.winner is not None


def test_complete_regression_retains_incumbent_without_instruction_fallback():
    baseline = run(lambda *args: None, max_candidates=1).candidates[0].emission
    incumbent = observe(baseline, identity(), cycles=50)
    result = run(lambda p, r, i: observe(p, i), incumbent=incumbent)
    assert result.winner is None and result.ranking['selection'] == 'incumbent_retained'
    assert any(row['status'] == 'cycle_regression' for row in result.outcomes)
    assert result.incumbent is incumbent
    with pytest.raises(ValueError, match='incumbent differs'):
        run(lambda *args: pytest.fail('domain must fail before evaluation'),
            incumbent=replace(incumbent, identity=replace(identity(), provider_sha256='3' * 64)))


def test_pending_and_adapter_infrastructure_remain_distinct():
    result = run(lambda *args: None)
    assert all(row['status'] == 'evaluation_pending' for row in result.outcomes)
    assert all('transformation' in row for row in result.outcomes)
    def unavailable(*args):
        raise RuntimeError('evaluation transport unavailable')
    with pytest.raises(RuntimeError, match='transport unavailable'):
        run(unavailable)


def test_unknown_incumbent_does_not_freeze_new_evaluation_and_tie_retains_control():
    baseline = run(lambda *args: None, max_candidates=1).candidates[0].emission
    unknown = observe(baseline, identity(), timing_status='unknown', cycles=None)
    calls = []
    result = run(lambda p, r, i: (calls.append(p), observe(p, i))[1], incumbent=unknown)
    assert len(calls) == 2 and result.winner is None
    assert all(row['status'] == 'cycle_ranked_incumbent_unranked' for row in result.outcomes)
    tied = run(lambda p, r, i: observe(p, i, cycles=100), incumbent=observe(baseline, identity(), cycles=100))
    assert tied.winner is None and any(row['status'] == 'cycle_tie' for row in tied.outcomes)
