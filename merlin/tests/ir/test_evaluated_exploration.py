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
