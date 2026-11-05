"""Simulated cycle fixtures exercise learning, never claim measured performance."""

import json
from dataclasses import replace

import pytest
from test_shape_policy import PATHS, case, execute

from merlin.transforms.pass_exploration import ExplorationGeometry, ValidationPartition
from merlin.transforms.pipeline_policy import (
    DimensionRef,
    FeaturePredicate,
    PipelineVariant,
    PropertyPredicate,
    RatioFeature,
    ShapePredicate,
)
from merlin.transforms.shape_policy import (
    ShapeScheduleClass,
    assess_shape_policy,
    propose_shape_policies,
)


def test_propose_validate_and_execute_unseen_current_ir_without_repeating_complete_panel():
    training = (case(3, cycles=110), case(4), case(6))
    heldout = case(5, cycles=75)
    design = ValidationPartition(tuple(c.geometry for c in training), (heldout.geometry,))
    proposals = propose_shape_policies(training, PATHS, design=design, incumbent="original", axes=((0, 0),))
    assert proposals
    proposal = proposals[0]
    assert proposal.shape_class.predicates == (ShapePredicate(0, 0, at_least=4),)
    assert proposal.report["training_only"] and not proposal.report["performance_qualified"]
    pending = proposal.requests(training)
    assert len(pending["pending_evaluations"]) == 1
    assert pending["pending_evaluations"][0]["geometry_sha256"] == heldout.geometry.identity
    assert pending["pending_evaluations"][0]["alternatives"] == ["original", "fold"]
    assert pending["missing_boundary_sources"] == []
    assert proposal.requests((*training, heldout))["pending_evaluations"] == []
    assessed = proposal.assess((*training, heldout))
    assert execute(assessed, m=8).receipt["selected_variant"] == "fold"
    assert "arith.addi" not in execute(assessed, m=8).artifact
    assert "arith.addi" in execute(assessed, m=8, resources=32).artifact  # ordinary legality still applies
    assert "arith.addi" in execute(assessed, m=8, domain=False).artifact
    assert proposal.assess(training).selected[0][1] == "original"
    # Advisory report dictionaries cannot loosen the frozen validation limit.
    proposal.report["max_regression_ratio"] = 100
    assert proposal.assess((*training, case(5, cycles=105))).selected[0][1] == "original"


def test_training_selects_path_and_guard_heldout_can_only_veto_never_retune():
    training = (case(3, cycles=110), case(4), case(6))
    heldout = case(5)
    design = ValidationPartition(tuple(c.geometry for c in training), (heldout.geometry,))
    proposal = propose_shape_policies(training, PATHS, design=design, incumbent="original", axes=((0, 0),))[0]
    for invalid in (case(5, cycles=105), case(5, unknown=True), replace(heldout, implementation_sha256="6"*64)):
        assessed = proposal.assess((*training, invalid))
        assert assessed.selected[0][1] == "original"
        assert bool(proposal.requests((*training, invalid))["pending_evaluations"]) == (
            invalid.observations[1][1].cycles is None or invalid.implementation_sha256 != "5"*64)
    with pytest.raises(ValueError, match="TRAINING"):
        propose_shape_policies((*training, heldout), PATHS, design=design, incumbent="original", axes=((0, 0),))
    with pytest.raises(ValueError, match="training feedback changed"):
        proposal.assess((training[0], case(4, cycles=60), training[2], heldout))
    assert propose_shape_policies((case(3, unknown=True), case(4, unknown=True)), PATHS,
        design=ValidationPartition((case(3).geometry, case(4).geometry), (heldout.geometry,)),
        incumbent="original", axes=((0, 0),)) == ()


def test_heldout_cannot_choose_a_different_training_runner_up():
    paths = (*PATHS, PipelineVariant("other", ("normalize", "emit")))

    def add_other(row, cycles):
        return replace(row, observations=(*row.observations,
                       ("other", replace(row.observations[1][1], cycles=cycles))))

    training = tuple(add_other(c, 115 if c.geometry.operand_shapes == ((3,),) else 90)
                     for c in (case(3, cycles=110), case(4), case(6)))
    heldout = add_other(case(5, cycles=110), 60)
    design = ValidationPartition(tuple(c.geometry for c in training), (heldout.geometry,))
    proposal = propose_shape_policies(training, paths, design=design, incumbent="original", axes=((0, 0),))[0]
    assert proposal.candidate == "fold"
    assessed = proposal.assess((*training, heldout))
    assert assessed.selected[0][1] == "original"
    assert {v.name for v in assessed.alternatives} == {"original", "fold"}


def rectangle(m, n, cycles=80, *, control_only=False):
    row = case(m, cycles=cycles)
    facts = replace(row.facts, shapes=((m, n),))
    return replace(row, geometry=ExplorationGeometry("integer", facts.shapes), facts=facts,
                   observations=row.observations[:1] if control_only else row.observations)


def test_two_dimensional_conjunction_requests_fixed_adjacent_boundaries_then_reuses_them():
    training = (rectangle(2, 2, 110), rectangle(4, 2), rectangle(6, 4),
                rectangle(4, 16, 120), rectangle(2, 16, 120))
    heldout = rectangle(5, 3)
    design = ValidationPartition(tuple(c.geometry for c in training), (heldout.geometry,))
    proposals = propose_shape_policies(training, PATHS, design=design, incumbent="original",
                                     axes=((0, 0), (0, 1)))
    proposal = proposals[0]
    assert len(proposal.shape_class.predicates) == 2
    missing = proposal.requests(training)["missing_boundary_sources"]
    assert {(r["axis"], r["value"]) for r in missing} == {(0, 3), (1, 5)}
    boundaries = (rectangle(3, 4, control_only=True), rectangle(4, 5, control_only=True))
    frozen = proposal.freeze_boundaries(tuple(c.geometry for c in boundaries))
    assert frozen.requests()["missing_boundary_sources"] == []
    assert frozen.assess((*training, heldout, *boundaries)).selected[0][1] == "fold"
    assert frozen.requests((*training, heldout, *boundaries))["pending_evaluations"] == []
    with pytest.raises(ValueError, match="never observations"):
        proposal.freeze_boundaries(boundaries)
    with pytest.raises(ValueError, match="fixed proposal request"):
        proposal.freeze_boundaries((rectangle(99, 99).geometry,))
    with pytest.raises(ValueError, match="distinct"):
        proposal.freeze_boundaries((training[0].geometry,))


def test_outside_class_candidate_legality_or_cycles_do_not_veto_guarded_path():
    training = (case(3, resources=None, unknown=True), case(4), case(6))
    heldout = case(5)
    design = ValidationPartition(tuple(c.geometry for c in training), (heldout.geometry,))
    proposal = propose_shape_policies(training, PATHS, design=design, incumbent="original", axes=((0, 0),))[0]
    assert proposal.assess((*training, heldout)).selected[0][1] == "fold"
    assert proposal.requests(training)["pending_evaluations"][0]["geometry_sha256"] == heldout.geometry.identity


def test_known_false_class_conjunct_dominates_irrelevant_unknown_property():
    rows = tuple(replace(c, facts=replace(c.facts, capabilities=(("batching", None if i == 0 else True),)))
                 for i, c in enumerate((case(2), case(3), case(4), case(5))))
    design = ValidationPartition(tuple(c.geometry for c in rows[:3]), (rows[3].geometry,))
    group = ShapeScheduleClass("guarded", (ShapePredicate(0, 0, at_least=4),
                              PropertyPredicate("capability", "batching")), "original", design)
    assert assess_shape_policy((group,), PATHS, rows, fallback="original").selected[0][1] == "fold"


def test_search_budget_and_original_path_restrictions_are_explicit():
    training = (case(3), case(4))
    design = ValidationPartition(tuple(c.geometry for c in training), (case(5).geometry,))
    result = propose_shape_policies(training, PATHS, design=design, incumbent="original",
                                    axes=((0, 0),), max_search_steps=1)
    assert len(result) <= 1
    if result:
        assert result[0].report["search_steps"] == 1 and result[0].report["search_truncated"]
    for invalid in ({"axes": ((0, 0), (0, 0))}, {"max_thresholds": 99}, {"max_proposals": True}):
        with pytest.raises(ValueError, match="bounded"):
            propose_shape_policies(training, PATHS, design=design, incumbent="original",
                                   **({"axes": ((0, 0),)} | invalid))


def test_explicit_rational_feature_assessment_keeps_exact_boundary_not_integer_truncation():
    cases = (rectangle(5, 4), rectangle(6, 4), rectangle(7, 4))
    design = ValidationPartition((cases[0].geometry, cases[1].geometry), (cases[2].geometry,))
    feature = FeaturePredicate(RatioFeature(DimensionRef(0, 0), DimensionRef(0, 1)), at_least=(3, 2))
    group = ShapeScheduleClass("aspect", (feature,), "original", design)
    assessed = assess_shape_policy((group,), PATHS, cases, fallback="original")
    assert assessed.selected[0][1] == "fold"
    # Exact 3/2 is absent: having 5/4 and 7/4 cannot replace the boundary.
    missing = replace(group, design=ValidationPartition((cases[0].geometry,), (cases[2].geometry,)))
    unsupported = assess_shape_policy((missing,), PATHS, (cases[0], cases[2]), fallback="original")
    assert unsupported.selected[0][1] == "original"
    assert "nearest_observed_below" in json.dumps(unsupported.report)


def test_self_contained_example_consumes_the_selected_shared_pass_on_new_source():
    import runpy
    from pathlib import Path

    example = Path(__file__).resolve().parents[3] / "examples/compiler/shape_policy_proposals.py"
    report = runpy.run_path(str(example))["main"]()
    assert report["observations"] == "simulated; no performance claim"
    assert report["selected_passes"] == ["normalize", "emit"]
    assert len(report["pending_validation"]["pending_evaluations"]) == 1


def test_proposed_aspect_ratio_keeps_fraction_and_executes_existing_registered_path():
    training = (rectangle(5, 4, 110), rectangle(6, 4), rectangle(8, 4))
    held = rectangle(7, 4, 75)
    design = ValidationPartition(tuple(c.geometry for c in training), (held.geometry,))
    feature = RatioFeature(DimensionRef(0, 0), DimensionRef(0, 1))
    proposal = propose_shape_policies(training, PATHS, design=design, incumbent='original', features=(feature,))[0]
    assert proposal.shape_class.predicates == (FeaturePredicate(feature, at_least=(3, 2)),)
    assert proposal.report['training_only'] and not proposal.report['performance_qualified']
    assert proposal.requests(training)['missing_boundary_sources'] == []
    assessed = proposal.assess((*training, held))
    assert assessed.selected[0][1] == 'fold'
    from test_pipeline_policy import registry, source
    from test_shape_policy import CONTEXT
    from merlin.transforms.pipeline_state import PipelineSession
    from merlin.transforms.pipeline_policy import PolicyFacts
    session = PipelineSession(source(), context=CONTEXT)
    facts = PolicyFacts(session.stamp(), shapes=((11, 4),), resources=(('private_bytes', 128),))
    emitted = assessed.execute(session, registry(), facts, output='emitted_ir', domain=proposal.domain)
    assert emitted.receipt['selected_variant'] == 'fold' and 'arith.addi' not in emitted.artifact
    assert proposal.assess((*training, rectangle(7, 4, 105))).selected[0][1] == 'original'
    with pytest.raises(ValueError, match='TRAINING'):
        propose_shape_policies((*training, held), PATHS, design=design, incumbent='original', features=(feature,))


def test_rational_requests_freeze_a_real_strict_outside_geometry_not_integer_plus_one():
    training = (rectangle(5, 4), rectangle(6, 4))
    held = rectangle(7, 5)
    feature = RatioFeature(DimensionRef(0, 0), DimensionRef(0, 1))
    proposals = propose_shape_policies(training, PATHS,
        design=ValidationPartition(tuple(c.geometry for c in training), (held.geometry,)),
        incumbent='original', features=(feature,), max_proposals=16)
    p = next(p for p in proposals if p.shape_class.predicates == (FeaturePredicate(feature, at_most=(3, 2)),))
    requests = p.requests(training)['missing_boundary_sources']
    assert len(requests) == 1 and requests[0]['comparison'] == 'above'
    assert requests[0]['value'] == {'numerator': 3, 'denominator': 2}
    outside = rectangle(7, 4, control_only=True)
    frozen = p.freeze_boundaries((outside.geometry,))
    assert frozen.requests()['missing_boundary_sources'] == []
    assert frozen.assess((*training, held, outside)).selected[0][1] == 'fold'


def resource_case(m, *, tile=4, cycles=80):
    row = case(m, cycles=cycles)
    return replace(row, facts=replace(row.facts, resources=(*row.facts.resources, ('tile_extent', tile))))


def test_resource_tile_requests_need_source_facts_without_inferred_tile_capacity():
    from merlin.transforms.pipeline_policy import ResourceRef, TileFeature
    feature = TileFeature(DimensionRef(0, 0), ResourceRef('tile_extent'))
    training = (resource_case(5), resource_case(6))
    held = resource_case(7)
    proposals = propose_shape_policies(training, PATHS,
        design=ValidationPartition(tuple(c.geometry for c in training), (held.geometry,)),
        incumbent='original', features=(feature,), max_proposals=16)
    p = next(p for p in proposals if p.shape_class.predicates == (FeaturePredicate(feature, at_most=(2, 1)),))
    request = p.requests(training)['missing_boundary_sources'][0]
    assert request['value'] == 3 and request['feature']['tile']['resource'] == 'tile_extent'
    outside = resource_case(9)
    with pytest.raises(ValueError, match='source/resource facts'):
        p.freeze_boundaries((outside.geometry,))
    frozen = p.freeze_boundaries((outside.geometry,), facts=(outside.facts,))
    assert frozen.requests()['missing_boundary_sources'] == []
    assert frozen.assess((*training, held, outside)).selected[0][1] == 'fold'
    with pytest.raises(ValueError, match='facts changed'):
        frozen.assess((*training, held, resource_case(9, tile=8)))
    # Missing/zero resources do not become a denominator or a static score.
    for extent in (None, 0):
        rows = (resource_case(5, tile=extent), resource_case(6, tile=extent))
        assert propose_shape_policies(rows, PATHS,
            design=ValidationPartition(tuple(c.geometry for c in rows), (held.geometry,)),
            incumbent='original', features=(feature,)) == ()


def test_mixed_dimension_and_resource_ratio_features_are_bounded_and_training_only():
    from merlin.transforms.pipeline_policy import ResourceRef, TileFeature
    ratio = RatioFeature(DimensionRef(0, 0), ResourceRef('tile_extent'))
    tail = TileFeature(DimensionRef(0, 0), ResourceRef('tile_extent'), 'tail_fraction')
    training = (resource_case(5, cycles=110), resource_case(6), resource_case(10))
    held = resource_case(14)
    design = ValidationPartition(tuple(c.geometry for c in training), (held.geometry,))
    proposals = propose_shape_policies(training, PATHS, design=design, incumbent='original',
                                      axes=((0, 0),), features=(ratio, tail), max_search_steps=32)
    assert proposals and all(p.report['search_steps'] <= 32 for p in proposals)
    assert any(any(type(g) is FeaturePredicate for g in p.shape_class.predicates) for p in proposals)
    for invalid in ((ratio, ratio), ('label',), (ratio, tail, ratio, tail)):
        with pytest.raises(ValueError, match='bounded'):
            propose_shape_policies(training, PATHS, design=design, incumbent='original', features=invalid)


def test_positive_tile_count_has_no_impossible_zero_neighbor_request():
    from merlin.transforms.pipeline_policy import ResourceRef, TileFeature
    rows = (resource_case(1), resource_case(2))
    feature = TileFeature(DimensionRef(0, 0), ResourceRef('tile_extent'))
    proposals = propose_shape_policies(rows, PATHS,
        design=ValidationPartition(tuple(r.geometry for r in rows), (resource_case(3).geometry,)),
        incumbent='original', features=(feature,))
    assert proposals
    assert all(p.shape_class.predicates == (FeaturePredicate(feature, at_most=(1, 1)),) for p in proposals)
    assert all(r['value'] != 0 for p in proposals for r in p.requests()['missing_boundary_sources'])
