import json
from dataclasses import replace
from hashlib import sha256

import pytest

from merlin.compile.tuning_knobs import MAX_KNOBS, Knob, KnobError, KnobSettings, bounded_settings, declare
from merlin.transforms.evaluated_exploration import CompletedObservation, ExplorationIdentity
from merlin.transforms.pass_exploration import EmittedProgram, MappingProposalKey
from merlin.transforms.tuning_selection import KnobTrial, select_settings, unique_emissions


def settings(tmp_path, **options):
    declare(
        tmp_path,
        {
            "order": Knob(("rows", "columns"), "rows"),
            "buffers": Knob((1, 2), 1),
            "optional": Knob((True, 1), True),
        },
    )
    return bounded_settings(tmp_path, **options)


def identity():
    return replace(
        ExplorationIdentity(*("1" * 64 for _ in range(10)), "elf", "modeled_roi"),
        facts_sha256="3" * 64,
        library_sha256="4" * 64,
    )


def domain(**changes):
    return replace(
        MappingProposalKey(
            "2" * 64, ((4, 16), (16, 8)), ("row", "column"), ("i8", "i8"), ("raw",), "3" * 64, "4" * 64, "5" * 64
        ),
        **changes,
    )


def trial(setting, data, cycles=None, **changes):
    emitted = EmittedProgram(data, "elf", "6" * 64)
    observed = None
    if cycles is not None:
        observed = CompletedObservation(
            identity(), sha256(data).hexdigest(), "7" * 64, "pass", "complete", True, cycles
        )
        observed = replace(observed, **changes)
    return KnobTrial(setting, emitted, observed)


def test_bounded_subset_pairs_preserve_control_and_never_write_settings(tmp_path):
    original = settings(tmp_path)
    before = (tmp_path / "tuning_settings.json").read_bytes()
    options = bounded_settings(tmp_path, knob_names=("buffers", "order"), max_changed_knobs=2)
    assert [option.changes for option in options] == [
        (),
        (("order", "columns"),),
        (("buffers", 2),),
        (("order", "columns"), ("buffers", 2)),
    ]
    assert all(option.values["optional"] is True for option in options)
    assert (tmp_path / "tuning_settings.json").read_bytes() == before
    assert bounded_settings(tmp_path, max_candidates=1) == original[:1]
    assert bounded_settings(tmp_path, knob_names=()) == original[:1]
    assert options[0].baseline and not options[1].baseline
    changed = options[0].values
    changed["order"] = "tampered"
    assert options[0].values["order"] == "rows"
    # Declared boolean True and integer1 remain distinct candidates.
    assert original[-1].values["optional"] == 1 and type(original[-1].values["optional"]) is int
    assert len({option.identity for option in original}) == len(original)


def test_absent_defaults_preserved_and_invalid_subset_or_present_value_refuses(tmp_path):
    settings(tmp_path)
    with pytest.raises(KnobError, match="subset"):
        bounded_settings(tmp_path, knob_names=("missing",))
    with pytest.raises(KnobError, match="subset"):
        bounded_settings(tmp_path, knob_names=("order", "order"))
    (tmp_path / "tuning_settings.json").write_text('{"order":"rows"}')
    options = bounded_settings(tmp_path, knob_names=("buffers",))
    assert options[0].values == {"order": "rows"}
    assert [option.changes for option in options] == [(), (("buffers", 1),), (("buffers", 2),)]
    (tmp_path / "tuning_settings.json").write_text('{"order":"invalid"}')
    with pytest.raises(KnobError, match="present baseline"):
        bounded_settings(tmp_path)


def test_actual_byte_dedup_and_cycle_choice_keep_settings_proposal_scope(tmp_path):
    options = settings(tmp_path)
    trials = (
        trial(options[0], b"baseline", 100),
        trial(options[1], b"fast", 70),
        trial(options[2], b"fast"),
        trial(options[3], b"baseline"),
    )
    assert unique_emissions(trials, identity=identity()) == trials[:2]
    selected = select_settings(domain(), identity=identity(), trials=trials)
    assert selected.selected is options[1] and selected.status == "cycle_improvement"
    assert selected.outcomes[2]["duplicate_of"] == options[1].identity
    assert selected.outcomes[2]["ranked_cycles"] == 70
    record = selected.record()
    assert record["mapping_key"] == domain().identity and not record["performance_qualification"]
    assert "revalidation" in record["reuse"]
    different_layout = domain(layouts=("column", "column"))
    assert (
        select_settings(different_layout, identity=identity(), trials=trials).record()["mapping_key"]
        != record["mapping_key"]
    )


@pytest.mark.parametrize(
    "changes",
    [
        dict(timing_status="unknown", cycles=None),
        dict(complete_roi=False),
        dict(cycles=None),
        dict(numerical_status="fail"),
    ],
)
def test_unknown_partial_failed_feedback_cannot_replace_passing_control(tmp_path, changes):
    options = settings(tmp_path)
    baseline = trial(options[0], b"baseline", 100)
    candidate = trial(options[1], b"candidate", 1)
    candidate = replace(candidate, observation=replace(candidate.observation, **changes))
    failed = KnobTrial(options[2], error="link refused")
    selected = select_settings(domain(), identity=identity(), trials=(baseline, candidate, failed))
    assert selected.selected is options[0] and selected.status == "baseline_retained"
    assert selected.outcomes[1]["ranked_cycles"] is None
    assert selected.outcomes[2]["status"] == "failed"


def test_unknown_control_ties_regressions_and_conflicting_duplicates_keep_control(tmp_path):
    options = settings(tmp_path)
    baseline = trial(options[0], b"baseline", 100)
    for cycles in (100, 120):
        selected = select_settings(
            domain(), identity=identity(), trials=(baseline, trial(options[1], b"candidate", cycles))
        )
        assert selected.selected is options[0]
    unknown = replace(baseline, observation=replace(baseline.observation, cycles=None, timing_status="unknown"))
    selected = select_settings(domain(), identity=identity(), trials=(unknown, trial(options[1], b"fast", 1)))
    assert selected.selected is options[0] and selected.status == "baseline_unranked"
    selected = select_settings(
        domain(), identity=identity(), trials=(baseline, trial(options[1], b"same", 20), trial(options[2], b"same", 21))
    )
    assert selected.selected is options[0]
    assert all(row["status"] == "conflicting_feedback" for row in selected.outcomes[1:])


def test_explicit_decision_margin_avoids_tiny_changes_without_changing_ranking(tmp_path):
    options = settings(tmp_path)
    trials = (trial(options[0], b"baseline", 10000), trial(options[1], b"candidate", 9999))
    assert select_settings(domain(), identity=identity(), trials=trials).selected is options[1]
    selected = select_settings(domain(), identity=identity(), trials=trials, minimum_relative_improvement=0.01)
    assert selected.selected is options[0] and selected.ranking["ranked"][0][1] == (9999,)
    assert selected.record()["ranking"]["minimum_relative_improvement"] == 0.01
    at_margin = (trials[0], trial(options[1], b"candidate", 9900))
    assert select_settings(domain(), identity=identity(), trials=at_margin,
                           minimum_relative_improvement=0.01).selected is options[1]
    for invalid in (True, -0.01, 1, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="tolerance"):
            select_settings(domain(), identity=identity(), trials=trials, minimum_relative_improvement=invalid)


def test_wrong_program_domain_library_or_baseline_order_refuse(tmp_path):
    options = settings(tmp_path)
    baseline = trial(options[0], b"baseline", 100)
    candidate = trial(options[1], b"candidate", 70)
    with pytest.raises(ValueError, match="actual emitted bytes"):
        replace(candidate, observation=replace(candidate.observation, program_sha256="0" * 64))
    with pytest.raises(ValueError, match="source/provider/ROI"):
        select_settings(
            domain(),
            identity=identity(),
            trials=(
                baseline,
                replace(
                    candidate,
                    observation=replace(candidate.observation, identity=replace(identity(), inputs_sha256="9" * 64)),
                ),
            ),
        )
    with pytest.raises(ValueError, match="compiler libraries"):
        select_settings(domain(library_sha256="9" * 64), identity=identity(), trials=(baseline, candidate))
    with pytest.raises(ValueError, match="baseline, first"):
        select_settings(domain(), identity=identity(), trials=(candidate, baseline))


def test_complete_existing_bundle_is_independent_of_generated_search_limit(tmp_path):
    declare(tmp_path, {f"option_{index}": Knob((False, True), False) for index in range(6)})
    choices = bounded_settings(tmp_path, max_candidates=32, max_changed_knobs=2)
    assert all(len(choice.changes) <= 2 for choice in choices)
    values = {f"option_{index}": True for index in range(6)}
    bundle = KnobSettings(json.dumps(values, sort_keys=True, separators=(",", ":")), tuple(values.items()))
    selected = select_settings(
        domain(),
        identity=identity(),
        trials=(trial(choices[0], b"baseline", 100), trial(bundle, b"existing-bundle", 90)),
    )
    assert selected.selected is bundle and len(selected.selected.changes) == 6
    with pytest.raises(KnobError, match="one or two changed knobs"):
        bounded_settings(tmp_path, max_changed_knobs=3)
    upper = {f"option_{index}": True for index in range(MAX_KNOBS)}
    KnobSettings(json.dumps(upper, sort_keys=True, separators=(",", ":")), tuple(upper.items()))
    upper["one_extra"] = True
    with pytest.raises(KnobError, match="distinct changes"):
        KnobSettings(json.dumps(upper, sort_keys=True, separators=(",", ":")), tuple(upper.items()))
    with pytest.raises(KnobError, match="complete immutable values"):
        replace(bundle, changes=(("option_0", False),))
