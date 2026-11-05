"""Conditional complete-cost arithmetic; these synthetic records qualify no timer."""

from dataclasses import asdict, replace

import pytest

from merlin.common.jsonio import canonical_sha256 as sha
from merlin.perf.component_cost import (
    COMPLETE_STAGES,
    ComponentCostRegion,
    ComponentCostScope,
    ComponentFeatureObservation,
)
from merlin.perf.offload_profitability import OffloadCostDecision, compare_complete_offload_cost
from merlin.perf.term import UNKNOWN, Bounds, PerformanceTerm, Provenance, Validity
from merlin.transforms.pipeline_state import AnalysisStamp, PipelineContext
from merlin.xdsl_dialects.lowering.global_plan import CycleInterval


def term(lower, upper, *, unit="ns", kind="measured"):
    return PerformanceTerm(
        "complete_forward",
        (lower + upper) / 2,
        unit,
        Provenance(kind, (sha([lower, upper, unit]),)),
        Validity("original full invocation and declared reset/reuse policy"),
        Bounds(lower, upper),
    )


def fingerprint(arguments):
    return sha(
        {
            **{key: asdict(arguments[key]) for key in ("original", "candidate", "scope", "stamp")},
            "regime": arguments["regime"],
            "original_selection_sha256": arguments["original_selection_sha256"],
            "candidate_selection_sha256": arguments["candidate_selection_sha256"],
            **{
                key: {regime: value.to_dict() for regime, value in arguments[key].items()}
                for key in ("original_costs", "candidate_costs")
            },
        }
    )


def request(*, bounds=(70, 80), unit="ns", regime="warm", before=None):
    scope = ComponentCostScope(
        sha("timer and exact unit"),
        sha("frozen numerical policy"),
        sha("inputs state positions cold reset and warm ownership"),
    )

    def observation(arm):
        return ComponentFeatureObservation(
            sha([arm, "compiler"]),
            sha("original member"),
            sha("original corpus"),
            sha("hardware"),
            scope.sha256,
            sha([arm, "qualified joint cell"]),
            sha([arm, "actual executable"]),
            sha([arm, "dependencies"]),
            sha("identical inputs and carried state"),
            (sha([arm, "actual timer product"]),),
            (
                ComponentCostRegion("whole", COMPLETE_STAGES, ("observed",), {"count": 1}, accounting="inclusive"),
                ComponentCostRegion("packing", ("packing",), ("observed",), {"count": 1}, parent="whole"),
            ),
            (ComponentCostRegion("whole", COMPLETE_STAGES, ("observed",), {"count": 1}),),
            "PASS",
            artifact_files=((arm + ".elf", sha([arm, "actual executable"])),),
            legality_status="PASS",
        )

    query = dict(
        original=observation("original"),
        candidate=observation("candidate"),
        scope=scope,
        regime=regime,
        original_selection_sha256=sha("original CPU route"),
        candidate_selection_sha256=sha("exact selected structural candidate"),
        original_costs={name: term(100, 120, unit=unit) for name in ("cold", "warm")},
        candidate_costs={name: term(*bounds, unit=unit) for name in ("cold", "warm")},
        stamp=AnalysisStamp(
            sha("actual current source"),
            PipelineContext(
                sha("hardware"),
                sha("frozen numerical policy"),
                sha("resources"),
                sha("runtime state cache input and completion"),
                sha("actual shared library"),
            ),
            "tensor",
        ),
    )
    if before is not None:
        before(query)
    expected, external, calls = fingerprint(query), {"current": True}, []

    def verifier(**arguments):
        # Independent ownership is simulated here only to test the public
        # protocol. This fixture supplies no physical timing qualification.
        assert set(arguments) == set(query) - {"verify_original"}
        calls.append(True)
        if not external["current"] or fingerprint(arguments) != expected:
            raise ValueError("original measurement owner refuses changed actual evidence")

    query["verify_original"] = verifier
    return query, external, calls


@pytest.mark.parametrize("unit", ["ns", "cycle"])
@pytest.mark.parametrize(
    "bounds,status,chosen",
    [
        ((70, 80), "OPPORTUNITY", True),
        ((90, 110), "TIE_OR_OVERLAP", False),
        ((90, 100), "TIE_OR_OVERLAP", False),
        ((100, 120), "TIE_OR_OVERLAP", False),
        ((130, 150), "REGRESSION", False),
    ],
)
def test_complete_interval_dominance_preserves_original_unit_and_regime(unit, bounds, status, chosen):
    query, _, calls = request(unit=unit, bounds=bounds)
    decision = compare_complete_offload_cost(**query)
    assert calls == [True, True]
    record = decision.record()
    assert record["unit"] == unit and record["regime"] == "warm"
    assert decision.status == status and decision.choose_candidate is chosen
    assert record["accounting"]["original"]["cold"]["included"] == ["whole"]
    assert set(record["accounting"]["original"]["cold"]["covered"]) == set(COMPLETE_STAGES)
    assert record["observations"]["original"]["executable_sha256"] == query["original"].executable_sha256
    assert "legality" in record["scope_note"]
    assert decision.stamp == query["stamp"] and decision.stamp is not query["stamp"]
    assert record["original_selection_sha256"] == query["original_selection_sha256"]
    assert record["candidate_selection_sha256"] == query["candidate_selection_sha256"]


def test_native_nanoseconds_never_construct_cycle_intervals(monkeypatch):
    query, _, _ = request(unit="ns")

    def no_cycle_conversion(*args, **kwargs):
        pytest.fail("a nanosecond observation was relabeled as cycles")

    monkeypatch.setattr(CycleInterval, "__post_init__", no_cycle_conversion)
    assert compare_complete_offload_cost(**query).choose_candidate


@pytest.mark.parametrize("regime", ["cold", "warm"])
def test_explicit_policy_compares_only_the_selected_regime_but_retains_both(regime):
    def before(query):
        other = "cold" if regime == "warm" else "warm"
        query["candidate_costs"][other] = term(130, 150)

    query, _, _ = request(regime=regime, before=before)
    record = compare_complete_offload_cost(**query).record()
    assert record["choose_candidate"] and set(record["costs"]["candidate"]) == {"cold", "warm"}


@pytest.mark.parametrize("unknown", ["value", "bound", "provenance", "context", "stage", "functional"])
def test_missing_cost_accounting_or_original_context_retains_original(unknown):
    def before(query):
        value = query["candidate_costs"]["warm"]
        if unknown == "value":
            query["candidate_costs"]["warm"] = PerformanceTerm.unknown(
                value.name, value.unit, value.provenance, value.validity, "completion timer unavailable"
            )
        elif unknown == "bound":
            query["candidate_costs"]["warm"] = replace(value, bounds=Bounds(UNKNOWN, 100))
        elif unknown == "provenance":
            query["candidate_costs"]["warm"] = replace(
                value, provenance=Provenance("assumed", ("no calibrated price",))
            )
        elif unknown == "context":
            query["stamp"] = replace(query["stamp"], context=replace(query["stamp"].context, library=None))
        elif unknown == "stage":
            incomplete = ComponentCostRegion("device_only", ("device",), ("observed",), {"count": 1})
            query["candidate"] = replace(query["candidate"], warm=(incomplete,))
        else:
            query["candidate"] = replace(query["candidate"], functional_status="UNKNOWN")

    query, _, _ = request(before=before)
    decision = compare_complete_offload_cost(**query)
    assert decision.status == "UNRESOLVED" and not decision.choose_candidate
    assert decision.record()["missing"]


@pytest.mark.parametrize("gate", ["functional_status", "legality_status"])
def test_known_failed_candidate_never_acquires_a_cost_based_permission(gate):
    query, _, _ = request(before=lambda q: q.update(candidate=replace(q["candidate"], **{gate: "FAIL"})))
    decision = compare_complete_offload_cost(**query)
    assert decision.status == "REFUSAL" and not decision.choose_candidate


@pytest.mark.parametrize(
    "mutation", ["unit", "timer", "input", "member", "regime", "extra", "missing", "screening", "artifact"]
)
def test_incompatible_or_incomplete_original_records_refuse(mutation):
    def before(query):
        if mutation == "unit":
            query["candidate_costs"]["warm"] = replace(query["candidate_costs"]["warm"], unit="cycle")
        elif mutation == "timer":
            query["scope"] = replace(query["scope"], timer_sha256=sha("other timer"))
        elif mutation in ("input", "member"):
            key = {"input": "inputs_sha256", "member": "member_sha256"}[mutation]
            query["candidate"] = replace(query["candidate"], **{key: sha("other")})
        elif mutation == "regime":
            query["regime"] = "unspecified"
        elif mutation == "extra":
            query["candidate_costs"]["steady"] = query["candidate_costs"]["warm"]
        elif mutation == "missing":
            query["candidate_costs"].pop("cold")
        elif mutation == "screening":
            query["candidate"] = {"promotion": "SCREENING_ONLY"}
        else:
            query["candidate"] = replace(query["candidate"], artifact_files=())

    if mutation == "screening":
        query, _, _ = request()
        before(query)
    else:
        query, _, _ = request(before=before)
    with pytest.raises((TypeError, ValueError)):
        compare_complete_offload_cost(**query)


@pytest.mark.parametrize(
    "field,value",
    [
        ("value", 0),
        ("value", -1),
        ("value", float("inf")),
        ("value", float("nan")),
        ("bounds", Bounds(0, 100)),
        ("bounds", Bounds(1, float("inf"))),
    ],
)
def test_zero_negative_nonfinite_or_unbounded_values_do_not_become_prices(field, value):
    query, _, _ = request()
    object.__setattr__(query["candidate_costs"]["warm"], field, value)
    with pytest.raises(ValueError):
        compare_complete_offload_cost(**query)


@pytest.mark.parametrize("mutation", ["cost", "source", "hardware", "cache", "region"])
def test_original_verifier_cannot_mutate_the_cost_query(mutation):
    query, _, _ = request()

    def faulty(**arguments):
        if mutation == "cost":
            arguments["candidate_costs"]["warm"] = term(1, 2)
        elif mutation == "source":
            object.__setattr__(arguments["stamp"], "source_sha256", sha("different source"))
        elif mutation in ("hardware", "cache"):
            field = "hardware" if mutation == "hardware" else "execution"
            object.__setattr__(arguments["stamp"].context, field, sha("different context"))
        else:
            arguments["candidate"].warm[0].context["count"] += 1

    query["verify_original"] = faulty
    with pytest.raises(ValueError, match="changed during verification"):
        compare_complete_offload_cost(**query)


def test_live_decision_reopens_original_evidence_and_detaches_report_copies():
    query, external, _ = request()
    decision = compare_complete_offload_cost(**query)
    report = decision.record()
    report["choose_candidate"] = False
    assert decision.choose_candidate
    external["current"] = False
    with pytest.raises(ValueError, match="original measurement owner"):
        decision.revalidate()


def test_report_json_and_arbitrary_callbacks_cannot_construct_live_decisions():
    query, _, _ = request()
    record = compare_complete_offload_cost(**query)._record
    with pytest.raises(ValueError, match="live complete-cost comparison"):
        OffloadCostDecision(record, lambda: record, object())


@pytest.mark.parametrize("key", ["original_selection_sha256", "candidate_selection_sha256"])
@pytest.mark.parametrize("value", [None, True, "", "different route", "0" * 63])
def test_missing_or_malformed_selection_identity_refuses_before_verification(key, value):
    query, _, calls = request()
    query[key] = value
    with pytest.raises(ValueError, match="selection digests"):
        compare_complete_offload_cost(**query)
    assert not calls


@pytest.mark.parametrize("key", ["original_selection_sha256", "candidate_selection_sha256"])
def test_selection_identities_are_required_and_joined_by_original_owner(key):
    query, _, _ = request()
    missing = dict(query)
    missing.pop(key)
    with pytest.raises(TypeError, match=key):
        compare_complete_offload_cost(**missing)
    # The same source/context stamp alone does not bind one structural plan.
    query[key] = sha("different selection for the same source and context")
    with pytest.raises(ValueError, match="original measurement owner"):
        compare_complete_offload_cost(**query)


@pytest.mark.parametrize("key", ["original_selection_sha256", "candidate_selection_sha256"])
def test_live_selection_record_cannot_be_rebound_to_another_plan(key):
    import json

    query, _, _ = request()
    decision = compare_complete_offload_cost(**query)
    record = decision.record()
    record[key] = sha("another plan with the same source stamp")
    object.__setattr__(decision, "_record", json.dumps(record, sort_keys=True, separators=(",", ":")))
    with pytest.raises(ValueError, match="decision changed"):
        decision.revalidate()


@pytest.mark.parametrize("control", [{"max_regions": 1}, {"max_evidence_bytes": 1}, {"max_regions": True}])
def test_cost_work_is_bounded_before_the_external_verifier(control):
    query, _, calls = request()
    with pytest.raises(ValueError, match="budget|bounded"):
        compare_complete_offload_cost(**query, **control)
    assert not calls


@pytest.mark.parametrize("verifier", [None, lambda **kwargs: True, lambda **kwargs: {"promotion": "SCREENING_ONLY"}])
def test_flags_and_screening_json_cannot_replace_the_original_live_verifier(verifier):
    query, _, _ = request()
    query["verify_original"] = verifier
    with pytest.raises(ValueError, match="original|verifier"):
        compare_complete_offload_cost(**query)
