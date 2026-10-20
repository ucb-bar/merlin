"""A new public-development profile does not reinterpret old Phase1 intake."""

import pytest
from data_helpers import sample
from merlin_timing import contracts as C
from merlin_timing import validation as V
from test_event_validation import original_reports


def test_development_is_explicit_and_never_issues_original_public_feedback():
    data = sample()
    data.update(schema=C.DEVELOPMENT_DATASET, profile=C.DEVELOPMENT_PROFILE, native_profile="phase1-public")
    for case in data["cases"]:
        case["source"]["phase"] = "phase2"
    C.dataset(data)
    original_reports(data)
    for row in data["records"]:
        raw = row["event_report"]["report"]
        raw["profile"] = "phase1-public"
        row["event_report"]["sha256"] = C.digest(raw)
    report = V.validate(data, prediction_kind="event_timing")
    assert report["qualified"]  # A synthetic arithmetic fixture only.
    assert report["public_feedback_eligible"] is False


@pytest.mark.parametrize(
    "schema,profile", [(C.DATASET, C.DEVELOPMENT_PROFILE), (C.DEVELOPMENT_DATASET, "phase1-public")]
)
def test_versions_never_silently_reinterpret_old_profiles(schema, profile):
    data = sample()
    data.update(schema=schema, profile=profile)
    with pytest.raises(C.ContractError, match="unsupported dataset"):
        C.dataset(data)


@pytest.mark.parametrize("visibility,phase", [("private", "phase2"), ("hidden", "phase1"), ("public", "historical")])
def test_public_development_profile_refuses_private_or_ambiguous_sources(visibility, phase):
    data = sample()
    data.update(schema=C.DEVELOPMENT_DATASET, profile=C.DEVELOPMENT_PROFILE, native_profile="phase1-public")
    data["cases"][0]["source"].update(visibility=visibility, phase=phase)
    with pytest.raises(C.ContractError, match="visibility/phase"):
        C.dataset(data)


def test_original_phase1_profile_still_refuses_phase2_training_labels():
    data = sample()
    data["cases"][0]["source"]["phase"] = "phase2"
    with pytest.raises(C.ContractError, match="visibility/phase"):
        C.dataset(data)


@pytest.mark.parametrize("profile", ["historical-private", "foreign", None])
def test_development_native_profile_cannot_replace_the_original_public_model(profile):
    data = sample()
    data.update(schema=C.DEVELOPMENT_DATASET, profile=C.DEVELOPMENT_PROFILE, native_profile=profile)
    with pytest.raises(C.ContractError, match="original public native profile"):
        C.dataset(data)
