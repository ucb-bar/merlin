"""Synthetic experiment design checks, not hardware qualification."""

from data_helpers import sample
from merlin_timing import archive as A
from merlin_timing import contracts as C
from merlin_timing import design as D
from merlin_timing.cli import main


def _plan():
    data = sample()
    records = data.pop("records")
    data["records"] = []
    return {
        "schema": A.PLAN,
        "archive_sha256": C.digest("synthetic-archive"),
        "dataset_template": data,
        "observations": [{"case_id": row["case_id"], "label": row["label"]} for row in records],
    }


def test_design_check_requires_measured_separated_variants_across_families():
    plan = _plan()
    result = D.check(plan)
    assert result["ready_for_native_qualification_attempt"] is True
    assert result["counts"]["measured_separated_held_pairs"] >= 20
    assert result["counts"]["families_with_eligible_pairs"] == 3

    # A repeat improves noise evidence, but must not create another variant pair.
    plan["observations"].append(dict(plan["observations"][-1]))
    assert D.check(plan)["counts"]["measured_separated_held_pairs"] == result["counts"]["measured_separated_held_pairs"]

    for observation in plan["observations"]:
        if observation["case_id"].startswith("test-") and not observation["case_id"].endswith("-0"):
            observation["label"] = dict(observation["label"], measured_now=False)
    result = D.check(plan)
    assert result["ready_for_native_qualification_attempt"] is False
    assert result["counts"]["measured_separated_held_pairs"] == 0
    assert result["checks"]["all_held_labels_available"] is False


def test_design_check_cli_writes_diagnostic_receipt(tmp_path):
    source, output = tmp_path / "plan.json", tmp_path / "design.json"
    C.write(source, _plan())
    assert main(["archive-design-check", "--plan", str(source), "--output", str(output)]) == 0
    assert C.read(output)["ready_for_native_qualification_attempt"] is True


def test_v2_missing_counter_review_does_not_erase_measured_pairs():
    plan = _plan()
    plan["schema"] = A.PLAN_V2
    for row in plan["observations"]:
        row["scope_binding"] = {"synthetic": True}
    original = D.check(plan)
    assert original["ready_for_native_qualification_attempt"]
    plan["observations"][0]["scope_binding"] = None
    result = D.check(plan)
    assert not result["ready_for_native_qualification_attempt"]
    assert not result["checks"]["all_counter_reviews_available"]
    assert result["counts"] == original["counts"]
