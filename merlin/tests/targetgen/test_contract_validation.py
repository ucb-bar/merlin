"""Contract loading + validation: the five plans validate, malformed input diagnoses."""
from __future__ import annotations

from merlin.validation.load import PLAN_FILES, REQUIRED_PLANS, load_all_plans
from merlin.validation.validate import validate_plan, validate_target_repo
from merlin.targetgen import pipeline
from merlin.targetgen.validate import validate_plans


def _build(out):
    return pipeline.build(
        target_name="toy_npu", out=out,
        emit=["xdsl", "mlir", "zephyr", "llvm-plan", "runtime"],
    )


def test_generated_repo_plans_load_and_validate(tmp_path):
    result = _build(tmp_path / "repo")
    plans = load_all_plans(result.out)
    assert set(plans) == set(PLAN_FILES)
    assert validate_target_repo(result.out) == []


def test_validate_plans_on_synthesized_dicts(tmp_path):
    result = _build(tmp_path / "repo")
    assert validate_plans(result.plans) == []


def test_missing_required_field_is_reported():
    bad = {"name": "x"}  # missing version, capabilities, ...
    problems = validate_plan(bad, "target_contract")
    assert problems
    assert any("version" in p for p in problems)


def test_non_mapping_is_reported():
    problems = validate_plan(["not", "a", "mapping"], "dialect_plan")
    assert problems
    assert any("mapping" in p.lower() for p in problems)


def test_missing_plan_file_is_reported(tmp_path):
    # Empty dir: every REQUIRED plan file is missing (dialect_plan is optional, so it is not flagged).
    problems = validate_target_repo(tmp_path)
    assert len(problems) >= len(REQUIRED_PLANS)
    assert any("missing" in p for p in problems)
