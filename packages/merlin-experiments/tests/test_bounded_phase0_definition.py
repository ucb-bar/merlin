"""Bounded definition generation keeps the denominator and strict lane obligations."""

import json
import runpy
from types import SimpleNamespace

import pytest
import yaml


@pytest.fixture
def definition_inputs(tmp_path, monkeypatch):
    from merlin.common.paths import repo_root
    from merlin.targetgen import functional_truth

    # The strict definition commits the independent truth lanes before any
    # compiler placement. These synthetic cases carry no schema, so the truth
    # commitment is stubbed: "support" cases are device obligations, the rest
    # are host guards.
    def expectation(case):
        device = case["overload"] == "support"
        return dict(lane="device" if device else "host-guard", truth="must_accelerate" if device else "not_native")

    monkeypatch.setattr(functional_truth, "expectation", expectation)

    root = repo_root()
    generate = runpy.run_path(str(root / "build_tools/scripts/make_bounded_phase0.py"))["generate"]
    repository = tmp_path / "repository"
    target = repository / "examples/gemmini/target"
    target.mkdir(parents=True)
    (target / "descriptor.yaml").write_text("target: gemmini\nworkload_spec: {applications: [example]}\n")
    cases = [dict(case_id=f"support::{n}", overload="support") for n in range(4)]
    cases += [dict(case_id=f"other::{n}", overload=f"other{n % 3}") for n in range(6)]
    suite = tmp_path / "suite.json"
    suite.write_text(json.dumps({"selected_cases": cases, "denominator_sha256": "original"}))
    support = tmp_path / "support.txt"
    support.write_text("support\n")
    args = SimpleNamespace(
        suite=suite,
        captures=tmp_path / "captures",
        limit=None,
        output=tmp_path / "definition",
        descriptor=target / "descriptor.yaml",
        software_spec=target / "software-spec.yaml",
        host_capabilities=target / "host-capabilities.yaml",
        execution_provider=tmp_path / "provider.py",
        capability_contract=target / "contract.yaml",
        performance_template=tmp_path / "performance.yaml",
        device_package=tmp_path / "provider",
        facts=tmp_path / "facts.json",
        overlay=tmp_path / "overlay.json",
        workers=2,
    )
    return generate, args


def test_strict_definition_retains_every_case_and_derives_both_lanes(definition_inputs):
    generate, args = definition_inputs
    generate(args)
    recipe = yaml.safe_load((args.output / "recipe.yaml").read_bytes())
    cohorts = recipe["core_aten"]["cohorts"]
    assert all(row["lane"] == "truth" for rows in cohorts.values() for row in rows)
    assert cohorts["hidden"][0]["select"] == ["support::0"]
    assert cohorts["public"][0]["select"] == ["support::1", "support::2", "support::3"]
    assert cohorts["host_guard"][0]["select"] == sorted(f"other::{n}" for n in range(6))
    all_ids = [key for rows in cohorts.values() for row in rows for key in row["select"]]
    assert len(all_ids) == len(set(all_ids)) == 10
    assert recipe["core_aten"]["workers"] == 2
    cases = json.loads((args.output / "cases.json").read_bytes())
    assert {case["overload"] for case in cases["cases"]} == {"support", "other0", "other1", "other2"}
    assert cases["denominator_sha256"] == "original"
    definition = json.loads((args.output / "definition.json").read_bytes())
    assert definition["lane_policy"] == "functional_truth"
    assert definition["device_count"] == 4 and definition["host_guard_count"] == 6
    expectations = json.loads((args.output / "expectations.json").read_bytes())
    assert sorted(expectations) == sorted(all_ids)
    assert "host_exceptions" not in definition and "support_count" not in definition


def test_prefix_limit_retains_disjoint_visibility_without_placement_selection(definition_inputs):
    generate, args = definition_inputs
    args.limit = 3
    generate(args)
    cases = json.loads((args.output / "cases.json").read_bytes())["cases"]
    assert [case["case_id"] for case in cases] == ["support::0", "support::1", "support::2"]
