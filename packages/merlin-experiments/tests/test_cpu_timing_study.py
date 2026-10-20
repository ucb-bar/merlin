"""Data-only AWS study planning and exact cost arithmetic; no study is run."""

import copy
import json

import pytest
from merlin_experiments.phase2 import evaluation_fidelity as F
from merlin_experiments.phase2 import timing_study as S
from merlin_experiments.phase2.contracts import StageGateError, sha256_file

from merlin.perf.component_screen import ComponentScreenPolicy


@pytest.fixture
def study_request(tmp_path):
    pins = {}
    for name in ("target", "memory", "cache", "clock"):
        path = tmp_path / name
        path.write_text("synthetic public source " + name)
        pins[name] = {"path": str(path), "sha256": sha256_file(path)}
    path = tmp_path / "registry.json"
    path.write_text(json.dumps({"schema": F.REGISTRY_SCHEMA, "public": ["a" * 64], "protected": ["b" * 64]}))
    work = tmp_path / "assigned"
    work.mkdir()
    return {
        "schema": S.SCHEMA,
        "target": "synthetic",
        "target_descriptor": pins["target"],
        "registry": {"path": str(path), "sha256": sha256_file(path)},
        "programs": [
            {"family": "dag", "capsule": "connected", "source_sha256": "a" * 64, "scope": "component_program"}
        ],
        "engines": [
            {"engine": "spike", "identity_sha256": "c" * 64},
            {"engine": "firesim_verilator_metasim", "identity_sha256": "d" * 64},
        ],
        "configuration": {name: pins[name] for name in ("memory", "cache", "clock")},
        "final_configuration": {name: copy.deepcopy(pins[name]) for name in ("memory", "cache", "clock")},
        "resources": {
            "provider": "aws",
            "instance_id": "i-synthetic",
            "cpu_ids": [0],
            "memory_bytes": 1 << 20,
            "disk_bytes": 1 << 20,
            "work_root": str(work),
        },
        "environments": {
            name: {"LC_ALL": "C"} for name in ("compile", "feature_extraction", "spike", "firesim_verilator_metasim")
        },
        "replicates": 2,
        "max_stage_seconds": 600,
    }


def test_plan_preserves_both_regimes_all_cost_stages_and_original_screen_minima(study_request, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("planning may not load a backend, query AWS or execute a tool")

    monkeypatch.setattr("merlin.runtime.backends.base.get_backend", forbidden)
    monkeypatch.setattr(S, "_aws_instance_id", forbidden)
    result = S.plan_timing_study(study_request)
    assert result["execution_performed"] is result["provider_operations_performed"] is False
    assert result["execution_substrate"] == "CPU"
    assert result["timing_status"] == "UNKNOWN" and result["promotion"] == "NONE"
    assert len(result["matrix"]) == 2 * 2 * 2 * 7
    assert {row["regime"] for row in result["matrix"]} == {"cold", "warm"}
    assert {row["stage"] for row in result["matrix"]} == set(S._STAGES)
    assert result["screening_policy"] == ComponentScreenPolicy().to_dict()
    assert result["screening_policy"]["minimum_decided"] == 100
    assert result["screening_policy"]["minimum_slice_decided"] == 20
    assert result["screening_policy"]["minimum_predictions"] == 20


@pytest.mark.parametrize(
    "field,value",
    [("command", ["do-arbitrary-work"]), ("callback", "module.function"), ("saved_pass", {"status": "PASS"})],
)
def test_public_study_definition_cannot_supply_commands_callbacks_or_saved_passes(study_request, field, value):
    study_request[field] = value
    with pytest.raises(StageGateError, match="unsupported fields"):
        S.plan_timing_study(study_request)


def test_study_refuses_protected_programs_and_non_aws_allocation(study_request):
    study_request["programs"][0]["source_sha256"] = "b" * 64
    with pytest.raises(StageGateError, match="before compiler freeze"):
        S.plan_timing_study(study_request)
    study_request["programs"][0]["source_sha256"] = "a" * 64
    study_request["resources"]["provider"] = "local"
    with pytest.raises(StageGateError, match="explicitly allocated AWS"):
        S.plan_timing_study(study_request)


def test_stage_deadlines_stay_at_original_reduced_simulation_ceiling(study_request):
    study_request["max_stage_seconds"] = 601
    with pytest.raises(StageGateError, match="stage deadline"):
        S.plan_timing_study(study_request)


def test_mismatched_final_memory_cache_clock_is_retained_without_timing_authority(study_request, tmp_path):
    path = tmp_path / "other-clock"
    path.write_text("a different final clock")
    study_request["final_configuration"]["clock"] = {"path": str(path), "sha256": sha256_file(path)}
    result = S.plan_timing_study(study_request)
    assert result["configuration_matched"] is False
    assert result["timing_status"] == "UNKNOWN"


def test_shared_machine_allocation_is_refused_before_any_stage_execution(study_request, monkeypatch):
    monkeypatch.setattr(S, "_aws_instance_id", lambda: "i-other")
    with pytest.raises(StageGateError, match="allocated AWS instance"):
        S._allocated(study_request)


def test_plan_cli_is_nonexecuting_and_requires_a_fresh_output(study_request, tmp_path, monkeypatch):
    path, output = tmp_path / "study_request.json", tmp_path / "plan.json"
    path.write_text(json.dumps(study_request))
    monkeypatch.setattr(S, "_allocated", lambda *_: pytest.fail("dry-run must not admit execution"))
    assert S.main(["--request", str(path), "--output", str(output)]) == 0
    assert json.loads(output.read_text())["execution_performed"] is False
    with pytest.raises(SystemExit):
        S.main(["--request", str(path), "--output", str(output)])


def test_break_even_charges_setup_calibration_and_strict_threshold():
    # M includes setup, calibration and held-validation costs. Each proposal
    # saves (1-.25)*100-25 = 50 seconds, so N=20 only repays M.
    equal = S.break_even(
        proposals=20, retained_rtl_fraction=0.25, rtl_seconds=100, extraction_seconds=25, calibration_seconds=1000
    )
    assert equal["beneficial"] is False and equal["minimum_proposals"] == 21
    gain = S.break_even(
        proposals=21, retained_rtl_fraction=0.25, rtl_seconds=100, extraction_seconds=25, calibration_seconds=1000
    )
    assert gain["beneficial"] is True and gain["net_saved_seconds"] == 50
    unknown = S.break_even(
        proposals=21, retained_rtl_fraction=0.25, rtl_seconds=None, extraction_seconds=25, calibration_seconds=1000
    )
    assert unknown["status"] == "UNKNOWN" and unknown["net_saved_seconds"] is None
    never = S.break_even(
        proposals=1000, retained_rtl_fraction=1, rtl_seconds=100, extraction_seconds=25, calibration_seconds=1000
    )
    assert never["beneficial"] is False and never["minimum_proposals"] is None
