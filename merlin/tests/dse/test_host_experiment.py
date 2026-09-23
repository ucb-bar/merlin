"""Methodology contract for the K1 CPU-host four-arm experiment."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import shutil
import sys
import threading

import pytest
import yaml

from merlin.common.artifacts import Artifact
from merlin.common.paths import repo_root
from merlin.compare.host_experiment import (
    HostExperimentSpec,
    HostPreflight,
    _calibration_private_capsule,
    _calibration_sample,
    _grader_package_tree_identity,
    _protocol_design_replacement_valid,
)


SPEC = repo_root() / "merlin/experiments/cpu_host_compiler_v0/experiment.yaml"
_PREFLIGHT_SPEC = importlib.util.spec_from_file_location(
    "cpu_host_preflight_under_test", SPEC.with_name("preflight.py"))
assert _PREFLIGHT_SPEC is not None and _PREFLIGHT_SPEC.loader is not None
_PREFLIGHT = importlib.util.module_from_spec(_PREFLIGHT_SPEC)
_PREFLIGHT_SPEC.loader.exec_module(_PREFLIGHT)
freeze_protocol = _PREFLIGHT.freeze_protocol
_COMPLETE_SPEC = importlib.util.spec_from_file_location(
    "cpu_host_complete_under_test", SPEC.with_name("complete_campaign.py"))
assert _COMPLETE_SPEC is not None and _COMPLETE_SPEC.loader is not None
_COMPLETE = importlib.util.module_from_spec(_COMPLETE_SPEC)
_COMPLETE_SPEC.loader.exec_module(_COMPLETE)
complete_campaign = _COMPLETE.complete_campaign
_LAUNCH_SPEC = importlib.util.spec_from_file_location(
    "cpu_host_launch_under_test", SPEC.with_name("launch.py"))
assert _LAUNCH_SPEC is not None and _LAUNCH_SPEC.loader is not None
_LAUNCH = importlib.util.module_from_spec(_LAUNCH_SPEC)
_LAUNCH_SPEC.loader.exec_module(_LAUNCH)
_RUN_SPEC = importlib.util.spec_from_file_location(
    "cpu_host_run_under_test", SPEC.with_name("run_arm.py"))
assert _RUN_SPEC is not None and _RUN_SPEC.loader is not None
_RUN = importlib.util.module_from_spec(_RUN_SPEC)
_RUN_SPEC.loader.exec_module(_RUN)


def _raw():
    return yaml.safe_load(SPEC.read_text(encoding="utf-8"))


def _bind_test_environment(raw: dict, root: Path, draft: HostPreflight) -> HostPreflight:
    """Bind a small non-live environment artifact for synthetic finalizer fixtures.

    Production protocols can only obtain ``capture_complete: true`` by running the controller with
    a live board probe. Finalizer unit tests do not execute tools or K1, but still exercise the exact
    manifest/source-digest plumbing rather than bypassing it.
    """
    root.mkdir(parents=True, exist_ok=True)
    bundle = root / "fixture_environment.sources.tar"
    bundle.write_bytes(b"synthetic finalizer fixture; not a live environment\n")
    source_index = {}
    for name, value in draft.evidence["paths"].items():
        if name == "environment_manifest":
            continue
        path = Path(value)
        if path.is_file():
            source_index[name] = {
                "path": str(path.resolve()), "archive_path": f"fixture/{name}",
                "size_bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
    manifest = root / "fixture_environment.json"
    manifest.write_text(json.dumps({
        "version": 1, "capture_complete": False, "local": {}, "k1": None,
        "source_bundle": {
            "path": bundle.name,
            "sha256": hashlib.sha256(bundle.read_bytes()).hexdigest(),
            "source_index": source_index, "entry_count": len(source_index),
        },
    }, sort_keys=True, separators=(",", ":")) + "\n")
    raw["environment"].update({
        "manifest": str(manifest.resolve()),
        "sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
    })
    return HostExperimentSpec.parse(raw).preflight(check_environment=False)


def test_curated_host_experiment_is_nested_and_has_all_grader_authorities():
    spec = HostExperimentSpec.from_yaml(SPEC)
    assert len(spec.arms) == 4
    assert all(a.capabilities < b.capabilities for a, b in zip(spec.arms, spec.arms[1:]))
    result = spec.preflight(check_environment=False)
    assert result.ready
    assert result.blockers == ()
    assert any("environment manifest will be captured" in warning
               for warning in result.warnings)
    assert result.evidence["grader_self_check"]["implemented_levels"] == ["L0", "L1", "L2", "L3"]
    assert all(result.evidence["grader_self_check"]["level_ready"].values())
    assert result.evidence["grader_corpus_check"]["ready"] is True
    assert result.evidence["development_corpus_sha256"] == spec.development_corpus["sha256"]
    assert result.evidence["materialized_corpus"]["corpus_sha256"] == (
        spec.development_corpus["materialized_sha256"])
    predecessor = result.evidence["replacement_predecessor"]
    assert predecessor["protocol_design_invalid_verified"] is True
    assert predecessor["treatment_or_provider_started"] is True
    assert predecessor["reason_codes"] == [
        "persistent_worker_audit_contract_contradiction",
        "undisclosed_capsule_descriptor_abi",
    ]
    assert predecessor["excluded_from_arm_outcomes"] is True
    assert all(row["matches"] for row in predecessor["evidence"].values())
    descriptor = result.evidence["grader_self_check"]["capsule_descriptor"]
    assert descriptor["ready"] is True
    assert descriptor["fixture_count"] == 6
    assert all(row["verified"] for row in descriptor["checks"])
    assert "capsule_descriptor_contract" in result.evidence["paths"]
    for evidence_name, path_key in (
            ("search_space_sha256", "space"), ("search_runner_sha256", "runner"),
            ("trusted_evaluator_sha256", "trusted_evaluator"),
            ("trusted_broker_sha256", "trusted_broker"),
            ("trusted_replay_sha256", "trusted_replay")):
        assert result.evidence[evidence_name] == hashlib.sha256(
            spec._repo_path(spec.search[path_key]).read_bytes()).hexdigest()
    assert result.evidence["trusted_search_budget"] == {
        "action_groups": 9,
        "maximum_distinct_incumbents": 10,
        "maximum_screen_candidate_evaluations": 104,
        "confirmation_capsules_per_split": 6,
        "maximum_confirmation_requests": 20,
        "confirmation_package_builds": 40,
        "confirmation_compiler_invocations": 280,
        "confirmation_spike_checks": 240,
        "expected_k1_program_invocations": 1440,
        "k1_program_invocations": 1920,
        "expected_confirmation_overhead_seconds": 1031.0,
        "planning_upper_confirmation_overhead_seconds": 4942.0,
        "expected_search_seconds": 17883.0,
        "planning_upper_search_seconds": 29342.0,
        "available_search_seconds": 30200,
        "fits_declared_arm": True,
    }
    assert spec.freeze["failure_policy"] == {
        "launch_all_scheduled_attempts": True,
        "retry_terminal_outcomes": False,
        "failed_primary_fallback": "forbidden",
        "one_shot_protocol_claim": True,
        "per_cell_atomic_consumption": True,
    }
    assert [row["ordinal"] for row in spec.agent["launch_plan"]] == list(range(16))
    assert {(row["arm"], row["repeat"]) for row in spec.agent["launch_plan"]} == {
        (arm.id, repeat) for arm in spec.arms for repeat in range(4)}
    sequences = spec.analysis_plan_config()["design"]["sequences"]
    assert [row["arm"] for row in spec.agent["launch_plan"]] == [
        arm for sequence in sequences for arm in sequence]
    assert result.evidence["analysis_plan_sha256"] == spec.analysis["sha256"]


def test_protocol_design_replacement_rejects_each_deep_semantic_or_hash_link():
    spec = HostExperimentSpec.from_yaml(SPEC)
    paths = {
        name: spec._repo_path(row["path"])
        for name, row in spec.replacement["evidence"].items()
    }
    documents = {
        name: (yaml.safe_load(path.read_text(encoding="utf-8"))
               if name == "frozen_protocol" else json.loads(path.read_text(encoding="utf-8")))
        for name, path in paths.items() if name != "arm4_raw_events"
    }
    hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest()
              for name, path in paths.items()}
    cells = paths["protocol_claim"].with_name(
        f'{spec.replacement["predecessor_protocol_inputs_sha256"]}.cells')

    def valid(candidate):
        return _protocol_design_replacement_valid(
            replacement=spec.replacement, documents=candidate,
            artifact_sha256=hashes, cells=cells,
            arm4_raw_size=paths["arm4_raw_events"].stat().st_size)

    assert valid(documents)
    mutations = {
        "exclusion classification": lambda value: value["campaign_exclusion"].__setitem__(
            "classification", "graded_failure"),
        "exclusion reasons": lambda value: value["campaign_exclusion"].__setitem__(
            "reason_codes", ["undisclosed_capsule_descriptor_abi"]),
        "treatment-start truth": lambda value: value["campaign_exclusion"].__setitem__(
            "treatment_or_provider_started", False),
        "holdout exclusion": lambda value: value["campaign_exclusion"].__setitem__(
            "excluded_from_holdout_capture", False),
        "non-reuse": lambda value: value["campaign_exclusion"]["non_reuse"].__setitem__(
            "submissions", False),
        "cell schedule": lambda value: value["campaign_exclusion"]["cells"][4].__setitem__(
            "arm", "arm1_raw_cpp"),
        "cell state": lambda value: value["campaign_exclusion"]["cells"][3].__setitem__(
            "state", "not_started"),
        "revoked claim hash": lambda value: value["claim_revocation"]["claim"].__setitem__(
            "sha256", "0" * 64),
        "revoked audit hash": lambda value: value["claim_revocation"][
            "design_audit"].__setitem__("sha256", "0" * 64),
        "revoked exclusion hash": lambda value: value["claim_revocation"][
            "campaign_exclusion"].__setitem__("sha256", "0" * 64),
        "cancellation campaign": lambda value: value[
            "arm4_controller_cancellation"].__setitem__("campaign_run_id", "other"),
        "cancellation protocol": lambda value: value[
            "arm4_controller_cancellation"].__setitem__("protocol_inputs_sha256", "0" * 64),
        "cancellation audit hash": lambda value: value[
            "arm4_controller_cancellation"]["design_audit"].__setitem__(
                "sha256", "0" * 64),
        "exclusion cancellation hash": lambda value: value["campaign_exclusion"][
            "arm4_partial_evidence"].__setitem__("controller_cancellation_sha256", "0" * 64),
    }
    for label, mutate in mutations.items():
        candidate = deepcopy(documents)
        mutate(candidate)
        assert not valid(candidate), label


def test_draft_cannot_authorize_a_live_campaign():
    spec = HostExperimentSpec.from_yaml(SPEC)
    result = spec.preflight(check_environment=False, require_frozen=True)
    assert not result.ready
    assert any("status is draft" in blocker for blocker in result.blockers)


def test_campaign_exclusion_tombstone_blocks_stale_authorization(tmp_path, monkeypatch):
    raw = deepcopy(_raw())
    raw["status"] = "protocol_frozen"
    raw["freeze"]["protocol_inputs_sha256"] = "a" * 64
    raw["environment"].update({"manifest": "/frozen/environment.json", "sha256": "b" * 64})
    spec_path = tmp_path / "frozen.yaml"
    spec_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    spec = HostExperimentSpec.from_yaml(spec_path)
    output = tmp_path / "runs"
    monkeypatch.setitem(spec.telemetry, "output_layout", str(output))
    claims = output / ".protocol_claims"; claims.mkdir(parents=True)
    claim_path = claims / f'{spec.freeze["protocol_inputs_sha256"]}.json'
    campaign = "pilot-campaign"
    claim_path.write_text(json.dumps({
        "version": 1, "status": "bound", "campaign_run_id": campaign,
        "protocol_inputs_sha256": spec.freeze["protocol_inputs_sha256"],
        "environment_manifest_sha256": spec.environment["sha256"],
        "analysis_plan_sha256": spec.analysis["sha256"],
        "spec_path": str(spec_path.resolve()),
        "spec_sha256": hashlib.sha256(spec_path.read_bytes()).hexdigest(),
    }), encoding="utf-8")
    exclusions = output / ".campaign_exclusions"; exclusions.mkdir()
    exclusions.joinpath(f"{campaign}.json").write_text(json.dumps({
        "authority": "controller_campaign_exclusion_v1",
        "campaign_run_id": campaign,
        "protocol_inputs_sha256": spec.freeze["protocol_inputs_sha256"],
        "excluded_from_arm_outcomes": True,
        "excluded_from_promotion": True,
    }), encoding="utf-8")

    arm = spec.arms[0]
    run_id = f"{campaign}__{arm.id}__r00__seed001"
    with pytest.raises(ValueError, match="controller-excluded"):
        _RUN._authorization_cell(spec, spec_path, arm.id, 1, run_id, claim_path)


def test_frozen_protocol_requires_content_addressed_environment_manifest():
    raw = deepcopy(_raw())
    draft = HostExperimentSpec.parse(raw).preflight(check_environment=False)
    raw["status"] = "protocol_frozen"
    raw["freeze"]["protocol_inputs_sha256"] = draft.evidence["protocol_inputs_sha256"]
    result = HostExperimentSpec.parse(raw).preflight(check_environment=False, require_frozen=True)
    assert not result.ready
    assert any("environment manifest is unresolved" in blocker for blocker in result.blockers)


def test_environment_contract_requires_exact_local_and_k1_rechecks():
    raw = deepcopy(_raw())
    raw["environment"]["require_exact_k1_before_each_cell"] = False
    with pytest.raises(ValueError, match="exact local and K1 identity"):
        HostExperimentSpec.parse(raw)


@pytest.mark.parametrize("mutation", ["order", "seed"])
def test_launch_plan_is_derived_from_frozen_williams_design_and_block_rule(mutation):
    raw = deepcopy(_raw())
    if mutation == "order":
        first_repeat = raw["agent"]["launch_plan"][:4]
        raw["agent"]["launch_plan"][:4] = list(reversed(first_repeat))
        for ordinal, row in enumerate(raw["agent"]["launch_plan"]):
            row["ordinal"] = ordinal
    else:
        for row in raw["agent"]["launch_plan"]:
            row["seed"] = -99
    with pytest.raises(ValueError, match=r"frozen 4x4 Williams schedule.+paired block"):
        HostExperimentSpec.parse(raw)


def test_analysis_plan_balances_positions_and_all_within_block_carryover_pairs():
    spec = HostExperimentSpec.from_yaml(SPEC)
    sequences = spec.analysis_plan_config()["design"]["sequences"]
    arm_ids = {arm.id for arm in spec.arms}
    assert all({sequence[position] for sequence in sequences} == arm_ids
               for position in range(4))
    transitions = [(left, right) for sequence in sequences
                   for left, right in zip(sequence, sequence[1:])]
    assert len(transitions) == len(set(transitions)) == 12
    assert set(transitions) == {
        (left, right) for left in arm_ids for right in arm_ids if left != right}
    design = spec.analysis_plan_config()["design"]
    assert design["carryover_scope"] == "within_block_transitions_only"
    assert design["block_boundary"]["retained_receipt_required"] is True


def test_replacement_predecessor_digest_tampering_fails_closed():
    raw = _raw()
    raw["replacement"]["evidence"]["design_audit"]["sha256"] = "0" * 64
    result = HostExperimentSpec.parse(raw).preflight(check_environment=False)
    assert result.ready is False
    assert "replacement predecessor evidence differs: design_audit" in result.errors


def test_provider_sampling_is_explicitly_unseeded():
    spec = HostExperimentSpec.from_yaml(SPEC)
    assert spec.agent["launch_seed_role"] == "campaign_metadata_only_not_provider_sampling"
    assert spec.analysis_plan_config()["design"]["provider_sampling_seeded"] is False


def test_block_boundary_enforces_washout_and_retains_qualifying_probe(monkeypatch):
    sleeps = []
    monkeypatch.setattr(_LAUNCH.time, "sleep", sleeps.append)

    class FakeSpec:
        def search_space_config(self):
            return {"board_environment": {
                "settle_attempts": 3, "settle_interval_seconds": 5,
                "online": "0-7"}}

        def preflight(self, **_kwargs):
            return HostPreflight((), (), (), {
                "protocol_inputs_sha256": "a" * 64,
                "analysis_plan_sha256": "b" * 64,
                "k1_board_state_ready": True,
                "k1_board_state_probe": {"authority": "driver_ssh_sysfs_procfs"},
            })

    gate, receipt = _LAUNCH._requalify_block_boundary(
        FakeSpec(), block=2, first_ordinal=8)
    assert gate.ready and sleeps == [5.0]
    assert receipt["authority"] == "frozen_k1_board_environment_gate"
    assert receipt["block"] == 2 and receipt["first_ordinal"] == 8
    assert receipt["qualifying_attempt_index"] == 0
    assert receipt["attempts"][0]["evidence"]["k1_board_state_ready"] is True


def test_only_passing_aa_noise_artifact_is_bound():
    spec = HostExperimentSpec.from_yaml(SPEC)
    failed = (repo_root() /
              "out/runs/k1_cpu/cpu-host-compiler/"
              "20260831T160455Z_k1-aa-noise-calibration_seed000_b8213a5/metrics/"
              "k1_aa_noise_calibration.json")
    bound = spec._repo_path(spec.search["calibration"]["noise_artifact"])
    assert bound.is_file()
    assert json.loads(bound.read_text())["status"] == "pass"
    assert bound != failed
    if failed.is_file():
        assert json.loads(failed.read_text())["status"] == "fail"


def test_protocol_freeze_allows_campaign_before_output_hashes_exist():
    raw = deepcopy(_raw())
    draft = HostExperimentSpec.parse(raw).preflight(check_environment=False)
    raw["status"] = "protocol_frozen"
    raw["freeze"]["protocol_inputs_sha256"] = draft.evidence["protocol_inputs_sha256"]
    spec = HostExperimentSpec.parse(raw)
    result = spec.preflight(check_environment=False, require_frozen=True)
    assert not result.ready
    assert not any(name in error for name in (
        "selected_policy_sha256", "runtime_sha256", "compiler_sha256")
                   for error in result.errors)


def test_completed_campaign_requires_post_campaign_output_hashes():
    raw = deepcopy(_raw())
    draft = HostExperimentSpec.parse(raw).preflight(check_environment=False)
    raw["status"] = "campaign_complete"
    raw["freeze"]["protocol_inputs_sha256"] = draft.evidence["protocol_inputs_sha256"]
    result = HostExperimentSpec.parse(raw).preflight(check_environment=False)
    assert {name for name in ("selected_policy_sha256", "runtime_sha256", "compiler_sha256")
            if any(name in error for error in result.errors)} == {
            "selected_policy_sha256", "runtime_sha256", "compiler_sha256"}


def test_freeze_protocol_cli_helper_is_atomic_and_round_trips(tmp_path):
    valid_raw = _raw()
    _bind_valid_calibration_fixtures(valid_raw, tmp_path / "calibrations")
    source = tmp_path / "draft.yaml"
    source.write_text(yaml.safe_dump(valid_raw, sort_keys=False), encoding="utf-8")
    output = tmp_path / "protocol_frozen.yaml"
    frozen, result = freeze_protocol(source, output, check_environment=False)
    assert result.ready
    assert frozen.status == "protocol_frozen"
    assert frozen.freeze["protocol_inputs_sha256"] == result.evidence[
        "protocol_inputs_sha256"]
    assert HostExperimentSpec.from_yaml(SPEC).status == "draft"

    broken_raw = deepcopy(valid_raw)
    broken_raw["agent"]["active_wall_seconds_per_arm"] = 6000
    broken = tmp_path / "broken.yaml"
    broken.write_text(yaml.safe_dump(broken_raw), encoding="utf-8")
    absent = tmp_path / "must_not_exist.yaml"
    with pytest.raises(ValueError, match="preflight is NO_GO"):
        freeze_protocol(broken, absent, check_environment=False)
    assert not absent.exists()
    assert not absent.with_name(f".{absent.name}.freeze.lock").exists()

    existing = tmp_path / "already_frozen.yaml"
    existing.write_text("do not replace\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        freeze_protocol(source, existing, check_environment=False)
    assert existing.read_text(encoding="utf-8") == "do not replace\n"
    assert not existing.with_name(f".{existing.name}.freeze.lock").exists()


def test_freeze_protocol_exclusive_reservation_prevents_concurrent_draft_mixup(
        tmp_path, monkeypatch):
    """A contender that loses the reservation cannot return the winner's frozen spec."""
    first_raw, second_raw = _raw(), _raw()
    _bind_valid_calibration_fixtures(first_raw, tmp_path / "first_calibrations")
    _bind_valid_calibration_fixtures(second_raw, tmp_path / "second_calibrations")
    first_raw["label"] = "first_concurrent_draft"
    second_raw["label"] = "second_concurrent_draft"
    first = tmp_path / "first.yaml"
    second = tmp_path / "second.yaml"
    first.write_text(yaml.safe_dump(first_raw, sort_keys=False), encoding="utf-8")
    second.write_text(yaml.safe_dump(second_raw, sort_keys=False), encoding="utf-8")
    output = tmp_path / "protocol_frozen.yaml"

    original_preflight = HostExperimentSpec.preflight
    first_in_preflight = threading.Event()
    release_first = threading.Event()

    def pause_first_draft(self, **kwargs):
        if self.source_path == first.resolve() and self.status == "draft":
            first_in_preflight.set()
            assert release_first.wait(timeout=5), "test did not release the first freezer"
        return original_preflight(self, **kwargs)

    monkeypatch.setattr(HostExperimentSpec, "preflight", pause_first_draft)
    first_result: list[object] = []
    first_error: list[BaseException] = []
    second_result: list[object] = []
    second_error: list[BaseException] = []

    def freeze_into(source, result, errors):
        try:
            result.append(freeze_protocol(source, output, check_environment=False))
        except BaseException as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    first_worker = threading.Thread(target=freeze_into, args=(first, first_result, first_error))
    first_worker.start()
    assert first_in_preflight.wait(timeout=5)
    second_worker = threading.Thread(
        target=freeze_into, args=(second, second_result, second_error))
    second_worker.start()
    try:
        second_worker.join(timeout=5)
    finally:
        release_first.set()
        first_worker.join(timeout=30)

    assert not first_worker.is_alive()
    assert not second_worker.is_alive()
    assert not first_error
    assert not second_result
    assert len(second_error) == 1
    assert isinstance(second_error[0], FileExistsError)
    assert "another protocol freezer" in str(second_error[0])
    frozen, result = first_result[0]
    assert result.ready
    assert frozen.label == "first_concurrent_draft"
    assert HostExperimentSpec.from_yaml(output).label == "first_concurrent_draft"
    assert not output.with_name(f".{output.name}.freeze.lock").exists()


def test_host_experiment_rejects_non_nested_treatments():
    raw = deepcopy(_raw())
    raw["arms"][2]["capabilities"] = raw["arms"][1]["capabilities"]
    with pytest.raises(ValueError, match="strictly nested"):
        HostExperimentSpec.parse(raw)


def test_host_experiment_requires_full_fidelity_aet_telemetry():
    raw = deepcopy(_raw())
    raw["telemetry"]["reasoning_tokens"] = False
    with pytest.raises(ValueError, match="full-fidelity"):
        HostExperimentSpec.parse(raw)


def test_protocol_digest_binds_freeze_methodology_and_external_source_content(tmp_path):
    raw = deepcopy(_raw())
    first = HostExperimentSpec.parse(raw).preflight(check_environment=False)
    raw["freeze"]["forbid_post_freeze_tuning"] = False
    with pytest.raises(ValueError, match="freeze must forbid"):
        HostExperimentSpec.parse(raw)

    # The exact dirty bytes, not only the dirty path name, identify external agent machinery.
    source = tmp_path / "external"
    source.mkdir()
    __import__("subprocess").run(["git", "init", "-q"], cwd=source, check=True)
    __import__("subprocess").run(["git", "config", "user.email", "test@example.com"],
                                 cwd=source, check=True)
    __import__("subprocess").run(["git", "config", "user.name", "Test"], cwd=source, check=True)
    tracked = source / "source.py"
    tracked.write_text("v1\n")
    __import__("subprocess").run(["git", "add", "source.py"], cwd=source, check=True)
    __import__("subprocess").run(["git", "commit", "-qm", "base"], cwd=source, check=True)
    raw = deepcopy(_raw())
    raw["telemetry"]["aet_source"] = str(source)
    raw["telemetry"]["chia_source"] = str(source)
    clean = HostExperimentSpec.parse(raw).preflight(check_environment=False)
    tracked.write_text("v2\n")
    dirty = HostExperimentSpec.parse(raw).preflight(check_environment=False)
    assert clean.evidence["protocol_inputs_sha256"] != dirty.evidence["protocol_inputs_sha256"]
    assert clean.evidence["aet_provenance"]["git_sha"] == dirty.evidence[
        "aet_provenance"]["git_sha"]
    assert clean.evidence["aet_provenance"]["dirty_content_sha256"] != dirty.evidence[
        "aet_provenance"]["dirty_content_sha256"]


def test_protocol_digest_binds_predeclared_selection_rule():
    raw = deepcopy(_raw())
    raw["freeze"]["selection"]["primary_repeat_index"] = 1
    with pytest.raises(ValueError, match="predeclare arm4/repeat0"):
        HostExperimentSpec.parse(raw)


def test_protocol_digest_binds_exact_prematerialized_arm_inputs(monkeypatch):
    first = HostExperimentSpec.from_yaml(SPEC).preflight(check_environment=False)
    from merlin.targetgen.generate import target_repo
    original = target_repo.generate_skeleton

    def changed(target):
        return [*original(target), Artifact("GENERATED_TREATMENT_CHANGE", "changed\n")]

    monkeypatch.setattr(target_repo, "generate_skeleton", changed)
    second = HostExperimentSpec.from_yaml(SPEC).preflight(check_environment=False)
    assert first.evidence["arm_workspace_inputs"]["arm1_raw_cpp"] == second.evidence[
        "arm_workspace_inputs"]["arm1_raw_cpp"]
    for arm in ("arm2_cpp_scaffold", "arm3_generated_cpu_dialect",
                "arm4_agentic_pass_authoring"):
        assert first.evidence["arm_workspace_inputs"][arm] != second.evidence[
            "arm_workspace_inputs"][arm]
    assert first.evidence["protocol_inputs_sha256"] != second.evidence[
        "protocol_inputs_sha256"]


def test_preflight_fails_closed_when_trusted_search_cannot_fit_arm_budget():
    raw = deepcopy(_raw())
    raw["agent"]["active_wall_seconds_per_arm"] = 6000
    result = HostExperimentSpec.parse(raw).preflight(check_environment=False)
    assert not result.ready
    assert result.evidence["trusted_search_budget"]["fits_declared_arm"] is False
    assert any("planning upper bound" in blocker for blocker in result.blockers)


def test_development_corpus_excludes_exact_paper_roster(tmp_path):
    raw = deepcopy(_raw())
    corpus = yaml.safe_load(
        (repo_root() / raw["development_corpus"]["manifest"]).read_text(encoding="utf-8"))
    corpus["paper_model_exclusion"]["forbidden_workloads"] = corpus[
        "paper_model_exclusion"]["forbidden_workloads"][:-1]
    corpus_path = tmp_path / "corpus.yaml"
    corpus_path.write_text(yaml.safe_dump(corpus), encoding="utf-8")
    raw["development_corpus"]["manifest"] = str(corpus_path)
    spec = HostExperimentSpec.parse(raw)
    result = spec.preflight(check_environment=False)
    assert any("does not exactly equal" in error for error in result.errors)
    assert any("digest differs" in error for error in result.errors)


def test_preflight_rejects_locked_corpus_that_cannot_satisfy_grader_tail_selection():
    """A digest-valid corpus can still make a paid arm structurally unwinnable."""
    raw = deepcopy(_raw())
    definition = repo_root() / "merlin/benchmarks/rvv_paper/development_corpus_v1.yaml"
    materialized = (
        repo_root() / "out/artifacts/rvv-development-corpus/k1_cpu/v1/latest"
    )
    lock = yaml.safe_load((materialized / "corpus_lock.yaml").read_text(encoding="utf-8"))
    raw["development_corpus"].update({
        "manifest": str(definition),
        "sha256": hashlib.sha256(definition.read_bytes()).hexdigest(),
        "materialized_capsules": str(materialized),
        "materialized_sha256": lock["corpus_sha256"],
    })
    result = HostExperimentSpec.parse(raw).preflight(check_environment=False)
    assert not result.ready
    assert any(
        "movement_layout" in message and "RVV tail" in message
        for message in (*result.errors, *result.blockers)
    )


def test_preflight_rederives_materialized_splits_from_frozen_definition(tmp_path):
    """A self-consistent lock cannot smuggle rows absent from the generic definition."""
    raw = deepcopy(_raw())
    source = repo_root() / raw["development_corpus"]["materialized_capsules"]
    materialized = tmp_path / "materialized"
    shutil.copytree(source.resolve(), materialized)

    identity = {
        "family": "contraction", "operation": "matmul", "dtype": "fp32",
        "shape": {"M": 999, "N": 999, "K": 999}, "layout": "row_row",
        "state": "stateless", "core_count": 1,
    }
    canonical = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    row = {
        "id": f"contraction-matmul-{digest[:16]}", "sha256": digest,
        "split": "train", **identity,
    }
    train = materialized / "public/train.jsonl"
    train.write_text(
        train.read_text(encoding="utf-8")
        + json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    lock_path = materialized / "corpus_lock.yaml"
    lock = yaml.safe_load(lock_path.read_text(encoding="utf-8"))
    payloads = {
        name: (materialized / name).read_bytes()
        for name in ("public/train.jsonl", "public/validation.jsonl", "sealed/heldout.jsonl")
    }
    aggregate = hashlib.sha256()
    for name in sorted(payloads):
        aggregate.update(name.encode("utf-8") + b"\0" + payloads[name])
    lock["files"] = {name: hashlib.sha256(value).hexdigest()
                     for name, value in payloads.items()}
    lock["corpus_sha256"] = aggregate.hexdigest()
    lock["capsule_count"] += 1
    lock["split_counts"]["train"] += 1
    lock_path.write_text(yaml.safe_dump(lock, sort_keys=False), encoding="utf-8")
    raw["development_corpus"]["materialized_capsules"] = str(materialized)
    raw["development_corpus"]["materialized_sha256"] = lock["corpus_sha256"]

    result = HostExperimentSpec.parse(raw).preflight(check_environment=False)
    assert not result.ready
    assert any("not the exact expansion" in error for error in result.errors)


def _bind_valid_calibration_fixtures(raw: dict, root: Path) -> None:
    """Give finalizer fixtures passing calibrations without weakening production validation."""
    root.mkdir(parents=True, exist_ok=True)
    for path_key, digest_key in (
            ("space", "space_sha256"), ("runner", "runner_sha256"),
            ("trusted_evaluator", "trusted_evaluator_sha256"),
            ("trusted_broker", "trusted_broker_sha256"),
            ("trusted_replay", "trusted_replay_sha256")):
        raw["search"][digest_key] = hashlib.sha256(
            (repo_root() / raw["search"][path_key]).read_bytes()).hexdigest()
    search_paths = {
        "cost_calibrator": raw["search"]["cost_calibrator"],
        "noise_calibrator": raw["search"]["noise_calibrator"],
        "grader": raw["grading"]["grader"],
        "search_runner": raw["search"]["runner"],
        "trusted_harness": raw["grading"]["trusted_harness"],
        "k1_monitor": raw["grading"]["k1_monitor"],
        "search_space": raw["search"]["space"],
        "trusted_evaluator": raw["search"]["trusted_evaluator"],
        "trusted_broker": raw["search"]["trusted_broker"],
        "k1_adapter": "merlin/python/merlin/mining/k1.py",
    }
    sources = {name: hashlib.sha256((repo_root() / path).read_bytes()).hexdigest()
               for name, path in search_paths.items()}
    raw["search"]["cost_calibrator_sha256"] = sources["cost_calibrator"]
    raw["search"]["noise_calibrator_sha256"] = sources["noise_calibrator"]
    space = yaml.safe_load((repo_root() / raw["search"]["space"]).read_text())
    budget = space["budget"]
    train = repo_root() / raw["development_corpus"]["materialized_capsules"] / \
        "public" / "train.jsonl"
    train_sha = hashlib.sha256(train.read_bytes()).hexdigest()
    cost_input = root / "cost_input"
    cost_input.mkdir()
    (cost_input / "manifest.yaml").write_text(
        "version: 1\nbuild:\n  command: [cmake, --build, build]\n")
    input_rows = [("manifest.yaml", "file", hashlib.sha256(
        (cost_input / "manifest.yaml").read_bytes()).hexdigest())]
    input_tree_sha = hashlib.sha256(json.dumps(
        input_rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    cost_submission = root / "cost_submission"
    cost_submission.mkdir()
    (cost_submission / "manifest.yaml").write_text(
        "version: 1\nbuild:\n  command: [/bin/true]\n")
    cost_rows = [("manifest.yaml", "file", hashlib.sha256(
        (cost_submission / "manifest.yaml").read_bytes()).hexdigest())]
    cost_tree_sha = hashlib.sha256(json.dumps(
        cost_rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    receipt = {
        "version": 1, "authority": "driver_private_prebuild",
        "submitted_manifest_sha256": input_rows[0][2],
        "private_manifest_sha256": cost_rows[0][2],
        "real_build_commands": [["cmake", "--build", "build"]],
        "real_build_logs": [{"returncode": 0}],
        "prebuild_tree_sha256": _grader_package_tree_identity(cost_input),
        "built_tree_sha256": "2" * 64,
        "sealed_prebuilt_tree_sha256": _grader_package_tree_identity(cost_submission),
        "submitted_entrypoint_identity": None,
        "built_entrypoint_identity": [0o755, "4" * 64],
        "private_build_override": ["/bin/true"], "policy_sha256": "5" * 64,
    }
    tool_path = Path(sys.executable).resolve()
    tool_identity = {"path": str(tool_path),
                     "sha256": hashlib.sha256(tool_path.read_bytes()).hexdigest(),
                     "mode": tool_path.stat().st_mode & 0o777}
    train_rows = [json.loads(line) for line in train.read_text().splitlines() if line.strip()]
    common = {"version": 1, "paid_work": False, "heldout_opened": False,
              "protocol_state_mutated": False, "public_split_sha256": train_sha,
              "public_context": {"authority": "complete_public_train",
                                 "capsule_ids": [row["id"] for row in train_rows],
                                 "row_count": len(train_rows),
                                 "rows_sha256": hashlib.sha256(json.dumps(
                                     train_rows, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()},
              "source_sha256": sources,
              "submission": str(cost_submission.resolve()),
              "submission_manifest_sha256": cost_rows[0][2],
              "submission_tree_sha256": cost_tree_sha,
              "prebuild_input_submission": str(cost_input.resolve()),
              "prebuild_input_manifest_sha256": input_rows[0][2],
              "prebuild_input_tree_sha256": input_tree_sha,
              "prebuild_input_package_sha256": receipt["prebuild_tree_sha256"],
              "prebuild_receipt": receipt,
              "prebuild_receipt_sha256": hashlib.sha256(json.dumps(
                  receipt, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
              "space": str((repo_root() / raw["search"]["space"]).resolve()),
              "space_sha256": sources["search_space"],
              "toolchain_identity": {name: dict(tool_identity) for name in (
                  "python", "bwrap", "prebuild_command_0", "private_build_override_0",
                  "spike_gcc", "spike_spike",
                  "spike_objdump", "k1_clang", "k1_objcopy", "ssh", "scp")}}
    nonce = b"\x12" * 32

    def private_authority(phase: str, per_family: int):
        public = _calibration_sample(
            train_rows, per_family=per_family,
            families=list(space["confirmation_families"]))
        private = [_calibration_private_capsule(row, nonce=nonce, phase=phase)
                   for row in public]
        records = [{"public": source, "private": measured}
                   for source, measured in zip(public, private, strict=True)]
        return private, {
            "version": 1,
            "authority": "trusted_broker_private_capsule_independent_calibration_nonce",
            "phase": phase, "split": "train", "nonce_hex": nonce.hex(),
            "nonce_sha256": hashlib.sha256(nonce).hexdigest(), "records": records,
            "records_sha256": hashlib.sha256(json.dumps(
                records, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        }

    confirm_private, confirm_authority = private_authority(
        "confirm", int(space["confirmation_samples_per_family"]))
    screen_private, screen_authority = private_authority(
        "screen", int(space["screen_samples_per_family"]))
    k1_capsule = confirm_private[0]
    programs = [{
        "index": index, "status": "pass", "total_seconds": 1.0,
        "start_monotonic_ns": index * 2_000_000_000,
        "end_monotonic_ns": index * 2_000_000_000 + 1_000_000_000,
        "capsule_id": k1_capsule["id"], "family": k1_capsule["family"],
        "checks": {"numeric_correctness": True, "wall_time": True},
        "kernel_text_sha256": "6" * 64, "seed": 7,
        "metrics": {"wall_ns": 100, "calls": 1}, "monitor": {"returncode": 0},
        "evidence": {
            "capsule": k1_capsule["id"], "family": k1_capsule["family"],
            "status": "pass", "checks": {"numeric_correctness": True, "wall_time": True},
            "kernel_text_sha256": "6" * 64, "seed": 7,
            "metrics": {"wall_ns": 100, "calls": 1}, "monitor": {"returncode": 0},
            "receipt_nonce": 11, "local_sha256": "7" * 64,
            "remote_sha256": "7" * 64,
        },
    } for index in range(12)]
    k1 = {**common, "private_shape_calibration": confirm_authority,
           "kind": "cpu_host_trusted_search_k1_program_calibration",
           "status": "pass",
           "checks": {"all_passed": True, "max_within_planning_upper": True,
                      "mean_within_expected": True},
           "declared": {
               "expected_seconds_per_program": float(budget["expected_seconds_per_k1_program"]),
               "planning_upper_seconds_per_program": float(
                   budget["planning_upper_seconds_per_k1_program"]),
           }, "programs": programs,
           "calibration_capsule": {key: k1_capsule.get(key)
                                   for key in ("id", "sha256", "family", "split")},
           "trusted_evaluation_observations": [{
               "capsule_id": k1_capsule["id"], "family": k1_capsule["family"],
               "correctness_ok": True, "board_condition_pairs": [{}] * 6}],
           "statistics": {"count": 12, "mean_seconds": 1.0,
                          "median_seconds": 1.0, "p95_seconds": 1.0,
                          "max_seconds": 1.0}}
    spike = {**common, "private_shape_calibration": screen_authority,
             "kind": "cpu_host_trusted_search_spike_candidate_calibration",
             "status": "pass",
             "checks": {"all_observations_passed": True,
                        "projection_within_expected_budget": True},
             "declared": {"expected_spike_screen_seconds": float(
                 budget["expected_spike_screen_seconds"]),
                 "maximum_candidate_evaluations": int(
                     budget["maximum_screen_candidate_evaluations"])},
             "capsules": 12, "completed_observations": 12,
             "maximum_candidate_evaluations": 104,
             "start_monotonic_ns": 1_000_000_000,
             "end_monotonic_ns": 2_000_000_000,
             "candidate_evaluation_seconds": 1.0,
             "projected_max_screen_seconds": 104.0,
             "observations": [{"capsule_id": capsule["id"],
                               "family": capsule["family"],
                               "correctness_ok": True}
                              for capsule in screen_private]}
    limits = {
        "package_build": {
            "expected_seconds": float(budget[
                "expected_seconds_per_confirmation_package_build"]),
            "planning_upper_seconds": float(budget[
                "planning_upper_seconds_per_confirmation_package_build"]),
        },
        "compiler_invocation": {
            "expected_seconds": float(budget[
                "expected_seconds_per_confirmation_compiler_invocation"]),
            "planning_upper_seconds": float(budget[
                "planning_upper_seconds_per_confirmation_compiler_invocation"]),
        },
        "spike_check": {
            "expected_seconds": float(budget[
                "expected_seconds_per_confirmation_spike_check"]),
            "planning_upper_seconds": float(budget[
                "planning_upper_seconds_per_confirmation_spike_check"]),
        },
    }
    confirmation = {**common, "private_shape_calibration": confirm_authority,
                    "kind": "cpu_host_confirmation_overhead_calibration",
                    "status": "pass", "checks": {
                        "all_toolchain_stages_passed": True,
                        "all_expected_costs_within_budget": True,
                        "all_maximum_costs_within_planning_upper": True},
                    "declared": limits, "calibration_repeats_per_capsule": 2,
                    "public_capsules": [
                        {key: capsule.get(key) for key in ("id", "sha256", "family")}
                        for capsule in confirm_private],
                    "spike_statuses": ["pass"] * 12,
                    "trusted_evaluation_observations": [{
                        "capsule_id": capsule["id"], "family": capsule["family"],
                        "correctness_ok": True,
                        "calibration_authority":
                        "exact_confirmation_pre_k1_stages_without_k1",
                    } for capsule in confirm_private]}
    for name, values in limits.items():
        stage_count = 14 if name == "compiler_invocation" else 12
        confirmation[name] = {
            "count": stage_count, "mean_seconds": values["expected_seconds"] / 2,
            "median_seconds": values["expected_seconds"] / 2,
            "p95_seconds": values["expected_seconds"] / 2,
            "max_seconds": values["expected_seconds"] / 2,
        }
    stage_membership = {
        name: [(capsule, side, mode)
               for capsule in confirm_private for side in ("parent", "candidate")
               for mode in (("rvv", "rvv_multicore")
                            if name == "compiler_invocation" and
                            capsule["family"] == "runtime_parallel"
                            else (("rvv",) if name == "compiler_invocation" else (None,)))]
        for name in limits
    }
    confirmation["stage_observations"] = {
        name: [{
            "index": index, "start_monotonic_ns": index * 2_000_000_000,
            "end_monotonic_ns": index * 2_000_000_000 + int(
                values["expected_seconds"] / 2 * 1e9),
            "wall_seconds": values["expected_seconds"] / 2,
            "stage": name, "capsule_id": capsule["id"],
            "family": capsule["family"], "side": side, "mode": mode,
            "status": "pass", "evidence": {"fixture": True},
        } for index, (capsule, side, mode) in enumerate(stage_membership[name])]
        for name, values in limits.items()
    }
    pair_orders = ["parent_candidate", "candidate_parent", "candidate_parent",
                   "parent_candidate", "parent_candidate", "candidate_parent"]
    board_state = {
        "authority": "driver_ssh_sysfs_procfs", "controller_monotonic_ns": 1,
        "returncode": 0,
        "state": {
            "online": "0-7",
            "governors": {str(index): "performance" for index in range(8)},
            "frequencies_khz": {str(index): "1600000" for index in range(8)},
            "temperatures_millic": {"0": "40000"},
            "loadavg": "0.1 0.1 0.1 1/1 1",
        },
    }

    def k1_pair_evidence(capsule, wall_ns):
        return {
            "capsule": capsule["id"], "family": capsule["family"], "status": "pass",
            "seed": 7, "checks": {"numeric_correctness": True, "wall_time": True},
            "metrics": {"wall_ns": wall_ns, "calls": 1},
            "kernel_text_sha256": "b" * 64, "receipt_nonce": 11,
            "local_sha256": "7" * 64, "remote_sha256": "7" * 64,
            "monitor": {"returncode": 0}, "ssh_returncode": 0,
            "board_wall_seconds": 0.1,
        }

    observations = [{
        "capsule_id": capsule["id"], "family": capsule["family"],
        "correctness_ok": True,
        "baseline_elapsed_ns": [1117] * 6, "baseline_calls": [1] * 6,
        "candidate_elapsed_ns": [1000] * 6, "candidate_calls": [1] * 6,
        "baseline_code_sha256": "b" * 64, "candidate_code_sha256": "b" * 64,
        "pair_orders": pair_orders, "excluded_board_condition_pairs": [],
        "k1_program_count": 12,
        "board_condition_pairs": [{
            "pair_id": pair_index, "attempt_id": pair_index,
            "order": pair_orders[pair_index], "seed": 7,
            "settle_probes": [deepcopy(board_state)],
            "before": deepcopy(board_state), "after": deepcopy(board_state),
            "valid": True,
            "measurements": {side: {
                "elapsed_ns": (1117 if side == "parent" else 1000),
                "calls": 1, "seed": 7,
                "evidence": k1_pair_evidence(
                    capsule, 1117 if side == "parent" else 1000),
            } for side in ("parent", "candidate")},
        } for pair_index in range(6)],
        "spike_gates": {label: {
            "compile_ok": True, "k1_compile_ok": True, "passed": True,
            "checks": {"rvv_correctness": True, "instruction_evidence": True,
                       "vlen_256": True, "cycle_measurement": True},
            "kernel_text_sha256": "c" * 64,
        } for label in ("parent", "candidate")},
    } for capsule in confirm_private]
    pairs = [{
        "capsule_id": capsule["id"], "family": capsule["family"],
        "pair_index": pair_index, "speedup_ratio": 1.117,
        "absolute_unit_deviation": abs(1.117 - 1.0),
    } for capsule in confirm_private for pair_index in range(6)]
    margin = float(space["noise_margin"])
    derivation = (
        "margin=max(0.02,ceil((exp(max(abs(log(pair_ratio)))+0.005)-1)*1000)/1000); "
        "lower_bound=1/(1+margin)")
    calibration_protocol = {
        "version": 1,
        "confirmation_samples_per_family": int(space["confirmation_samples_per_family"]),
        "confirmation_families": list(space["confirmation_families"]),
        "measurement_repeats": int(space["measurement_repeats"]),
        "board_environment": dict(space["board_environment"]),
        "private_shape_authority":
            "trusted_broker_private_capsule_independent_calibration_nonce",
        "public_context": "complete_public_train",
        "search_package_authority": "driver_private_prebuild",
        "derivation": derivation,
    }
    calibration_protocol_sha = hashlib.sha256(json.dumps(
        calibration_protocol, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()
    noise = {**common, "private_shape_calibration": confirm_authority,
        "version": 2, "kind": "cpu_host_k1_order_balanced_aa_noise_calibration",
        "status": "pass", "paid_work": False, "heldout_opened": False,
        "checks": {"six_families": True, "six_valid_pairs_per_family": True,
                   "all_correct": True, "identical_k1_text": True,
                   "no_heldout_argument": True},
        "calibrator_sha256": sources["noise_calibrator"],
        "grader_sha256": sources["grader"], "runner_sha256": sources["search_runner"],
        "trusted_harness_sha256": sources["trusted_harness"],
        "k1_monitor_sha256": sources["k1_monitor"],
        "calibration_protocol": calibration_protocol,
        "calibration_protocol_sha256": calibration_protocol_sha,
        "calibration_lineage": {
            "version": 1, "stage": "noise_pre_result",
            "pre_result_protocol_sha256": calibration_protocol_sha,
            "raw_input_tree_sha256": input_tree_sha,
            "raw_input_package_sha256": receipt["prebuild_tree_sha256"],
            "output_field": "noise_margin",
        },
        "derivation": derivation,
        "public_train_sha256": train_sha,
        "observations": observations, "pairs": pairs,
        "maximum_absolute_pair_deviation": abs(1.117 - 1.0),
        "maximum_absolute_log_ratio": math.log(1.117),
        "padded_log_half_width": math.log(1.117) + 0.005,
        "derived_noise_margin": margin, "upper_speedup_bound": 1.0 + margin,
        "lower_speedup_bound": 1.0 / (1.0 + margin),
    }
    noise["source_sha256"] = {
        name: digest for name, digest in sources.items() if name != "search_space"}
    noise.pop("space_sha256", None)
    noise_path = root / "noise.json"
    noise_path.write_text(json.dumps(noise))
    noise_sha = hashlib.sha256(noise_path.read_bytes()).hexdigest()
    cost_lineage = {
        "version": 1, "stage": "cost_post_noise_result",
        "predecessor_stage": "noise_pre_result",
        "noise_authority": str(noise_path.resolve()),
        "noise_authority_sha256": noise_sha,
        "pre_result_protocol_sha256": calibration_protocol_sha,
        "derived_noise_margin": margin,
        "raw_input_tree_sha256": input_tree_sha,
        "raw_input_package_sha256": receipt["prebuild_tree_sha256"],
        "final_space_sha256": sources["search_space"],
    }
    for cost_value in (k1, spike, confirmation):
        cost_value["calibration_lineage"] = dict(cost_lineage)
    for key, value in (("k1", k1), ("spike", spike),
                       ("confirmation_overhead", confirmation), ("noise", noise)):
        path = root / f"{key}.json"
        if key != "noise":
            path.write_text(json.dumps(value))
        artifact_key = ("confirmation_overhead_artifact" if key == "confirmation_overhead"
                        else f"{key}_artifact")
        sha_key = ("confirmation_overhead_sha256" if key == "confirmation_overhead"
                   else f"{key}_sha256")
        raw["search"]["calibration"][artifact_key] = str(path.resolve())
        raw["search"]["calibration"][sha_key] = hashlib.sha256(path.read_bytes()).hexdigest()


def test_preflight_rejects_cost_artifacts_bound_to_a_different_noise_file(
        tmp_path: Path) -> None:
    raw = _raw()
    root = tmp_path / "calibrations"
    _bind_valid_calibration_fixtures(raw, root)
    original = root / "noise.json"
    alternate = root / "noise_reencoded.json"
    alternate.write_text(
        json.dumps(json.loads(original.read_text()), indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    assert hashlib.sha256(alternate.read_bytes()).hexdigest() != hashlib.sha256(
        original.read_bytes()).hexdigest()
    raw["search"]["calibration"]["noise_artifact"] = str(alternate.resolve())
    raw["search"]["calibration"]["noise_sha256"] = hashlib.sha256(
        alternate.read_bytes()).hexdigest()

    result = HostExperimentSpec.parse(raw).preflight(check_environment=False)

    assert not result.ready
    assert sum("lacks exact A/A predecessor" in blocker for blocker in result.blockers) == 3
    assert not any("A/A noise calibration" in blocker for blocker in result.blockers)


_COST_CALIBRATION_ARTIFACTS = (
    ("k1", "k1_artifact", "k1_sha256"),
    ("spike", "spike_artifact", "spike_sha256"),
    ("confirmation_overhead", "confirmation_overhead_artifact",
     "confirmation_overhead_sha256"),
)


def _repoint_calibration_space(raw: dict, root: Path, space: Path) -> None:
    """Rebind all four calibration authorities to one optimization-space path."""
    noise_path = root / "noise.json"
    noise = json.loads(noise_path.read_text(encoding="utf-8"))
    noise["space"] = str(space)
    noise_path.write_text(json.dumps(noise), encoding="utf-8")
    noise_sha = hashlib.sha256(noise_path.read_bytes()).hexdigest()
    raw["search"]["calibration"]["noise_artifact"] = str(noise_path.resolve())
    raw["search"]["calibration"]["noise_sha256"] = noise_sha
    for name, artifact_key, sha_key in _COST_CALIBRATION_ARTIFACTS:
        path = root / f"{name}.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        value["space"] = str(space)
        value["calibration_lineage"]["noise_authority_sha256"] = noise_sha
        path.write_text(json.dumps(value), encoding="utf-8")
        raw["search"]["calibration"][artifact_key] = str(path.resolve())
        raw["search"]["calibration"][sha_key] = hashlib.sha256(
            path.read_bytes()).hexdigest()


def test_preflight_accepts_calibrations_bound_to_a_relocated_identical_space(
        tmp_path: Path) -> None:
    """A protocol tree moved to an immutable snapshot keeps its calibration authorities."""
    raw = _raw()
    root = tmp_path / "calibrations"
    _bind_valid_calibration_fixtures(raw, root)
    frozen = repo_root() / raw["search"]["space"]
    in_place = HostExperimentSpec.parse(deepcopy(raw)).preflight(check_environment=False)
    relocated = tmp_path / "relocated" / frozen.name
    relocated.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(frozen, relocated)
    assert relocated.resolve() != frozen.resolve()
    _repoint_calibration_space(raw, root, relocated)

    result = HostExperimentSpec.parse(raw).preflight(check_environment=False)

    assert not any("optimization space" in blocker
                   for blocker in (*in_place.blockers, *result.blockers))
    assert result.blockers == in_place.blockers
    assert result.errors == in_place.errors


@pytest.mark.parametrize("relocation", ["altered", "absent"])
def test_preflight_rejects_calibrations_whose_recorded_space_is_not_the_frozen_bytes(
        tmp_path: Path, relocation: str) -> None:
    """Relocation is content-addressed: substituted or missing space bytes still fail closed."""
    raw = _raw()
    root = tmp_path / "calibrations"
    _bind_valid_calibration_fixtures(raw, root)
    frozen = repo_root() / raw["search"]["space"]
    relocated = tmp_path / "relocated" / frozen.name
    relocated.parent.mkdir(parents=True, exist_ok=True)
    if relocation == "altered":
        relocated.write_bytes(frozen.read_bytes() + b"\n# tampered\n")
        assert hashlib.sha256(relocated.read_bytes()).hexdigest() != hashlib.sha256(
            frozen.read_bytes()).hexdigest()
    _repoint_calibration_space(raw, root, relocated)

    result = HostExperimentSpec.parse(raw).preflight(check_environment=False)

    assert not result.ready
    assert sum("optimization space" in blocker for blocker in result.blockers) == 4
    assert any("A/A noise calibration does not bind the frozen optimization space"
               in blocker for blocker in result.blockers)


def _fake_complete_campaign(tmp_path: Path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    raw = deepcopy(_raw())
    for path_key, digest_key in (
            ("space", "space_sha256"), ("runner", "runner_sha256"),
            ("trusted_evaluator", "trusted_evaluator_sha256"),
            ("trusted_broker", "trusted_broker_sha256"),
            ("trusted_replay", "trusted_replay_sha256")):
        raw["search"][digest_key] = hashlib.sha256(
            (repo_root() / raw["search"][path_key]).read_bytes()).hexdigest()
    _bind_valid_calibration_fixtures(raw, tmp_path / "calibrations")
    draft = HostExperimentSpec.parse(raw).preflight(check_environment=False)
    draft = _bind_test_environment(raw, tmp_path / "environment", draft)
    assert draft.ready, draft.to_dict()
    raw["status"] = "protocol_frozen"
    raw["freeze"]["protocol_inputs_sha256"] = draft.evidence["protocol_inputs_sha256"]
    source = tmp_path / "protocol_frozen.yaml"
    source.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    runs_root = tmp_path / "runs"
    planned, results = [], []
    frozen_spec = HostExperimentSpec.from_yaml(source)
    grader_layout = _COMPLETE._expected_grader_layout(frozen_spec)
    arms_by_id = {arm.id: arm for arm in frozen_spec.arms}
    for frozen_cell in frozen_spec.agent["launch_plan"]:
            repeat, seed = int(frozen_cell["repeat"]), int(frozen_cell["seed"])
            arm = arms_by_id[str(frozen_cell["arm"])]
            run_id = f"campaign__{arm.id}__r{repeat:02d}__seed{seed:03d}"
            run_dir = runs_root / run_id
            archive = run_dir / "artifacts" / "compiler_submission"
            archive.mkdir(parents=True)
            manifest = {"version": 1, "build": {"command": ["python3", "build.py"]},
                        "compiler": {"command": ["build/compiler", "{input_mlir}",
                                                  "{output_dir}", "{mode}", "{harts}",
                                                  "{vlen_bits}"]},
                        "policy": "policy.yaml"}
            (archive / "manifest.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False))
            (archive / "build.py").write_text("# fixture build\n")
            (archive / "policy.yaml").write_text(
                json.dumps({"arm": arm.id, "repeat": repeat}) + "\n")
            (archive / "compiler.cc").write_text(f"// {arm.id} {repeat}\n")
            if "deterministic_candidate_search" in arm.capabilities:
                search = archive / "search"
                search.mkdir()
                (search / "search_record.json").write_text('{"status":"converged"}\n')
                shutil.copy2(archive / "policy.yaml", search / "selected_policy.json")
            from merlin.benchharness.host_agent import (
                _submission_package_digest, _submission_source_digest)
            policy_sha = hashlib.sha256((archive / "policy.yaml").read_bytes()).hexdigest()
            compiler_seal = {
                "version": 1, "status": "sealed", "policy_sha256": policy_sha,
                "selected_policy_sha256": policy_sha,
                "compiler_source_sha256": _submission_source_digest(archive),
                "compiler_package_sha256": _submission_package_digest(archive),
            }
            search_status = ("pass" if "deterministic_candidate_search" in arm.capabilities
                             else "not_required")
            if search_status == "pass":
                ledger = run_dir / "metrics" / "trusted_search_ledger"
                ledger.mkdir(parents=True)
                (ledger / "index.json").write_text('{"version":1,"evaluations":{"one":{}}}')
                search_seal = {
                    "version": 1, "status": "pass",
                    "checks": {name: True for name in _COMPLETE._TRUSTED_SEARCH_CHECKS},
                    "selected_policy_sha256": hashlib.sha256(
                        (search / "selected_policy.json").read_bytes()).hexdigest(),
                    "search_record_sha256": hashlib.sha256(
                        (search / "search_record.json").read_bytes()).hexdigest(),
                    "trusted_ledger_sha256": hashlib.sha256(
                        (ledger / "index.json").read_bytes()).hexdigest(),
                    "trusted_evaluation_count": 1, "trusted_evaluation_wall_ns": 1,
                    "trusted_broker_wall_ns": 1,
                }
                compiler_seal.update({
                    "search_status": "pass",
                    "search_record_sha256": search_seal["search_record_sha256"],
                })
            else:
                search_seal = {"version": 1, "status": "not_required", "arm": arm.id}
                compiler_seal.update({"search_status": "not_required",
                                      "search_record_sha256": None})
            (run_dir / "contracts").mkdir()
            expected_arm_input = draft.evidence["arm_workspace_inputs"][arm.id]
            input_lock = expected_arm_input["input_lock"]
            input_lock_sha = expected_arm_input["input_lock_sha256"]
            (run_dir / "contracts" / "preflight.json").write_text(json.dumps({
                "ready": True, "errors": [], "blockers": [], "evidence": {
                    "protocol_inputs_sha256": raw["freeze"]["protocol_inputs_sha256"],
                    "frozen_environment": {
                        "manifest_sha256": raw["environment"]["sha256"],
                        "capture_complete": True,
                        "local_identity_matches": True,
                        "k1_identity_matches": True,
                    },
                    "arm_workspace_inputs": {
                        arm.id: expected_arm_input,
                    },
                },
            }))
            (run_dir / "contracts" / "workspace_input_lock.json").write_text(
                json.dumps(input_lock))
            (run_dir / "contracts" / "workspace_input_audit.json").write_text(json.dumps({
                "ok": True, "changed_or_missing": [], "unexpected": [],
                "input_lock_sha256": input_lock_sha,
            }))
            (run_dir / "contracts" / "compiler_seal.json").write_text(
                json.dumps(compiler_seal))
            (run_dir / "contracts" / "trusted_search_seal.json").write_text(
                json.dumps(search_seal))
            metrics = run_dir / "metrics"
            metrics.mkdir(exist_ok=True)
            summary = {
                "agent_success": True, "workspace_inputs_unchanged": True,
                "agent_failure_class": None,
                "aet_reconciled": True, "grader_returncode": 0, "grader_status": "pass",
                "trusted_search_status": search_status, "compiler_seal_status": "sealed",
                "billing_mode": "subscription_notional", "active_wall_seconds": 10.0,
                "grader_wall_seconds": 2.0,
                "trusted_search_wall_seconds": (1e-9 if search_status == "pass" else 0.0),
                "wall_seconds": 13.0,
            }
            (metrics / "summary_metrics.json").write_text(json.dumps(summary))
            def passing_record(level, capsule, mode):
                family = grader_layout["rows"][capsule]["family"]
                checks = {name: True for name in _COMPLETE._LEVEL_CHECKS[level]}
                base = {"capsule": capsule, "mode": mode, "family": family,
                        "checks": checks}
                if level == "L0":
                    source_sha = hashlib.sha256(f"{capsule}:{mode}".encode()).hexdigest()
                    input_sha = hashlib.sha256(f"input:{capsule}:{mode}".encode()).hexdigest()
                    lowered_sha = hashlib.sha256(f"lowered:{capsule}:{mode}".encode()).hexdigest()
                    vector_mode = mode != "scalar"
                    harts = int(grader_layout["rows"][capsule]["core_count"] if
                                mode == "rvv_multicore" else 1)
                    return {**base, "returncode": 0, "wall_seconds": 0.01,
                            "stdout_tail": "", "stderr_tail": "", "ok": True,
                            "metadata": {"version": 1,
                                         "capsule_sha256": grader_layout["rows"][capsule]["sha256"],
                                         "requested_mode": mode, "actual_mode": mode,
                                         "fallback_used": False, "harts": harts,
                                         "vlen_bits": 256,
                                         "vlen_policy": ("scalable_vl" if vector_mode
                                                         else "not_applicable"),
                                         "tail_policy": ("dynamic_vl" if vector_mode
                                                         else "not_applicable"),
                                         "transformations": ["test-lowering"],
                                         "source_sha256": source_sha},
                            "source_sha256": source_sha, "syntax_stderr": "",
                            "verifier_stderr": "", "source_size_bytes": 100,
                            "input_mlir_sha256": input_sha,
                            "lowered_mlir_sha256": lowered_sha,
                            "syntax_returncode": 0, "verifier_returncode": 0,
                            "buffer_plan": grader_layout["buffer_plans"][capsule]}
                if level == "L1":
                    nonce = 7
                    trials = [{"seed": index + 1, "returncode": 0, "wall_seconds": 0.01,
                               "stdout_tail": ("MERLIN_TRUSTED_RESULT version=1 "
                                               f"seed={index + 1} nonce={nonce} "
                                               "memory=1 numeric=1\n"), "stderr_tail": "",
                               "ok": True} for index in range(3)]
                    return {**base, "status": "pass", "build_wall_seconds": 0.01,
                            "build_stderr_tail": "", "build_returncode": 0,
                            "receipt_nonce": nonce, "trials": trials}
                if level == "L2":
                    return {**base, "status": "pass", "tail_case": True, "seed": 1,
                            "receipt_nonce": 7, "trusted_receipt": True,
                            "vector_instructions": ["vsetvli", "vadd.vv"],
                            "vector_dataflow": {
                                "version": 1, "function_found": True, "useful": True,
                                "source_vector_loads": ["vle32.v"],
                                "computed_vector_registers": ["v8"],
                                "output_vector_stores": ["vse32.v"],
                                "output_scalar_stores": [],
                                "output_scalar_overwrites": [],
                                "required_execution_pcs": [1, 2, 3],
                                "vector_instructions": ["vsetvli", "vadd.vv"],
                            },
                            "linked_vector_dataflow": {
                                "version": 1, "function_found": True, "useful": True,
                                "source_vector_loads": ["vle32.v"],
                                "computed_vector_registers": ["v8"],
                                "output_vector_stores": ["vse32.v"],
                                "output_scalar_stores": [],
                                "output_scalar_overwrites": [],
                                "required_execution_pcs": [1, 2, 3],
                                "vector_instructions": ["vsetvli", "vadd.vv"],
                            },
                            "executed_vector_dataflow": True,
                            "required_pc_trace_lines": [
                                "core 0: 0x1", "core 0: 0x2", "core 0: 0x3"],
                            "spike_trace_sha256": "b" * 64,
                            "kernel_text_sha256": "a" * 64,
                            "build_logs": [{"returncode": 0, "wall_seconds": 0.01,
                                            "stderr_tail": ""} for _ in range(6)],
                            "spike_cycles": 1, "spike_returncode": 0,
                            "wall_seconds": 0.01,
                            "stdout_tail": ("MERLIN_TRUSTED_RESULT version=1 seed=1 "
                                            "nonce=7 vlenb=32 cycles=1 calls=20\n"),
                            "stderr_tail": ""}
                harts = 8 if mode == "rvv_multicore" else 1
                output_count = int(grader_layout["buffer_plans"][capsule]["output_count"])
                shard_floor = output_count // harts
                shard_ceil = (output_count + harts - 1) // harts
                affinity = "0" if harts == 1 else f"0-{harts-1}"
                monitor = {"version": 1, "returncode": 0, "timed_out": False,
                           "wall_ns": 1, "requested_harts": harts, "max_tasks": harts + 1,
                           "tids_observed": harts + 1, "active_tids": harts,
                           "cpus_observed": list(range(harts)),
                           "active_cpus": list(range(harts)),
                           "affinity_samples": [affinity],
                           "pinned_affinities_observed": list(range(harts)),
                           "pinned_runtime_cpus": list(range(harts)),
                           "running_cpus_observed": list(range(harts)),
                           "max_simultaneous_running_cpus": harts,
                           "peak_rss_kb": 1,
                           "child_stdout": ("K1_METRIC vlenb 32\n"
                                            f"K1_METRIC affinity_count {harts}\n"
                                            "K1_METRIC wall_ns 1\n"
                                            "K1_METRIC time_ticks 1\n"
                                            "K1_METRIC calls 20\n"
                                            "K1_METRIC audit_call 7\n"
                                            "K1_METRIC audit_wall_ns 1\n"
                                            "K1_METRIC audit_time_ticks 1\n"
                                            "K1_METRIC correctness_checks 21\n"
                                            f"K1_METRIC pinned_hart_mask {(1 << harts) - 1 if harts > 1 else 0}\n"
                                            f"K1_METRIC worker_hart_mask {((1 << harts) - 1) & ~1 if harts > 1 else 0}\n"
                                            f"K1_METRIC productive_worker_hart_mask {((1 << harts) - 1) & ~1 if harts > 1 else 0}\n"
                                            f"K1_METRIC pthread_create_attempts {harts - 1 if harts > 1 else 0}\n"
                                            f"K1_METRIC pthread_creates {harts - 1 if harts > 1 else 0}\n"
                                            "K1_METRIC pthread_create_failures 0\n"
                                            f"K1_METRIC pthread_completions {harts - 1 if harts > 1 else 0}\n"
                                            f"K1_METRIC pthread_affinity_attempts {harts if harts > 1 else 0}\n"
                                            f"K1_METRIC pthread_affinity_successes {harts if harts > 1 else 0}\n"
                                            "K1_METRIC pthread_affinity_failures 0\n"
                                            f"K1_METRIC minimum_worker_cpu_ns {100 if harts > 1 else 0}\n"
                                            f"K1_METRIC counterfactual_create_attempts {harts - 1 if harts > 1 else 0}\n"
                                            f"K1_METRIC counterfactual_creates {harts - 1 if harts > 1 else 0}\n"
                                            "K1_METRIC counterfactual_create_failures 0\n"
                                            f"K1_METRIC counterfactual_suppressed_starts {harts - 1 if harts > 1 else 0}\n"
                                            "K1_METRIC counterfactual_worker_dependence 1\n"
                                            f"K1_METRIC audit_serialized_callbacks {harts - 1}\n"
                                            f"K1_METRIC audit_output_elements {output_count}\n"
                                            f"K1_METRIC audit_output_coverage {output_count}\n"
                                            f"K1_METRIC audit_owner_min_elements {shard_floor}\n"
                                            f"K1_METRIC audit_owner_max_elements {shard_ceil}\n"
                                            "K1_METRIC audit_ownership_violations 0\n"
                                            "K1_METRIC audit_balanced_shards 1\n"
                                            "K1_METRIC peak_rss_kb 1\n"
                                            "MERLIN_TRUSTED_RESULT version=1 seed=1 "
                                            "nonce=7 memory=1 numeric=1\n"), "child_stderr": ""}
                metrics_evidence = {"vlenb": 32, "affinity_count": harts, "wall_ns": 1,
                                    "time_ticks": 1, "calls": 20,
                                    "audit_call": 7, "audit_wall_ns": 1,
                                    "audit_time_ticks": 1, "correctness_checks": 21,
                                    "pinned_hart_mask": ((1 << harts) - 1
                                                         if harts > 1 else 0),
                                    "worker_hart_mask": (((1 << harts) - 1) & ~1
                                                         if harts > 1 else 0),
                                    "productive_worker_hart_mask": (
                                        ((1 << harts) - 1) & ~1 if harts > 1 else 0),
                                    "pthread_create_attempts": harts - 1 if harts > 1 else 0,
                                    "pthread_creates": harts - 1 if harts > 1 else 0,
                                    "pthread_create_failures": 0,
                                    "pthread_completions": harts - 1 if harts > 1 else 0,
                                    "pthread_affinity_attempts": harts if harts > 1 else 0,
                                    "pthread_affinity_successes": harts if harts > 1 else 0,
                                    "pthread_affinity_failures": 0,
                                    "minimum_worker_cpu_ns": 100 if harts > 1 else 0,
                                    "counterfactual_create_attempts": (
                                        harts - 1 if harts > 1 else 0),
                                    "counterfactual_creates": harts - 1 if harts > 1 else 0,
                                    "counterfactual_create_failures": 0,
                                    "counterfactual_suppressed_starts": (
                                        harts - 1 if harts > 1 else 0),
                                    "counterfactual_worker_dependence": 1,
                                    "audit_serialized_callbacks": harts - 1,
                                    "audit_output_elements": output_count,
                                    "audit_output_coverage": output_count,
                                    "audit_owner_min_elements": shard_floor,
                                    "audit_owner_max_elements": shard_ceil,
                                    "audit_ownership_violations": 0,
                                    "audit_balanced_shards": 1,
                                    "peak_rss_kb": 1}
                return {**base, "status": "pass", "harts": harts,
                        "build_wall_seconds": 0.01, "build_stderr_tail": "",
                        "build_returncode": 0, "seed": 1, "receipt_nonce": 7,
                        "metrics": metrics_evidence,
                        "monitor": monitor, "kernel_text_sha256": "b" * 64,
                        "local_sha256": "a" * 64,
                        "remote_sha256": "a" * 64, "board_wall_seconds": 0.01,
                        "ssh_returncode": 0, "ssh_stderr_tail": ""}

            level_records = {
                level: [passing_record(level, capsule, mode)
                        for capsule, mode in grader_layout["records"][level]]
                for level in ("L0", "L1", "L2", "L3")}
            (metrics / "grader_result.json").write_text(json.dumps({
                "version": 1, "status": "pass", "wall_seconds": 2.0,
                "implemented_levels": ["L0", "L1", "L2", "L3"],
                "selected_capsules": grader_layout["selected"],
                "tail_capsules": grader_layout["tails"],
                "multicore_capsule": grader_layout["multicore"],
                "build": {"commands": [{"command": ["python3", "build.py"],
                                           "returncode": 0}],
                          "policy_sha256": compiler_seal["policy_sha256"]},
                "compiler_seal": {"status": "pass", "checks": {
                    "sealed": True, "policy_sha256": True, "compiler_source_sha256": True,
                    "compiler_package_sha256": True},
                    "seal_sha256": hashlib.sha256((
                        run_dir / "contracts" / "compiler_seal.json").read_bytes()).hexdigest()},
                "contracts": {
                    "target_contract": {
                        "path": str(frozen_spec._repo_path(frozen_spec.target_contract).resolve()),
                        "sha256": hashlib.sha256(frozen_spec._repo_path(
                            frozen_spec.target_contract).read_bytes()).hexdigest()},
                    "dialect_plan": {
                        "path": str(frozen_spec._repo_path(frozen_spec.dialect_plan).resolve()),
                        "sha256": hashlib.sha256(frozen_spec._repo_path(
                            frozen_spec.dialect_plan).read_bytes()).hexdigest()},
                },
                "trusted_search": ({"status": "pass", "checks": {
                    "driver_verified": True, "policy_byte_match": True,
                    "independent_convergence_sweep": True, "deterministic_replay": True,
                    "heldout_never_opened": True},
                    "seal_sha256": hashlib.sha256((
                        run_dir / "contracts" / "trusted_search_seal.json").read_bytes()).hexdigest()}
                    if search_status == "pass" else {"status": "not_required"}),
                "levels": {
                    "L0": {"status": "pass", "records": level_records["L0"],
                           "scalar_rvv_source_change": {
                               capsule: True for capsule in grader_layout["selected"]}},
                    "L1": {"status": "pass", "records": level_records["L1"],
                           "authority": "native_scalar_reference_with_asan_ubsan_and_guards"},
                    "L2": {"status": "pass", "records": level_records["L2"],
                           "authority": "spike_rv64gcv_vlen256"},
                    "L3": {"status": "pass", "records": level_records["L3"],
                           "authority": "spacemit_k1_linux_csr_and_proc_monitor"},
                },
            }))
            (metrics / "driver_wall_timing.json").write_text(json.dumps({
                "version": 1, "authority": "driver_monotonic_ns",
                "start_monotonic_ns": 1_000_000_000,
                "end_monotonic_ns": 14_000_000_000,
                "wall_seconds": 13.0,
            }))
            token_turn = {
                "turn": 0, "source_event": "turn.completed", "input_tokens": 10,
                "cached_input_tokens": 5, "cache_write_input_tokens": 1,
                "output_tokens": 3, "reasoning_output_tokens": 2,
                "uncached_input_tokens": 4, "provider_reported": True, "event_index": 1,
            }
            (metrics / "codex_reconciliation.json").write_text(json.dumps({
                "ok": True,
                "raw_events": {"reconciled": True, "raw_event_count": 1},
                "token_ledger": {
                    "num_turns": 1, "all_match": True, "subset_invariants_hold": True,
                    "checks": {
                        "uncached_input": {"ledger": 4}, "cache_read": {"ledger": 5},
                        "cache_write": {"ledger": 1}, "output": {"ledger": 3},
                        "reasoning": {"ledger": 2},
                    },
                },
            }))
            (metrics / "token_ledger.jsonl").write_text(json.dumps(token_turn) + "\n")
            agent = run_dir / "agent"
            agent.mkdir()
            (agent / "tools.jsonl").write_text("")
            attempts = [{"index": 0, "tools": [], "turns": [{
                key: value for key, value in token_turn.items() if key != "turn"}]}]
            (agent / "run_result.json").write_text(json.dumps({
                "attempts": attempts, "active_wall_s": 10.0,
                "resolved_model": "gpt-5.6-sol", "requested_model": "gpt-5.6-sol",
                "status": "completed", "usage_complete": True}))
            raw_event = json.dumps({"type": "turn.completed", "usage": {
                "input_tokens": 10, "cached_input_tokens": 5,
                "cache_write_input_tokens": 1, "output_tokens": 3,
                "reasoning_output_tokens": 2,
            }}, separators=(",", ":"))
            raw_directory = agent / "aet_raw"
            raw_directory.mkdir()
            (raw_directory / "attempt_0000.jsonl").write_text(raw_event + "\n")
            timestamped_directory = agent / "aet_timestamped"
            timestamped_directory.mkdir()
            (timestamped_directory / "attempt_0000.jsonl").write_text(json.dumps({
                "ts": "2026-01-01T00:00:00+00:00", "line": raw_event}) + "\n")
            (run_dir / "run_record.json").write_text(json.dumps({
                "run_id": run_id, "project": "merlin", "suite": "k1_cpu/cpu-host-compiler",
                "target": "k1_cpu", "method": arm.id, "seed": seed,
                "experiment": frozen_spec.label, "arm": arm.id,
                "model": frozen_spec.agent["model"],
                "billing_mode": frozen_spec.agent["billing"],
                "environment_manifest_sha256": raw["environment"]["sha256"],
                "analysis_plan_sha256": frozen_spec.analysis["sha256"],
                "provider_sampling_seeded": False,
                "spec": str(source.resolve())}))
            item = {"ordinal": int(frozen_cell["ordinal"]),
                    "arm": arm.id, "repeat": repeat, "seed": seed,
                    "run_id": run_id, "run_dir": str(run_dir)}
            planned.append(item)
            results.append({**item, "returncode": 0, "run_identity_ok": True})
    claim_dir = runs_root / ".protocol_claims"
    claim_dir.mkdir(parents=True)
    claim = claim_dir / f"{raw['freeze']['protocol_inputs_sha256']}.json"
    claim.write_text(json.dumps({
        "version": 1, "status": "bound",
        "protocol_inputs_sha256": raw["freeze"]["protocol_inputs_sha256"],
        "environment_manifest_sha256": raw["environment"]["sha256"],
        "analysis_plan_sha256": frozen_spec.analysis["sha256"],
        "spec_path": str(source.resolve()),
        "spec_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "campaign_run_id": "campaign",
    }))
    cells = claim.with_name(f"{claim.stem}.cells")
    cells.mkdir()
    for row in planned:
        (cells / f"{int(row['ordinal']):02d}.consumed.json").write_text(json.dumps({
            "version": 1, "status": "authorized",
            "protocol_inputs_sha256": raw["freeze"]["protocol_inputs_sha256"],
            "environment_manifest_sha256": raw["environment"]["sha256"],
            "analysis_plan_sha256": frozen_spec.analysis["sha256"],
            "campaign_run_id": "campaign", "ordinal": int(row["ordinal"]),
            "arm": row["arm"], "repeat": int(row["repeat"]),
            "seed": int(row["seed"]), "run_id": row["run_id"],
        }))
    launch = runs_root / "campaign" / "contracts" / "launch.json"
    launch.parent.mkdir(parents=True)
    boundary_root = launch.parent / "block_boundaries"
    boundary_root.mkdir()
    environment = frozen_spec.search_space_config()["board_environment"]
    block_boundaries = []
    for block in range(4):
        qualifying = {
            "verdict": "GO", "ready": True, "errors": [], "blockers": [], "warnings": [],
            "evidence": {
                "protocol_inputs_sha256": raw["freeze"]["protocol_inputs_sha256"],
                "analysis_plan_sha256": frozen_spec.analysis["sha256"],
                "k1_board_state_ready": True,
                "k1_board_state_probe": {"authority": "driver_ssh_sysfs_procfs"},
            },
        }
        receipt = {
            "version": 1, "authority": "frozen_k1_board_environment_gate",
            "block": block, "first_ordinal": block * 4,
            "mandatory_washout_seconds": float(environment["settle_interval_seconds"]),
            "stabilization_attempt_limit": int(environment["settle_attempts"]),
            "board_environment": environment, "attempts": [qualifying],
            "qualifying_attempt_index": 0, "ready": True,
        }
        receipt_path = boundary_root / f"{block:02d}.json"
        receipt_path.write_text(json.dumps(receipt))
        block_boundaries.append({
            "block": block, "first_ordinal": block * 4,
            "path": str(receipt_path.resolve()),
            "sha256": hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
        })
    launch.write_text(json.dumps({
        "version": 2, "sequential": True,
        "terminal_failure_policy": "record_and_continue",
        "retry_terminal_outcomes": False, "campaign_run_id": "campaign",
        "launch_seed": frozen_spec.agent["launch_seed"],
        "launch_seed_role": frozen_spec.agent["launch_seed_role"],
        "provider_sampling_seeded": False,
        "run_id_scheme": "{campaign_run_id}__{arm}__r{repeat:02d}__seed{seed:03d}",
        "runs_root": str(runs_root),
        "authorization_claim": str(claim.resolve()),
        "authorization_claim_sha256": hashlib.sha256(claim.read_bytes()).hexdigest(),
        "protocol_inputs_sha256": raw["freeze"]["protocol_inputs_sha256"],
        "environment_manifest_sha256": raw["environment"]["sha256"],
        "analysis_plan_sha256": frozen_spec.analysis["sha256"],
        "block_boundaries": block_boundaries,
        "planned": planned, "results": results}, indent=2))
    return source, launch, results


def _mark_fake_run_as_graded_failure(launch: Path, results: list[dict], *, arm: str,
                                     repeat: int) -> dict:
    """Turn one complete fake run into a genuine, fully retained grader failure."""
    result = next(row for row in results if row["arm"] == arm and row["repeat"] == repeat)
    result["returncode"] = 1
    run_dir = Path(result["run_dir"])
    summary_path = run_dir / "metrics" / "summary_metrics.json"
    summary = json.loads(summary_path.read_text())
    summary.update({"grader_returncode": 1, "grader_status": "fail"})
    summary_path.write_text(json.dumps(summary))
    grader_path = run_dir / "metrics" / "grader_result.json"
    grader = json.loads(grader_path.read_text())
    grader["status"] = "fail"
    grader["levels"]["L3"]["status"] = "fail"
    failed_record = grader["levels"]["L3"]["records"][0]
    failed_record["checks"]["numeric_correctness"] = False
    failed_record["monitor"]["returncode"] = 1
    failed_record["monitor"]["child_stdout"] = failed_record["monitor"][
        "child_stdout"].replace("PASS seed=1", "FAIL seed=1")
    failed_record["status"] = "fail"
    grader_path.write_text(json.dumps(grader))
    raw = json.loads(launch.read_text())
    raw["results"] = results
    launch.write_text(json.dumps(raw))
    return result


def _upgrade_fake_launch_to_v3(launch: Path, results: list[dict]) -> None:
    raw = json.loads(launch.read_text())
    raw["version"] = 3
    for result in results:
        run_dir = Path(result["run_dir"])
        summary_path = run_dir / "metrics" / "summary_metrics.json"
        summary = json.loads(summary_path.read_text())
        outcome = "graded_pass" if result["returncode"] == 0 else "graded_fail"
        result.update({"attempted": True, "executed": True, "cell_status": "executed",
                       "terminal_class": outcome})
        summary["terminal_class"] = outcome
        summary_path.write_text(json.dumps(summary))
        input_audit = json.loads(
            (run_dir / "contracts" / "workspace_input_audit.json").read_text())
        search = json.loads((run_dir / "contracts" / "trusted_search_seal.json").read_text())
        compiler = json.loads((run_dir / "contracts" / "compiler_seal.json").read_text())
        grader = json.loads((run_dir / "metrics" / "grader_result.json").read_text())
        reconciliation = json.loads(
            (run_dir / "metrics" / "codex_reconciliation.json").read_text())
        (run_dir / "contracts" / "terminal_outcome.json").write_text(json.dumps({
            "version": 1, "run_id": result["run_id"], "arm": result["arm"],
            "terminal_class": outcome, "paper_evidence_eligible": True,
            "promotion_eligible": outcome == "graded_pass",
            "checks": {
                "agent_success": summary["agent_success"],
                "agent_failure_class": summary.get("agent_failure_class"),
                "workspace_input_audit": input_audit["ok"],
                "aet_reconciled": reconciliation["ok"],
                "trusted_search_status": search["status"],
                "compiler_seal_status": compiler["status"],
                "compiler_seal_failure_class": compiler.get("failure_class"),
                "grader_returncode": summary["grader_returncode"],
                "grader_status": grader["status"],
                "grader_failure_class": grader.get("failure_class"),
            },
        }))
    raw["results"] = results
    launch.write_text(json.dumps(raw))


def _mark_fake_search_failure(launch: Path, results: list[dict]) -> dict:
    result = next(row for row in results
                  if row["arm"] == "arm3_generated_cpu_dialect" and row["repeat"] == 0)
    run_dir = Path(result["run_dir"])
    result.update(returncode=1, terminal_class="treatment_search_fail")
    search = {"version": 1, "status": "fail",
              "failure_class": "treatment_search_fail",
              "reason": "agent did not publish converged search artifacts", "checks": {}}
    compiler = {"version": 1, "status": "not_run",
                "reason": "trusted search did not verify"}
    grader = {"version": 1, "status": "not_run", "wall_seconds": 0.0,
              "reason": "trusted search/compiler seal did not verify"}
    (run_dir / "contracts" / "trusted_search_seal.json").write_text(json.dumps(search))
    (run_dir / "contracts" / "compiler_seal.json").write_text(json.dumps(compiler))
    (run_dir / "metrics" / "grader_result.json").write_text(json.dumps(grader))
    shutil.rmtree(run_dir / "artifacts" / "compiler_submission")
    summary_path = run_dir / "metrics" / "summary_metrics.json"
    summary = json.loads(summary_path.read_text())
    summary.update({"trusted_search_status": "fail", "compiler_seal_status": "not_run",
                    "grader_status": "not_run", "grader_returncode": 2,
                    "grader_wall_seconds": 0.0, "trusted_search_wall_seconds": 0.0,
                    "terminal_class": "treatment_search_fail"})
    summary_path.write_text(json.dumps(summary))
    input_audit = json.loads(
        (run_dir / "contracts" / "workspace_input_audit.json").read_text())
    reconciliation = json.loads(
        (run_dir / "metrics" / "codex_reconciliation.json").read_text())
    (run_dir / "contracts" / "terminal_outcome.json").write_text(json.dumps({
        "version": 1, "run_id": result["run_id"], "arm": result["arm"],
        "terminal_class": "treatment_search_fail", "paper_evidence_eligible": False,
        "promotion_eligible": False,
        "checks": {"agent_success": summary["agent_success"],
                   "agent_failure_class": summary.get("agent_failure_class"),
                   "workspace_input_audit": input_audit["ok"],
                   "aet_reconciled": reconciliation["ok"],
                   "trusted_search_status": "fail", "compiler_seal_status": "not_run",
                   "compiler_seal_failure_class": compiler.get("failure_class"),
                   "grader_returncode": 2, "grader_status": "not_run",
                   "grader_failure_class": grader.get("failure_class")},
    }))
    raw = json.loads(launch.read_text()); raw["results"] = results
    launch.write_text(json.dumps(raw))
    return result


def test_launcher_executes_every_planned_attempt_after_a_nonzero_result(tmp_path, monkeypatch):
    plan = []
    for ordinal in range(3):
        run_dir = tmp_path / f"run-{ordinal}"
        run_dir.mkdir()
        (run_dir / "run_record.json").write_text(json.dumps({"run_id": f"run-{ordinal}"}))
        plan.append({"arm": f"arm-{ordinal}", "repeat": 0, "seed": 1,
                     "run_id": f"run-{ordinal}", "run_dir": str(run_dir)})
    returncodes = iter((1, 0, 1))
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return type("Result", (), {"returncode": next(returncodes)})()

    monkeypatch.setattr(_LAUNCH.subprocess, "run", fake_run)
    got = _LAUNCH._execute_live_plan(plan, lambda row: ["runner", row["run_id"]])
    assert len(calls) == 3
    assert [row["returncode"] for row in got] == [1, 0, 1]
    assert all(row["run_identity_ok"] is True for row in got)


def test_live_protocol_claim_is_one_shot(tmp_path):
    (tmp_path / "frozen.yaml").write_text("status: protocol_frozen\n")
    first = _LAUNCH._claim_protocol_once(
        tmp_path, protocol_sha256="a" * 64, environment_manifest_sha256="b" * 64,
        analysis_plan_sha256="c" * 64,
        spec_path=tmp_path / "frozen.yaml")
    assert first.is_file()
    with pytest.raises(FileExistsError, match="already claimed"):
        _LAUNCH._claim_protocol_once(
            tmp_path, protocol_sha256="a" * 64, environment_manifest_sha256="b" * 64,
            analysis_plan_sha256="c" * 64,
            spec_path=tmp_path / "frozen.yaml")


def test_exact_arm_cell_authorization_is_consumed_once(tmp_path):
    raw = deepcopy(_raw())
    draft = HostExperimentSpec.parse(raw).preflight(check_environment=False)
    raw["status"] = "protocol_frozen"
    raw["freeze"]["protocol_inputs_sha256"] = draft.evidence["protocol_inputs_sha256"]
    source = tmp_path / "frozen.yaml"
    source.write_text(yaml.safe_dump(raw, sort_keys=False))
    spec = HostExperimentSpec.from_yaml(source)
    claim = _LAUNCH._claim_protocol_once(
        tmp_path, protocol_sha256=spec.freeze["protocol_inputs_sha256"],
        environment_manifest_sha256=str(spec.environment["sha256"]),
        analysis_plan_sha256=str(spec.analysis["sha256"]), spec_path=source)
    campaign = "campaign"
    _LAUNCH._bind_protocol_claim(claim, campaign, spec.agent["launch_plan"])
    cell = spec.agent["launch_plan"][0]
    run_id = (f"{campaign}__{cell['arm']}__r{cell['repeat']:02d}__seed{cell['seed']:03d}")
    # Point the protocol's output root at the temporary claim root without altering frozen semantics.
    object.__setattr__(spec, "telemetry", {**spec.telemetry, "output_layout": str(tmp_path)})
    authorized, consumed = _RUN._authorization_cell(
        spec, source, cell["arm"], cell["seed"], run_id, claim)
    _RUN._consume_authorization_cell(authorized, consumed)
    assert consumed.is_file() and not authorized.exists()
    with pytest.raises(ValueError, match="already been consumed"):
        _RUN._authorization_cell(spec, source, cell["arm"], cell["seed"], run_id, claim)


def test_campaign_completion_requires_all_4x4_and_selects_without_outcomes(tmp_path):
    source, launch, results = _fake_complete_campaign(tmp_path / "unretained")
    output = tmp_path / "campaign_complete.yaml"
    completed = complete_campaign(source, launch, output)
    assert completed.status == "campaign_complete"
    record = completed.freeze["campaign_record"]
    assert record["completed_run_count"] == 16
    assert len(record["block_boundary_receipts"]) == 4
    assert record["selection"]["selected_run_id"] == next(
        row["run_id"] for row in results
        if row["arm"] == "arm4_agentic_pass_authoring" and row["repeat"] == 0)
    assert record["selection"]["selection_outcome_fields_used"] == []
    assert record["selection"]["heldout_outcome_used"] is False
    assert record["telemetry"]["tokens"] == {
        "input_tokens": 160, "cached_input_tokens": 80,
        "cache_write_input_tokens": 16, "output_tokens": 48,
        "reasoning_output_tokens": 32, "uncached_input_tokens": 64}
    assert record["telemetry"]["tool_calls"] == 0


def test_campaign_completion_rejects_controller_exclusion_tombstone(tmp_path):
    source, launch, _ = _fake_complete_campaign(tmp_path)
    launch_value = json.loads(launch.read_text())
    exclusion = (Path(launch_value["runs_root"]) / ".campaign_exclusions" /
                 f'{launch_value["campaign_run_id"]}.json')
    exclusion.parent.mkdir()
    exclusion.write_text("{}", encoding="utf-8")
    output = tmp_path / "must_not_exist.yaml"
    with pytest.raises(ValueError, match="controller-excluded campaign"):
        complete_campaign(source, launch, output)
    assert not output.exists()


def test_campaign_completion_rechecks_exclusion_immediately_before_publish(
        tmp_path, monkeypatch):
    source, launch, _ = _fake_complete_campaign(tmp_path)
    launch_value = json.loads(launch.read_text())
    exclusion = (Path(launch_value["runs_root"]) / ".campaign_exclusions" /
                 f'{launch_value["campaign_run_id"]}.json')
    real_check = _COMPLETE._assert_campaign_not_excluded
    calls = 0

    def publish_tombstone_on_final_check(runs_root, campaign_run_id):
        nonlocal calls
        calls += 1
        if calls == 2:
            exclusion.parent.mkdir()
            exclusion.write_text("{}", encoding="utf-8")
        return real_check(runs_root, campaign_run_id)

    monkeypatch.setattr(_COMPLETE, "_assert_campaign_not_excluded",
                        publish_tombstone_on_final_check)
    output = tmp_path / "must_not_exist.yaml"
    with pytest.raises(ValueError, match="controller-excluded campaign"):
        complete_campaign(source, launch, output)
    assert calls == 2
    assert not output.exists()


def test_campaign_completion_rejects_tampered_block_boundary_receipt(tmp_path):
    source, launch, _ = _fake_complete_campaign(tmp_path)
    launch_value = json.loads(launch.read_text())
    receipt = Path(launch_value["block_boundaries"][0]["path"])
    value = json.loads(receipt.read_text())
    value["mandatory_washout_seconds"] = 0
    receipt.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="block-boundary"):
        complete_campaign(source, launch, tmp_path / "must_not_exist.yaml")


def test_campaign_environment_binding_reaches_claim_cells_launch_runs_and_live_preflight(tmp_path):
    source, launch, results = _fake_complete_campaign(tmp_path)
    spec = HostExperimentSpec.from_yaml(source)
    expected = spec.environment["sha256"]
    launch_value = json.loads(launch.read_text())
    claim_path = Path(launch_value["authorization_claim"])
    claim = json.loads(claim_path.read_text())
    cells = claim_path.with_name(f"{claim_path.stem}.cells")
    first_cell = json.loads(next(cells.glob("*.consumed.json")).read_text())
    first_run = Path(results[0]["run_dir"])
    run_record_path = first_run / "run_record.json"
    preflight_path = first_run / "contracts" / "preflight.json"
    assert launch_value["environment_manifest_sha256"] == expected
    assert claim["environment_manifest_sha256"] == expected
    assert first_cell["environment_manifest_sha256"] == expected
    assert json.loads(run_record_path.read_text())["environment_manifest_sha256"] == expected
    assert json.loads(preflight_path.read_text())["evidence"]["frozen_environment"] == {
        "manifest_sha256": expected, "capture_complete": True,
        "local_identity_matches": True, "k1_identity_matches": True,
    }

    launch_value["environment_manifest_sha256"] = "0" * 64
    launch.write_text(json.dumps(launch_value))
    with pytest.raises(ValueError, match="different frozen environment"):
        complete_campaign(source, launch, tmp_path / "bad-launch.yaml")
    launch_value["environment_manifest_sha256"] = expected
    launch.write_text(json.dumps(launch_value))

    run_record = json.loads(run_record_path.read_text())
    run_record.pop("environment_manifest_sha256")
    run_record_path.write_text(json.dumps(run_record))
    with pytest.raises(ValueError, match="run identity does not match"):
        complete_campaign(source, launch, tmp_path / "bad-run.yaml")
    run_record["environment_manifest_sha256"] = expected
    run_record_path.write_text(json.dumps(run_record))

    preflight = json.loads(preflight_path.read_text())
    preflight["evidence"]["frozen_environment"]["k1_identity_matches"] = False
    preflight_path.write_text(json.dumps(preflight))
    with pytest.raises(ValueError, match="preflight/workspace identity"):
        complete_campaign(source, launch, tmp_path / "bad-preflight.yaml")


def test_campaign_completion_retains_failed_comparison_arms_without_retry_or_selection(tmp_path):
    source, launch, results = _fake_complete_campaign(tmp_path)
    failed = _mark_fake_run_as_graded_failure(
        launch, results, arm="arm1_raw_cpp", repeat=0)
    completed = complete_campaign(source, launch, tmp_path / "campaign_complete.yaml")
    assert completed.status == "campaign_complete"
    record = completed.freeze["campaign_record"]
    observed = next(row for row in record["runs"] if row["run_id"] == failed["run_id"])
    assert observed["outcome"] == "graded_fail"
    assert record["outcome_counts"] == {"graded_pass": 15, "graded_fail": 1}
    assert record["selection"]["selected_run_id"] != failed["run_id"]
    assert record["selection"]["selection_outcome_fields_used"] == []
    assert record["retries_after_observed_failure"] == 0


@pytest.mark.parametrize("tamper", [
    "monitor_timeout", "missing_time_ticks", "stale_per_call", "excess_worker_attempt",
    "failed_worker_attempt", "empty_worker", "busy_decoy_workers",
])
def test_campaign_completion_rejects_incomplete_k1_monitor_authority(tmp_path, tamper):
    source, launch, results = _fake_complete_campaign(tmp_path)
    grader_path = Path(results[0]["run_dir"]) / "metrics" / "grader_result.json"
    grader = json.loads(grader_path.read_text())
    l3 = (next(row for row in grader["levels"]["L3"]["records"] if row["harts"] > 1)
          if tamper in {"excess_worker_attempt", "failed_worker_attempt", "empty_worker",
                        "busy_decoy_workers"}
          else grader["levels"]["L3"]["records"][0])
    if tamper == "monitor_timeout":
        l3["monitor"]["timed_out"] = True
    elif tamper == "missing_time_ticks":
        del l3["metrics"]["time_ticks"]
        l3["monitor"]["child_stdout"] = "\n".join(
            line for line in l3["monitor"]["child_stdout"].splitlines()
            if not line.startswith("K1_METRIC time_ticks ")) + "\n"
    else:
        field, value = {
            "stale_per_call": ("correctness_checks", 1),
            "excess_worker_attempt": ("pthread_create_attempts", 8),
            "failed_worker_attempt": ("pthread_create_failures", 1),
            "empty_worker": ("minimum_worker_cpu_ns", 0),
            "busy_decoy_workers": ("counterfactual_worker_dependence", 0),
        }[tamper]
        l3["metrics"][field] = value
        l3["monitor"]["child_stdout"] = "\n".join(
            f"K1_METRIC {field} {value}" if line.startswith(f"K1_METRIC {field} ") else line
            for line in l3["monitor"]["child_stdout"].splitlines()) + "\n"
    grader_path.write_text(json.dumps(grader))
    with pytest.raises(ValueError, match="grader L3"):
        complete_campaign(source, launch, tmp_path / f"campaign-{tamper}.yaml")


def test_campaign_completion_rejects_downstream_failure_when_matching_l0_passed(tmp_path):
    source, launch, results = _fake_complete_campaign(tmp_path)
    result = results[0]
    result["returncode"] = 1
    run_dir = Path(result["run_dir"])
    summary_path = run_dir / "metrics" / "summary_metrics.json"
    summary = json.loads(summary_path.read_text())
    summary.update({"grader_returncode": 1, "grader_status": "fail"})
    summary_path.write_text(json.dumps(summary))
    grader_path = run_dir / "metrics" / "grader_result.json"
    grader = json.loads(grader_path.read_text())
    grader["status"] = "fail"
    grader["levels"]["L1"]["status"] = "fail"
    passed_l1 = grader["levels"]["L1"]["records"][0]
    grader["levels"]["L1"]["records"][0] = {
        "capsule": passed_l1["capsule"], "family": passed_l1["family"],
        "mode": "scalar", "status": "fail", "reason": "L0 scalar artifact failed",
    }
    grader_path.write_text(json.dumps(grader))
    launch_payload = json.loads(launch.read_text())
    launch_payload["results"] = results
    launch.write_text(json.dumps(launch_payload))
    with pytest.raises(ValueError, match="matching L0 passed"):
        complete_campaign(source, launch, tmp_path / "must_not_exist.yaml")


def test_campaign_completion_rejects_infrastructure_failure_as_treatment_failure(tmp_path):
    source, launch, results = _fake_complete_campaign(tmp_path)
    result = results[0]
    result["returncode"] = 1
    run_dir = Path(result["run_dir"])
    summary_path = run_dir / "metrics" / "summary_metrics.json"
    summary = json.loads(summary_path.read_text())
    summary.update({"grader_returncode": 1, "grader_status": "fail"})
    summary_path.write_text(json.dumps(summary))
    grader_path = run_dir / "metrics" / "grader_result.json"
    grader = json.loads(grader_path.read_text())
    grader["status"] = "fail"
    grader["levels"]["L2"]["status"] = "fail"
    passed_l2 = grader["levels"]["L2"]["records"][0]
    grader["levels"]["L2"]["records"][0] = {
        "capsule": passed_l2["capsule"], "family": passed_l2["family"],
        "mode": "rvv", "tail_case": True, "status": "fail",
        "reason": "Spike tools absent: ['spike']",
    }
    grader_path.write_text(json.dumps(grader))
    launch_payload = json.loads(launch.read_text())
    launch_payload["results"] = results
    launch.write_text(json.dumps(launch_payload))
    with pytest.raises(ValueError, match="no recognized failure evidence"):
        complete_campaign(source, launch, tmp_path / "must_not_exist.yaml")


def test_campaign_completion_rederives_l0_source_change_mapping(tmp_path):
    source, launch, results = _fake_complete_campaign(tmp_path)
    run_dir = Path(results[0]["run_dir"])
    grader_path = run_dir / "metrics" / "grader_result.json"
    grader = json.loads(grader_path.read_text())
    capsule = grader["selected_capsules"][0]
    l0 = {record["mode"]: record for record in grader["levels"]["L0"]["records"]
          if record["capsule"] == capsule}
    l0["rvv"]["source_sha256"] = l0["scalar"]["source_sha256"]
    l0["rvv"]["metadata"]["source_sha256"] = l0["scalar"]["source_sha256"]
    # Preserve the forged producer claim: the finalizer must recompute rather than trust it.
    assert grader["levels"]["L0"]["scalar_rvv_source_change"][capsule] is True
    grader_path.write_text(json.dumps(grader))
    with pytest.raises(ValueError, match="source-change mapping is not rederived"):
        complete_campaign(source, launch, tmp_path / "must_not_exist.yaml")


def test_campaign_completion_rederives_l0_no_fallback_from_metadata(tmp_path):
    source, launch, results = _fake_complete_campaign(tmp_path)
    grader_path = Path(results[0]["run_dir"]) / "metrics" / "grader_result.json"
    grader = json.loads(grader_path.read_text())
    record = next(record for record in grader["levels"]["L0"]["records"]
                  if record["mode"] == "scalar")
    record["metadata"]["fallback_used"] = True
    assert record["checks"]["fallback_forbidden"] is True
    grader_path.write_text(json.dumps(grader))
    with pytest.raises(ValueError, match="checks disagree with retained metadata"):
        complete_campaign(source, launch, tmp_path / "must_not_exist.yaml")


@pytest.mark.parametrize(("level", "record"), [
    ("L0", {"capsule": "c", "family": "contraction", "mode": "scalar", "ok": False,
            "reason": "compiler invocation timed out", "timeout_seconds": 300,
            "wall_seconds": 300.0}),
    ("L1", {"capsule": "c", "family": "contraction", "mode": "scalar",
            "status": "fail", "reason": "trusted native execution timed out",
            "build_wall_seconds": 0.1, "build_stderr_tail": "", "build_returncode": 0,
            "trials": [], "timed_out_trial_index": 0, "timed_out_seed": 7,
            "timeout_seconds": 45}),
    ("L1", {"capsule": "c", "family": "contraction", "mode": "scalar",
            "status": "fail", "reason": "trusted native build timed out",
            "build_logs": [{"returncode": 0, "wall_seconds": 0.1, "stderr_tail": ""}],
            "failed_stage_index": 1, "timeout_seconds": 120}),
    ("L2", {"capsule": "c", "family": "contraction", "mode": "rvv",
            "tail_case": True, "status": "fail", "reason": "Spike execution timed out",
            "seed": 7, "vector_instructions": ["vadd.vv"],
            "vector_dataflow": {"version": 1, "useful": False},
            "kernel_text_sha256": "a" * 64,
            "build_logs": [{"returncode": 0, "wall_seconds": 0.1, "stderr_tail": ""}
                           for _ in range(6)],
            "timeout_seconds": 180, "wall_seconds": 180.0}),
    ("L3", {"capsule": "c", "family": "runtime_parallel", "mode": "rvv_multicore",
            "harts": 8, "status": "fail", "reason": "K1 cross-build failed",
            "build_wall_seconds": 0.1, "build_stderr_tail": "compile failed",
            "build_returncode": 1, "failed_stage_index": 0,
            "build_logs": [{"returncode": 1, "wall_seconds": 0.1,
                            "stderr_tail": "compile failed"}]}),
])
def test_candidate_timeouts_are_substantive_treatment_failures(level, record):
    _COMPLETE._validate_early_failure(level, record, l0_outcomes={})


@pytest.mark.parametrize(("reason", "stage", "syntax_returncode", "syntax_ok"), [
    ("C syntax check timed out", "c_syntax", None, False),
    ("MLIR verifier timed out", "mlir_verifier", 0, True),
])
def test_l0_tool_timeouts_retain_artifact_and_partial_check_evidence(
        reason, stage, syntax_returncode, syntax_ok):
    plan = {"buffers": []}
    checks = {name: True for name in _COMPLETE._LEVEL_CHECKS["L0"]}
    checks.update(c_syntax=syntax_ok, mlir_verifier=False)
    record = {
        "capsule": "c", "family": "contraction", "mode": "scalar", "ok": False,
        "reason": reason, "returncode": 0, "wall_seconds": 0.1,
        "stdout_tail": "", "stderr_tail": "", "checks": checks,
        "metadata": {}, "source_sha256": "a" * 64, "source_size_bytes": 10,
        "input_mlir_sha256": "b" * 64, "lowered_mlir_sha256": "c" * 64,
        "buffer_plan": plan, "timeout_seconds": 60, "timed_out_stage": stage,
        "syntax_returncode": syntax_returncode, "verifier_returncode": None,
        "syntax_stderr": "partial", "verifier_stderr": "",
    }
    _COMPLETE._validate_early_failure(
        "L0", record, l0_outcomes={}, expected_buffer_plan=plan)


@pytest.mark.parametrize(("level", "reason", "mode", "extra"), [
    ("L1", "trusted native build failed", "scalar", {}),
    ("L3", "K1 cross-build failed", "rvv_multicore", {"harts": 8}),
])
def test_build_failure_aggregate_must_equal_terminal_stage_log(level, reason, mode, extra):
    record = {
        "capsule": "c", "family": "runtime_parallel", "mode": mode,
        "status": "fail", "reason": reason, "failed_stage_index": 0,
        "build_logs": [{"returncode": 1, "wall_seconds": 0.1, "stderr_tail": "stage"}],
        "build_wall_seconds": 0.1, "build_returncode": 2,
        "build_stderr_tail": "aggregate", **extra,
    }
    with pytest.raises(ValueError, match="aggregate build evidence contradicts"):
        _COMPLETE._validate_early_failure(level, record, l0_outcomes={})


def test_failed_predeclared_primary_completes_campaign_but_cannot_promote(tmp_path):
    source, launch, results = _fake_complete_campaign(tmp_path)
    failed = _mark_fake_run_as_graded_failure(
        launch, results, arm="arm4_agentic_pass_authoring", repeat=0)
    completed = complete_campaign(source, launch, tmp_path / "campaign_unpromoted.yaml")
    assert completed.status == "campaign_complete_unpromoted"
    record = completed.freeze["campaign_record"]
    assert record["promotion"] == {
        "status": "ineligible",
        "predeclared_run_id": failed["run_id"],
        "reason": "predeclared primary outcome is graded_fail",
    }
    assert completed.freeze["selected_run_id"] == "unresolved"
    assert completed.freeze["selected_compiler_package"] == "unresolved"
    assert completed.freeze["selected_policy_sha256"] == "unresolved"
    assert completed.freeze["runtime_sha256"] == "unresolved"
    assert completed.freeze["compiler_sha256"] == "unresolved"


def test_v3_campaign_retains_typed_search_failure_without_calling_it_graded(tmp_path):
    source, launch, results = _fake_complete_campaign(tmp_path)
    _upgrade_fake_launch_to_v3(launch, results)
    failed = _mark_fake_search_failure(launch, results)
    completed = complete_campaign(source, launch, tmp_path / "campaign_complete.yaml")
    record = completed.freeze["campaign_record"]
    row = next(item for item in record["runs"] if item["run_id"] == failed["run_id"])
    assert row["outcome"] == "treatment_search_fail"
    assert record["outcome_counts"]["treatment_search_fail"] == 1
    assert record["promotion"]["status"] == "promoted"


@pytest.mark.parametrize("arm_id", ["arm1_raw_cpp", "arm3_generated_cpu_dialect"])
def test_v3_campaign_retains_reconciled_agent_timeout_with_full_telemetry(tmp_path, arm_id):
    source, launch, results = _fake_complete_campaign(tmp_path)
    _upgrade_fake_launch_to_v3(launch, results)
    result = next(row for row in results if row["arm"] == arm_id and row["repeat"] == 0)
    run_dir = Path(result["run_dir"])
    result.update(returncode=1, terminal_class="treatment_agent_fail")
    search_path = run_dir / "contracts" / "trusted_search_seal.json"
    search = json.loads(search_path.read_text())
    if arm_id == "arm3_generated_cpu_dialect":
        search = {
            "version": 1, "status": "fail", "failure_class": "treatment_search_fail",
            "checks": {"private_ledger": True},
            "reason": "trusted search artifacts are incomplete",
        }
        search_path.write_text(json.dumps(search))
    summary_path = run_dir / "metrics" / "summary_metrics.json"
    summary = json.loads(summary_path.read_text())
    summary.update({"agent_success": False, "agent_failure_class": "treatment_agent_fail",
                    "compiler_seal_status": "not_run", "grader_returncode": 2,
                    "grader_status": "not_run", "grader_wall_seconds": 0.0,
                    "trusted_search_status": search["status"],
                    "terminal_class": "treatment_agent_fail"})
    summary_path.write_text(json.dumps(summary))
    compiler = {"version": 1, "status": "not_run",
                "failure_class": "treatment_agent_fail",
                "reason": "reconciled Codex attempt did not complete"}
    grader = {"version": 1, "status": "not_run", "wall_seconds": 0.0,
              "reason": "trusted search/compiler seal did not verify"}
    (run_dir / "contracts" / "compiler_seal.json").write_text(json.dumps(compiler))
    (run_dir / "metrics" / "grader_result.json").write_text(json.dumps(grader))
    shutil.rmtree(run_dir / "artifacts" / "compiler_submission")
    run_result_path = run_dir / "agent" / "run_result.json"
    run_result = json.loads(run_result_path.read_text())
    run_result.update(status="failed", resolved_model=None)
    run_result["attempts"][0]["failure_class"] = "timeout"
    run_result_path.write_text(json.dumps(run_result))
    input_audit = json.loads(
        (run_dir / "contracts" / "workspace_input_audit.json").read_text())
    reconciliation = json.loads(
        (run_dir / "metrics" / "codex_reconciliation.json").read_text())
    (run_dir / "contracts" / "terminal_outcome.json").write_text(json.dumps({
        "version": 1, "run_id": result["run_id"], "arm": result["arm"],
        "terminal_class": "treatment_agent_fail", "paper_evidence_eligible": False,
        "promotion_eligible": False,
        "checks": {"agent_success": False, "agent_failure_class": "treatment_agent_fail",
                   "workspace_input_audit": input_audit["ok"],
                   "aet_reconciled": reconciliation["ok"],
                   "trusted_search_status": search["status"],
                   "compiler_seal_status": "not_run",
                   "compiler_seal_failure_class": "treatment_agent_fail",
                   "grader_returncode": 2, "grader_status": "not_run",
                   "grader_failure_class": None},
    }))
    payload = json.loads(launch.read_text()); payload["results"] = results
    launch.write_text(json.dumps(payload))
    completed = complete_campaign(source, launch, tmp_path / "campaign_complete.yaml")
    record = completed.freeze["campaign_record"]
    observed = next(row for row in record["runs"] if row["run_id"] == result["run_id"])
    assert observed["outcome"] == "treatment_agent_fail"
    assert observed["codex_attempts"] == 1
    assert observed["tokens"]["input_tokens"] == 10


def test_v3_campaign_retains_submission_build_failure_evidence(tmp_path):
    source, launch, results = _fake_complete_campaign(tmp_path)
    _upgrade_fake_launch_to_v3(launch, results)
    result = next(row for row in results if row["arm"] == "arm1_raw_cpp" and row["repeat"] == 0)
    run_dir = Path(result["run_dir"])
    result.update(returncode=1, terminal_class="treatment_build_fail")
    grader = {
        "version": 1, "status": "treatment_build_fail",
        "failure_class": "treatment_build_fail",
        "implemented_levels": ["L0", "L1", "L2", "L3"],
        "reason": "submission build failed",
        "build_failure": {"commands": [{
            "command": ["python3", "build.py"], "returncode": 7,
            "wall_seconds": 0.2, "stdout_tail": "", "stderr_tail": "compile error",
        }], "failed_stage_index": 0, "returncode": 7},
        "wall_seconds": 0.2,
    }
    (run_dir / "metrics" / "grader_result.json").write_text(json.dumps(grader))
    summary_path = run_dir / "metrics" / "summary_metrics.json"
    summary = json.loads(summary_path.read_text())
    summary.update({"grader_returncode": 1, "grader_status": "treatment_build_fail",
                    "grader_wall_seconds": 0.2, "terminal_class": "treatment_build_fail"})
    summary_path.write_text(json.dumps(summary))
    input_audit = json.loads(
        (run_dir / "contracts" / "workspace_input_audit.json").read_text())
    reconciliation = json.loads(
        (run_dir / "metrics" / "codex_reconciliation.json").read_text())
    search = json.loads((run_dir / "contracts" / "trusted_search_seal.json").read_text())
    compiler = json.loads((run_dir / "contracts" / "compiler_seal.json").read_text())
    (run_dir / "contracts" / "terminal_outcome.json").write_text(json.dumps({
        "version": 1, "run_id": result["run_id"], "arm": result["arm"],
        "terminal_class": "treatment_build_fail", "paper_evidence_eligible": False,
        "promotion_eligible": False,
        "checks": {"agent_success": True, "agent_failure_class": None,
                   "workspace_input_audit": input_audit["ok"],
                   "aet_reconciled": reconciliation["ok"],
                   "trusted_search_status": search["status"],
                   "compiler_seal_status": compiler["status"],
                   "compiler_seal_failure_class": None,
                   "grader_returncode": 1, "grader_status": "treatment_build_fail",
                   "grader_failure_class": "treatment_build_fail"},
    }))
    payload = json.loads(launch.read_text()); payload["results"] = results
    launch.write_text(json.dumps(payload))
    completed = complete_campaign(source, launch, tmp_path / "campaign_complete.yaml")
    assert completed.freeze["campaign_record"]["outcome_counts"]["treatment_build_fail"] == 1


def test_nonzero_launch_code_without_matching_grader_failure_is_not_scored(tmp_path):
    source, launch, results = _fake_complete_campaign(tmp_path)
    results[0]["returncode"] = 1
    raw = json.loads(launch.read_text())
    raw["results"] = results
    launch.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="no recognized complete pass/fail outcome"):
        complete_campaign(source, launch, tmp_path / "must_not_exist.yaml")


def test_campaign_completion_rejects_cross_protocol_run_transplant(tmp_path):
    source, launch, results = _fake_complete_campaign(tmp_path)
    first = Path(results[0]["run_dir"])
    preflight_path = first / "contracts" / "preflight.json"
    preflight = json.loads(preflight_path.read_text())
    preflight["evidence"]["protocol_inputs_sha256"] = "0" * 64
    preflight_path.write_text(json.dumps(preflight))
    with pytest.raises(ValueError, match="differs from frozen protocol"):
        complete_campaign(source, launch, tmp_path / "must_not_exist.yaml")


def test_campaign_completion_is_atomic_and_refuses_missing_run(tmp_path):
    source, launch, _ = _fake_complete_campaign(tmp_path)
    raw = json.loads(launch.read_text())
    raw["results"].pop()
    launch.write_text(json.dumps(raw))
    output = tmp_path / "must_not_exist.yaml"
    with pytest.raises(ValueError, match="exact four arms x four blocks"):
        complete_campaign(source, launch, output)
    assert not output.exists()


def test_campaign_completion_derives_every_seed_and_run_id_from_launch_plan(tmp_path):
    source, launch, _ = _fake_complete_campaign(tmp_path)
    raw = json.loads(launch.read_text())
    raw["results"][0]["seed"] += 1
    launch.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="differs from deterministic frozen launch plan"):
        complete_campaign(source, launch, tmp_path / "must_not_exist.yaml")


def test_campaign_completion_requires_exact_frozen_launch_chronology(tmp_path):
    source, launch, _ = _fake_complete_campaign(tmp_path)
    raw = json.loads(launch.read_text())
    raw["planned"].reverse()
    raw["results"].reverse()
    launch.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="frozen launch chronology"):
        complete_campaign(source, launch, tmp_path / "must_not_exist.yaml")


def test_campaign_completion_rejects_cumulative_or_unretained_aet_evidence(tmp_path):
    source, launch, results = _fake_complete_campaign(tmp_path / "unretained")
    first = Path(results[0]["run_dir"])
    ledger = first / "metrics" / "token_ledger.jsonl"
    row = json.loads(ledger.read_text())
    row["input_tokens"] = 99  # Not the delta captured in agent/run_result.json.
    ledger.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="not the per-turn delta"):
        complete_campaign(source, launch, tmp_path / "must_not_exist.yaml")

    source, launch, results = _fake_complete_campaign(tmp_path)
    first = Path(results[0]["run_dir"])
    (first / "agent" / "aet_raw" / "attempt_9999.jsonl").write_text("{}\n")
    with pytest.raises(ValueError, match="exactly one non-symlink JSONL stream per attempt"):
        complete_campaign(source, launch, tmp_path / "must_not_exist_2.yaml")


def test_campaign_completion_rejects_missing_token_fields_even_if_ledgers_agree(tmp_path):
    source, launch, results = _fake_complete_campaign(tmp_path)
    first = Path(results[0]["run_dir"])
    token_path = first / "metrics" / "token_ledger.jsonl"
    token = json.loads(token_path.read_text())
    token.pop("input_tokens")
    token["uncached_input_tokens"] = 0
    token_path.write_text(json.dumps(token) + "\n")
    result_path = first / "agent" / "run_result.json"
    result = json.loads(result_path.read_text())
    result["attempts"][0]["turns"][0].pop("input_tokens")
    result["attempts"][0]["turns"][0]["uncached_input_tokens"] = 0
    result_path.write_text(json.dumps(result))
    reconciliation_path = first / "metrics" / "codex_reconciliation.json"
    reconciliation = json.loads(reconciliation_path.read_text())
    reconciliation["token_ledger"]["checks"]["uncached_input"]["ledger"] = 0
    reconciliation_path.write_text(json.dumps(reconciliation))
    with pytest.raises(ValueError, match="missing required full-fidelity field|retained-stream rederivation"):
        complete_campaign(source, launch, tmp_path / "must_not_exist.yaml")


def test_campaign_completion_rejects_malformed_or_fabricated_tool_ledger(tmp_path):
    source, launch, results = _fake_complete_campaign(tmp_path)
    tools = Path(results[0]["run_dir"]) / "agent" / "tools.jsonl"
    tools.write_text("{not-json}\n")
    with pytest.raises(ValueError, match="not valid JSON"):
        complete_campaign(source, launch, tmp_path / "must_not_exist.yaml")

    source, launch, results = _fake_complete_campaign(tmp_path / "fabricated")
    tools = Path(results[0]["run_dir"]) / "agent" / "tools.jsonl"
    tools.write_text(json.dumps({"item_id": "fabricated", "kind": "command_execution"}) + "\n")
    with pytest.raises(ValueError, match="differs from deterministic retained-stream rederivation"):
        complete_campaign(source, launch, tmp_path / "must_not_exist_2.yaml")


@pytest.mark.parametrize("field, value, message", [
    ("active_wall_seconds", -1.0, "active_wall_seconds"),
    ("grader_wall_seconds", -1.0, "grader_wall_seconds"),
    ("trusted_search_wall_seconds", -1.0, "trusted_search_wall_seconds"),
    ("wall_seconds", -1.0, "wall_seconds"),
    ("active_wall_seconds", 11.0, "differs from its retained authoritative evidence"),
    ("grader_wall_seconds", 3.0, "differs from its retained authoritative evidence"),
    ("trusted_search_wall_seconds", 1.0, "differs from its retained authoritative evidence"),
    ("wall_seconds", 999999.0, "differs from its retained authoritative evidence"),
])
def test_campaign_completion_rejects_negative_or_corrupted_summary_times(
        tmp_path, field, value, message):
    source, launch, results = _fake_complete_campaign(tmp_path)
    summary_path = Path(results[0]["run_dir"]) / "metrics" / "summary_metrics.json"
    summary = json.loads(summary_path.read_text())
    summary[field] = value
    summary_path.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match=message):
        complete_campaign(source, launch, tmp_path / "must_not_exist.yaml")


def test_campaign_completion_exclusive_reservation_blocks_concurrent_publisher(tmp_path):
    source, launch, _ = _fake_complete_campaign(tmp_path)
    output = tmp_path / "campaign_complete.yaml"
    reservation = output.with_name(f".{output.name}.completion.lock")
    reservation.write_text("other finalizer\n")
    with pytest.raises(FileExistsError, match="another campaign finalizer"):
        complete_campaign(source, launch, output)
    assert not output.exists()
    assert reservation.read_text() == "other finalizer\n"


def test_campaign_completion_never_publishes_failed_round_trip(tmp_path, monkeypatch):
    source, launch, _ = _fake_complete_campaign(tmp_path)
    output = tmp_path / "campaign_complete.yaml"
    real_preflight = HostExperimentSpec.preflight
    calls = 0

    def fail_only_final_round_trip(self, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return real_preflight(self, **kwargs)
        return HostPreflight((), ("forced final verification failure",), (), {})

    monkeypatch.setattr(HostExperimentSpec, "preflight", fail_only_final_round_trip)
    with pytest.raises(ValueError, match="round-trip is NO_GO"):
        complete_campaign(source, launch, output)
    assert not output.exists()
    assert not output.with_name(f".{output.name}.completion.lock").exists()
