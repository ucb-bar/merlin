from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from merlin.plotting import cpu_host_experiment_figures as figures
from merlin.plotting import cpu_host_beam_figures as beam_figures


pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("matplotlib") is None, reason="matplotlib optional extra is absent")


_ARMS = [
    "arm1_raw_cpp", "arm2_cpp_scaffold", "arm3_generated_cpu_dialect",
    "arm4_agentic_pass_authoring",
]
_ACTION = {
    "id": "dynamic_vl", "group": "vector_length", "action_class": "heuristic",
    "stage": 10, "value": "scalable",
    "affected_families": ["contraction", "elementwise_map", "reduction"],
    "evidence": ["target_contract"],
}


def _candidate(actions: list[dict]) -> dict:
    canonical = sorted(actions, key=lambda row: (row["stage"], row["group"], row["id"]))
    payload = [{key: value for key, value in action.items() if key != "evidence"}
               for action in canonical]
    return {"version": 1, "candidate_sha256": figures._canonical_sha256(payload),
            "actions": canonical}


def _campaign(tmp_path: Path) -> dict:
    rows = []
    ordinal = 0
    for repeat in range(4):
        for arm_index, arm in enumerate(_ARMS):
            run_id = f"run-{ordinal}"
            run_dir = tmp_path / "runs" / run_id
            (run_dir / "metrics").mkdir(parents=True)
            (run_dir / "contracts").mkdir()
            package = run_dir / "artifacts" / "compiler_submission"
            package.mkdir(parents=True)
            grader = {"version": 1, "status": "pass", "implemented_levels": list(
                figures._LEVELS), "levels": {
                    level: {"status": "pass"} for level in figures._LEVELS}}
            grader_path = run_dir / "metrics" / "grader_result.json"
            grader_path.write_text(json.dumps(grader), encoding="utf-8")

            search_arm = arm in _ARMS[2:]
            if search_arm:
                parent, selected = _candidate([]), _candidate([_ACTION])
                search_dir = package / "search"
                search_dir.mkdir()
                policy_path = search_dir / "selected_policy.json"
                policy_bytes = json.dumps(selected, sort_keys=True)
                policy_path.write_text(policy_bytes, encoding="utf-8")
                accepted = [{
                    "candidate": selected,
                    "train": {
                        "split": "train", "parent_candidate_sha256": parent["candidate_sha256"],
                        "candidate_sha256": selected["candidate_sha256"], "failures": [],
                        "affected_median_speedup": 1.20 + repeat / 100,
                    },
                    "validation": {
                        "split": "validation",
                        "parent_candidate_sha256": parent["candidate_sha256"],
                        "candidate_sha256": selected["candidate_sha256"], "failures": [],
                        "affected_median_speedup": 1.16 + repeat / 100,
                    },
                }]
                search_record = {
                    "version": 1, "status": "converged", "heldout_visible": False,
                    "selected_policy_sha256": figures._sha256(policy_path),
                    "accepted": accepted,
                }
                search_record_path = search_dir / "search_record.json"
                search_record_path.write_text(json.dumps(search_record), encoding="utf-8")
                policy = selected
                search_seal = {
                    "version": 1, "status": "pass",
                    "search_record_sha256": figures._sha256(search_record_path),
                    "selected_policy_sha256": figures._sha256(policy_path),
                }
            else:
                policy = {"version": 1, "name": f"fixed-{arm}"}
                policy_bytes = yaml.safe_dump(policy, sort_keys=True)
                search_seal = {"version": 1, "status": "not_required", "arm": arm}
            (package / "policy.yaml").write_text(policy_bytes, encoding="utf-8")
            (package / "compiler.bin").write_bytes(bytes([arm_index + 1]) * (32 + ordinal))
            (package / "manifest.yaml").write_text(
                yaml.safe_dump({"version": 1, "policy": "policy.yaml"}), encoding="utf-8")
            search_seal_path = run_dir / "contracts" / "trusted_search_seal.json"
            search_seal_path.write_text(json.dumps(search_seal), encoding="utf-8")
            policy_sha = figures._sha256(package / "policy.yaml")
            compiler_seal = {
                "version": 1, "status": "sealed", "policy_sha256": policy_sha,
                "compiler_source_sha256": "b" * 64,
                "compiler_package_sha256": figures._submission_package_digest(package),
                "search_status": "pass" if search_arm else "not_required",
                "search_record_sha256": search_seal.get("search_record_sha256"),
                "selected_policy_sha256": (
                    search_seal.get("selected_policy_sha256") if search_arm else policy_sha),
            }
            input_tokens = 100 * (arm_index + 1) + repeat * 10
            output_tokens = 20 * (arm_index + 1) + repeat
            rows.append({
                "ordinal": ordinal, "arm": arm, "repeat": repeat, "seed": repeat + 1,
                "run_id": run_id, "run_dir": str(run_dir), "outcome": "graded_pass",
                "grader_result_sha256": figures._sha256(grader_path),
                "compiler_seal": compiler_seal,
                "search_seal_sha256": figures._sha256(search_seal_path),
                "tokens": {
                    "input_tokens": input_tokens,
                    "cached_input_tokens": input_tokens // 2,
                    "cache_write_input_tokens": input_tokens // 10,
                    "uncached_input_tokens": input_tokens - input_tokens // 2 - input_tokens // 10,
                    "output_tokens": output_tokens,
                    "reasoning_output_tokens": output_tokens // 2,
                },
                "tool_calls": 10 * (arm_index + 1) + repeat,
                "timing_seconds": {
                    "active_wall_seconds": float(30 * (arm_index + 1) + repeat),
                    "grader_wall_seconds": float(5 + repeat),
                    "trusted_search_wall_seconds": float(arm_index * 2),
                    "wall_seconds": float(40 * (arm_index + 1) + repeat),
                },
            })
            ordinal += 1
    return {
        "version": 1, "expected_run_count": 16, "completed_run_count": 16,
        "analysis_plan_sha256": "a" * 64, "runs": rows,
    }


def _install_spec(monkeypatch, record: dict, tmp_path: Path, *,
                  status: str = "campaign_complete"):
    digest = figures._canonical_sha256(record)
    space_path = tmp_path / "optimization_space.yaml"
    space = {
        "version": 1, "status": "frozen_definition", "minimum_families": 1,
        "confirmation_width": 1, "actions": [_ACTION],
    }
    space_path.write_text(yaml.safe_dump(space), encoding="utf-8")
    check = SimpleNamespace(ready=True, to_dict=lambda: {
        "ready": True, "errors": [], "blockers": [], "warnings": []})
    spec = SimpleNamespace(
        status=status,
        arms=[SimpleNamespace(
            id=arm, capabilities=((figures._SEARCH_CAPABILITY,) if arm in _ARMS[2:] else ()))
            for arm in _ARMS],
        agent={"billing": "subscription_notional"},
        analysis={"sha256": "a" * 64},
        search={"space": str(space_path), "space_sha256": figures._sha256(space_path)},
        freeze={"campaign_record": record, "campaign_record_sha256": digest},
        preflight=lambda **_kwargs: check,
        search_space_config=lambda: space,
        _repo_path=lambda value: Path(value),
    )
    monkeypatch.setattr(figures.HostExperimentSpec, "from_yaml", lambda _path: spec)
    return spec


def _write_beam_evaluation(ledger: Path, *, parent: dict, candidate: dict,
                           split: str, phase: str, repeats: int, ordinal: int) -> tuple[str, dict]:
    key = f"{parent['candidate_sha256']}:{candidate['candidate_sha256']}:{split}:{phase}"
    observation = {
        "capsule_id": "synthetic-public-capsule",
        "family": "contraction", "correctness_ok": True,
        "parent_candidate_sha256": parent["candidate_sha256"],
        "candidate_sha256": candidate["candidate_sha256"],
        "baseline_code_sha256": "1" * 64, "candidate_code_sha256": "2" * 64,
        "code_digest_authority": ("compiled_kernel_object_text_section" if phase == "screen"
                                  else "measured_k1_kernel_object_text_section"),
    }
    if phase == "screen":
        observation.update({
            "baseline_cycles": 100, "candidate_cycles": 80,
            "screen_authority": "spike_rv64gcv_mcycle_trusted_harness",
        })
        measured_repeats = 1
    else:
        observation.update({
            "baseline_elapsed_ns": [100] * repeats, "baseline_calls": [1] * repeats,
            "candidate_elapsed_ns": [80] * repeats, "candidate_calls": [1] * repeats,
            "timing_authority": "spacemit_k1_elapsed_ns_div_completed_calls",
        })
        measured_repeats = repeats
    observation_path = ledger / "observations" / f"observation-{ordinal}.jsonl"
    observation_path.write_text(json.dumps(observation, sort_keys=True) + "\n", encoding="utf-8")
    policy_sha, parent_sha, capsules_sha = "3" * 64, "4" * 64, "5" * 64
    evaluation = {
        "parent_candidate_sha256": parent["candidate_sha256"],
        "candidate_sha256": candidate["candidate_sha256"], "split": split, "phase": phase,
        "policy_sha256": policy_sha, "parent_policy_sha256": parent_sha,
        "capsules_sha256": capsules_sha, "private_capsules_sha256": "6" * 64,
        "private_capsule_ids": [f"private-id-never-exported-{ordinal}"],
        "observations": str(observation_path.relative_to(ledger)),
        "observations_sha256": figures._sha256(observation_path),
        "measurement_repeats": measured_repeats, "request_multiplicity": 1,
        "wall_ns": ordinal + 11,
    }
    request_id = f"request{ordinal}"
    request = {
        "version": 1, "split": split, "phase": phase, "repeats": measured_repeats,
        "policy": f"/controller/workspace/private/policy-{ordinal}.json",
        "parent_policy": f"/controller/workspace/private/parent-{ordinal}.json",
        "capsules": f"/controller/workspace/private/capsules-{ordinal}.jsonl",
    }
    request_path = ledger / "requests" / f"{request_id}.json"
    request_path.write_text(json.dumps(request, sort_keys=True), encoding="utf-8")
    parsed = {
        **request,
        "parent_candidate_sha256": parent["candidate_sha256"],
        "candidate_sha256": candidate["candidate_sha256"],
        "parent_policy_sha256": parent_sha, "policy_sha256": policy_sha,
        "capsules_sha256": capsules_sha,
    }
    receipt = {
        "version": 1, "authority": "driver_trusted_search_broker",
        "request_id": request_id, "status": "pass",
        "request_sha256": figures._sha256(request_path),
        "request_artifact": str(request_path.relative_to(ledger)),
        "evaluation_key": key, "cache_hit": False, "multiplicity": 1,
        "parsed_request": parsed, "response_sha256": evaluation["observations_sha256"],
        "wall_ns": ordinal + 3,
    }
    receipt_path = ledger / "receipts" / f"{request_id}.json"
    receipt_path.write_text(json.dumps(receipt, sort_keys=True), encoding="utf-8")
    receipt_index = {
        "path": str(receipt_path.relative_to(ledger)), "sha256": figures._sha256(receipt_path),
        "status": "pass", "evaluation_key": key, "cache_hit": False,
        "multiplicity": 1, "response_sha256": evaluation["observations_sha256"],
    }
    return key, {"evaluation": evaluation, "request_id": request_id,
                 "receipt_index": receipt_index}


def _install_one_synthetic_beam(record: dict, spec, *, selected_row: dict) -> None:
    for row in record["runs"]:
        if row["arm"] not in _ARMS[2:]:
            continue
        seal_path = Path(row["run_dir"]) / "contracts" / "trusted_search_seal.json"
        if row is not selected_row:
            seal_path.write_text(json.dumps({
                "version": 1, "status": "fail", "failure_class": "treatment_search_fail",
                "reason": "synthetic unavailable cell",
            }), encoding="utf-8")
            row["search_seal_sha256"] = figures._sha256(seal_path)
            row["outcome"] = "treatment_search_fail"
            row["compiler_seal"] = {
                "version": 1, "status": "not_run", "failure_class": "treatment_search_fail",
            }
            shutil.rmtree(Path(row["run_dir"]) / "artifacts" / "compiler_submission")

    package = Path(selected_row["run_dir"]) / "artifacts" / "compiler_submission"
    selected_path = package / "search" / "selected_policy.json"
    selected = json.loads(selected_path.read_text(encoding="utf-8"))
    parent = _candidate([])
    repeats = 3
    ledger = Path(selected_row["run_dir"]) / "metrics" / "trusted_search_ledger"
    for directory in (ledger / "observations", ledger / "requests", ledger / "receipts"):
        directory.mkdir(parents=True, exist_ok=True)
    evaluation_rows = [
        _write_beam_evaluation(ledger, parent=parent, candidate=selected,
                               split="train", phase="screen", repeats=repeats, ordinal=0),
        _write_beam_evaluation(ledger, parent=parent, candidate=selected,
                               split="train", phase="confirm", repeats=repeats, ordinal=1),
        _write_beam_evaluation(ledger, parent=parent, candidate=selected,
                               split="validation", phase="confirm", repeats=repeats, ordinal=2),
    ]
    evaluations = {key: value["evaluation"] for key, value in evaluation_rows}
    receipts = {value["request_id"]: value["receipt_index"] for _, value in evaluation_rows}
    index = {
        "version": 1, "authority": "trusted_spacemit_k1_outside_agent_sandbox",
        "heldout_opened": False, "space_sha256": spec.search["space_sha256"],
        "measurement_repeats": repeats, "evaluations": evaluations,
        "terminal_receipts": receipts,
        # This deliberately contains private controller data. The renderer must neither require
        # its identities for labels nor leak them into its manifest.
        "private_shape_corpus": {"secret": "never-export-me"},
    }
    index_path = ledger / "index.json"
    index_path.write_text(json.dumps(index, sort_keys=True), encoding="utf-8")
    accepted_entry = {"candidate": selected}
    for split in ("train", "validation"):
        accepted_entry[split] = {
            "parent_candidate_sha256": parent["candidate_sha256"],
            "candidate_sha256": selected["candidate_sha256"], "split": split,
            "failures": [], "median_speedup": 1.25, "minimum_speedup": 1.25,
            "affected_median_speedup": 1.25,
            "observations_sha256": evaluations[
                f"{parent['candidate_sha256']}:{selected['candidate_sha256']}:{split}:confirm"
            ]["observations_sha256"],
        }
    search_record = {
        "version": 1, "status": "converged", "heldout_visible": False,
        "selection_policy": "spike_screen_then_k1_confirmation",
        "space_sha256": spec.search["space_sha256"],
        "selected_policy_sha256": figures._sha256(selected_path),
        "sample_counts": {"screen_train": 1, "confirmation_train": 1,
                          "confirmation_validation": 1},
        "acceptance_thresholds": {
            "calibrated_upper_margin": 0.02,
            "affected_train_median_strictly_above": 1.02,
            "affected_validation_median_strictly_above": 1.02,
            "validation_minimum_at_least": 1 / 1.02,
            "minimum_pairwise_wins_per_affected_capsule": 2,
        },
        "confirmation_families": ["contraction"], "confirmation_width": 1,
        "measurement_repeats": repeats, "accepted": [accepted_entry],
        "sweeps": [
            {"sweep": 0, "incumbent": parent["candidate_sha256"],
             "screened": [selected["candidate_sha256"]],
             "confirmed": [selected["candidate_sha256"]],
             "promoted": [selected["candidate_sha256"]],
             "winner": selected["candidate_sha256"]},
            {"sweep": 1, "incumbent": selected["candidate_sha256"],
             "screened": [], "confirmed": [], "promoted": [], "winner": None},
        ],
        "empty_sweeps": 1, "required_empty_sweeps": 1,
    }
    record_path = package / "search" / "search_record.json"
    record_path.write_text(json.dumps(search_record, sort_keys=True), encoding="utf-8")
    search_seal = {
        "version": 1, "status": "pass", "search_record_sha256": figures._sha256(record_path),
        "selected_policy_sha256": figures._sha256(selected_path),
        "trusted_ledger_sha256": figures._sha256(index_path),
        "trusted_evaluation_count": len(evaluations),
        "trusted_evaluation_wall_ns": sum(value["wall_ns"] for value in evaluations.values()),
        "trusted_broker_wall_ns": 100,
    }
    seal_path = Path(selected_row["run_dir"]) / "contracts" / "trusted_search_seal.json"
    seal_path.write_text(json.dumps(search_seal, sort_keys=True), encoding="utf-8")
    selected_row["search_seal_sha256"] = figures._sha256(seal_path)
    selected_row["compiler_seal"].update({
        "policy_sha256": figures._sha256(selected_path),
        "selected_policy_sha256": figures._sha256(selected_path),
        "search_record_sha256": figures._sha256(record_path),
    })
    selected_row["compiler_seal"]["compiler_package_sha256"] = (
        figures._submission_package_digest(package))


def test_arm_resource_figure_uses_all_cells_and_discloses_notional_cost(
        tmp_path: Path, monkeypatch) -> None:
    record = _campaign(tmp_path)
    _install_spec(monkeypatch, record, tmp_path)
    campaign = tmp_path / "campaign.yaml"
    campaign.write_text("sealed completed campaign\n", encoding="utf-8")

    output = figures.generate_cpu_host_resource_figure(
        campaign, output_dir=tmp_path / "arm-figures")

    assert (output / "arm1_4_resource_cost.png").is_file()
    svg = (output / "arm1_4_resource_cost.svg").read_text(encoding="utf-8")
    assert "Token cost" in svg
    assert "Time cost" in svg
    assert "Tool-interaction cost" in svg
    assert "subscription_notional supplies no per-run currency amount" in svg
    outcome_svg = (output / "arm1_4_compiler_outcomes.svg").read_text(encoding="utf-8")
    assert "Generic compiler gates" in outcome_svg
    assert "Final accepted marginal" in outcome_svg
    assert "Compiler package size" in outcome_svg
    assert "Deterministic generic-search result" in outcome_svg
    assert "no paper holdout is read" in outcome_svg
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["claim_scope"] == (
        "descriptive_small_n_all_sixteen_predeclared_cells_generic_development_only")
    assert manifest["currency_cost"] == {
        "status": "not_available", "reason": "subscription_notional"}
    # Arm 1 provider totals are [120, 131, 142, 153]; the manifest retains all four and median.
    arm1 = manifest["summaries"]["arm1_raw_cpp"]["provider_tokens"]
    assert arm1["every_cell"] == [120, 131, 142, 153]
    assert arm1["median"] == 136.5
    outcomes = manifest["outcome_summaries"]
    assert outcomes["scope"] == "generic_development_only_no_paper_holdouts"
    assert outcomes["levels"]["arm1_raw_cpp"]["L3"] == {
        "pass": 4, "fail": 0, "not_reached": 0}
    assert outcomes["selected_policy_action_count"]["arm1_raw_cpp"]["median"] is None
    assert outcomes["selected_policy_action_count"][
        "arm3_generated_cpu_dialect"]["every_cell"] == [1, 1, 1, 1]
    assert outcomes["generic_train_validation_speedup"][
        "arm4_agentic_pass_authoring"]["validation"]["every_cell"] == [
            1.16, 1.17, 1.18, 1.19]
    assert len(manifest["figures"]) == 4
    with pytest.raises(FileExistsError):
        figures.generate_cpu_host_resource_figure(campaign, output_dir=output)


def test_arm_resource_figure_rejects_incomplete_token_fidelity_before_output(
        tmp_path: Path, monkeypatch) -> None:
    record = _campaign(tmp_path)
    del record["runs"][0]["tokens"]["reasoning_output_tokens"]
    _install_spec(monkeypatch, record, tmp_path)
    campaign = tmp_path / "campaign.yaml"
    campaign.write_text("sealed completed campaign\n", encoding="utf-8")
    output = tmp_path / "figures"

    with pytest.raises(ValueError, match="full-fidelity token"):
        figures.generate_cpu_host_resource_figure(campaign, output_dir=output)

    assert not output.exists()


def test_arm_resource_figure_rejects_noncompleted_campaign(
        tmp_path: Path, monkeypatch) -> None:
    record = _campaign(tmp_path)
    _install_spec(monkeypatch, record, tmp_path, status="protocol_frozen")
    campaign = tmp_path / "campaign.yaml"
    campaign.write_text("frozen but incomplete\n", encoding="utf-8")

    with pytest.raises(ValueError, match="completed sixteen-cell"):
        figures.generate_cpu_host_resource_figure(
            campaign, output_dir=tmp_path / "figures")


def test_arm_resource_figure_rejects_nonfinite_accounting(
        tmp_path: Path, monkeypatch) -> None:
    record = _campaign(tmp_path)
    record["runs"][0]["timing_seconds"]["wall_seconds"] = float("nan")
    _install_spec(monkeypatch, record, tmp_path)
    campaign = tmp_path / "campaign.yaml"
    campaign.write_text("sealed completed campaign\n", encoding="utf-8")

    with pytest.raises(ValueError, match="non-negative number"):
        figures.generate_cpu_host_resource_figure(
            campaign, output_dir=tmp_path / "figures")


def test_arm_outcome_figure_rejects_post_completion_grader_edit_before_output(
        tmp_path: Path, monkeypatch) -> None:
    record = _campaign(tmp_path)
    _install_spec(monkeypatch, record, tmp_path)
    grader = Path(record["runs"][0]["run_dir"]) / "metrics" / "grader_result.json"
    value = json.loads(grader.read_text(encoding="utf-8"))
    value["levels"]["L3"]["status"] = "fail"
    grader.write_text(json.dumps(value), encoding="utf-8")
    campaign = tmp_path / "campaign.yaml"
    campaign.write_text("sealed completed campaign\n", encoding="utf-8")
    output = tmp_path / "figures"

    with pytest.raises(ValueError, match="grader result differs"):
        figures.generate_cpu_host_resource_figure(campaign, output_dir=output)
    assert not output.exists()


def test_arm_outcome_figure_rejects_search_ratio_edit_even_with_updated_local_file_hash(
        tmp_path: Path, monkeypatch) -> None:
    record = _campaign(tmp_path)
    search_row = next(row for row in record["runs"]
                      if row["arm"] == "arm3_generated_cpu_dialect")
    record_path = (Path(search_row["run_dir"]) / "artifacts" / "compiler_submission" /
                   "search" / "search_record.json")
    value = json.loads(record_path.read_text(encoding="utf-8"))
    value["accepted"][0]["validation"]["affected_median_speedup"] = float("nan")
    record_path.write_text(json.dumps(value), encoding="utf-8")
    # A mutable local seal is not enough: the completed campaign embeds the original package digest.
    seal_path = Path(search_row["run_dir"]) / "contracts" / "trusted_search_seal.json"
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    seal["search_record_sha256"] = figures._sha256(record_path)
    seal_path.write_text(json.dumps(seal), encoding="utf-8")
    search_row["search_seal_sha256"] = figures._sha256(seal_path)
    _install_spec(monkeypatch, record, tmp_path)
    campaign = tmp_path / "campaign.yaml"
    campaign.write_text("sealed completed campaign\n", encoding="utf-8")
    output = tmp_path / "figures"

    with pytest.raises(ValueError, match="compiler package differs"):
        figures.generate_cpu_host_resource_figure(campaign, output_dir=output)
    assert not output.exists()


def test_arm_outcome_figure_counts_a_digest_bound_level_failure(
        tmp_path: Path, monkeypatch) -> None:
    record = _campaign(tmp_path)
    row = next(value for value in record["runs"]
               if value["arm"] == "arm1_raw_cpp" and value["repeat"] == 0)
    grader_path = Path(row["run_dir"]) / "metrics" / "grader_result.json"
    grader = json.loads(grader_path.read_text(encoding="utf-8"))
    grader["status"] = "fail"
    grader["levels"]["L2"]["status"] = "fail"
    grader_path.write_text(json.dumps(grader), encoding="utf-8")
    row["outcome"] = "graded_fail"
    row["grader_result_sha256"] = figures._sha256(grader_path)
    _install_spec(monkeypatch, record, tmp_path)
    campaign = tmp_path / "campaign.yaml"
    campaign.write_text("sealed completed campaign\n", encoding="utf-8")

    output = figures.generate_cpu_host_resource_figure(
        campaign, output_dir=tmp_path / "figures")
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["outcome_summaries"]["levels"]["arm1_raw_cpp"]["L2"] == {
        "pass": 3, "fail": 1, "not_reached": 0}


def test_arm_outcome_figure_rejects_selected_action_outside_frozen_catalogue(
        tmp_path: Path, monkeypatch) -> None:
    record = _campaign(tmp_path)
    row = next(value for value in record["runs"]
               if value["arm"] == "arm4_agentic_pass_authoring" and value["repeat"] == 0)
    package = Path(row["run_dir"]) / "artifacts" / "compiler_submission"
    unknown = {**_ACTION, "id": "model_specific_secret_action"}
    selected, parent = _candidate([unknown]), _candidate([])
    encoded = json.dumps(selected, sort_keys=True)
    policy_path = package / "search" / "selected_policy.json"
    manifest_policy = package / "policy.yaml"
    policy_path.write_text(encoded, encoding="utf-8")
    manifest_policy.write_text(encoded, encoding="utf-8")
    record_path = package / "search" / "search_record.json"
    search_record = json.loads(record_path.read_text(encoding="utf-8"))
    search_record["selected_policy_sha256"] = figures._sha256(policy_path)
    for entry in search_record["accepted"]:
        entry["candidate"] = selected
        for split in ("train", "validation"):
            entry[split]["parent_candidate_sha256"] = parent["candidate_sha256"]
            entry[split]["candidate_sha256"] = selected["candidate_sha256"]
    record_path.write_text(json.dumps(search_record), encoding="utf-8")
    seal_path = Path(row["run_dir"]) / "contracts" / "trusted_search_seal.json"
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    seal["search_record_sha256"] = figures._sha256(record_path)
    seal["selected_policy_sha256"] = figures._sha256(policy_path)
    seal_path.write_text(json.dumps(seal), encoding="utf-8")
    row["search_seal_sha256"] = figures._sha256(seal_path)
    row["compiler_seal"]["policy_sha256"] = figures._sha256(manifest_policy)
    row["compiler_seal"]["selected_policy_sha256"] = figures._sha256(policy_path)
    row["compiler_seal"]["search_record_sha256"] = figures._sha256(record_path)
    row["compiler_seal"]["compiler_package_sha256"] = figures._submission_package_digest(package)
    _install_spec(monkeypatch, record, tmp_path)
    campaign = tmp_path / "campaign.yaml"
    campaign.write_text("sealed completed campaign\n", encoding="utf-8")
    output = tmp_path / "figures"

    with pytest.raises(ValueError, match="outside the frozen optimization space"):
        figures.generate_cpu_host_resource_figure(campaign, output_dir=output)
    assert not output.exists()


def test_beam_figure_reconstructs_tree_without_exporting_private_identities(
        tmp_path: Path, monkeypatch) -> None:
    record = _campaign(tmp_path)
    spec = _install_spec(monkeypatch, record, tmp_path)
    # Beam verification uses the same loader function, whose global HostExperimentSpec lives in
    # cpu_host_experiment_figures. No treatment workspace is opened by the plotting code.
    selected_row = next(row for row in record["runs"]
                        if row["arm"] == "arm3_generated_cpu_dialect" and row["repeat"] == 0)
    _install_one_synthetic_beam(record, spec, selected_row=selected_row)
    spec.freeze["campaign_record_sha256"] = figures._canonical_sha256(record)
    campaign = tmp_path / "campaign.yaml"
    campaign.write_text("sealed completed campaign\n", encoding="utf-8")

    output = beam_figures.generate_cpu_host_beam_figures(
        campaign, output_dir=tmp_path / "beam-figures")

    assert (output / "arm3_4_beam_coverage.png").is_file()
    assert (output / "beam_arm3_generated_cpu_dialect_r01.svg").is_file()
    manifest_text = (output / "manifest.json").read_text(encoding="utf-8")
    manifest = json.loads(manifest_text)
    passing = next(cell for cell in manifest["cells"] if cell["status"] == "pass")
    assert passing["evaluation_count"] == 3
    assert passing["selected_policy"]["action_ids"] == ["dynamic_vl"]
    assert [sweep["winner_candidate_sha256"] for sweep in passing["sweeps"]] == [
        _candidate([_ACTION])["candidate_sha256"], None]
    assert manifest["privacy"] == {
        "broker_request_workspace_paths": "not_dereferenced_or_exported",
        "controller_private_capsule_identities": "omitted",
    }
    assert "private-id-never-exported" not in manifest_text
    assert "/controller/workspace/private" not in manifest_text
    assert "synthetic-public-capsule" not in manifest_text
    # One coverage pair plus one verified tree pair. Failed cells are explicit in the coverage and
    # do not receive fabricated empty trees.
    assert len(manifest["figures"]) == 4


def test_beam_figure_fails_closed_on_missing_derived_candidate_evaluation(
        tmp_path: Path, monkeypatch) -> None:
    record = _campaign(tmp_path)
    spec = _install_spec(monkeypatch, record, tmp_path)
    selected_row = next(row for row in record["runs"]
                        if row["arm"] == "arm4_agentic_pass_authoring" and row["repeat"] == 0)
    _install_one_synthetic_beam(record, spec, selected_row=selected_row)
    ledger = Path(selected_row["run_dir"]) / "metrics" / "trusted_search_ledger"
    index_path = ledger / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    missing_key = next(key for key in index["evaluations"] if key.endswith(":train:screen"))
    del index["evaluations"][missing_key]
    index_path.write_text(json.dumps(index, sort_keys=True), encoding="utf-8")
    seal_path = Path(selected_row["run_dir"]) / "contracts" / "trusted_search_seal.json"
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    seal["trusted_ledger_sha256"] = figures._sha256(index_path)
    seal["trusted_evaluation_count"] -= 1
    seal_path.write_text(json.dumps(seal, sort_keys=True), encoding="utf-8")
    selected_row["search_seal_sha256"] = figures._sha256(seal_path)
    spec.freeze["campaign_record_sha256"] = figures._canonical_sha256(record)
    campaign = tmp_path / "campaign.yaml"
    campaign.write_text("sealed completed campaign\n", encoding="utf-8")
    output = tmp_path / "beam-figures"

    with pytest.raises(ValueError, match="request/receipt associations are incomplete"):
        beam_figures.generate_cpu_host_beam_figures(campaign, output_dir=output)
    assert not output.exists()


def test_beam_figure_rejects_a_digest_refreshed_but_structurally_false_ranking(
        tmp_path: Path, monkeypatch) -> None:
    record = _campaign(tmp_path)
    spec = _install_spec(monkeypatch, record, tmp_path)
    selected_row = next(row for row in record["runs"]
                        if row["arm"] == "arm3_generated_cpu_dialect" and row["repeat"] == 0)
    _install_one_synthetic_beam(record, spec, selected_row=selected_row)
    package = Path(selected_row["run_dir"]) / "artifacts" / "compiler_submission"
    record_path = package / "search" / "search_record.json"
    search_record = json.loads(record_path.read_text(encoding="utf-8"))
    search_record["sweeps"][0]["screened"] = []
    record_path.write_text(json.dumps(search_record, sort_keys=True), encoding="utf-8")
    seal_path = Path(selected_row["run_dir"]) / "contracts" / "trusted_search_seal.json"
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    seal["search_record_sha256"] = figures._sha256(record_path)
    seal_path.write_text(json.dumps(seal, sort_keys=True), encoding="utf-8")
    selected_row["search_seal_sha256"] = figures._sha256(seal_path)
    selected_row["compiler_seal"]["search_record_sha256"] = figures._sha256(record_path)
    selected_row["compiler_seal"]["compiler_package_sha256"] = (
        figures._submission_package_digest(package))
    spec.freeze["campaign_record_sha256"] = figures._canonical_sha256(record)
    campaign = tmp_path / "campaign.yaml"
    campaign.write_text("sealed completed campaign\n", encoding="utf-8")
    output = tmp_path / "beam-figures"

    with pytest.raises(ValueError, match="screened ranking is incomplete"):
        beam_figures.generate_cpu_host_beam_figures(campaign, output_dir=output)
    assert not output.exists()


def test_beam_figure_requires_completed_campaign_before_output(
        tmp_path: Path, monkeypatch) -> None:
    record = _campaign(tmp_path)
    _install_spec(monkeypatch, record, tmp_path, status="protocol_frozen")
    campaign = tmp_path / "campaign.yaml"
    campaign.write_text("not completed\n", encoding="utf-8")
    output = tmp_path / "beam-figures"

    with pytest.raises(ValueError, match="completed sixteen-cell campaign"):
        beam_figures.generate_cpu_host_beam_figures(campaign, output_dir=output)
    assert not output.exists()
