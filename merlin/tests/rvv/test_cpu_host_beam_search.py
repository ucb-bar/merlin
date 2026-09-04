"""Executable, non-agentic beam search supplied to CPU-host Arms 3 and 4."""
from __future__ import annotations

import importlib.util
import json
import math
import statistics
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import yaml
import pytest

from merlin.common.paths import repo_root


RUNNER = repo_root() / "merlin/experiments/cpu_host_compiler_v0/beam_search.py"
SPACE = repo_root() / "merlin/experiments/cpu_host_compiler_v0/optimization_space_v1.yaml"
README = repo_root() / "merlin/experiments/cpu_host_compiler_v0/README.md"
CALIBRATOR = repo_root() / "merlin/experiments/cpu_host_compiler_v0/calibrate_search_noise.py"


def _module():
    spec = importlib.util.spec_from_file_location("cpu_host_beam_search", RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _broker_module():
    path = repo_root() / "merlin/experiments/cpu_host_compiler_v0/trusted_search_broker.py"
    spec = importlib.util.spec_from_file_location("cpu_host_trusted_broker", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _calibrator_module():
    spec = importlib.util.spec_from_file_location("cpu_host_noise_calibrator", CALIBRATOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _capsules(split: str) -> list[dict]:
    families = [
        ("contraction", "matmul", "fp32", "row_row", 1),
        ("elementwise_map", "relu", "fp32", "contiguous", 1),
        ("reduction", "sum", "fp32", "contiguous", 1),
        ("movement_layout", "copy", "fp32", "operation_defined", 1),
        ("fusion_epilogue", "matmul_bias", "fp32", "row_row", 1),
        ("runtime_parallel", "static_partition", "fp32", "contiguous", 4),
    ]
    return [{"id": f"{split}-{family}", "sha256": f"{index + 1:064x}", "split": split,
             "family": family, "operation": operation, "dtype": dtype, "layout": layout,
             "core_count": cores}
            for index, (family, operation, dtype, layout, cores) in enumerate(families)]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_curated_search_space_is_generic_and_non_agentic():
    raw = yaml.safe_load(SPACE.read_text(encoding="utf-8"))
    assert raw["selection"]["screen"] == "trusted_spike_mcycle_train"
    assert raw["selection"]["rank_split"] == "k1_train_confirmation"
    assert raw["selection"]["promotion_split"] == "k1_validation_confirmation"
    assert raw["selection"]["heldout_visible"] is False
    assert raw["confirmation_families"] == [
        "contraction", "elementwise_map", "reduction", "movement_layout",
        "fusion_epilogue", "runtime_parallel"]
    assert raw["confirmation_width"] == 1
    assert raw["screen_samples_per_family"] == 2
    assert raw["search_package"] == {
        "submitted_manifest_requires_real_build": True,
        "controller_private_prebuild": True,
        "private_build_override": ["/bin/true"],
        "candidate_time_build_then_forbidden": True,
    }
    assert raw["measurement_repeats"] == 6
    assert raw["required_empty_sweeps"] == 1
    assert raw["budget"]["maximum_confirmation_requests"] == 20
    assert raw["budget"]["confirmation_package_builds"] == 40
    assert raw["budget"]["confirmation_compiler_invocations"] == 280
    assert raw["budget"]["confirmation_spike_checks"] == 240
    assert raw["budget"]["k1_program_invocations"] == 1920
    encoded = json.dumps(raw).lower()
    assert not any(name in encoded for name in (
        "gemma2_2b", "tinyllama_1_1b", "smolvla", "resnet50_v1_5", "lstmnetvit"))


def test_readme_matches_frozen_width_one_search_budget():
    raw = yaml.safe_load(SPACE.read_text(encoding="utf-8"))
    budget = raw["budget"]
    prose = " ".join(README.read_text(encoding="utf-8").split())
    assert raw["confirmation_width"] == 1
    assert "deterministic width-one top survivor" in prose
    assert "two from each of all six generic families" in prose
    assert "one controller-private post-freeze shape from each of all six generic families" in prose
    for key, label in (
        ("maximum_confirmation_requests", "confirmation requests"),
        ("confirmation_package_builds", "policy-package builds"),
        ("confirmation_compiler_invocations", "compiler invocations"),
        ("confirmation_spike_checks", "Spike confirmation checks"),
        ("k1_program_invocations", "K1 program invocations"),
    ):
        assert f"{budget[key]:,} {label}" in prose
    assert (
        f"complete search budget is {budget['expected_search_seconds']:,} seconds expected "
        f"and {budget['planning_upper_search_seconds']:,} seconds planning-upper"
    ) in prose
    assert (
        f"{budget['planning_upper_seconds_per_k1_program']} seconds planning-upper "
        "per K1 program"
    ) in prose


def test_curated_confirmation_panel_covers_all_six_generic_families():
    module = _module()
    raw = module._load_space(SPACE)
    rows = module.select_semantic_sample(
        _capsules("train"), per_family=raw["confirmation_samples_per_family"],
        families=raw["confirmation_families"])
    assert [row["family"] for row in rows] == [
        "contraction", "elementwise_map", "fusion_epilogue", "movement_layout",
        "reduction", "runtime_parallel"]


def test_curated_spike_panel_covers_all_six_generic_families():
    module = _module()
    raw = module._load_space(SPACE)
    rows = module.select_semantic_sample(
        _capsules("train"), per_family=raw["screen_samples_per_family"])
    assert {row["family"] for row in rows} == {
        "contraction", "elementwise_map", "reduction", "movement_layout",
        "fusion_epilogue", "runtime_parallel",
    }


def test_trusted_broker_deadline_fails_closed_before_an_evaluation():
    broker = _broker_module()
    with pytest.raises(RuntimeError, match="planning wall budget"):
        with broker._wall_deadline(time.monotonic_ns() - 1):
            raise AssertionError("expired trusted evaluation was entered")


def test_trusted_broker_rejects_unmeasured_source_below_submission_search(tmp_path):
    broker = _broker_module()
    submission = tmp_path / "submission"
    (submission / "search").mkdir(parents=True)
    (submission / "search" / "compiler_helper.py").write_text("# unmeasured dependency\n")
    with pytest.raises(ValueError, match="must be empty"):
        broker._reject_presearch_files(submission)


def test_broker_tree_digest_includes_nested_directory_named_search(tmp_path):
    broker = _broker_module()
    submission = tmp_path / "submission"
    nested = submission / "src" / "search"
    nested.mkdir(parents=True)
    (submission / "manifest.yaml").write_text("version: 1\npolicy: policy.json\n")
    (submission / "policy.json").write_text("{}\n")
    source = nested / "compiler.cpp"
    source.write_text("one\n")
    before = broker._tree_digest(submission)
    source.write_text("two\n")
    assert broker._tree_digest(submission) != before


def test_controller_private_capsule_shapes_are_changed_and_reproducible():
    broker = _broker_module()
    row = {"id": "public", "sha256": "a" * 64, "split": "train",
           "family": "contraction", "operation": "matmul", "dtype": "fp32",
           "shape": {"M": 3, "N": 5, "K": 7}, "layout": "row_row",
           "state": "stateless", "core_count": 1}
    first = broker._private_capsule(row, secret=b"s" * 32, phase="screen", split="train")
    second = broker._private_capsule(row, secret=b"s" * 32, phase="screen", split="train")
    assert first == second
    assert first["shape"] != row["shape"]
    assert first["id"] != row["id"] and first["sha256"] != row["sha256"]


def test_beam_search_uses_exact_public_samples_and_converges(tmp_path: Path):
    module = _module()
    train, validation = tmp_path / "train.jsonl", tmp_path / "validation.jsonl"
    _write_jsonl(train, _capsules("train")); _write_jsonl(validation, _capsules("validation"))
    space = tmp_path / "space.yaml"
    space.write_text(yaml.safe_dump({
        "version": 1, "status": "frozen_definition", "screen_samples_per_family": 1,
        "confirmation_samples_per_family": 1, "confirmation_width": 1,
        "max_sweeps": 4, "measurement_repeats": 3, "noise_margin": 0.02,
        "minimum_families": 3, "required_empty_sweeps": 1,
        "actions": [{"id": "generic_fast", "group": "schedule", "action_class": "knob",
                     "stage": 1, "value": "fast"}],
    }), encoding="utf-8")
    evaluator = tmp_path / "evaluator.py"
    evaluator.write_text('''
import argparse, json
p=argparse.ArgumentParser(); p.add_argument("--phase"); p.add_argument("--policy"); p.add_argument("--parent-policy"); p.add_argument("--capsules")
p.add_argument("--split"); p.add_argument("--repeats",type=int); p.add_argument("--output")
a=p.parse_args(); rows=[]
for line in open(a.capsules):
 c=json.loads(line); row={"capsule_id":c["id"],"family":c["family"],
 "correctness_ok":True,"baseline_code_sha256":"a"*64,"candidate_code_sha256":"b"*64}
 if a.phase=="screen": row.update(baseline_cycles=100,candidate_cycles=80)
 else: row.update(baseline_elapsed_ns=[500000000]*a.repeats,baseline_calls=[100]*a.repeats,
                  candidate_elapsed_ns=[500000000]*a.repeats,candidate_calls=[200]*a.repeats)
 rows.append(row)
with open(a.output,"w") as f:
 [f.write(json.dumps(row)+"\\n") for row in rows]
''', encoding="utf-8")
    record = module.run_search(
        space_path=space, train_path=train, validation_path=validation,
        evaluator=[sys.executable, str(evaluator)], output=tmp_path / "out")
    assert record["status"] == "converged"
    assert record["heldout_visible"] is False
    assert record["sample_counts"] == {"screen_train": 6, "confirmation_train": 6,
                                        "confirmation_validation": 6}
    assert record["empty_sweeps"] == 1
    assert len(record["accepted"]) == 1
    assert record["accepted"][0]["validation"]["median_speedup"] == 2.0
    selected = json.loads((tmp_path / "out" / "selected_policy.json").read_text())
    assert selected == record["accepted"][0]["candidate"]
    assert "status" not in selected


def test_paired_speedup_preserves_pairs_instead_of_ratio_of_medians():
    module = _module()
    baseline_elapsed = [1, 100, 101]
    candidate_elapsed = [1, 2, 100]
    calls = [1, 1, 1]
    paired = module._median_paired_speedup(
        baseline_elapsed, calls, candidate_elapsed, calls)
    ratio_of_medians = statistics.median(baseline_elapsed) / statistics.median(
        candidate_elapsed)
    assert paired == 1.01
    assert ratio_of_medians == 50.0


def test_promotion_uses_reciprocal_log_symmetric_regression_bound():
    module = _module()
    train = {"affected_median_speedup": 1.3}
    validation = {
        "failures": [], "families": ["a", "b", "c"],
        "affected_median_speedup": 1.3, "minimum_speedup": 0.81,
    }
    # At a 20% upper tolerance, 0.81 passes the asymmetric 1-margin rule but is below the
    # reciprocal log-symmetric bound 1/1.2.
    assert module._promotion_eligible(train, validation, 3, 0.2) is False
    validation["minimum_speedup"] = 0.84
    assert module._promotion_eligible(train, validation, 3, 0.2) is True


def test_aa_margin_covers_maximum_log_deviation_with_frozen_padding_and_rounding():
    calibration = _calibrator_module()
    derived = calibration._derive_multiplicative_margin(
        [math.exp(-0.03), 1.0, math.exp(0.01)])
    assert derived["maximum_absolute_log_ratio"] == pytest.approx(0.03)
    assert derived["upper_margin"] == pytest.approx(0.036)
    assert derived["upper_speedup_bound"] == pytest.approx(1.036)
    assert derived["lower_speedup_bound"] == pytest.approx(1.0 / 1.036)
    with pytest.raises(ValueError, match="finite and positive"):
        calibration._derive_multiplicative_margin([1.0, 0.0])


def test_aa_calibration_protocol_excludes_derived_margin_and_keeps_twelve_capsules():
    calibration = _calibrator_module()
    space = yaml.safe_load(SPACE.read_text(encoding="utf-8"))
    protocol = calibration._calibration_protocol(space)
    assert "noise_margin" not in protocol
    assert (protocol["confirmation_samples_per_family"] *
            len(protocol["confirmation_families"])) == 6
    assert space["screen_samples_per_family"] * len(protocol["confirmation_families"]) == 12
    changed = dict(space)
    changed["noise_margin"] = float(space["noise_margin"]) + 0.123
    assert calibration._calibration_protocol(changed) == protocol


def test_semantic_sample_rejects_a_sealed_row():
    module = _module()
    rows = _capsules("train")
    rows[0]["split"] = "heldout"
    try:
        module.select_semantic_sample(rows)
    except ValueError as exc:
        assert "public train/validation" in str(exc)
    else:
        raise AssertionError("sealed capsule was accepted")


def test_evaluator_requires_real_sha256_code_digests(tmp_path: Path):
    module = _module()
    parent = module._candidate([])
    candidate = module._candidate([{"id": "x", "group": "g", "action_class": "knob",
                                    "stage": 1, "value": 1,
                                    "affected_families": ["contraction", "elementwise_map",
                                      "reduction", "movement_layout", "fusion_epilogue",
                                      "runtime_parallel"]}])
    evaluator = tmp_path / "bad_digest.py"
    evaluator.write_text('''
import argparse,json
p=argparse.ArgumentParser(); p.add_argument("--phase"); p.add_argument("--policy"); p.add_argument("--parent-policy"); p.add_argument("--capsules")
p.add_argument("--split"); p.add_argument("--repeats",type=int); p.add_argument("--output")
a=p.parse_args(); out=open(a.output,"w")
for line in open(a.capsules):
 c=json.loads(line); out.write(json.dumps({"capsule_id":c["id"],"family":c["family"],
 "baseline_elapsed_ns":[10000]*a.repeats,"baseline_calls":[100]*a.repeats,
 "candidate_elapsed_ns":[16000]*a.repeats,"candidate_calls":[200]*a.repeats,
 "correctness_ok":True,
 "baseline_code_sha256":"not-a-digest","candidate_code_sha256":"also-bad"})+"\\n")
out.close()
''')
    result = module._evaluate(parent, candidate, "train", _capsules("train"),
                              [sys.executable, str(evaluator)], tmp_path / "evaluations", 5)
    assert result["failures"]
    assert all("SHA-256" in failure for failure in result["failures"])


def test_trusted_evaluator_balances_six_pairs_after_one_invalid_replacement(
        tmp_path: Path, monkeypatch):
    grader_path = repo_root() / "merlin/experiments/cpu_host_compiler_v0/grader.py"
    spec = importlib.util.spec_from_file_location("test_trusted_host_grader", grader_path)
    grader = importlib.util.module_from_spec(spec); spec.loader.exec_module(grader)
    submission = tmp_path / "submission"; submission.mkdir()
    (submission / "manifest.yaml").write_text("version: 1\npolicy: policy.json\n")
    (submission / "policy.json").write_text("{}\n")
    monkeypatch.setattr(grader, "_build", lambda package: (
        {"policy": "policy.json", "compiler": {"command": []}}, {}))
    monkeypatch.setattr(grader, "_freeze_tree", lambda package: None)
    sequence = []
    seeds = []
    compile_modes = []
    def compile_one(package, manifest, row, mode, operation_codes, root):
        label = package.parents[0].name
        compile_modes.append((label, mode))
        return {"ok": True, "mode": mode,
                "metadata": {"harts": 8 if mode == "rvv_multicore" else 1},
                "source_sha256": ("a" if label == "parent" else "b") * 64,
                "_kernel_path": str(package / "kernel.c"), "label": label}
    monkeypatch.setattr(grader, "_compile_one", compile_one)
    monkeypatch.setattr(grader, "_grade_spike", lambda *args, **kwargs: {
        "checks": {"rvv_correctness": True, "instruction_evidence": True,
                   "vlen_256": True, "cycle_measurement": True},
        "kernel_text_sha256": ("a" if args[1]["label"] == "parent" else "b") * 64})
    def grade_k1(row, record, operation_codes, root, *, seed=None):
        sequence.append(record["label"])
        seeds.append(seed)
        return {"status": "pass", "seed": seed,
                "kernel_text_sha256": ("a" if record["label"] == "parent" else "b") * 64,
                "metrics": {"wall_ns": len(sequence) * 10, "calls": 10}}
    monkeypatch.setattr(grader, "_grade_k1", grade_k1)
    monkeypatch.setattr(grader, "_k1_connection", lambda: {"host": "fake"})
    monkeypatch.setattr(grader, "_k1_lock", lambda connection: nullcontext())
    monkeypatch.setattr(grader, "_probe_k1_state", lambda connection: {
        "authority": "driver_ssh_sysfs_procfs", "returncode": 0, "state": {}})
    monkeypatch.setattr(grader, "_k1_state_ready", lambda probe, contract: True)
    pair_gate_calls = 0
    def pair_ok(before, after, contract):
        nonlocal pair_gate_calls
        pair_gate_calls += 1
        return pair_gate_calls > 1
    monkeypatch.setattr(grader, "_k1_state_pair_ok", pair_ok)
    row = {"id": "train-runtime", "family": "runtime_parallel",
           "operation": "static_partition", "core_count": 8}
    parent = {"version": 1, "candidate_sha256": "a" * 64, "actions": []}
    candidate = {"version": 1, "candidate_sha256": "c" * 64,
                 "actions": [{"id": "x"}]}
    observations = grader.evaluate_public_policy_k1(
        submission=submission, capsules=[row], parent=parent, candidate=candidate, repeats=6,
        public_rows=[row], board_environment={
            "settle_attempts": 1, "settle_interval_seconds": 0,
            "maximum_invalid_pair_replacements_per_capsule": 2})
    assert sequence == ["parent", "candidate", "parent", "candidate",
                        "candidate", "parent", "candidate", "parent",
                        "parent", "candidate", "parent", "candidate", "candidate", "parent"]
    assert compile_modes == [
        ("parent", "rvv"), ("parent", "rvv_multicore"),
        ("candidate", "rvv"), ("candidate", "rvv_multicore")]
    assert observations[0]["baseline_elapsed_ns"] == [30, 60, 80, 90, 110, 140]
    assert observations[0]["candidate_elapsed_ns"] == [40, 50, 70, 100, 120, 130]
    assert observations[0]["baseline_calls"] == [10] * 6
    assert observations[0]["candidate_calls"] == [10] * 6
    assert len(observations[0]["excluded_board_condition_pairs"]) == 1
    assert observations[0]["excluded_board_condition_pairs"][0]["attempt_id"] == 0
    assert observations[0]["k1_program_count"] == 14
    assert all(seeds[index] == seeds[index + 1] for index in range(0, 14, 2))
    assert [pair["seed"] for pair in observations[0]["board_condition_pairs"]] == seeds[2::2]
    assert observations[0]["correctness_ok"] is True
    assert observations[0]["baseline_code_sha256"] == "a" * 64
    assert observations[0]["candidate_code_sha256"] == "b" * 64
    assert observations[0]["code_digest_authority"] == "measured_k1_kernel_object_text_section"
