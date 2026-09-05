from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import inspect
import json
import shutil
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = (ROOT / "merlin/experiments/cpu_host_compiler_v0/qualify_multicore_effect.py")
SPEC = importlib.util.spec_from_file_location("cpu_host_multicore_effect", SCRIPT)
assert SPEC and SPEC.loader
Q = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(Q)


def test_trusted_dynamic_loader_supports_dataclass_modules(tmp_path):
    module_path = tmp_path / "trusted_dataclass_module.py"
    module_path.write_text(
        "from __future__ import annotations\n"
        "from dataclasses import dataclass\n"
        "@dataclass\n"
        "class Receipt:\n"
        "    value: int\n",
        encoding="utf-8",
    )
    name = "cpu_host_multicore_effect_dynamic_dataclass_test"
    sys.modules.pop(name, None)
    module = Q._load(name, module_path)
    assert module.Receipt(7).value == 7
    assert sys.modules[name] is module
    sys.modules.pop(name, None)


def _row(*, digest: str = "1" * 64, harts: int = 8, reuse: int = 1,
         operation: str = "static_partition") -> dict:
    return {
        "core_count": harts, "dtype": "fp32", "family": "runtime_parallel",
        "id": f"runtime_parallel-{operation}-{digest[:16]}", "layout": "contiguous",
        "operation": operation, "sha256": digest, "shape": {"work_items": 1024},
        "split": "train", "state": {"reuse_count": reuse},
    }


def _write_public(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
                    encoding="utf-8")


def _submission(path: Path) -> Path:
    path.mkdir()
    (path / "manifest.yaml").write_text(
        "version: 1\nbuild:\n  command: [/bin/true]\ncompiler:\n"
        "  command: [compiler, '{input_mlir}', '{output_dir}', '{mode}', '{harts}', "
        "'{vlen_bits}']\npolicy: policy.json\n", encoding="utf-8")
    (path / "compiler").write_text("compiler\n", encoding="utf-8")
    (path / "compiler").chmod(0o755)
    (path / "policy.json").write_text("{}\n", encoding="utf-8")
    return path


def _noise(path: Path, *, public: Path, submission: Path, margin: float,
           prebuild_receipt: dict) -> Path:
    space = yaml.safe_load(Q.DEFAULT_SPACE.read_text(encoding="utf-8"))
    protocol = {
        "version": 1,
        "measurement_repeats": 6,
        "board_environment": space["board_environment"],
    }
    raw_tree = Q._tree_sha256(submission)
    raw_package = Q._package_tree_sha256(submission)
    protocol_sha = Q._canonical_sha256(protocol)
    value = {
        "version": 2,
        "kind": "cpu_host_k1_order_balanced_aa_noise_calibration",
        "status": "pass", "paid_work": False, "heldout_opened": False,
        "protocol_state_mutated": False,
        "checks": {"six_families": True, "six_valid_pairs_per_family": True,
                   "all_correct": True, "identical_k1_text": True,
                   "no_heldout_argument": True},
        "public_train_sha256": Q._sha256(public),
        "calibration_protocol": protocol,
        "calibration_protocol_sha256": protocol_sha,
        "calibration_lineage": {
            "version": 1, "stage": "noise_pre_result",
            "pre_result_protocol_sha256": protocol_sha,
            "raw_input_tree_sha256": raw_tree,
            "raw_input_package_sha256": raw_package,
            "output_field": "noise_margin",
        },
        "prebuild_input_tree_sha256": raw_tree,
        "prebuild_input_package_sha256": raw_package,
        "submission": str(submission.resolve()),
        "derived_noise_margin": margin,
        "prebuild_receipt": prebuild_receipt,
    }
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


class _Runner:
    @staticmethod
    def _candidate(actions):
        assert actions == []
        return {"version": 1, "actions": [], "candidate_sha256": "c" * 64}


class _Grader:
    def __init__(self, *, speedup: float = 2.0, invalid_first: bool = False,
                 incomplete: bool = False, prebuild_mismatch: bool = False,
                 physical_overlap_failure_call: int | None = None):
        self.speedup = speedup
        self.invalid_first = invalid_first
        self.incomplete = incomplete
        self.prebuild_mismatch = prebuild_mismatch
        self.physical_overlap_failure_call = physical_overlap_failure_call
        self.pair_gate_calls = 0
        self.probe_calls = 0
        self.grade_calls = 0
        self.entrypoint_identity = [0o755, "e" * 64]

    def expected_prebuild_receipt(self, submission):
        tree = Q._package_tree_sha256(submission)
        self.entrypoint_identity = [
            (submission / "compiler").stat().st_mode & 0o777,
            Q._sha256(submission / "compiler"),
        ]
        return {
            "version": 1, "authority": "driver_private_prebuild",
            "submitted_manifest_sha256": Q._sha256(submission / "manifest.yaml"),
            "private_manifest_sha256": Q._sha256(submission / "manifest.yaml"),
            "real_build_commands": [["/bin/true"]],
            "real_build_logs": [], "prebuild_tree_sha256": tree,
            "built_tree_sha256": "d" * 64, "sealed_prebuilt_tree_sha256": tree,
            "submitted_entrypoint_identity": None,
            "built_entrypoint_identity": self.entrypoint_identity,
            "private_build_override": ["/bin/true"],
            "policy_sha256": Q._sha256(submission / "policy.json"),
        }

    def prepare_prebuilt_search_package(self, *, submission, destination, build_override):
        assert build_override == ["/bin/true"]
        shutil.copytree(submission, destination)
        receipt = self.expected_prebuild_receipt(destination)
        if self.prebuild_mismatch:
            receipt["built_tree_sha256"] = "0" * 64
        return receipt

    @staticmethod
    def _codes(rows, field):
        return {value: index + 1 for index, value in enumerate(
            sorted({row[field] for row in rows}))}

    @staticmethod
    def _install_search_policy(package, candidate):
        assert candidate["actions"] == []

    @staticmethod
    def _build(package):
        return yaml.safe_load((package / "manifest.yaml").read_text(encoding="utf-8")), {
            "authority": "mock_private_noop_build"}

    @staticmethod
    def _freeze_tree(package):
        pass

    @staticmethod
    def _buffer_plan(row):
        count = row["shape"]["work_items"]
        return {"input0_kind": "fp32", "input1_kind": "fp32", "output_kind": "fp32",
                "input0_count": count, "input1_count": count, "output_count": count,
                "dim0": count, "dim1": 0, "dim2": 0, "state0": row["state"]["reuse_count"]}

    def _compile_one(self, package, manifest, row, mode, operation_codes, root):
        work = root / f"{row['id']}_{mode}"
        out = work / "output"
        out.mkdir(parents=True)
        input_mlir = work / "input.mlir"
        input_mlir.write_text("same public capsule input\n", encoding="utf-8")
        (out / "kernel.c").write_text(f"/* {mode} */\n", encoding="utf-8")
        (out / "lowered.mlir").write_text("same input, lowered\n", encoding="utf-8")
        harts = 1 if mode == "rvv" else row["core_count"]
        metadata = {"capsule_sha256": row["sha256"], "actual_mode": mode,
                    "requested_mode": mode, "harts": harts, "fallback_used": False}
        (out / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
        return {"capsule": row["id"], "family": row["family"], "mode": mode,
                "ok": True, "metadata": metadata,
                "input_mlir_sha256": Q._sha256(input_mlir),
                "buffer_plan": self._buffer_plan(row),
                "source_sha256": Q._sha256(out / "kernel.c"),
                "lowered_mlir_sha256": Q._sha256(out / "lowered.mlir"),
                "_kernel_path": str(out / "kernel.c")}

    @staticmethod
    def _grade_spike(row, compiled, operation_codes, root):
        return {"status": "pass", "capsule": row["id"], "checks": {
            "rvv_correctness": True, "instruction_evidence": True,
            "vlen_256": True, "cycle_measurement": True},
            "kernel_text_sha256": "d" * 64}

    @staticmethod
    def _search_spike_correct(record):
        return record["status"] == "pass" and all(record["checks"].values())

    @staticmethod
    def _k1_connection():
        return {"host": "mock"}

    @staticmethod
    def _k1_lock(connection):
        return contextlib.nullcontext()

    def _probe_k1_state(self, connection):
        self.probe_calls += 1
        invalid = self.invalid_first and self.probe_calls == 2
        return {"authority": "mock", "returncode": 0,
                "state": {"ready": True, "pair_valid": not invalid}}

    @staticmethod
    def _k1_state_ready(probe, contract):
        return probe["returncode"] == 0 and probe["state"]["ready"]

    def _k1_state_pair_ok(self, before, after, contract):
        self.pair_gate_calls += 1
        return bool(after["state"]["pair_valid"])

    @staticmethod
    def _k1_timing_authority(metrics, harts, output_count):
        return True, True

    def _grade_k1(self, row, compiled, operation_codes, root, *, seed):
        self.grade_calls += 1
        mode = compiled["mode"]
        harts = 1 if mode == "rvv" else row["core_count"]
        per_call = 2000.0 if mode == "rvv" else 2000.0 / self.speedup
        checks = {name: True for name in Q.REQUIRED_K1_CHECKS}
        if self.incomplete:
            checks.pop("audit_attribution")
        work = root / f"{row['id']}_{mode}"
        work.mkdir(parents=True)
        artifacts = {
            "kernel.o": f"kernel-object-{mode}".encode(),
            "trusted_harness.o": b"trusted-harness-object",
            "kernel.text.bin": f"kernel-text-{mode}".encode(),
            "capsule_k1": f"binary-{mode}-{seed}-{self.grade_calls}".encode(),
        }
        for name, payload in artifacts.items():
            (work / name).write_bytes(payload)
        output_count = self._buffer_plan(row)["output_count"]
        metrics = {
            "wall_ns": int(per_call * 20), "calls": 20,
            "audit_output_coverage": output_count,
            "audit_owner_min_elements": output_count // harts,
            "audit_owner_max_elements": (output_count + harts - 1) // harts,
            "audit_ownership_violations": 0, "audit_balanced_shards": 1,
        }
        expected = list(range(harts))
        child_stdout = "".join(
            f"K1_METRIC {name} {value}\n" for name, value in metrics.items()) + (
            f"MERLIN_TRUSTED_RESULT version=1 seed={seed} nonce={self.grade_calls} "
            "memory=1 numeric=1\n")
        return {
            "capsule": row["id"], "family": row["family"], "mode": mode,
            "harts": harts, "status": "pass", "seed": seed, "checks": checks,
            "metrics": metrics,
            "monitor": {"returncode": 0, "timed_out": False,
                        "requested_harts": harts, "wall_ns": 1,
                        "affinity_samples": ["0" if harts == 1 else f"0-{harts-1}"],
                        "pinned_affinities_observed": expected,
                        "pinned_runtime_cpus": expected, "active_cpus": expected,
                        "running_cpus_observed": expected,
                        "max_simultaneous_running_cpus": (
                            1 if harts == 1 or self.grade_calls ==
                            self.physical_overlap_failure_call else harts),
                        "child_stdout": child_stdout},
            "receipt_nonce": self.grade_calls,
            "build_returncode": 0, "kernel_text_sha256": hashlib.sha256(
                artifacts["kernel.text.bin"]).hexdigest(),
            "local_sha256": hashlib.sha256(artifacts["capsule_k1"]).hexdigest(),
            "remote_sha256": hashlib.sha256(artifacts["capsule_k1"]).hexdigest(),
            "ssh_returncode": 0,
            "board_wall_seconds": 0.5,
        }


def _run(tmp_path: Path, grader: _Grader, *, noise_margin: float | None = None):
    row = _row()
    public = tmp_path / "train.jsonl"
    _write_public(public, [row])
    submission = _submission(tmp_path / "submission")
    margin = float(yaml.safe_load(Q.DEFAULT_SPACE.read_text(encoding="utf-8"))["noise_margin"])
    noise = _noise(tmp_path / "noise.json", public=public, submission=submission,
                   margin=margin if noise_margin is None else noise_margin,
                   prebuild_receipt=grader.expected_prebuild_receipt(submission))
    original_path, original_sha = Q.DEFAULT_PUBLIC_TRAIN, Q.CANONICAL_PUBLIC_TRAIN_SHA256
    Q.DEFAULT_PUBLIC_TRAIN, Q.CANONICAL_PUBLIC_TRAIN_SHA256 = public, Q._sha256(public)
    try:
        return Q.qualify(
            submission=submission, public_train=public, space_path=Q.DEFAULT_SPACE,
            noise_authority=noise, prebuilt_destination=tmp_path / "prebuilt",
            receipts_destination=tmp_path / "receipts", grader=grader, runner=_Runner(),
            semantic_noise_validator=lambda **_kwargs: [])
    finally:
        Q.DEFAULT_PUBLIC_TRAIN, Q.CANONICAL_PUBLIC_TRAIN_SHA256 = original_path, original_sha


def test_deterministic_capsule_selection_is_pre_measurement_and_public():
    rows = [
        _row(digest="f" * 64, harts=4, reuse=64),
        _row(digest="e" * 64, harts=8, reuse=8),
        _row(digest="2" * 64, harts=8, reuse=32),
        _row(digest="3" * 64, harts=8, reuse=1),
        _row(digest="1" * 64, harts=8, reuse=32, operation="producer_consumer"),
        _row(digest="0" * 64, harts=8, reuse=128, operation="single_hart"),
    ]
    selected = Q.select_public_runtime_capsule(rows)
    assert selected["sha256"] == "3" * 64
    assert selected["operation"] == "static_partition"
    assert selected["state"]["reuse_count"] == 1
    assert selected["split"] == "train"


def test_selection_rejects_repeated_work_amplification():
    try:
        Q.select_public_runtime_capsule([_row(reuse=65536)])
    except ValueError as error:
        assert "static_partition multicore capsule" in str(error)
    else:
        raise AssertionError("qualification accepted an amplified repeated-work capsule")


def test_pass_retains_full_balanced_raw_receipts_and_speedup(tmp_path):
    result = _run(tmp_path, _Grader(speedup=2.0))
    assert result["status"] == "pass"
    assert result["heldout_opened"] is False
    assert result["checks"]["equivalent_semantic_work_per_call"] is True
    assert result["effect"]["effect_direction"] == "multicore_faster"
    assert result["effect"]["multicore_speedup_qualified"] is True
    assert result["transcript"]["pair_orders"] == list(Q.PAIR_ORDERS)
    assert result["transcript"]["k1_program_count"] == 12
    assert len(result["raw_receipts"]["pairs"]) == 6
    assert len(result["raw_receipts"]["k1_build_artifacts"]) == 48
    assert result["publication_bundle_replay"]["status"] == "pass"
    assert result["checks"]["logical_shards_balanced"] is True
    assert result["checks"]["physical_process_window_observed"] is True
    assert {row["name"] for row in result["raw_receipts"]["compiled_artifacts"]} == {
        "input.mlir", "kernel.c", "lowered.mlir", "metadata.json"}
    assert Q._tree_sha256(Path(result["raw_receipts"]["root"])) == \
           result["raw_receipts"]["tree_sha256"]
    for receipt in result["raw_receipts"]["pairs"]:
        path = Path(receipt["path"])
        assert path.is_file()
        assert Q._sha256(path) == receipt["sha256"]
        pair = json.loads(path.read_text(encoding="utf-8"))
        assert set(pair["measurements"]) == {"rvv", "rvv_multicore"}
        assert set(pair["measurements"]["rvv"]["evidence"]["checks"]) == \
               Q.REQUIRED_K1_CHECKS


def test_invalid_environment_pair_is_retained_and_replaced_without_path_collision(tmp_path):
    grader = _Grader(speedup=2.0, invalid_first=True)
    result = _run(tmp_path, grader)
    assert result["status"] == "pass"
    assert result["transcript"]["attempt_count"] == 7
    assert result["transcript"]["k1_program_count"] == 14
    assert len(result["transcript"]["excluded_pairs"]) == 1
    attempts = [Path(row["path"]).name for row in result["raw_receipts"]["pairs"]]
    assert len(attempts) == len(set(attempts)) == 7


def test_incomplete_grader_receipt_fails_closed(tmp_path):
    try:
        _run(tmp_path, _Grader(incomplete=True))
    except ValueError as error:
        assert "complete passing grader K1 evidence" in str(error)
    else:
        raise AssertionError("qualification accepted incomplete K1 evidence")


def test_consistent_slowdown_is_effect_but_never_a_speedup_claim(tmp_path):
    result = _run(tmp_path, _Grader(speedup=0.5))
    assert result["status"] == "pass"
    assert result["effect"]["effect_direction"] == "multicore_slower"
    assert result["effect"]["directionally_consistent_effect"] is True
    assert result["effect"]["multicore_speedup_qualified"] is False


def test_effect_within_aa_margin_is_a_retained_negative_result(tmp_path):
    result = _run(tmp_path, _Grader(speedup=1.0))
    assert result["status"] == "fail"
    assert result["effect"]["effect_direction"] == "not_distinguishable_from_aa_noise"
    assert result["checks"]["directionally_consistent_effect"] is False


def test_noise_margin_must_equal_final_frozen_space(tmp_path):
    try:
        _run(tmp_path, _Grader(), noise_margin=0.987)
    except ValueError as error:
        assert "final derived margin" in str(error)
    else:
        raise AssertionError("qualification accepted an unbound A/A margin")


def test_api_and_cli_have_no_heldout_input():
    assert "heldout" not in inspect.signature(Q.qualify).parameters
    assert "--heldout" not in SCRIPT.read_text(encoding="utf-8")


def test_noncanonical_train_path_fails_before_any_measurement(tmp_path):
    path = tmp_path / "renamed-train.jsonl"
    _write_public(path, [_row()])
    try:
        Q._require_canonical_public_train(path)
    except ValueError as error:
        assert "exact canonical public train" in str(error)
    else:
        raise AssertionError("qualification accepted a noncanonical train path")


def test_exact_aa_prebuilt_copy_rejects_any_sealed_tree_drift(tmp_path):
    grader = _Grader()
    submission = _submission(tmp_path / "submission")
    receipt = grader.expected_prebuild_receipt(submission)
    noise = {"submission": str(submission)}
    (submission / "policy.json").write_text('{"drift":true}\n', encoding="utf-8")
    try:
        Q._copy_aa_sealed_prebuilt(
            noise=noise, aa_receipt=receipt, destination=tmp_path / "copy")
    except ValueError as error:
        assert "sealed prebuilt package differs" in str(error)
    else:
        raise AssertionError("qualification copied a prebuilt tree that differed from A/A")


def test_publication_replay_rejects_raw_receipt_tampering(tmp_path):
    result = _run(tmp_path, _Grader(speedup=2.0))
    pair = Path(result["raw_receipts"]["pairs"][0]["path"])
    value = json.loads(pair.read_text(encoding="utf-8"))
    value["measurements"]["rvv"]["elapsed_ns"] += 1
    pair.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    try:
        Q.verify_saved_qualification(result, grader=_Grader())
    except ValueError as error:
        assert "raw receipt tree differs" in str(error)
    else:
        raise AssertionError("publication replay accepted a modified raw pair receipt")


def test_physical_process_window_gate_is_narrow_and_reports_sampling_limit():
    monitor = {
        "requested_harts": 8, "timed_out": False, "wall_ns": 1,
        "affinity_samples": ["0-7"], "pinned_affinities_observed": [0, 3],
        "pinned_runtime_cpus": [0, 3], "active_cpus": [0, 3],
        "running_cpus_observed": [0, 3], "max_simultaneous_running_cpus": 2,
    }
    evidence = Q._physical_process_window_evidence(monitor, 8)
    assert evidence["qualified"] is True
    assert evidence["maximum_simultaneously_sampled_running_cpus"] == 2
    assert "does not claim that all requested harts" in evidence["claim"]
    monitor["max_simultaneous_running_cpus"] = 1
    assert Q._physical_process_window_evidence(monitor, 8)["qualified"] is False
    monitor["max_simultaneous_running_cpus"] = 2
    monitor["active_cpus"] = [9]
    assert Q._physical_process_window_evidence(monitor, 8)["qualified"] is False


def test_live_observability_failure_fails_closed_with_exact_retained_receipt(tmp_path):
    grader = _Grader(physical_overlap_failure_call=2)
    try:
        _run(tmp_path, grader)
    except ValueError as error:
        assert "physical_process_window.sampled_parallel_overlap" in str(error)
        assert "live K1 evidence failed closed" in str(error)
    else:
        raise AssertionError("qualification retried a failed physical observation")
    failures = list((tmp_path / "receipts" / "failures").glob("*.json"))
    assert [path.name for path in failures] == [
        "attempt_00_pair_00_rvv_multicore.json"]
    failure = json.loads(failures[0].read_text(encoding="utf-8"))
    assert failure["mode"] == "rvv_multicore"
    assert failure["attempt_id"] == 0
    assert failure["pair_id"] == 0
    assert failure["assessment"]["checks"][
        "physical_process_window.sampled_parallel_overlap"] is False
    assert failure["evidence"]["monitor"][
        "max_simultaneous_running_cpus"] == 1
    assert {entry["name"] for entry in failure["retained_build_artifacts"]} == set(
        Q._K1_ARTIFACT_NAMES)
