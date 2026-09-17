#!/usr/bin/env python3
"""Measure order-balanced A/A K1 throughput noise on the frozen public confirmation panel."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import secrets
from pathlib import Path
from typing import Any

import yaml

from merlin.common.artifacts import finish_run, start_run
from merlin.common.paths import repo_root


HERE = Path(__file__).resolve().parent
DEFAULT_SPACE = HERE / "optimization_space_v1.yaml"
DERIVATION = (
    "margin=max(0.02,ceil((exp(max(abs(log(pair_ratio)))+0.005)-1)*1000)/1000); "
    "lower_bound=1/(1+margin)")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tree_sha256(root: Path) -> str:
    """Content-address one calibration package without trusting directory metadata."""
    rows: list[tuple[str, str, str]] = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            rows.append((relative, "symlink", path.readlink().as_posix()))
        elif path.is_file():
            rows.append((relative, "file", _sha256(path)))
    return hashlib.sha256(json.dumps(
        rows, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()


def _calibration_protocol(space: dict[str, Any]) -> dict[str, Any]:
    """Project only pre-result inputs; ``noise_margin`` is the calibration output."""
    return {
        "version": 1,
        "confirmation_samples_per_family": int(space["confirmation_samples_per_family"]),
        "confirmation_families": list(space["confirmation_families"]),
        "measurement_repeats": int(space["measurement_repeats"]),
        "board_environment": dict(space["board_environment"]),
        "private_shape_authority":
            "trusted_broker_private_capsule_independent_calibration_nonce",
        "public_context": "complete_public_train",
        "search_package_authority": "driver_private_prebuild",
        "derivation": DERIVATION,
    }


def _canonical_sha256(value: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()


def _derive_multiplicative_margin(ratios: list[float]) -> dict[str, float]:
    """Return a conservative symmetric log-throughput A/A tolerance.

    With only 36 predeclared pairs, an empirical percentile is too unstable to justify a paper
    threshold.  Cover the maximum absolute log-ratio observed in the full panel, add 0.005 in log
    space, round the upper multiplicative margin upward to 0.1 percentage point, and retain the
    predeclared 2% floor.  The reciprocal lower bound preserves symmetry in ratio space.
    """
    if not ratios or any(not math.isfinite(value) or value <= 0 for value in ratios):
        raise ValueError("A/A calibration ratios must be finite and positive")
    maximum_log_deviation = max(abs(math.log(value)) for value in ratios)
    padded_log_half_width = maximum_log_deviation + 0.005
    raw_upper_margin = math.exp(padded_log_half_width) - 1.0
    upper_margin = max(0.02, math.ceil(raw_upper_margin * 1000.0) / 1000.0)
    return {
        "maximum_absolute_log_ratio": maximum_log_deviation,
        "padded_log_half_width": padded_log_half_width,
        "upper_margin": upper_margin,
        "upper_speedup_bound": 1.0 + upper_margin,
        "lower_speedup_bound": 1.0 / (1.0 + upper_margin),
    }


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def calibrate(*, submission: Path, public_train: Path, space_path: Path,
              prebuilt_destination: Path, grader=None, runner=None, broker=None,
              calibration_nonce: bytes | None = None,
              toolchain_identity: dict[str, Any] | None = None) -> dict[str, Any]:
    submission, public_train, space_path, prebuilt_destination = (
        submission.resolve(), public_train.resolve(), space_path.resolve(),
        prebuilt_destination.resolve())
    if space_path != DEFAULT_SPACE.resolve():
        raise ValueError("noise calibration requires the exact frozen optimization_space_v1.yaml path")
    grader_path = HERE / "grader.py"
    runner_path = HERE / "beam_search.py"
    harness_path = HERE / "trusted_harness.c"
    monitor_path = HERE / "k1_monitor.py"
    grader = grader or _load("merlin_host_noise_grader", grader_path)
    runner = runner or _load("merlin_host_noise_runner", runner_path)
    broker = broker or _load("merlin_host_noise_broker", HERE / "trusted_search_broker.py")
    costs = _load("merlin_host_cost_calibration_helpers", HERE / "calibrate_search_costs.py")
    space = yaml.safe_load(space_path.read_text(encoding="utf-8"))
    rows = costs._jsonl(public_train)
    nonce = calibration_nonce if calibration_nonce is not None else secrets.token_bytes(32)
    if not isinstance(nonce, bytes) or len(nonce) != 32:
        raise ValueError("noise calibration nonce must be exactly 32 bytes")
    before = (_sha256(public_train), _sha256(space_path), _tree_sha256(submission))
    prebuild_receipt = grader.prepare_prebuilt_search_package(
        submission=submission, destination=prebuilt_destination,
        build_override=list(space["search_package"]["private_build_override"]))
    sample, private_authority = costs._private_panel(
        rows=rows, runner=runner, broker=broker, nonce=nonce,
        per_family=int(space["confirmation_samples_per_family"]),
        families=list(space["confirmation_families"]), phase="confirm")
    common = costs._common(
        submission=prebuilt_destination, prebuild_input=submission,
        prebuild_receipt=prebuild_receipt, public_train=public_train, public_rows=rows,
        space_path=space_path, private_shape_calibration=private_authority,
        toolchain_identity=(toolchain_identity if toolchain_identity is not None else
                            costs._toolchain(grader=grader, kind="k1-program",
                                             prebuild_receipt=prebuild_receipt)))
    # ``noise_margin`` is this calibration's output.  Binding the full space digest here would
    # invalidate the authority when that output is written back, so bind only the exact path and
    # the canonical pre-result protocol projection below.
    common["source_sha256"] = {
        name: digest for name, digest in common["source_sha256"].items()
        if name != "search_space"}
    common.pop("space_sha256", None)
    baseline = runner._candidate([])
    observations = grader.evaluate_public_policy_k1(
        submission=prebuilt_destination, capsules=sample, parent=baseline, candidate=baseline,
        repeats=int(space["measurement_repeats"]), public_rows=rows,
        board_environment=dict(space["board_environment"]))
    if ((_sha256(public_train), _sha256(space_path), _tree_sha256(submission)) != before or
            _tree_sha256(prebuilt_destination) != common["submission_tree_sha256"]):
        raise RuntimeError("noise calibration mutated a sealed input or prebuilt package")
    pair_rows: list[dict[str, Any]] = []
    for observation in observations:
        for pair_index, values in enumerate(zip(
                observation["baseline_elapsed_ns"], observation["baseline_calls"],
                observation["candidate_elapsed_ns"], observation["candidate_calls"],
                strict=True)):
            base_elapsed, base_calls, candidate_elapsed, candidate_calls = values
            ratio = (base_elapsed / base_calls) / (candidate_elapsed / candidate_calls)
            pair_rows.append({
                "capsule_id": observation["capsule_id"], "family": observation["family"],
                "pair_index": pair_index, "speedup_ratio": ratio,
                "absolute_unit_deviation": abs(ratio - 1.0),
            })
    margin = _derive_multiplicative_margin(
        [float(row["speedup_ratio"]) for row in pair_rows])
    checks = {
        "six_families": {row["family"] for row in observations} == set(
            space["confirmation_families"]),
        "six_valid_pairs_per_family": all(
            len(row["board_condition_pairs"]) == int(space["measurement_repeats"])
            for row in observations),
        "all_correct": all(row["correctness_ok"] for row in observations),
        "identical_k1_text": all(
            row["baseline_code_sha256"] == row["candidate_code_sha256"]
            for row in observations),
        "no_heldout_argument": True,
    }
    calibration_protocol = _calibration_protocol(space)
    calibration_protocol_sha256 = _canonical_sha256(calibration_protocol)
    calibration_lineage = {
        "version": 1,
        "stage": "noise_pre_result",
        "pre_result_protocol_sha256": calibration_protocol_sha256,
        "raw_input_tree_sha256": common["prebuild_input_tree_sha256"],
        "raw_input_package_sha256": common["prebuild_input_package_sha256"],
        "output_field": "noise_margin",
    }
    return {
        "version": 2, "kind": "cpu_host_k1_order_balanced_aa_noise_calibration",
        "status": "pass" if all(checks.values()) else "fail", "checks": checks,
        **common,
        "version": 2,
        "public_train_sha256": _sha256(public_train),
        "calibration_protocol": calibration_protocol,
        "calibration_protocol_sha256": calibration_protocol_sha256,
        "calibration_lineage": calibration_lineage,
        "grader_sha256": _sha256(grader_path),
        "runner_sha256": _sha256(runner_path),
        "trusted_harness_sha256": _sha256(harness_path),
        "k1_monitor_sha256": _sha256(monitor_path),
        "calibrator_sha256": _sha256(Path(__file__)),
        "derivation": DERIVATION,
        "maximum_absolute_pair_deviation": max(
            row["absolute_unit_deviation"] for row in pair_rows),
        "maximum_absolute_log_ratio": margin["maximum_absolute_log_ratio"],
        "padded_log_half_width": margin["padded_log_half_width"],
        "derived_noise_margin": margin["upper_margin"],
        "upper_speedup_bound": margin["upper_speedup_bound"],
        "lower_speedup_bound": margin["lower_speedup_bound"],
        "pairs": pair_rows, "observations": observations,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--public-train", type=Path, required=True)
    parser.add_argument("--space", type=Path, default=DEFAULT_SPACE)
    args = parser.parse_args(argv)
    handle = start_run(
        suite="cpu-host-compiler", method="k1-aa-noise-calibration", target="k1_cpu",
        extra={"paid_work": False, "heldout_opened": False})
    result: dict[str, Any] | None = None
    output = handle.run_dir / "metrics" / "k1_aa_noise_calibration.json"
    retained_input = handle.run_dir / "artifacts_dir" / "prebuild_input_submission"
    prebuilt = handle.run_dir / "artifacts_dir" / "prebuilt_search_package"
    try:
        try:
            cost_helpers = _load(
                "merlin_host_noise_retention_helpers", HERE / "calibrate_search_costs.py")
            retained_submission = cost_helpers._retain_input_submission(
                args.submission, retained_input)
            result = calibrate(
                submission=retained_submission, public_train=args.public_train.resolve(),
                space_path=args.space.resolve(), prebuilt_destination=prebuilt)
        except Exception as error:
            result = {
                "version": 2,
                "kind": "cpu_host_k1_order_balanced_aa_noise_calibration",
                "status": "fail",
                "paid_work": False,
                "heldout_opened": False,
                "error_class": type(error).__name__,
                "error": str(error),
                "error_evidence": getattr(error, "evidence", None),
                "submission": str(args.submission.resolve()),
                "public_train": str(args.public_train.resolve()),
                "space": str(args.space.resolve()),
                "calibrator_sha256": _sha256(Path(__file__)),
                "grader_sha256": _sha256(HERE / "grader.py"),
                "runner_sha256": _sha256(HERE / "beam_search.py"),
                "trusted_harness_sha256": _sha256(HERE / "trusted_harness.c"),
                "k1_monitor_sha256": _sha256(HERE / "k1_monitor.py"),
            }
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(output)
        return 0 if result["status"] == "pass" else 2
    finally:
        finish_run(handle, status=("ok" if result and result["status"] == "pass" else "error"),
                   summary={
                       "ready": bool(result and result["status"] == "pass"),
                       "prebuild_input_submission": str(retained_input.resolve()),
                       "prebuilt_search_package": str(prebuilt.resolve()),
                   })


if __name__ == "__main__":
    raise SystemExit(main())
