#!/usr/bin/env python3
"""Produce public-only timing authorities for the frozen CPU-host search budget.

This controller never accepts a heldout path.  It selects deterministic rows from the exact public
train JSONL, invokes the same trusted grader entry points as search, records monotonic timing for every
measured operation, and writes one fail-closed artifact inside an AET timestamped run directory.  Every
cost authority requires the already-issued A/A authority as a content-addressed predecessor.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import secrets
import shutil
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable

import yaml

from merlin.common.artifacts import finish_run, start_run
from merlin.common.paths import repo_root


HERE = Path(__file__).resolve().parent
DEFAULT_PUBLIC_TRAIN = (
    repo_root() / "out/artifacts/rvv-development-corpus/k1_cpu/v2/latest/public/train.jsonl")
DEFAULT_SPACE = HERE / "optimization_space_v1.yaml"
KINDS = {
    "k1-program": ("cpu_host_trusted_search_k1_program_calibration",
                    "k1_program_calibration.json"),
    "spike-screen": ("cpu_host_trusted_search_spike_candidate_calibration",
                     "spike_screen_calibration.json"),
    "confirmation-overhead": ("cpu_host_confirmation_overhead_calibration",
                              "confirmation_overhead_calibration.json"),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tree_sha256(root: Path) -> str:
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


def _package_tree_sha256(root: Path) -> str:
    """Mode-sensitive package identity matching the source-sealed grader receipt."""
    rows: list[tuple[str, str, int, str | None]] = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        stat = path.lstat()
        if path.is_symlink() or not (path.is_file() or path.is_dir()):
            raise ValueError(f"calibration package contains a non-regular entry: {relative}")
        rows.append((relative.as_posix(), "dir" if path.is_dir() else "file",
                     stat.st_mode & 0o777,
                     None if path.is_dir() else _sha256(path)))
    return hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest()


def _retain_input_submission(source: Path, destination: Path) -> Path:
    """Retain the unbuilt compiler package inside the timestamped authority run."""
    source = source.resolve()
    destination = destination.resolve()
    if not source.is_dir() or source.is_symlink():
        raise ValueError("calibration submission must be a real compiler package")
    if destination.exists():
        raise ValueError("retained calibration input destination already exists")
    links = [path.relative_to(source).as_posix()
             for path in source.rglob("*") if path.is_symlink()]
    if links:
        raise ValueError(f"calibration submission symlinks are forbidden: {links[:8]}")
    shutil.copytree(source, destination, symlinks=False)
    if _tree_sha256(source) != _tree_sha256(destination):
        raise RuntimeError("retained calibration input differs from submitted package")
    return destination


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load trusted module {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict) or value.get("split") != "train":
            raise ValueError(f"{path}:{line_number}: calibration accepts public train rows only")
        rows.append(value)
    if not rows:
        raise ValueError("public train split is empty")
    return rows


def _source_sha256() -> dict[str, str]:
    paths = {
        "cost_calibrator": Path(__file__).resolve(),
        "noise_calibrator": HERE / "calibrate_search_noise.py",
        "grader": HERE / "grader.py",
        "search_runner": HERE / "beam_search.py",
        "trusted_harness": HERE / "trusted_harness.c",
        "k1_monitor": HERE / "k1_monitor.py",
        "search_space": HERE / "optimization_space_v1.yaml",
        "trusted_evaluator": HERE / "trusted_evaluator.py",
        "trusted_broker": HERE / "trusted_search_broker.py",
        "k1_adapter": repo_root() / "merlin/python/merlin/mining/k1.py",
    }
    return {name: _sha256(path) for name, path in paths.items()}


def _safe(value: Any) -> Any:
    return json.loads(json.dumps(value, default=str))


def _summary(samples: list[float]) -> dict[str, Any]:
    if not samples or any(not math.isfinite(value) or value <= 0 for value in samples):
        raise ValueError("timing samples must be finite and positive")
    ordered = sorted(samples)
    return {
        "count": len(samples),
        "mean_seconds": statistics.mean(samples),
        "median_seconds": statistics.median(samples),
        "p95_seconds": ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)],
        "max_seconds": max(samples),
    }


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()


def _cost_calibration_lineage(*, noise_authority_path: Path, space_path: Path,
                              raw_input_tree_sha256: str,
                              raw_input_package_sha256: str) -> dict[str, Any]:
    """Bind a cost authority to the already-issued pre-result A/A authority.

    The predecessor's content hash is the ordering edge: a cost calibration cannot be issued by
    this producer until the A/A artifact exists.  The A/A protocol deliberately excludes its
    output field, while this later stage requires that output to be present in the final space.
    """
    noise_path = noise_authority_path.resolve()
    if noise_path.is_symlink() or not noise_path.is_file():
        raise ValueError("cost calibration requires a regular A/A noise authority artifact")
    try:
        noise = json.loads(noise_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("cost calibration A/A noise authority is unreadable") from exc
    space = yaml.safe_load(space_path.read_text(encoding="utf-8"))
    noise_lineage = noise.get("calibration_lineage") if isinstance(noise, dict) else None
    expected_noise_lineage = {
        "version": 1,
        "stage": "noise_pre_result",
        "pre_result_protocol_sha256": noise.get("calibration_protocol_sha256"),
        "raw_input_tree_sha256": noise.get("prebuild_input_tree_sha256"),
        "raw_input_package_sha256": noise.get("prebuild_input_package_sha256"),
        "output_field": "noise_margin",
    } if isinstance(noise, dict) else None
    if (not isinstance(noise, dict) or noise.get("version") != 2 or
            noise.get("kind") != "cpu_host_k1_order_balanced_aa_noise_calibration" or
            noise.get("status") != "pass" or noise_lineage != expected_noise_lineage):
        raise ValueError("cost calibration requires a passing pre-result A/A authority")
    if noise.get("prebuild_input_tree_sha256") != raw_input_tree_sha256:
        raise ValueError("cost and A/A calibrations must share one raw compiler input tree")
    if noise.get("prebuild_input_package_sha256") != raw_input_package_sha256:
        raise ValueError("cost and A/A calibrations must share one mode-sensitive raw package")
    if float(noise.get("derived_noise_margin", -1)) != float(space.get("noise_margin", -2)):
        raise ValueError("final optimization space does not contain the A/A-derived noise margin")
    protocol_sha256 = str(noise.get("calibration_protocol_sha256", ""))
    if len(protocol_sha256) != 64 or any(c not in "0123456789abcdef" for c in protocol_sha256):
        raise ValueError("A/A authority lacks a pre-result protocol digest")
    return {
        "version": 1,
        "stage": "cost_post_noise_result",
        "predecessor_stage": "noise_pre_result",
        "noise_authority": str(noise_path),
        "noise_authority_sha256": _sha256(noise_path),
        "pre_result_protocol_sha256": protocol_sha256,
        "derived_noise_margin": float(noise["derived_noise_margin"]),
        "raw_input_tree_sha256": raw_input_tree_sha256,
        "raw_input_package_sha256": raw_input_package_sha256,
        "final_space_sha256": _sha256(space_path),
    }


def _common(*, submission: Path, prebuild_input: Path, prebuild_receipt: dict[str, Any],
            public_train: Path, public_rows: list[dict[str, Any]], space_path: Path,
            private_shape_calibration: dict[str, Any],
            toolchain_identity: dict[str, Any]) -> dict[str, Any]:
    return {
        "version": 1,
        "paid_work": False,
        "heldout_opened": False,
        "protocol_state_mutated": False,
        "public_train": str(public_train.resolve()),
        "public_split_sha256": _sha256(public_train),
        "public_context": {
            "authority": "complete_public_train",
            "capsule_ids": [str(row["id"]) for row in public_rows],
            "row_count": len(public_rows),
            "rows_sha256": _canonical_sha256(public_rows),
        },
        "submission": str(submission.resolve()),
        "submission_manifest_sha256": _sha256(submission / "manifest.yaml"),
        "submission_tree_sha256": _tree_sha256(submission),
        "prebuild_input_submission": str(prebuild_input.resolve()),
        "prebuild_input_manifest_sha256": _sha256(prebuild_input / "manifest.yaml"),
        "prebuild_input_tree_sha256": _tree_sha256(prebuild_input),
        "prebuild_input_package_sha256": _package_tree_sha256(prebuild_input),
        "prebuild_receipt": _safe(prebuild_receipt),
        "prebuild_receipt_sha256": _canonical_sha256(prebuild_receipt),
        "space": str(space_path.resolve()),
        "space_sha256": _sha256(space_path),
        "private_shape_calibration": private_shape_calibration,
        "toolchain_identity": toolchain_identity,
        "source_sha256": _source_sha256(),
    }


def _baseline(runner) -> dict[str, Any]:
    return runner._candidate([])


def _tool_identity(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise ValueError(f"timing-relevant executable is absent: {resolved}")
    return {"path": str(resolved), "sha256": _sha256(resolved),
            "mode": resolved.stat().st_mode & 0o777}


def _resolve_tool(command: str) -> Path:
    if Path(command).is_absolute():
        return Path(command)
    resolved = shutil.which(command)
    if not resolved:
        raise ValueError(f"timing-relevant executable is absent: {command}")
    return Path(resolved)


def _toolchain(*, grader, kind: str, prebuild_receipt: dict[str, Any]) -> dict[str, Any]:
    tools = {"python": Path(sys.executable), "bwrap": _resolve_tool("bwrap")}
    for index, command in enumerate(prebuild_receipt["real_build_commands"]):
        tools[f"prebuild_command_{index}"] = _resolve_tool(str(command[0]))
    for index, command in enumerate(prebuild_receipt["private_build_override"]):
        tools[f"private_build_override_{index}"] = _resolve_tool(str(command))
    spike = grader._spike_tools()
    tools.update({f"spike_{name}": Path(path) for name, path in spike.items()})
    if kind == "k1-program":
        cc = grader._k1_cc()
        if cc is None:
            raise ValueError("K1 calibration requires the configured K1 compiler")
        tools.update({"k1_clang": cc, "k1_objcopy": cc.with_name("llvm-objcopy"),
                      "ssh": _resolve_tool("ssh"), "scp": _resolve_tool("scp")})
    return {name: _tool_identity(path) for name, path in sorted(tools.items())}


def _private_panel(*, rows: list[dict[str, Any]], runner, broker, nonce: bytes,
                   per_family: int, families: list[str], phase: str
                   ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    public = runner.select_semantic_sample(
        rows, per_family=per_family, families=families)
    private = [broker._private_capsule(
        row, secret=nonce, phase=phase, split="train") for row in public]
    records = [{"public": _safe(source), "private": _safe(measured)}
               for source, measured in zip(public, private, strict=True)]
    return private, {
        "version": 1,
        "authority": "trusted_broker_private_capsule_independent_calibration_nonce",
        "phase": phase,
        "split": "train",
        "nonce_hex": nonce.hex(),
        "nonce_sha256": hashlib.sha256(nonce).hexdigest(),
        "records": records,
        "records_sha256": _canonical_sha256(records),
    }


def _produce_k1(*, submission: Path, capsule: dict[str, Any],
                 public_rows: list[dict[str, Any]], common: dict[str, Any],
                 space: dict[str, Any], grader, runner,
                 monotonic_ns: Callable[[], int]) -> dict[str, Any]:
    # The authority is a per-program orchestration cost, not a family performance estimate.  One
    # deterministically selected public capsule keeps compiler/input work fixed while the trusted
    # parent/candidate sequence supplies the predeclared valid pairs plus any board-condition
    # replacement attempts retained by the evaluator.
    candidate = _baseline(runner)
    programs: list[dict[str, Any]] = []
    original = grader._grade_k1

    def timed_grade(*args, **kwargs):
        started = monotonic_ns()
        result = original(*args, **kwargs)
        ended = monotonic_ns()
        programs.append({
            "index": len(programs), "start_monotonic_ns": started,
            "end_monotonic_ns": ended, "total_seconds": (ended - started) / 1e9,
            "capsule_id": result.get("capsule"), "family": result.get("family"),
            "status": result.get("status"), "checks": _safe(result.get("checks", {})),
            "kernel_text_sha256": result.get("kernel_text_sha256"),
            "seed": result.get("seed"), "metrics": _safe(result.get("metrics", {})),
            "monitor": _safe(result.get("monitor", {})),
            "evidence": _safe(result),
        })
        return result

    grader._grade_k1 = timed_grade
    try:
        observations = grader.evaluate_public_policy_k1(
            submission=submission, capsules=[capsule], parent=candidate, candidate=candidate,
            repeats=int(space["measurement_repeats"]), public_rows=public_rows,
            board_environment=dict(space["board_environment"]))
    finally:
        grader._grade_k1 = original
    totals = [float(row["total_seconds"]) for row in programs]
    measured = _summary(totals)
    budget = space["budget"]
    # A board-condition rejection consumes the parent+candidate programs before the trusted
    # evaluator can discard that pair and replace it.  The retained observation is the authority
    # for the exact invocation count; requiring ``2 * repeats`` would incorrectly reject an
    # otherwise valid calibration whenever the predeclared replacement mechanism is exercised.
    observation = observations[0] if isinstance(observations, list) and len(observations) == 1 \
        and isinstance(observations[0], dict) else {}
    valid_pairs = observation.get("board_condition_pairs")
    excluded_pairs = observation.get("excluded_board_condition_pairs")
    repeats = int(space["measurement_repeats"])
    maximum_replacements = int(
        space["board_environment"]["maximum_invalid_pair_replacements_per_capsule"])
    expected_programs = observation.get("k1_program_count")
    program_count_bound = (
        isinstance(valid_pairs, list) and len(valid_pairs) == repeats and
        all(isinstance(pair, dict) and pair.get("valid") is True for pair in valid_pairs) and
        isinstance(excluded_pairs, list) and
        len(excluded_pairs) <= maximum_replacements and
        all(isinstance(pair, dict) and pair.get("valid") is False
            for pair in excluded_pairs) and
        type(expected_programs) is int and
        expected_programs == 2 * (len(valid_pairs) + len(excluded_pairs))
    )
    checks = {
        "all_passed": (program_count_bound and len(programs) == expected_programs and all(
            row["status"] == "pass" and row["checks"] and all(row["checks"].values())
            for row in programs)),
        "mean_within_expected": measured["mean_seconds"] <= float(
            budget["expected_seconds_per_k1_program"]),
        "max_within_planning_upper": measured["max_seconds"] <= float(
            budget["planning_upper_seconds_per_k1_program"]),
    }
    return {
        **common,
        "kind": KINDS["k1-program"][0], "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "declared": {
            "expected_seconds_per_program": float(budget["expected_seconds_per_k1_program"]),
            "planning_upper_seconds_per_program": float(
                budget["planning_upper_seconds_per_k1_program"]),
        },
        "calibration_capsule": {
            key: capsule.get(key) for key in ("id", "sha256", "family", "split")},
        "programs": programs, "statistics": measured,
        "trusted_evaluation_observations": _safe(observations),
    }


def _produce_spike(*, submission: Path, panel: list[dict[str, Any]],
                   public_rows: list[dict[str, Any]], common: dict[str, Any],
                   space: dict[str, Any], grader, runner,
                   monotonic_ns: Callable[[], int]) -> dict[str, Any]:
    candidate = _baseline(runner)
    started = monotonic_ns()
    observations = grader.evaluate_public_policy_spike(
        submission=submission, capsules=panel, parent=candidate, candidate=candidate,
        public_rows=public_rows)
    ended = monotonic_ns()
    elapsed = (ended - started) / 1e9
    budget = space["budget"]
    maximum = int(budget["maximum_screen_candidate_evaluations"])
    projected = elapsed * maximum
    checks = {
        "all_observations_passed": (
            len(panel) == 12 and len(observations) == 12 and
            all(row.get("correctness_ok") is True for row in observations)),
        "projection_within_expected_budget": projected <= float(
            budget["expected_spike_screen_seconds"]),
    }
    return {
        **common,
        "kind": KINDS["spike-screen"][0],
        "status": "pass" if all(checks.values()) else "fail", "checks": checks,
        "declared": {
            "expected_spike_screen_seconds": float(budget["expected_spike_screen_seconds"]),
            "maximum_candidate_evaluations": maximum,
        },
        "capsules": len(panel), "completed_observations": len(observations),
        "maximum_candidate_evaluations": maximum,
        "start_monotonic_ns": started, "end_monotonic_ns": ended,
        "candidate_evaluation_seconds": elapsed,
        "projected_max_screen_seconds": projected,
        "observations": _safe(observations),
    }


def _produce_confirmation(*, submission: Path, panel: list[dict[str, Any]],
                          public_rows: list[dict[str, Any]], common: dict[str, Any],
                          space: dict[str, Any],
                          grader, runner, monotonic_ns: Callable[[], int]) -> dict[str, Any]:
    candidate = _baseline(runner)
    stage_rows: dict[str, list[dict[str, Any]]] = {
        "package_build": [], "compiler_invocation": [], "spike_check": []}
    originals = {
        "package_build": grader._build,
        "compiler_invocation": grader._compile_one,
        "spike_check": grader._grade_spike,
    }
    active_capsule: dict[str, Any] | None = None

    def wrapper(name: str):
        original = originals[name]

        def timed(*args, **kwargs):
            started = monotonic_ns()
            result = original(*args, **kwargs)
            ended = monotonic_ns()
            if name == "package_build":
                passed = True
                evidence = {"manifest": _safe(result[0]), "build_logs": _safe(result[1])}
            elif name == "compiler_invocation":
                passed = result.get("ok") is True
                evidence = _safe(result)
            else:
                passed = grader._search_spike_correct(result)
                evidence = _safe(result)
            if active_capsule is None:
                raise RuntimeError("confirmation timing occurred without an active capsule")
            capsule_stage_index = sum(
                row["capsule_id"] == active_capsule["id"] for row in stage_rows[name])
            per_side = 2 if (name == "compiler_invocation" and
                             active_capsule["family"] == "runtime_parallel") else 1
            if capsule_stage_index >= per_side * 2:
                raise RuntimeError("confirmation stage exceeded broker-equivalent multiplicity")
            side = ("parent", "candidate")[capsule_stage_index // per_side]
            mode = str(args[3]) if name == "compiler_invocation" and len(args) > 3 else None
            stage_rows[name].append({
                "index": len(stage_rows[name]), "start_monotonic_ns": started,
                "end_monotonic_ns": ended, "wall_seconds": (ended - started) / 1e9,
                "stage": name, "capsule_id": active_capsule["id"],
                "family": active_capsule["family"],
                "side": side, "mode": mode,
                "status": "pass" if passed else "fail", "evidence": evidence,
            })
            return result

        return timed

    grader._build = wrapper("package_build")
    grader._compile_one = wrapper("compiler_invocation")
    grader._grade_spike = wrapper("spike_check")
    evaluations = []
    try:
        # One trusted evaluation per capsule gives two observations (parent and A/A candidate) for
        # each package/build/Spike stage: six families x two = twelve retained samples per stage.
        for capsule in panel:
            active_capsule = capsule
            evaluations.extend(grader.evaluate_public_policy_confirmation_stages(
                submission=submission, capsules=[capsule], parent=candidate,
                candidate=candidate, public_rows=public_rows))
    finally:
        for name, original in originals.items():
            attribute = {"package_build": "_build", "compiler_invocation": "_compile_one",
                         "spike_check": "_grade_spike"}[name]
            setattr(grader, attribute, original)
    budget = space["budget"]
    declared = {
        "package_build": {
            "expected_seconds": float(
                budget["expected_seconds_per_confirmation_package_build"]),
            "planning_upper_seconds": float(
                budget["planning_upper_seconds_per_confirmation_package_build"]),
        },
        "compiler_invocation": {
            "expected_seconds": float(
                budget["expected_seconds_per_confirmation_compiler_invocation"]),
            "planning_upper_seconds": float(
                budget["planning_upper_seconds_per_confirmation_compiler_invocation"]),
        },
        "spike_check": {
            "expected_seconds": float(budget["expected_seconds_per_confirmation_spike_check"]),
            "planning_upper_seconds": float(
                budget["planning_upper_seconds_per_confirmation_spike_check"]),
        },
    }
    summaries = {name: _summary([float(row["wall_seconds"]) for row in values])
                 for name, values in stage_rows.items()}
    expected_stage_counts = {
        "package_build": len(panel) * 2,
        "compiler_invocation": (
            len(panel) + sum(row["family"] == "runtime_parallel" for row in panel)) * 2,
        "spike_check": len(panel) * 2,
    }
    all_pass = (len(panel) == 6 and len(evaluations) == 6 and all(
        len(values) == expected_stage_counts[name] and
        all(row["status"] == "pass" for row in values)
        for name, values in stage_rows.items()))
    mean_ok = all(summaries[name]["mean_seconds"] <= limits["expected_seconds"]
                  for name, limits in declared.items())
    max_ok = all(summaries[name]["max_seconds"] <= limits["planning_upper_seconds"]
                 for name, limits in declared.items())
    checks = {
        "all_toolchain_stages_passed": all_pass,
        "all_expected_costs_within_budget": mean_ok,
        "all_maximum_costs_within_planning_upper": max_ok,
    }
    return {
        **common,
        "kind": KINDS["confirmation-overhead"][0],
        "status": "pass" if all(checks.values()) else "fail", "checks": checks,
        "declared": declared, "calibration_repeats_per_capsule": 2,
        "public_capsules": [{key: row.get(key) for key in ("id", "sha256", "family")}
                            for row in panel],
        "spike_statuses": [row["status"] for row in stage_rows["spike_check"]],
        **summaries, "stage_observations": stage_rows,
        "trusted_evaluation_observations": _safe(evaluations),
    }


def calibrate(*, kind: str, submission: Path, public_train: Path, space_path: Path,
              prebuilt_destination: Path, noise_authority: Path,
              grader=None, runner=None, broker=None,
              calibration_nonce: bytes | None = None,
              toolchain_identity: dict[str, Any] | None = None,
              monotonic_ns: Callable[[], int] = time.monotonic_ns) -> dict[str, Any]:
    submission, public_train, space_path, prebuilt_destination = (
        submission.resolve(), public_train.resolve(), space_path.resolve(),
        prebuilt_destination.resolve())
    if kind not in KINDS:
        raise ValueError(f"unknown calibration kind {kind!r}")
    if space_path != DEFAULT_SPACE.resolve():
        raise ValueError("calibration requires the exact frozen optimization_space_v1.yaml path")
    if not submission.is_dir() or submission.is_symlink() or not (
            submission / "manifest.yaml").is_file():
        raise ValueError("calibration submission must be a real compiler package")
    grader = grader or _load("merlin_host_cost_grader", HERE / "grader.py")
    runner = runner or _load("merlin_host_cost_runner", HERE / "beam_search.py")
    broker = broker or _load("merlin_host_cost_broker", HERE / "trusted_search_broker.py")
    rows = _jsonl(public_train)
    space = yaml.safe_load(space_path.read_text(encoding="utf-8"))
    if not isinstance(space, dict):
        raise ValueError("optimization space is not a mapping")
    nonce = calibration_nonce if calibration_nonce is not None else secrets.token_bytes(32)
    if not isinstance(nonce, bytes) or len(nonce) != 32:
        raise ValueError("calibration nonce must be exactly 32 bytes")
    before = (_sha256(public_train), _sha256(space_path), _tree_sha256(submission))
    raw_input_package_sha256 = _package_tree_sha256(submission)
    calibration_lineage = _cost_calibration_lineage(
        noise_authority_path=noise_authority, space_path=space_path,
        raw_input_tree_sha256=before[2],
        raw_input_package_sha256=raw_input_package_sha256)
    prebuild_receipt = grader.prepare_prebuilt_search_package(
        submission=submission, destination=prebuilt_destination,
        build_override=list(space["search_package"]["private_build_override"]))
    if (not isinstance(prebuild_receipt, dict) or
            prebuild_receipt.get("authority") != "driver_private_prebuild" or
            prebuild_receipt.get("private_build_override") != ["/bin/true"] or
            prebuild_receipt.get("prebuild_tree_sha256") != raw_input_package_sha256):
        raise RuntimeError("trusted grader emitted an invalid private-prebuild receipt")
    prebuilt_tree = _tree_sha256(prebuilt_destination)
    phase = "screen" if kind == "spike-screen" else "confirm"
    per_family = int(space[
        "screen_samples_per_family" if phase == "screen"
        else "confirmation_samples_per_family"])
    panel, private_shape_calibration = _private_panel(
        rows=rows, runner=runner, broker=broker, nonce=nonce, per_family=per_family,
        families=list(space["confirmation_families"]), phase=phase)
    common = _common(
        submission=prebuilt_destination, prebuild_input=submission,
        prebuild_receipt=prebuild_receipt, public_train=public_train, public_rows=rows,
        space_path=space_path, private_shape_calibration=private_shape_calibration,
        toolchain_identity=(toolchain_identity if toolchain_identity is not None else
                            _toolchain(grader=grader, kind=kind,
                                       prebuild_receipt=prebuild_receipt)))
    common["calibration_lineage"] = calibration_lineage
    producer = {"k1-program": _produce_k1, "spike-screen": _produce_spike,
                "confirmation-overhead": _produce_confirmation}[kind]
    selected = {"capsule": panel[0]} if kind == "k1-program" else {"panel": panel}
    result = producer(
        submission=prebuilt_destination, public_rows=rows, common=common, space=space,
        grader=grader, runner=runner, monotonic_ns=monotonic_ns, **selected)
    after = (_sha256(public_train), _sha256(space_path), _tree_sha256(submission))
    if after != before:
        raise RuntimeError("public split, optimization space, or compiler package mutated during calibration")
    if _tree_sha256(prebuilt_destination) != prebuilt_tree:
        raise RuntimeError("private prebuilt package mutated during calibration")
    return result


def _failure(kind: str, *, submission: Path, public_train: Path,
             error: Exception) -> dict[str, Any]:
    return {
        "version": 1, "kind": KINDS[kind][0], "status": "fail",
        "paid_work": False, "heldout_opened": False, "protocol_state_mutated": False,
        "submission": str(submission.resolve()), "public_train": str(public_train.resolve()),
        "source_sha256": _source_sha256(), "error_class": type(error).__name__,
        "error": str(error),
    }


def run_calibration(*, kind: str, submission: Path, public_train: Path, space_path: Path,
                    prebuilt_destination: Path, noise_authority: Path,
                    grader=None, runner=None, broker=None,
                    calibration_nonce: bytes | None = None,
                    toolchain_identity: dict[str, Any] | None = None,
                    monotonic_ns: Callable[[], int] = time.monotonic_ns) -> dict[str, Any]:
    """Run one authority producer and always return a serializable pass/fail artifact."""
    try:
        return calibrate(
            kind=kind, submission=submission, public_train=public_train,
            space_path=space_path, prebuilt_destination=prebuilt_destination,
            noise_authority=noise_authority,
            grader=grader, runner=runner, broker=broker,
            calibration_nonce=calibration_nonce,
            toolchain_identity=toolchain_identity,
            monotonic_ns=monotonic_ns)
    except Exception as error:
        return _failure(kind, submission=submission, public_train=public_train, error=error)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=tuple(KINDS), required=True)
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--public-train", type=Path, default=DEFAULT_PUBLIC_TRAIN)
    parser.add_argument("--space", type=Path, default=DEFAULT_SPACE)
    parser.add_argument("--noise-authority", type=Path, required=True)
    args = parser.parse_args(argv)
    handle = start_run(
        suite="cpu-host-compiler", method=f"{args.kind}-calibration", target="k1_cpu", seed=0,
        extra={"paid_work": False, "heldout_opened": False, "kind": args.kind,
               "submission": str(args.submission.resolve()),
               "public_train": str(args.public_train.resolve()),
               "space": str(args.space.resolve())})
    output = handle.run_dir / "metrics" / KINDS[args.kind][1]
    retained_input = handle.run_dir / "artifacts_dir" / "prebuild_input_submission"
    prebuilt = handle.run_dir / "artifacts_dir" / "prebuilt_search_package"
    finish_status = "error"
    summary: dict[str, Any] = {
        "calibration_kind": args.kind, "calibration_status": "interrupted_or_failed",
        "paid_work": False, "heldout_opened": False,
    }
    try:
        retained_submission = _retain_input_submission(args.submission, retained_input)
        result = run_calibration(
            kind=args.kind, submission=retained_submission, public_train=args.public_train,
            space_path=args.space, prebuilt_destination=prebuilt,
            noise_authority=args.noise_authority)
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        passed = result.get("status") == "pass"
        finish_status = "ok" if passed else "error"
        summary = {
            "calibration_kind": args.kind, "calibration_status": result.get("status"),
            "artifact": str(output.resolve()), "artifact_sha256": _sha256(output),
            "prebuild_input_submission": str(retained_input.resolve()),
            "prebuilt_search_package": str(prebuilt.resolve()),
            "paid_work": False, "heldout_opened": False}
        print(json.dumps({"status": result.get("status"), "artifact": str(output.resolve()),
                          "sha256": _sha256(output)}, indent=2))
        return 0 if passed else 1
    finally:
        # KeyboardInterrupt/SystemExit remain control-flow signals and never become authority
        # artifacts, but their timestamped AET run is still closed as an error.
        finish_run(handle, status=finish_status, summary=summary)


if __name__ == "__main__":
    raise SystemExit(main())
