#!/usr/bin/env python3
"""Qualify a real K1 multicore timing effect on one public runtime capsule.

This is a post-A/A, pre-campaign qualification.  It compares the exact same public capsule ABI in
``rvv`` (one hart) and ``rvv_multicore`` (the capsule's declared hart count), using the source-sealed
grader's compiler, Spike, K1 harness, monitor, board lock, and board-condition gates.  The controller
has deliberately no held-out input and writes every raw K1 receipt into its timestamped AET run.

"Performance effect" is intentionally bidirectional: a consistent slowdown proves that the real
multicore path affects execution but does not claim an improvement.  The artifact separately records
whether multicore is faster.  This distinction prevents a qualification gate from silently turning a
negative result into a speedup claim.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import re
import secrets
import shutil
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

import yaml

from merlin.common.artifacts import finish_run, start_run
from merlin.common.paths import repo_root


HERE = Path(__file__).resolve().parent
DEFAULT_PUBLIC_TRAIN = (
    repo_root() / "out/artifacts/rvv-development-corpus/k1_cpu/v2/latest/public/train.jsonl")
CANONICAL_PUBLIC_TRAIN_SHA256 = "d567f8a61d7c834c561b3663cd1d829ab8301d852fd7e4cf58311042811ea9b1"
DEFAULT_SPACE = HERE / "optimization_space_v1.yaml"
PAIR_ORDERS = (
    "rvv_rvv_multicore",
    "rvv_multicore_rvv",
    "rvv_multicore_rvv",
    "rvv_rvv_multicore",
    "rvv_rvv_multicore",
    "rvv_multicore_rvv",
)
REQUIRED_K1_CHECKS = frozenset({
    "exact_mode", "no_fallback", "numeric_correctness", "trusted_parent_receipt",
    "per_call_correctness", "csr_vlen", "exact_affinity", "exact_task_count",
    "active_harts", "audit_attribution", "wall_time", "peak_rss", "upload_integrity",
})
_K1_ARTIFACT_NAMES = ("kernel.o", "trusted_harness.o", "kernel.text.bin", "capsule_k1")
_METRIC_LINE = re.compile(r"K1_METRIC ([a-z_]+) ([0-9]+)")


class K1EvidenceError(ValueError):
    """A fail-closed K1 evidence rejection with machine-readable predicate results."""

    def __init__(self, message: str, assessment: dict[str, Any]):
        self.assessment = assessment
        failed = [name for name, passed in assessment["checks"].items() if passed is not True]
        self.failed_checks = tuple(failed)
        super().__init__(f"{message}; failed checks: {', '.join(failed)}")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()


def _digest(value: Any) -> bool:
    text = str(value or "")
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _tree_sha256(root: Path) -> str:
    rows: list[tuple[str, str, str]] = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            rows.append((relative, "symlink", path.readlink().as_posix()))
        elif path.is_file():
            rows.append((relative, "file", _sha256(path)))
    return _canonical_sha256(rows)


def _package_tree_sha256(root: Path) -> str:
    """Match the grader's mode-sensitive raw-package identity."""
    rows: list[tuple[str, str, int, str | None]] = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        stat = path.lstat()
        if path.is_symlink() or not (path.is_file() or path.is_dir()):
            raise ValueError(f"qualification package contains a non-regular entry: {relative}")
        rows.append((relative.as_posix(), "dir" if path.is_dir() else "file",
                     stat.st_mode & 0o777, None if path.is_dir() else _sha256(path)))
    return hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest()


def _retain_input_submission(source: Path, destination: Path) -> Path:
    source, destination = source.resolve(), destination.resolve()
    if not source.is_dir() or source.is_symlink():
        raise ValueError("multicore qualification requires a real compiler package")
    if destination.exists():
        raise ValueError("retained qualification input destination already exists")
    links = [path.relative_to(source).as_posix()
             for path in source.rglob("*") if path.is_symlink()]
    if links:
        raise ValueError(f"qualification submission symlinks are forbidden: {links[:8]}")
    shutil.copytree(source, destination, symlinks=False)
    if (_tree_sha256(source) != _tree_sha256(destination) or
            _package_tree_sha256(source) != _package_tree_sha256(destination)):
        raise RuntimeError("retained qualification input differs from submitted package")
    return destination


def _copy_aa_sealed_prebuilt(*, noise: dict[str, Any], aa_receipt: dict[str, Any],
                             destination: Path) -> Path:
    """Copy the exact A/A-sealed package; never substitute a fresh nondeterministic rebuild."""
    raw_source = Path(str(noise.get("submission", "")))
    source = raw_source.resolve()
    if raw_source.is_symlink() or not source.is_dir() or destination.exists():
        raise ValueError("A/A sealed prebuilt package is absent, symlinked, or has a used destination")
    expected_tree = aa_receipt.get("sealed_prebuilt_tree_sha256")
    if not _digest(expected_tree) or _package_tree_sha256(source) != expected_tree:
        raise ValueError("A/A sealed prebuilt package differs from its receipt")
    manifest_path = source / "manifest.yaml"
    if (not manifest_path.is_file() or _sha256(manifest_path) !=
            aa_receipt.get("private_manifest_sha256")):
        raise ValueError("A/A sealed prebuilt manifest differs from its receipt")
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    compiler = manifest.get("compiler") if isinstance(manifest, dict) else None
    command = compiler.get("command") if isinstance(compiler, dict) else None
    if not isinstance(command, list) or not command or any(not isinstance(item, str)
                                                           for item in command):
        raise ValueError("A/A sealed prebuilt compiler command is invalid")
    first = command[0]
    raw_entrypoint = command[1] if first in {
        "python3", "/usr/bin/python3", "/usr/bin/python3.12"} and len(command) > 1 else first
    relative = Path(raw_entrypoint)
    raw_entrypoint_path = source / relative
    entrypoint = raw_entrypoint_path.resolve()
    if (relative.is_absolute() or ".." in relative.parts or
            not entrypoint.is_relative_to(source) or raw_entrypoint_path.is_symlink() or
            not entrypoint.is_file()):
        raise ValueError("A/A sealed prebuilt compiler entrypoint is invalid")
    identity = [entrypoint.stat().st_mode & 0o777, _sha256(entrypoint)]
    if identity != aa_receipt.get("built_entrypoint_identity"):
        raise ValueError("A/A sealed prebuilt compiler entrypoint differs from its receipt")
    raw_policy = source / str(manifest.get("policy", ""))
    policy = raw_policy.resolve()
    if (not policy.is_relative_to(source) or not policy.is_file() or raw_policy.is_symlink() or
            _sha256(policy) != aa_receipt.get("policy_sha256")):
        raise ValueError("A/A sealed prebuilt policy differs from its receipt")
    shutil.copytree(source, destination, symlinks=False)
    if _package_tree_sha256(destination) != expected_tree:
        raise RuntimeError("qualification copy differs from the exact A/A sealed prebuilt package")
    return destination


def _require_canonical_public_train(path: Path) -> None:
    canonical = DEFAULT_PUBLIC_TRAIN.resolve()
    if path.resolve() != canonical:
        raise ValueError("qualification requires the exact canonical public train corpus path")
    if not canonical.is_file() or _sha256(canonical) != CANONICAL_PUBLIC_TRAIN_SHA256:
        raise ValueError("canonical public train corpus differs from its frozen digest")


def _executable_source_paths() -> dict[str, Path]:
    """Return every local source whose code participates in the qualification decision."""
    return {
        "qualifier": Path(__file__).resolve(),
        "grader": (HERE / "grader.py").resolve(),
        "runner": (HERE / "beam_search.py").resolve(),
        "trusted_harness": (HERE / "trusted_harness.c").resolve(),
        "k1_monitor": (HERE / "k1_monitor.py").resolve(),
        "noise_validator": (
            repo_root() / "merlin/python/merlin/compare/host_experiment.py").resolve(),
        "calibration_sources": (HERE / "calibrate_search_costs.py").resolve(),
    }


def _source_snapshot() -> dict[str, dict[str, Any]]:
    snapshot: dict[str, dict[str, Any]] = {}
    for name, path in sorted(_executable_source_paths().items()):
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"qualification executable source is absent or symlinked: {name}")
        snapshot[name] = {
            "path": str(path), "sha256": _sha256(path), "size_bytes": path.stat().st_size,
        }
    return snapshot


def _inventory_entry(root: Path, path: Path, **fields: Any) -> dict[str, Any]:
    root, path = root.resolve(), path.resolve()
    if path.is_symlink() or not path.is_file() or not path.is_relative_to(root):
        raise ValueError(f"raw evidence file escapes its receipt root: {path}")
    return {
        **fields,
        "relative_path": path.relative_to(root).as_posix(),
        "path": str(path),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
        "mode": path.stat().st_mode & 0o777,
    }


def _verify_inventory_entry(root: Path, entry: dict[str, Any]) -> Path:
    relative = Path(str(entry.get("relative_path", "")))
    if (relative.is_absolute() or not relative.parts or ".." in relative.parts or
            relative.as_posix() != str(entry.get("relative_path", ""))):
        raise ValueError("raw evidence inventory contains an unsafe relative path")
    path = root.resolve() / relative
    resolved = path.resolve()
    if (resolved != path or path.is_symlink() or not path.is_file() or
            not resolved.is_relative_to(root.resolve()) or _sha256(path) != entry.get("sha256") or
            path.stat().st_size != entry.get("size_bytes") or
            path.stat().st_mode & 0o777 != entry.get("mode")):
        raise ValueError(f"raw evidence inventory identity differs: {relative.as_posix()}")
    if entry.get("path") != str(resolved):
        raise ValueError(f"raw evidence absolute/relative paths disagree: {relative.as_posix()}")
    return path


def _retain_source_snapshot(root: Path, snapshot: dict[str, dict[str, Any]]) \
        -> list[dict[str, Any]]:
    inventory: list[dict[str, Any]] = []
    for name, identity in sorted(snapshot.items()):
        source = Path(str(identity["path"]))
        suffix = source.suffix or ".source"
        target = root / "executable_sources" / f"{name}{suffix}"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        entry = _inventory_entry(root, target, role="executable_source", source_role=name)
        if (entry["sha256"] != identity["sha256"] or
                entry["size_bytes"] != identity["size_bytes"]):
            raise RuntimeError(f"retained executable source differs: {name}")
        inventory.append(entry)
    return inventory


def _retain_authority_inputs(root: Path, *, public_train: Path, space: Path,
                             noise_authority: Path) -> list[dict[str, Any]]:
    inventory: list[dict[str, Any]] = []
    for role, source, name in (
            ("canonical_public_train", public_train, "public_train.jsonl"),
            ("frozen_optimization_space", space, "optimization_space_v1.yaml"),
            ("aa_noise_authority", noise_authority, "noise_authority.json")):
        if source.is_symlink() or not source.is_file():
            raise ValueError(f"qualification authority input is absent or symlinked: {role}")
        target = root / "authority_inputs" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        entry = _inventory_entry(root, target, role="authority_input", authority_role=role)
        if entry["sha256"] != _sha256(source):
            raise RuntimeError(f"retained qualification authority input differs: {role}")
        inventory.append(entry)
    return inventory


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load trusted module {path}")
    module = importlib.util.module_from_spec(spec)
    # ``dataclasses`` resolves postponed annotations through ``sys.modules``
    # while a module is executing.  Register the trusted module for the
    # duration of the load, just as Python's normal import machinery does.
    previous = sys.modules.get(name)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        if previous is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous
        raise
    return module


def _jsonl_public_train(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict) or value.get("split") != "train":
            raise ValueError(f"{path}:{line_number}: qualification accepts public train rows only")
        rows.append(value)
    if not rows:
        raise ValueError("public train split is empty")
    return rows


def select_public_runtime_capsule(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Select one public, genuine deterministic-static-partition capsule.

    Selection is independent of measured performance: the explicit ``static_partition`` operation,
    exactly one reuse, maximum harts, maximum output work-items, then lexical SHA.  Unit reuse is
    deliberate: it prevents a synthetic repeated-work loop from manufacturing a timing difference;
    both modes perform one semantic output computation per element.
    """
    candidates = [row for row in rows
                  if row.get("split") == "train" and
                  row.get("family") == "runtime_parallel" and
                  row.get("operation") == "static_partition" and
                  isinstance(row.get("state"), dict) and
                  row["state"].get("reuse_count") == 1 and
                  isinstance(row.get("core_count"), int) and
                  not isinstance(row.get("core_count"), bool) and
                  int(row["core_count"]) > 1]
    if not candidates:
        raise ValueError("public train has no genuine static_partition multicore capsule")

    def declared_axes(row: dict[str, Any]) -> tuple[int, int]:
        shape = row.get("shape") if isinstance(row.get("shape"), dict) else {}
        state = row.get("state") if isinstance(row.get("state"), dict) else {}
        work_items, reuse = shape.get("work_items"), state.get("reuse_count", 1)
        if (not isinstance(work_items, int) or isinstance(work_items, bool) or work_items <= 0 or
                not isinstance(reuse, int) or isinstance(reuse, bool) or reuse <= 0):
            raise ValueError("runtime_parallel capsule has invalid public work declaration")
        return work_items, reuse

    maximum_harts = max(int(row["core_count"]) for row in candidates)
    candidates = [row for row in candidates if int(row["core_count"]) == maximum_harts]
    maximum_work = max(declared_axes(row)[0] for row in candidates)
    candidates = [row for row in candidates if declared_axes(row)[0] == maximum_work]
    selected = min(candidates, key=lambda row: str(row.get("sha256", "")))
    if not _digest(selected.get("sha256")):
        raise ValueError("selected public runtime capsule lacks a content digest")
    return json.loads(json.dumps(selected))


def _default_semantic_noise_problems(*, noise: dict[str, Any], public_train: Path,
                                     space: dict[str, Any], rows: list[dict[str, Any]]) -> list[str]:
    """Invoke the campaign's full independent A/A receipt validator lazily."""
    host = _load("merlin_multicore_host_preflight", repo_root() /
                 "merlin/python/merlin/compare/host_experiment.py")
    costs = _load("merlin_multicore_cost_sources", HERE / "calibrate_search_costs.py")
    return host._validate_calibration_semantics(
        label="noise_calibration", value=noise, train_sha256=_sha256(public_train),
        source_sha256=costs._source_sha256(), space=space, train_rows=rows)


def _noise_lineage(*, noise_path: Path, public_train: Path, space_path: Path,
                   space: dict[str, Any], rows: list[dict[str, Any]],
                   raw_tree_sha256: str, raw_package_sha256: str,
                   semantic_validator: Callable[..., list[str]] | None) -> tuple[dict[str, Any],
                                                                                 dict[str, Any]]:
    noise_path = noise_path.resolve()
    if noise_path.is_symlink() or not noise_path.is_file():
        raise ValueError("multicore qualification requires a regular A/A authority artifact")
    try:
        noise = json.loads(noise_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("A/A authority is unreadable") from exc
    expected_checks = {
        "six_families": True, "six_valid_pairs_per_family": True,
        "all_correct": True, "identical_k1_text": True, "no_heldout_argument": True,
    }
    protocol = noise.get("calibration_protocol") if isinstance(noise, dict) else None
    lineage = noise.get("calibration_lineage") if isinstance(noise, dict) else None
    expected_lineage = {
        "version": 1,
        "stage": "noise_pre_result",
        "pre_result_protocol_sha256": noise.get("calibration_protocol_sha256"),
        "raw_input_tree_sha256": raw_tree_sha256,
        "raw_input_package_sha256": raw_package_sha256,
        "output_field": "noise_margin",
    } if isinstance(noise, dict) else None
    try:
        margin = float(noise.get("derived_noise_margin", -1))
        final_margin = float(space.get("noise_margin", -2))
    except (TypeError, ValueError):
        margin = final_margin = -1.0
    basic_ok = (
        isinstance(noise, dict) and noise.get("version") == 2 and
        noise.get("kind") == "cpu_host_k1_order_balanced_aa_noise_calibration" and
        noise.get("status") == "pass" and noise.get("checks") == expected_checks and
        noise.get("paid_work") is False and noise.get("heldout_opened") is False and
        noise.get("protocol_state_mutated") is False and
        noise.get("public_train_sha256") == _sha256(public_train) and
        isinstance(protocol, dict) and
        protocol.get("measurement_repeats") == len(PAIR_ORDERS) and
        protocol.get("board_environment") == space.get("board_environment") and
        _canonical_sha256(protocol) == noise.get("calibration_protocol_sha256") and
        lineage == expected_lineage and
        noise.get("prebuild_input_tree_sha256") == raw_tree_sha256 and
        noise.get("prebuild_input_package_sha256") == raw_package_sha256 and
        math.isfinite(margin) and margin > 0 and margin == final_margin)
    if not basic_ok:
        raise ValueError("A/A authority does not bind the public split, raw compiler, board gate, "
                         "and final derived margin")
    validator = semantic_validator or _default_semantic_noise_problems
    problems = validator(noise=noise, public_train=public_train, space=space, rows=rows)
    if problems:
        raise ValueError("A/A authority failed full receipt replay: " + "; ".join(problems))
    return noise, {
        "version": 1,
        "stage": "multicore_effect_post_noise_result",
        "predecessor_stage": "noise_pre_result",
        "noise_authority": str(noise_path),
        "noise_authority_sha256": _sha256(noise_path),
        "pre_result_protocol_sha256": noise["calibration_protocol_sha256"],
        "derived_noise_margin": margin,
        "raw_input_tree_sha256": raw_tree_sha256,
        "raw_input_package_sha256": raw_package_sha256,
        "final_space_sha256": _sha256(space_path),
    }


def _safe(value: Any) -> Any:
    return json.loads(json.dumps(value, default=str))


def _write_receipt(root: Path, relative: str, value: Any) -> dict[str, Any]:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise ValueError(f"raw receipt path already exists: {path}")
    path.write_text(json.dumps(_safe(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return _inventory_entry(root, path, role="json_receipt")


def _retain_compiled_artifacts(receipts: Path, compiled: dict[str, dict[str, Any]]) \
        -> list[dict[str, Any]]:
    inventory: list[dict[str, Any]] = []
    for mode, record in sorted(compiled.items()):
        kernel = Path(str(record.get("_kernel_path", "")))
        source_dir = kernel.parent
        sources = {
            "input.mlir": source_dir.parent / "input.mlir",
            "kernel.c": source_dir / "kernel.c",
            "lowered.mlir": source_dir / "lowered.mlir",
            "metadata.json": source_dir / "metadata.json",
        }
        expected_hashes = {
            "input.mlir": record.get("input_mlir_sha256"),
            "kernel.c": record.get("source_sha256"),
            "lowered.mlir": record.get("lowered_mlir_sha256"),
        }
        for name, source in sources.items():
            if not source.is_file():
                raise ValueError(f"compiled {mode} artifact disappeared before retention: {name}")
            expected = expected_hashes.get(name)
            if expected is not None and _sha256(source) != expected:
                raise ValueError(f"compiled {mode} artifact differs from its compile receipt: {name}")
            target = receipts / "compiled" / mode / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            inventory.append(_inventory_entry(
                receipts, target, role="compiled_artifact", mode_name=mode, name=name))
    return inventory


def _retain_k1_build_artifacts(*, receipts: Path, run_root: Path, row: dict[str, Any],
                               mode: str, attempt_id: int,
                               evidence: dict[str, Any]) -> list[dict[str, Any]]:
    work = run_root / f"{row['id']}_{mode}"
    inventory: list[dict[str, Any]] = []
    for name in _K1_ARTIFACT_NAMES:
        source = work / name
        if not source.is_file() or source.is_symlink():
            raise ValueError(f"K1 {mode} build omitted retained evidence {name}")
        target = receipts / "k1_builds" / f"attempt_{attempt_id:02d}" / mode / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        entry = _inventory_entry(
            receipts, target, role="k1_build_artifact", mode_name=mode,
            attempt_id=attempt_id, name=name)
        if name == "capsule_k1" and entry["sha256"] != evidence.get("local_sha256"):
            raise ValueError("retained K1 executable differs from the uploaded executable receipt")
        if name == "kernel.text.bin" and entry["sha256"] != evidence.get(
                "kernel_text_sha256"):
            raise ValueError("retained K1 text differs from the kernel text receipt")
        inventory.append(entry)
    return inventory


def _equivalent_work_contract(row: dict[str, Any], compiled: dict[str, dict[str, Any]],
                              grader) -> dict[str, Any]:
    one, many = compiled["rvv"], compiled["rvv_multicore"]
    harts = int(row["core_count"])
    buffer_plan = _safe(grader._buffer_plan(row))
    invariant = {
        "capsule_sha256": row["sha256"], "family": row["family"],
        "operation": row["operation"], "dtype": row["dtype"], "layout": row["layout"],
        "shape": _safe(row["shape"]), "state": _safe(row["state"]),
        "buffer_plan": buffer_plan,
    }
    checks = {
        "same_input_mlir": one.get("input_mlir_sha256") == many.get("input_mlir_sha256") and
                           _digest(one.get("input_mlir_sha256")),
        "same_capsule_digest": all(record.get("metadata", {}).get("capsule_sha256") ==
                                   row["sha256"] for record in (one, many)),
        "same_buffer_plan": one.get("buffer_plan") == many.get("buffer_plan") == buffer_plan,
        "exact_modes": (one.get("mode"), many.get("mode")) == ("rvv", "rvv_multicore") and
                       one.get("metadata", {}).get("actual_mode") == "rvv" and
                       many.get("metadata", {}).get("actual_mode") == "rvv_multicore",
        "exact_harts": one.get("metadata", {}).get("harts") == 1 and
                       many.get("metadata", {}).get("harts") == harts,
        "no_fallback": all(record.get("metadata", {}).get("fallback_used") is False
                           for record in (one, many)),
        "both_l0_pass": all(record.get("ok") is True for record in (one, many)),
    }
    return {
        "version": 1,
        "authority": "same_public_semantic_problem_and_c_abi_trusted_harness_per_call",
        "invariant": invariant,
        "modes": {"rvv": {"harts": 1}, "rvv_multicore": {"harts": harts}},
        "checks": checks,
        "equivalent": all(checks.values()),
        "claim": (
            "Both modes receive the same public capsule, MLIR problem, buffers, C function ABI, "
            "fresh seeded inputs, and required outputs per call. Hart count and implementation are "
            "the treatments; equal dynamic instruction counts or identical internal work are not claimed."),
    }


def _parse_monitor_metrics(child_stdout: str) -> dict[str, int]:
    rows = []
    for line in child_stdout.splitlines():
        match = _METRIC_LINE.fullmatch(line)
        if match:
            rows.append((match.group(1), int(match.group(2))))
    if not rows or len({name for name, _ in rows}) != len(rows):
        raise ValueError("K1 monitor stdout has absent or duplicate metric lines")
    return dict(rows)


def _physical_process_window_evidence(monitor: dict[str, Any], harts: int) -> dict[str, Any]:
    """Grade physical process-window samples without claiming full-hart simultaneity."""
    expected = list(range(harts))
    expected_set = set(expected)
    expected_affinity = "0" if harts == 1 else f"0-{harts - 1}"
    try:
        simultaneous = int(monitor.get("max_simultaneous_running_cpus", -1))
        monitor_wall = int(monitor.get("wall_ns", 0))
        pinned = monitor.get("pinned_affinities_observed")
        pinned_runtime = monitor.get("pinned_runtime_cpus")
        active = monitor.get("active_cpus")
        running = monitor.get("running_cpus_observed")
        sampled_sets = (pinned, pinned_runtime, active, running)
        sets_well_formed = all(
            isinstance(values, list) and bool(values) and len(values) == len(set(values)) and
            set(values) <= expected_set for values in sampled_sets)
        checks = {
            "requested_harts": monitor.get("requested_harts") == harts,
            "not_timed_out": monitor.get("timed_out") is False,
            "monitor_wall_time": monitor_wall > 0,
            "process_affinity": monitor.get("affinity_samples") == [expected_affinity],
            "sampled_cpu_sets_are_nonempty_requested_subsets": sets_well_formed,
            # Sampling can prove that at least two harts overlapped; it cannot prove all H
            # overlapped in one instant.  Preserve the raw maximum and make only this narrow claim.
            "sampled_parallel_overlap": simultaneous >= (1 if harts == 1 else 2),
        }
    except (TypeError, ValueError):
        checks = {"monitor_fields_well_typed": False}
        simultaneous = -1
    return {
        "version": 1,
        "authority": "independent_procfs_process_window_sampling",
        "checks": checks,
        "qualified": all(checks.values()),
        "requested_harts": harts,
        "requested_harts_set": expected,
        "sampled_pinned_harts": monitor.get("pinned_affinities_observed"),
        "sampled_pinned_runtime_harts": monitor.get("pinned_runtime_cpus"),
        "sampled_active_harts": monitor.get("active_cpus"),
        "sampled_running_harts": monitor.get("running_cpus_observed"),
        "maximum_simultaneously_sampled_running_cpus": simultaneous,
        "claim": (
            "The process had the exact requested affinity; every sampled task CPU belonged to that "
            "set, and at least two CPUs were sampled running concurrently for multicore. Sampled "
            "CPU subsets and maximum overlap are reported exactly. This does not claim that all "
            "requested harts were sampled or ran simultaneously."),
    }


def _logical_shard_evidence(metrics: dict[str, Any], harts: int,
                            output_count: int) -> dict[str, Any]:
    return {
        "version": 1,
        "authority": "trusted_serial_poison_to_correct_first_transition_audit",
        "harts": harts,
        "output_elements": output_count,
        "coverage": metrics.get("audit_output_coverage"),
        "owner_min_elements": metrics.get("audit_owner_min_elements"),
        "owner_max_elements": metrics.get("audit_owner_max_elements"),
        "ownership_violations": metrics.get("audit_ownership_violations"),
        "balanced": metrics.get("audit_balanced_shards") == 1,
        "claim": (
            "The excluded serial audit attributes balanced first poison-to-correct output "
            "transitions to logical callbacks. It does not claim physical concurrency, exclusive "
            "writes, or equal dynamic instruction counts."),
    }


def _require_complete_k1_evidence(*, evidence: dict[str, Any], row: dict[str, Any],
                                  mode: str, harts: int, seed: int, grader
                                  ) -> tuple[int, int, dict[str, Any], dict[str, Any]]:
    if not isinstance(evidence, dict):
        assessment = {
            "version": 1, "mode": mode, "harts": harts,
            "checks": {"grader.evidence_object": False},
        }
        raise K1EvidenceError(
            f"{mode} measurement lacks the complete passing grader K1 evidence", assessment)
    checks = evidence.get("checks")
    metrics = evidence.get("metrics")
    monitor = evidence.get("monitor")
    grader_checks = {
        "grader.status_pass": evidence.get("status") == "pass",
        "grader.capsule_exact": evidence.get("capsule") == row["id"],
        "grader.family_exact": evidence.get("family") == row["family"],
        "grader.mode_exact": evidence.get("mode") == mode,
        "grader.harts_exact": evidence.get("harts") == harts,
        "grader.seed_exact": evidence.get("seed") == seed,
        "grader.check_set_exact": (
            isinstance(checks, dict) and set(checks) == REQUIRED_K1_CHECKS),
        "grader.checks_all_true": (
            isinstance(checks, dict) and all(value is True for value in checks.values())),
        "grader.metrics_object": isinstance(metrics, dict),
        "grader.monitor_object": isinstance(monitor, dict),
    }
    if not all(grader_checks.values()):
        assessment = {
            "version": 1, "mode": mode, "harts": harts, "checks": grader_checks,
        }
        raise K1EvidenceError(
            f"{mode} measurement lacks the complete passing grader K1 evidence", assessment)
    assert isinstance(metrics, dict) and isinstance(monitor, dict)
    wall_ns, calls = metrics.get("wall_ns"), metrics.get("calls")
    receipt_nonce = evidence.get("receipt_nonce")
    metrics_parse_error = None
    try:
        parsed_metrics = _parse_monitor_metrics(str(monitor.get("child_stdout", "")))
    except ValueError as error:
        parsed_metrics, metrics_parse_error = None, str(error)
    trusted_lines = [line for line in str(monitor.get("child_stdout", "")).splitlines()
                     if line.startswith("MERLIN_TRUSTED_RESULT ")]
    expected_receipt = (f"MERLIN_TRUSTED_RESULT version=1 seed={seed} nonce={receipt_nonce} "
                        "memory=1 numeric=1")
    physical = _physical_process_window_evidence(monitor, harts)
    output_count = int(grader._buffer_plan(row)["output_count"])
    per_call, attribution = grader._k1_timing_authority(
        metrics, harts, output_count)
    board_wall = evidence.get("board_wall_seconds")
    board_wall_ok = (isinstance(board_wall, (int, float)) and
                     not isinstance(board_wall, bool) and math.isfinite(float(board_wall)) and
                     float(board_wall) > 0)
    replay_checks = {
        "timing.wall_ns_positive_int": (
            isinstance(wall_ns, int) and not isinstance(wall_ns, bool) and wall_ns > 0),
        "timing.calls_positive_int": (
            isinstance(calls, int) and not isinstance(calls, bool) and calls > 0),
        "receipt.nonce_positive_int": (
            isinstance(receipt_nonce, int) and not isinstance(receipt_nonce, bool) and
            receipt_nonce > 0),
        "build.returncode_zero": evidence.get("build_returncode") == 0,
        "kernel_text.digest": _digest(evidence.get("kernel_text_sha256")),
        "binary.local_digest": _digest(evidence.get("local_sha256")),
        "binary.upload_digest_match": (
            evidence.get("remote_sha256") == evidence.get("local_sha256")),
        "ssh.returncode_zero": evidence.get("ssh_returncode") == 0,
        "monitor.returncode_zero": monitor.get("returncode") == 0,
        "monitor.metrics_unique_present_and_exact": (
            metrics_parse_error is None and parsed_metrics == metrics),
        "receipt.trusted_marker_exact": trusted_lines == [expected_receipt],
        "monitor.requested_harts_exact": monitor.get("requested_harts") == harts,
        "monitor.not_timed_out": monitor.get("timed_out") is False,
        "board_wall_seconds.positive_finite": board_wall_ok,
        "grader.per_call_timing_authority": per_call is True,
        "grader.logical_attribution": attribution is True,
        **{
            f"physical_process_window.{name}": passed is True
            for name, passed in physical["checks"].items()
        },
    }
    assessment = {
        "version": 1, "mode": mode, "harts": harts, "checks": replay_checks,
        "monitor_metrics_parse_error": metrics_parse_error,
        "physical_process_window_evidence": physical,
    }
    if not all(replay_checks.values()):
        raise K1EvidenceError(f"{mode} measurement failed K1 evidence requirements", assessment)
    return wall_ns, calls, _logical_shard_evidence(metrics, harts, output_count), physical


def _paired_k1(*, grader, row: dict[str, Any], compiled: dict[str, dict[str, Any]],
               operation_codes: dict[str, int], board_environment: dict[str, Any],
               receipts: Path) -> dict[str, Any]:
    valid_pairs: list[dict[str, Any]] = []
    excluded_pairs: list[dict[str, Any]] = []
    pair_inventory: list[dict[str, Any]] = []
    build_inventory: list[dict[str, Any]] = []
    code_digests: dict[str, str] = {}
    maximum_replacements = int(
        board_environment["maximum_invalid_pair_replacements_per_capsule"])
    connection = grader._k1_connection()
    repeat = pair_attempt = 0
    with grader._k1_lock(connection):
        while repeat < len(PAIR_ORDERS):
            order = PAIR_ORDERS[repeat]
            settle_probes: list[dict[str, Any]] = []
            for settle_attempt in range(int(board_environment["settle_attempts"])):
                before = grader._probe_k1_state(connection)
                settle_probes.append(_safe(before))
                if grader._k1_state_ready(before, board_environment):
                    break
                if settle_attempt + 1 < int(board_environment["settle_attempts"]):
                    time.sleep(float(board_environment["settle_interval_seconds"]))
            else:
                raise ValueError("K1 did not enter the frozen pre-pair environment")
            seed = secrets.randbits(63) or 1
            modes = ("rvv", "rvv_multicore") if order == "rvv_rvv_multicore" else \
                    ("rvv_multicore", "rvv")
            measurements: dict[str, dict[str, Any]] = {}
            for mode in modes:
                run_root = Path(tempfile.mkdtemp(
                    prefix=f"pair_{repeat}_attempt_{pair_attempt}_{mode}_"))
                try:
                    evidence = grader._grade_k1(
                        row, compiled[mode], operation_codes, run_root, seed=seed)
                    build_artifacts = _retain_k1_build_artifacts(
                        receipts=receipts, run_root=run_root, row=row, mode=mode,
                        attempt_id=pair_attempt, evidence=evidence)
                finally:
                    shutil.rmtree(run_root, ignore_errors=True)
                build_inventory.extend(build_artifacts)
                harts = 1 if mode == "rvv" else int(row["core_count"])
                try:
                    wall_ns, calls, logical, physical = _require_complete_k1_evidence(
                        evidence=evidence, row=row, mode=mode, harts=harts, seed=seed,
                        grader=grader)
                except K1EvidenceError as error:
                    failure = {
                        "version": 1,
                        "authority": "controller_live_k1_evidence_failure",
                        "pair_id": repeat,
                        "attempt_id": pair_attempt,
                        "order": order,
                        "mode": mode,
                        "harts": harts,
                        "seed": seed,
                        "assessment": error.assessment,
                        "retained_build_artifacts": build_artifacts,
                        "evidence": _safe(evidence),
                    }
                    failure_receipt = _write_receipt(
                        receipts,
                        f"failures/attempt_{pair_attempt:02d}_pair_{repeat:02d}_{mode}.json",
                        failure)
                    raise ValueError(
                        f"{mode} live K1 evidence failed closed; "
                        f"reason={error}; "
                        f"failed checks: {', '.join(error.failed_checks)}; "
                        f"raw failure receipt={failure_receipt['relative_path']} "
                        f"sha256={failure_receipt['sha256']}") from error
                digest = str(evidence["kernel_text_sha256"])
                if mode in code_digests and code_digests[mode] != digest:
                    raise ValueError(f"{mode} K1 kernel text changed between paired repetitions")
                code_digests.setdefault(mode, digest)
                measurements[mode] = {
                    "elapsed_ns": wall_ns, "calls": calls, "seed": seed,
                    "logical_shard_evidence": logical,
                    "physical_process_window_evidence": physical,
                    "retained_build_artifacts": build_artifacts,
                    "evidence": _safe(evidence),
                }
            after = grader._probe_k1_state(connection)
            valid = grader._k1_state_pair_ok(before, after, board_environment)
            raw = {
                "version": 1, "pair_id": repeat, "attempt_id": pair_attempt,
                "order": order, "seed": seed, "settle_probes": settle_probes,
                "before": _safe(before), "measurements": measurements,
                "after": _safe(after), "valid": bool(valid),
            }
            receipt = _write_receipt(
                receipts, f"pairs/attempt_{pair_attempt:02d}_pair_{repeat:02d}.json", raw)
            pair_inventory.append(receipt)
            record = {**raw, "raw_receipt": receipt}
            pair_attempt += 1
            if not valid:
                excluded_pairs.append(record)
                if len(excluded_pairs) > maximum_replacements:
                    raise ValueError("K1 exceeded the frozen invalid-pair replacement limit")
                continue
            valid_pairs.append(record)
            repeat += 1
    return {
        "pair_orders": list(PAIR_ORDERS), "valid_pairs": valid_pairs,
        "excluded_pairs": excluded_pairs, "attempt_count": pair_attempt,
        "k1_program_count": 2 * pair_attempt, "kernel_text_sha256": code_digests,
        "raw_receipt_inventory": pair_inventory,
        "k1_build_artifact_inventory": build_inventory,
    }


def _replay_pair_receipts(*, receipts: Path, transcript: dict[str, Any], grader,
                          row: dict[str, Any], board_environment: dict[str, Any]
                          ) -> tuple[dict[str, Any], dict[str, Any]]:
    """Reload raw pair files and reconstruct the complete transcript from disk."""
    pair_entries = transcript.get("raw_receipt_inventory")
    build_entries = transcript.get("k1_build_artifact_inventory")
    if not isinstance(pair_entries, list) or not isinstance(build_entries, list):
        raise ValueError("qualification transcript lacks raw receipt inventories")
    disk_records: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for entry in pair_entries:
        path = _verify_inventory_entry(receipts, entry)
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("raw K1 pair receipt is invalid JSON") from exc
        if not isinstance(value, dict):
            raise ValueError("raw K1 pair receipt is not an object")
        disk_records.append((value, entry))

    if len(disk_records) != int(transcript.get("attempt_count", -1)):
        raise ValueError("raw K1 pair receipt count differs from the transcript")
    disk_records.sort(key=lambda item: int(item[0].get("attempt_id", -1)))
    if [record.get("attempt_id") for record, _ in disk_records] != list(
            range(len(disk_records))):
        raise ValueError("raw K1 pair attempts are not one complete chronology")

    expected_by_attempt = {
        int(record["attempt_id"]): record
        for record in [*transcript.get("valid_pairs", []),
                       *transcript.get("excluded_pairs", [])]
    }
    if set(expected_by_attempt) != set(range(len(disk_records))):
        raise ValueError("in-memory K1 pair transcript has missing or duplicate attempts")

    build_by_attempt_mode: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for entry in build_entries:
        _verify_inventory_entry(receipts, entry)
        key = (int(entry.get("attempt_id", -1)), str(entry.get("mode_name", "")))
        build_by_attempt_mode.setdefault(key, []).append(entry)

    valid_pairs: list[dict[str, Any]] = []
    excluded_pairs: list[dict[str, Any]] = []
    code_digests: dict[str, str] = {}
    current_pair_id = 0
    for raw, receipt_entry in disk_records:
        attempt_id = int(raw.get("attempt_id", -1))
        expected = expected_by_attempt[attempt_id]
        if raw != {key: value for key, value in expected.items() if key != "raw_receipt"}:
            raise ValueError("raw K1 pair receipt differs from the in-memory transcript")
        if (current_pair_id >= len(PAIR_ORDERS) or raw.get("version") != 1 or
                raw.get("pair_id") != current_pair_id or
                raw.get("order") != PAIR_ORDERS[current_pair_id]):
            raise ValueError("raw K1 pair chronology/order differs from the frozen protocol")
        settle = raw.get("settle_probes")
        before, after = raw.get("before"), raw.get("after")
        if (not isinstance(settle, list) or not settle or before != settle[-1] or
                any(grader._k1_state_ready(probe, board_environment)
                    for probe in settle[:-1]) or
                not grader._k1_state_ready(before, board_environment)):
            raise ValueError("raw K1 pair settle chronology is invalid")
        recomputed_valid = grader._k1_state_pair_ok(before, after, board_environment)
        if bool(raw.get("valid")) != bool(recomputed_valid):
            raise ValueError("raw K1 pair board-condition decision cannot be replayed")
        seed = raw.get("seed")
        if not isinstance(seed, int) or isinstance(seed, bool) or seed <= 0:
            raise ValueError("raw K1 pair seed is invalid")
        measurements = raw.get("measurements")
        if not isinstance(measurements, dict) or set(measurements) != {"rvv", "rvv_multicore"}:
            raise ValueError("raw K1 pair lacks both exact modes")
        for mode, harts in (("rvv", 1), ("rvv_multicore", int(row["core_count"]))):
            measurement = measurements[mode]
            if not isinstance(measurement, dict) or measurement.get("seed") != seed:
                raise ValueError("raw K1 pair measurement seed differs")
            wall_ns, calls, logical, physical = _require_complete_k1_evidence(
                evidence=measurement.get("evidence"), row=row, mode=mode,
                harts=harts, seed=seed, grader=grader)
            if (measurement.get("elapsed_ns"), measurement.get("calls")) != (wall_ns, calls):
                raise ValueError("raw K1 pair timing differs from its monitor receipt")
            if (measurement.get("logical_shard_evidence") != logical or
                    measurement.get("physical_process_window_evidence") != physical):
                raise ValueError("raw K1 pair attribution summaries cannot be replayed")
            artifacts = measurement.get("retained_build_artifacts")
            expected_artifacts = build_by_attempt_mode.get((attempt_id, mode), [])
            if artifacts != expected_artifacts or {
                    str(entry.get("name")) for entry in expected_artifacts} != set(
                        _K1_ARTIFACT_NAMES):
                raise ValueError("raw K1 pair build-artifact inventory differs")
            artifact_paths = {
                str(entry["name"]): _verify_inventory_entry(receipts, entry)
                for entry in expected_artifacts}
            evidence = measurement["evidence"]
            if (_sha256(artifact_paths["capsule_k1"]) != evidence.get("local_sha256") or
                    _sha256(artifact_paths["kernel.text.bin"]) !=
                    evidence.get("kernel_text_sha256")):
                raise ValueError("retained K1 binary/text does not bind the raw execution receipt")
            digest = str(evidence["kernel_text_sha256"])
            if mode in code_digests and code_digests[mode] != digest:
                raise ValueError("K1 kernel text changed across raw pair receipts")
            code_digests.setdefault(mode, digest)
        record = {**raw, "raw_receipt": receipt_entry}
        if raw["valid"]:
            valid_pairs.append(record)
            current_pair_id += 1
        else:
            excluded_pairs.append(record)
    if current_pair_id != len(PAIR_ORDERS):
        raise ValueError("raw K1 pair receipts do not contain six valid replacements")
    replayed = {
        "pair_orders": list(PAIR_ORDERS), "valid_pairs": valid_pairs,
        "excluded_pairs": excluded_pairs, "attempt_count": len(disk_records),
        "k1_program_count": 2 * len(disk_records), "kernel_text_sha256": code_digests,
        "raw_receipt_inventory": pair_entries,
        "k1_build_artifact_inventory": build_entries,
    }
    if replayed != transcript:
        raise ValueError("disk-replayed K1 transcript differs from the controller transcript")
    return replayed, {
        "version": 1,
        "authority": "independent_disk_receipt_replay",
        "status": "pass",
        "attempts_replayed": len(disk_records),
        "k1_programs_replayed": 2 * len(disk_records),
        "transcript_sha256": _canonical_sha256(replayed),
    }


def _effect_summary(transcript: dict[str, Any], *, margin: float,
                    minimum_directional_pairs: int) -> dict[str, Any]:
    pairs: list[dict[str, Any]] = []
    upper, lower = 1.0 + margin, 1.0 / (1.0 + margin)
    for record in transcript["valid_pairs"]:
        one, many = record["measurements"]["rvv"], record["measurements"]["rvv_multicore"]
        one_per_call = one["elapsed_ns"] / one["calls"]
        many_per_call = many["elapsed_ns"] / many["calls"]
        speedup = one_per_call / many_per_call
        direction = "multicore_faster" if speedup > upper else (
            "multicore_slower" if speedup < lower else "within_aa_noise")
        pairs.append({
            "pair_id": record["pair_id"], "attempt_id": record["attempt_id"],
            "order": record["order"], "seed": record["seed"],
            "rvv_ns_per_call": one_per_call,
            "rvv_multicore_ns_per_call": many_per_call,
            "multicore_speedup": speedup, "margin_classification": direction,
        })
    speedups = [float(pair["multicore_speedup"]) for pair in pairs]
    faster = sum(pair["margin_classification"] == "multicore_faster" for pair in pairs)
    slower = sum(pair["margin_classification"] == "multicore_slower" for pair in pairs)
    geometric_mean = math.exp(statistics.mean(math.log(value) for value in speedups))
    if faster >= minimum_directional_pairs and geometric_mean > upper:
        direction = "multicore_faster"
    elif slower >= minimum_directional_pairs and geometric_mean < lower:
        direction = "multicore_slower"
    else:
        direction = "not_distinguishable_from_aa_noise"
    return {
        "version": 1, "noise_margin": margin, "lower_bound": lower,
        "upper_bound": upper, "minimum_directional_pairs": minimum_directional_pairs,
        "pairs": pairs, "multicore_faster_pairs": faster, "multicore_slower_pairs": slower,
        "within_noise_pairs": len(pairs) - faster - slower,
        "geometric_mean_multicore_speedup": geometric_mean,
        "median_multicore_speedup": statistics.median(speedups),
        "effect_direction": direction,
        "directionally_consistent_effect": direction in {
            "multicore_faster", "multicore_slower"},
        "multicore_speedup_qualified": direction == "multicore_faster",
    }


def verify_saved_qualification(value: dict[str, Any] | str | Path, *, grader=None) \
        -> dict[str, Any]:
    """Independently replay a completed qualification and its raw on-disk evidence."""
    if isinstance(value, (str, Path)):
        result_path = Path(value).resolve()
        try:
            value = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("qualification result is invalid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("qualification result must be an object")
    raw = value.get("raw_receipts")
    if not isinstance(raw, dict):
        raise ValueError("qualification result lacks its raw receipt bundle")
    root = Path(str(raw.get("root", ""))).resolve()
    if root.is_symlink() or not root.is_dir() or _tree_sha256(root) != raw.get("tree_sha256"):
        raise ValueError("qualification raw receipt tree differs from its result seal")
    categories = (
        [raw.get("pre_k1")], raw.get("compiled_artifacts"), raw.get("pairs"),
        raw.get("k1_build_artifacts"), raw.get("executable_sources"),
        raw.get("authority_inputs"),
    )
    entries: list[dict[str, Any]] = []
    for category in categories:
        if not isinstance(category, list) or any(not isinstance(entry, dict)
                                                for entry in category):
            raise ValueError("qualification raw receipt inventory is malformed")
        entries.extend(category)
    paths = [str(entry.get("relative_path", "")) for entry in entries]
    if len(paths) != len(set(paths)):
        raise ValueError("qualification raw receipt inventory contains duplicate paths")
    for entry in entries:
        _verify_inventory_entry(root, entry)
    actual_files = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}
    if actual_files != set(paths):
        raise ValueError("qualification raw receipt inventory does not exactly cover its tree")

    snapshot = value.get("source_snapshot")
    if not isinstance(snapshot, dict) or _canonical_sha256(snapshot) != value.get(
            "source_snapshot_sha256"):
        raise ValueError("qualification executable source snapshot is malformed")
    source_entries = {str(entry.get("source_role")): entry
                      for entry in raw["executable_sources"]}
    if set(source_entries) != set(snapshot):
        raise ValueError("retained executable sources do not exactly cover the source snapshot")
    for name, identity in snapshot.items():
        entry = source_entries[name]
        if (entry.get("sha256") != identity.get("sha256") or
                entry.get("size_bytes") != identity.get("size_bytes")):
            raise ValueError(f"retained executable source identity differs: {name}")

    authority_entries = {str(entry.get("authority_role")): entry
                         for entry in raw["authority_inputs"]}
    expected_authorities = {
        "canonical_public_train": value.get("public_train_sha256"),
        "frozen_optimization_space": value.get("space_sha256"),
        "aa_noise_authority": value.get("lineage", {}).get("noise_authority_sha256"),
    }
    if set(authority_entries) != set(expected_authorities) or any(
            authority_entries[role].get("sha256") != digest
            for role, digest in expected_authorities.items()):
        raise ValueError("retained qualification authority inputs differ from the result lineage")

    if grader is None:
        current_sources = _source_snapshot()
        if current_sources != snapshot:
            raise ValueError("publication replay executable sources differ from the run snapshot")
        grader = _load("merlin_multicore_publication_replay_grader", HERE / "grader.py")

    pre_k1_path = _verify_inventory_entry(root, raw["pre_k1"])
    try:
        pre_k1 = json.loads(pre_k1_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("qualification pre-K1 receipt is invalid JSON") from exc
    if (not isinstance(pre_k1, dict) or
            pre_k1.get("row") != value.get("selected_public_capsule") or
            pre_k1.get("work_contract") != value.get("equivalent_work_contract") or
            pre_k1.get("spike_gate") != value.get("spike_gate")):
        raise ValueError("qualification pre-K1 receipt differs from the published gates")
    compiled_receipts = pre_k1.get("compiled")
    if not isinstance(compiled_receipts, dict) or set(compiled_receipts) != {
            "rvv", "rvv_multicore"}:
        raise ValueError("qualification pre-K1 receipt lacks both compiled modes")
    compiled_entries = {(str(entry.get("mode_name")), str(entry.get("name"))): entry
                        for entry in raw["compiled_artifacts"]}
    for mode, record in compiled_receipts.items():
        expected = {
            "input.mlir": record.get("input_mlir_sha256"),
            "kernel.c": record.get("source_sha256"),
            "lowered.mlir": record.get("lowered_mlir_sha256"),
        }
        metadata_path = _verify_inventory_entry(
            root, compiled_entries[(mode, "metadata.json")])
        if json.loads(metadata_path.read_text(encoding="utf-8")) != record.get("metadata"):
            raise ValueError(f"retained {mode} metadata differs from its pre-K1 receipt")
        for name, digest in expected.items():
            path = _verify_inventory_entry(root, compiled_entries[(mode, name)])
            if _sha256(path) != digest:
                raise ValueError(f"retained {mode} {name} differs from its pre-K1 receipt")

    space_path = Path(str(value.get("space", ""))).resolve()
    if not space_path.is_file() or _sha256(space_path) != value.get("space_sha256"):
        raise ValueError("qualification frozen space differs during publication replay")
    space = yaml.safe_load(space_path.read_text(encoding="utf-8"))
    replayed, replay = _replay_pair_receipts(
        receipts=root, transcript=value.get("transcript"), grader=grader,
        row=value.get("selected_public_capsule"),
        board_environment=dict(space["board_environment"]))
    if replay != value.get("disk_receipt_replay"):
        raise ValueError("qualification disk replay receipt differs from a fresh replay")
    effect = value.get("effect")
    recomputed_effect = _effect_summary(
        replayed, margin=float(effect["noise_margin"]),
        minimum_directional_pairs=int(effect["minimum_directional_pairs"]))
    if recomputed_effect != effect:
        raise ValueError("qualification effect summary cannot be rederived from raw receipts")
    return {
        "version": 1,
        "authority": "standalone_publication_bundle_replay",
        "status": "pass",
        "raw_tree_sha256": raw["tree_sha256"],
        "inventory_file_count": len(entries),
        "transcript_sha256": replay["transcript_sha256"],
        "effect_sha256": _canonical_sha256(effect),
    }


def qualify(*, submission: Path, public_train: Path, space_path: Path,
            noise_authority: Path, prebuilt_destination: Path, receipts_destination: Path,
            grader=None, runner=None,
            semantic_noise_validator: Callable[..., list[str]] | None = None) -> dict[str, Any]:
    submission, public_train, space_path = (
        submission.resolve(), public_train.resolve(), space_path.resolve())
    prebuilt_destination = prebuilt_destination.resolve()
    receipts_destination = receipts_destination.resolve()
    if space_path != DEFAULT_SPACE.resolve():
        raise ValueError("qualification requires the exact frozen optimization_space_v1.yaml path")
    if receipts_destination.exists():
        raise ValueError("raw K1 receipt destination already exists")
    receipts_destination.mkdir(parents=True)
    _require_canonical_public_train(public_train)
    source_before = _source_snapshot()
    source_inventory = _retain_source_snapshot(receipts_destination, source_before)
    grader = grader or _load("merlin_multicore_grader", HERE / "grader.py")
    runner = runner or _load("merlin_multicore_runner", HERE / "beam_search.py")
    space = yaml.safe_load(space_path.read_text(encoding="utf-8"))
    if not isinstance(space, dict):
        raise ValueError("optimization space must be a mapping")
    rows = _jsonl_public_train(public_train)
    raw_tree, raw_package = _tree_sha256(submission), _package_tree_sha256(submission)
    before = (_sha256(public_train), _sha256(space_path), raw_tree, raw_package,
              _sha256(noise_authority.resolve()), _canonical_sha256(source_before))
    noise, lineage = _noise_lineage(
        noise_path=noise_authority, public_train=public_train, space_path=space_path,
        space=space, rows=rows, raw_tree_sha256=raw_tree,
        raw_package_sha256=raw_package, semantic_validator=semantic_noise_validator)
    authority_input_inventory = _retain_authority_inputs(
        receipts_destination, public_train=public_train, space=space_path,
        noise_authority=noise_authority.resolve())
    row = select_public_runtime_capsule(rows)
    if int(row["core_count"]) != int(space["board_environment"]["frequency_core_count"]):
        raise ValueError("selected public capsule does not exercise every frozen K1 hart")
    aa_prebuild = noise.get("prebuild_receipt")
    if not isinstance(aa_prebuild, dict):
        raise ValueError("A/A authority lacks its full sealed prebuild receipt")
    exact_prebuild_fields = (
        "submitted_manifest_sha256", "private_manifest_sha256", "real_build_commands",
        "prebuild_tree_sha256", "built_tree_sha256", "sealed_prebuilt_tree_sha256",
        "submitted_entrypoint_identity", "built_entrypoint_identity",
        "private_build_override", "policy_sha256",
    )
    if any(field not in aa_prebuild for field in exact_prebuild_fields):
        raise ValueError("A/A authority prebuild receipt is incomplete")
    _copy_aa_sealed_prebuilt(
        noise=noise, aa_receipt=aa_prebuild, destination=prebuilt_destination)
    prebuild_receipt = _safe(aa_prebuild)
    lineage.update({
        "aa_built_tree_sha256": aa_prebuild["built_tree_sha256"],
        "aa_sealed_prebuilt_tree_sha256": aa_prebuild["sealed_prebuilt_tree_sha256"],
        "qualification_built_tree_sha256": prebuild_receipt["built_tree_sha256"],
        "qualification_sealed_prebuilt_tree_sha256": prebuild_receipt[
            "sealed_prebuilt_tree_sha256"],
        "exact_prebuild_fields": list(exact_prebuild_fields),
        "qualification_prebuilt_authority": "exact_copy_of_aa_sealed_prebuilt_package",
    })
    toolchain_identity = _safe(noise.get("toolchain_identity"))
    if semantic_noise_validator is None:
        costs = _load("merlin_multicore_timing_tools", HERE / "calibrate_search_costs.py")
        current_toolchain = costs._toolchain(
            grader=grader, kind="k1-program", prebuild_receipt=prebuild_receipt)
        if current_toolchain != toolchain_identity:
            raise ValueError("qualification timing toolchain differs from the A/A authority")

    operation_codes = grader._codes(rows, "operation")
    with tempfile.TemporaryDirectory(prefix="merlin-host-multicore-effect-") as temporary:
        root = Path(temporary)
        package = root / "submission"
        shutil.copytree(prebuilt_destination, package, symlinks=False)
        candidate = runner._candidate([])
        grader._install_search_policy(package, candidate)
        manifest, build_receipt = grader._build(package)
        grader._freeze_tree(package)
        compile_root = root / "compile"
        compile_root.mkdir()
        compiled = {
            mode: grader._compile_one(package, manifest, row, mode, operation_codes, compile_root)
            for mode in ("rvv", "rvv_multicore")
        }
        work_contract = _equivalent_work_contract(row, compiled, grader)
        if not work_contract["equivalent"]:
            raise ValueError("RVV and multicore artifacts do not implement one equivalent capsule ABI")
        compiled_inventory = _retain_compiled_artifacts(receipts_destination, compiled)
        spike_root = root / "spike"
        spike_root.mkdir()
        spike = grader._grade_spike(row, compiled["rvv"], operation_codes, spike_root)
        spike_gate = {
            "rvv_spike_correct": grader._search_spike_correct(spike),
            "rvv_multicore_l0_correct": compiled["rvv_multicore"].get("ok") is True,
            "evidence": _safe(spike),
        }
        if not all(spike_gate[key] for key in (
                "rvv_spike_correct", "rvv_multicore_l0_correct")):
            raise ValueError("pre-K1 Spike/L0 gate failed")
        pre_k1_receipt = _write_receipt(receipts_destination, "pre_k1.json", {
            "row": row, "work_contract": work_contract,
            "compiled": {mode: _safe({key: value for key, value in record.items()
                                      if key != "_kernel_path"})
                         for mode, record in compiled.items()},
            "spike_gate": spike_gate, "build_receipt": build_receipt,
        })
        transcript = _paired_k1(
            grader=grader, row=row, compiled=compiled, operation_codes=operation_codes,
            board_environment=dict(space["board_environment"]),
            receipts=receipts_destination)

    source_after = _source_snapshot()
    current = (_sha256(public_train), _sha256(space_path), _tree_sha256(submission),
               _package_tree_sha256(submission), _sha256(noise_authority.resolve()),
               _canonical_sha256(source_after))
    if current != before or _package_tree_sha256(prebuilt_destination) != \
            prebuild_receipt["sealed_prebuilt_tree_sha256"]:
        raise RuntimeError(
            "qualification mutated a sealed input, executable source, or prebuilt compiler package")
    transcript, disk_replay = _replay_pair_receipts(
        receipts=receipts_destination, transcript=transcript, grader=grader, row=row,
        board_environment=dict(space["board_environment"]))
    effect = _effect_summary(
        transcript, margin=float(noise["derived_noise_margin"]),
        minimum_directional_pairs=int(space["selection"]["minimum_pairwise_wins"]))
    checks = {
        "public_train_only": row["split"] == "train",
        "aa_authority_bound": lineage["noise_authority_sha256"] == _sha256(
            noise_authority.resolve()),
        "equivalent_semantic_work_per_call": work_contract["equivalent"],
        "both_modes_l0_pass": all(work_contract["checks"].values()),
        "pre_k1_spike_gate": spike_gate["rvv_spike_correct"] and
                             spike_gate["rvv_multicore_l0_correct"],
        "six_valid_pairs": len(transcript["valid_pairs"]) == len(PAIR_ORDERS),
        "balanced_order": transcript["pair_orders"].count("rvv_rvv_multicore") == 3 and
                          transcript["pair_orders"].count("rvv_multicore_rvv") == 3,
        "complete_k1_evidence": transcript["k1_program_count"] ==
                                2 * transcript["attempt_count"],
        "independent_disk_receipt_replay": disk_replay["status"] == "pass",
        "logical_shards_balanced": all(
            measurement["logical_shard_evidence"]["balanced"]
            for pair in transcript["valid_pairs"] for measurement in pair[
                "measurements"].values()),
        "physical_process_window_observed": all(
            measurement["physical_process_window_evidence"]["qualified"]
            for pair in transcript["valid_pairs"] for measurement in pair[
                "measurements"].values()),
        "frozen_board_condition_gate": all(pair["valid"] is True
                                           for pair in transcript["valid_pairs"]),
        "directionally_consistent_effect": effect["directionally_consistent_effect"],
    }
    result = {
        "version": 1,
        "kind": "cpu_host_k1_public_multicore_performance_effect_qualification",
        "status": "pass" if all(checks.values()) else "fail",
        "paid_work": False,
        "heldout_opened": False,
        "protocol_state_mutated": False,
        "checks": checks,
        "qualification_interpretation": (
            "A pass proves a directionally consistent effect outside the A/A margin. "
            "Logical shard balance and cumulative physical timed-worker observations are separate. "
            "Only effect.multicore_speedup_qualified supports a speedup claim; all-hart simultaneous "
            "execution and equal dynamic implementation work are not claimed."),
        "lineage": lineage,
        "public_train": str(public_train),
        "public_train_sha256": _sha256(public_train),
        "public_context_sha256": _canonical_sha256(rows),
        "space": str(space_path),
        "space_sha256": _sha256(space_path),
        "submission": str(submission),
        "submission_tree_sha256": raw_tree,
        "submission_package_sha256": raw_package,
        "prebuilt_submission": str(prebuilt_destination),
        "prebuild_receipt": _safe(prebuild_receipt),
        "toolchain_identity": toolchain_identity,
        "selected_public_capsule": row,
        "selection_rule": (
            "static_partition_reuse1_then_max_harts_then_max_work_items_then_lexical_sha256"),
        "equivalent_work_contract": work_contract,
        "spike_gate": spike_gate,
        "transcript": transcript,
        "disk_receipt_replay": disk_replay,
        "effect": effect,
        "raw_receipts": {
            "root": str(receipts_destination),
            "tree_sha256": _tree_sha256(receipts_destination),
            "pre_k1": pre_k1_receipt,
            "compiled_artifacts": compiled_inventory,
            "pairs": transcript["raw_receipt_inventory"],
            "k1_build_artifacts": transcript["k1_build_artifact_inventory"],
            "executable_sources": source_inventory,
            "authority_inputs": authority_input_inventory,
        },
        "source_snapshot": source_before,
        "source_snapshot_sha256": _canonical_sha256(source_before),
        "source_sha256": {name: identity["sha256"]
                          for name, identity in source_before.items()},
    }
    result["publication_bundle_replay"] = verify_saved_qualification(result, grader=grader)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--public-train", type=Path, default=DEFAULT_PUBLIC_TRAIN)
    parser.add_argument("--space", type=Path, default=DEFAULT_SPACE)
    parser.add_argument("--noise-authority", type=Path, required=True)
    args = parser.parse_args(argv)
    handle = start_run(
        suite="cpu-host-compiler", method="k1-multicore-effect-qualification",
        target="k1_cpu", extra={"paid_work": False, "heldout_opened": False})
    result: dict[str, Any] | None = None
    output = handle.run_dir / "metrics" / "k1_multicore_effect_qualification.json"
    retained_input = handle.run_dir / "artifacts_dir" / "prebuild_input_submission"
    prebuilt = handle.run_dir / "artifacts_dir" / "prebuilt_search_package"
    receipts = handle.run_dir / "artifacts_dir" / "raw_k1_receipts"
    try:
        try:
            retained = _retain_input_submission(args.submission, retained_input)
            result = qualify(
                submission=retained, public_train=args.public_train,
                space_path=args.space, noise_authority=args.noise_authority,
                prebuilt_destination=prebuilt, receipts_destination=receipts)
        except Exception as error:
            result = {
                "version": 1,
                "kind": "cpu_host_k1_public_multicore_performance_effect_qualification",
                "status": "fail", "paid_work": False, "heldout_opened": False,
                "protocol_state_mutated": False,
                "error_class": type(error).__name__, "error": str(error),
                "submission": str(args.submission.resolve()),
                "public_train": str(args.public_train.resolve()),
                "space": str(args.space.resolve()),
                "noise_authority": str(args.noise_authority.resolve()),
                "source_sha256": {"qualifier": _sha256(Path(__file__)),
                                  "grader": _sha256(HERE / "grader.py"),
                                  "trusted_harness": _sha256(HERE / "trusted_harness.c"),
                                  "k1_monitor": _sha256(HERE / "k1_monitor.py")},
            }
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(output)
        return 0 if result["status"] == "pass" else 2
    finally:
        finish_run(handle, status=("ok" if result and result["status"] == "pass" else "error"),
                   summary={
                       "ready": bool(result and result["status"] == "pass"),
                       "effect_direction": (result or {}).get("effect", {}).get(
                           "effect_direction"),
                       "multicore_speedup_qualified": (result or {}).get("effect", {}).get(
                           "multicore_speedup_qualified", False),
                       "raw_receipts": str(receipts),
                   })


if __name__ == "__main__":
    raise SystemExit(main())
