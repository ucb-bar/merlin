#!/usr/bin/env python3
"""Run a small, pinned Gemmini baseline/candidate differential experiment.

This is deliberately *not* the agentic Phase-P campaign.  It is a bounded engineering experiment:
the caller names two exact package trees and an explicit kernel list; each package/kernel cell is
screened for correctness on Spike and then timed once on Verilator, serially, at reduced host
priority.  The report is useful provisional evidence, not a replacement for Phase-P's optimization
loop, replicates, controls, receipts, isolation, hidden regrade, or statistical contract.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import re
import shutil
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

import yaml

import _pbcommon as PB
from merlin.benchharness import hash_tree, repo_sha
from merlin.targetgen import capsule_runner as CR


_CONTRACT = str(PB.REPO / "merlin/contract")
_CORPUS_SECTIONS = ("golden_kernels", "model_kernels", "attention_kernels",
                    "conv_kernels", "movement_kernels")
_DIGEST_RE = re.compile(r"[0-9a-fA-F]{64}")
_RUN_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_MAX_TIMEOUT_S = 3600
_TREE_HASH_EXCLUSIONS = {"build", "__pycache__", ".git"}
_CLAIM_SCOPE = (
    "Provisional pinned baseline/candidate differential only; not the full agentic Phase-P contract."
)
_LIMITATIONS = (
    "One Verilator observation per package/kernel cell; no replicate statistics.",
    "No agent optimization loop, negative controls, falsifier families, hidden regrade, or broker receipts.",
    "Package trees are frozen read-only, but this helper does not provide Phase-P's full entrypoint sandbox.",
    "Spike establishes correctness only; any Spike cycle value is rejected as performance evidence.",
)


class ProvisionalExperimentError(RuntimeError):
    """A fail-closed refusal at the provisional experiment boundary."""


@dataclasses.dataclass(frozen=True)
class ExperimentConfig:
    baseline_package: Path
    baseline_sha256: str
    candidate_package: Path
    candidate_sha256: str
    kernels: str
    run_id: str
    timeout_s: int = 900
    nice_increment: int = 10
    output_root: Path = PB.RUNS
    kernels_root: Path = PB.KERNELS
    hardware_counters: bool = True
    counter_unit: str | None = None


def _atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validated_digest(value: str, label: str) -> str:
    if not isinstance(value, str) or _DIGEST_RE.fullmatch(value) is None:
        raise ProvisionalExperimentError(f"{label} must be an exact 64-hex SHA-256")
    return value.lower()


def _validate_tree_surface(root: Path, label: str) -> Path:
    supplied = Path(root)
    if supplied.is_symlink():
        raise ProvisionalExperimentError(f"{label} package root must not be a symlink: {supplied}")
    resolved = supplied.resolve()
    if not resolved.is_dir():
        raise ProvisionalExperimentError(f"{label} package is not a directory: {resolved}")
    for path in resolved.rglob("*"):
        if path.is_symlink():
            raise ProvisionalExperimentError(f"{label} package contains a live symlink: {path}")
    return resolved


def _excluded_source_paths(root: Path) -> list[str]:
    # Record each excluded subtree root, not every bytecode file below it.
    return sorted(path.relative_to(root).as_posix() for path in root.rglob("*")
                  if path.name in _TREE_HASH_EXCLUSIONS)


def _materialize_hashed_tree(source: Path, snapshot: Path, label: str) -> str:
    """Freeze exactly the bytes covered by ``hash_tree``, omitting its explicit exclusions.

    Real Python packages commonly contain disposable ``__pycache__`` directories after functional
    grading.  Copying those unhashed bytes would weaken the pin, while refusing the package would make
    the helper unusable on the preserved candidates it is meant to compare.  The frozen executable
    tree therefore contains precisely the hash domain and records the excluded source paths.
    """
    source = _validate_tree_surface(source, label)
    snapshot = Path(snapshot).resolve()
    if snapshot.exists() or snapshot.is_symlink():
        raise ProvisionalExperimentError(f"{label} frozen tree already exists: {snapshot}")
    before = hash_tree(source)["sha256"]
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        source,
        snapshot,
        symlinks=False,
        ignore=lambda _directory, names: [name for name in names if name in _TREE_HASH_EXCLUSIONS],
    )
    observed = hash_tree(snapshot)["sha256"]
    source_after = hash_tree(source)["sha256"]
    if source_after != before:
        raise ProvisionalExperimentError(
            f"{label} source changed during materialization: {before} != {source_after}")
    if observed != before:
        raise ProvisionalExperimentError(
            f"{label} frozen copy changed during materialization: {before} != {observed}")
    for path in sorted(snapshot.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        if path.is_dir():
            path.chmod(0o555)
        elif path.is_file():
            path.chmod(0o555 if path.stat().st_mode & 0o111 else 0o444)
    snapshot.chmod(0o555)
    return str(observed)


def inspect_package(path: Path, expected_sha256: str, label: str) -> dict:
    """Validate and describe one exact Gemmini OOT package before allocating a run."""
    expected = _validated_digest(expected_sha256, f"{label} digest")
    root = _validate_tree_surface(path, label)
    manifest_path = root / "manifest.yaml"
    if not manifest_path.is_file():
        raise ProvisionalExperimentError(f"{label} package has no manifest.yaml: {root}")
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, Mapping):
        raise ProvisionalExperimentError(f"{label} manifest.yaml is not a mapping")
    if manifest.get("artifact_type") != "mlir_oot_target_backend":
        raise ProvisionalExperimentError(
            f"{label} package is not an mlir_oot_target_backend")
    if manifest.get("target") != PB.TARGET:
        raise ProvisionalExperimentError(
            f"{label} package targets {manifest.get('target')!r}, not {PB.TARGET!r}")
    if manifest.get("integrity_exempt") is not False:
        raise ProvisionalExperimentError(
            f"{label} package must explicitly declare integrity_exempt: false")
    observed = hash_tree(root)
    if observed.get("sha256") != expected:
        raise ProvisionalExperimentError(
            f"{label} package digest mismatch: expected {expected}, observed {observed.get('sha256')}")
    return {
        "label": label,
        "source_path": str(root),
        "expected_sha256": expected,
        "source_tree_sha256": observed["sha256"],
        "source_tree_files": observed["n_files"],
        "tree_hash_exclusions": sorted(_TREE_HASH_EXCLUSIONS),
        "excluded_source_paths": _excluded_source_paths(root),
        "manifest_sha256": _sha256_file(manifest_path),
        "declared_provenance": {
            key: manifest.get(key)
            for key in ("artifact_type", "target", "package_id", "language", "authoring",
                        "provenance", "integrity_exempt")
            if key in manifest
        },
    }


def _selected_corpus(selection: str, kernels_root: Path) -> list[dict]:
    """Resolve an explicit, ordered kernel list; the broad ``all`` spelling is refused."""
    requested = [value.strip() for value in str(selection).split(",") if value.strip()]
    if not requested or any(value.lower() == "all" for value in requested):
        raise ProvisionalExperimentError(
            "--kernels must be a non-empty explicit comma-separated list; 'all' is not admitted")
    if len(requested) != len(set(requested)):
        raise ProvisionalExperimentError("--kernels contains a duplicate kernel id")
    corpus_path = Path(kernels_root) / "kernel_corpus.yaml"
    doc = yaml.safe_load(corpus_path.read_text(encoding="utf-8"))
    if not isinstance(doc, Mapping):
        raise ProvisionalExperimentError("performance kernel corpus is not a mapping")
    rows = [row for section in _CORPUS_SECTIONS for row in (doc.get(section) or [])]
    if any(not isinstance(row, Mapping) for row in rows):
        raise ProvisionalExperimentError("performance kernel corpus contains a non-mapping row")
    ids = [str(row.get("id") or "") for row in rows]
    if any(not kernel_id for kernel_id in ids) or len(ids) != len(set(ids)):
        raise ProvisionalExperimentError("performance corpus has missing or duplicate kernel ids")
    by_id = {str(row["id"]): dict(row) for row in rows}
    missing = [kernel_id for kernel_id in requested if kernel_id not in by_id]
    if missing:
        raise ProvisionalExperimentError(f"unknown performance kernel id(s): {missing}")
    for kernel_id in requested:
        if not (Path(kernels_root) / kernel_id).is_dir():
            raise ProvisionalExperimentError(f"kernel directory is missing: {kernel_id}")
    return [by_id[kernel_id] for kernel_id in requested]


def _normalize_spike_tier(tier: object) -> dict:
    """Retain Spike correctness while making its timing categorically non-citable."""
    row = tier if isinstance(tier, Mapping) else {}
    status = row.get("status")
    raw_cycle_present = row.get("cycles") is not None
    provenance_valid = (row.get("derived_from_rtl") is not True
                        and row.get("cycle_accurate") is not True)
    return {
        "simulator": "spike",
        "tier": "L2",
        "purpose": "correctness_only",
        "status": status,
        "correct": status == "pass" and provenance_valid,
        "cycles": None,
        "cycles_admitted_as_performance_evidence": False,
        "raw_cycle_value_rejected": raw_cycle_present,
        "derived_from_rtl": False,
        "cycle_accurate": False,
    }


def _normalize_verilator_tier(tier: object) -> dict:
    """Admit an L3 cycle count only with passing, explicit RTL/cycle-accurate provenance."""
    row = tier if isinstance(tier, Mapping) else {}
    cycles = row.get("cycles")
    reasons: list[str] = []
    if row.get("status") != "pass":
        reasons.append(f"L3 status is {row.get('status')!r}, not 'pass'")
    if not isinstance(cycles, int) or isinstance(cycles, bool) or cycles <= 0:
        reasons.append("L3 cycles are not a positive integer")
    if row.get("derived_from_rtl") is not True:
        reasons.append("L3 evidence is not explicitly derived_from_rtl")
    if row.get("cycle_accurate") is not True:
        reasons.append("L3 evidence is not explicitly cycle_accurate")
    admitted = not reasons
    return {
        "simulator": "verilator",
        "tier": "L3",
        "status": row.get("status"),
        "correct": row.get("status") == "pass",
        "cycles": cycles if admitted else None,
        "cycles_admitted_as_performance_evidence": admitted,
        "derived_from_rtl": row.get("derived_from_rtl") is True,
        "cycle_accurate": row.get("cycle_accurate") is True,
        "evidence": row.get("evidence"),
        "timing": row.get("timing"),
        "concurrency": row.get("concurrency"),
        "counters": row.get("counters"),
        "timing_observations": row.get("timing_observations"),
        "timing_capability": row.get("timing_capability"),
        "utilization": row.get("utilization"),
        "refusal": "; ".join(reasons) if reasons else None,
    }


def _run_cell(package: Path, package_digest: str, lane: str, kernel: Mapping,
              kernels_root: Path, cells_root: Path, timeout_s: int,
              adapters: Mapping) -> dict:
    kernel_id = str(kernel["id"])
    record = {"kernel": kernel_id, "lane": lane, "package_tree_sha256": package_digest,
              "valid": False}
    try:
        capsule = dict(CR.load_capsule(Path(kernels_root) / kernel_id, contract=_CONTRACT))
        capsule["required_oracle_tiers"] = ["L0", "L1", "L2", "L3"]
        grade = CR.run_capsule(
            capsule,
            str(package),
            runs_root=str(cells_root / kernel_id / lane),
            run_id="capsule",
            contract=_CONTRACT,
            oracle_adapters={"L2": adapters["L2"], "L3": adapters["L3"]},
            timeout=timeout_s,
            target=PB.TARGET,
            workers=1,
        )
        tiers = grade.get("tiers") or {}
        record["grade_status"] = grade.get("status")
        record["spike"] = _normalize_spike_tier(tiers.get("L2"))
        record["verilator"] = _normalize_verilator_tier(tiers.get("L3"))
        record["valid"] = (grade.get("status") == "pass"
                           and record["spike"]["correct"] is True
                           and record["verilator"]["cycles_admitted_as_performance_evidence"] is True)
        if grade.get("failure"):
            record["failure"] = {
                key: grade["failure"].get(key) for key in ("plane", "category", "detail")
            }
    except Exception as exc:  # preserve the other bounded cells and report NO_GO at completion
        record["error"] = f"{type(exc).__name__}: {str(exc)[:500]}"
        record["traceback"] = traceback.format_exc()[-1600:]
    return record


def _paired_results(cells: Sequence[Mapping], kernel_ids: Sequence[str]) -> list[dict]:
    indexed = {(str(cell.get("kernel")), str(cell.get("lane"))): cell for cell in cells}
    pairs: list[dict] = []
    for kernel_id in kernel_ids:
        baseline = indexed.get((kernel_id, "baseline"))
        candidate = indexed.get((kernel_id, "candidate"))
        b_cycles = ((baseline or {}).get("verilator") or {}).get("cycles")
        c_cycles = ((candidate or {}).get("verilator") or {}).get("cycles")
        valid = bool(baseline and candidate and baseline.get("valid") and candidate.get("valid"))
        speedup = (b_cycles / c_cycles) if valid else None
        pairs.append({
            "kernel": kernel_id,
            "baseline_valid": bool(baseline and baseline.get("valid")),
            "candidate_valid": bool(candidate and candidate.get("valid")),
            "baseline_verilator_cycles": b_cycles,
            "candidate_verilator_cycles": c_cycles,
            "candidate_speedup_vs_baseline": speedup,
            "valid": valid,
        })
    return pairs


def _markdown(document: Mapping) -> str:
    packages = document.get("packages") or {}
    pairs = document.get("pairs") or []
    lines = [
        "# Provisional Gemmini differential experiment",
        "",
        f"**Status: {document.get('status', 'NO_GO')}**",
        "",
        _CLAIM_SCOPE,
        "",
        "This report does **not** claim the missing full agentic Phase-P contract.",
        "",
        "## Pinned packages",
        "",
        "| Lane | Package ID | Frozen tree SHA-256 | Manifest SHA-256 |",
        "|---|---|---|---|",
    ]
    for lane in ("baseline", "candidate"):
        package = packages.get(lane) or {}
        declared = package.get("declared_provenance") or {}
        lines.append(
            f"| {lane} | {declared.get('package_id', '')} | "
            f"`{package.get('snapshot_tree_sha256', '')}` | `{package.get('manifest_sha256', '')}` |")
    lines.extend([
        "",
        "## Results",
        "",
        "Spike is correctness-only; its cycles are always `null` and never enter this table.",
        "",
        "| Kernel | Baseline valid | Candidate valid | Baseline L3 cycles | Candidate L3 cycles | Speedup |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for pair in pairs:
        speedup = pair.get("candidate_speedup_vs_baseline")
        speedup_text = f"{speedup:.4f}x" if isinstance(speedup, (int, float)) else "—"
        lines.append(
            f"| {pair.get('kernel')} | {pair.get('baseline_valid')} | "
            f"{pair.get('candidate_valid')} | {pair.get('baseline_verilator_cycles') or '—'} | "
            f"{pair.get('candidate_verilator_cycles') or '—'} | {speedup_text} |")
    lines.extend(["", "## Boundaries", ""])
    lines.extend(f"- {item}" for item in document.get("limitations") or [])
    refusals = document.get("refusals") or []
    if refusals:
        lines.extend(["", "## Refusals", ""])
        lines.extend(f"- {item}" for item in refusals)
    return "\n".join(lines) + "\n"


def _write_reports(run_dir: Path, document: dict, kernel_ids: Sequence[str]) -> None:
    document["pairs"] = _paired_results(document.get("cells") or [], kernel_ids)
    expected = 2 * len(kernel_ids)
    valid_cells = sum(cell.get("valid") is True for cell in document.get("cells") or [])
    document["completion"] = {
        "expected_package_kernel_cells": expected,
        "reported_package_kernel_cells": len(document.get("cells") or []),
        "valid_package_kernel_cells": valid_cells,
        "complete": (len(document.get("cells") or []) == expected
                     and valid_cells == expected
                     and not document.get("refusals")),
    }
    document["status"] = "GO" if document["completion"]["complete"] else "NO_GO"
    _atomic_json(run_dir / "provisional_differential.json", document)
    (run_dir / "provisional_differential.md").write_text(_markdown(document), encoding="utf-8")


def _validate_config(config: ExperimentConfig) -> None:
    if _RUN_ID_RE.fullmatch(config.run_id) is None or config.run_id in (".", ".."):
        raise ProvisionalExperimentError("run id must be a simple non-empty directory name")
    if not 1 <= config.timeout_s <= _MAX_TIMEOUT_S:
        raise ProvisionalExperimentError(
            f"cell timeout must be between 1 and {_MAX_TIMEOUT_S} seconds")
    if not 1 <= config.nice_increment <= 19:
        raise ProvisionalExperimentError("nice increment must be between 1 and 19")
    if config.counter_unit is not None:
        unit = str(config.counter_unit).strip()
        if (not config.hardware_counters or not unit
                or any(not (char.isalnum() or char == "_") for char in unit)):
            raise ProvisionalExperimentError(
                "counter_unit requires hardware counters and must be one identifier token")


def run_experiment(config: ExperimentConfig) -> tuple[int, Path, dict]:
    """Run the bounded experiment, returning ``(exit_code, run_directory, report)``."""
    _validate_config(config)
    source_packages = {
        "baseline": inspect_package(config.baseline_package, config.baseline_sha256, "baseline"),
        "candidate": inspect_package(config.candidate_package, config.candidate_sha256, "candidate"),
    }
    _selected_corpus(config.kernels, config.kernels_root)
    run_dir = Path(config.output_root) / config.run_id
    if run_dir.exists() or run_dir.is_symlink():
        raise ProvisionalExperimentError(
            f"provisional run directory already exists; choose a fresh --run-id: {run_dir}")
    run_dir.mkdir(parents=True)

    package_snapshots: dict[str, Path] = {}
    for lane, source in source_packages.items():
        snapshot = run_dir / "packages" / lane
        observed = _materialize_hashed_tree(Path(source["source_path"]), snapshot, lane)
        if observed != source["expected_sha256"]:
            raise ProvisionalExperimentError(
                f"{lane} frozen package digest changed: {observed} != {source['expected_sha256']}")
        source["snapshot_path"] = str(snapshot)
        source["snapshot_tree_sha256"] = observed
        package_snapshots[lane] = snapshot

    workload_snapshot = run_dir / "workload" / "kernels"
    workload_sha256 = _materialize_hashed_tree(config.kernels_root, workload_snapshot, "workload")
    corpus = _selected_corpus(config.kernels, workload_snapshot)
    kernel_ids = [str(row["id"]) for row in corpus]
    document: dict = {
        "schema": "provisional_gemmini_differential_v1",
        "claim_scope": _CLAIM_SCOPE,
        "status": "NO_GO",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "runner_repo_head": repo_sha(),
        "run_id": config.run_id,
        "bounds": {
            "kernels": kernel_ids,
            "timeout_s_per_package_kernel": config.timeout_s,
            "package_kernel_cells": 2 * len(kernel_ids),
            "sequential": True,
            "workers_per_cell": 1,
            "verilator_observations_per_cell": 1,
            "requested_nice_increment": config.nice_increment,
            "effective_nice": None,
            "hardware_counters": config.hardware_counters,
            "counter_unit": str(config.counter_unit).upper() if config.counter_unit else None,
        },
        "packages": source_packages,
        "workload": {
            "snapshot_path": str(workload_snapshot),
            "tree_sha256": workload_sha256,
            "tree_hash_exclusions": sorted(_TREE_HASH_EXCLUSIONS),
        },
        "cells": [],
        "pairs": [],
        "refusals": [],
        "limitations": list(_LIMITATIONS),
    }
    _write_reports(run_dir, document, kernel_ids)

    adapters: Mapping = {}
    try:
        document["bounds"]["effective_nice"] = os.nice(config.nice_increment)
        defaults = CR.default_adapters()
        if "L2" not in defaults or "L3" not in defaults:
            raise ProvisionalExperimentError("Gemmini default adapters do not provide both L2 and L3")
        adapters = defaults
    except Exception as exc:
        document["refusals"].append(f"could not establish low-priority L2/L3 execution: {exc}")

    previous_counters = os.environ.get("MERLIN_HW_COUNTERS")
    previous_unit = os.environ.get("MERLIN_HW_COUNTER_UNIT")
    try:
        if config.hardware_counters:
            os.environ["MERLIN_HW_COUNTERS"] = "1"
        else:
            os.environ.pop("MERLIN_HW_COUNTERS", None)
        if config.counter_unit:
            os.environ["MERLIN_HW_COUNTER_UNIT"] = str(config.counter_unit).upper()
        else:
            os.environ.pop("MERLIN_HW_COUNTER_UNIT", None)
        if not document["refusals"]:
            cells_root = run_dir / "cells"
            for kernel in corpus:
                for lane in ("baseline", "candidate"):
                    print(f"[{kernel['id']}] {lane}: Spike correctness + Verilator timing", flush=True)
                    document["cells"].append(_run_cell(
                        package_snapshots[lane], source_packages[lane]["snapshot_tree_sha256"],
                        lane, kernel, workload_snapshot, cells_root, config.timeout_s, adapters))
                    _write_reports(run_dir, document, kernel_ids)
    finally:
        if previous_counters is None:
            os.environ.pop("MERLIN_HW_COUNTERS", None)
        else:
            os.environ["MERLIN_HW_COUNTERS"] = previous_counters
        if previous_unit is None:
            os.environ.pop("MERLIN_HW_COUNTER_UNIT", None)
        else:
            os.environ["MERLIN_HW_COUNTER_UNIT"] = previous_unit

    for lane, snapshot in package_snapshots.items():
        after = hash_tree(snapshot)["sha256"]
        source_packages[lane]["snapshot_tree_sha256_after"] = after
        if after != source_packages[lane]["snapshot_tree_sha256"]:
            document["refusals"].append(f"{lane} frozen package changed during measurement")
    workload_after = hash_tree(workload_snapshot)["sha256"]
    document["workload"]["tree_sha256_after"] = workload_after
    if workload_after != workload_sha256:
        document["refusals"].append("frozen workload changed during measurement")
    _write_reports(run_dir, document, kernel_ids)
    return (0 if document["status"] == "GO" else 2), run_dir, document


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-package", type=Path, required=True)
    parser.add_argument("--baseline-sha256", required=True)
    parser.add_argument("--candidate-package", type=Path, required=True)
    parser.add_argument("--candidate-sha256", required=True)
    parser.add_argument("--kernels", required=True,
                        help="explicit comma-separated kernel ids; 'all' is intentionally refused")
    parser.add_argument("--run-id", required=True,
                        help="fresh output directory name under the canonical perf-bench run root")
    parser.add_argument("--timeout", type=int, default=900,
                        help=f"timeout per package/kernel cell (1..{_MAX_TIMEOUT_S}s)")
    parser.add_argument("--nice-increment", type=int, default=10,
                        help="positive host niceness increment inherited by simulator children")
    parser.add_argument("--hardware-counters", action=argparse.BooleanOptionalAction, default=True,
                        help="instrument each cycle window with an RTL-capacity-checked counter set")
    parser.add_argument("--counter-unit",
                        help="select a unit family from the shipped counter header (default: occupancy)")
    args = parser.parse_args(argv)
    try:
        code, run_dir, document = run_experiment(ExperimentConfig(
            baseline_package=args.baseline_package,
            baseline_sha256=args.baseline_sha256,
            candidate_package=args.candidate_package,
            candidate_sha256=args.candidate_sha256,
            kernels=args.kernels,
            run_id=args.run_id,
            timeout_s=args.timeout,
            nice_increment=args.nice_increment,
            hardware_counters=args.hardware_counters,
            counter_unit=args.counter_unit,
        ))
    except ProvisionalExperimentError as exc:
        parser.error(str(exc))
    print(f"{document['status']}: {run_dir / 'provisional_differential.md'}", flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
