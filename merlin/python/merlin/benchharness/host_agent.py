"""Isolated Codex/AET primitives for the CPU-host compiler experiment.

This module knows no experiment path.  Callers supply the frozen task, contracts, public corpus,
arm capabilities, and output directories.  It stages an answer-free workspace, builds the outer
``bwrap`` boundary used by Codex, and converts raw Codex attempts into one reconciled AET trajectory.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml

from merlin.common.artifacts import write_all
from merlin.common.paths import repo_root


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tree_lock(root: Path, *, excluded: frozenset[str] = frozenset({".git"})) -> dict[str, str]:
    rows: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if any(part in excluded for part in relative.parts):
            continue
        if path.is_symlink():
            rows[relative.as_posix()] = "symlink:" + hashlib.sha256(
                os.readlink(path).encode("utf-8")).hexdigest()
        elif path.is_file():
            rows[relative.as_posix()] = "file:" + _sha256(path)
        elif path.exists():
            rows[relative.as_posix()] = "non_regular"
    return rows


@dataclass(frozen=True)
class StagedHostWorkspace:
    path: Path
    input_lock: dict[str, str]
    input_lock_sha256: str


def stage_host_workspace(
    destination: str | Path,
    *,
    task_path: str | Path,
    target_contract_path: str | Path,
    dialect_plan_path: str | Path,
    submission_contract_path: str | Path,
    public_corpus_dir: str | Path,
    search_space_path: str | Path,
    search_runner_path: str | Path,
    trusted_evaluator_path: str | Path | None = None,
    arm_id: str,
    capabilities: Iterable[str],
    treatment: str,
) -> StagedHostWorkspace:
    """Create one arm's writable workspace from public, content-locked inputs only."""
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to reuse host-agent workspace: {destination}")
    destination.mkdir(parents=True)
    (destination / "submission").mkdir()
    # The agent may create arbitrary transient build/search state only below this directory.  Keeping
    # it outside the immutable input lock avoids treating compiler caches as protocol mutations while
    # still making every durable deliverable pass through submission/ and its independent seal.
    (destination / "scratch").mkdir()
    (destination / "contracts").mkdir()
    (destination / "corpus" / "public").mkdir(parents=True)

    task = Path(task_path).resolve()
    target_contract = Path(target_contract_path).resolve()
    dialect_plan = Path(dialect_plan_path).resolve()
    submission_contract = Path(submission_contract_path).resolve()
    public_corpus = Path(public_corpus_dir).resolve()
    search_space = Path(search_space_path).resolve()
    search_runner = Path(search_runner_path).resolve()
    trusted_evaluator = (Path(trusted_evaluator_path).resolve()
                         if trusted_evaluator_path is not None else None)
    shutil.copy2(task, destination / "TASK.md")
    shutil.copy2(target_contract, destination / "contracts" / "target_contract.yaml")
    shutil.copy2(dialect_plan, destination / "contracts" / "dialect_plan.yaml")
    shutil.copy2(submission_contract, destination / "contracts" / "SUBMISSION_CONTRACT.md")
    # The compiler input ABI is public protocol, not grader trivia.  Stage the exact renderer used
    # by the trusted grader plus one synthetic fixture per generic family so treatments can test
    # their parsers without seeing sealed capsules or paper workloads.
    from merlin.benchharness import capsule_descriptor
    shutil.copy2(
        Path(capsule_descriptor.__file__).resolve(),
        destination / "contracts" / "capsule_descriptor.py")
    capsule_descriptor.write_conformance_fixtures(
        destination / "contracts" / "descriptor_fixtures")
    for name in ("train.jsonl", "validation.jsonl"):
        source = public_corpus / name
        if not source.is_file():
            raise FileNotFoundError(f"public corpus split is missing: {source}")
        shutil.copy2(source, destination / "corpus" / "public" / name)

    caps = tuple(sorted(set(str(value) for value in capabilities)))
    plan = yaml.safe_load(dialect_plan.read_text(encoding="utf-8"))
    target_name = str(plan.get("target") or "cpu_host")
    arm_record = {"version": 1, "arm": arm_id, "treatment": treatment,
                  "capabilities": list(caps)}
    (destination / "contracts" / "arm.yaml").write_text(
        yaml.safe_dump(arm_record, sort_keys=False), encoding="utf-8")

    # Arm 2 receives only the neutral C++ repository scaffold. Arm 3/4 additionally receive the
    # generated dialect artifacts and deterministic-search contract. The generators consume only
    # the public dialect plan, so this cannot encode paper workloads or results.
    if "cpp_targetgen_scaffold" in caps:
        from merlin.targetgen.generate import target_repo
        write_all(target_repo.generate_skeleton(target_name), destination / "starter")
    if "generated_cpu_dialect" in caps:
        from merlin.targetgen.generate import mlir_scaffold, xdsl
        write_all([*mlir_scaffold.generate(plan), *xdsl.generate(plan)], destination / "starter")
    if "deterministic_candidate_search" in caps:
        (destination / "policy").mkdir()
        if (not search_space.is_file() or not search_runner.is_file() or
                trusted_evaluator is None or not trusted_evaluator.is_file()):
            raise FileNotFoundError(
                "deterministic search capability requires its frozen space, runner, and trusted shim")
        shutil.copy2(search_space, destination / "policy" / "optimization_space.yaml")
        shutil.copy2(search_runner, destination / "policy" / "beam_search.py")
        shutil.copy2(trusted_evaluator, destination / "policy" / "trusted_evaluator.py")
        (destination / "policy" / "beam_search.py").chmod(0o555)
        (destination / "policy" / "trusted_evaluator.py").chmod(0o555)
        (destination / "policy" / "README.md").write_text(
            "Candidate choice is deterministic. Run beam_search.py with output fixed at "
            "scratch/search_work and evaluator `/usr/bin/python3 -B "
            "policy/trusted_evaluator.py`. Only after convergence, copy "
            "scratch/search_work/search_record.json and scratch/search_work/selected_policy.json into "
            "submission/search/; no other file is permitted there. The shim reaches "
            "the driver-owned broker: trusted Spike screens every candidate across all six generic "
            "families, then the deterministic top survivor receives exactly six balanced K1 "
            "measurement pairs on all six predeclared confirmation families. Correctness and emitted-code "
            "digests are recorded outside this workspace. It exposes only the exact public "
            "train/validation samples and has no heldout argument. Copy selected_policy.json byte-for-"
            "byte to the manifest-declared submission policy. A private replay seal is required.\n",
            encoding="utf-8")

    # An isolated repository keeps Codex rooted in this workspace. No history or remote is staged.
    subprocess.run(["git", "init", "-q"], cwd=destination, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    lock = _tree_lock(destination)
    encoded = json.dumps(lock, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return StagedHostWorkspace(destination, lock, hashlib.sha256(encoded).hexdigest())


def _submission_digest(submission: Path, *, include_policy: bool,
                       exclude_search: bool = False) -> str:
    """Digest every executable/source byte in a submission package.

    ``.git`` is transport metadata, and the manifest-selected policy is omitted only from the
    source digest so it can be pinned independently.  In particular, the final ``search/`` tree is
    part of both post-campaign seals.  The private trusted-observation ledger is driver-owned and
    never resides below ``submission/``.
    """
    manifest = yaml.safe_load((submission / "manifest.yaml").read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("submission manifest must be a mapping")
    policy = Path(str(manifest.get("policy", ""))).as_posix()
    rows = []
    for path in sorted(submission.rglob("*")):
        relative = path.relative_to(submission)
        if path.is_symlink():
            raise ValueError(f"submission symlink is forbidden: {path.relative_to(submission)}")
        if any(part == ".git" for part in relative.parts):
            raise ValueError("submission .git metadata is forbidden from sealed compiler packages")
        if (not path.is_file() or
                (not include_policy and relative.as_posix() == policy) or
                (exclude_search and relative.parts and relative.parts[0] == "search")):
            continue
        rows.append((relative.as_posix(), _sha256(path)))
    encoded = json.dumps(rows, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _submission_source_digest(submission: Path) -> str:
    return _submission_digest(submission, include_policy=False)


def _submission_package_digest(submission: Path) -> str:
    return _submission_digest(submission, include_policy=True)


def _submission_presearch_digest(submission: Path) -> str:
    """Identity captured by the broker before the agent writes final search outputs."""
    return _submission_digest(submission, include_policy=False, exclude_search=True)


def _package_tree_identity(root: Path, *, file_sha_overrides: dict[str, str] | None = None) -> str:
    overrides = file_sha_overrides or {}
    rows = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if ".git" in relative.parts or path.is_symlink() or not (path.is_file() or path.is_dir()):
            raise ValueError(f"invalid sealed package entry: {relative}")
        stat = path.lstat()
        rows.append((relative.as_posix(), "dir" if path.is_dir() else "file",
                     stat.st_mode & 0o777, None if path.is_dir() else
                     overrides.get(relative.as_posix(), _sha256(path))))
    return hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest()


def _compiler_entrypoint_identity(package: Path, manifest: dict[str, Any]) -> list[Any] | None:
    command = manifest.get("compiler", {}).get("command")
    if not isinstance(command, list) or not command:
        return None
    first = command[0]
    raw = command[1] if first in {"python3", "/usr/bin/python3", "/usr/bin/python3.12"} \
        and len(command) > 1 else first
    relative = Path(str(raw))
    if relative.is_absolute() or ".." in relative.parts:
        return None
    entrypoint = (package / relative).resolve()
    if (not entrypoint.is_relative_to(package.resolve()) or not entrypoint.is_file()
            or entrypoint.is_symlink()):
        return None
    return [entrypoint.stat().st_mode & 0o777, _sha256(entrypoint)]


def create_compiler_seal(*, workspace: str | Path, search_seal: dict[str, Any]) -> dict[str, Any]:
    """Create the deterministic post-campaign package seal required before heldout access."""
    submission = Path(workspace).resolve() / "submission"
    manifest = yaml.safe_load((submission / "manifest.yaml").read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("submission manifest must be a mapping")
    policy = (submission / str(manifest.get("policy", ""))).resolve()
    if not policy.is_relative_to(submission) or not policy.is_file():
        raise ValueError("manifest policy is absent or escapes the submission")
    if search_seal.get("status") not in {"pass", "not_required"}:
        raise ValueError("compiler cannot be sealed before its required search verifies")
    return {
        "version": 1,
        "status": "sealed",
        "policy_sha256": _sha256(policy),
        "compiler_source_sha256": _submission_source_digest(submission),
        "compiler_package_sha256": _submission_package_digest(submission),
        "search_status": search_seal["status"],
        "search_record_sha256": search_seal.get("search_record_sha256"),
        "selected_policy_sha256": search_seal.get("selected_policy_sha256", _sha256(policy)),
    }


def _search_semantics(value: Any) -> Any:
    """Remove only nondeterministic transport/time fields before trusted replay comparison."""
    if isinstance(value, dict):
        return {key: _search_semantics(item) for key, item in value.items()
                if key not in {"command", "wall_ns"}}
    if isinstance(value, list):
        return [_search_semantics(item) for item in value]
    return value


def _jsonl_objects(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path} contains a non-object JSONL row")
            rows.append(value)
    return rows


def _recompute_private_capsule(
        row: dict[str, Any], *, secret: bytes, phase: str, split: str) -> dict[str, Any]:
    """Independently reproduce the controller-private shape transform for sealing."""
    private = dict(row)
    shape = dict(row.get("shape", {}))
    for name, value in sorted(shape.items()):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            continue
        material = f"{phase}:{split}:{row['id']}:{name}".encode()
        delta = 1 + int.from_bytes(hashlib.sha256(secret + material).digest()[:2], "big") % 13
        shape[name] = value + delta
    if shape == row.get("shape"):
        raise ValueError("controller-private capsule shape did not change")
    private["shape"] = shape
    identity = {key: private[key] for key in (
        "family", "operation", "dtype", "shape", "layout", "state", "core_count")}
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    digest = hashlib.sha256(encoded).hexdigest()
    private["sha256"] = digest
    private["id"] = f"private-{private['family']}-{digest[:16]}"
    return private


def _private_shape_corpus_valid(
        ledger: Path, record: Any, *,
        expected_samples: dict[str, list[dict[str, Any]]]) -> bool:
    """Verify every private row from its retained secret and frozen public source."""
    try:
        if (not isinstance(record, dict) or
                set(record) != {"authority", "secret_hex", "splits"} or
                record["authority"] != "controller_private_after_compiler_snapshot" or
                not isinstance(record["secret_hex"], str) or len(record["secret_hex"]) != 64):
            return False
        secret = bytes.fromhex(record["secret_hex"])
        split_records = record["splits"]
        expected_keys = {"screen:train", "confirm:train", "confirm:validation"}
        if not isinstance(split_records, dict) or set(split_records) != expected_keys:
            return False
        for key, artifact_record in split_records.items():
            phase, split = key.split(":", 1)
            if not isinstance(artifact_record, dict) or set(artifact_record) != {
                    "path", "sha256", "count", "aliases"}:
                return False
            artifact = (ledger / str(artifact_record["path"])).resolve()
            expected_artifact = (ledger / "private_corpus" / f"{phase}_{split}.jsonl").resolve()
            if (artifact != expected_artifact or not artifact.is_relative_to(ledger) or
                    not artifact.is_file() or artifact.is_symlink() or
                    _sha256(artifact) != artifact_record["sha256"]):
                return False
            private_rows = _jsonl_objects(artifact)
            aliases = artifact_record["aliases"]
            public_rows = expected_samples.get(key)
            if (not isinstance(artifact_record["count"], int) or
                    artifact_record["count"] != len(private_rows) or
                    not isinstance(public_rows, list) or len(public_rows) != len(private_rows) or
                    not isinstance(aliases, dict) or
                    list(aliases) != [str(row.get("id", "")) for row in private_rows] or
                    list(aliases.values()) != [str(row["id"]) for row in public_rows]):
                return False
            for private_row, public_row in zip(private_rows, public_rows, strict=True):
                if private_row != _recompute_private_capsule(
                        public_row, secret=secret, phase=phase, split=split):
                    return False
        return True
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def _expected_private_samples(
        *, runner_path: Path, space_path: Path, train_path: Path,
        validation_path: Path) -> dict[str, list[dict[str, Any]]]:
    spec = importlib.util.spec_from_file_location(
        "merlin_host_seal_frozen_search_runner", runner_path)
    if spec is None or spec.loader is None:
        raise ValueError("cannot load the frozen search runner")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    space = yaml.safe_load(space_path.read_text(encoding="utf-8"))
    if not isinstance(space, dict):
        raise ValueError("frozen search space must be a mapping")
    source_rows = {
        "train": _jsonl_objects(train_path),
        "validation": _jsonl_objects(validation_path),
    }
    result: dict[str, list[dict[str, Any]]] = {}
    for phase, split in (("screen", "train"), ("confirm", "train"),
                         ("confirm", "validation")):
        count = int(space[
            "screen_samples_per_family" if phase == "screen"
            else "confirmation_samples_per_family"])
        families = ((list(space.get("confirmation_families", ())) or None)
                    if phase == "confirm" else None)
        result[f"{phase}:{split}"] = runner.select_semantic_sample(
            source_rows[split], per_family=count, families=families)
    return result


def _trusted_terminal_receipt_valid(
        ledger: Path, receipt_path: Path, *, receipt_id: str,
        index_record: dict[str, Any], evaluations: dict[str, Any]) -> dict[str, Any] | None:
    """Re-derive one broker receipt's request/evaluation association."""
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        request = (ledger / str(receipt["request_artifact"])).resolve()
        request_value = json.loads(request.read_text(encoding="utf-8"))
        parsed = receipt.get("parsed_request")
        if (not isinstance(receipt, dict) or receipt.get("version") != 1 or
                receipt.get("authority") != "driver_trusted_search_broker" or
                receipt.get("request_id") != receipt_id or
                receipt.get("status") not in {"pass", "fail"} or
                not request.is_relative_to(ledger) or not request.is_file() or
                request.is_symlink() or _sha256(request) != receipt.get("request_sha256") or
                not isinstance(receipt.get("wall_ns"), int) or receipt["wall_ns"] < 0 or
                not isinstance(request_value, dict)):
            return None
        if receipt["status"] == "fail" and parsed is None:
            if (receipt.get("response_sha256") is not None or
                    receipt.get("failure_class") not in {
                        "treatment_build_fail", "treatment_agent_fail",
                        "treatment_search_fail", "harness_invalid"} or
                    not isinstance(receipt.get("error"), str)):
                return None
        else:
            parsed_keys = {"version", "split", "phase", "repeats",
                           "policy", "parent_policy", "capsules",
                           "parent_candidate_sha256", "candidate_sha256",
                           "parent_policy_sha256", "policy_sha256", "capsules_sha256"}
            if (not isinstance(parsed, dict) or set(parsed) != parsed_keys or
                    parsed.get("version") != 1 or
                    any(request_value.get(name) != parsed.get(name)
                        for name in ("split", "phase", "repeats", "policy",
                                     "parent_policy", "capsules"))):
                return None
            evaluation_key = (
                f"{parsed['parent_candidate_sha256']}:{parsed['candidate_sha256']}:"
                f"{parsed['split']}:{parsed['phase']}")
            evaluation = evaluations.get(evaluation_key)
            if (receipt.get("evaluation_key") != evaluation_key or
                    not isinstance(receipt.get("multiplicity"), int) or
                    receipt["multiplicity"] < 1 or
                    not isinstance(receipt.get("cache_hit"), bool)):
                return None
            if receipt["status"] == "pass":
                if (not isinstance(evaluation, dict) or
                        receipt.get("response_sha256") != evaluation.get(
                            "observations_sha256") or
                        any(parsed.get(name) != evaluation.get(name) for name in (
                            "parent_candidate_sha256", "candidate_sha256", "split", "phase",
                            "parent_policy_sha256", "policy_sha256", "capsules_sha256"))):
                    return None
            elif (receipt.get("response_sha256") is not None or
                  receipt.get("failure_class") not in {
                      "treatment_build_fail", "treatment_agent_fail",
                      "treatment_search_fail", "harness_invalid"} or
                  not isinstance(receipt.get("error"), str)):
                return None
        expected_index = {
            "path": str(receipt_path.relative_to(ledger)), "sha256": _sha256(receipt_path),
            "status": receipt["status"], "evaluation_key": receipt.get("evaluation_key"),
            "cache_hit": receipt.get("cache_hit"),
            "multiplicity": receipt.get("multiplicity"),
            "response_sha256": receipt.get("response_sha256"),
        }
        return receipt if index_record == expected_index else None
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _terminal_receipt_summary(ledger: Path, index: dict[str, Any]) -> dict[str, Any]:
    receipts = index.get("terminal_receipts")
    evaluations = index.get("evaluations")
    receipt_directory = ledger / "receipts"
    if not isinstance(receipts, dict) or not receipts or not isinstance(evaluations, dict):
        return {"integrity": False, "all_pass": False, "failure_classes": []}
    files = sorted(receipt_directory.glob("*.json")) if receipt_directory.is_dir() else []
    if ([path.name for path in files] != sorted(f"{key}.json" for key in receipts) or
            any(path.is_symlink() for path in files)):
        return {"integrity": False, "all_pass": False, "failure_classes": []}
    values: list[dict[str, Any]] = []
    for receipt_id, record in receipts.items():
        if not isinstance(record, dict):
            return {"integrity": False, "all_pass": False, "failure_classes": []}
        value = _trusted_terminal_receipt_valid(
            ledger, receipt_directory / f"{receipt_id}.json", receipt_id=receipt_id,
            index_record=record, evaluations=evaluations)
        if value is None:
            return {"integrity": False, "all_pass": False, "failure_classes": []}
        values.append(value)
    passing = [value for value in values if value["status"] == "pass"]
    by_key: dict[str, list[dict[str, Any]]] = {}
    for value in passing:
        by_key.setdefault(str(value["evaluation_key"]), []).append(value)
    association_ok = set(by_key) == set(evaluations)
    for key, group in by_key.items():
        ordered = sorted(group, key=lambda value: value["multiplicity"])
        expected = list(range(1, len(ordered) + 1))
        evaluation = evaluations.get(key, {})
        association_ok = association_ok and (
            [value["multiplicity"] for value in ordered] == expected and
            [value["cache_hit"] for value in ordered] == [False] + [True] * (len(ordered) - 1) and
            evaluation.get("request_multiplicity") == len(ordered) and
            all(value["response_sha256"] == evaluation.get("observations_sha256")
                for value in ordered))
    return {
        "integrity": bool(association_ok),
        "all_pass": bool(association_ok and len(passing) == len(values)),
        "failure_classes": [value.get("failure_class") for value in values
                            if value["status"] == "fail"],
    }


def _verify_trusted_search(
    *,
    workspace: str | Path,
    ledger: str | Path,
    space_path: str | Path,
    runner_path: str | Path,
    replay_path: str | Path,
    train_path: str | Path,
    validation_path: str | Path,
) -> dict[str, Any]:
    """Replay and seal an Arm 3/4 search using only private trusted observations."""
    workspace, ledger = Path(workspace).resolve(), Path(ledger).resolve()
    search = workspace / "submission" / "search"
    submitted_record = search / "search_record.json"
    submitted_selected = search / "selected_policy.json"
    checks: dict[str, bool] = {
        "private_ledger": (ledger / "index.json").is_file(),
        "search_record_present": submitted_record.is_file(),
        "selected_policy_present": submitted_selected.is_file(),
    }
    checks["exact_final_search_file_set"] = (
        search.is_dir() and {
            path.relative_to(search).as_posix() for path in search.rglob("*") if path.is_file()
        } == {"search_record.json", "selected_policy.json"} and
        not any(path.is_symlink() for path in search.rglob("*")))
    if not all(checks.values()):
        failure_class = "harness_invalid" if not checks["private_ledger"] \
            else "treatment_search_fail"
        if checks["private_ledger"]:
            try:
                incomplete_index = json.loads(
                    (ledger / "index.json").read_text(encoding="utf-8"))
                receipt_summary = _terminal_receipt_summary(ledger, incomplete_index)
                checks["terminal_receipt_associations"] = receipt_summary["integrity"]
                checks["all_requests_have_passing_terminal_receipts"] = receipt_summary[
                    "all_pass"]
                receipt_failures = set(receipt_summary["failure_classes"])
                if incomplete_index.get("terminal_receipts") and not receipt_summary["integrity"]:
                    failure_class = "harness_invalid"
                elif receipt_failures:
                    failure_class = (
                        "harness_invalid" if (not receipt_summary["integrity"] or
                                              "harness_invalid" in receipt_failures) else
                        "treatment_build_fail" if "treatment_build_fail" in receipt_failures else
                        "treatment_agent_fail" if "treatment_agent_fail" in receipt_failures else
                        "treatment_search_fail")
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                failure_class = "harness_invalid"
        return {"version": 1, "status": "fail", "checks": checks,
                "failure_class": failure_class,
                "reason": "trusted search artifacts are incomplete"}
    index = json.loads((ledger / "index.json").read_text(encoding="utf-8"))
    checks["heldout_never_opened"] = index.get("heldout_opened") is False
    checks["six_balanced_paired_measurements"] = index.get("measurement_repeats") == 6
    budget = index.get("budget", {})
    budget_counters = {
        "screen_evaluations": ("screen_evaluations_used", "screen_evaluation_limit"),
        "confirmation_requests": ("confirmation_requests_used", "confirmation_request_limit"),
        "package_builds": ("package_builds_used", "package_build_limit"),
        "compiler_invocations": ("compiler_invocations_used", "compiler_invocation_limit"),
        "spike_checks": ("spike_checks_used", "spike_check_limit"),
        "k1_programs": ("k1_programs_used", "k1_program_limit"),
    }
    for label, (used_name, limit_name) in budget_counters.items():
        try:
            used, limit = int(budget[used_name]), int(budget[limit_name])
            checks[f"{label}_budget"] = 0 <= used <= limit
        except (KeyError, TypeError, ValueError):
            checks[f"{label}_budget"] = False
    checks["search_wall_deadline"] = (
        isinstance(budget, dict) and budget.get("deadline_exceeded") is False and
        isinstance(budget.get("planning_upper_search_seconds"), (int, float)) and
        float(budget["planning_upper_search_seconds"]) > 0)
    checks["submission_source_unchanged"] = (
        index.get("submission_tree_sha256") ==
        _submission_presearch_digest(workspace / "submission"))
    prebuild = index.get("private_prebuild")
    submission_snapshot = ledger / "submission_snapshot"
    prebuilt_snapshot = ledger / "prebuilt_search_package"
    snapshots_present = submission_snapshot.is_dir() and prebuilt_snapshot.is_dir()
    checks["private_prebuild_snapshots_present"] = snapshots_present
    submitted_manifest_path = submission_snapshot / "manifest.yaml"
    private_manifest_path = prebuilt_snapshot / "manifest.yaml"
    try:
        submitted_manifest = yaml.safe_load(submitted_manifest_path.read_text(encoding="utf-8"))
        private_manifest = yaml.safe_load(private_manifest_path.read_text(encoding="utf-8"))
        reconstructed = dict(private_manifest)
        reconstructed["build"] = submitted_manifest["build"]
        manifests_differ_only_by_build = reconstructed == submitted_manifest
        snapshot_identity = (
            _submission_presearch_digest(submission_snapshot) ==
            index.get("submission_tree_sha256") and
            _submission_presearch_digest(prebuilt_snapshot) ==
            index.get("prebuilt_package_sha256") and
            _sha256(submitted_manifest_path) == prebuild.get("submitted_manifest_sha256") and
            _sha256(private_manifest_path) == prebuild.get("private_manifest_sha256") and
            manifests_differ_only_by_build and
            _package_tree_identity(submission_snapshot) == prebuild.get("prebuild_tree_sha256") and
            _package_tree_identity(
                prebuilt_snapshot,
                file_sha_overrides={"manifest.yaml": _sha256(submitted_manifest_path)}) ==
            prebuild.get("built_tree_sha256") and
            _package_tree_identity(prebuilt_snapshot) ==
            prebuild.get("sealed_prebuilt_tree_sha256") and
            _compiler_entrypoint_identity(prebuilt_snapshot, private_manifest) ==
            prebuild.get("built_entrypoint_identity"))
    except (OSError, AttributeError, TypeError, KeyError, ValueError, yaml.YAMLError):
        snapshot_identity = False
    checks["private_prebuild_snapshot_identity"] = bool(snapshot_identity)
    real_logs = prebuild.get("real_build_logs") if isinstance(prebuild, dict) else None
    checks["private_prebuild_logs_valid"] = (
        isinstance(real_logs, list) and bool(real_logs) and
        [row.get("command") for row in real_logs if isinstance(row, dict)] ==
        prebuild.get("real_build_commands") and
        all(isinstance(row, dict) and row.get("returncode") == 0 and
            isinstance(row.get("wall_seconds"), (int, float)) and row["wall_seconds"] >= 0 and
            isinstance(row.get("stdout_tail"), str) and isinstance(row.get("stderr_tail"), str)
            for row in real_logs))
    checks["controller_private_prebuild"] = (
        isinstance(prebuild, dict) and
        prebuild.get("authority") == "driver_private_prebuild" and
        prebuild.get("private_build_override") == ["/bin/true"] and
        isinstance(prebuild.get("real_build_commands"), list) and
        bool(prebuild["real_build_commands"]) and
        prebuild["real_build_commands"] != [["/bin/true"]] and
        checks["private_prebuild_snapshots_present"] and
        checks["private_prebuild_snapshot_identity"] and
        checks["private_prebuild_logs_valid"] and
        all(isinstance(value, str) and len(value) == 64
            for value in (index.get("prebuilt_package_sha256"),
                          prebuild.get("submitted_manifest_sha256"),
                          prebuild.get("private_manifest_sha256"))))
    try:
        expected_private_samples = _expected_private_samples(
            runner_path=Path(runner_path).resolve(), space_path=Path(space_path).resolve(),
            train_path=Path(train_path).resolve(), validation_path=Path(validation_path).resolve())
    except Exception:
        expected_private_samples = {}
    checks["controller_private_shape_corpus"] = _private_shape_corpus_valid(
        ledger, index.get("private_shape_corpus"), expected_samples=expected_private_samples)
    evaluations = index.get("evaluations")
    private_splits = (index.get("private_shape_corpus") or {}).get("splits", {}) \
        if isinstance(index.get("private_shape_corpus"), dict) else {}
    checks["evaluations_bind_private_shape_corpus"] = (
        isinstance(evaluations, dict) and bool(evaluations) and
        all(isinstance(record, dict) and
            isinstance(private_splits.get(f"{record.get('phase')}:{record.get('split')}"), dict) and
            record.get("private_capsules_sha256") == private_splits[
                f"{record.get('phase')}:{record.get('split')}"]["sha256"] and
            record.get("private_capsule_ids") == list(private_splits[
                f"{record.get('phase')}:{record.get('split')}"]["aliases"])
            for record in evaluations.values()))
    receipt_summary = _terminal_receipt_summary(ledger, index)
    checks["terminal_receipt_associations"] = receipt_summary["integrity"]
    checks["all_requests_have_passing_terminal_receipts"] = receipt_summary["all_pass"]
    broker_terminal = index.get("broker_terminal")
    checks["broker_terminal_timing"] = (
        isinstance(broker_terminal, dict) and broker_terminal.get("status") == "stopped" and
        isinstance(broker_terminal.get("start_monotonic_ns"), int) and
        isinstance(broker_terminal.get("end_monotonic_ns"), int) and
        isinstance(broker_terminal.get("wall_ns"), int) and broker_terminal["wall_ns"] >= 0 and
        broker_terminal["end_monotonic_ns"] - broker_terminal["start_monotonic_ns"] ==
        broker_terminal["wall_ns"])

    with tempfile.TemporaryDirectory(prefix="merlin-trusted-search-replay-") as temporary:
        official = Path(temporary) / "official"
        command = [
            sys.executable, str(Path(runner_path).resolve()),
            "--space", str(Path(space_path).resolve()),
            "--train", str(Path(train_path).resolve()),
            "--validation", str(Path(validation_path).resolve()),
            "--output", str(official), "--", sys.executable,
            str(Path(replay_path).resolve()), "--ledger", str(ledger),
        ]
        proc = subprocess.run(command, capture_output=True, text=True, timeout=1800)
        checks["deterministic_replay"] = proc.returncode == 0
        if proc.returncode:
            return {"version": 1, "status": "fail", "checks": checks,
                    "failure_class": ("treatment_search_fail"
                                      if checks.get("all_requests_have_passing_terminal_receipts")
                                      else "harness_invalid"),
                    "reason": "trusted observation ledger cannot replay the frozen search",
                    "replay_stderr_tail": proc.stderr[-4000:]}
        official_record = json.loads(
            (official / "search_record.json").read_text(encoding="utf-8"))
        untrusted_record = json.loads(submitted_record.read_text(encoding="utf-8"))
        checks["search_record_matches_replay"] = (
            _search_semantics(untrusted_record) == _search_semantics(official_record))
        checks["selected_policy_matches_replay"] = (
            submitted_selected.read_bytes() == (official / "selected_policy.json").read_bytes())
        checks["independent_convergence_sweep"] = (
            official_record.get("status") == "converged" and
            int(official_record.get("required_empty_sweeps", 0)) == 1 and
            int(official_record.get("empty_sweeps", 0)) >= 1 and
            len(official_record.get("sweeps", ())) >= 1 and
            official_record["sweeps"][-1].get("winner") is None)
        checks["staged_spike_k1_policy"] = (
            official_record.get("selection_policy") == "spike_screen_then_k1_confirmation")

    manifest = yaml.safe_load(
        (workspace / "submission" / "manifest.yaml").read_text(encoding="utf-8"))
    relative = Path(str(manifest.get("policy", "")))
    policy = (workspace / "submission" / relative).resolve()
    checks["manifest_policy_inside_submission"] = policy.is_relative_to(
        (workspace / "submission").resolve())
    checks["submission_policy_byte_match"] = (
        policy.is_file() and policy.read_bytes() == submitted_selected.read_bytes())
    status = "pass" if all(checks.values()) else "fail"
    controller_checks = {
        "private_ledger", "heldout_never_opened", "six_balanced_paired_measurements",
        "controller_private_prebuild", "private_prebuild_snapshots_present",
        "private_prebuild_snapshot_identity", "private_prebuild_logs_valid",
        "controller_private_shape_corpus", "evaluations_bind_private_shape_corpus",
        "terminal_receipt_associations", "broker_terminal_timing",
    }
    receipt_failures = set(receipt_summary["failure_classes"])
    if status == "pass":
        failure_class = None
    elif (any(not checks.get(name, False) for name in controller_checks) or
          "harness_invalid" in receipt_failures):
        failure_class = "harness_invalid"
    elif "treatment_build_fail" in receipt_failures:
        failure_class = "treatment_build_fail"
    elif "treatment_agent_fail" in receipt_failures:
        failure_class = "treatment_agent_fail"
    else:
        failure_class = "treatment_search_fail"
    return {
        "version": 1,
        "status": status,
        "failure_class": failure_class,
        "checks": checks,
        "selected_policy_sha256": _sha256(submitted_selected),
        "search_record_sha256": _sha256(submitted_record),
        "trusted_ledger_sha256": _sha256(ledger / "index.json"),
        "trusted_evaluation_count": len(index.get("evaluations", {})),
        "trusted_evaluation_wall_ns": sum(
            int(value.get("wall_ns", 0)) for value in index.get("evaluations", {}).values()
            if isinstance(value, dict)),
        "trusted_broker_wall_ns": int(broker_terminal.get("wall_ns", 0))
        if isinstance(broker_terminal, dict) else 0,
    }


def verify_trusted_search(**kwargs: Any) -> dict[str, Any]:
    """Fail-closed public wrapper that always returns a machine-readable seal."""
    try:
        return _verify_trusted_search(**kwargs)
    except Exception as exc:
        return {"version": 1, "status": "fail", "checks": {},
                "failure_class": "harness_invalid",
                "reason": f"trusted search verification error: {type(exc).__name__}: {exc}"}


def audit_staged_inputs(staged: StagedHostWorkspace) -> dict[str, Any]:
    """Detect input mutation while allowing only declared output/transport/scratch namespaces."""
    current = _tree_lock(staged.path)
    immutable = {name: digest for name, digest in staged.input_lock.items()
                 if not name.startswith(("submission/", "scratch/"))}
    changed = sorted(name for name, digest in immutable.items() if current.get(name) != digest)
    sanctioned = {"submission", "scratch", ".trusted_search_channel"}
    unexpected = sorted(name for name in current
                        if not any(name == root or name.startswith(root + "/")
                                   for root in sanctioned)
                        and name not in immutable)
    return {"version": 2, "ok": not changed and not unexpected,
            "sanctioned_mutable_roots": ["submission", "scratch", ".trusted_search_channel"],
            "changed_or_missing": changed, "unexpected": unexpected,
            "input_lock_sha256": staged.input_lock_sha256}


def prepare_isolated_codex_home(destination: str | Path) -> tuple[Path, Path | None]:
    """Create an empty per-run Codex home; credentials remain bind-mounted from the real home."""
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    # The mount destination must exist but contains no secret in the run bundle.
    (destination / "auth.json").touch(mode=0o600)
    real_home = Path(os.environ.get("CODEX_HOME") or (Path.home() / ".codex"))
    auth = real_home / "auth.json"
    return destination, auth if auth.is_file() else None


def codex_bwrap_argv(
    workspace: str | Path,
    codex_home: str | Path,
    *,
    output_path: str | Path | None = None,
) -> list[str]:
    """Return a deny-by-default outer sandbox for one Codex subprocess.

    The checkout and all of ``/scratch*`` remain hidden. Only the arm workspace, isolated Codex
    state, credentials, and compiler executables are mounted back. ``output_path`` is the temporary
    ``--output-last-message`` file allocated by Chia and must be rebound over the sandbox's /tmp.
    """
    from merlin.targetgen.sandbox import bwrap

    workspace, codex_home = Path(workspace).resolve(), Path(codex_home).resolve()
    argv = bwrap.base_argv(workspace, {}, repo=repo_root())
    # Codex does not need Claude state; mask the broad bind base_argv supplies for Claude harnesses.
    claude_home = Path.home() / ".claude"
    if claude_home.is_dir():
        argv += ["--tmpfs", str(claude_home)]

    real_codex_home = Path(os.environ.get("CODEX_HOME") or (Path.home() / ".codex"))
    packages = real_codex_home / "packages"
    if packages.is_dir():
        argv += ["--ro-bind", str(packages), str(packages)]
    argv += ["--bind", str(codex_home), str(codex_home)]
    auth = real_codex_home / "auth.json"
    if auth.is_file():
        argv += ["--bind", str(auth), str(codex_home / "auth.json")]
    argv += ["--setenv", "CODEX_HOME", str(codex_home)]
    argv += ["--setenv", "PYTHONDONTWRITEBYTECODE", "1"]

    tool_roots = [repo_root() / ".venv", repo_root() / "third_party" / "llvm-install"]
    cross = Path("/scratch2/agustin/merlin/build/host-merlin-release/install")
    if cross.is_dir():
        tool_roots.append(cross)
    for root in tool_roots:
        if root.is_dir():
            argv += ["--ro-bind", str(root), str(root)]
    path_dirs = [str(root / "bin") for root in tool_roots if (root / "bin").is_dir()]
    argv += ["--setenv", "PATH", ":".join([*path_dirs, "/usr/bin", "/bin"])]

    if output_path is not None:
        output = Path(output_path).resolve()
        if not output.is_file():
            raise FileNotFoundError(f"Codex output transport file is absent: {output}")
        argv += ["--bind", str(output), str(output)]
    return argv


def probe_codex_bwrap_runtime(codex_binary: str | Path, *, timeout: int = 30) -> dict[str, Any]:
    """Exercise the production Codex mount boundary without contacting the provider.

    ``codex --version`` is local and token-free.  Running it behind the exact bwrap prefix catches
    user-namespace and mount failures before a one-shot protocol is claimed or a cell authorization
    is consumed.  Every execution error fails closed and leaves bounded diagnostic evidence.
    """
    started_ns = time.monotonic_ns()
    result: dict[str, Any] = {
        "version": 1,
        "authority": "production_codex_bwrap_local_version_probe",
        "token_or_provider_work": False,
        "codex_binary": str(Path(codex_binary).resolve()),
        "ready": False,
    }
    try:
        with tempfile.TemporaryDirectory(prefix="merlin-codex-bwrap-probe-") as temporary:
            root = Path(temporary).resolve()
            workspace = root / "workspace"
            workspace.mkdir()
            codex_home, _ = prepare_isolated_codex_home(root / "codex-home")
            output = root / "last-message.txt"
            output.touch(mode=0o600)
            argv = [
                *codex_bwrap_argv(
                    workspace, codex_home, output_path=output),
                str(Path(codex_binary).resolve()),
                "--version",
            ]
            proc = subprocess.run(
                argv, stdin=subprocess.DEVNULL, capture_output=True, text=True,
                timeout=timeout, check=False)
            result.update({
                "returncode": proc.returncode,
                "stdout": proc.stdout.strip()[:1000],
                "stderr": proc.stderr.strip()[:1000],
                "ready": proc.returncode == 0,
            })
    except subprocess.TimeoutExpired as exc:
        result.update({
            "returncode": None,
            "failure": "timeout",
            "stdout": (exc.stdout or "")[:1000] if isinstance(exc.stdout, str) else "",
            "stderr": (exc.stderr or "")[:1000] if isinstance(exc.stderr, str) else "",
        })
    except Exception as exc:
        result.update({
            "returncode": None,
            "failure": type(exc).__name__,
            "stderr": str(exc)[:1000],
        })
    result["elapsed_ns"] = time.monotonic_ns() - started_ns
    return result


def write_codex_bwrap_wrapper(
    path: str | Path,
    *,
    workspace: str | Path,
    codex_home: str | Path,
    codex_binary: str | Path,
) -> Path:
    """Write an executable that dynamically binds Chia's output file then execs Codex in bwrap."""
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    prefix = codex_bwrap_argv(workspace, codex_home)
    script = f'''#!/usr/bin/env python3
import os
import sys

PREFIX = {prefix!r}
CODEX = {str(Path(codex_binary).resolve())!r}

args = sys.argv[1:]
output = None
for flag in ("--output-last-message", "-o"):
    if flag in args:
        index = args.index(flag)
        if index + 1 < len(args):
            output = os.path.realpath(args[index + 1])
            break
if not output or not os.path.isfile(output):
    raise SystemExit("missing Codex --output-last-message transport file")
argv = PREFIX + ["--bind", output, output, CODEX, *args]
os.execv("/usr/bin/bwrap", argv)
'''
    path.write_text(script, encoding="utf-8")
    path.chmod(0o700)
    return path


def _link(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def record_codex_trajectory(
    *,
    run_result: Any,
    run_id: str,
    model: str,
    run_dir: str | Path,
    logger: Any,
) -> dict[str, Any]:
    """Import every attempt once into AET and retain separate raw/timestamp/tool ledgers."""
    from aet.trajectory.importers.codex import import_codex_run
    from aet.trajectory.reconcile import reconcile_codex, token_ledger_rows, tool_ledger_rows
    from aet.trajectory.recording import emit_trajectory

    run_dir = Path(run_dir)
    raw_import = run_dir / "agent" / "aet_raw"
    ts_import = run_dir / "agent" / "aet_timestamped"
    raw_import.mkdir(parents=True, exist_ok=False)
    ts_import.mkdir(parents=True, exist_ok=False)
    for attempt in run_result.attempts:
        raw = Path(attempt.raw_event_path or "")
        timestamped = Path(attempt.arrival_timestamped_event_path or "")
        if not raw.is_file() or not timestamped.is_file():
            raise ValueError(f"attempt {attempt.index} lacks raw or timestamped Codex evidence")
        name = f"attempt_{attempt.index:04d}.jsonl"
        _link(raw, raw_import / name)
        _link(timestamped, ts_import / name)

    trajectory, normalized = import_codex_run(
        raw_import, timestamped=ts_import, model=model, billing_mode="subscription",
        run_id=run_id)
    reconciliation = reconcile_codex(normalized, trajectory, admin_usd=None)
    if not reconciliation["ok"]:
        raise ValueError("Codex/AET raw-event or token reconciliation failed")
    emit_trajectory(trajectory, logger, run_dir)

    metrics = run_dir / "metrics"
    (metrics / "codex_reconciliation.json").write_text(
        json.dumps(reconciliation, indent=2) + "\n", encoding="utf-8")
    with (metrics / "token_ledger.jsonl").open("w", encoding="utf-8") as stream:
        for row in token_ledger_rows(normalized):
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    tools = run_dir / "agent" / "tools.jsonl"
    with tools.open("w", encoding="utf-8") as stream:
        for row in tool_ledger_rows(normalized):
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    return {"trajectory": trajectory.to_dict(), "reconciliation": reconciliation,
            "raw_attempts": len(run_result.attempts)}
