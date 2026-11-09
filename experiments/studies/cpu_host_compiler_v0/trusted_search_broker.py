#!/usr/bin/env python3
"""Driver-side trusted evaluator for CPU-host policy search.

The agent can request only one operation: evaluate a frozen-space candidate on the exact
content-selected train or validation sample.  This process runs outside Codex's bwrap, owns the K1
connection, and writes a private replay ledger under the AET run.  It never opens the sealed split.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import signal
import shutil
import sys
import time
import re
import secrets
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import yaml


HERE = Path(__file__).resolve().parent


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import trusted module {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: expected an object")
        rows.append(value)
    return rows


def _private_capsule(row: dict[str, Any], *, secret: bytes, phase: str, split: str
                     ) -> dict[str, Any]:
    """Create a reproducible controller-private shape after compiler-source freeze."""
    private = dict(row)
    shape = dict(row.get("shape", {}))
    for name, value in sorted(shape.items()):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            continue
        material = f"{phase}:{split}:{row['id']}:{name}".encode()
        delta = 1 + int.from_bytes(
            hashlib.sha256(secret + material).digest()[:2], "big") % 13
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


def _tree_digest(root: Path, *, ignored: frozenset[str] = frozenset({"search"})) -> str:
    manifest = yaml.safe_load((root / "manifest.yaml").read_text(encoding="utf-8"))
    policy_relative = Path(str(manifest.get("policy", ""))).as_posix()
    rows = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if any(part == ".git" for part in relative.parts):
            raise ValueError("submission .git metadata is forbidden")
        if (not path.is_file() or relative.as_posix() == policy_relative or
                (relative.parts and relative.parts[0] in ignored)):
            continue
        rows.append((relative.as_posix(), _sha256(path)))
    payload = json.dumps(rows, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _reject_symlinks(root: Path) -> None:
    links = [path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_symlink()]
    if links:
        raise ValueError(f"submission symlinks are forbidden: {links[:8]}")
    metadata = [path.relative_to(root).as_posix() for path in root.rglob("*")
                if ".git" in path.relative_to(root).parts]
    if metadata:
        raise ValueError(f"submission .git metadata is forbidden: {metadata[:8]}")


def _reject_presearch_files(submission: Path) -> None:
    """Search work is outside the package; no unmeasured dependency may hide below search/."""
    search_root = submission / "search"
    search_files = ([path.relative_to(submission).as_posix()
                     for path in search_root.rglob("*") if path.is_file()]
                    if search_root.exists() else [])
    if search_files:
        raise ValueError(
            "submission/search must be empty before trusted evaluation; run beam work "
            f"outside the package: {search_files[:8]}")


def _reject_public_specialization(submission: Path, public_rows: list[dict[str, Any]]) -> None:
    needles = set()
    for row in public_rows:
        needles.update((str(row["id"]), str(row["sha256"]),
                        json.dumps(row.get("shape", {}), sort_keys=True, separators=(",", ":"))))
    hits = []
    for path in sorted(submission.rglob("*")):
        if not path.is_file() or path.stat().st_size > 16 * 1024 * 1024:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if any(needle and needle in text for needle in needles):
            hits.append(path.relative_to(submission).as_posix())
    if hits:
        raise ValueError(f"submission embeds exact public capsule identities/shapes: {hits[:8]}")


def _require_real_build_manifest(submission: Path, space: dict[str, Any]) -> None:
    """Require a real submitted build; only the trusted controller may install a no-op override."""
    invariant = space.get("search_package")
    expected = {
        "submitted_manifest_requires_real_build": True,
        "controller_private_prebuild": True,
        "private_build_override": ["/bin/true"],
        "candidate_time_build_then_forbidden": True,
    }
    if invariant != expected:
        raise ValueError("frozen search package invariant is absent or malformed")
    manifest = yaml.safe_load(
        (submission / "manifest.yaml").read_text(encoding="utf-8"))
    build = manifest.get("build") if isinstance(manifest, dict) else None
    command = build.get("command") if isinstance(build, dict) else None
    if (not isinstance(command, list) or not command or command == ["/bin/true"] or
            any(not isinstance(part, str) or not part for part in command)):
        raise ValueError(
            "submitted manifest must retain a reproducible real build command; "
            "the no-op build override is controller-private")


def _inside(path: Path, root: Path) -> Path:
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("trusted evaluator request path escapes the staged workspace")
    return resolved


def _regular_bounded(path: Path, *, maximum_bytes: int, label: str) -> Path:
    try:
        stat = path.lstat()
    except OSError as exc:
        raise ValueError(f"{label} is absent") from exc
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular non-symlink file")
    if stat.st_size > maximum_bytes:
        raise ValueError(f"{label} exceeds its frozen size limit")
    return path


def _broker_failure_class(exc: Exception) -> str:
    """Preserve treatment-owned failures without relabeling controller outages as observations."""
    declared = getattr(exc, "failure_class", None)
    if declared in {
            "treatment_build_fail", "treatment_agent_fail",
            "treatment_search_fail", "harness_invalid"}:
        return str(declared)
    if isinstance(exc, TimeoutError):
        return ("treatment_build_fail" if "package-build" in str(exc)
                else "treatment_search_fail")
    if isinstance(exc, (ValueError, json.JSONDecodeError)):
        return "treatment_search_fail"
    return "harness_invalid"


@contextmanager
def _wall_deadline(deadline_monotonic_ns: int):
    """Hard-stop one trusted evaluation at the frozen monotonic search deadline."""
    remaining = (deadline_monotonic_ns - time.monotonic_ns()) / 1e9
    if remaining <= 0:
        raise RuntimeError("trusted search exceeded its frozen planning wall budget")
    previous_handler = signal.getsignal(signal.SIGALRM)

    def expired(_signum, _frame):
        raise TimeoutError("trusted evaluation exceeded the frozen search wall deadline")

    signal.signal(signal.SIGALRM, expired)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, remaining)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, *previous_timer)
        signal.signal(signal.SIGALRM, previous_handler)


def _validate_candidate(policy_path: Path, space: dict[str, Any]) -> dict[str, Any]:
    candidate = json.loads(policy_path.read_text(encoding="utf-8"))
    if not isinstance(candidate, dict) or candidate.get("version") != 1:
        raise ValueError("candidate policy must be a version-1 object")
    actions = candidate.get("actions")
    if not isinstance(actions, list):
        raise ValueError("candidate actions must be a list")
    allowed = {str(action["id"]): action for action in space["actions"]}
    if len({str(action.get("id", "")) for action in actions}) != len(actions):
        raise ValueError("candidate action ids must be unique")
    for action in actions:
        expected = allowed.get(str(action.get("id", "")))
        if expected is None or action != expected:
            raise ValueError("candidate contains an action outside the frozen optimization space")
    canonical = sorted(actions, key=lambda row: (int(row["stage"]), str(row["group"]),
                                                  str(row["id"])))
    if actions != canonical:
        raise ValueError("candidate actions are not in canonical order")
    payload = [{key: value for key, value in action.items() if key != "evidence"}
               for action in canonical]
    digest = hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    if candidate.get("candidate_sha256") != digest:
        raise ValueError("candidate_sha256 does not match the frozen canonical action list")
    return candidate


def _validate_parent_child(parent: dict[str, Any], child: dict[str, Any]) -> None:
    parent_ids = {str(action["id"]) for action in parent["actions"]}
    child_ids = {str(action["id"]) for action in child["actions"]}
    if not parent_ids < child_ids or len(child_ids - parent_ids) != 1:
        raise ValueError("trusted evaluator candidate must add exactly one action to its parent")


def _is_sha256(value: Any) -> bool:
    text = str(value or "")
    return len(text) == 64 and all(char in "0123456789abcdef" for char in text)


def _validate_observations(rows: list[dict[str, Any]], capsules: list[dict[str, Any]],
                           repeats: int, phase: str,
                           board_environment: dict[str, Any] | None = None) -> None:
    expected = {str(row["id"]): row for row in capsules}
    actual = [str(row.get("capsule_id", "")) for row in rows]
    if len(actual) != len(set(actual)) or set(actual) != set(expected):
        raise RuntimeError("trusted grader did not cover the exact public capsule sample once")
    for row in rows:
        capsule = expected[str(row["capsule_id"])]
        if row.get("family") != capsule.get("family") or not isinstance(
                row.get("correctness_ok"), bool):
            raise RuntimeError("trusted grader emitted malformed correctness evidence")
        if phase == "confirm":
            for name in ("baseline_elapsed_ns", "baseline_calls",
                         "candidate_elapsed_ns", "candidate_calls"):
                samples = row.get(name)
                if (not isinstance(samples, list) or len(samples) != repeats or
                        any(not isinstance(value, int) or isinstance(value, bool) or value <= 0
                            for value in samples)):
                    raise RuntimeError(
                        f"trusted grader did not emit exactly {repeats} {name} samples")
            expected_orders = ["parent_candidate", "candidate_parent", "candidate_parent",
                               "parent_candidate", "parent_candidate", "candidate_parent"]
            conditions = row.get("board_condition_pairs")
            excluded = row.get("excluded_board_condition_pairs")
            if (row.get("pair_orders") != expected_orders or not isinstance(conditions, list) or
                    len(conditions) != repeats or
                    any(not isinstance(pair, dict) or pair.get("pair_id") != index or
                        pair.get("order") != expected_orders[index] or pair.get("valid") is not True or
                        not isinstance(pair.get("seed"), int) or pair["seed"] < 1 or
                        set(pair.get("measurements", {})) != {"parent", "candidate"} or
                        any(pair["measurements"].get(label) != {
                                "elapsed_ns": row[
                                    "baseline_elapsed_ns" if label == "parent"
                                    else "candidate_elapsed_ns"][index],
                                "calls": row[
                                    "baseline_calls" if label == "parent"
                                    else "candidate_calls"][index],
                                "seed": pair["seed"],
                            } for label in ("parent", "candidate")) or
                        pair.get("before", {}).get("authority") != "driver_ssh_sysfs_procfs" or
                        pair.get("after", {}).get("authority") != "driver_ssh_sysfs_procfs" or
                        pair.get("before", {}).get("returncode") != 0 or
                        pair.get("after", {}).get("returncode") != 0
                        for index, pair in enumerate(conditions))):
                raise RuntimeError("trusted K1 pair ordering/board-condition evidence is invalid")
            maximum_excluded = int((board_environment or {}).get(
                "maximum_invalid_pair_replacements_per_capsule", -1))
            attempts = [*conditions, *(excluded if isinstance(excluded, list) else [])]
            if (not isinstance(excluded, list) or len(excluded) > maximum_excluded or
                    row.get("k1_program_count") != 2 * (repeats + len(excluded)) or
                    sorted(pair.get("attempt_id") for pair in attempts
                           if isinstance(pair, dict)) != list(range(len(attempts))) or
                    any(not isinstance(pair, dict) or pair.get("valid") is not False or
                        not isinstance(pair.get("settle_probes"), list) or
                        not pair["settle_probes"] or pair.get("before") != pair["settle_probes"][-1]
                        for pair in excluded) or
                    any(not isinstance(pair.get("settle_probes"), list) or
                        not pair["settle_probes"] or pair.get("before") != pair["settle_probes"][-1]
                        for pair in conditions)):
                raise RuntimeError("trusted K1 exclusion/replacement evidence is invalid")
        elif (not isinstance(row.get("baseline_cycles"), int) or
              not isinstance(row.get("candidate_cycles"), int) or
              row["baseline_cycles"] <= 0 or row["candidate_cycles"] <= 0):
            raise RuntimeError("trusted Spike screen emitted invalid cycle counts")
        if not all(_is_sha256(row.get(name)) for name in (
                "baseline_code_sha256", "candidate_code_sha256")):
            raise RuntimeError("trusted grader emitted a malformed code SHA-256")
        expected_code_authority = ("measured_k1_kernel_object_text_section"
                                   if phase == "confirm"
                                   else "compiled_kernel_object_text_section")
        if row.get("code_digest_authority") != expected_code_authority:
            raise RuntimeError("trusted grader code digest is not executable .text authority")
        if phase == "confirm":
            if row.get("timing_authority") != "spacemit_k1_elapsed_ns_div_completed_calls":
                raise RuntimeError("trusted grader timing authority is not K1 silicon")
            if repeats != 6:
                raise RuntimeError("trusted grader did not preserve the frozen balanced repeat count")
        elif row.get("screen_authority") != "spike_rv64gcv_mcycle_trusted_harness":
            raise RuntimeError("trusted screen authority is not Spike mcycle")


class Broker:
    def __init__(self, *, workspace: Path, public: Path, space_path: Path,
                 runner_path: Path, grader_path: Path, ledger: Path) -> None:
        self.workspace = workspace.resolve()
        self.public = public.resolve()
        self.space_path = space_path.resolve()
        self.runner = _load_module("cpu_host_trusted_beam", runner_path.resolve())
        self.grader = _load_module("cpu_host_trusted_grader", grader_path.resolve())
        self.space = yaml.safe_load(self.space_path.read_text(encoding="utf-8"))
        self.repeats = int(self.space["measurement_repeats"])
        self.budget = dict(self.space["budget"])
        group_counts: dict[str, int] = {}
        for action in self.space["actions"]:
            group = str(action["group"])
            group_counts[group] = group_counts.get(group, 0) + 1
        if "maximum_screen_candidate_evaluations" not in self.budget:
            remaining = sum(group_counts.values())
            maximum = 0
            for group_size in sorted(group_counts.values()):
                maximum += remaining
                remaining -= group_size
            self.budget["maximum_screen_candidate_evaluations"] = maximum
        confirmation_families = self.space.get("confirmation_families") or (
            "contraction", "elementwise_map", "reduction", "movement_layout",
            "fusion_epilogue", "runtime_parallel")
        capsules = len(confirmation_families) * int(
            self.space["confirmation_samples_per_family"])
        requests = (len(group_counts) + 1) * int(self.space["confirmation_width"]) * 2
        self.budget.setdefault("maximum_confirmation_requests", requests)
        self.budget.setdefault("confirmation_package_builds", requests * 2)
        runtime_capsules = (int(self.space["confirmation_samples_per_family"])
                            if "runtime_parallel" in confirmation_families else 0)
        self.budget.setdefault(
            "confirmation_compiler_invocations", requests * (capsules + runtime_capsules) * 2)
        self.budget.setdefault("confirmation_spike_checks", requests * capsules * 2)
        self.broker_started_ns = time.monotonic_ns()
        self.search_started_ns: int | None = None
        self.k1_programs_used = 0
        self.screen_evaluations_used = 0
        self.confirmation_requests_used = 0
        self.package_builds_used = 0
        self.compiler_invocations_used = 0
        self.spike_checks_used = 0
        self.evaluation_multiplicity: dict[str, int] = {}
        self.ledger = ledger.resolve()
        self.observations = self.ledger / "observations"
        self.observations.mkdir(parents=True, exist_ok=False)
        self.receipts = self.ledger / "receipts"
        self.receipts.mkdir()
        self.requests = self.ledger / "requests"
        self.requests.mkdir()
        self.private_corpus = self.ledger / "private_corpus"
        self.private_corpus.mkdir()
        self.submission_snapshot = self.ledger / "submission_snapshot"
        self.prebuilt_snapshot = self.ledger / "prebuilt_search_package"
        self.samples: dict[tuple[str, str], tuple[list[dict[str, Any]], bytes]] = {}
        self.public_rows_all: list[dict[str, Any]] = []
        self.private_samples: dict[tuple[str, str], tuple[list[dict[str, Any]], dict[str, str]]] = {}
        self.private_shape_secret = secrets.token_bytes(32)
        for split in ("train", "validation"):
            source = self.public / f"{split}.jsonl"
            rows = _jsonl(source)
            self.public_rows_all.extend(rows)
            if any(row.get("split") != split for row in rows):
                raise ValueError(f"trusted public {split} file contains another split")
            phases = ("screen", "confirm") if split == "train" else ("confirm",)
            for phase in phases:
                count = int(self.space[
                    "screen_samples_per_family" if phase == "screen"
                    else "confirmation_samples_per_family"])
                selected = self.runner.select_semantic_sample(
                    rows, per_family=count,
                    families=((list(self.space.get("confirmation_families", ())) or None)
                              if phase == "confirm" else None))
                encoded = b"".join(
                    (json.dumps(row, sort_keys=True) + "\n").encode("utf-8")
                    for row in selected)
                self.samples[phase, split] = selected, encoded
        self.all_public = [
            *self.samples["screen", "train"][0],
            *self.samples["confirm", "validation"][0],
        ]
        self.index: dict[str, Any] = {
            "version": 1,
            "authority": "trusted_spacemit_k1_outside_agent_sandbox",
            "heldout_opened": False,
            "space_sha256": _sha256(self.space_path),
            "public_split_sha256": {
                split: _sha256(self.public / f"{split}.jsonl")
                for split in ("train", "validation")
            },
            "measurement_repeats": self.repeats,
            "budget": {
                "k1_program_limit": int(self.budget["k1_program_invocations"]),
                "screen_evaluation_limit": int(
                    self.budget["maximum_screen_candidate_evaluations"]),
                "confirmation_request_limit": int(
                    self.budget["maximum_confirmation_requests"]),
                "package_build_limit": int(self.budget["confirmation_package_builds"]),
                "compiler_invocation_limit": int(
                    self.budget["confirmation_compiler_invocations"]),
                "spike_check_limit": int(self.budget["confirmation_spike_checks"]),
                "planning_upper_search_seconds": float(
                    self.budget["planning_upper_search_seconds"]),
                "reserved_presearch_seconds": float(
                    self.budget["reserved_agent_seconds"]),
                "screen_candidate_stage_cap_seconds": (
                    float(self.budget["planning_upper_spike_screen_seconds"]) /
                    int(self.budget["maximum_screen_candidate_evaluations"])),
                "package_build_stage_cap_seconds": float(
                    self.budget["planning_upper_seconds_per_confirmation_package_build"]),
                "compiler_invocation_stage_cap_seconds": float(
                    self.budget["planning_upper_seconds_per_confirmation_compiler_invocation"]),
                "spike_check_stage_cap_seconds": float(
                    self.budget["planning_upper_seconds_per_confirmation_spike_check"]),
                "k1_program_stage_cap_seconds": float(
                    self.budget["planning_upper_seconds_per_k1_program"]),
                "k1_programs_used": 0,
                "screen_evaluations_used": 0,
                "confirmation_requests_used": 0,
                "package_builds_used": 0,
                "compiler_invocations_used": 0,
                "spike_checks_used": 0,
                "search_started_monotonic_ns": None,
                "presearch_deadline_exceeded": False,
                "stage_cap_exceeded": False,
                "deadline_exceeded": False,
            },
            "submission_tree_sha256": None,
            "prebuilt_package_sha256": None,
            "private_prebuild": None,
            "private_shape_corpus": None,
            "evaluations": {},
            "terminal_receipts": {},
        }
        self._write_index()

    def _materialize_private_samples(self) -> None:
        if self.private_samples:
            return
        records: dict[str, Any] = {
            "authority": "controller_private_after_compiler_snapshot",
            "secret_hex": self.private_shape_secret.hex(), "splits": {},
        }
        for (phase, split), (public_rows, _encoded) in sorted(self.samples.items()):
            private_rows = [_private_capsule(
                row, secret=self.private_shape_secret, phase=phase, split=split)
                for row in public_rows]
            aliases = {str(private["id"]): str(public["id"])
                       for public, private in zip(public_rows, private_rows, strict=True)}
            path = self.private_corpus / f"{phase}_{split}.jsonl"
            path.write_text("".join(json.dumps(row, sort_keys=True) + "\n"
                                    for row in private_rows), encoding="utf-8")
            self.private_samples[phase, split] = private_rows, aliases
            records["splits"][f"{phase}:{split}"] = {
                "path": str(path.relative_to(self.ledger)), "sha256": _sha256(path),
                "count": len(private_rows), "aliases": aliases,
            }
        self.index["private_shape_corpus"] = records
        self._write_index()

    def _write_index(self) -> None:
        temporary = self.ledger / "index.json.tmp"
        temporary.write_text(json.dumps(self.index, indent=2, sort_keys=True) + "\n",
                             encoding="utf-8")
        os.replace(temporary, self.ledger / "index.json")

    def parse_request(self, request: dict[str, Any]) -> dict[str, Any]:
        """Validate one agent message and derive its controller-owned evaluation identity."""
        if request.get("version") != 1:
            raise ValueError("trusted evaluator request must be version 1")
        split = str(request.get("split", ""))
        phase = str(request.get("phase", ""))
        if (phase, split) not in self.samples:
            raise ValueError("trusted evaluator phase/split is not exposed")
        expected_repeats = 1 if phase == "screen" else self.repeats
        if int(request.get("repeats", -1)) != expected_repeats:
            raise ValueError(
                f"trusted evaluator {phase} requires exactly {expected_repeats} measurements")
        policy_path = _regular_bounded(
            _inside(Path(str(request.get("policy", ""))), self.workspace),
            maximum_bytes=256 * 1024, label="candidate policy")
        parent_path = _regular_bounded(
            _inside(Path(str(request.get("parent_policy", ""))), self.workspace),
            maximum_bytes=256 * 1024, label="parent policy")
        capsules_path = _regular_bounded(
            _inside(Path(str(request.get("capsules", ""))), self.workspace),
            maximum_bytes=16 * 1024 * 1024, label="candidate capsule request")
        candidate = _validate_candidate(policy_path, self.space)
        parent = _validate_candidate(parent_path, self.space)
        _validate_parent_child(parent, candidate)
        expected_rows, expected_bytes = self.samples[phase, split]
        if capsules_path.read_bytes() != expected_bytes:
            raise ValueError("requested capsules differ from the exact trusted semantic sample")
        key = f"{parent['candidate_sha256']}:{candidate['candidate_sha256']}:{split}:{phase}"
        binding = {
            "version": 1, "split": split, "phase": phase,
            "repeats": expected_repeats,
            "policy": str(request.get("policy", "")),
            "parent_policy": str(request.get("parent_policy", "")),
            "capsules": str(request.get("capsules", "")),
            "parent_candidate_sha256": parent["candidate_sha256"],
            "candidate_sha256": candidate["candidate_sha256"],
            "parent_policy_sha256": _sha256(parent_path),
            "policy_sha256": _sha256(policy_path),
            "capsules_sha256": _sha256(capsules_path),
        }
        return {"split": split, "phase": phase, "expected_repeats": expected_repeats,
                "policy_path": policy_path, "parent_path": parent_path,
                "capsules_path": capsules_path, "candidate": candidate, "parent": parent,
                "expected_rows": expected_rows, "expected_bytes": expected_bytes,
                "evaluation_key": key, "parsed_request": binding}

    def handle(self, request: dict[str, Any], *, parsed: dict[str, Any] | None = None
               ) -> tuple[bytes, dict[str, Any]]:
        parsed = self.parse_request(request) if parsed is None else parsed
        split, phase = parsed["split"], parsed["phase"]
        expected_repeats = int(parsed["expected_repeats"])
        policy_path, parent_path = parsed["policy_path"], parsed["parent_path"]
        capsules_path = parsed["capsules_path"]
        candidate, parent = parsed["candidate"], parsed["parent"]
        expected_rows = parsed["expected_rows"]
        key = str(parsed["evaluation_key"])
        cached = self.index["evaluations"].get(key)
        if isinstance(cached, dict):
            artifact = self.ledger / str(cached["observations"])
            if cached.get("policy_sha256") != _sha256(policy_path):
                raise ValueError("candidate digest was reused with different policy bytes")
            if cached.get("parent_policy_sha256") != _sha256(parent_path):
                raise ValueError("parent digest was reused with different policy bytes")
            data = artifact.read_bytes()
            if _sha256(artifact) != cached.get("observations_sha256"):
                raise RuntimeError("cached trusted observations changed after evaluation")
            multiplicity = self.evaluation_multiplicity.get(
                key, int(cached.get("request_multiplicity", 1))) + 1
            self.evaluation_multiplicity[key] = multiplicity
            cached["request_multiplicity"] = multiplicity
            self._write_index()
            return data, {"evaluation_key": key, "cache_hit": True,
                          "multiplicity": multiplicity,
                          "parsed_request": parsed["parsed_request"],
                          "response_sha256": hashlib.sha256(data).hexdigest()}

        submission = self.workspace / "submission"
        if not (submission / "manifest.yaml").is_file():
            raise ValueError("submission/manifest.yaml must exist before trusted policy search")
        search = submission / "search"
        if search.exists() and any(search.rglob("*")):
            raise ValueError(
                "submission/search must remain absent until trusted search convergence")
        _reject_symlinks(submission)
        if self.search_started_ns is None:
            presearch_seconds = (time.monotonic_ns() - self.broker_started_ns) / 1e9
            if presearch_seconds > float(self.budget["reserved_agent_seconds"]):
                self.index["budget"]["presearch_deadline_exceeded"] = True
                self._write_index()
                raise RuntimeError(
                    "compiler package was not ready inside the frozen pre-search reserve")
            _require_real_build_manifest(submission, self.space)
            self.search_started_ns = time.monotonic_ns()
            self.index["budget"]["search_started_monotonic_ns"] = self.search_started_ns
            self._write_index()
        submission_digest = _tree_digest(submission)
        first_digest = self.index.get("submission_tree_sha256")
        if first_digest is None:
            _reject_presearch_files(submission)
            _reject_public_specialization(submission, self.public_rows_all)
            shutil.copytree(submission, self.submission_snapshot, symlinks=False)
            snapshot_digest = _tree_digest(self.submission_snapshot)
            if snapshot_digest != submission_digest:
                shutil.rmtree(self.submission_snapshot)
                raise ValueError("submission changed while the trusted snapshot was captured")
            self.index["submission_tree_sha256"] = snapshot_digest
            prebuild_started = time.monotonic_ns()
            prebuild = self.grader.prepare_prebuilt_search_package(
                submission=self.submission_snapshot,
                destination=self.prebuilt_snapshot,
                build_override=list(self.space["search_package"]["private_build_override"]),
            )
            prebuild["controller_wall_ns"] = time.monotonic_ns() - prebuild_started
            self.index["private_prebuild"] = prebuild
            self.index["prebuilt_package_sha256"] = _tree_digest(self.prebuilt_snapshot)
            self._write_index()
        elif first_digest != submission_digest:
            raise ValueError("submission source changed after trusted search began")
        self._materialize_private_samples()
        measured_rows, private_aliases = self.private_samples[phase, split]

        started = time.monotonic_ns()
        assert self.search_started_ns is not None
        elapsed_seconds = (started - self.search_started_ns) / 1e9
        if elapsed_seconds > float(self.budget["planning_upper_search_seconds"]):
            raise RuntimeError("trusted search exceeded its frozen planning wall budget")
        if phase == "screen":
            if self.screen_evaluations_used + 1 > int(
                    self.budget["maximum_screen_candidate_evaluations"]):
                raise RuntimeError("trusted Spike screen would exceed the frozen candidate budget")
            self.screen_evaluations_used += 1
            self.index["budget"]["screen_evaluations_used"] = self.screen_evaluations_used
            self._write_index()
            overall_deadline = self.search_started_ns + int(
                float(self.budget["planning_upper_search_seconds"]) * 1e9)
            stage_deadline = started + int(
                float(self.budget["planning_upper_spike_screen_seconds"]) /
                int(self.budget["maximum_screen_candidate_evaluations"]) * 1e9)
            try:
                with _wall_deadline(min(overall_deadline, stage_deadline)):
                    rows = self.grader.evaluate_public_policy_spike(
                        submission=self.prebuilt_snapshot, capsules=measured_rows,
                        parent=parent, candidate=candidate, public_rows=self.all_public)
            except TimeoutError:
                if stage_deadline <= overall_deadline:
                    self.index["budget"]["stage_cap_exceeded"] = True
                else:
                    self.index["budget"]["deadline_exceeded"] = True
                self._write_index()
                raise
        else:
            requests = 1
            package_builds = 2
            runtime_rows = sum(row.get("family") == "runtime_parallel" for row in expected_rows)
            compiler_invocations = (len(expected_rows) + runtime_rows) * 2
            spike_checks = len(expected_rows) * 2
            maximum_programs = len(expected_rows) * 2 * (
                self.repeats + int(self.space["board_environment"][
                    "maximum_invalid_pair_replacements_per_capsule"]))
            proposed = {
                "confirmation_requests_used": (
                    self.confirmation_requests_used + requests,
                    int(self.budget["maximum_confirmation_requests"])),
                "package_builds_used": (
                    self.package_builds_used + package_builds,
                    int(self.budget["confirmation_package_builds"])),
                "compiler_invocations_used": (
                    self.compiler_invocations_used + compiler_invocations,
                    int(self.budget["confirmation_compiler_invocations"])),
                "spike_checks_used": (
                    self.spike_checks_used + spike_checks,
                    int(self.budget["confirmation_spike_checks"])),
                "k1_programs_used": (
                    self.k1_programs_used + maximum_programs,
                    int(self.budget["k1_program_invocations"])),
            }
            exceeded = [name for name, (used, limit) in proposed.items() if used > limit]
            if exceeded:
                raise RuntimeError(
                    f"trusted confirmation would exceed frozen stage budgets: {exceeded}")
            self.confirmation_requests_used += requests
            self.package_builds_used += package_builds
            self.compiler_invocations_used += compiler_invocations
            self.spike_checks_used += spike_checks
            for name, value in (
                    ("confirmation_requests_used", self.confirmation_requests_used),
                    ("package_builds_used", self.package_builds_used),
                    ("compiler_invocations_used", self.compiler_invocations_used),
                    ("spike_checks_used", self.spike_checks_used)):
                self.index["budget"][name] = value
            self._write_index()
            deadline = self.search_started_ns + int(
                float(self.budget["planning_upper_search_seconds"]) * 1e9)
            stage_caps = {
                "package_build": float(
                    self.budget["planning_upper_seconds_per_confirmation_package_build"]),
                "compiler_invocation": float(
                    self.budget["planning_upper_seconds_per_confirmation_compiler_invocation"]),
                "spike_check": float(
                    self.budget["planning_upper_seconds_per_confirmation_spike_check"]),
                "k1_program": float(
                    self.budget["planning_upper_seconds_per_k1_program"]),
            }
            try:
                with _wall_deadline(deadline):
                    rows = self.grader.evaluate_public_policy_k1(
                        submission=self.prebuilt_snapshot, capsules=measured_rows,
                        parent=parent, candidate=candidate, repeats=self.repeats,
                        public_rows=self.all_public,
                        board_environment=dict(self.space["board_environment"]),
                        deadline_monotonic_ns=deadline, stage_caps=stage_caps)
            except TimeoutError as exc:
                if "stage cap" in str(exc):
                    self.index["budget"]["stage_cap_exceeded"] = True
                else:
                    self.index["budget"]["deadline_exceeded"] = True
                self._write_index()
                raise
            actual_programs = sum(int(row.get("k1_program_count", 0)) for row in rows)
            if not 0 < actual_programs <= maximum_programs:
                raise RuntimeError("trusted grader emitted an invalid K1 program count")
            self.k1_programs_used += actual_programs
            if self.k1_programs_used > int(self.budget["k1_program_invocations"]):
                raise RuntimeError("trusted confirmation exceeded its frozen K1 program budget")
            self.index["budget"]["k1_programs_used"] = self.k1_programs_used
            self._write_index()
        for row in rows:
            private_id = str(row.get("capsule_id", ""))
            if private_id not in private_aliases:
                raise RuntimeError("trusted grader emitted an unknown private capsule identity")
            row["capsule_id"] = private_aliases[private_id]
        _validate_observations(
            rows, expected_rows, expected_repeats, phase,
            board_environment=(dict(self.space["board_environment"])
                               if phase == "confirm" else None))
        if any(row.get("parent_candidate_sha256") != parent["candidate_sha256"] or
               row.get("candidate_sha256") != candidate["candidate_sha256"] for row in rows):
            raise RuntimeError("trusted grader observation is not bound to the requested parent/child")
        data = b"".join(
            (json.dumps(row, sort_keys=True) + "\n").encode("utf-8") for row in rows)
        relative = (f"observations/{parent['candidate_sha256']}_to_"
                    f"{candidate['candidate_sha256']}_{split}_{phase}.jsonl")
        artifact = self.ledger / relative
        artifact.write_bytes(data)
        self.index["evaluations"][key] = {
            "parent_candidate_sha256": parent["candidate_sha256"],
            "candidate_sha256": candidate["candidate_sha256"],
            "split": split,
            "phase": phase,
            "policy_sha256": _sha256(policy_path),
            "parent_policy_sha256": _sha256(parent_path),
            "capsules_sha256": _sha256(capsules_path),
            "private_capsules_sha256": self.index["private_shape_corpus"]["splits"][
                f"{phase}:{split}"]["sha256"],
            "private_capsule_ids": [str(row["id"]) for row in measured_rows],
            "observations": relative,
            "observations_sha256": _sha256(artifact),
            "measurement_repeats": expected_repeats,
            "request_multiplicity": 1,
            "wall_ns": time.monotonic_ns() - started,
        }
        self.evaluation_multiplicity[key] = 1
        self._write_index()
        return data, {"evaluation_key": key, "cache_hit": False, "multiplicity": 1,
                      "parsed_request": parsed["parsed_request"],
                      "response_sha256": hashlib.sha256(data).hexdigest()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--public", type=Path, required=True)
    parser.add_argument("--space", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--grader", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--poll", type=float, default=0.2)
    args = parser.parse_args(argv)
    broker = Broker(workspace=args.workspace, public=args.public, space_path=args.space,
                    runner_path=args.runner, grader_path=args.grader, ledger=args.ledger)
    channel = args.workspace.resolve() / ".trusted_search_channel"
    channel.mkdir(parents=True, exist_ok=True)
    (channel / "READY").write_text("ready\n", encoding="utf-8")
    original_parent = os.getppid()
    seen: set[str] = set()
    while True:
        for request_path in sorted(channel.glob("req_*.json")):
            if request_path.name in seen:
                continue
            seen.add(request_path.name)
            request_id = request_path.stem.removeprefix("req_")
            safe_id = (request_id if re.fullmatch(r"[A-Za-z0-9_]{1,160}", request_id)
                       else "invalid_" + hashlib.sha256(
                           request_path.name.encode()).hexdigest()[:24])
            response = channel / f"resp_{safe_id}.jsonl"
            error = channel / f"error_{safe_id}.txt"
            started_ns = time.monotonic_ns()
            request_sha256: str | None = None
            request_artifact: Path | None = None
            parsed: dict[str, Any] | None = None
            terminal: dict[str, Any]
            try:
                _regular_bounded(
                    request_path, maximum_bytes=64 * 1024, label="trusted broker request")
                request_bytes = request_path.read_bytes()
                request_sha256 = hashlib.sha256(request_bytes).hexdigest()
                request_artifact = broker.requests / f"{safe_id}.json"
                request_temporary = request_artifact.with_suffix(".json.tmp")
                request_temporary.write_bytes(request_bytes)
                os.replace(request_temporary, request_artifact)
                if safe_id != request_id:
                    raise ValueError("trusted broker request id is invalid")
                request_value = json.loads(request_bytes)
                parsed = broker.parse_request(request_value)
                response_bytes, association = broker.handle(request_value, parsed=parsed)
                response.write_bytes(response_bytes)
                if association["response_sha256"] != _sha256(response):
                    raise RuntimeError("trusted response digest changed during publication")
                terminal = {
                    "version": 1, "authority": "driver_trusted_search_broker",
                    "request_id": request_id, "status": "pass",
                    "request_sha256": request_sha256,
                    "request_artifact": str(request_artifact.relative_to(broker.ledger)),
                    **association,
                    "wall_ns": time.monotonic_ns() - started_ns,
                }
            except Exception as exc:  # fail one request closed without losing the private ledger
                message = f"trusted broker: {type(exc).__name__}: {exc}"
                error.write_text(message + "\n", encoding="utf-8")
                declared_class = _broker_failure_class(exc)
                terminal = {
                    "version": 1, "authority": "driver_trusted_search_broker",
                    "request_id": request_id, "safe_request_id": safe_id, "status": "fail",
                    "request_sha256": request_sha256, "response_sha256": None,
                    "failure_class": declared_class,
                    "error": message[:4000], "wall_ns": time.monotonic_ns() - started_ns,
                }
                if request_artifact is not None and request_artifact.is_file():
                    terminal["request_artifact"] = str(
                        request_artifact.relative_to(broker.ledger))
                if parsed is not None:
                    terminal.update({
                        "evaluation_key": parsed["evaluation_key"], "cache_hit": False,
                        "multiplicity": broker.evaluation_multiplicity.get(
                            str(parsed["evaluation_key"]), 0) + 1,
                        "parsed_request": parsed["parsed_request"],
                    })
            encoded = (json.dumps(terminal, indent=2, sort_keys=True) + "\n").encode("utf-8")
            ledger_receipt = broker.receipts / f"{safe_id}.json"
            ledger_temporary = ledger_receipt.with_suffix(".json.tmp")
            ledger_temporary.write_bytes(encoded)
            os.replace(ledger_temporary, ledger_receipt)
            broker.index["terminal_receipts"][safe_id] = {
                "path": str(ledger_receipt.relative_to(broker.ledger)),
                "sha256": _sha256(ledger_receipt), "status": terminal["status"],
                "evaluation_key": terminal.get("evaluation_key"),
                "cache_hit": terminal.get("cache_hit"),
                "multiplicity": terminal.get("multiplicity"),
                "response_sha256": terminal.get("response_sha256"),
            }
            broker._write_index()
            channel_receipt = channel / f"receipt_{safe_id}.json"
            channel_temporary = channel_receipt.with_suffix(".json.tmp")
            channel_temporary.write_bytes(encoded)
            os.replace(channel_temporary, channel_receipt)
            (channel / f"done_{safe_id}").write_text("done\n", encoding="utf-8")
        if os.getppid() != original_parent or (channel / "STOP").exists():
            broker.index["broker_terminal"] = {
                "status": "stopped", "start_monotonic_ns": broker.broker_started_ns,
                "end_monotonic_ns": time.monotonic_ns(),
            }
            broker.index["broker_terminal"]["wall_ns"] = (
                broker.index["broker_terminal"]["end_monotonic_ns"] - broker.broker_started_ns)
            broker._write_index()
            break
        time.sleep(args.poll)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
