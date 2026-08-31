"""Fail-closed boundary between the completed functional Arm-4 run and performance.

The performance campaign is allowed to consume exactly one content-addressed functional
submission.  It copies that submission into a run-private workspace, makes the copy read-only,
records the functional fork, and executes every untrusted package entrypoint in a credential-free,
networkless bwrap sandbox.  Pure helpers in this module make the refusal paths testable without
starting a simulator.
"""
from __future__ import annotations

import contextlib
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Mapping, Sequence

import yaml

from merlin.benchharness import hash_tree
from merlin.perf.fork import ForkPoint, candidate_states, check_invariants, fork_from
from merlin.targetgen import oot_runner
from merlin.targetgen.oracle_schedule import CapsuleState, Verdict
from merlin.targetgen.sandbox import bwrap as BW
from merlin.targetgen.sandbox import toolchain as TC
from merlin.targetgen.sandbox.answer_surfaces import answer_surfaces
from merlin.targetgen.target_experiment import TargetExperiment


class CampaignGateError(RuntimeError):
    """A phase-boundary or completion condition was not proved."""


@dataclass(frozen=True)
class FunctionalRun:
    run_dir: Path
    submission_dir: Path
    run_id: str
    digest: str
    public_capsules: int
    hidden_capsules: int
    public_score: dict
    hidden_score: dict
    frozen_at: str


@dataclass(frozen=True)
class PackageSandboxPolicy:
    argv: tuple[str, ...]
    coverage_gap: tuple[str, ...]
    required_tools: tuple[TC.ToolProbe, ...]
    workspace: Path
    package: Path
    target_experiment: TargetExperiment


def _mapping_file(path: Path, *, yaml_file: bool = False) -> dict:
    if not path.is_file():
        raise CampaignGateError(f"required campaign evidence is absent: {path}")
    try:
        doc = yaml.safe_load(path.read_text(encoding="utf-8")) if yaml_file else json.loads(
            path.read_text(encoding="utf-8"))
    except (OSError, ValueError, yaml.YAMLError) as exc:
        raise CampaignGateError(f"campaign evidence is unreadable at {path}: {exc}") from exc
    if not isinstance(doc, dict):
        raise CampaignGateError(f"campaign evidence must be a mapping: {path}")
    return doc


def _full_ratio(value: object, *, label: str) -> int:
    parts = str(value or "").split("/")
    if len(parts) != 2:
        raise CampaignGateError(f"{label} must state an explicit passed/total ratio")
    try:
        passed, total = (int(x) for x in parts)
    except ValueError as exc:
        raise CampaignGateError(f"{label} has a malformed passed/total ratio: {value!r}") from exc
    if total <= 0:
        raise CampaignGateError(f"{label} must be non-vacuous, not {passed}/{total}")
    if passed != total:
        raise CampaignGateError(f"{label} is incomplete: {passed}/{total}")
    return total


def _validate_score(score: dict, *, label: str, expected: int) -> None:
    n_capsules = score.get("n_capsules")
    n_passed = score.get("n_passed")
    rows = score.get("per_capsule")
    if (not isinstance(n_capsules, int) or isinstance(n_capsules, bool)
            or not isinstance(n_passed, int) or isinstance(n_passed, bool)
            or n_capsules <= 0 or n_passed != n_capsules or n_capsules != expected):
        raise CampaignGateError(
            f"{label} score is not a full non-vacuous {expected}/{expected} grade")
    if score.get("functional_pass") != 1 or score.get("gradeable") is not True:
        raise CampaignGateError(f"{label} score is not gradeable and functionally complete")
    if score.get("integrity_status") != "clean" or score.get("integrity_exempt") is not False:
        raise CampaignGateError(f"{label} score did not pass the integrity gate")
    if not isinstance(rows, list) or len(rows) != expected:
        raise CampaignGateError(f"{label} score has no complete per-capsule evidence")
    names: set[str] = set()
    for row in rows:
        if not isinstance(row, dict) or not row.get("capsule") or row.get("status") != "pass":
            raise CampaignGateError(f"{label} contains a capsule without a passing verdict")
        name = str(row["capsule"])
        if name in names:
            raise CampaignGateError(f"{label} repeats capsule {name!r}")
        names.add(name)
        tiers = row.get("tiers") or {}
        if tiers.get("L2") != "pass" or tiers.get("L3") != "pass":
            raise CampaignGateError(f"{label} capsule {name!r} did not earn both L2 and L3")


def _validate_clean_run(environment: dict, summary: dict) -> None:
    if environment.get("sandbox") != "bwrap":
        raise CampaignGateError("functional run was not executed in the required bwrap sandbox")
    if environment.get("isolation_violations") != []:
        raise CampaignGateError("functional run lacks a clean isolation-violation audit")
    inputs = environment.get("bundle_input_snapshot")
    snapshot_digest = inputs.get("content_sha256") if isinstance(inputs, dict) else None
    if (not isinstance(inputs, dict) or inputs.get("version") != 2
            or not isinstance(snapshot_digest, str) or len(snapshot_digest) != 64
            or any(c not in "0123456789abcdef" for c in snapshot_digest)
            or not isinstance(inputs.get("n_files"), int) or isinstance(inputs.get("n_files"), bool)
            or inputs["n_files"] <= 0
            or not isinstance(inputs.get("n_bytes"), int) or isinstance(inputs.get("n_bytes"), bool)
            or inputs["n_bytes"] <= 0):
        raise CampaignGateError(
            "functional run lacks a complete immutable bundle-input snapshot v2 record")
    mask = environment.get("golden_mask_selftest")
    if (not isinstance(mask, dict) or mask.get("leaked_answer_files") != []
            or int(mask.get("n_answer_files_masked") or 0) <= 0):
        raise CampaignGateError("functional run did not prove a non-vacuous clean answer mask")
    if summary.get("converged") is not True:
        raise CampaignGateError("functional QA loop did not converge")
    rounds = summary.get("rounds")
    if not isinstance(rounds, list) or not rounds:
        raise CampaignGateError("functional QA evidence has no completed round")
    for row in rounds:
        if (not isinstance(row, dict) or row.get("answer_access_clean") is not True
                or row.get("audit_hits") != []):
            raise CampaignGateError("functional QA round failed the answer-access audit")
    finalize = summary.get("finalize")
    if (not isinstance(finalize, dict)
            or finalize.get("answer_access_clean") is not True
            or finalize.get("audit_hits") != []
            or finalize.get("regrade_all_pass") is not True):
        raise CampaignGateError("functional finalization did not pass clean regrade and audit gates")


def _has_symlink(root: Path) -> Path | None:
    if root.is_symlink():
        return root
    for path in root.rglob("*"):
        if path.is_symlink():
            return path
    return None


def _digest_excluded_path(root: Path) -> Path | None:
    """Return executable state omitted by the functional harness's historical tree hash."""
    excluded = {"build", "__pycache__", ".git"}
    for path in root.rglob("*"):
        if excluded & set(path.relative_to(root).parts):
            return path
    return None


def inspect_functional_run(run_root: Path, run_id: str, expected_digest: str) -> FunctionalRun:
    """Validate one explicitly named, fully graded Arm-4 functional run.

    No directory search is performed: the caller supplies both the run ID and the whole-submission
    SHA-256, and all independently recorded digests plus the bytes on disk must agree with it.
    """
    if not run_id:
        raise CampaignGateError("an explicit functional run id is required")
    if Path(run_id).name != run_id or run_id in (".", ".."):
        raise CampaignGateError("functional run id must be a simple directory name")
    if (len(expected_digest) != 64 or expected_digest.lower() != expected_digest
            or any(c not in "0123456789abcdef" for c in expected_digest)):
        raise CampaignGateError("functional submission digest must be an explicit lowercase SHA-256")

    arm_root = (Path(run_root) / "merlin_assisted").resolve()
    run_dir = arm_root / run_id
    if run_dir.is_symlink() or not run_dir.is_dir() or arm_root not in run_dir.resolve().parents:
        raise CampaignGateError(f"explicit functional run does not resolve safely: {run_dir}")
    run_dir = run_dir.resolve()
    submission = run_dir / "submission"
    if not submission.is_dir():
        raise CampaignGateError(f"functional submission is absent: {submission}")
    linked = _has_symlink(submission)
    if linked is not None:
        raise CampaignGateError(f"functional submission contains a live symlink: {linked}")
    excluded = _digest_excluded_path(submission)
    if excluded is not None:
        raise CampaignGateError(
            f"functional submission contains a digest-excluded path: {excluded}")

    environment = _mapping_file(run_dir / "environment.yaml", yaml_file=True)
    summary = _mapping_file(run_dir / "qa_loop_summary.yaml", yaml_file=True)
    freeze = _mapping_file(run_dir / "freeze.json")
    manifest = _mapping_file(run_dir / "run_manifest.yaml", yaml_file=True)
    public_score = _mapping_file(run_dir / "grading_public" / "score_capsule.json")
    hidden_score = _mapping_file(run_dir / "grading_hidden" / "score_capsule.json")

    if environment.get("run_id") != run_id or manifest.get("run_id") != run_id:
        raise CampaignGateError("functional evidence names a different run id")
    if not str(environment.get("bundle_id") or "").startswith("merlin_assisted_rtlchecks_"):
        raise CampaignGateError("functional run is not from the Arm-4 RTL-checks bundle")
    _validate_clean_run(environment, summary)
    if (manifest.get("integrity_status") != "clean" or manifest.get("integrity_exempt") is not False
            or manifest.get("gradeable") is not True):
        raise CampaignGateError("functional run did not pass integrity and gradeability gates")
    public = manifest.get("public_dev") or {}
    hidden = manifest.get("hidden") or {}
    if public.get("functional_pass") != 1 or hidden.get("functional_pass") != 1:
        raise CampaignGateError("functional run did not pass both public and hidden grades")
    if public.get("highest_tier") != "L3":
        raise CampaignGateError("functional public run did not reach the required L3 tier")
    n_public = _full_ratio(public.get("passed"), label="public functional grade")
    n_hidden = _full_ratio(hidden.get("passed"), label="hidden functional grade")
    _validate_score(public_score, label="public functional grade", expected=n_public)
    _validate_score(hidden_score, label="hidden functional grade", expected=n_hidden)
    if {str(r["capsule"]) for r in public_score["per_capsule"]} & {
            str(r["capsule"]) for r in hidden_score["per_capsule"]}:
        raise CampaignGateError("public and hidden grades reuse a capsule identity")
    if freeze.get("workspace_mutable_after_freeze") is not False or not freeze.get("frozen_at"):
        raise CampaignGateError("functional freeze did not record an immutable workspace and timestamp")

    observed = hash_tree(submission)["sha256"]
    recorded = {
        "requested": expected_digest,
        "run_manifest": manifest.get("submission_sha256"),
        "freeze": freeze.get("submission_sha256"),
        "freeze_recheck": freeze.get("submission_sha256_recheck"),
        "submission_tree": observed,
    }
    if any(value != expected_digest for value in recorded.values()):
        raise CampaignGateError(
            f"explicit functional submission digest does not match every record: {recorded}")
    return FunctionalRun(
        run_dir=run_dir,
        submission_dir=submission.resolve(),
        run_id=run_id,
        digest=expected_digest,
        public_capsules=n_public,
        hidden_capsules=n_hidden,
        public_score=public_score,
        hidden_score=hidden_score,
        frozen_at=str(freeze.get("frozen_at") or manifest.get("frozen_at") or ""),
    )


def materialize_perf_workspace(record: FunctionalRun, perf_root: Path) -> Path:
    """Copy the exact functional package into a new performance-only workspace."""
    perf_root = Path(perf_root).resolve()
    workspace = perf_root / "workspace"
    snapshot = workspace / "submission"
    observed = materialize_readonly_tree(record.submission_dir, snapshot)
    if observed != record.digest:
        raise CampaignGateError(
            f"copied performance submission digest {observed} does not match {record.digest}")
    return snapshot


def materialize_readonly_tree(source: Path, snapshot: Path) -> str:
    """Copy a symlink-free, fully hashed tree once and make the destination read-only."""
    source = Path(source).resolve()
    snapshot = Path(snapshot).resolve()
    if not source.is_dir():
        raise CampaignGateError(f"immutable campaign source is not a directory: {source}")
    if snapshot.exists() or snapshot.is_symlink():
        raise CampaignGateError(f"immutable campaign snapshot already exists: {snapshot}")
    linked = _has_symlink(source)
    if linked is not None:
        raise CampaignGateError(f"immutable campaign source contains a live symlink: {linked}")
    excluded = _digest_excluded_path(source)
    if excluded is not None:
        raise CampaignGateError(f"immutable campaign source contains a digest-excluded path: {excluded}")
    before = hash_tree(source)["sha256"]
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, snapshot, symlinks=False)
    linked = _has_symlink(snapshot)
    if linked is not None:
        raise CampaignGateError(f"immutable campaign snapshot contains a symlink: {linked}")
    observed = hash_tree(snapshot)["sha256"]
    if observed != before:
        raise CampaignGateError(
            f"immutable campaign copy changed during materialization: {before} != {observed}")
    for path in sorted(snapshot.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        if path.is_dir():
            path.chmod(0o555)
        elif path.is_file():
            path.chmod(0o555 if path.stat().st_mode & 0o111 else 0o444)
    snapshot.chmod(0o555)
    return str(observed)


def functional_fork(record: FunctionalRun) -> ForkPoint:
    """Convert the public+hidden L2/L3 evidence into the immutable Phase-F fork record."""
    states: list[CapsuleState] = []
    tiers: set[str] = set()
    for score in (record.public_score, record.hidden_score):
        for row in score["per_capsule"]:
            passed = {str(tier): Verdict(str(status), record.digest)
                      for tier, status in (row.get("tiers") or {}).items() if status == "pass"}
            tiers.update(passed)
            states.append(CapsuleState(str(row["capsule"]), record.digest, verdicts=passed))
    if not states or not tiers:
        raise CampaignGateError("cannot fork from an empty functional verdict set")
    return fork_from(states, tier_order=sorted(tiers), digest=record.digest,
                     recorded_at=record.frozen_at or None)


def check_fork(fork: ForkPoint, snapshot: Path):
    """Check the copied compiler's current bytes against every Phase-F invariant."""
    digest = hash_tree(Path(snapshot))["sha256"]
    states = candidate_states(fork, digest=digest)
    return check_invariants(fork, states, provenance=fork.provenance)


def _remove_agent_home_mounts(argv: Sequence[str]) -> list[str]:
    """Remove the agent-launch credential bind from a package-only bwrap prefix.

    ``base_argv`` also serves paid agent launches and therefore binds ``~/.claude``.  A compiled
    submission never needs that state.  Strip both that bind and its nested projects mask, then clear
    the inherited environment so API/provider credentials cannot enter through environment variables.
    """
    home_claude = str(Path(os.path.expanduser("~/.claude")))
    out: list[str] = []

    def is_agent_home_path(value: str) -> bool:
        path = Path(value)
        home = Path(home_claude)
        return path == home or home in path.parents

    i = 0
    while i < len(argv):
        if argv[i] in ("--bind", "--ro-bind", "--bind-try", "--ro-bind-try") and i + 2 < len(argv):
            if is_agent_home_path(argv[i + 2]):
                i += 3
                continue
        if argv[i] == "--tmpfs" and i + 1 < len(argv) and is_agent_home_path(argv[i + 1]):
            i += 2
            continue
        out.append(argv[i])
        i += 1
    if any(is_agent_home_path(value) for value in out if value.startswith("/")):
        raise CampaignGateError("package sandbox still exposes the agent credential directory")
    return out


def package_sandbox_policy(te: TargetExperiment, workspace: Path,
                           package: Path) -> PackageSandboxPolicy:
    """Derive a credential-free, answer-closed bwrap policy for untrusted package entrypoints."""
    workspace = Path(workspace).resolve()
    package = Path(package).resolve()
    if not workspace.is_dir() or not package.is_dir() or workspace == package:
        raise CampaignGateError("performance sandbox workspace and copied package must be distinct directories")
    # Empty bundle is intentional: the package sees its input files and the derived toolchain, not the
    # experiment repository.  The policy-only escape avoids requiring an agent-input snapshot when no
    # bundle grants exist and does not expose any live input itself.
    argv = BW.base_argv(workspace, {}, _policy_test_live_inputs=True)
    argv = _remove_agent_home_mounts(argv)
    argv += ["--unshare-net", "--clearenv", "--setenv", "HOME", "/tmp",
             "--setenv", "XDG_RUNTIME_DIR", "/tmp/.xdg"]
    argv += TC.toolchain_binds(te)
    argv = _remove_agent_home_mounts(argv)
    surfaces = answer_surfaces(te)
    argv = BW.apply_answer_masks(argv, surfaces)
    argv += ["--ro-bind", str(package), str(package)]
    gaps = tuple(str(surface.path) for surface in BW.coverage_gap(argv, surfaces))
    if gaps:
        raise CampaignGateError(f"package sandbox exposes derived answer surfaces: {gaps}")
    probes = tuple(TC.required_tool_probes(te))
    if not probes:
        raise CampaignGateError("package sandbox derives zero required tool probes")
    return PackageSandboxPolicy(tuple(argv), gaps, probes, workspace, package, te)


def run_tool_probes(policy: PackageSandboxPolicy, *, timeout: int = 60) -> list[dict]:
    """Run every descriptor-derived tool probe in the exact package sandbox."""
    rows: list[dict] = []
    for probe in policy.required_tools:
        proc = subprocess.run(
            [*policy.argv, "bash", "-c", TC.sandbox_env(policy.target_experiment, policy.workspace)
             + probe.cmd],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        row = {"label": probe.label, "returncode": proc.returncode,
               "stdout": (proc.stdout or "")[-400:], "stderr": (proc.stderr or "")[-400:]}
        rows.append(row)
        if proc.returncode != 0:
            raise CampaignGateError(
                f"required sandbox tool probe {probe.label!r} failed with rc={proc.returncode}")
    if len(rows) != len(policy.required_tools):
        raise CampaignGateError("not every required package-sandbox tool probe ran")
    return rows


def _inside(path: Path, root: Path) -> bool:
    path = path.resolve()
    root = root.resolve()
    return path == root or root in path.parents


@contextlib.contextmanager
def boxed_entrypoints(policy: PackageSandboxPolicy) -> Iterator[None]:
    """Route oot_runner's untrusted package execution through ``policy`` for this serial campaign."""
    original_entrypoint = oot_runner.run_entrypoint
    original_build = oot_runner.build_package

    def refuse_build(pkg, *, timeout: int = 1800) -> None:
        build = pkg.manifest.get("build") or {}
        if any(build.get(key) for key in ("configure", "command")):
            raise CampaignGateError(
                "Arm-4 performance package declares an untrusted build step; no host build is allowed")

    def run_boxed(pkg, name: str, input_mlir: Path, output_json: Path | None = None,
                  *, timeout: int = 600) -> subprocess.CompletedProcess:
        if pkg.directory.resolve() != policy.package:
            raise CampaignGateError("capsule runner attempted to execute a package outside the snapshot")
        input_mlir = Path(input_mlir).resolve()
        output_json = Path(output_json).resolve() if output_json is not None else None
        if not _inside(input_mlir, policy.workspace):
            raise CampaignGateError("untrusted package input is outside the performance workspace")
        if output_json is not None and not _inside(output_json, policy.workspace):
            raise CampaignGateError("untrusted package output is outside the performance workspace")
        argv = oot_runner._resolve_argv(pkg, name, input_mlir, output_json)
        if oot_runner._needs_interpreter(pkg, argv):
            argv = [sys.executable, *argv]
        shell = TC.sandbox_env(policy.target_experiment, policy.workspace) + 'exec "$@"'
        return subprocess.run(
            [*policy.argv, "--chdir", str(policy.package), "bash", "-c", shell,
             "perf-package", *argv],
            capture_output=True,
            text=True,
            timeout=timeout,
        )

    oot_runner.build_package = refuse_build
    oot_runner.run_entrypoint = run_boxed
    try:
        yield
    finally:
        oot_runner.run_entrypoint = original_entrypoint
        oot_runner.build_package = original_build


def completion_report(results: Sequence[Mapping], expected: Mapping[str, Sequence[str]]) -> dict:
    """Count every expected Arm-4 cell without letting an incomplete run look like an empty pass."""
    expected_cells = {(str(kernel), str(sim)) for kernel, sims in expected.items() for sim in sims}
    if not expected_cells:
        raise CampaignGateError("performance campaign has zero expected Arm-4 cells")
    if any(not sims for sims in expected.values()):
        raise CampaignGateError("performance campaign contains a kernel with zero expected simulators")
    by_kernel: dict[str, Mapping] = {}
    for row in results:
        kernel = str(row.get("kernel") or "")
        if not kernel or kernel in by_kernel:
            raise CampaignGateError(f"performance results repeat or omit a kernel identity: {kernel!r}")
        by_kernel[kernel] = row

    reported = correct = cycles_measured = 0
    failed = 0
    for kernel, sim in sorted(expected_cells):
        row = by_kernel.get(kernel) or {}
        arm = ((row.get("approaches") or {}).get("arm4") or {})
        per_sim = arm.get("per_sim") or {}
        cell = per_sim.get(sim)
        if not isinstance(cell, Mapping):
            continue
        reported += 1
        is_correct = cell.get("correct") is True
        cycles = cell.get("cycles")
        has_cycles = (isinstance(cycles, (int, float)) and not isinstance(cycles, bool)
                      and cycles > 0)
        correct += int(is_correct)
        cycles_measured += int(has_cycles)
        failed += int(not (is_correct and has_cycles))
    missing = len(expected_cells) - reported
    counts = {"expected": len(expected_cells), "reported": reported, "correct": correct,
              "cycles_measured": cycles_measured, "failed": failed, "missing": missing,
              "complete": not missing and not failed}
    return counts


def completion_counts(results: Sequence[Mapping], expected: Mapping[str, Sequence[str]]) -> dict:
    """Require a correct, positive-cycle result for every expected Arm-4 kernel/simulator cell."""
    counts = completion_report(results, expected)
    if not counts["complete"]:
        raise CampaignGateError(
            f"Arm-4 performance reported {counts['reported']} of {counts['expected']} expected cells; "
            f"{counts['failed']} reported cell(s) failed correctness or positive-cycle measurement")
    return counts
