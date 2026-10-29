"""Opt-in bounded author studies; measurement stays outside the author clock.

This policy changes lifecycle and reporting, never numerical admission. Private
snapshots have no grading credit and are never fed back to an author.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import threading
import time
from pathlib import Path

SECTION = "study_protocol"


def validate(policy):
    value = policy.get(SECTION)
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) != {
        "version", "author_s", "evaluation", "scope", "snapshot_seconds"
    }:
        raise ValueError("study protocol requires explicit version, clock, evaluation, scope and snapshots")
    if (value["version"] != 1 or type(value["author_s"]) is not int
            or value["author_s"] != policy["duration_s"] - policy["drain_s"]):
        raise ValueError("study author clock must equal duration minus drain")
    if value["evaluation"] != "deferred" or value["scope"] != "selected_capsules":
        raise ValueError("study protocol requires deferred selected-capsule evaluation")
    points = value["snapshot_seconds"]
    if (not isinstance(points, list) or not points or points[0] != 0
            or any(type(t) is not int or not 0 <= t <= value["author_s"] for t in points)
            or points != sorted(set(points)) or points[-1] != value["author_s"]):
        raise ValueError("study snapshots must be ordered unique offsets spanning the author clock")
    return value


def selected():
    from . import joint_control
    state = joint_control.selected()
    return validate(state["policy"]) if state else None


def prompt(policy):
    spec = validate(policy)
    if spec is None:
        return ""
    return (
        f"\nBounded study: {spec['author_s'] / 3600:g} hours of authoring. "
        "Optimize the common selected public capsule suite, including its connected programs. "
        "Additional private whole-model campaigns are outside this study. Numerical and model "
        "feedback is provisional; final functional and physical evaluation happens after freeze. "
        "Private transfer sources and results are unavailable during authoring.\n"
    )


def _write(path, value):
    temporary = path.with_suffix(".next")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _source_hash(root):
    from .run_inputs import outside_snapshot
    digest = hashlib.sha256()
    for path in sorted(Path(root).rglob("*")):
        relative = path.relative_to(root)
        if outside_snapshot(relative.parts):
            continue
        if path.is_symlink():
            raise ValueError("study compiler source contains a symlink")
        if path.is_file():
            digest.update(relative.as_posix().encode())
            digest.update(b"\0")
            digest.update(path.read_bytes())
            digest.update(b"\0")
    return digest.hexdigest()


def snapshot(source, destination, *, offset_s, started_at, now=None):
    """Copy one coherent source version or explicitly refuse a moving source."""
    from .run_inputs import BUILD_STATE, SNAPSHOT_IGNORED, graded_source_hash, strip_build_state
    source, destination = Path(source), Path(destination)
    if destination.exists() or destination.is_symlink():
        raise ValueError("study snapshot already exists")
    if source.is_symlink() or not source.is_dir():
        raise ValueError("study source must be an ordinary directory")
    before = _source_hash(source)
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        shutil.copytree(source, destination, ignore=shutil.ignore_patterns(*SNAPSHOT_IGNORED, *BUILD_STATE))
        strip_build_state(destination)
        digest = graded_source_hash(destination)
        if before != digest or _source_hash(source) != digest:
            raise ValueError("compiler changed while taking study snapshot")
    except BaseException:
        if destination.is_dir() and not destination.is_symlink():
            shutil.rmtree(destination)
        raise
    for member in destination.rglob("*"):
        if member.is_file():
            member.chmod(member.stat().st_mode & ~0o222)
    observed = time.time() if now is None else now
    return dict(
        offset_s=offset_s, observed_elapsed_s=observed - started_at,
        source=str(destination.resolve()), compiler_sha256=digest,
        digest_basis="graded_source_hash", status="frozen", author_visible=False,
    )


class SnapshotMonitor:
    def __init__(self, workspace, run_dir, state):
        self.source = Path(workspace) / "submission"
        self.root = Path(run_dir) / "study_snapshots"
        self.started = state["started_at_epoch"]
        self.spec = validate(state["policy"])
        self.stopped = threading.Event()
        self.root.mkdir(mode=0o700, exist_ok=True)
        self.index = self.root / "index.json"
        if self.index.exists():
            self.record = json.loads(self.index.read_text())
            if self.record["started_at_epoch"] != self.started or self.record["protocol"] != self.spec:
                raise ValueError("study snapshot monitor changed on resume")
        else:
            self.record = dict(schema="merlin.study.snapshots.v1", started_at_epoch=self.started,
                               protocol=self.spec, author_visible=False, snapshots=[])
        # The first checkpoint is synchronous, before the first author invocation.
        self.tick()
        self.thread = threading.Thread(target=self._watch, name="study-snapshots", daemon=True)
        self.thread.start()

    def tick(self):
        existing = {row["offset_s"] for row in self.record["snapshots"]}
        elapsed = time.time() - self.started
        for offset in self.spec["snapshot_seconds"]:
            if offset in existing or elapsed < offset:
                continue
            try:
                row = snapshot(self.source, self.root / str(offset), offset_s=offset, started_at=self.started)
            except (OSError, ValueError, RuntimeError) as error:
                row = dict(offset_s=offset, observed_elapsed_s=elapsed, status="unavailable",
                           reason=f"{type(error).__name__}: {error}")
            self.record["snapshots"].append(row)
            _write(self.index, self.record)

    def _watch(self):
        while not self.stopped.wait(1):
            self.tick()

    def close(self):
        self.stopped.set()
        self.thread.join(timeout=30)
        if self.thread.is_alive():
            raise RuntimeError("study snapshot copy did not finish")
        self.tick()


def _metrics_roster(context, *, public_roots=None, contract=None):
    """Read the same admitted public core, scale and program owners as grading."""
    from merlin.targetgen.capsule_common import discover_capsules
    from merlin.targetgen.target_experiment import load_target_experiment

    from .feedback import public_programs, public_scalability

    experiment = load_target_experiment(context.descriptor)
    roots = experiment.graded_roots() if public_roots is None else public_roots
    core = discover_capsules(roots, labels={"public", "dev"},
                             contract=contract or context.repo / "merlin" / "contract")
    excluded = set(experiment.effective_exclusions(cap["name"] for cap in core))
    suites = {"core": sorted(cap["name"] for cap in core if cap["name"] not in excluded)}
    scalability = public_scalability.selected(context)
    suites["scalability"] = sorted(scalability[1]["cases"]) if scalability else []
    programs = public_programs.selected(context)
    suites["programs"] = sorted(programs[2]) if programs else []
    names = [name for values in suites.values() for name in values]
    if not names or any(not isinstance(name, str) or not name for name in names):
        raise ValueError("study metrics require a nonempty actual public roster")
    if len(names) != len(set(names)):
        raise ValueError("study metric case identities must be unique across suites")
    return suites


def write_metrics_context(workspace, run_dir, state, snapshot_index, *, public_roster, environment):
    """Persist the bounded public measurement context before any author starts.

    Only roster names, timestamps and source identities are recorded. No hidden
    source, oracle value or initial ELF is synthesized. Resume retains the same
    initial snapshot and original author clock; optional separately qualified
    initial-case evidence already present in the context is preserved.
    """
    workspace, run_dir = Path(workspace), Path(run_dir)
    protocol = validate(state["policy"])
    if protocol is None or state["started_at_epoch"] is None:
        raise ValueError("study metrics require a started native study clock")
    names = [name for values in public_roster.values() for name in values]
    if not names or len(names) != len(set(names)):
        raise ValueError("study metrics require an unambiguous nonempty public roster")
    initial = [row for row in snapshot_index["snapshots"] if row["offset_s"] == 0]
    if len(initial) != 1:
        raise ValueError("study metrics require the original zero-offset observation")
    initial = initial[0]
    seeded = bool(state["policy"].get("compiler_origin") or environment.get("seed_submission"))
    empty = (environment.get("backend_origin") or {}).get("schema") == "merlin.empty_backend_origin.v1"
    origin = "seeded" if seeded else "scratch" if empty else "UNKNOWN"
    context = dict(
        schema="merlin.study.metrics_context.v1", workspace=str(workspace.resolve()),
        run_id=run_dir.name, started_at=state["started_at_epoch"],
        observation_end_at=state["started_at_epoch"] + protocol["author_s"],
        observation_window_kind="declared author clock", author_s=protocol["author_s"],
        backend_origin=origin, public_case_ids=sorted(names), public_cases_by_suite=public_roster,
        initial_source_sha256=initial.get("compiler_sha256") if initial.get("status") == "frozen" else None,
        initial_source_status=initial.get("status"),
        initial_source_observed_elapsed_s=initial.get("observed_elapsed_s"),
        initial_case_elf_sha256={}, source_snapshot_index=str((run_dir / "study_snapshots/index.json").resolve()),
        operator_only=True, author_visible=False, qualification_issued=False,
    )
    path = run_dir / "study-metrics-context.json"
    if path.exists():
        prior = json.loads(path.read_text())
        for key, value in context.items():
            if key == "initial_case_elf_sha256":
                continue
            if prior.get(key) != value:
                raise ValueError("study metrics context changed on resume: " + key)
        return prior
    _write(path, context)
    return context


def start_snapshots(workspace, run_dir, *, context=None, public_roots=None, contract=None, environment=None):
    from . import joint_control
    state = joint_control.selected()
    if state is None or validate(state["policy"]) is None:
        return None
    if state["started_at_epoch"] is None:
        raise ValueError("study snapshots require the started author clock")
    if context is None or environment is None:
        raise ValueError("study metrics require the admitted invocation and run environment")
    roster = _metrics_roster(context, public_roots=public_roots, contract=contract)
    monitor = SnapshotMonitor(workspace, run_dir, state)
    try:
        write_metrics_context(workspace, run_dir, state, monitor.record,
                              public_roster=roster, environment=environment)
    except BaseException:
        monitor.stopped.set()
        monitor.thread.join(timeout=30)
        raise
    return monitor
