"""Find every experiment on this machine and say, with its basis, whether it is running.

Experiments are found in three bounded places:

* managed launch owners: ``<tasks>/<task>/.tmp/<launch>/process/owner.json``
  (``operator.joint_native_owner.v1``), which pin the selection naming the run directory;
* run roots: ``<tasks>/<task>/merlin/out/runs/<target>/<suite>/<run_id>/`` that carry agent rounds;
* explicit extra run directories given by the operator.

An owner record says ``native_runner_started`` forever, so it never decides state alone. The state is
derived from the owner process (pid alive *and* its cmdline equals the recorded argv), the run's own
completion and stop records, and how recently its records changed -- and every state carries the record
that decided it. A new launch is always a new experiment; nothing merges two runs.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

from .io import Reader, clean, epoch, is_dir, is_file, mtime, obj, scan_dir

#: States, most to least active. ``ACTIVE`` is a run without a managed owner whose records still change.
STATES = ("LIVE", "STALLED", "ACTIVE", "INTERRUPTED", "FINISHED", "ENDED", "STORED")

OWNER_SCHEMA = "operator.joint_native_owner.v1"
#: A live run whose records have not changed for this long is shown as STALLED.
DEFAULT_STALE_S = 15 * 60
#: A run without a managed owner counts as ACTIVE while its records changed this recently.
DEFAULT_ACTIVE_S = 5 * 60


def tasks_root(explicit: Path | None = None) -> Path | None:
    """The directory holding every task folder: explicit, ``MERLIN_CONSOLE_TASKS_ROOT``, or derived."""
    if explicit is not None:
        return Path(explicit)
    env = os.environ.get("MERLIN_CONSOLE_TASKS_ROOT")
    if env:
        return Path(env)
    task_dir = os.environ.get("TASK_DIR")
    if task_dir:
        return Path(task_dir).parent
    try:
        from merlin.common.paths import repo_root

        checkout = repo_root().resolve()
    except Exception:  # noqa: BLE001 -- discovery must work without an importable core checkout
        return None
    # <tasks>/<task>/<repo>: the checkout of one task folder.
    return checkout.parent.parent if checkout.parent.parent.name == "tasks" else None


RACY_S = 2.0

#: A run directory is an experiment when it carries agent rounds or an offline construction study.
EXPERIMENT_MARKERS = (
    "rounds",
    "joint_campaign.json",
    "construction_study_result.json",
    "CONSTRUCTION_ACTIONS.json",
)


def is_experiment(run_dir: Path) -> bool:
    return any(os.path.exists(run_dir / marker) for marker in EXPERIMENT_MARKERS)


def _proc_stat(pid: int, proc_root: Path) -> str | None:
    try:
        raw = (proc_root / str(pid) / "stat").read_bytes()
        return raw.decode().rsplit(")", 1)[1].split()[0]
    except (OSError, IndexError, UnicodeError):
        return None


def owner_alive(owner: dict, proc_root: Path = Path("/proc")) -> bool:
    """The recorded process is alive and still runs exactly the recorded argv (pid reuse guard)."""
    pid, expected = owner.get("pid"), owner.get("argv")
    if type(pid) is not int or not isinstance(expected, list) or not expected:
        return False
    state = _proc_stat(pid, proc_root)
    if state is None or state in ("Z", "X"):
        return False
    if Path(str(expected[0])).name == "taskset" and expected[1:2] == ["-c"]:
        expected = expected[3:]
    try:
        raw = (proc_root / str(pid) / "cmdline").read_bytes()
    except OSError:
        return False
    return [x.decode(errors="replace") for x in raw.split(b"\0") if x] == expected


RUNNER_MODULE = "merlin_experiments.phase1"


def runner_processes(proc_root: Path = Path("/proc"), limit: int = 8192) -> dict[str, int]:
    """``run_id -> pid`` for live ``python -m <runner> ... --run-id <id>`` processes (exact argv tokens)."""
    found: dict[str, int] = {}
    for entry in scan_dir(proc_root, limit):
        if not entry.name.isdigit():
            continue
        try:
            raw = (Path(entry.path) / "cmdline").read_bytes()
        except OSError:
            continue
        argv = [x.decode(errors="replace") for x in raw.split(b"\0") if x]
        if "-m" not in argv or "--run-id" not in argv:
            continue
        module_at, run_at = argv.index("-m") + 1, argv.index("--run-id") + 1
        # The interpreter itself runs the module (only interpreter flags precede -m); a wrapper
        # whose own argv merely carries the runner command line is not the runner.
        direct = Path(argv[0]).name.startswith("python") and all(a.startswith("-") for a in argv[1 : module_at - 1])
        if direct and module_at < len(argv) and argv[module_at] == RUNNER_MODULE and run_at < len(argv):
            state = _proc_stat(int(entry.name), proc_root)
            if state not in (None, "Z", "X"):
                found.setdefault(argv[run_at], int(entry.name))
    return found


def workspace_of(run_dir: Path, reader: Reader) -> str | None:
    """The single agent workspace a run's ``environment.yaml`` names (lines, not a YAML parse)."""
    lines = reader.text_tail(run_dir / "environment.yaml", 4 << 20)
    found = set()
    for line in lines:
        stripped = line.strip()
        value = (
            stripped.partition(":")[2].strip()
            if stripped.startswith("workspace_path:")
            else (stripped[2:].strip() if stripped.startswith("- /") else "")
        )
        value = value.strip("'\"")
        if "/agent-workspaces/" in value and value.endswith("/workspace"):
            found.add(value)
    return next(iter(found)) if len(found) == 1 else None


class Discovery:
    """Bounded machine-wide experiment discovery with per-directory change caching."""

    def __init__(
        self,
        root: Path | None = None,
        *,
        tasks: list[Path] | None = None,
        extra_runs: list[Path] | None = None,
        proc_root: Path = Path("/proc"),
        stale_s: float = DEFAULT_STALE_S,
        active_s: float = DEFAULT_ACTIVE_S,
        max_tasks: int = 512,
        max_launches: int = 2048,
        max_runs: int = 512,
    ) -> None:
        self.root = tasks_root(root)
        self.tasks = [Path(t) for t in tasks] if tasks else None
        self.extra_runs = [Path(p) for p in extra_runs or []]
        self.proc_root = Path(proc_root)
        self.stale_s, self.active_s = stale_s, active_s
        self.max_tasks, self.max_launches, self.max_runs = max_tasks, max_launches, max_runs
        self.reader = Reader(budget=64 << 20)
        self._owner_cache: dict[str, tuple[int, list[tuple[Path, dict]]]] = {}
        self._run_cache: dict[str, tuple[int, list[Path]]] = {}
        self._confirmed: set[str] = set()
        self.notices: list[str] = []

    def task_dirs(self) -> list[Path]:
        if self.tasks is not None:
            return self.tasks
        if self.root is None:
            return []
        return [
            Path(e.path)
            for e in scan_dir(self.root, self.max_tasks)
            if e.is_dir(follow_symlinks=False) and not e.name.startswith(".")
        ]

    def _owners(self, task: Path) -> list[tuple[Path, dict]]:
        tmp = task / ".tmp"
        try:
            stamp = os.stat(tmp).st_mtime_ns
        except OSError:
            return []
        cached = self._owner_cache.get(str(tmp))
        if cached and cached[0] == stamp:
            return cached[1]
        found = []
        for entry in scan_dir(tmp, self.max_launches):
            path = Path(entry.path) / "process" / "owner.json"
            if not entry.is_dir(follow_symlinks=False) or not is_file(path):
                continue
            doc = self.reader.document(path, limit=256 << 10)
            if doc and doc.get("schema") == OWNER_SCHEMA:
                found.append((path, doc))
        self._owner_cache[str(tmp)] = (stamp, found)
        return found

    def _runs(self, task: Path, runners: dict[str, int] | None = None) -> list[Path]:
        """Run directories that are experiments, or that a live runner names (still in setup)."""
        out: list[Path] = []
        runs = task / "merlin" / "out" / "runs"
        for target in scan_dir(runs, 64):
            if not target.is_dir(follow_symlinks=False):
                continue
            for suite in scan_dir(Path(target.path), 64):
                if not suite.is_dir(follow_symlinks=False):
                    continue
                suite_path = Path(suite.path)
                try:
                    stamp = os.stat(suite_path).st_mtime_ns
                except OSError:
                    continue
                cached = self._run_cache.get(str(suite_path))
                # A listing taken within a timestamp tick of a change may miss it ("racy" mtime), so a
                # directory changed in the last RACY_S seconds is listed again.
                if not cached or cached[0] != stamp or time.time() - stamp / 1e9 < RACY_S:
                    found = [
                        Path(e.path) for e in scan_dir(suite_path, self.max_runs) if e.is_dir(follow_symlinks=False)
                    ]
                    cached = (stamp, found)
                    self._run_cache[str(suite_path)] = cached
                # A run directory gains its markers after it is created, so membership is re-checked
                # each scan (a few stats per directory); a confirmed experiment stays one.
                for run_dir in cached[1]:
                    key = str(run_dir)
                    if key not in self._confirmed and is_experiment(run_dir):
                        self._confirmed.add(key)
                    if key in self._confirmed or run_dir.name in (runners or {}):
                        out.append(run_dir)
        return out

    def _last_record(self, run_dir: Path, workspace: str | None, owner_path: Path | None) -> float | None:
        probes = [
            run_dir / "rounds",
            run_dir / "qa_history",
            run_dir / "_qa_work",
            run_dir / "oot_commits.jsonl",
            run_dir / "construction_study_result.json",
        ]
        if workspace:
            probes += [
                Path(workspace) / "qa" / "progress.json",
                Path(workspace) / ".qa_channel",
                Path(workspace) / "construction_study" / "events.jsonl",
            ]
        if owner_path is not None:
            probes.append(owner_path.parent / "native.log")
        times = [t for t in map(mtime, probes) if t is not None]
        return max(times) if times else None

    def _interruption(self, run_dir: Path) -> str | None:
        stop = self.reader.document(run_dir / "joint_epoch_stop.json", limit=1 << 20)
        if stop is not None:
            return f"joint_epoch_stop.json ({clean(stop.get('reason') or 'no reason recorded', 80)})"
        rounds = run_dir / "rounds"
        summaries = sorted(e.name for e in scan_dir(rounds, 4096) if e.name.endswith(".codex_summary.json"))
        if summaries:
            doc = obj(self.reader.document(rounds / summaries[-1], limit=1 << 20))
            if doc.get("operator_interrupted") is True:
                reason = clean(doc.get("interruption_reason"), 60) or "no reason"
                return f"{summaries[-1]}: operator interrupted ({reason})"
            if doc.get("timed_out") is True:
                return f"{summaries[-1]}: round timed out"
        return None

    def classify(
        self,
        run_dir: Path,
        owner: dict | None,
        owner_alive_now: bool,
        last_record: float | None,
        now: float,
        runner_pid: int | None = None,
    ) -> tuple[str, str]:
        """``(state, basis)`` from the strongest record available; see the module docstring."""
        age = now - last_record if last_record is not None else None
        if not is_dir(run_dir):
            return "ENDED", "run directory no longer present; only its launch owner record remains"
        if owner_alive_now:
            if age is not None and age > self.stale_s:
                return "STALLED", f"owner pid {owner.get('pid')} alive; no record changed for {int(age // 60)} min"
            return "LIVE", f"owner pid {owner.get('pid')} alive with its recorded argv"
        if runner_pid is not None:
            if age is not None and age > self.stale_s:
                return "STALLED", f"runner pid {runner_pid} alive; no record changed for {int(age // 60)} min"
            return "LIVE", f"runner pid {runner_pid} runs this --run-id (no managed owner)"
        summary = run_dir / "joint_campaign_summary.json"
        if is_file(summary):
            return "FINISHED", "joint_campaign_summary.json present (completion record)"
        study = run_dir / "construction_study_result.json"
        if is_file(study) and not is_dir(run_dir / "rounds"):
            result = obj(self.reader.document(study, limit=4 << 20))
            return (
                "FINISHED",
                f"construction_study_result.json present (status {clean(result.get('status'), 30) or 'not recorded'})",
            )
        reason = self._interruption(run_dir)
        if reason:
            return "INTERRUPTED", reason
        if owner is not None:
            return "INTERRUPTED", f"owner pid {owner.get('pid')} gone without a completion record"
        if age is not None and age <= self.active_s:
            return "ACTIVE", f"no managed owner; records changed {int(age)} s ago"
        return "ENDED", "no managed owner, no completion record, records unchanged"

    def _experiment(
        self,
        run_dir: Path,
        task: Path | None,
        owner_path: Path | None,
        owner: dict | None,
        now: float,
        runners: dict[str, int] | None = None,
    ) -> dict[str, Any]:
        alive = owner_alive(owner, self.proc_root) if owner else False
        workspace = workspace_of(run_dir, self.reader)
        last = self._last_record(run_dir, workspace, owner_path)
        runner = (runners or {}).get(run_dir.name)
        state, basis = self.classify(run_dir, owner, alive, last, now, None if alive else runner)
        parts = run_dir.parts
        target, suite = (parts[-3], parts[-2]) if len(parts) >= 3 else (None, None)
        started = epoch(obj(owner).get("started_at_unix")) or mtime(run_dir / "environment.yaml")
        launch = owner_path.parent.parent if owner_path else None
        return {
            "id": str(run_dir),
            "run_id": run_dir.name,
            "kind": "run",
            "task": str(task) if task else None,
            "task_name": task.name if task else None,
            "run_dir": str(run_dir),
            "workspace": workspace,
            "target": target,
            "suite": suite,
            "launch_dir": str(launch) if launch else None,
            "owner": (
                {
                    "path": str(owner_path),
                    "pid": owner.get("pid"),
                    "alive": alive,
                    "selection": obj(owner.get("selection")).get("path"),
                }
                if owner
                else None
            ),
            "runner_pid": owner.get("pid") if alive else runner,
            "started_at": started,
            "last_record_at": last,
            "state": state,
            "state_basis": basis,
        }

    def scan(self, now: float | None = None) -> list[dict[str, Any]]:
        """Every experiment found, newest activity first within state order."""
        now = time.time() if now is None else now
        self.reader.begin()
        found: dict[str, dict[str, Any]] = {}
        runners = runner_processes(self.proc_root)
        for task in self.task_dirs():
            for owner_path, owner in self._owners(task):
                run = owner.get("native_run_dir")
                if not isinstance(run, str) or not run.startswith("/"):
                    continue
                run_dir = Path(run)
                key = str(run_dir)
                previous = found.get(key)
                # Several launches may name one run directory; the newest launch owns it.
                if previous and (previous.get("started_at") or 0) >= (epoch(owner.get("started_at_unix")) or 0):
                    continue
                found[key] = self._experiment(run_dir, task, owner_path, owner, now, runners)
            for run_dir in self._runs(task, runners):
                if str(run_dir) not in found:
                    found[str(run_dir)] = self._experiment(run_dir, task, None, None, now, runners)
        for run_dir in self.extra_runs:
            if str(run_dir) not in found:
                item = self._experiment(run_dir, None, None, None, now, runners)
                if not is_experiment(run_dir):
                    item.update(kind="reference", state="STORED", state_basis="operator-named record directory")
                found[str(run_dir)] = item
        order = {state: index for index, state in enumerate(STATES)}
        self.notices = sorted(self.reader.notices)
        return sorted(found.values(), key=lambda e: (order[e["state"]], -(e.get("last_record_at") or 0)))


def default_selection(experiments: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The experiment to open first: the most recently active live one, else the most recent record."""
    live = [e for e in experiments if e["state"] in ("LIVE", "STALLED", "ACTIVE")]
    pool = live or experiments
    return max(pool, key=lambda e: e.get("last_record_at") or 0, default=None)


__all__ = ["Discovery", "STATES", "default_selection", "owner_alive", "runner_processes", "tasks_root", "workspace_of"]
