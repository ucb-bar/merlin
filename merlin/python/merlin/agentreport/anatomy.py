"""Everything one run did, on one clock — the input to a single-run anatomy figure.

The cross-run figures answer "how do these runs compare". This answers "what does one run actually
look like", which needs finer material than a comparison can carry: every tool call with its
category and duration, every verdict with its per-capsule detail, and the token curve, all rebased
onto one wall clock.

THE CATEGORIES ARE THE POINT. A capsule-bench agent spends its time on two different things, and the
run's shape is mostly the ratio between them:

``feedback``
    asking whether it is right yet -- the self-check and the oracle. The agent cannot do this itself;
    it waits.
``development``
    authoring and inspecting -- edits, reads, greps, builds, the Merlin tools.

Measured on the run this was written for: authoring 46 calls costing about one second in total,
against 80 feedback calls costing 12,218 s. Development is free and feedback is the entire clock,
which is not what the call COUNTS suggest (281 inspect calls against 69 self-checks) and is exactly
why both are carried here.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from merlin.agentreport.spans import SpanSet

#: Category -> (display label, whether it is feedback rather than development). Ordered as a stacked
#: band reads, development at the bottom.
CATEGORIES: dict[str, tuple[str, bool]] = {
    "author": ("writing compiler code", False),
    "inspect": ("reading files / searching", False),
    "merlin_tool": ("Merlin helper (ISA encoder, CCA check)", False),
    "build": ("compiling", False),
    "shell": ("other shell commands", False),
    "selfcheck": ("self-check: \u201cis my compiler right yet?\u201d", True),
    "oracle": ("RTL simulation job (runs in background)", True),
}

#: Grading-plane ids -> what a plane actually means. The verdict speaks in these ids and nobody
#: outside the project can read them; a figure that repeats them is not explaining anything.
PLANE_PLAIN: dict[str, str] = {
    "lanes": "which hardware unit ran the work",
    "model": "whole-model numerical result",
    "model_execution": "proof the whole model really executed",
    "build": "the submission did not build",
    "parse": "the emitted IR did not parse",
    "trace_check": "the instruction trace did not match",
    "integrity": "the submission read something it must not",
}

#: Command substrings that identify a category. Harness vocabulary, not target facts: these are the
#: names the harness stages, so they change with the harness and never with the hardware.
_AUTHOR_TOOLS = frozenset({"edit", "write", "multiedit", "notebookedit"})
_SELFCHECK = ("agent_selfcheck.py",)
_ORACLE = ("simjob.py",)
_MERLIN_TOOLS = ("isa_tools", "cca_contract", "action_catalog")
_BUILD = ("pytest", "make ", "cmake", "ninja", "mlir-opt", "clang", "gcc ", "llc ")
_INSPECT = ("cat ", "sed -n", "grep", "rg ", "ls ", "find ", "head ", "tail ", "wc ", "nl ")


def categorize(tool: str, command: str) -> str:
    """Which kind of work one completed call was. Checked feedback-first: a self-check invoked
    THROUGH a shell is feedback, whatever the shell looks like."""
    name = (tool or "").strip().lower()
    text = (command or "").lower()
    if any(k in text for k in _SELFCHECK):
        return "selfcheck"
    if any(k in text for k in _ORACLE):
        return "oracle"
    if name in _AUTHOR_TOOLS:
        return "author"
    if any(k in text for k in _MERLIN_TOOLS):
        return "merlin_tool"
    if any(k in text for k in _BUILD):
        return "build"
    if any(k in text for k in _INSPECT):
        return "inspect"
    return "shell"


@dataclass
class Call:
    t_s: float
    duration_s: float
    category: str


@dataclass
class Verdict:
    t_s: float
    n_passed: int
    n_capsules: int
    name: str = ""
    #: capsule -> status, so a reader can see WHICH capsule turned green when.
    per_capsule: dict = field(default_factory=dict)
    #: capsule -> deepest tier that passed, for the same reason.
    per_capsule_tier: dict = field(default_factory=dict)


@dataclass
class Blocked:
    """A capsule that never passed, and the plane its verdict actually turned on.

    Worth carrying separately because "the agent kept working and the score did not move" and "the
    remaining capsules could not move" look identical on a step plot and mean opposite things. On the
    run this was written for, two of the four unresolved capsules PASS their RTL tier and fail on a
    lane contract or a routing rule, and a third is marked incomplete because a lane was never
    measured on this path at all -- none of which is a numerical failure the agent could fix by
    trying harder."""

    capsule: str
    status: str
    plane: str = ""
    category: str = ""
    deepest_tier_passed: str = ""
    detail: str = ""


@dataclass
class SimEvent:
    """One oracle evaluation of one capsule, placed on the run's clock.

    The grader records how long each evaluation took but not when it began; the per-capsule
    ``run_manifest.yaml`` records ``created_at``, written when the capsule FINISHES. The start is
    therefore inferred as ``created_at - adapter_wall_s`` and is exact only to the extent that the
    adapter's own wall covers the evaluation -- which it does in the functional lane, where the
    adapter really did wait. Marked ``inferred_start`` so a reader knows the ends are measured and
    the starts are reconstructed."""

    capsule: str
    tier: str
    engine: str
    start_s: float
    end_s: float
    sim_active_s: float
    build_s: float = 0.0
    oracle_wait_s: float = 0.0
    workers: int | None = None
    grade: str = ""
    inferred_start: bool = True


@dataclass
class GradeCost:
    """What one grade spent, split into building the agent's package and running it.

    The distinction is the whole answer to "why does a self-check take six minutes when the
    simulation is twenty seconds": every grade recompiles and re-emits every capsule from scratch.
    Measured on one run, a spike-only grade spends 408 s building to run 18.6 s of simulation."""

    grade: str
    n_capsules: int
    build_s: float
    sim_active_s: float
    oracle_wait_s: float
    adapter_wall_s: float


@dataclass
class Anatomy:
    run_id: str = ""
    target: str = ""
    arm: str = ""
    model: str = ""
    wall_s: float = 0.0
    calls: list[Call] = field(default_factory=list)
    verdicts: list[Verdict] = field(default_factory=list)
    token_curve: list[dict] = field(default_factory=list)
    cost_curve: list[dict] = field(default_factory=list)
    blocked: list[Blocked] = field(default_factory=list)
    sim_events: list[SimEvent] = field(default_factory=list)
    grade_costs: list[GradeCost] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"run_id": self.run_id, "target": self.target, "arm": self.arm, "model": self.model,
                "wall_s": self.wall_s, "calls": [asdict(c) for c in self.calls],
                "verdicts": [asdict(v) for v in self.verdicts],
                "token_curve": self.token_curve, "cost_curve": self.cost_curve,
                "blocked": [asdict(b) for b in self.blocked],
                "sim_events": [asdict(e) for e in self.sim_events],
                "grade_costs": [asdict(g) for g in self.grade_costs], "notes": self.notes}


_TIERS = ("L0", "L1", "L2", "L3", "L4")


def _deepest_tier(tiers) -> str:
    if not isinstance(tiers, dict):
        return ""
    got = [t for t in _TIERS if tiers.get(t) == "pass"]
    return got[-1] if got else ""


def run_started_at(run_dir: Path) -> float | None:
    """Absolute epoch of the run's start, the one origin every series here shares.

    Without it each series anchors on its own first event, and series that begin at different times
    silently slide against each other -- measured on one run, the simulator strip sat 5.4 minutes to
    the left of the verdicts it belongs to, which is enough to put a grade's simulations before the
    grade that requested them."""
    import yaml
    for name, path in (("timing", run_dir / "qa_loop_summary.yaml"),
                       ("timing", run_dir / "qa_loop_state.yaml")):
        if not path.is_file():
            continue
        try:
            doc = yaml.safe_load(path.read_text(encoding="utf-8", errors="ignore")) or {}
        except Exception:  # noqa: BLE001
            continue
        stamp = ((doc.get(name) or {}).get("started_at")
                 if isinstance(doc.get(name), dict) else None)
        if isinstance(stamp, str) and stamp:
            from datetime import datetime
            try:
                return datetime.fromisoformat(stamp).timestamp()
            except ValueError:
                continue
    env = run_dir / "environment.yaml"
    if env.is_file():
        try:
            doc = yaml.safe_load(env.read_text(encoding="utf-8", errors="ignore")) or {}
            stamp = doc.get("started_at")
            if isinstance(stamp, str) and stamp:
                from datetime import datetime
                return datetime.fromisoformat(stamp).timestamp()
        except Exception:  # noqa: BLE001
            pass
    return None


def read_verdicts(run_dir: Path, t0: float | None = None) -> list[Verdict]:
    """Every verdict this run produced, on the run's own clock.

    Ordered and timed by the grader's own ``graded_at`` where present, else the file mtime. The two
    are not the same kind of fact and the caller is told which, but for ORDERING both work."""
    history = run_dir / "qa_history"
    if not history.is_dir():
        return []
    rows: list[tuple[float, Path, dict]] = []
    for path in sorted(history.glob("verdict*.json")):
        try:
            doc = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
        except (ValueError, OSError):
            continue
        if not isinstance(doc, dict) or not isinstance(doc.get("n_capsules"), int):
            continue
        stamp = doc.get("graded_at")
        when: float | None = None
        if isinstance(stamp, str) and stamp:
            from datetime import datetime
            try:
                when = datetime.fromisoformat(stamp.replace("Z", "+00:00")).timestamp()
            except ValueError:
                when = None
        rows.append((when if when is not None else path.stat().st_mtime, path, doc))
    if not rows:
        return []
    rows.sort(key=lambda r: r[0])
    if t0 is None:
        t0 = rows[0][0]
    out = []
    for when, path, doc in rows:
        per = {}
        tier = {}
        for c in doc.get("per_capsule") or []:
            if isinstance(c, dict) and c.get("capsule"):
                per[str(c["capsule"])] = str(c.get("status") or "")
                tier[str(c["capsule"])] = _deepest_tier(c.get("tiers"))
        out.append(Verdict(t_s=when - t0, n_passed=int(doc.get("n_passed") or 0),
                           n_capsules=int(doc["n_capsules"]), name=path.stem,
                           per_capsule=per, per_capsule_tier=tier))
    return out


def read_blocked(run_dir: Path) -> list[Blocked]:
    """Capsules unresolved at the last verdict, with the plane each actually turned on."""
    history = run_dir / "qa_history"
    if not history.is_dir():
        return []
    latest = None
    for path in sorted(history.glob("verdict*.json")):
        try:
            doc = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
        except (ValueError, OSError):
            continue
        if isinstance(doc, dict) and isinstance(doc.get("n_capsules"), int):
            latest = doc
    if latest is None:
        return []
    out = []
    for c in latest.get("per_capsule") or []:
        if not isinstance(c, dict) or c.get("status") == "pass":
            continue
        out.append(Blocked(capsule=str(c.get("capsule") or ""), status=str(c.get("status") or ""),
                           plane=str(c.get("failure_plane") or ""),
                           category=str(c.get("failure_category") or ""),
                           deepest_tier_passed=_deepest_tier(c.get("tiers")),
                           detail=str(c.get("failure_detail") or "")[:400]))
    return out


def read_sim_events(run_dir: Path, t0: float | None = None) -> list[SimEvent]:
    """Every oracle evaluation across every grade this run produced, on one clock.

    ``t0`` anchors the clock; when absent the earliest evaluation becomes zero. Evaluations from
    different grades share the axis deliberately -- the question this answers is how much simulator
    work overlapped, and that is a property of the whole run rather than of one grade."""
    import yaml
    work = run_dir / "_qa_work"
    if not work.is_dir():
        return []
    out: list[SimEvent] = []
    for grade in sorted(work.glob("runs_*")):
        for path in grade.rglob("capsule_result.json"):
            try:
                doc = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
            except (ValueError, OSError):
                continue
            if not isinstance(doc, dict):
                continue
            manifest = path.parent / "run_manifest.yaml"
            created = None
            if manifest.is_file():
                try:
                    created = (yaml.safe_load(manifest.read_text(encoding="utf-8",
                                                                 errors="ignore")) or {}).get("created_at")
                except Exception:  # noqa: BLE001
                    created = None
            if not isinstance(created, str):
                continue
            from datetime import datetime
            try:
                end = datetime.fromisoformat(created).timestamp()
            except ValueError:
                continue
            for tier, entry in (doc.get("tiers") or {}).items():
                if not isinstance(entry, dict):
                    continue
                timing = entry.get("timing")
                if not isinstance(timing, dict) or not timing.get("sim_active_s"):
                    continue
                wall = float(timing.get("adapter_wall_s") or timing.get("sim_active_s") or 0.0)
                conc = entry.get("concurrency")
                out.append(SimEvent(
                    capsule=str(doc.get("capsule") or path.parent.name), tier=str(tier),
                    engine=str(entry.get("engine") or ""), start_s=end - wall, end_s=end,
                    sim_active_s=float(timing.get("sim_active_s") or 0.0),
                    build_s=float(timing.get("build_s") or 0.0),
                    oracle_wait_s=float(timing.get("oracle_wait_s") or 0.0),
                    workers=(int(conc["workers"]) if isinstance(conc, dict)
                             and isinstance(conc.get("workers"), int) else None),
                    grade=grade.name))
    if not out:
        return []
    base = t0 if t0 is not None else min(e.start_s for e in out)
    for e in out:
        e.start_s -= base
        e.end_s -= base
    return sorted(out, key=lambda e: e.start_s)


def read_grade_costs(run_dir: Path) -> list[GradeCost]:
    """Build / simulate / queue, summed per grade."""
    work = run_dir / "_qa_work"
    if not work.is_dir():
        return []
    out = []
    for grade in sorted(work.glob("runs_*")):
        build = sim = wait = wall = 0.0
        n = 0
        for path in grade.rglob("capsule_result.json"):
            try:
                doc = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
            except (ValueError, OSError):
                continue
            if not isinstance(doc, dict):
                continue
            seen = False
            for entry in (doc.get("tiers") or {}).values():
                timing = entry.get("timing") if isinstance(entry, dict) else None
                if not isinstance(timing, dict):
                    continue
                build += timing.get("build_s") or 0.0
                sim += timing.get("sim_active_s") or 0.0
                wait += timing.get("oracle_wait_s") or 0.0
                wall += timing.get("adapter_wall_s") or 0.0
                seen = True
            n += seen
        if n:
            out.append(GradeCost(grade.name, n, build, sim, wait, wall))
    return out


def build_anatomy(run_dir: Path, spanset: SpanSet, *, run_id: str, target: str, arm: str,
                  model: str, token_curve=None, cost_curve=None) -> Anatomy:
    """Assemble one run's full record. Spans and verdicts keep their own clocks; both start at 0."""
    a = Anatomy(run_id=run_id, target=target, arm=arm, model=model, wall_s=spanset.wall_s)
    for sp in spanset.spans:
        a.calls.append(Call(sp.start_s, sp.duration_s, categorize(sp.kind, sp.detail)))
    # One origin for every series. The transcript already starts at zero, so anchoring the grader's
    # two series on the run's own start puts all three on the same clock.
    started = run_started_at(run_dir)
    a.verdicts = read_verdicts(run_dir, started)
    a.blocked = read_blocked(run_dir)
    a.sim_events = read_sim_events(run_dir, started)
    a.grade_costs = read_grade_costs(run_dir)
    if started is None:
        a.notes.append("this run recorded no start time, so the grader's series are anchored on "
                       "their own first event and may sit a few minutes off the transcript's clock")
    a.token_curve = list(token_curve or [])
    a.cost_curve = list(cost_curve or [])
    if a.verdicts and a.wall_s > 0:
        drift = abs((a.verdicts[-1].t_s) - a.wall_s) / a.wall_s
        if drift > 0.25:
            a.notes.append(
                f"the verdict clock and the transcript clock disagree by {drift:.0%} "
                f"({a.verdicts[-1].t_s / 3600:.1f} h of grading against {a.wall_s / 3600:.1f} h of "
                f"transcript); they are two independent records and are drawn on their own scales")
    return a
