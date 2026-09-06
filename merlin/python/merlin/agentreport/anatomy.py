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
    "author": ("authoring an edit", False),
    "inspect": ("inspecting files", False),
    "merlin_tool": ("Merlin tool", False),
    "build": ("build / compile", False),
    "shell": ("other shell", False),
    "selfcheck": ("self-check", True),
    "oracle": ("oracle job", True),
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
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"run_id": self.run_id, "target": self.target, "arm": self.arm, "model": self.model,
                "wall_s": self.wall_s, "calls": [asdict(c) for c in self.calls],
                "verdicts": [asdict(v) for v in self.verdicts],
                "token_curve": self.token_curve, "cost_curve": self.cost_curve,
                "blocked": [asdict(b) for b in self.blocked], "notes": self.notes}


_TIERS = ("L0", "L1", "L2", "L3", "L4")


def _deepest_tier(tiers) -> str:
    if not isinstance(tiers, dict):
        return ""
    got = [t for t in _TIERS if tiers.get(t) == "pass"]
    return got[-1] if got else ""


def read_verdicts(run_dir: Path) -> list[Verdict]:
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


def build_anatomy(run_dir: Path, spanset: SpanSet, *, run_id: str, target: str, arm: str,
                  model: str, token_curve=None, cost_curve=None) -> Anatomy:
    """Assemble one run's full record. Spans and verdicts keep their own clocks; both start at 0."""
    a = Anatomy(run_id=run_id, target=target, arm=arm, model=model, wall_s=spanset.wall_s)
    for sp in spanset.spans:
        a.calls.append(Call(sp.start_s, sp.duration_s, categorize(sp.kind, sp.detail)))
    a.verdicts = read_verdicts(run_dir)
    a.blocked = read_blocked(run_dir)
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
