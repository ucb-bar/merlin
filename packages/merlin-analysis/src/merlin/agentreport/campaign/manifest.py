"""The campaign manifest: which run directories, agent homes and launch logs make up one campaign.

A campaign is a chain of continuations (each seeded from the previous one's compiler) whose launches
may have been restarted several times. Nothing about a particular campaign lives in code: the caller
names every path in a YAML manifest (or on the command line for a single run) and this module only
validates and resolves it.

Schema ``merlin.agentreport.campaign_manifest.v1``::

    schema: merlin.agentreport.campaign_manifest.v1
    campaign: <name shown in notes>
    target: <target id; an output folder axis only>
    root: <optional base for every relative path below>
    reserved_cores: 20                  # size of the CPU reservation the runs shared
    rtl_engine: gsim                    # engine identity a certifying (L3) row must carry
    simulator_labels: {l2: ..., l3: ...}  # names shown for the L2 and L3 simulators
    attempts:                           # in launch order; restarts of one continuation repeat its name
      - continuation: v5
        attempt: 1                      # optional, for labels
        run_dir: <run directory>
        workspace: <agent workspace>    # optional
        agent_home: <agent CLI home with sessions/**/rollout-*.jsonl>   # optional
        launch_log: <runner stdout; its cpu_reservation_verified line is the launch time>  # optional
        accounting_log: <systemd-run log>            # optional
        accounting_entry: -1            # which "Running as unit" block of that log belongs here
        live: false                     # true for an attempt still running
    cohorts: [...]                      # test-case grouping rules in display order, see ``cohort_of``
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

SCHEMA = "merlin.agentreport.campaign_manifest.v1"


@dataclass
class Attempt:
    continuation: str
    attempt: int | None
    run_dir: Path | None
    workspace: Path | None
    agent_home: Path | None
    launch_log: Path | None
    accounting_log: Path | None
    accounting_entry: int
    live: bool

    @property
    def id(self) -> str:
        return self.continuation if self.attempt is None else f"{self.continuation}a{self.attempt}"


@dataclass
class Manifest:
    campaign: str
    target: str | None
    reserved_cores: int
    rtl_engine: str
    attempts: list[Attempt]
    cohorts: list[dict[str, Any]] = field(default_factory=list)
    labels: dict[str, str] = field(default_factory=lambda: {"l2": "L2 sim", "l3": "L3 sim"})
    source: Path | None = None
    raw: dict[str, Any] = field(default_factory=dict)


def _path(root: Path | None, value: Any) -> Path | None:
    if not value:
        return None
    p = Path(str(value)).expanduser()
    if not p.is_absolute() and root is not None:
        p = root / p
    return p


def from_dict(doc: dict[str, Any], *, base: Path | None = None, source: Path | None = None) -> Manifest:
    if not isinstance(doc, dict):
        raise ValueError("campaign manifest must be a mapping")
    if doc.get("schema") not in (None, SCHEMA):
        raise ValueError(f"unsupported campaign manifest schema {doc.get('schema')!r}")
    root = _path(base, doc.get("root")) or base
    attempts = []
    for i, row in enumerate(doc.get("attempts") or []):
        if not isinstance(row, dict) or not row.get("continuation"):
            raise ValueError(f"attempt {i}: 'continuation' is required")
        attempts.append(
            Attempt(
                continuation=str(row["continuation"]),
                attempt=int(row["attempt"]) if row.get("attempt") is not None else None,
                run_dir=_path(root, row.get("run_dir")),
                workspace=_path(root, row.get("workspace")),
                agent_home=_path(root, row.get("agent_home")),
                launch_log=_path(root, row.get("launch_log")),
                accounting_log=_path(root, row.get("accounting_log")),
                accounting_entry=int(row.get("accounting_entry", -1)),
                live=bool(row.get("live", False)),
            )
        )
    if not attempts:
        raise ValueError("campaign manifest lists no attempts")
    ids = [a.id for a in attempts]
    if len(set(ids)) != len(ids):
        raise ValueError(f"attempt ids must be unique: {ids}")
    return Manifest(
        campaign=str(doc.get("campaign") or "campaign"),
        target=str(doc["target"]) if doc.get("target") else None,
        reserved_cores=int(doc.get("reserved_cores", 20)),
        rtl_engine=str(doc.get("rtl_engine", "gsim")),
        attempts=attempts,
        cohorts=list(doc.get("cohorts") or []),
        labels={"l2": "L2 sim", "l3": "L3 sim", **{k: str(v) for k, v in (doc.get("simulator_labels") or {}).items()}},
        source=source,
        raw=doc,
    )


def load(path: str | Path) -> Manifest:
    path = Path(path)
    doc = yaml.safe_load(path.read_text())
    return from_dict(doc, base=path.parent, source=path)


def single_run(
    run_dir: str | Path,
    *,
    workspace: str | Path | None = None,
    agent_home: str | Path | None = None,
    launch_log: str | Path | None = None,
    accounting_log: str | Path | None = None,
    target: str | None = None,
    live: bool = False,
) -> Manifest:
    """A one-attempt manifest for the command-line shorthand."""
    row = {
        "continuation": Path(run_dir).name,
        "run_dir": str(run_dir),
        "workspace": str(workspace) if workspace else None,
        "agent_home": str(agent_home) if agent_home else None,
        "launch_log": str(launch_log) if launch_log else None,
        "accounting_log": str(accounting_log) if accounting_log else None,
        "live": live,
    }
    return from_dict({"schema": SCHEMA, "campaign": Path(run_dir).name, "target": target, "attempts": [row]})


def _matches(rule: dict[str, Any], name: str, meta: dict[str, Any], calibration: set[str]) -> bool:
    if rule.get("default"):
        return True
    if rule.get("calibration_profile") and name in calibration:
        return True
    if any(name.startswith(p) for p in rule.get("name_prefixes") or []):
        return True
    if rule.get("corpus_cohort") and meta.get("cohort") == rule["corpus_cohort"]:
        return True
    if meta.get("kind") and meta.get("kind") in (rule.get("kinds") or []):
        return True
    if meta.get("op") and meta.get("op") in (rule.get("ops") or []):
        return True
    low = name.lower()
    return any(s in low for s in rule.get("name_contains") or [])


def cohort_of(rules: list[dict[str, Any]], name: str, meta: dict[str, Any], calibration: set[str]) -> str | None:
    """First matching rule by ``priority`` (default: manifest order); plain string tests, no patterns.

    The manifest lists cohorts in DISPLAY order; ``priority`` decides which rule claims a test case
    that several would match (e.g. a calibration member that is also a dense matmul)."""
    ordered = sorted(enumerate(rules), key=lambda ir: (ir[1].get("priority", ir[0]), ir[0]))
    for _, rule in ordered:
        if _matches(rule, name, meta or {}, calibration):
            return str(rule["id"])
    return None
