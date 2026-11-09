"""Judge one recorded round again: replay its transcript audit, receipt join and edit authority.

A round's status (authored, refused, failed) was decided once, by the code of the day, and that
decision attributes every byte the round asked to measure.  When the audit itself is in question -- a
parser false positive refused a clean round, or a fix to the audit lands -- the operator needs to see
what today's audit says about the SAME evidence: the round's own transcript, the broker receipts the
host sealed, and the round workspace's final bytes.  :func:`audit_round` reruns exactly the round
driver's owners over them (:func:`.rounds.join_receipts`, the shared transcript audit, the run's edit
authority, phase 1's status rule) and compares the replayed status with the recorded one.

It writes nothing and changes no attribution: a disagreement is a finding for the operator, who
decides (``admin``) what to do about it, never something this replay settles on its own.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from merlin_experiments.phase2.authoring import authored_round_status
from merlin_experiments.phase2.transcript_audit import audit_codex_transcript

from . import rounds as RND
from .identity import read_json

AUDIT_SCHEMA = "merlin.phase2.whole_model_measured.round_audit.v1"


class RoundAuditError(RuntimeError):
    """The round's evidence is missing or unreadable, so there is nothing to judge."""


def _target_experiment(run_dir: Path, target: str) -> Any:
    from merlin.targetgen.target_experiment import descriptor_for, load_target_experiment

    config = read_json(run_dir / "whole_model_objective_config.json") or {}
    descriptor = config.get("descriptor") or descriptor_for(target)
    if descriptor is None:
        raise RoundAuditError(f"no target descriptor for {target!r}")
    return load_target_experiment(Path(descriptor))


def _edits(run_dir: Path, candidate: Path, recorded: Mapping[str, Any]) -> dict[str, Any]:
    """The whole-package rule over the round's final bytes.  A run that froze an edit contract is not
    re-judged here (freezing one again would write into the run); its recorded verdict is shown."""
    if (run_dir / "edit_authority").is_dir():
        return {"status": "not_replayed", "why": "the run froze an edit contract", "recorded": recorded.get("edits")}
    try:
        return RND.whole_package_edits(run_dir / "seed" / "submission", candidate)
    except ValueError as exc:
        return {"status": "refused", "reason": str(exc)}


def audit_round(
    run_dir: Path,
    index: int,
    *,
    target_experiment: Any = None,
    audit_token_set: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, Any]:
    """Replay round ``index`` (0-based, as its files are named) of ``run_dir``; see the module doc."""
    run_dir = Path(run_dir)
    stage = run_dir / "stage"
    name = f"round_{int(index):02d}"
    recorded = read_json(stage / "rounds" / f"{name}{RND.ROUND_SUFFIX}")
    transcript = stage / "rounds" / f"{name}.transcript.jsonl"
    candidate = stage / "agent_workspaces" / name / "submission"
    receipts = stage / "control" / name / "receipts.jsonl"
    if recorded is None:
        raise RoundAuditError(f"{run_dir} has no record of round {index}")
    for what, path in (("transcript", transcript), ("round workspace", candidate)):
        if not path.exists():
            raise RoundAuditError(f"round {index}'s {what} is gone ({path}); there is nothing to replay")
    if target_experiment is None:
        record = read_json(run_dir / "run.json") or {}
        target_experiment = _target_experiment(run_dir, str(record.get("target") or ""))
    audit = audit_codex_transcript(
        transcript, target_experiment, candidate, RND.ACTIONS, audit_token_set=audit_token_set
    )
    joined = RND.join_receipts(receipts, audit)
    edits = _edits(run_dir, candidate, recorded)
    refusals = [] if joined["joined"] else [str(joined["reason"])]
    if edits.get("status") == "refused":
        refusals.append(f"edit authority: {edits.get('reason')}")
    exit_code = recorded.get("agent_exit_code")
    replayed = (
        authored_round_status(agent_exit_code=int(exit_code), audit_clean=audit.get("clean"), refusals=refusals)
        if exit_code is not None
        else {"status": recorded.get("status"), "why": "no agent exit was recorded; only the audit is replayed"}
    )
    hits: dict[str, list[int]] = {}
    for hit in audit.get("hits") or ():
        line = str(hit.get("line") or "")
        hits.setdefault(str(hit.get("kind")), []).append(int(line) if line.isdigit() else -1)
    return {
        "schema": AUDIT_SCHEMA,
        "run_dir": str(run_dir),
        "round": int(index),
        "recorded": {
            "status": recorded.get("status"),
            "why": recorded.get("why"),
            "audit_clean": (recorded.get("audit") or {}).get("clean"),
            "agent_exit_code": exit_code,
        },
        "replayed": {
            "status": replayed.get("status"),
            "why": replayed.get("why"),
            "audit_clean": audit.get("clean"),
            "commands_seen": audit.get("commands_seen"),
            "hits": {kind: sorted(set(lines)) for kind, lines in sorted(hits.items())},
            "receipts": joined,
            "edits": edits,
        },
        "agrees": replayed.get("status") == recorded.get("status"),
    }


def transcript_lines(run_dir: Path, index: int, numbers: Sequence[int], *, width: int = 1500) -> dict[int, str]:
    """The round transcript's own lines ``numbers`` (1-based, as the audit's hits name them)."""
    path = Path(run_dir) / "stage" / "rounds" / f"round_{int(index):02d}.transcript.jsonl"
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return {n: lines[n - 1][:width] for n in numbers if 0 < n <= len(lines)}


def format_audit(document: Mapping[str, Any]) -> str:
    recorded, replayed = document["recorded"], document["replayed"]
    lines = [
        f"round {document['round']} of {document['run_dir']}",
        f"  recorded: {recorded['status']} (audit clean {recorded['audit_clean']}) -- {recorded['why']}",
        f"  replayed: {replayed['status']} (audit clean {replayed['audit_clean']}) -- {replayed['why']}",
        f"  {'AGREES' if document['agrees'] else 'DISAGREES'}; commands seen {replayed['commands_seen']}",
        f"  receipts: {json.dumps(replayed['receipts'])}",
        f"  edits: {json.dumps(replayed['edits'], default=str)[:300]}",
    ]
    for kind, numbers in replayed["hits"].items():
        lines.append(f"  hit {kind}: lines {numbers}")
    return "\n".join(lines)


__all__ = ["AUDIT_SCHEMA", "RoundAuditError", "audit_round", "format_audit", "transcript_lines"]
