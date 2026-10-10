"""Read a checkpointed paired Phase 2 experiment: trials x members x {tuning, held_out, held_out_form}.

The experiment root holds ``state/checkpoint.NNNN.<sha>.json`` (the controller's append-only chain,
:mod:`..phase2.checkpoint_admission`), ``agent_visible/*commitment.json`` and, once sealed,
``experiment_manifest.<sha>.json``.  The chain names everything else:

* ``candidate:<trial>`` -> the trial's authoring stage directory (``performance_candidate.json`` beside
  ``rounds/``, ``control/round_NN/{receipts.jsonl, feedback/sha256/*.json}``, ``cost_time_toolcalls.yaml``,
  ``metrics/token_ledger.jsonl``);
* ``measurement:<trial>:<label>`` -> the cell's ``campaign_manifest.json`` in its measurement directory
  (``paired_cycles.json``, ``paired_completion_cells.json``, ``raw_results/``, ``measurement_reuse.json``,
  operator-only ``reference_comparison.json``).

A cell still being measured is not in the chain yet; ``--measurement-root`` / ``--stage-root`` let the
view find ``<experiment>__<trial>__<label>/`` directories as they appear.

TIMES.  These records carry no wall clock: a checkpoint's, a commitment's, a reveal's and a raw
result's time is its file's modification time (each is written once and sealed), and the page says so.
A measurement slot is reconstructed from raw-result write times and recorded durations and packed into
the campaign's declared fan-out -- labelled as a reconstruction, never as a recorded slot.

PRIVATE MATERIAL.  Held-out member names are replaced by ordinal labels and ``reference_comparison.json``
is reduced to "present" unless ``operator_private``.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from . import records as R

TUNING, HELD_OUT = "tuning", "held_out"
PROFILE_ACTIONS = (
    "profile-tuning-member",
    "profile-whole-model",
    "analyze-command-buffers",
    "profile-reduced-global-witness",
    "tuning-gsim-feedback",
    "analyze-whole-model",
    "inspect-optimization-surfaces",
)


def _owner() -> Any:
    from ..phase2 import checkpoint_admission as AD

    return AD


def is_experiment_root(path: Path) -> bool:
    path = Path(path)
    return (
        (path / "state").is_dir()
        and any((path / "state").glob("checkpoint.*.json"))
        or any(path.glob("experiment_manifest.*.json"))
    )


def _json(path: Path) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _mtime(path: Path | None) -> float | None:
    try:
        return os.stat(path).st_mtime if path else None
    except OSError:
        return None


def checkpoints(root: Path, inventory: R.Inventory) -> list[dict[str, Any]]:
    state = Path(root) / "state"
    rows = []
    for path in sorted(state.glob("checkpoint.*.json")):
        body = _json(path)
        if not isinstance(body, Mapping):
            rows.append({"index": None, "stage": None, "at": _mtime(path), "path": str(path), "unreadable": True})
            continue
        rows.append(
            {
                "index": body.get("index"),
                "stage": body.get("stage"),
                "at": _mtime(path),
                "path": str(path),
                "evidence": body.get("evidence") if isinstance(body.get("evidence"), Mapping) else {},
            }
        )
    inventory.note("state/checkpoint.*.json", state, "read" if rows else "absent", f"{len(rows)} checkpoints")
    return rows


def expected_stages(form: bool) -> list[str]:
    AD = _owner()
    trials = list(AD.TRIALS)
    labels = [TUNING, HELD_OUT] + ([AD.FORM_HOLDOUT_MEASUREMENT_LABEL] if form else [])
    out = ["predeclared", "holdout_committed"] + (["form_holdout_committed"] if form else [])
    out += [f"candidate:{t}" for t in trials] + [f"functional_regrade:{t}" for t in trials]
    out += ["holdout_revealed"] + (["form_holdout_revealed"] if form else [])
    out += ["heldout_gsim_certificate"] + (["form_heldout_gsim_certificate"] if form else [])
    out += ["statistics_predeclared"] + [f"measurement:{t}:{label}" for t in trials for label in labels]
    return out


# --------------------------------------------------------------------------- one measurement cell
def _single_observations(manifest: Mapping[str, Any]) -> set[tuple[str, str]]:
    plan = manifest.get("measurement_plan") if isinstance(manifest.get("measurement_plan"), Mapping) else {}
    out = set()
    policy = plan.get("replicate_policy") if isinstance(plan.get("replicate_policy"), Mapping) else {}
    for row in policy.get("single_observation") or ():
        if isinstance(row, Mapping):
            out.add((str(row.get("family")), str(row.get("capsule"))))
    for entry in plan.get("schedule") or ():
        if isinstance(entry, Mapping) and entry.get("gsim_observation") == "single_gsim_observation":
            out.add((str(entry.get("family")), str(entry.get("capsule"))))
    return out


def _slots(directory: Path, manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Raw executions as reconstructed spans: end = the raw record's write time, start = end minus its
    recorded duration (adapter wall, else active simulation), lane = first free slot of the fan-out."""
    index = _json(directory / "raw_results.index.json")
    if not isinstance(index, list):
        return []
    fanout = manifest.get("execution_fanout") if isinstance(manifest.get("execution_fanout"), Mapping) else {}
    width = fanout.get("effective") if isinstance(fanout.get("effective"), int) and fanout.get("effective") > 0 else 1
    spans = []
    for entry in index:
        if not isinstance(entry, Mapping):
            continue
        path = Path(str(entry.get("path") or ""))
        if not path.is_absolute():
            path = directory / path
        end = _mtime(path)
        if end is None:
            continue
        raw = _json(path) or {}
        outcome = ((raw.get("measurement") or {}).get("execution_outcome") or {}) if isinstance(raw, Mapping) else {}
        tier = ((outcome.get("gsim") or {}).get("tier_outcome") or {}) if isinstance(outcome, Mapping) else {}
        timing = tier.get("timing") if isinstance(tier.get("timing"), Mapping) else {}
        duration = next(
            (timing[k] for k in ("adapter_wall_s", "sim_active_s") if isinstance(timing.get(k), int | float)), None
        )
        spans.append(
            {
                "start": end - duration if duration else None,
                "end": end,
                "duration": duration,
                "execution": entry.get("execution_index"),
                "arm": entry.get("arm"),
                "family": entry.get("family"),
                "capsule": entry.get("capsule"),
                "replicate": entry.get("replicate"),
            }
        )
    spans.sort(key=lambda s: s["start"] if s["start"] is not None else s["end"])
    free = [0.0] * width
    for span in spans:
        start = span["start"] if span["start"] is not None else span["end"]
        lane = next((i for i, t in enumerate(free) if t <= start + 1e-6), min(range(width), key=lambda i: free[i]))
        span["slot"] = lane
        free[lane] = span["end"]
    return spans


def read_cell(directory: Path, inventory: R.Inventory, *, operator_private: bool) -> dict[str, Any]:
    directory = Path(directory)
    manifest = inventory.json(f"{directory.name}/campaign_manifest.json", directory / "campaign_manifest.json")
    cycles = _json(directory / "paired_cycles.json")
    results = _json(directory / "paired_completion_cells.json")
    if isinstance(results, Mapping):
        results = results.get("cells")
    reuse = _json(directory / "measurement_reuse.json")
    reference = _json(directory / "reference_comparison.json")
    manifest = manifest or {}
    single = _single_observations(manifest)
    members: dict[tuple[str, str], dict[str, Any]] = {}
    for row in cycles if isinstance(cycles, list) else ():
        if not isinstance(row, Mapping):
            continue
        key = (str(row.get("family")), str(row.get("capsule")))
        member = members.setdefault(key, {"pairs": [], "single": key in single})
        member["pairs"].append(
            {
                k: row.get(k)
                for k in (
                    "replicate",
                    "baseline_cycles",
                    "candidate_cycles",
                    "baseline_over_candidate",
                    "comparable",
                    "baseline_carried",
                    "candidate_carried",
                )
            }
        )
    for row in results if isinstance(results, list) else ():
        if not isinstance(row, Mapping) or row.get("simulator") == "spike":
            continue
        key = (str(row.get("family")), str(row.get("capsule")))
        member = members.setdefault(key, {"pairs": [], "single": key in single})
        member.setdefault("results", []).append(
            {k: row.get(k) for k in ("arm", "replicate", "correct", "cycles", "citable")}
            | {
                "reused": (row.get("provenance") or {}).get("reused_measurement")
                if isinstance(row.get("provenance"), Mapping)
                else None
            }
        )
    completion = manifest.get("completion") if isinstance(manifest.get("completion"), Mapping) else None
    ref = None
    if isinstance(reference, Mapping):
        summary = reference.get("summary") if isinstance(reference.get("summary"), Mapping) else {}
        ref = {"present": True, "cells": summary.get("cells")}
        if operator_private:
            ref.update(
                {
                    "summary": dict(summary),
                    "rows": [r for r in reference.get("rows") or () if isinstance(r, Mapping)],
                    "reference": reference.get("reference"),
                }
            )
    return {
        "dir": str(directory),
        "exists": directory.is_dir(),
        "status": manifest.get("status"),
        "refusal": manifest.get("refusal"),
        "completion": completion,
        "fanout": manifest.get("execution_fanout"),
        "members": {f"{f}/{c}": v for (f, c), v in sorted(members.items())},
        "reuse": {k: reuse.get(k) for k in ("cited_cells", "measured_here", "carried", "unstated", "auditable")}
        if isinstance(reuse, Mapping)
        else None,
        "slots": _slots(directory, manifest),
        "reference": ref,
        "updated": _mtime(directory / "paired_completion_cells.json"),
        "functional_run_id": manifest.get("functional_run_id"),
        "functional_submission_sha256": manifest.get("functional_submission_sha256"),
    }


# --------------------------------------------------------------------------- one trial's authoring stage
def _feedback(stage: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(stage.glob("control/round_*/feedback/sha256/*.json")):
        doc = _json(path)
        if not isinstance(doc, Mapping):
            continue
        for cell in doc.get("cells") or ():
            if not isinstance(cell, Mapping):
                continue
            roofline = cell.get("roofline") if isinstance(cell.get("roofline"), Mapping) else {}
            executed = cell.get("executed_commands") if isinstance(cell.get("executed_commands"), Mapping) else {}
            rows.append(
                {
                    "round": doc.get("round"),
                    "invocation": doc.get("invocation"),
                    "at": _mtime(path),
                    "member": f"{cell.get('family')}/{cell.get('capsule')}",
                    "verdict": cell.get("verdict"),
                    "measured": cell.get("measured"),
                    "skip_reason": cell.get("skip_reason"),
                    "baseline_cycles": cell.get("baseline_gsim_cycles"),
                    "candidate_cycles": cell.get("candidate_gsim_cycles"),
                    "baseline_over_candidate": cell.get("baseline_over_candidate"),
                    "roofline_status": roofline.get("status"),
                    "limiter": roofline.get("limiter"),
                    "baseline_over_roofline": roofline.get("baseline_over_roofline"),
                    "candidate_over_roofline": roofline.get("candidate_over_roofline"),
                    "executed": {
                        arm: {
                            k: (executed.get(arm) or {}).get(k)
                            for k in (
                                "status",
                                "accelerator_commands",
                                "retired_instructions",
                                "by_class",
                                "local_memory",
                                "why",
                            )
                        }
                        for arm in ("baseline", "candidate")
                        if isinstance(executed.get(arm), Mapping)
                    },
                }
            )
    return rows


def _receipts(stage: Path, inventory: R.Inventory) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(stage.glob("control/round_*/receipts.jsonl")):
        number = path.parent.name.partition("_")[2]
        for row in inventory.jsonl(f"{stage.name}/{path.parent.name}/receipts.jsonl", path) or ():
            rows.append(
                {
                    "round": int(number) if number.isdigit() else number,
                    "index": row.get("index"),
                    "action": row.get("action"),
                    "state": row.get("state"),
                    "returncode": row.get("returncode"),
                    "elapsed_s": row.get("elapsed_s"),
                    "bindings": row.get("bindings"),
                    "rejection": row.get("rejection_reason"),
                }
            )
    return rows


def read_stage(stage: Path, inventory: R.Inventory, tails) -> dict[str, Any]:
    from . import activity as A

    stage = Path(stage)
    candidate = inventory.json(f"{stage.name}/performance_candidate.json", stage / "performance_candidate.json")
    cost = inventory.yaml(f"{stage.name}/cost_time_toolcalls.yaml", stage / "cost_time_toolcalls.yaml")
    accounting = (
        ((candidate or {}).get("telemetry") or {}).get("accounting")
        if isinstance((candidate or {}).get("telemetry"), Mapping)
        else None
    )
    per_round = []
    for r in (
        ((candidate or {}).get("agent") or {}).get("rounds") or ()
        if isinstance((candidate or {}).get("agent"), Mapping)
        else ()
    ):
        if isinstance(r, Mapping):
            acc = (
                ((r.get("telemetry") or {}).get("accounting") or {}) if isinstance(r.get("telemetry"), Mapping) else {}
            )
            per_round.append(
                {
                    "round": r.get("round"),
                    "tokens_total": acc.get("tokens_total"),
                    "tokens_output": acc.get("tokens_output"),
                    "tool_calls": acc.get("tool_calls"),
                }
            )
    activity = A.read(stage, R.Inventory(), tails) if (stage / "rounds").is_dir() else None
    actions = []
    for call in (activity or {}).get("calls") or ():
        command = str(call.get("command") or "")
        hit = next((a for a in PROFILE_ACTIONS if a in command), None)
        if hit:
            actions.append(
                {
                    "action": hit,
                    "start": call["start"],
                    "end": call["end"],
                    "failed": call.get("failed"),
                    "command": command[:200],
                }
            )
    tokens = accounting or cost or {}
    return {
        "dir": str(stage),
        "exists": stage.is_dir(),
        "tokens": {
            k: tokens.get(k)
            for k in (
                "tokens_total",
                "tokens_input",
                "tokens_cached",
                "tokens_output",
                "tool_calls",
                "wall_time_seconds",
                "subscription_notional_usd",
            )
        }
        if tokens
        else None,
        "rounds": per_round,
        "receipts": _receipts(stage, inventory),
        "feedback": _feedback(stage),
        "actions": actions,
        "activity": activity,
    }


# --------------------------------------------------------------------------- the experiment
def summary(
    root: Path,
    inventory: R.Inventory,
    tails,
    *,
    operator_private: bool = False,
    measurement_root: Path | None = None,
    stage_root: Path | None = None,
) -> dict[str, Any]:
    AD = _owner()
    root = Path(root)
    chain = checkpoints(root, inventory)
    form = (root / "agent_visible" / "form_holdout_commitment.json").is_file() or any(
        str(c.get("stage")).startswith("form_") for c in chain
    )
    labels = [TUNING, HELD_OUT] + ([AD.FORM_HOLDOUT_MEASUREMENT_LABEL] if form else [])
    stages: dict[str, Path] = {}
    cells: dict[str, Path] = {}
    for c in chain:
        stage, evidence = str(c.get("stage") or ""), c.get("evidence") or {}
        kind, _, rest = stage.partition(":")
        if kind == "candidate" and evidence.get("record"):
            stages[rest] = Path(str(evidence["record"])).parent
        elif kind == "measurement" and evidence.get("path"):
            cells[rest] = Path(str(evidence["path"])).parent
    experiment_id = next((p.name.rpartition("__")[0] for p in stages.values()), None) or next(
        (p.name.split("__")[0] for p in cells.values()), None
    )
    if stage_root is not None and Path(stage_root).is_dir():
        for p in Path(stage_root).glob("*__trial_*"):
            if experiment_id in (None, p.name.rpartition("__")[0]):
                stages.setdefault(p.name.rpartition("__")[2], p)
    if measurement_root is not None and Path(measurement_root).is_dir():
        for p in Path(measurement_root).glob("*__trial_*__*"):
            parts = p.name.split("__")
            if len(parts) == 3 and experiment_id in (None, parts[0]):
                cells.setdefault(f"{parts[1]}:{parts[2]}", p)
    trials = list(AD.TRIALS)
    cell_rows = {
        key: read_cell(path, inventory, operator_private=operator_private) for key, path in sorted(cells.items())
    }
    stage_rows = {t: read_stage(path, inventory, tails) for t, path in sorted(stages.items())}
    manifest_path = next(iter(sorted(root.glob("experiment_manifest.*.json"))), None)
    final = inventory.json("experiment_manifest", manifest_path) if manifest_path else None
    if manifest_path is None:
        inventory.note("experiment_manifest.<sha>.json", root, "absent", "not sealed yet")
    stats = (final or {}).get("statistics") if isinstance((final or {}).get("statistics"), Mapping) else None
    holdout = []
    for name, stage_name in (
        ("holdout_commitment.json", "holdout_committed"),
        ("form_holdout_commitment.json", "form_holdout_committed"),
    ):
        path = root / "agent_visible" / name
        if path.is_file():
            holdout.append({"event": f"commit ({name})", "at": _mtime(path), "source": "file time"})
    for c in chain:
        if str(c.get("stage") or "") in (
            "holdout_committed",
            "form_holdout_committed",
            "holdout_revealed",
            "form_holdout_revealed",
            "heldout_gsim_certificate",
            "form_heldout_gsim_certificate",
        ):
            evidence = c.get("evidence") or {}
            manifest = Path(str(evidence["manifest"])) if evidence.get("manifest") else None
            revealed = _json(manifest) if manifest else None
            members = len((revealed or {}).get("members") or ()) if isinstance(revealed, Mapping) else None
            holdout.append(
                {
                    "event": c["stage"],
                    "at": c["at"],
                    "source": "checkpoint file time",
                    "members": members,
                    "digest": evidence.get("manifest_sha256") or evidence.get("public_sha256"),
                }
            )
    done = [str(c.get("stage")) for c in chain]
    bindings = sorted(
        {(c.get("functional_run_id"), c.get("functional_submission_sha256")) for c in cell_rows.values()}
        - {(None, None)},
        key=str,
    )
    return {
        "functional": [{"run_id": run, "submission_sha256": sha} for run, sha in bindings],
        "kind": "paired",
        "root": str(root),
        "experiment_id": experiment_id,
        "operator_private": operator_private,
        "trials": trials,
        "labels": labels,
        "form": form,
        "chain": chain,
        "expected": expected_stages(form),
        "done": done,
        "cells": cell_rows,
        "stages": stage_rows,
        "statistics": {
            "aggregate": stats.get("aggregate"),
            "status": stats.get("status"),
            "per_trial": stats.get("per_trial"),
        }
        if stats
        else None,
        "sealed": final is not None,
        "holdout": sorted(holdout, key=lambda h: h["at"] or 0.0),
    }


__all__ = ["PROFILE_ACTIONS", "expected_stages", "is_experiment_root", "read_cell", "read_stage", "summary"]
