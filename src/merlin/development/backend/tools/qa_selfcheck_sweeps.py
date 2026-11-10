"""Standalone advisory sweep selection, requests and ranking; no grader or golden access."""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from pathlib import Path

if __package__:
    from .qa_selfcheck_protocol import (
        _SUBMISSION,
        PROTOCOL_VERSION,
        _atomic_write,
        _read_json,
        _reply_is_usable,
        _request_budget_seconds,
        _request_id,
        _requester_identity,
        _source_label,
        _submission_digest,
        _variant_error,
    )
else:
    from qa_selfcheck_protocol import (
        _SUBMISSION,
        PROTOCOL_VERSION,
        _atomic_write,
        _read_json,
        _reply_is_usable,
        _request_budget_seconds,
        _request_id,
        _requester_identity,
        _source_label,
        _submission_digest,
        _variant_error,
    )

_SWEEP_SCHEMA = "merlin.selfcheck_sweep.v1"


_EXPERIMENT_SCHEMA = "merlin.idle_tuning_experiment.v1"


_MODEL_KEYS = (
    "selection_sha256",
    "model_sha256",
    "timing_mode",
    "timing_epoch",
    "scope_sha256",
    "comparison_domain_sha256",
)


def _sweep_candidates(ws: Path, variants: str, knobs: list[str]) -> list[dict]:
    """Validate the complete candidate list before making any request visible."""
    candidates = []
    for value in variants.split(",") if variants else ():
        label, error = _source_label(ws, value.strip())
        error = error or (_variant_error(ws, label) if label != _SUBMISSION else "--sweep names variants/")
        if error:
            raise ValueError(error)
        candidates.append({"submission": label, "key": "variant:" + label.partition("/")[2]})
    if knobs:
        declared = _read_json(ws / _SUBMISSION / "TUNING_KNOBS.json")
        if not isinstance(declared, dict) or declared.get("schema") != "merlin.tuning_knobs.v1":
            raise ValueError("--knob needs submission/TUNING_KNOBS.json (merlin.tuning_knobs.v1)")
        for text in knobs:
            name, equals, raw = text.partition("=")
            try:
                values = json.loads("[" + raw + "]") if equals else []
            except ValueError as exc:
                raise ValueError("--knob takes NAME=V1,V2 with JSON scalar values") from exc
            spec = (declared.get("knobs") or {}).get(name)
            allowed = spec.get("values") if isinstance(spec, dict) else None
            if not values or not isinstance(allowed, list):
                raise ValueError("--knob needs a declared knob and at least one JSON scalar value")
            for value in values:
                if not any(type(value) is type(v) and value == v for v in allowed):
                    raise ValueError(f"--knob {name}: value is not declared in TUNING_KNOBS.json")
                candidates.append(
                    {
                        "submission": _SUBMISSION,
                        "tuning_overrides": {name: value},
                        "key": "knob:" + name + "=" + json.dumps(value),
                    }
                )
    unique = {c["key"]: c for c in candidates}
    if not unique:
        raise ValueError("--sweep needs variant trees or --knob NAME=V1,V2")
    return list(unique.values())


def _sweep_cases(ws: Path, capsules: str, bundle: str) -> str:
    if bundle:
        if capsules != "all":
            raise ValueError("--bundle selects its own cases; drop --capsules")
        ledger = _read_json(ws / "qa" / "idle_tuning.json") or {}
        chosen = next((b for b in ledger.get("bundles") or () if b.get("representative") == bundle), None)
        if chosen is None:
            raise ValueError("no matching bundle in qa/idle_tuning.json; use --capsules X,Y")
        capsules = ",".join([chosen["representative"], *chosen.get("siblings", [])])
    names = [name.strip() for name in capsules.split(",") if name.strip()]
    if not names or any(name in ("all", "changed", "failing") or ":" in name for name in names):
        raise ValueError("--sweep needs named --capsules X,Y or --bundle REPRESENTATIVE")
    return ",".join(dict.fromkeys(names))


def _case_groups(ws: Path, names: list[str]) -> list[tuple[str, list[str]]]:
    """Route public names by the staged inventory, never by a case-name convention.

    A performance bundle may include a core/calibration control or a separately
    selected program. Each native suite owns a distinct input roster. Partition
    before publishing requests; changing the suite cannot grant another input.
    Historical workspaces without an inventory retain their scalability route.
    """
    if not names:
        raise ValueError("public case routing needs named cases")
    path = ws / "PUBLIC_CAPABILITIES.json"
    if not path.exists() and not path.is_symlink():
        return [("scalability", list(dict.fromkeys(names)))]
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 16 << 20:
        raise ValueError("public case routing requires an ordinary bounded inventory file")
    inventory = _read_json(path)
    members = inventory.get("members") if isinstance(inventory, dict) else None
    if not isinstance(members, dict):
        raise ValueError("public case routing requires the staged capability inventory")
    groups = {}
    for name in dict.fromkeys(names):
        row = members.get(name)
        suite = row.get("selfcheck_suite") if isinstance(row, dict) else None
        if suite not in {"core", "scalability", "programs"}:
            raise ValueError("case has no declared public self-check suite: " + str(name))
        groups.setdefault(suite, []).append(name)
    if not groups:
        raise ValueError("public case routing needs named cases")
    return list(groups.items())


def _sweep_rows(ws: Path, document: dict) -> dict:
    """Read verified public detail pointers; incomplete or altered details never supply cycle evidence."""
    found = {}
    for row in document.get("cases") or document.get("per_capsule") or ():
        if not isinstance(row, dict):
            continue
        detail = row.get("detail")
        if isinstance(detail, dict):
            raw_path = detail.get("path")
            path = ws / raw_path if isinstance(raw_path, str) else ws
            try:
                path.resolve().relative_to(ws.resolve())
                data = path.read_bytes()
                if hashlib.sha256(data).hexdigest() != detail.get("sha256"):
                    continue
                full = json.loads(data)
                if not isinstance(full, dict) or full.get("capsule") != row.get("capsule"):
                    continue
                row = full
            except (OSError, ValueError):
                continue
        name = row.get("capsule")
        if isinstance(name, str):
            found[name] = row
    return found


def _sweep_passed(row: dict) -> bool:
    tier = (row.get("tiers") or {}).get("L2")
    if tier is not None:
        return (tier.get("status") if isinstance(tier, dict) else tier) == "pass"
    return row.get("pass") is True or row.get("status") == "pass"


def _sweep_rank(ws: Path, manifest: dict, responses: dict) -> tuple[list, list]:
    """Compare each member with this sweep's baseline only, under identical case/model scopes."""
    comparisons, rankings, groups = [], [], {}
    for request in manifest["requests"]:
        if request.get("key") == "baseline" or request["request_id"] in manifest.get(
            "baseline_request_ids", [manifest["baseline_request_id"]]
        ):
            continue
        baseline = responses.get(request.get("baseline_request_id", manifest["baseline_request_id"])) or {}
        before_rows = _sweep_rows(ws, baseline)
        document = responses.get(request["request_id"])
        if not isinstance(document, dict):
            continue
        group = []
        candidate_sha = (document.get("variant") or {}).get("sha256") or document.get("submission_sha256")
        after_rows = _sweep_rows(ws, document)
        names = request.get("capsules", list(after_rows))
        names = names.split(",") if isinstance(names, str) else names
        for name in dict.fromkeys(names):
            row = after_rows.get(name) or {}
            before = before_rows.get(name) or {}
            bp, cp = before.get("performance") or {}, row.get("performance") or {}
            before_cycles, after_cycles = bp.get("estimated_whole_soc_cycles"), cp.get("estimated_whole_soc_cycles")
            reason = next(("estimates differ in " + k for k in _MODEL_KEYS if bp.get(k) != cp.get(k)), None)
            if request.get("tuning_overrides") and baseline.get("submission_sha256") != document.get(
                "submission_sha256"
            ):
                reason = "knob copy and baseline differ in submission_sha256"
            required = ("model_sha256", "timing_mode", "scope_sha256", "comparison_domain_sha256", "elf_sha256")
            if reason is None and (
                any(not p.get(k) for p in (bp, cp) for k in required)
                or any(type(p.get("timing_epoch")) is not int for p in (bp, cp))
            ):
                reason = "incomplete model, case scope or executable identity"
            if any(
                type(c) not in (int, float) or not math.isfinite(c) or c <= 0 for c in (before_cycles, after_cycles)
            ):
                reason = "no complete model estimate on both sides"
            unchanged = _sweep_passed(before) and _sweep_passed(row)
            identity = {
                "candidate_sha256": candidate_sha,
                "baseline_sha256": baseline.get("submission_sha256"),
                "candidate_base_sha256": document.get("submission_sha256"),
                "case": name,
                "candidate_elf_sha256": cp.get("elf_sha256"),
                "baseline_elf_sha256": bp.get("elf_sha256"),
                "model": {k: cp.get(k) for k in _MODEL_KEYS},
                "baseline_model": {k: bp.get(k) for k in _MODEL_KEYS},
            }
            comparison = {
                "test_case": name,
                "candidate": request["key"],
                "identity": identity,
                "numerics": "unchanged" if unchanged else "changed_or_unknown",
                "cycles_before": before_cycles,
                "cycles_after": after_cycles,
                "delta_pct": round(100 * (after_cycles / before_cycles - 1), 2) if reason is None else None,
                "ratio": after_cycles / before_cycles if reason is None else None,
                "request_id": request["request_id"],
            }
            if reason:
                comparison["not_compared"] = reason
            comparisons.append(comparison)
            group.append(comparison)
        ranked = groups.setdefault(
            (request["key"], candidate_sha),
            {
                "candidate": request["key"],
                "candidate_sha256": candidate_sha,
                "rows": [],
                "request_ids": [],
                "errors": [],
            },
        )
        ranked["rows"].extend(group)
        ranked["request_ids"].append(request["request_id"])
        if document.get("error"):
            ranked["errors"].append(document["error"])
    for ranked in groups.values():
        group = ranked.pop("rows")
        usable = [c for c in group if c["numerics"] == "unchanged" and c["ratio"] is not None]
        rankings.append(
            {
                "candidate": ranked["candidate"],
                "candidate_sha256": ranked["candidate_sha256"],
                "geomean_ratio": round(math.exp(sum(math.log(c["ratio"]) for c in usable) / len(usable)), 4)
                if usable
                else None,
                "cases_compared": len(usable),
                "faster": sum(c["ratio"] < 1 for c in usable),
                "slower": sum(c["ratio"] > 1 for c in usable),
                "numerics_changed_or_unknown": sum(c["numerics"] != "unchanged" for c in group),
                "error": ranked["errors"][0] if ranked["errors"] else None,
                "request_id": ranked["request_ids"][0],
                "request_ids": ranked["request_ids"],
            }
        )
    rankings.sort(
        key=lambda r: (
            r["numerics_changed_or_unknown"] > 0,
            r["geomean_ratio"] is None,
            r["geomean_ratio"] or 0,
            r["candidate"],
        )
    )
    comparisons.sort(
        key=lambda c: (c["numerics"] != "unchanged", c["delta_pct"] is None, c["delta_pct"] or 0, c["test_case"])
    )
    return rankings, comparisons


def _run_sweep(a, ws: Path, ch: Path, manifest: dict | None = None) -> int:
    if manifest is None:
        candidates = _sweep_candidates(ws, a.sweep or "", a.knob)
        cases = _sweep_cases(ws, a.capsules, a.bundle)
        groups = _case_groups(ws, cases.split(","))
        sid = "sweep_" + _request_id()
        base = _submission_digest(ws / _SUBMISSION)
        now = time.time_ns()
        deadline = time.time() + _request_budget_seconds(a.timeout, capsules=cases, workers=a.workers)
        caller = os.environ.get("CODEX_THREAD_ID", "").strip()[:128]
        requests, bodies = [], []
        baselines = []
        for suite, names in groups:
            baseline_id = f"{sid}_{len(requests)}"
            baselines.append(baseline_id)
            for candidate in [{"submission": _SUBMISSION, "key": "baseline"}, *candidates]:
                label, rid = candidate["submission"], f"{sid}_{len(requests)}"
                digest = base if label == _SUBMISSION else _submission_digest(ws / label)
                member = {
                    **candidate,
                    "request_id": rid,
                    "submission_sha256": digest,
                    "suite": suite,
                    "capsules": list(names),
                    "baseline_request_id": baseline_id,
                }
                requests.append(member)
                body = {
                    "protocol": PROTOCOL_VERSION,
                    "request_id": rid,
                    "requested_at_unix_ns": now,
                    "requester": _requester_identity(),
                    "deadline_unix_ns": int(deadline * 1e9),
                    "submission": label,
                    "submission_sha256": digest,
                    "suite": suite,
                    "sim": "spike",
                    "capsules": ",".join(names),
                    "workers": a.workers,
                    "timeout": a.timeout,
                    **({"tiers": "L0,L1,L2"} if suite == "core" else {}),
                    **({"caller_thread_id": caller} if caller else {}),
                }
                body["tuning_experiment"] = {
                    "schema": _EXPERIMENT_SCHEMA,
                    "base_sha256": base,
                    "candidate": {"kind": "baseline", "key": "baseline"},
                    "test_cases": list(names),
                    "origin": "author_sweep",
                    "sweep": sid,
                }
                if candidate["key"] != "baseline":
                    kind = "knob" if candidate.get("tuning_overrides") else "variant"
                    body["tuning_experiment"] = {
                        "schema": _EXPERIMENT_SCHEMA,
                        "base_sha256": base,
                        "candidate": {"kind": kind, "key": candidate["key"]},
                        "test_cases": list(names),
                        "origin": "author_sweep",
                        "sweep": sid,
                    }
                    if kind == "knob":
                        body["tuning_overrides"] = candidate["tuning_overrides"]
                bodies.append(body)
        manifest = {
            "schema": _SWEEP_SCHEMA,
            "sweep_id": sid,
            "baseline_request_id": requests[0]["request_id"],
            "baseline_request_ids": baselines,
            "requests": requests,
            "capsules": cases,
            "deadline_unix_ns": int(deadline * 1e9),
        }
        _atomic_write(ch / f"{sid}.json", json.dumps(manifest))
        for body in bodies:
            rid = body["request_id"]
            _atomic_write(ch / f"detached_{rid}", json.dumps({"sweep": sid, "at_unix_ns": now}))
            _atomic_write(ch / f"req_{rid}.json", json.dumps(body))
    sid = manifest["sweep_id"]
    until = manifest["deadline_unix_ns"] / 1e9
    if a.wait_budget > 0:
        until = min(until, time.time() + a.wait_budget)
    responses, faults = {}, {}
    while True:
        for request in manifest["requests"]:
            rid = request["request_id"]
            if rid in responses or rid in faults or not (ch / f"done_{rid}").exists():
                continue
            doc = _read_json(ch / f"resp_{rid}.json")
            if not isinstance(doc, dict):
                faults[rid] = "unparseable self-check response"
                continue
            ok, why = _reply_is_usable(doc, rid, request["submission_sha256"], label=request["submission"])
            if ok and request.get("tuning_overrides"):
                variant = doc.get("variant") or {}
                if (
                    variant.get("kind") != "knob_overrides"
                    or variant.get("applied") is not True
                    or variant.get("tuning_overrides") != request["tuning_overrides"]
                ):
                    ok, why = False, "response does not bind the requested private knob copy"
            if ok:
                responses[rid] = doc
            else:
                faults[rid] = why
        complete = len(responses) + len(faults) == len(manifest["requests"])
        if complete or time.time() >= until:
            break
        time.sleep(0.4)
    rankings, comparisons = _sweep_rank(ws, manifest, responses)
    path = ws / "selfcheck_out" / f"{sid}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "schema": _SWEEP_SCHEMA,
        "sweep_id": sid,
        "official": False,
        "advisory_only": True,
        "status": "complete" if complete else "in_progress",
        "n_completed": len(responses),
        "baseline_status": (
            "current"
            if all(
                _submission_digest(ws / _SUBMISSION) == (responses.get(rid) or {}).get("submission_sha256")
                for rid in manifest.get("baseline_request_ids", [manifest["baseline_request_id"]])
            )
            else "older_source"
        ),
        "n_expected": len(manifest["requests"]),
        "errors": faults,
        "all_pass": (all(d.get("all_pass") is True for d in responses.values()) if complete and not faults else None),
        "ranked_candidates": rankings,
        "comparisons": comparisons,
        "responses": [f".qa_channel/resp_{r['request_id']}.json" for r in manifest["requests"]],
        "resume_with": f"python agent_selfcheck.py --attach {sid}",
        "all": path.relative_to(ws).as_posix(),
    }
    _atomic_write(path, json.dumps(report, indent=2))
    compact = {**report, "ranked_candidates": rankings[:12], "comparisons": comparisons[:12]}
    text = json.dumps(compact, indent=2)
    print(text)
    if a.out:
        Path(a.out).write_text(text)
    if complete:
        return 2 if faults else (0 if report["all_pass"] else 1)
    return 3 if time.time() < manifest["deadline_unix_ns"] / 1e9 else 2
