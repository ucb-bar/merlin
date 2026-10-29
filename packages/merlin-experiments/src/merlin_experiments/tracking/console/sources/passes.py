"""The pass matrix: what the library offers, what the agent's compiler declares and runs, and what it costs.

Rows by ``kind``:

* ``command`` -- the compiler manifest's ``commands`` in pipeline order (declared and used); how many
  capsules exercised each (``qa/tier_state.json`` component fingerprints) and how often a change to it
  invalidated a cached tier result (``[promote] ... invalidated by <command>`` in the launch log);
* ``family`` -- ``shared_pass_contract.families`` with their recorded status;
* ``control`` -- ``shared_pass_contract.controls.choices`` (declared; used when any choice is enabled);
* ``requirement`` -- capsule ``pass_requirements`` tokens and how many capsules require each;
* ``mechanism`` -- the run's optimization catalog (present in the library, not a claim of use).

Profiling rows (``host_pass_profile``) are joined when a record carrying them exists; otherwise
``profile_wall_s`` is None. The manifest is read from the submission on disk, or from the run's
snapshot repository with a read-only ``git show``.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from ..io import clean, is_file, num, obj, rows, scan_dir

CHURN_BYTES = 4 << 20
PROFILE_NAMES = ("pass_observation.json", "host_pass_profile.json", "pipeline_receipt.json")
#: Recorded family statuses that mean the family runs in the compiler (anything else is declared only).
APPLIED_STATES = frozenset({"applied", "active", "enabled", "implemented", "used", "promoted"})


def _manifest(ctx, run: Path, workspace: Path | None) -> tuple[dict | None, str | None]:
    candidates = []
    if workspace is not None:
        candidates.append(workspace / "submission" / "manifest.yaml")
    candidates += [run / "joint_frozen_submission" / "manifest.yaml", run / "oot" / "manifest.yaml"]
    for path in candidates:
        if is_file(path):
            doc = ctx.reader.yaml(path, limit=4 << 20)
            if isinstance(doc, dict):
                return doc, str(path)
    repo = run / "oot"
    head = ctx.reader.identity(repo / ".git" / "HEAD")
    if head is None:
        return None, None
    cached = ctx.state.get("git_manifest")
    if cached and cached[0] == (head, ctx.reader.identity(repo / ".git" / "index")):
        return cached[1], cached[2]
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), "show", "HEAD:manifest.yaml"], capture_output=True, timeout=10, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None, None
    doc = None
    if out.returncode == 0 and len(out.stdout) < 4 << 20:
        import yaml

        try:
            doc = yaml.safe_load(out.stdout)
        except yaml.YAMLError:
            doc = None
    doc = doc if isinstance(doc, dict) else None
    source = f"{repo}@HEAD:manifest.yaml" if doc else None
    ctx.state["git_manifest"] = ((head, ctx.reader.identity(repo / ".git" / "index")), doc, source)
    return doc, source


def churn(lines: list[str]) -> dict[str, dict[str, int]]:
    """``command -> {tier: count}`` from ``[promote] <capsule> <tier> invalidated by <command> (...)`` lines."""
    out: dict[str, dict[str, int]] = {}
    for line in lines:
        head, marker, rest = line.partition(" invalidated by ")
        if not marker or not head.startswith("[promote] "):
            continue
        words = head.split()
        tier = words[-1] if len(words) >= 3 else "?"
        command = rest.split(" ", 1)[0].strip()
        if command:
            counts = out.setdefault(command, {})
            counts[tier] = counts.get(tier, 0) + 1
    return out


def _exercised(ctx, workspace: Path | None) -> dict[str, int]:
    if workspace is None:
        return {}
    ledger = ctx.reader.document(workspace / "qa" / "tier_state.json", limit=16 << 20)
    counts: dict[str, int] = {}
    for name, state in obj(ledger).items():
        if name.startswith("<"):
            continue
        seen = set()
        for tier in ("L2", "L3"):
            seen.update(k for k in obj(obj(obj(state).get(tier)).get("components")) if not k.startswith("<"))
        for command in seen:
            counts[command] = counts.get(command, 0) + 1
    return counts


def _profiles(ctx, run: Path, workspace: Path | None) -> tuple[dict[str, float], str | None]:
    """Summed ``total_wall_ns`` per pass name from a recorded ``host_pass_profile``, if one exists."""
    roots = [p for p in (workspace, run) if p is not None]
    for root in roots:
        for entry in scan_dir(root, 4096):
            if entry.name not in PROFILE_NAMES:
                continue
            doc = ctx.reader.document(Path(entry.path))
            profile = obj(doc).get("host_pass_profile") or obj(doc).get("passes")
            totals: dict[str, float] = {}
            for row in rows(profile):
                wall = num(row.get("total_wall_ns") or row.get("action_wall_ns"))
                if isinstance(row.get("name"), str) and wall is not None:
                    totals[row["name"]] = totals.get(row["name"], 0) + wall / 1e9
            if totals:
                return totals, entry.path
    return {}, None


def _row(name: str, kind: str, **values: Any) -> dict[str, Any]:
    row = {
        "name": clean(name, 80),
        "kind": kind,
        "declared": False,
        "used": None,
        "status": None,
        "boundary": None,
        "required_by": None,
        "exercised_by": None,
        "churn": None,
        "profile_wall_s": None,
        "source": None,
    }
    row.update(values)
    return row


def matrix(ctx) -> tuple[list[dict[str, Any]], dict[str, Any], list[str], dict[str, Any]]:
    run = Path(ctx.experiment["run_dir"])
    workspace = Path(ctx.experiment["workspace"]) if ctx.experiment.get("workspace") else None
    notes: list[str] = []
    manifest, manifest_source = _manifest(ctx, run, workspace)
    commands = obj(obj(manifest).get("commands"))
    contract = obj(obj(manifest).get("shared_pass_contract"))
    exercised = _exercised(ctx, workspace)
    log = Path(ctx.experiment["launch_dir"]) / "process" / "native.log" if ctx.experiment.get("launch_dir") else None
    invalidations = churn(ctx.reader.text_tail(log, CHURN_BYTES)) if log and is_file(log) else {}
    profiles, profile_source = _profiles(ctx, run, workspace)
    out = []
    for name in commands:
        out.append(
            _row(
                name,
                "command",
                declared=True,
                used=True,
                exercised_by=exercised.get(name, 0) if exercised else None,
                churn=sum(invalidations.get(name, {}).values()) if log else None,
                profile_wall_s=profiles.get(name),
                source=manifest_source,
            )
        )
    boundaries = {b: obj(v).get("command") for b, v in obj(contract.get("boundaries")).items()}
    for family in rows(contract.get("families"), 256):
        out.append(
            _row(
                family.get("name") or "?",
                "family",
                declared=True,
                status=clean(family.get("status"), 24) or None,
                boundary=clean(family.get("boundary"), 32) or None,
                used=None,
                profile_wall_s=profiles.get(family.get("name")),
                source=manifest_source,
            )
        )
    for name, choice in obj(obj(contract.get("controls")).get("choices")).items():
        enabled = obj(choice).get("enabled")
        out.append(
            _row(
                name,
                "control",
                declared=True,
                used=any(v is True for v in enabled) if isinstance(enabled, list) else None,
                source=manifest_source,
            )
        )
    requirements = obj(obj(obj(ctx.shared.get("capsules")).get("counts")).get("pass_requirements"))
    for token, count in sorted(requirements.items()):
        out.append(_row(token, "requirement", required_by=count, source="capsule pass_requirements"))
    catalog = ctx.reader.document(run / "OPTIMIZATION_CATALOG.json", limit=16 << 20)
    mechanisms = rows(obj(catalog).get("mechanisms"), 512)
    for mechanism in mechanisms:
        compiler = obj(mechanism.get("compiler"))
        out.append(
            _row(
                mechanism.get("id") or "?",
                "mechanism",
                status=clean(mechanism.get("facet"), 24) or None,
                source=clean(compiler.get("module"), 80) or "OPTIMIZATION_CATALOG.json",
            )
        )
    library = ctx.reader.document(run / "COMPILER_LIBRARY_API.json", limit=32 << 20)
    modules = obj(library).get("public_modules")
    if manifest is None:
        notes.append("compiler manifest not found in the submission or the snapshot repository")
    if log is None or not is_file(log):
        notes.append("launch log not available; invalidation churn not recorded")
    if not profiles:
        notes.append("no host_pass_profile record in this run; pass wall time not recorded")
    counts = {
        "library_modules": len(modules) if isinstance(modules, list) else None,
        "catalog_mechanisms": len(mechanisms) if catalog else None,
        "declared": sum(1 for r in out if r["declared"]),
        "used": sum(1 for r in out if r["used"]),
        "required": len(requirements),
        "profiled": len(profiles),
    }
    detail = {
        "commands": commands,
        "families": rows(contract.get("families"), 256),
        "boundaries": boundaries,
        "invalidations": invalidations,
        "mechanisms": mechanisms,
        "manifest_source": manifest_source,
        "profile_source": profile_source,
    }
    return out, counts, notes, detail


def read(ctx) -> dict[str, Any]:
    out, counts, notes, detail = matrix(ctx)
    by_command = {command: boundary for boundary, command in detail["boundaries"].items() if command}
    # What the compiler actually runs: its manifest commands in pipeline order, plus the shared-pass
    # families and controls recorded as applied or enabled. Declared-only rows stay in the matrix.
    applied = []
    for order, row in enumerate((r for r in out if r["kind"] == "command"), start=1):
        applied.append({**row, "order": order, "boundary": by_command.get(row["name"])})
    for row in out:
        if row["kind"] == "family" and str(row.get("status") or "").lower() in APPLIED_STATES:
            applied.append({**row, "order": None})
        elif row["kind"] == "control" and row.get("used"):
            applied.append({**row, "order": None})
    counts["applied"] = len(applied)
    counts["not_applied"] = sum(1 for r in out if r["declared"] and r not in applied and r["kind"] != "command")
    return {
        "passes": out,
        "applied": applied,
        "counts": counts,
        "pipeline": [clean(n, 80) for n in detail["commands"]],
        "boundaries": detail["boundaries"],
        "notes": notes,
    }


def inspect(ctx, key: str) -> dict[str, Any]:
    out, _, _, detail = matrix(ctx)
    matches = [r for r in out if r["name"] == key]
    if not matches:
        return {"title": f"pass {clean(key, 80)}", "sections": [], "paths": [], "notice": "not in the pass matrix"}
    sections = []
    for row in matches:
        lines = [f"{k}: {v}" for k, v in row.items() if k not in ("name", "kind") and v is not None]
        if row["kind"] == "command":
            argv = obj(detail["commands"].get(key)).get("argv")
            if isinstance(argv, list):
                lines.append("argv " + clean(" ".join(str(a) for a in argv), 300))
            for tier, count in sorted(obj(detail["invalidations"].get(key)).items()):
                lines.append(f"invalidated {tier} results {count} times (bounded log tail)")
            for boundary, command in detail["boundaries"].items():
                if command == key:
                    lines.append(f"shared-pass boundary: {boundary}")
        elif row["kind"] == "family":
            family = next((f for f in detail["families"] if f.get("name") == key), {})
            lines.append("requirements " + ", ".join(clean(r, 40) for r in family.get("requirements") or []))
            if family.get("reason"):
                lines.append("reason " + clean(family["reason"], 300))
        elif row["kind"] == "mechanism":
            mechanism = next((m for m in detail["mechanisms"] if m.get("id") == key), {})
            compiler = obj(mechanism.get("compiler"))
            lines += [
                f"mechanism {clean(mechanism.get('mechanism'), 300)}",
                f"{clean(compiler.get('module'), 80)}.{clean(compiler.get('symbol'), 60)}"
                f" ({clean(compiler.get('path'), 120)})",
                "required traits " + ", ".join(clean(t, 30) for t in mechanism.get("required_traits") or []),
            ]
        elif row["kind"] == "requirement" and ctx.operator_private:
            names = [
                c["name"]
                for c in obj(ctx.shared.get("capsules")).get("capsules") or []
                if key in (c.get("pass_requirements") or [])
            ]
            lines.append(f"required by {len(names)}: " + ", ".join(names[:30]) + (" ..." if len(names) > 30 else ""))
        sections.append({"title": row["kind"], "lines": lines})
    paths = [p for p in (detail["manifest_source"], detail["profile_source"]) if p]
    return {"title": f"pass {key}", "sections": sections, "paths": paths}


__all__ = ["churn", "inspect", "matrix", "read"]
