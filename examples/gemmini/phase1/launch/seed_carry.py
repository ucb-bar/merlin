"""What a seeded launch carries from the seed run besides its compiler and its verdicts.

``fresh_seed.build_evidence`` calls :func:`build` for the ``variants`` and ``performance`` carry keys.
Everything lands in the reviewed evidence directory (pinned by its tree digest with the rest of the
seed evidence). ``joint_seed.stage_evidence`` copies it to the workspace root, outside ``submission/``,
and ``joint_seed.stage_run_evidence`` copies the ``RUN_PRIVATE/`` part into the run directory, where
the grader (not the author) reads it:

* ``SEED_VARIANTS/<name>/``: every variant tree the seed run left, whole (``build``, caches and links
  left out, as the tree digest leaves them out), and ``SEED_VARIANTS/OUTCOMES.json``: per variant the
  files it changed against the seed compiler and the seed run's own records that name it;
* ``SEED_PERFORMANCE/``: the seed run's per-case performance history (``perf_history.jsonl``) and
  ``baseline.json``, the per-case row of its last completed grade (cycles, roofline, utilization,
  headroom, fences), the baseline every "vs seed" figure is read against;
* ``SEED_NOTES/workspace/`` (the seed author's own workspace notes), ``SEED_TOOLS/`` (its helper
  scripts) and ``SEED_EXPERIMENTS/evidence/`` (its evidence records, bounded);
* ``RUN_PRIVATE/seed_timing/<elf sha256>.json``: the seed run's complete timing-model projections keyed
  by the bytes of the executable they observed, so an unchanged executable has model cycles from the
  first grade without a new Spike observation (``feedback.seed_timing``);
  ``RUN_PRIVATE/joint_fast_estimate_pairs.json``: calibration pairs from the seed's complete estimates
  (census of the same program, complete timing-model cycles), so the fast estimate's own fit is ready
  at minute one; ``RUN_PRIVATE/seed_baseline.json``: ``baseline.json`` again, for the feedback.

A from-scratch plan never reaches this module.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

VARIANTS_DIR, PERFORMANCE_DIR, TOOLS_DIR, NOTES_DIR = "SEED_VARIANTS", "SEED_PERFORMANCE", "SEED_TOOLS", "SEED_NOTES"
EXPERIMENTS_DIR, RUN_PRIVATE, INDEX = "SEED_EXPERIMENTS", "RUN_PRIVATE", "SEED_INDEX.md"
#: Left out of every copied tree, exactly as ``merlin.common.tree_hash`` leaves them out of a digest.
SKIP = frozenset({"build", "__pycache__", ".git"})
#: Workspace-root notes the harness writes for every run; the seed author's own notes are the others.
HARNESS_NOTES = frozenset({"TASK.md", "ALLOWED_MERLIN_TOOLS.md", "PUBLIC_CAPABILITIES.md", "PREVIOUS_SESSION.md"})
MAX_EVIDENCE_BYTES = 256 << 10
BASELINE_SCHEMA = "merlin.phase1.seed_baseline.v1"
TIMING_SCHEMA = "merlin.phase1.seed_timing.v1"


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


class _Store:
    """Copies files into the evidence directory, hard-linking repeated contents to one inode there.

    Only this operator-owned directory is linked; the workspace stages plain copies of it, and nothing
    is ever linked to the seed run's own files."""

    def __init__(self):
        self.by_digest: dict[str, Path] = {}

    def put(self, source: Path, target: Path) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        digest = _sha(source)
        first = self.by_digest.get(digest)
        if first is not None:
            os.link(first, target)
            return
        shutil.copyfile(source, target)
        target.chmod(0o644)
        self.by_digest[digest] = target


def _files(root: Path):
    """``{relative posix path: path}`` of the regular files under ``root``, as the tree digest sees them."""
    out = {}
    for path in sorted(Path(root).rglob("*")):
        relative = path.relative_to(root)
        if SKIP & set(relative.parts) or path.is_symlink() or not path.is_file():
            continue
        out[relative.as_posix()] = path
    return out


# --------------------------------------------------------------------------- variants


def variants(workspace: Path, seed_source: Path, out: Path, store: _Store) -> dict:
    """Copy every variant tree and record what it changed and which seed records name it."""
    root = Path(workspace) / "variants"
    baseline = {k: _sha(p) for k, p in _files(seed_source).items()} if Path(seed_source).is_dir() else {}
    records = sorted(
        f"{kind}/{p.name}" for kind in ("evidence", "ab") for p in (Path(workspace) / kind).glob("*")
    )
    outcomes = {}
    for directory in sorted(p for p in root.glob("*") if p.is_dir() and not p.is_symlink()):
        files = _files(directory)
        for relative, path in files.items():
            store.put(path, out / VARIANTS_DIR / directory.name / relative)
        digests = {k: _sha(p) for k, p in files.items()}
        changed = sorted(k for k, v in digests.items() if k in baseline and baseline[k] != v)
        added = sorted(k for k in digests if k not in baseline)
        removed = sorted(k for k in baseline if k not in digests)
        outcomes[directory.name] = {
            "files": len(files),
            "changed": changed[:200],
            "added": added[:200],
            "removed": removed[:200],
            "changed_count": len(changed) + len(added) + len(removed),
            "seed_records": [r for r in records if directory.name in r][:40],
        }
    if outcomes:
        (out / VARIANTS_DIR).mkdir(parents=True, exist_ok=True)
        (out / VARIANTS_DIR / "OUTCOMES.json").write_text(json.dumps({
            "against": "the seed compiler (submission/ at the start of this run)",
            "seed_records": "paths under SEED_EXPERIMENTS/ (evidence/ and ab/ of the seed workspace)",
            "variants": outcomes}, indent=1) + "\n")
    return {"variants": len(outcomes)}


# --------------------------------------------------------------------------- performance


def _number(value):
    return value if type(value) in (int, float) and value == value and value >= 0 else None


def _headroom_entries(grade: dict) -> dict:
    brief = (grade or {}).get("feedback_brief") or {}
    block = ((brief.get("performance_priority") or {}).get("development_headroom") or {})
    out = {}
    for entry in block.get("largest_headroom") or ():
        if isinstance(entry, dict) and isinstance(entry.get("test_case"), str):
            out[entry["test_case"]] = entry
    return out


def _history(workspace: Path) -> dict:
    """``{case: newest modeled perf-history row}`` and ``{case: fewest modeled cycles}``."""
    newest, best = {}, {}
    try:
        lines = (Path(workspace) / "qa" / "perf_history.jsonl").read_text(encoding="utf-8").splitlines()
    except OSError:
        return {"newest": newest, "best": best}
    for line in lines:
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if not isinstance(row, dict) or row.get("kind") != "modeled" or _number(row.get("cycles")) is None:
            continue
        case = row.get("test_case")
        newest[case] = row
        best[case] = min(best.get(case, row["cycles"]), row["cycles"])
    return {"newest": newest, "best": best}


def baseline(workspace: Path, seed_run: str) -> dict:
    """Per case, the seed's last completed grade row with its performance numbers."""
    workspace = Path(workspace)
    grade = _json(workspace / "qa" / "latest_completed_grade.json") or {}
    headroom = _headroom_entries(grade)
    history = _history(workspace)
    cases = {}
    for row in grade.get("per_capsule") or ():
        if not isinstance(row, dict) or not isinstance(row.get("capsule"), str):
            continue
        name = row["capsule"]
        detail = _json(workspace / ((row.get("detail") or {}).get("path") or "")) or {}
        performance = detail.get("performance") if isinstance(detail.get("performance"), dict) else {}
        compact = row.get("performance") if isinstance(row.get("performance"), dict) else {}
        complete = performance.get("status") in ("PROVISIONAL", "QUALIFIED") and performance.get(
            "complete_event_accounting") is True
        roofline = performance.get("structural_roofline") or {}
        counts = ((performance.get("feature_counts") or {}).get("target") or {})
        fast = performance.get("fast_estimate") if isinstance(performance.get("fast_estimate"), dict) else {}
        listed = headroom.get(name) or {}
        fence = listed.get("fence_overlap") if isinstance(listed.get("fence_overlap"), dict) else {}
        modeled = history["newest"].get(name) or {}
        cycles = _number(performance.get("estimated_whole_soc_cycles")) if complete else None
        evidence = "modeled: Spike timing model (seed's last completed grade)" if cycles is not None else None
        if cycles is None and _number(modeled.get("cycles")) is not None:
            cycles, evidence = modeled["cycles"], "modeled: Spike timing model (seed's newest timed row)"
        if cycles is None and fast.get("status") == "ESTIMATED" and _number(fast.get("cycles")) is not None:
            cycles, evidence = fast["cycles"], "estimated: fast census estimate (seed)"
        cases[name] = {
            "status": row.get("status"),
            "cycles": cycles,
            "evidence": evidence,
            "best_cycles": history["best"].get(name),
            "model_sha256": performance.get("model_sha256") or (modeled.get("performance_identity") or {}).get(
                "model_sha256"),
            "selection_sha256": performance.get("selection_sha256"),
            "elf_sha256": performance.get("elf_sha256"),
            "roofline_bound_cycles": _number(roofline.get("structural_compute_floor_cycles")),
            "bound_by": compact.get("bound_by") or modeled.get("bound_by"),
            "array_utilization_pct": compact.get("array_utilization_pct", modeled.get("array_utilization_pct")),
            "headroom_cycles": _number(listed.get("headroom_cycles"))
            or _number((performance.get("search_priority") or {}).get("diagnostic_headroom")),
            "fences": _number(counts.get("fences")),
            "fence_per_tile": _number(fence.get("fence_per_tile")),
            "graded_source_sha256": (row.get("graded") or {}).get("graded_source_sha256"),
            "currency": (row.get("graded") or {}).get("currency"),
        }
    return {
        "schema": BASELINE_SCHEMA,
        "seed_run": seed_run,
        "graded_source_sha256": grade.get("graded_source_sha256"),
        "graded_at": grade.get("graded_at"),
        "note": ("The seed's per-case numbers from its last completed grade (a row may come from an older "
                 "seed version: see currency). Model cycles compare only under the same model_sha256."),
        "cases": cases,
    }


def timing(workspace: Path, out: Path, seed_run: str) -> dict:
    """The seed's complete timing-model projections keyed by the executable bytes they observed."""
    newest = {}
    for path in sorted((Path(workspace) / "qa" / "cases").glob("*/*.json")):
        detail = _json(path)
        if not isinstance(detail, dict):
            continue
        performance = detail.get("performance")
        if (not isinstance(performance, dict) or performance.get("status") not in ("PROVISIONAL", "QUALIFIED")
                or performance.get("complete_event_accounting") is not True
                or not isinstance(performance.get("elf_sha256"), str)
                or _number(performance.get("estimated_whole_soc_cycles")) is None):
            continue
        key = tuple(performance.get(k) for k in (
            "elf_sha256", "capsule_sha256", "selection_sha256", "model_sha256", "facts_sha256", "timing_epoch"))
        record = {
            "capsule": detail.get("capsule"),
            "key": dict(zip(("elf_sha256", "capsule_sha256", "selection_sha256", "model_sha256", "facts_sha256",
                             "timing_epoch"), key)),
            "seed_run": seed_run,
            "performance": performance,
            "compiler_diagnostics": detail.get("compiler_diagnostics"),
            "mtime_ns": path.stat().st_mtime_ns,
        }
        held = newest.get((detail.get("capsule"), *key))
        if held is None or held["mtime_ns"] < record["mtime_ns"]:
            newest[(detail.get("capsule"), *key)] = record
    by_elf: dict = {}
    for record in newest.values():
        record.pop("mtime_ns")
        by_elf.setdefault(record["key"]["elf_sha256"], []).append(record)
    directory = out / RUN_PRIVATE / "seed_timing"
    for elf, records in by_elf.items():
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"{elf}.json").write_text(json.dumps(
            {"schema": TIMING_SCHEMA, "seed_run": seed_run, "records": records}, sort_keys=True) + "\n")
    return {"timed_executables": len(by_elf), "timed_records": len(newest)}


def calibration_pairs(workspace: Path, run_dir: Path, out: Path) -> dict:
    """Fast-estimate calibration pairs from the seed's complete estimates and the census of the same ELF.

    Every complete estimate the seed recorded (``qa/cases``) is paired with the decoded instruction
    stream of the executable it timed, from any retained case work tree of the seed run whose
    executable has those bytes (the executable itself, else the digest its readback build recorded:
    retention keeps the decoded stream longer than the executable)."""
    from merlin.targetgen.elf_lanes import PACKAGE_ELF_NAME
    from merlin_experiments.phase1.feedback import fast_estimate as FE

    run_dir = Path(run_dir)
    ledger = _json(run_dir / FE.LEDGER) or {}
    pairs = [p for p in ledger.get("pairs") or () if isinstance(p, dict)]
    roots = {}
    for root in [*run_dir.glob("_qa_work/fruns_*/runs/*/*"), *run_dir.glob("_qa_work/selfchecks/*/runs/*/*")]:
        if not (root / "generated" / FE.TRACE).is_file():
            continue
        elf = root / "generated" / PACKAGE_ELF_NAME
        digest = _sha(elf) if elf.is_file() else ((_json(root / "generated" / "readback_build.json") or {}).get(
            "elf_sha256"))
        if isinstance(digest, str):
            roots.setdefault((root.name, digest), root)
    seen, counts = set(), {"complete_estimates": 0, "no_work_tree": 0, "partial_census": 0}
    for path in sorted((Path(workspace) / "qa" / "cases").glob("*/*.json")):
        detail = _json(path) or {}
        performance = detail.get("performance") if isinstance(detail.get("performance"), dict) else {}
        cycles, elf_sha = _number(performance.get("estimated_whole_soc_cycles")), performance.get("elf_sha256")
        if performance.get("complete_event_accounting") is not True or not cycles or elf_sha in seen:
            continue
        seen.add(elf_sha)
        counts["complete_estimates"] += 1
        root = roots.get((detail.get("capsule"), elf_sha))
        if root is None:
            counts["no_work_tree"] += 1
            continue
        counted = FE.census(FE.trace_for(root))
        if not counted or not counted.get("complete"):
            counts["partial_census"] += 1
            continue
        pairs.append({"floor": FE.floor(performance.get("structural_roofline")), "commands": counted["commands"],
                      "fences": counted["fences"], "cycles": cycles, "facts_sha256": performance.get("facts_sha256")})
    if not pairs:
        return {"calibration_pairs": 0, "calibration_sources": counts}
    fits = {}
    for facts in {p.get("facts_sha256") for p in pairs}:
        same = [p for p in pairs if p.get("facts_sha256") == facts]
        fits[str(facts)] = FE.fit([(p.get("floor"), p.get("commands"), p.get("fences"), p.get("cycles")) for p in same])
    target = out / RUN_PRIVATE / FE.LEDGER
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({"schema": FE.SCHEMA, "pairs": pairs[-FE._KEEP:], "fits": fits}, sort_keys=True))
    return {"calibration_pairs": len(pairs), "calibration_sources": counts}


def performance(workspace: Path, run_dir: Path, out: Path, seed_run: str) -> dict:
    workspace, out = Path(workspace), Path(out)
    (out / PERFORMANCE_DIR).mkdir(parents=True, exist_ok=True)
    history = workspace / "qa" / "perf_history.jsonl"
    if history.is_file():
        shutil.copyfile(history, out / PERFORMANCE_DIR / "perf_history.jsonl")
    document = baseline(workspace, seed_run)
    text = json.dumps(document, indent=1, sort_keys=True) + "\n"
    (out / PERFORMANCE_DIR / "baseline.json").write_text(text)
    (out / RUN_PRIVATE).mkdir(parents=True, exist_ok=True)
    (out / RUN_PRIVATE / "seed_baseline.json").write_text(text)
    summary = {"perf_history_rows": sum(1 for _ in history.open()) if history.is_file() else 0,
               "baseline_cases": len(document["cases"]),
               "baseline_cases_with_cycles": sum(1 for c in document["cases"].values() if c["cycles"] is not None)}
    summary.update(timing(workspace, out, seed_run))
    summary.update(calibration_pairs(workspace, run_dir, out))
    return summary


# --------------------------------------------------------------------------- notes, tools, records


def workspace_notes(workspace: Path, out: Path, store: _Store) -> dict:
    workspace, copied = Path(workspace), {"notes": [], "tools": 0, "evidence": 0}
    for path in sorted(workspace.glob("*.md")):
        if path.is_file() and not path.is_symlink() and path.name not in HARNESS_NOTES:
            store.put(path, out / NOTES_DIR / "workspace" / path.name)
            copied["notes"].append(path.name)
    for relative, path in _files(workspace / "tools").items() if (workspace / "tools").is_dir() else ():
        store.put(path, out / TOOLS_DIR / relative)
        copied["tools"] += 1
    for relative, path in _files(workspace / "evidence").items() if (workspace / "evidence").is_dir() else ():
        target = out / EXPERIMENTS_DIR / "evidence" / relative
        if path.stat().st_size > MAX_EVIDENCE_BYTES:
            target.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "rb") as handle:
                target.write_bytes(handle.read(MAX_EVIDENCE_BYTES) + b"\n[truncated by the seeded launch]\n")
        else:
            store.put(path, target)
        copied["evidence"] += 1
    return copied


INDEX_LINES = (
    ("SEED_INDEX.md", "this list: everything the previous run left you, outside submission/"),
    ("PREVIOUS_SESSION.md", "where the previous session stopped and what it was doing next"),
    ("OPTIMIZATION_RULES.json", "the previous run's optimization rules with their evidence; keep maintaining it"),
    ("SEED_VARIANTS/", "every variant compiler tree the previous run left, whole; OUTCOMES.json says what each "
                       "changed against submission/ and which records name it. Copy one into variants/ to "
                       "continue it, or into submission/ to promote it"),
    ("SEED_PERFORMANCE/baseline.json", "per case, the previous run's cycles, roofline, utilization, headroom and "
                                       "fences: the baseline your grades are compared with (vs_seed)"),
    ("SEED_PERFORMANCE/perf_history.jsonl", "the previous run's per-case performance history, every grade"),
    ("SEED_EXPERIMENTS/", "the previous run's A/B records (ab/), its evidence records (evidence/) and an index"),
    ("SEED_TOOLS/", "the helper scripts the previous author wrote"),
    ("SEED_NOTES/", "the previous run's reports and notes (workspace/ holds its own workspace notes); may be stale"),
    ("SEED_EVIDENCE/", "operator-prepared seed analyses (baseline map, structure diff), when present"),
)


def write_index(out: Path, summary: dict) -> None:
    """``SEED_INDEX.md``: one plain line for each carried item that is present."""
    out = Path(out)
    lines = ["# What the previous run left you", "",
             "Everything below sits at the workspace root, outside `submission/`. The compiler in `submission/` "
             "is the previous run's; its proven verdicts and model cycles carry for unchanged executables, and "
             "the timing-model calibration starts from its estimates.", ""]
    for name, text in INDEX_LINES:
        present = name in ("SEED_INDEX.md", "PREVIOUS_SESSION.md") or (out / name.rstrip("/")).exists()
        if present:
            lines.append(f"* `{name}`: {text}.")
    counts = summary.get("performance") or {}
    if counts:
        lines += ["", f"Seed baseline: {counts.get('baseline_cases_with_cycles', 0)} of {counts.get('baseline_cases', 0)} "
                  f"cases with cycles; {counts.get('timed_executables', 0)} executables carry their model cycles; "
                  f"{counts.get('calibration_pairs', 0)} calibration pairs."]
    (out / INDEX).write_text("\n".join(lines) + "\n")


def build(seed: dict, source: Path, out: Path) -> dict:
    """Stage the variants and performance carry into ``out``; returns counts for the plan record."""
    run, carry = seed.get("run") or {}, seed["carry"]
    workspace = Path(run["workspace"]) if run.get("workspace") else None
    summary, store = {}, _Store()
    if workspace is None or not workspace.is_dir():
        return summary
    if carry.get("variants"):
        summary["variants"] = variants(workspace, Path(source), Path(out), store)
        summary["workspace_notes"] = workspace_notes(workspace, Path(out), store)
    if carry.get("performance"):
        summary["performance"] = performance(workspace, Path(run["run_dir"]), Path(out), run["run_id"])
    return summary
