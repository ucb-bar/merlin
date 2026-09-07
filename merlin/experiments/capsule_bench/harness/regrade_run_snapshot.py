#!/usr/bin/env python3
"""Deterministically regrade an archived submission without mutating its run directory.

The source run supplies identity and expected corpus hashes. Grading uses the current grader against
the current target corpus only after its public contract bytes and hidden snapshot are shown to match
the archived run. Each repeat receives a fresh copy of the exact archived submission.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

import yaml

from merlin.common.paths import merlin_dir, repo_root
from merlin.targetgen.target_experiment import descriptor_for, load_target_experiment


def _contract_digest(roots: list[Path]) -> str:
    """Hash only public, agent-visible capsule contract files across category roots."""
    digest = hashlib.sha256()
    names = frozenset({"capsule.yaml", "capsule.interface.mlir", "README.md"})
    for root in sorted(Path(r) for r in roots):
        category = root.name
        for path in sorted(p for p in root.rglob("*") if p.is_file() and p.name in names):
            rel = f"{category}/{path.relative_to(root).as_posix()}"
            digest.update(rel.encode() + b"\0" + path.read_bytes())
    return digest.hexdigest()


def _signature(score: dict) -> list[dict]:
    """Stable verdict projection: exclude timings and other expected run-to-run diagnostics."""
    rows = []
    for item in score.get("per_capsule", []):
        row = {key: item.get(key) for key in ("capsule", "status", "failure_plane",
                                              "mismatch_count")}
        row["numeric"] = item.get("numeric", item.get("numeric_status"))
        row["trace"] = item.get("trace", item.get("trace_status"))
        row["tiers"] = {
            name: ((value or {}).get("status") if isinstance(value, dict) else value)
            for name, value in sorted((item.get("tiers") or {}).items())
        }
        rows.append(row)
    return sorted(rows, key=lambda row: str(row.get("capsule")))


def _score_summary(score: dict) -> dict:
    rows = score.get("per_capsule", [])
    failure_stages = Counter()
    for row in rows:
        if row.get("status") == "pass":
            continue
        failed_tiers = [name for name, value in sorted((row.get("tiers") or {}).items())
                        if ((value or {}).get("status") if isinstance(value, dict) else value) == "fail"]
        stage = row.get("failure_plane") or (failed_tiers[0] if failed_tiers else None)
        failure_stages[str(stage or row.get("numeric") or row.get("trace") or "unclassified")] += 1
    return {
        "n_passed": score.get("n_passed"),
        "n_capsules": score.get("n_capsules"),
        "n_declined": score.get("n_declined"),
        "statuses": dict(Counter(str(row.get("status")) for row in rows)),
        "failure_stages": dict(failure_stages),
    }


def _completed_repeat_score(repeat: Path) -> Path | None:
    """Return the public score only when the post-freeze grader completed both phases."""
    public = repeat / "grading_public/score_capsule.json"
    hidden = repeat / "grading_hidden/score_capsule.json"
    if public.is_file() and hidden.is_file() and (repeat / "final_report.md").is_file():
        return public
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, help="immutable archived source run")
    parser.add_argument("--out", required=True, help="new output directory; must not already exist")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--resume", action="store_true",
                        help="reuse only fully completed repeats in an interrupted output directory")
    parser.add_argument("--validate-only", action="store_true",
                        help="verify source submission and corpus identity without running the grader")
    args = parser.parse_args(argv)

    source = Path(args.run_dir).resolve(strict=True)
    out = Path(args.out).resolve(strict=False)
    if out.exists() and not args.resume:
        raise SystemExit(f"refusing to overwrite regrade output: {out}")
    if args.resume and not out.is_dir():
        raise SystemExit(f"cannot resume missing regrade output: {out}")
    if args.repeats < 1:
        raise SystemExit("--repeats must be positive")
    environment = yaml.safe_load((source / "environment.yaml").read_text(encoding="utf-8")) or {}
    target = str((environment.get("task_scope") or {}).get("target") or "")
    descriptor = descriptor_for(target) if target else None
    if descriptor is None:
        raise SystemExit(f"cannot resolve target descriptor for archived target {target!r}")
    os.environ["MERLIN_TARGET_EXPERIMENT"] = str(descriptor)
    experiment = load_target_experiment(descriptor)
    source_submission = source / "submission"
    if not (source_submission / "manifest.yaml").is_file():
        raise SystemExit(f"archived submission is missing manifest.yaml: {source_submission}")

    archived_ws = Path(environment.get("workspace_path") or "")
    archived_public = [archived_ws / root.name for root in experiment.graded_roots()]
    if not all(root.is_dir() for root in archived_public):
        raise SystemExit("archived public capsule view is incomplete; cannot prove corpus identity")
    archived_public_digest = _contract_digest(archived_public)
    current_public = experiment.graded_roots()
    current_public_digest = _contract_digest(current_public)
    if current_public_digest != archived_public_digest:
        raise SystemExit("public capsule contract drifted since the archived run; refusing mixed regrade")

    live_hidden_roots = experiment.hidden_roots()
    if len(live_hidden_roots) != 1:
        raise SystemExit(f"expected exactly one hidden root, found {len(live_hidden_roots)}")
    harness = merlin_dir() / "experiments/capsule_bench/harness"
    if str(harness) not in sys.path:
        sys.path.insert(0, str(harness))
    import run_baseline_qa_loop as loop  # noqa: PLC0415
    expected_hidden = (environment.get("hidden_capsule_snapshot") or {}).get("content_sha256")
    hidden_rel = live_hidden_roots[0].resolve(strict=False).relative_to(repo_root())
    candidates = [*live_hidden_roots]
    candidates.extend(sorted((descriptor.parent / "_qa_ws").glob(
        f"*/bundle_inputs/repo/{hidden_rel.as_posix()}")))
    hidden_root = None
    hidden_record = None
    observed = []
    for candidate in candidates:
        try:
            record = loop._subtree_snapshot_record(candidate)
        except (OSError, RuntimeError):
            continue
        observed.append(record["content_sha256"])
        if record["content_sha256"] == expected_hidden:
            hidden_root, hidden_record = candidate, record
            break
    if hidden_root is None or hidden_record is None:
        raise SystemExit("no available hidden capsule snapshot matches the archived run; refusing mixed "
                         f"regrade (expected {expected_hidden}, observed {sorted(set(observed))})")

    validation = {
        "source_run": str(source),
        "target": target,
        "public_contract_sha256": current_public_digest,
        "hidden_capsule_snapshot": hidden_record,
    }
    if args.validate_only:
        print(json.dumps(validation, indent=2))
        return 0

    if not args.resume:
        out.mkdir(parents=True)
    grader = descriptor.parent / "scripts/grade_agent_run.py"
    public_arg = ",".join(str(root) for root in current_public)
    hidden_arg = str(hidden_root)
    signatures: list[dict[str, list[dict]]] = []
    repeats = []
    for index in range(args.repeats):
        repeat = out / f"repeat_{index:02d}"
        score_path = _completed_repeat_score(repeat) if args.resume else None
        reused = score_path is not None
        returncode = None
        if repeat.exists() and not reused:
            raise SystemExit(f"repeat directory exists but is incomplete; refusing overwrite: {repeat}")
        if not reused:
            repeat.mkdir()
            shutil.copytree(source_submission, repeat / "submission",
                            ignore=shutil.ignore_patterns("build", "__pycache__", ".git"))
            command = [sys.executable, str(grader), "--run-dir", str(repeat),
                       "--arm", str(environment.get("arm") or "unknown"),
                       "--model", str(environment.get("model") or "unknown"),
                       "--capsules", public_arg, "--hidden-capsules", hidden_arg]
            proc = subprocess.run(command, cwd=repo_root(), text=True, capture_output=True)
            returncode = proc.returncode
            (repeat / "grader.stdout.log").write_text(proc.stdout, encoding="utf-8")
            (repeat / "grader.stderr.log").write_text(proc.stderr, encoding="utf-8")
            score_path = repeat / "grading_public/score_capsule.json"
        if not score_path.is_file():
            raise SystemExit(f"regrade {index} produced no public score (rc={returncode})")
        score = json.loads(score_path.read_text(encoding="utf-8"))
        hidden_score_path = repeat / "grading_hidden/score_capsule.json"
        if not hidden_score_path.is_file():
            raise SystemExit(f"regrade {index} produced no hidden score (rc={returncode})")
        hidden_score = json.loads(hidden_score_path.read_text(encoding="utf-8"))
        signatures.append({"public": _signature(score), "hidden": _signature(hidden_score)})
        repeats.append({"repeat": index, "grader_returncode": returncode, "reused": reused,
                        "public": _score_summary(score), "hidden": _score_summary(hidden_score),
                        "path": str(repeat)})

    deterministic = all(signature == signatures[0] for signature in signatures[1:])
    summary = {
        **validation,
        "source_submission_sha256": loop._subtree_snapshot_record(source_submission)["content_sha256"],
        "repeats": repeats,
        "deterministic_per_capsule_verdict": deterministic,
    }
    (out / "regrade_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if deterministic else 2


if __name__ == "__main__":
    raise SystemExit(main())
