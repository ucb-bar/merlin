"""Materialize and grade a descriptor-declared post-search evaluation cohort.

Search and evaluation answer different questions.  The paid search cohort may provide iterative
feedback at a fast oracle tier; this module creates a fresh, immutable view for a named evaluation
stage and makes that stage's oracle tier mandatory in the copied capsule contracts.  It therefore
cannot silently turn an optional GSIM observation into a certified result.

The source capsules are never changed.  The emitted manifest records the descriptor, source and
materialized tree digests, stage ordering, and the exact engine preflight used for the run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from merlin.common.paths import repo_root
from .contract.materialize import materialize_public_capsules
from .target_experiment import TargetExperiment, descriptor_for, load_target_experiment


def _tree_sha256(root: Path, *, exclude: frozenset[str] = frozenset()) -> str:
    """Content-and-relative-path digest for a symlink-free capsule/cohort tree."""
    rows: list[bytes] = []
    for path in sorted(root.rglob("*")):
        if path.relative_to(root).as_posix() in exclude:
            continue
        if path.is_symlink():
            raise ValueError(f"evaluation input contains a symlink: {path}")
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix().encode("utf-8")
        rows.append(rel + b"\0" + hashlib.sha256(path.read_bytes()).digest())
    return hashlib.sha256(b"\n".join(rows)).hexdigest()


def _source_capsules(te: TargetExperiment) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for root in te.graded_roots():
        for path in sorted(root.glob("*/capsule.yaml")):
            name = path.parent.name
            if name in out:
                raise ValueError(f"duplicate capsule {name!r} across {te.graded_roots()}")
            doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            if doc.get("label") in {"public", "dev"}:
                out[name] = path.parent
    return out


def engine_preflight(te: TargetExperiment, stage_name: str) -> dict[str, Any]:
    """Resolve the declared evaluation engine and reject unattributed executable bytes."""
    from .capsule_runner import describe_l3_engine

    stage = te.evaluation_cohort(stage_name)
    requested = str(stage["oracle_engine"])
    tier = str(stage["oracle_tier"])
    if tier != "L3":
        # L3 is the only post-loop engine selector currently exposed by the target-neutral runner.
        # Refuse instead of pretending the L3 selection says anything about another tier.
        raise ValueError(f"evaluation engine preflight currently supports L3, got {tier}")
    selected = describe_l3_engine(te.target, te.sim_via)
    reason = str(selected.get("reason") or "")
    problems = []
    if selected.get("available") is not True:
        problems.append("declared evaluation engine is unavailable")
    if selected.get("engine") != requested:
        problems.append(
            f"descriptor requests {requested!r}, resolver selected {selected.get('engine')!r}")
    if "UNRECORDED" in reason or "no build receipt" in reason.lower():
        problems.append("selected engine executable is not tied to an RTL build receipt")
    return {
        "stage": stage_name,
        "required_tier": tier,
        "requested_engine": requested,
        "selected": selected,
        "ok": not problems,
        "problems": problems,
    }


def _predecessor_pass_evidence(
    te: TargetExperiment,
    stage_name: str,
    candidate: Path,
    predecessor_cohort: str | Path | None,
    predecessor_score: str | Path | None,
) -> dict[str, Any] | None:
    """Validate the physical pass that gates a dependent evaluation stage.

    A descriptor's ``after: <stage>_pass`` is an executable dependency, not documentation.  The
    predecessor score must cover exactly the predecessor cohort, clear its mandatory physical tier,
    and name the same byte-frozen package.  Both evidence files remain content-addressed dependencies
    of the new cohort so editing a result after materialization is detected.
    """
    after = str(te.evaluation_cohort(stage_name)["after"])
    if after == "search_converged":
        if predecessor_cohort is not None or predecessor_score is not None:
            raise ValueError(
                f"evaluation stage {stage_name!r} follows search convergence, not an evaluation pass")
        return None
    if not after.endswith("_pass"):
        raise ValueError(f"evaluation stage {stage_name!r} has unsupported dependency {after!r}")
    predecessor_stage = after.removesuffix("_pass")
    if predecessor_cohort is None or predecessor_score is None:
        raise ValueError(
            f"evaluation stage {stage_name!r} requires --predecessor-cohort and "
            f"--predecessor-score proving {predecessor_stage!r} passed")

    cohort_root = Path(predecessor_cohort)
    score_path = Path(predecessor_score)
    if cohort_root.is_symlink() or not cohort_root.is_dir():
        raise ValueError(f"predecessor cohort is not a regular directory: {cohort_root}")
    if score_path.is_symlink() or not score_path.is_file():
        raise ValueError(f"predecessor score is not a regular file: {score_path}")
    prior = validate_evaluation_cohort(cohort_root, te, candidate)
    if prior.get("stage") != predecessor_stage:
        raise ValueError(
            f"predecessor cohort is stage {prior.get('stage')!r}, expected {predecessor_stage!r}")

    score = json.loads(score_path.read_text(encoding="utf-8"))
    expected_names = sorted(row["name"] for row in prior["capsules"])
    expected_n = len(expected_names)
    problems: list[str] = []
    package = score.get("package")
    if not isinstance(package, str) or Path(package).resolve() != candidate.resolve():
        problems.append("score package does not resolve to the frozen candidate")
    if score.get("integrity_status") != "clean":
        problems.append("score integrity_status is not clean")
    if score.get("gradeable") is not True:
        problems.append("score is not gradeable")
    if score.get("n_capsules") != expected_n or score.get("n_passed") != expected_n or expected_n == 0:
        problems.append(
            f"score is not an exact all-pass ({score.get('n_passed')}/{score.get('n_capsules')}, "
            f"expected {expected_n}/{expected_n})")

    rows = score.get("per_capsule")
    if not isinstance(rows, list):
        problems.append("score has no per_capsule evidence")
        rows = []
    row_names = [row.get("capsule") for row in rows if isinstance(row, dict)]
    if len(row_names) != len(set(row_names)) or sorted(row_names) != expected_names:
        problems.append("score per_capsule names do not exactly cover the predecessor cohort")
    required_tier = str(prior["required_oracle_tier"])
    for row in rows:
        if not isinstance(row, dict):
            problems.append("score contains a malformed per_capsule row")
            continue
        if row.get("status") != "pass" or (row.get("tiers") or {}).get(required_tier) != "pass":
            problems.append(
                f"{row.get('capsule', '<unnamed>')} did not pass required tier {required_tier}")
    if (score.get("pass_evidence") or {}).get("rtl_backed") != expected_n:
        problems.append("not every predecessor pass is backed by elaborated-RTL evidence")
    if problems:
        raise ValueError("predecessor pass evidence rejected: " + "; ".join(problems))

    return {
        "stage": predecessor_stage,
        "cohort": str(cohort_root.resolve()),
        "cohort_manifest_sha256": hashlib.sha256(
            (cohort_root / ".evaluation_cohort.json").read_bytes()).hexdigest(),
        "score": str(score_path.resolve()),
        "score_sha256": hashlib.sha256(score_path.read_bytes()).hexdigest(),
        "candidate_tree_sha256": prior["candidate_tree_sha256"],
        "required_oracle_tier": required_tier,
        "n_capsules": expected_n,
        "n_passed": expected_n,
        "rtl_backed": expected_n,
    }


def materialize_evaluation_cohort(
    dest: str | Path,
    te: TargetExperiment,
    stage_name: str,
    candidate: str | Path,
    *,
    predecessor_cohort: str | Path | None = None,
    predecessor_score: str | Path | None = None,
) -> dict[str, Any]:
    """Create one descriptor-declared frozen-candidate cohort at ``dest``.

    ``dest`` must be absent or empty.  Refusing an existing populated directory avoids combining two
    descriptor revisions or leaving stale capsules in the denominator.
    """
    stage = te.evaluation_cohort(stage_name)
    dest = Path(dest)
    candidate = Path(candidate)
    if candidate.is_symlink() or not candidate.is_dir():
        raise ValueError(f"frozen evaluation candidate is not a regular directory: {candidate}")
    if not any(path.is_file() for path in candidate.rglob("*")):
        raise ValueError(f"frozen evaluation candidate has no files: {candidate}")
    candidate_digest = _tree_sha256(candidate)
    predecessor = _predecessor_pass_evidence(
        te, stage_name, candidate, predecessor_cohort, predecessor_score)
    if dest.exists() and any(dest.iterdir()):
        raise ValueError(f"evaluation destination is not empty: {dest}")
    dest.mkdir(parents=True, exist_ok=True)

    sources = _source_capsules(te)
    include = tuple(stage["include_capsules"])
    missing = sorted(set(include) - set(sources))
    if missing:
        raise ValueError(f"evaluation cohort {stage_name!r} names absent capsules: {missing}")
    required_roles = set(stage.get("require_source_roles") or ())
    source_records = []
    for name in include:
        src = sources[name]
        doc = yaml.safe_load((src / "capsule.yaml").read_text(encoding="utf-8")) or {}
        role = str(doc.get("source_role") or "")
        if required_roles and role not in required_roles:
            raise ValueError(
                f"evaluation cohort {stage_name!r} requires source role(s) "
                f"{sorted(required_roles)}, but {name} is {role!r}")
        source_records.append({"name": name, "source_role": role,
                               "source_tree_sha256": _tree_sha256(src)})

    excluded = set(sources) - set(include)
    written = materialize_public_capsules(
        dest, tier_ceiling=str(stage["oracle_tier"]), corpus_roots=te.graded_roots(),
        exclude=excluded,
    )
    if set(written) != set(include):
        raise ValueError(
            f"materialized cohort differs from declaration: wrote={written}, declared={sorted(include)}")

    # This is a POST-SEARCH certification contract.  The source capsule intentionally leaves L3
    # optional so cheap search can use its sibling; in this isolated copy the stage's tier is mandatory.
    required_tier = str(stage["oracle_tier"])
    for name in written:
        path = dest / name / "capsule.yaml"
        doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        tiers = [str(t) for t in (doc.get("required_oracle_tiers") or [])]
        if required_tier not in tiers:
            tiers.append(required_tier)
        doc["required_oracle_tiers"] = sorted(
            set(tiers), key=lambda t: int(t[1:]) if t.startswith("L") and t[1:].isdigit() else 999)
        doc.pop("unreachable_required_oracle_tiers", None)
        doc.pop("oracle_tier_ceiling", None)
        # Application-derived search capsules normally cap themselves at L2 and point at a reduced
        # `_l3` sibling.  A frozen direct-evaluation copy deliberately lifts that search-time cap: it
        # is the same full workload, now required to execute at the stage's declared physical tier.
        doc.pop("max_oracle_tier", None)
        doc["evaluation_stage"] = {
            "name": stage_name,
            "policy": stage["policy"],
            "after": stage["after"],
            "required_oracle_tier": required_tier,
            "oracle_engine": stage["oracle_engine"],
        }
        path.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")

    preflight = engine_preflight(te, stage_name)
    record = {
        "schema": "descriptor_evaluation_cohort_v1",
        "target": te.target,
        "stage": stage_name,
        "policy": stage["policy"],
        "after": stage["after"],
        "required_oracle_tier": required_tier,
        "oracle_engine": stage["oracle_engine"],
        "descriptor": str(te.path.relative_to(repo_root())),
        "descriptor_sha256": te.descriptor_sha256,
        "candidate": str(candidate.resolve()),
        "candidate_tree_sha256": candidate_digest,
        "capsules": sorted(source_records, key=lambda row: row["name"]),
        "n_capsules": len(written),
        "engine_preflight": preflight,
    }
    if predecessor is not None:
        record["predecessor_pass_evidence"] = predecessor
    record["materialized_tree_sha256"] = _tree_sha256(dest)
    (dest / ".evaluation_cohort.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return record


def validate_evaluation_cohort(
    root: str | Path, te: TargetExperiment, candidate: str | Path,
) -> dict[str, Any]:
    root = Path(root)
    record_path = root / ".evaluation_cohort.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    if record.get("schema") != "descriptor_evaluation_cohort_v1":
        raise ValueError(f"unsupported evaluation cohort record: {record_path}")
    if record.get("target") != te.target or record.get("descriptor_sha256") != te.descriptor_sha256:
        raise ValueError("evaluation cohort does not belong to the loaded target descriptor")
    candidate = Path(candidate)
    if candidate.is_symlink() or not candidate.is_dir():
        raise ValueError(f"frozen evaluation candidate is unavailable: {candidate}")
    if _tree_sha256(candidate) != record.get("candidate_tree_sha256"):
        raise ValueError("frozen evaluation candidate content digest mismatch")
    stage = te.evaluation_cohort(str(record.get("stage")))
    names = sorted(path.name for path in root.iterdir() if path.is_dir())
    if names != sorted(stage["include_capsules"]):
        raise ValueError("evaluation cohort contents differ from its descriptor stage")
    for name in names:
        doc = yaml.safe_load((root / name / "capsule.yaml").read_text(encoding="utf-8")) or {}
        if stage["oracle_tier"] not in (doc.get("required_oracle_tiers") or []):
            raise ValueError(f"{name} does not require evaluation tier {stage['oracle_tier']}")
        if doc.get("max_oracle_tier") is not None:
            raise ValueError(f"{name} still carries a search-time maximum oracle tier")
        marker = doc.get("evaluation_stage") or {}
        if marker.get("name") != record["stage"] or marker.get("oracle_engine") != stage["oracle_engine"]:
            raise ValueError(f"{name} carries the wrong evaluation-stage marker")
    expected_digest = record.get("materialized_tree_sha256")
    # The manifest itself was deliberately absent when its digest was calculated.
    actual_digest = _tree_sha256(root, exclude=frozenset({record_path.name}))
    if actual_digest != expected_digest:
        raise ValueError("evaluation cohort content digest mismatch")
    predecessor = record.get("predecessor_pass_evidence")
    if predecessor is not None:
        manifest_path = Path(str(predecessor.get("cohort", ""))) / ".evaluation_cohort.json"
        score_path = Path(str(predecessor.get("score", "")))
        if (
            not manifest_path.is_file()
            or hashlib.sha256(manifest_path.read_bytes()).hexdigest()
            != predecessor.get("cohort_manifest_sha256")
        ):
            raise ValueError("predecessor cohort evidence digest mismatch")
        if (
            not score_path.is_file()
            or hashlib.sha256(score_path.read_bytes()).hexdigest()
            != predecessor.get("score_sha256")
        ):
            raise ValueError("predecessor score evidence digest mismatch")
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Materialize a frozen post-search evaluation cohort")
    parser.add_argument("--target", required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--dest", required=True)
    parser.add_argument("--candidate", required=True,
                        help="frozen compiler package evaluated by this stage")
    parser.add_argument("--predecessor-cohort",
                        help="materialized predecessor cohort required by an after: *_pass stage")
    parser.add_argument("--predecessor-score",
                        help="grader score proving the predecessor cohort passed its physical tier")
    parser.add_argument("--replace-empty", action="store_true",
                        help="remove an existing empty destination before materializing")
    args = parser.parse_args(argv)
    descriptor = descriptor_for(args.target)
    if descriptor is None:
        raise SystemExit(f"no target experiment descriptor for {args.target!r}")
    te = load_target_experiment(descriptor)
    dest = Path(args.dest)
    if args.replace_empty and dest.is_dir() and not any(dest.iterdir()):
        dest.rmdir()
    record = materialize_evaluation_cohort(
        dest, te, args.stage, args.candidate,
        predecessor_cohort=args.predecessor_cohort,
        predecessor_score=args.predecessor_score,
    )
    validate_evaluation_cohort(dest, te, args.candidate)
    print(json.dumps(record, indent=2, sort_keys=True))
    if not record["engine_preflight"]["ok"]:
        print("evaluation cohort materialized, but engine preflight failed; refusing to call it runnable")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
