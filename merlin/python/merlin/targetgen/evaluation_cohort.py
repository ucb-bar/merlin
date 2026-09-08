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


def materialize_evaluation_cohort(
    dest: str | Path, te: TargetExperiment, stage_name: str,
) -> dict[str, Any]:
    """Create one descriptor-declared frozen-candidate cohort at ``dest``.

    ``dest`` must be absent or empty.  Refusing an existing populated directory avoids combining two
    descriptor revisions or leaving stale capsules in the denominator.
    """
    stage = te.evaluation_cohort(stage_name)
    dest = Path(dest)
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
        "capsules": sorted(source_records, key=lambda row: row["name"]),
        "n_capsules": len(written),
        "engine_preflight": preflight,
    }
    record["materialized_tree_sha256"] = _tree_sha256(dest)
    (dest / ".evaluation_cohort.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return record


def validate_evaluation_cohort(root: str | Path, te: TargetExperiment) -> dict[str, Any]:
    root = Path(root)
    record_path = root / ".evaluation_cohort.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    if record.get("schema") != "descriptor_evaluation_cohort_v1":
        raise ValueError(f"unsupported evaluation cohort record: {record_path}")
    if record.get("target") != te.target or record.get("descriptor_sha256") != te.descriptor_sha256:
        raise ValueError("evaluation cohort does not belong to the loaded target descriptor")
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
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Materialize a frozen post-search evaluation cohort")
    parser.add_argument("--target", required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--dest", required=True)
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
    record = materialize_evaluation_cohort(dest, te, args.stage)
    validate_evaluation_cohort(dest, te)
    print(json.dumps(record, indent=2, sort_keys=True))
    if not record["engine_preflight"]["ok"]:
        print("evaluation cohort materialized, but engine preflight failed; refusing to call it runnable")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
