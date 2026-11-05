"""Pre-launch preparation of a seed compiler for a seeded open-compiler run.

    python -m merlin_experiments.phase1.seed_prep --seed <frozen compiler dir> \\
        --export <run-framework export> --out <dir> --workspace <seed run workspace> \\
        [--run-dir <seed run>] [--cases A,B,...] [--compare LABEL=TREE[,perf_history.jsonl] ...]

Default (launch path): read the seed run's records -- seconds, no simulation.  Baseline cycles
come from the workspace's qa/perf_history.jsonl and graded case rows (each row names the source
it was measured on), the calibrated fast estimate from recorded counts fills cases without a
complete timing, and variant decisions come from the run's A/B records, guarded by a compile-only
check that the change reaches its recorded cases and changes no unrecorded sampled case.
``--measure`` is the opt-in path that re-times on the Spike cycle model instead.

Writes into ``--out``: ``SEED_BASELINE.json`` / ``SEED_BASELINE.md`` (per-case map, ranked gaps
per family, promotion list), ``SEED_STRUCTURE.md`` (library routes per family against other
runs' compilers) and ``promoted-compiler/`` when a variant is promoted.  Performance only: no
functional suite and no physical simulation run here.
"""

from __future__ import annotations

import argparse
import filecmp
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import baseline as B
from . import records
from . import report as R
from . import structure as S
from . import variants as V

SCHEMA = "merlin.phase1.seed_prep.v1"


def _tree_sha(path: Path) -> str:
    from merlin.common.tree_hash import hash_tree

    return hash_tree(path)["sha256"]


def _discover_descriptor(export: Path) -> Path:
    hits = sorted((export / "out" / "artifacts").glob("*/source-release/payload/experiment/target_experiment.yaml"))
    if len(hits) != 1:
        raise SystemExit(f"--descriptor is required: found {len(hits)} candidate descriptors in the export")
    return hits[0]


def _discover_suite_root(descriptor: Path) -> Path:
    corpus = descriptor.parent.parent / "corpus"
    hits = [d for d in sorted(corpus.iterdir()) if (d / "selection.json").is_file() and (d / "cases").is_dir()]
    if len(hits) != 1:
        raise SystemExit(f"--suite-root is required: found {len(hits)} staged suites under {corpus}")
    return hits[0]


def _pin(path: Path) -> str:
    import hashlib

    return json.dumps({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})


def _run_dir_inputs(run_dir: Path | None) -> dict:
    """The seed run's own grading selections, when its directory is given."""
    found: dict = {}
    if run_dir is None:
        return found
    library = run_dir / "compiler_library"
    if (library / "contract.json").is_file():
        found["compiler_library"] = library
    campaign = run_dir / "joint_campaign.json"
    if campaign.is_file():
        selection = (json.loads(campaign.read_text(encoding="utf-8")) or {}).get("performance_selection")
        path = selection.get("path") if isinstance(selection, dict) else selection
        if path:
            found["performance_selection"] = Path(path)
    full = run_dir / "FULL_COMPILER_SOURCE_SELECTION.json"
    if full.is_file():
        found["full_compiler_selection"] = full
    host = run_dir / "public_owned_host_build_selection.json"
    if host.is_file():
        found["owned_host_selection"] = host
    return found


def _env_pairs(values: list[str], files: list[str]) -> dict[str, str]:
    env: dict[str, str] = {}
    for name in files:
        text = Path(name).read_text(encoding="utf-8")
        if text.lstrip().startswith("{"):
            env.update({str(k): str(v) for k, v in json.loads(text).items()})
            continue
        for line in text.splitlines():
            key, sep, value = line.strip().partition("=")
            if sep and key and not key.startswith("#"):
                env[key] = value
    for item in values:
        key, sep, value = item.partition("=")
        if not sep:
            raise SystemExit(f"--env expects KEY=VALUE, got {item!r}")
        env[key] = value
    return env


def _select_cases(args, suite_root: Path) -> tuple[list[str], str]:
    available = sorted(p.name for p in (suite_root / "cases").iterdir() if p.is_dir())
    passing = None
    if args.grade:
        grade = json.loads(Path(args.grade).read_text(encoding="utf-8"))
        passing = {row["capsule"] for row in grade.get("per_capsule") or [] if row.get("status") == "pass"}
    pool = [c for c in available if passing is None or c in passing]
    scope = f"{len(pool)} passing cases of the staged suite `{suite_root.name}`"
    if args.cases:
        wanted = [c for c in args.cases.split(",") if c]
        missing = [c for c in wanted if c not in available]
        if missing:
            raise SystemExit(f"unknown cases: {', '.join(missing)}")
        return wanted, f"{len(wanted)} named cases (bounded sample of {scope})"
    return pool, scope


def _history_cycles(path: Path, model_sha256: str | None) -> dict[str, int]:
    """Newest complete timing-model cycles per case from a run's perf history."""
    out: dict[str, tuple[float, int]] = {}
    if not path.is_file():
        return {}
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if row.get("kind") != "modeled" or not isinstance(row.get("cycles"), int):
            continue
        ident = (row.get("performance_identity") or {}).get("model_sha256")
        if model_sha256 and ident and ident != model_sha256:
            continue
        stamp = float(row.get("t") or 0)
        case = row.get("test_case")
        if case and (case not in out or stamp >= out[case][0]):
            out[case] = (stamp, row["cycles"])
    return {case: cycles for case, (_t, cycles) in out.items()}


def _family_effects(rows: list[dict], comparisons: dict[str, dict[str, int]]) -> dict[str, list[str]]:
    """Per family and comparator: how the seed's timed cases compare on the same cases and model."""
    effects: dict[str, list[str]] = {}
    seed = {r["case"]: r for r in rows if r.get("status") == "timed"}
    for label, cycles in comparisons.items():
        per_family: dict[str, list[tuple[float, str, int, int]]] = {}
        for case, row in seed.items():
            other = cycles.get(case)
            if other:
                per_family.setdefault(row["family"], []).append(
                    (row["model_cycles"] / other, case, row["model_cycles"], other))
        for family, items in per_family.items():
            items.sort(reverse=True)
            ahead = [i for i in items if i[0] > 1.0]
            if ahead:
                ratio, case, mine, theirs = ahead[0]
                line = (f"{label} is faster on {len(ahead)}/{len(items)} shared cases under the same model; "
                        f"largest gap {ratio:.2f}x on {case} ({mine:,} seed vs {theirs:,} model cycles): "
                        f"inspect its routes above for this family")
            else:
                ratio, case, mine, theirs = items[-1]
                line = (f"seed is at least as fast as {label} on all {len(items)} shared cases; "
                        f"its widest lead is {1 / ratio:.2f}x on {case} ({mine:,} vs {theirs:,})")
            effects.setdefault(f"family:{family}:{label}", []).append(line)
    return effects


def _grader(args, found: dict, export: Path, descriptor: Path, suite_root: Path) -> B.Grader:
    """The opt-in measuring grader: the export's own grading with the run's pinned selections."""
    facts = args.rtl_facts or (descriptor.parent / "rtl_facts" / "facts.json")
    env = _env_pairs(args.env, args.env_file)
    selection = args.performance_selection or found.get("performance_selection")
    if selection is None:
        raise SystemExit("--performance-selection (or --run-dir) is required with --measure: it pins the model")
    env["MERLIN_JOINT_PERFORMANCE_SELECTION"] = _pin(Path(selection))
    full = args.full_compiler_selection or found.get("full_compiler_selection")
    if full is not None:
        record = json.loads(Path(full).read_text(encoding="utf-8"))
        env["MERLIN_FULL_COMPILER_SELECTION"] = json.dumps(
            {k: record[k] for k in ("root", "manifest_path", "manifest_sha256")}, sort_keys=True)
    host = args.owned_host_selection or found.get("owned_host_selection")
    if host is not None:
        env["MERLIN_MODEL_OWNED_HOST_BUILD_SELECTION"] = _pin(Path(host))
    return B.Grader(export=export, descriptor=descriptor, suite_root=suite_root, python=args.python,
                    rtl_facts=facts if facts.is_file() else None,
                    compiler_library=args.compiler_library or found.get("compiler_library"),
                    env=env, cpus=args.cpus, nice=args.nice, budget_s=args.budget)


def _already_in_seed(change: V.VariantChange, merged: dict[str, str], seed: Path) -> bool:
    for rel in change.files:
        target = seed / rel
        if not target.is_file():
            return False
        if rel in merged:
            if target.read_text(encoding="utf-8", errors="replace") != merged[rel]:
                return False
        elif not filecmp.cmp(Path(change.root) / rel, target, shallow=False):
            return False
    return True


def _compile_grader(args, export: Path, descriptor: Path, suite_root: Path) -> B.Grader:
    """Just enough of the grading environment to run the compiler's own emit command."""
    return B.Grader(export=export, descriptor=descriptor, suite_root=suite_root, python=args.python,
                    env=_env_pairs(args.env, args.env_file), cpus=args.cpus, nice=args.nice)


def _promote_records(args, seed: Path, seed_sha: str, workspace: Path, rows: dict[str, dict], log,
                     export: Path, descriptor: Path, suite_root: Path) -> dict:
    """Variant decisions from the run's A/B records; nothing is simulated."""
    variants_root = Path(args.variants)
    extra = [p for p in (variants_root.parent / "checkpoints").glob("*") if p.is_dir()]
    found = V.discover(variants_root, seed, extra)
    wanted, excluded = set(args.variant or []), set(args.exclude_variant or [])
    by_name = {c.name: (c, m) for c, m in found}
    recorded = records.variant_records(workspace)
    compiler = _compile_grader(args, export, descriptor, suite_root) if args.check_emission else None
    seed_digest: dict[str, str | None] = {}

    def capsule(case: str) -> Path | None:
        # The staged suite first, then any public corpus directory of the same release.
        for path in (suite_root / "cases" / case, *sorted(suite_root.parent.glob(f"*/{case}"))):
            if (path / "capsule.interface.mlir").is_file():
                return path / "capsule.interface.mlir"
        return None

    def digest(tree: Path, case: str, tag: str) -> str | None:
        mlir = capsule(case)
        if mlir is None:
            return None
        return B.emission_digest(tree, mlir, Path(args.out).resolve() / "work" / "emit" / tag / case, compiler)

    # A variant is judged on every case it was compared on, not only the sampled ones.
    extra_cases = sorted({i["case"] for items in recorded.values() for i in items} - set(rows))
    seed_rows = dict(rows)
    seed_rows.update({r["case"]: r for r in records.baseline(seed_sha, extra_cases, workspace)})
    result: dict = {"mode": "records", "examined": 0, "promoted": [], "candidates": [], "not_applicable": [],
                    "discovered": {}, "not_examined": []}
    for change, _m in found:
        result["discovered"][change.status] = result["discovered"].get(change.status, 0) + 1
    for change, merged in found:
        if (wanted and change.name not in wanted) or change.name in excluded:
            continue
        result["examined"] += 1
        status = change.status
        if status == "applicable" and _already_in_seed(change, merged, seed):
            status = "already_in_seed"
        if status != "applicable":
            result["not_applicable"].append({"name": change.name, "status": status})
            continue
        decision = records.decide(status, recorded.get(change.name, []), seed_rows)
        entry = {"name": change.name, "files": change.files, "reason": decision["reason"],
                 "cases": decision["cases"], "gain_cycles": decision["gain_cycles"],
                 "affected_cases": [c["case"] for c in decision["cases"]]}
        if decision["promote"] and compiler is not None:
            # Compile only (no simulation): a promotion may not change a sampled case it has no record for.
            cand = Path(args.out).resolve() / "work" / "candidates" / change.name
            V.materialize(seed, cand, [(change, merged)])
            recorded_cases = [c["case"] for c in decision["cases"]]
            sample = [c for c in rows if c not in recorded_cases]
            every = recorded_cases + sample
            with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
                missing = [c for c in every if c not in seed_digest]
                for case, value in zip(missing, pool.map(lambda c: digest(seed, c, "_seed"), missing)):
                    seed_digest[case] = value
                mine = dict(zip(every, pool.map(lambda c, t=cand, n=change.name: digest(t, c, n), every)))
            unrecorded = [c for c in sample if mine[c] != seed_digest[c]]
            carried = [c for c in recorded_cases if mine[c] is not None and mine[c] != seed_digest[c]]
            entry["unrecorded_changed_cases"] = unrecorded
            entry["recorded_cases_changed_on_seed"] = carried
            if not carried:
                decision["promote"] = False
                entry["reason"] = (decision["reason"] + "; but applied to the seed it leaves the emitted code of "
                                   "its recorded cases unchanged, so the recorded gain does not carry over")
            elif unrecorded:
                decision["promote"] = False
                entry["reason"] = (decision["reason"] + f"; but it also changes the emitted code of "
                                   f"{len(unrecorded)} sampled case(s) with no A/B record "
                                   f"({', '.join(unrecorded[:4])}); time them with --measure")
        (result["promoted"] if decision["promote"] else result["candidates"]).append(entry)
        log(f"variant {change.name}: {entry['reason']}")
    if result["promoted"]:
        order = sorted(result["promoted"], key=lambda e: -e["gain_cycles"])
        tree = Path(args.out).resolve() / "promoted-compiler"
        result["promoted_tree_applied"] = V.materialize(seed, tree, [by_name[e["name"]] for e in order])
        result["promoted_tree"] = str(tree)
        result["promoted_tree_note"] = ("record-based: each promotion's numbers come from its own variant tree; "
                                        "re-time the promoted tree with --measure before relying on the sum")
    return result


def _promote(args, grader: B.Grader, seed: Path, cases: list[str], rows: dict[str, dict], work: Path,
             suite_root: Path, log) -> dict:
    variants_root = Path(args.variants)
    extra = [p for p in (variants_root.parent / "checkpoints").glob("*") if p.is_dir()]
    found = V.discover(variants_root, seed, extra)
    wanted, excluded = set(args.variant or []), set(args.exclude_variant or [])
    examined = [(c, m) for c, m in found if (not wanted or c.name in wanted) and c.name not in excluded]
    result: dict = {"examined": len(examined), "promoted": [], "candidates": [], "not_applicable": []}
    statuses: dict[str, int] = {}
    for change, _merged in found:
        statuses[change.status] = statuses.get(change.status, 0) + 1
    result["discovered"] = statuses
    result["not_examined"] = [c.name for c, _m in found if c.status == "applicable" and (
        (wanted and c.name not in wanted) or c.name in excluded)]
    default_digest: dict[str, str | None] = {}

    def digest(tree: Path, case: str, tag: str) -> str | None:
        mlir = suite_root / "cases" / case / "capsule.interface.mlir"
        return B.emission_digest(tree, mlir, work / "emit" / tag / case, grader)

    progress = work / "promotion_progress.json"
    done: set[str] = set()
    if args.resume and progress.is_file():
        # Keep decisions already made for this seed and sample; they are re-listed, not re-run.
        previous = json.loads(progress.read_text(encoding="utf-8"))
        for key in ("promoted", "candidates", "not_applicable"):
            for entry in previous.get(key) or []:
                if entry.get("name") in {c.name for c, _m in examined}:
                    result[key].append(entry)
                    done.add(entry["name"])
    for change, merged in examined:
        if change.name in done:
            continue
        if change.status != "applicable":
            result["not_applicable"].append({"name": change.name, "status": change.status})
            continue
        cand = work / "candidates" / change.name
        if not V.materialize(seed, cand, [(change, merged)]):
            result["not_applicable"].append({"name": change.name, "status": "not_applied"})
            continue
        if all((seed / rel).is_file() and filecmp.cmp(cand / rel, seed / rel, shallow=False)
               for rel in change.files):
            # The merged change equals the seed's own files: the seed already contains it.
            result["not_applicable"].append({"name": change.name, "status": "already_in_seed"})
            log(f"variant {change.name}: already in the seed")
            progress.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
            continue
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            missing = [c for c in cases if c not in default_digest]
            for case, value in zip(missing, pool.map(lambda c: digest(seed, c, "_seed"), missing)):
                default_digest[case] = value
            mine = list(pool.map(lambda c, name=change.name, tree=cand: digest(tree, c, name), cases))
        affected = [case for case, value in zip(cases, mine) if value != default_digest[case]]
        entry = {"name": change.name, "files": change.files, "affected_cases": affected, "cases": []}
        if not affected:
            entry["reason"] = "emitted bytes unchanged on every sampled case"
            result["candidates"].append(entry)
            log(f"variant {change.name}: no emitted change")
            progress.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
            continue
        timed = B.run_cases(grader, cand, affected[: args.max_affected], work / "variant-runs" / change.name,
                            args.workers)
        before_total = after_total = 0
        ok = True
        for case in affected[: args.max_affected]:
            new, old = timed[case], rows.get(case) or {}
            item = {"case": case, "before": old.get("model_cycles"), "after": new.get("model_cycles"),
                    "before_evidence": old.get("status"), "after_evidence": new.get("status"),
                    "numerics": new.get("numerics")}
            entry["cases"].append(item)
            if new.get("numerics") != "pass":
                ok = False
            if item["before"] and item["after"] and old.get("status") == new.get("status") == "timed":
                before_total += item["before"]
                after_total += item["after"]
                if item["after"] > item["before"]:
                    ok = False
            else:
                ok = False
        untimed = len(affected) - min(len(affected), args.max_affected)
        if ok and after_total < before_total and untimed == 0:
            entry["reason"] = f"faster {before_total:,}->{after_total:,} model cycles, numerics pass on all changed cases"
            entry["gain_cycles"] = before_total - after_total
            result["promoted"].append(entry)
        else:
            entry["reason"] = ("not faster on every changed case with complete timing and passing numerics"
                               + (f"; {untimed} changed cases untimed" if untimed else ""))
            result["candidates"].append(entry)
        log(f"variant {change.name}: {entry['reason']}")
        progress.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    if result["promoted"]:
        order = sorted(result["promoted"], key=lambda e: -e["gain_cycles"])
        by_name = {c.name: (c, m) for c, m in examined}
        tree = Path(args.out) / "promoted-compiler"
        applied = V.materialize(seed, tree, [by_name[e["name"]] for e in order])
        result["promoted_tree"] = str(tree)
        result["promoted_tree_applied"] = applied
        # A combined tree emits exactly what each single promotion emitted on its changed cases,
        # or the case is flagged for re-timing; numbers are never carried over silently.
        recheck = []
        for e in order:
            if e["name"] not in applied:
                continue
            for case in e["affected_cases"]:
                if digest(tree, case, "_promoted") != digest(work / "candidates" / e["name"], case, e["name"]):
                    recheck.append(case)
        result["promoted_tree_retime"] = sorted(set(recheck))
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="seed_prep", description=__doc__.split("\n\n")[0])
    ap.add_argument("--seed", required=True, type=Path, help="frozen seed compiler directory")
    ap.add_argument("--export", required=True, type=Path, help="run-framework export the runs grade with")
    ap.add_argument("--out", required=True, type=Path, help="directory that receives the seed products")
    ap.add_argument("--run-dir", type=Path, help="the seed run's directory (library, selections)")
    ap.add_argument("--workspace", type=Path, help="the seed run's workspace (qa/ records, variants/)")
    ap.add_argument("--no-emission-check", dest="check_emission", action="store_false",
                    help="skip the compile-only check that a record-based promotion changes no unrecorded case")
    ap.add_argument("--measure", action="store_true",
                    help="opt-in: re-time on the Spike cycle model instead of reading the run's records")
    ap.add_argument("--descriptor", type=Path)
    ap.add_argument("--suite-root", type=Path, help="staged performance suite (cases/<id>/)")
    ap.add_argument("--rtl-facts", type=Path)
    ap.add_argument("--compiler-library", type=Path)
    ap.add_argument("--performance-selection", type=Path, help="timing-model selection (pins the model)")
    ap.add_argument("--full-compiler-selection", type=Path)
    ap.add_argument("--owned-host-selection", type=Path, help="public owned-host build selection (host leaves)")
    ap.add_argument("--env", action="append", default=[], help="KEY=VALUE for grading children")
    ap.add_argument("--env-file", action="append", default=[], help="JSON object or KEY=VALUE lines")
    ap.add_argument("--python", default=B.default_python(), help="interpreter for grading children")
    ap.add_argument("--cpus", help="CPU list for children (taskset -c), optional; unpinned by default")
    ap.add_argument("--nice", type=int, help="optional niceness of the children (with ionice -c3)")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--budget", type=int, default=900, help="per-case timing-model budget, seconds")
    ap.add_argument("--grade", help="latest_completed_grade.json: restrict to the seed's passing cases")
    ap.add_argument("--cases", help="comma-separated case ids (a bounded sample)")
    ap.add_argument("--baseline-json", type=Path, help="reuse rows from an earlier SEED_BASELINE.json")
    ap.add_argument("--variants", help="the seed run's variants/ directory")
    ap.add_argument("--variant", action="append", help="only these variant names")
    ap.add_argument("--resume", action="store_true", help="keep variant decisions from an interrupted run")
    ap.add_argument("--exclude-variant", action="append", help="skip these variant names")
    ap.add_argument("--max-affected", type=int, default=6, help="changed cases timed per variant")
    ap.add_argument("--compare", action="append", default=[],
                    help="LABEL=TREE[,HISTORY] another run's compiler and optional perf_history.jsonl")
    ap.add_argument("--per-family", type=int, default=8)
    args = ap.parse_args(argv)

    seed, export, out = args.seed.resolve(), args.export.resolve(), args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    work = out / "work"
    work.mkdir(exist_ok=True)
    started = time.time()

    def log(message: str) -> None:
        line = f"{time.strftime('%H:%M:%S', time.gmtime())} {message}"
        print(line, flush=True)
        with (out / "seed_prep.log").open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")

    found = _run_dir_inputs(args.run_dir)
    descriptor = (args.descriptor or _discover_descriptor(export)).resolve()
    suite_root = (args.suite_root or _discover_suite_root(descriptor)).resolve()
    workspace = args.workspace.resolve() if args.workspace else None
    if workspace is not None:
        if args.grade is None and workspace.joinpath(*records.LATEST_GRADE).is_file():
            args.grade = str(workspace.joinpath(*records.LATEST_GRADE))
        if args.variants is None and (workspace / "variants").is_dir():
            args.variants = str(workspace / "variants")
    seed_sha = _tree_sha(seed)
    cases, scope = _select_cases(args, suite_root)
    rows: dict[str, dict] = {}
    grader = None
    if args.measure:
        grader = _grader(args, found, export, descriptor, suite_root)
        if args.baseline_json and args.baseline_json.is_file():
            for row in json.loads(args.baseline_json.read_text(encoding="utf-8"))["baseline"]["cases"]:
                if row.get("status") != "UNKNOWN":  # an UNKNOWN row is timed again
                    rows[row["case"]] = row
        todo = [c for c in cases if c not in rows]
        log(f"baseline: {len(todo)} cases to time ({len(rows)} reused), workers={args.workers}, cpus={args.cpus}")
        rows.update(B.run_cases(grader, seed, todo, work / "seed-runs", args.workers,
                                progress=lambda r: log(f"  {r['case']}: {r['status']} {r.get('model_cycles')}"
                                                       f" ({r.get('wall_s')} s)")))
        ordered = [rows[c] for c in cases if c in rows]
        review = next((r.get("model_review") for r in ordered if r.get("model_review")), "pinned model")
        method = (f"measured now: complete Spike timing model ({review}) within {args.budget} s per case, "
                  "else the calibrated fast census estimate (labelled)")
        selection_pin = json.loads(grader.env["MERLIN_JOINT_PERFORMANCE_SELECTION"])
    else:
        if workspace is None:
            raise SystemExit("--workspace (the seed run's workspace) is required unless --measure is given")
        measured: dict[str, dict] = {}
        if args.baseline_json and args.baseline_json.is_file():
            earlier = json.loads(args.baseline_json.read_text(encoding="utf-8"))
            if (earlier.get("seed") or {}).get("sha256") == seed_sha:
                measured = {r["case"]: r for r in earlier["baseline"]["cases"]}
        ordered = records.baseline(seed_sha, cases, workspace, measured=measured)
        rows = {r["case"]: r for r in ordered}
        log(f"baseline from records: {sum(r['status'] == 'timed' for r in ordered)} timed, "
            f"{sum(r['status'] == 'fast_estimate' for r in ordered)} fast estimates, "
            f"{sum(r['status'] == 'UNKNOWN' for r in ordered)} unknown, "
            f"{sum(r.get('source_match') == 'seed' for r in ordered)} on the seed's exact bytes")
        method = ("records of the seed run (qa/perf_history.jsonl and the last completed grade's case rows), "
                  "no simulation; the calibrated fast estimate from recorded counts where no complete timing "
                  "was recorded; each row names the source its record was measured on")
        selection_pin = None
    model_sha = next((r.get("model_sha256") for r in ordered if r.get("model_sha256")), None)
    payload = {
        "schema": SCHEMA,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed": {"path": str(seed), "sha256": seed_sha},
        "export": {"path": str(export), "descriptor": str(descriptor), "suite_root": str(suite_root)},
        "records": str(workspace) if workspace else None,
        "baseline": {
            "scope": scope,
            "method": method,
            "model_sha256": model_sha,
            "performance_selection": selection_pin,
            "cases": ordered,
            "ranked_gaps": {f: [r["case"] for r in items]
                            for f, items in R.ranked_gaps(ordered, args.per_family).items()},
        },
    }
    R.write_baseline(out, payload, args.per_family)

    if args.variants:
        if args.measure:
            log("promotion: comparing variants on the sampled cases (measured)")
            payload["promotion"] = _promote(args, grader, seed, cases, rows, work, suite_root, log)
        else:
            log("promotion: variant decisions from the run's A/B records")
            payload["promotion"] = _promote_records(args, seed, seed_sha, workspace, rows, log,
                                                    export, descriptor, suite_root)
        R.write_baseline(out, payload, args.per_family)

    others: dict[str, S.TreeRoutes] = {}
    histories: dict[str, dict[str, int]] = {}
    for item in args.compare:
        label, _, rest = item.partition("=")
        tree, _, history = rest.partition(",")
        others[label] = S.analyze_tree(Path(tree))
        if history:
            histories[label] = _history_cycles(Path(history), model_sha)
    seed_routes = S.analyze_tree(seed)
    comparison = S.compare_families(seed_routes, others)
    effects = _family_effects(ordered, histories)
    R.write_structure(out, str(seed), sorted(others), comparison, effects)
    payload["structure"] = {
        "seed": seed_routes.to_json(),
        "others": {k: v.to_json() for k, v in others.items()},
        "effects": effects,
    }
    payload["elapsed_s"] = round(time.time() - started, 1)
    R.write_baseline(out, payload, args.per_family)
    log(f"done in {payload['elapsed_s']} s: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
