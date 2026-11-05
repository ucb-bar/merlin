"""Advantage check for a seeded launch: does the seeded run really start with every advantage?

    seed_rehearsal.py --spec SEEDED_SPEC [--export-id ID] [--var ARTIFACTS=SCRATCH] --cases all|NAME,NAME|@FILE
                      [--edit SCRIPT] [--edit-cases NAMES] [--unseeded N] [--out DIR] [--skip-plan]
                      [--start-grade OUT]

No author and no model tokens. It writes the plan's artifacts directory (re-plans it; select a scratch
one with ``--var ARTIFACTS=...`` beside a prepared launch), the plan's certificate store and ``--out``:

1. **plan**: ``fresh_campaign.py ordering`` and ``plan`` of the export, exactly as a launch prepares
   (seed copy, evidence pin, verdict carry, guidance addendum);
2. **stage**: the runtime's own staging into a scratch workspace and run directory
   (``joint_seed.stage``, the full-compiler fill and the joint campaign state, as the session does),
   then an inventory of what the author will find, as a pass/fail table;
3. **zero-edit grade**: the seed compiler, unchanged, graded over ``--cases`` of the PLANNED run's
   sealed release (core and scalability, staged as the session stages them) by the self-check evaluator (``merlin_experiments.phase1.feedback.selfcheck``) in the
   campaign runtime with the plan's environment. Every case the seed proved should come back carried
   (``carried from <seed run>``) with model cycles attached and no Spike timing observation;
4. **unseeded comparison** (``--unseeded N``): the same evaluator on N of those cases without any seed
   state (empty certificate store, no seeded grade), and the per-case median scaled to the corpus;
5. **edited grade** (``--edit SCRIPT``): SCRIPT runs in a copy of the staged submission and should
   change what some cases emit; the copy is graded over ``--edit-cases``: changed cases re-execute,
   unchanged ones still carry;
6. **brief**: whether the performance brief of the first grade shows the seed's per-case cycles.

``--start-grade OUT`` also writes the zero-edit grade as ``SEED_START_GRADE.json``/``.md`` (status,
carried and cycles per test case) into OUT, for staging as seed evidence: the seeded run's minute-0
starting grade.

Writes ``report.json`` and ``report.md`` under ``--out`` and prints the table. Exit 0 when every
inventory row passes and every case the seed proved was carried by the zero-edit grade.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


# --------------------------------------------------------------------------- operator side


def _context(args):
    sys.path.insert(0, str(HERE))
    import fresh_campaign as F

    return F.Context(args.spec, export_id=args.export_id, overrides=args.var or ())


def _runtime_env(plan: dict, extra=None) -> dict:
    env = dict(os.environ, **plan["environment"])
    env.update(extra or {})
    return env


def _python(plan: dict) -> str:
    return plan["argv"][0]


def _call(command, *, env, cwd, log: Path) -> int:
    with open(log, "ab") as handle:
        handle.write(f"\n$ {' '.join(map(str, command))}\n".encode())
        handle.flush()
        return subprocess.run(list(map(str, command)), env=env, cwd=cwd, stdout=handle,
                              stderr=subprocess.STDOUT).returncode


def _in_runtime(plan: dict, out: Path, step: str, *rest, extra=None) -> dict:
    """Run ``seed_rehearsal.py <step>`` inside the campaign runtime (its interpreter, export, environment)."""
    # The export's own copy when it has one (the runtime under test), else this file.
    script = Path(plan["cwd"]) / "examples/gemmini/phase1/launch" / Path(__file__).name
    script = script if script.is_file() else Path(__file__).resolve()
    result = out / f"{step.strip('_')}-{rest[0] if step == '_grade' else 'workspace'}.json"
    if step == "_grade":  # each grade's operator work trees apart, so its raw results can be read back
        (out / f"tmp-{rest[0]}").mkdir(exist_ok=True)
        extra = dict(extra or {}, TMPDIR=str(out / f"tmp-{rest[0]}"))
    code = _call([_python(plan), "-B", script, step, str(out), *rest, "--result", result],
                 env=_runtime_env(plan, extra), cwd=plan["cwd"], log=out / "rehearsal.log")
    if code or not result.is_file():
        raise SystemExit(f"{step} failed (rc={code}); see {out / 'rehearsal.log'}")
    return json.loads(result.read_text())


def inventory(stage: dict, out: Path, plan: dict) -> list[dict]:
    ws, run = out / "workspace", out / "run"
    guidance = None
    for flag in ("--operator-errata",):
        if flag in plan["argv"]:
            guidance = Path(plan["argv"][plan["argv"].index(flag) + 1])
    addendum = Path((plan.get("seed") or {}).get("guidance_addendum", {}).get("path", ""))

    def nonempty(path: Path) -> bool:
        if path.is_dir():
            return any(p.is_file() and p.stat().st_size for p in path.rglob("*"))
        return path.is_file() and path.stat().st_size > 0

    def pairs(path: Path) -> int:
        try:
            return len(json.loads(path.read_text()).get("pairs") or ())
        except (OSError, ValueError):
            return 0

    def cycles(path: Path) -> int:
        try:
            return sum(1 for c in json.loads(path.read_text())["cases"].values() if c.get("cycles"))
        except (OSError, ValueError, KeyError):
            return 0

    rows = [
        ("SEED_INDEX.md", nonempty(ws / "SEED_INDEX.md"), "workspace root index of everything carried"),
        ("SEED_VARIANTS", nonempty(ws / "SEED_VARIANTS"),
         f"{sum(1 for p in (ws / 'SEED_VARIANTS').glob('*') if p.is_dir())} variant trees"),
        ("variant outcomes", nonempty(ws / "SEED_VARIANTS" / "OUTCOMES.json"), "SEED_VARIANTS/OUTCOMES.json"),
        ("perf history", nonempty(ws / "SEED_PERFORMANCE" / "perf_history.jsonl"), "SEED_PERFORMANCE/perf_history.jsonl"),
        ("baseline cycles", cycles(ws / "SEED_PERFORMANCE" / "baseline.json") > 0,
         f"{cycles(ws / 'SEED_PERFORMANCE' / 'baseline.json')} cases with seed cycles"),
        ("OPTIMIZATION_RULES", nonempty(ws / "OPTIMIZATION_RULES.json"), "OPTIMIZATION_RULES.json"),
        ("PREVIOUS_SESSION", nonempty(ws / "PREVIOUS_SESSION.md"), "PREVIOUS_SESSION.md"),
        ("A/B + evidence records", nonempty(ws / "SEED_EXPERIMENTS"), "SEED_EXPERIMENTS/"),
        ("seed tools and notes", nonempty(ws / "SEED_TOOLS") or nonempty(ws / "SEED_NOTES"), "SEED_TOOLS/, SEED_NOTES/"),
        ("calibration", pairs(run / "joint_fast_estimate_pairs.json") >= 12,
         f"{pairs(run / 'joint_fast_estimate_pairs.json')} fast-estimate pairs in the run directory"),
        ("carried timing", nonempty(run / "seed_timing"),
         f"{sum(1 for _ in (run / 'seed_timing').glob('*.json'))} executables with seed model cycles"),
        ("carried verdicts", bool(stage.get("certificate_carry_over")) and bool((plan.get("seed") or {}).get(
            "certificates", {}).get("carried")), f"{((plan.get('seed') or {}).get('certificates') or {}).get('carried')} "
         "verdicts re-keyed onto this judge"),
        ("guidance addendum", bool(guidance and addendum.is_file()
                                   and addendum.read_text().strip()[:200] in guidance.read_text()),
         "seed addendum inside the run's guidance"),
        ("seeded grade order", plan["environment"].get("MERLIN_CAPSULE_ORDER") in (None, "", "seeded")
         and bool(plan["environment"].get("MERLIN_SEEDED_GRADE")), "MERLIN_SEEDED_GRADE set, order seeded"),
    ]
    return [{"item": a, "pass": bool(b), "detail": c} for a, b, c in rows]


def _cases(spec: str, corpus: Path) -> dict:
    core = sorted(p.name for p in (corpus / "public").iterdir() if p.is_dir())
    scal = sorted(p.name for p in (corpus / "scalability" / "cases").iterdir() if p.is_dir()) if (
        corpus / "scalability" / "cases").is_dir() else []
    if spec == "all":
        wanted = set(core) | set(scal)
    elif spec.startswith("@"):
        wanted = {line.strip() for line in Path(spec[1:]).read_text().splitlines() if line.strip()}
    else:
        wanted = {name.strip() for name in spec.split(",") if name.strip()}
    return {"core": [c for c in core if c in wanted], "scalability": [c for c in scal if c in wanted]}


def _summary(grade: dict, proven: set) -> dict:
    rows = grade.get("cases") or {}
    carried = sorted(n for n, r in rows.items() if r.get("carried_from"))
    executed = sorted(n for n, r in rows.items() if r.get("executed_tiers"))
    missed = sorted(n for n in proven if n in rows and n not in carried)
    with_cycles = sorted(n for n in carried if rows[n].get("cycles"))
    timed = sorted(n for n, r in rows.items() if r.get("timing_observed"))
    seed_timed = sorted(n for n, r in rows.items() if r.get("timing_carried"))
    return {
        "cases": len(rows), "passed": sum(1 for r in rows.values() if r.get("status") == "pass"),
        "carried": len(carried), "executed": len(executed), "proven_not_carried": missed,
        "carried_with_model_cycles": len(with_cycles), "spike_timing_observations": len(timed),
        "timing_carried_from_seed": len(seed_timed), "wall_s": grade.get("wall_s"),
        "executed_cases": executed[:40],
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--spec", required=True)
    ap.add_argument("--export-id")
    ap.add_argument("--var", action="append", default=[], metavar="NAME=VALUE",
                    help="spec variable override, e.g. ARTIFACTS=<scratch> to keep a real launch plan untouched")
    ap.add_argument("--cases", default="all")
    ap.add_argument("--out")
    ap.add_argument("--edit", help="executable run inside a copy of the staged submission")
    ap.add_argument("--edit-cases", help="cases graded after --edit (default: --cases)")
    ap.add_argument("--unseeded", type=int, default=0, help="unseeded comparison sample size (0: skip)")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--skip-plan", action="store_true")
    ap.add_argument("--start-grade", metavar="OUT", type=Path,
                    help="write SEED_START_GRADE.json/.md (the zero-edit grade per test case) into OUT")
    args = ap.parse_args(argv)
    ctx = _context(args)
    artifacts = ctx.path("${ARTIFACTS}")
    out = Path(args.out) if args.out else artifacts / "rehearsal"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    framework = ctx.path("${FRAMEWORK}")
    python = ctx.x("${PYTHON}")
    report: dict = {"spec": str(Path(args.spec).resolve()), "export": str(framework), "timings_s": {}}
    started = time.monotonic()
    if not args.skip_plan:
        env = dict(os.environ, PYTHONPATH=":".join(str(framework / p) for p in (
            "src", "packages/merlin-experiments/src", "packages/merlin-timing/src")))
        launcher = framework / "examples/gemmini/phase1/launch/fresh_campaign.py"
        for step in ("ordering", "plan"):
            forwarded = [item for name in args.var for item in ("--var", name)]
            code = _call([python, "-B", launcher, step, "--spec", args.spec, "--export-id", ctx.lookup("EXPORT_ID"),
                          *forwarded],
                         env=env, cwd=framework, log=out / "rehearsal.log")
            if code:
                raise SystemExit(f"fresh_campaign.py {step} failed; see {out / 'rehearsal.log'}")
    report["timings_s"]["plan"] = round(time.monotonic() - started, 1)
    plan = json.loads((artifacts / "proposed-launch.json").read_text())
    if plan.get("seed") is None:
        raise SystemExit("the spec plans no seed")
    report["carry"] = {k: plan["seed"].get(k) for k in ("carry", "certificates", "evidence")}

    t = time.monotonic()
    stage = _in_runtime(plan, out, "_stage", str(artifacts / "proposed-launch.json"))
    report["timings_s"]["stage"] = round(time.monotonic() - t, 1)
    report["inventory"] = inventory(stage, out, plan)

    corpus = Path(stage["corpus"])
    cases = _cases(args.cases, corpus)
    extra = {"MERLIN_JOINT_CAMPAIGN_STATE": str(out / "run" / "joint_campaign.json")}
    extra.update(stage.get("environment") or {})
    proven = set(stage.get("proven") or ())
    grades = {}
    t = time.monotonic()
    grades["zero_edit"] = _in_runtime(plan, out, "_grade", "zero_edit", str(out / "workspace" / "submission"),
                                      json.dumps(cases), str(args.workers), extra=extra)
    report["timings_s"]["zero_edit_grade"] = round(time.monotonic() - t, 1)
    report["zero_edit"] = _summary(grades["zero_edit"], proven & set(cases["core"] + cases["scalability"]))
    if args.start_grade is not None:
        report["start_grade"] = start_grade(grades["zero_edit"], args.start_grade)

    if args.unseeded:
        # An even spread over each suite (every k-th case), not its alphabetical head.
        step = max(1, -(-sum(map(len, cases.values())) // args.unseeded))
        sample = {k: v[::step] for k, v in cases.items()}
        unseeded = dict(extra, MERLIN_TIER_CERT_CACHE=str(out / "unseeded-store"), MERLIN_SEEDED_GRADE="",
                        MERLIN_TIER_CERT_PORTABLE="", MERLIN_SEED_TIMING="0")
        grades["unseeded"] = _in_runtime(plan, out, "_grade", "unseeded", str(out / "workspace" / "submission"),
                                         json.dumps(sample), str(args.workers), extra=unseeded)
        per_case = [r["wall_s"] for r in grades["unseeded"]["cases"].values() if r.get("wall_s")]
        report["unseeded"] = dict(_summary(grades["unseeded"], set()), sample=sum(map(len, sample.values())),
                                  per_case_median_s=statistics.median(per_case) if per_case else None)
        sampled = max(1, report["unseeded"]["sample"])
        # Scaled by case count at the same worker count: an estimate, not a measurement.
        report["unseeded"]["estimated_all_cases_s"] = round(
            grades["unseeded"]["wall_s"] * sum(map(len, cases.values())) / sampled, 1)
    if args.edit:
        edited = out / "edited-submission"
        shutil.copytree(out / "workspace" / "submission", edited)
        code = subprocess.run([args.edit], cwd=edited).returncode
        if code:
            raise SystemExit(f"--edit failed with rc={code}")
        chosen = _cases(args.edit_cases or args.cases, corpus)
        t = time.monotonic()
        grades["edited"] = _in_runtime(plan, out, "_grade", "edited", str(edited), json.dumps(chosen),
                                       str(args.workers), extra=extra)
        report["timings_s"]["edited_grade"] = round(time.monotonic() - t, 1)
        report["edited"] = _summary(grades["edited"], proven & set(chosen["core"] + chosen["scalability"]))
    report["brief"] = grades["zero_edit"].get("brief")
    report["timings_s"]["total"] = round(time.monotonic() - started, 1)
    ok = all(row["pass"] for row in report["inventory"]) and not report["zero_edit"]["proven_not_carried"]
    report["advantage_check"] = "PASS" if ok else "FAIL"
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    text = render(report)
    (out / "report.md").write_text(text)
    print(text)
    return 0 if ok else 1


def render(report: dict) -> str:
    lines = [f"# Seed advantage check: {report['advantage_check']}", "", "| item | result | detail |", "|---|---|---|"]
    lines += [f"| {r['item']} | {'PASS' if r['pass'] else 'FAIL'} | {r['detail']} |" for r in report["inventory"]]
    for name in ("zero_edit", "unseeded", "edited"):
        block = report.get(name)
        if block:
            lines += ["", f"## {name}", "", "```", json.dumps(block, indent=1), "```"]
    if report.get("brief") is not None:
        lines += ["", "## performance brief (first grade)", "", "```", json.dumps(report["brief"], indent=1)[:4000], "```"]
    lines += ["", f"timings (s): {json.dumps(report['timings_s'])}"]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- runtime side


def _stage(out: Path, proposed: str, result: Path) -> None:
    """Stage the seed exactly as the session does, into ``out/workspace`` and ``out/run``."""
    import fresh_seed
    from merlin_experiments.phase1 import fresh_backend, joint_control, joint_seed
    from merlin_experiments.phase1 import full_compiler_workspace as FCW

    plan = json.loads(Path(proposed).read_text())
    options = fresh_seed.options_from_argv(plan["argv"])
    fresh_backend.validate(options)
    joint_control.validate(options)
    workspace, run_dir = out / "workspace", out / "run"
    (workspace / "submission").mkdir(parents=True)
    run_dir.mkdir()
    staged = joint_seed.stage(options, workspace, run_dir)
    selection = FCW.select(options.full_compiler_root, options.full_compiler_manifest,
                           options.full_compiler_manifest_sha256)
    FCW.stage(selection, workspace / "submission")
    joint_control.configure(options, run_dir)
    joint_control.start()  # the session starts the clock before authoring; grades read it
    seed_run = plan["seed"]["run"]
    corpus = planned_corpus(plan, run_dir, out / "corpus")
    planning_inventory(corpus, workspace, run_dir)
    from merlin.targetgen import seeded_grade

    environment = {FCW.ENV: json.dumps({"root": options.full_compiler_root,
                                        "manifest_path": options.full_compiler_manifest,
                                        "manifest_sha256": options.full_compiler_manifest_sha256})}
    owned = Path(seed_run["run_dir"]) / "public_owned_host_build_selection.json"
    if owned.is_file():
        import hashlib

        environment["MERLIN_MODEL_OWNED_HOST_BUILD_SELECTION"] = json.dumps(
            {"path": str(owned), "sha256": hashlib.sha256(owned.read_bytes()).hexdigest()})
    result.write_text(json.dumps({
        "certificate_carry_over": (staged.get("certificate_carry_over") or {}).get("enabled"),
        "seed_evidence": staged.get("seed_evidence"),
        "previous_session": bool(staged.get("previous_session")),
        "corpus": str(corpus),
        "proven": sorted(seeded_grade.proven()),
        "environment": environment,
    }, indent=2, default=str))


def planned_corpus(plan: dict, run_dir: Path, corpus: Path) -> Path:
    """The PLANNED run's sealed release, core and scalability, staged as the session stages it.

    The seed run's own corpus can differ from the planned one (fresh10 graded xs_capsules from a
    seed run on another release), and its frozen scalability snapshot then fails the planned
    descriptor's pin. The minute-0 starting grade must be the planned run's."""
    import yaml
    from merlin.targetgen.target_experiment import load_target_experiment
    from merlin_experiments.phase1 import corpus_inputs

    argv = plan["argv"]

    def value(flag):
        return argv[argv.index(flag) + 1]

    repo = Path(value("--repo")) if "--repo" in argv else Path(plan["cwd"])
    bundle = yaml.safe_load(Path(value("--bundle-manifest")).read_text())
    corpus_inputs.stage(run_dir, load_target_experiment(Path(value("--descriptor"))), bundle,
                        contract=repo / "merlin/contract")
    staged = run_dir / "private_corpus_input"
    # The session seals its copy read-only; this one is the rehearsal's to grade from and remove.
    for path in [staged, *staged.rglob("*")]:
        path.chmod(path.stat().st_mode | (0o700 if path.is_dir() else 0o600))
    shutil.move(str(staged), str(corpus))
    return corpus


def planning_inventory(corpus: Path, workspace: Path, run_dir: Path) -> None:
    """The planned corpus's public planning inventory (workload roles), as the session stages it."""
    from merlin.targetgen.capsule_common import discover_capsules
    from merlin_experiments.phase1 import capability_inventory, joint_control

    contract = corpus / "contract"
    native = discover_capsules(corpus / "public", labels={"public", "dev"}, contract=contract)
    scale = (discover_capsules(corpus / "scalability" / "cases", labels={"public", "dev"}, contract=contract)
             if (corpus / "scalability" / "cases").is_dir() else [])
    inventory = capability_inventory.build(native, scalability=scale, policy=joint_control.selected()["policy"])
    data = json.dumps(inventory, sort_keys=True, indent=2, allow_nan=False) + "\n"
    for root in (workspace, run_dir):
        (root / capability_inventory.FILENAME).write_text(data)


def start_grade(grade: dict, out: Path) -> dict:
    """``SEED_START_GRADE.json``/``.md``: the unchanged seed on the planned run's grader and corpus."""
    rows = {name: {key: row.get(key) for key in ("status", "carried_from", "cycles", "cycles_evidence", "suite")}
            for name, row in (grade.get("cases") or {}).items()}
    passed = sorted(name for name, row in rows.items() if row["status"] == "pass")
    carried = sorted(name for name in passed if rows[name]["carried_from"])
    document = {"schema": "merlin.seed_start_grade.v1",
                "what": "the seed compiler, unchanged, graded on this run's grader and corpus before launch",
                "cases": len(rows), "passed": len(passed), "carried": len(carried), "wall_s": grade.get("wall_s"),
                "per_case": rows}
    out.mkdir(parents=True, exist_ok=True)
    (out / "SEED_START_GRADE.json").write_text(json.dumps(document, indent=1, sort_keys=True) + "\n")
    lines = ["# Seed starting grade", "",
             f"The seed compiler, unchanged, graded on this run's grader before launch: **{len(passed)} of "
             f"{len(rows)} test cases pass**, {len(carried)} of them carried from the seed run (no re-execution).",
             "", "This is your starting point. After your edits, a test case that passed here and fails now is a "
             "regression your change introduced; the grade feedback names the error.", "",
             "| test case | suite | status | cycles |", "|---|---|---|---|"]
    lines += [f"| {name} | {row['suite'] or ''} | {row['status']}{' (carried)' if row['carried_from'] else ''} | "
              f"{row['cycles'] if row['cycles'] is not None else ''} |" for name, row in sorted(rows.items())]
    (out / "SEED_START_GRADE.md").write_text("\n".join(lines) + "\n")
    return {key: document[key] for key in ("cases", "passed", "carried", "wall_s")}


def _row(compact: dict, detail: dict, raw: dict) -> dict:
    """One case: its verdict, which tiers ran or carried, and where its cycles came from."""
    tiers = raw.get("tiers") or {}
    reuse = raw.get("tier_reuse") or {}
    performance = detail.get("performance") if isinstance(detail.get("performance"), dict) else {}
    carried_timing = isinstance(performance.get("carried_from"), dict)
    complete = performance.get("status") in ("PROVISIONAL", "QUALIFIED")
    fast = performance.get("fast_estimate") if isinstance(performance.get("fast_estimate"), dict) else {}
    return {
        "status": "pass" if compact.get("pass") is True or compact.get("status") == "pass" else (
            raw.get("status") or compact.get("status") or "unknown"),
        "carried_from": reuse.get("carried_from") or compact.get("carried_from"),
        # A whole-model case keeps no tier_reuse block: the certificate cache never covers it.
        "executed_tiers": [t for t in (reuse.get("executed") if reuse else tiers) or () if (tiers.get(t) or {}).get("status") not in (
            "skipped", "unavailable", None)],
        "carried_tiers": list(reuse.get("carried") or ()),
        "cycles": performance.get("estimated_whole_soc_cycles") if complete else fast.get("cycles"),
        "cycles_evidence": ("carried from seed" if carried_timing else "timing model" if complete
                            else "fast estimate" if fast.get("cycles") else None),
        "timing_carried": carried_timing,
        "timing_observed": complete and not carried_timing,
        "timing_reason": performance.get("reason"),
    }


def _raw_results(root: Path) -> dict:
    """``{case: capsule_result}`` from the evaluator's operator-side work trees under ``root``."""
    out = {}
    for path in sorted(Path(root).rglob("capsule_result.json"), key=lambda p: p.stat().st_mtime_ns):
        try:
            document = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if isinstance(document, dict) and isinstance(document.get("capsule"), str):
            out[document["capsule"]] = document
    return out


def _grade(out: Path, label: str, submission: str, cases_json: str, workers: str, result: Path) -> None:
    """Grade ``submission`` over the cases with the self-check evaluator; per-case carry and timing."""
    from merlin.targetgen.target_experiment import load_target_experiment
    from merlin_experiments.phase1.context import load_context
    from merlin_experiments.phase1.feedback.public_scalability import install
    from merlin_experiments.phase1.feedback.selfcheck import main as selfcheck

    descriptor = Path(os.environ["MERLIN_TARGET_EXPERIMENT"])
    context = load_context(descriptor, repo=Path(os.environ["MERLIN_REPO_ROOT"]))
    corpus = out / "corpus"
    cases = json.loads(cases_json)
    # Self-checks run in the workspace, as the author's do; each grade keeps its own copy of qa/.
    work = out / "workspace"
    os.chdir(work)
    rows, briefs, wall = {}, [], {}
    started = time.monotonic()
    for suite in ("core", "scalability"):
        if not cases.get(suite):
            continue
        output = out / f"{label}-{suite}.json"
        argv = ["--suite", suite, "--sim", "spike", "--submission", submission, "--capsules", ",".join(cases[suite]),
                "--workers", workers, "--timeout", "7200", "--out", str(output)]
        if suite == "core":
            argv += ["--capsules-root", str(corpus / "public"), "--contract", str(corpus / "contract")]
        else:
            install(load_target_experiment(descriptor), corpus / "scalability")
        t = time.monotonic()
        try:
            selfcheck(argv, context=context)
        except SystemExit:
            pass
        wall[suite] = time.monotonic() - t
        document = json.loads(output.read_text()) if output.is_file() else {}
        # Core work trees live under TMPDIR; the scalability suite keeps its own beside its output.
        raw = {**_raw_results(Path(os.environ["TMPDIR"])), **_raw_results(output.with_suffix(f".{suite}"))}
        # Core reports compact rows with a detail file; scalability reports full rows under "cases".
        for compact in document.get("per_capsule") or document.get("cases") or ():
            name = compact.get("capsule")
            if not isinstance(name, str):
                continue
            detail_path = (compact.get("detail") or {}).get("path")
            detail = json.loads((work / detail_path).read_text()) if detail_path and (work / detail_path).is_file() else compact
            rows[name] = dict(_row(compact, detail, raw.get(name) or {}), suite=suite,
                              wall_s=round(wall[suite] / max(1, len(cases[suite])), 2))
        if isinstance(document.get("performance_brief"), dict):
            briefs.append(document["performance_brief"])
    if (work / "qa").is_dir():
        shutil.copytree(work / "qa", out / f"qa-{label}", dirs_exist_ok=True)
    brief = None
    for document in briefs:
        entries = [*(document.get("largest_headroom") or []), *(document.get("diagnostic_controls") or [])]
        brief = {
            "entries": len(entries),
            "with_seed_cycles": sum(1 for e in entries if e.get("seed_cycles")),
            "with_vs_seed": sum(1 for e in entries if e.get("vs_seed") is not None),
            "sample": [{k: e.get(k) for k in ("test_case", "cycles", "evidence", "seed_cycles", "vs_seed")}
                       for e in entries[:6]],
            "vs_seed": document.get("vs_seed"),
        }
        break
    result.write_text(json.dumps({"label": label, "wall_s": round(time.monotonic() - started, 1), "cases": rows,
                                  "order": list(rows), "brief": brief}, indent=2, default=str))


def _runtime_main(argv) -> int:
    step, out, *rest = argv
    index = rest.index("--result")
    result = Path(rest[index + 1])
    rest = rest[:index] + rest[index + 2:]
    sys.path.insert(0, str(HERE))
    if step == "_stage":
        _stage(Path(out), rest[0], result)
    elif step == "_grade":
        _grade(Path(out), *rest, result=result)
    else:
        raise SystemExit(f"unknown step {step}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1].startswith("_"):
        raise SystemExit(_runtime_main(sys.argv[1:]))
    raise SystemExit(main())
