"""Seeded fresh campaigns: the ``seed:`` section of a fresh campaign spec.

A from-scratch spec has no ``seed:`` section and nothing here runs for it. With one, ``plan``
first renders the ordinary fresh launch (every fresh invariant is checked on that unseeded argv),
then turns it into a reviewed source-only joint seed (``merlin_experiments.phase1.joint_seed``):

* the seed is a directory holding the compiler source (for example a finished run's
  ``joint_frozen_submission`` or ``fcand_*`` snapshot), pinned by its tree digest, plus optional
  evidence: a top-level ``SEED_EVIDENCE/`` directory (moved out of the compiler copy) and any
  ``evidence:`` paths the spec lists;
* the copy drops ``strip`` members, and moves notes that describe the old harness (``notes``
  patterns, default ``REPORT.md`` and ``docs/*iteration*``) to ``SEED_NOTES/`` outside the
  compiler, unless the compiler's own sources name the file;
* ``carry`` (default: everything, when the seed names its ``run``) carries the seed run's
  evidence: ``certificates`` and ``results`` re-key its tier verdicts for byte-identical
  executables onto this export's judge (``merlin_experiments.phase1.seed_certificates``, with
  ``MERLIN_TIER_CERT_PORTABLE=1``) when the judge decides verdicts the same way in both exports
  (identical bytes, comment/docstring-only code changes, or changes listed in ``judge_review``
  with their exact old and new digests); otherwise those tiers re-run and the plan's
  ``seed-certificates.json`` lists each changed judge file for review; ``rules`` stages its ``OPTIMIZATION_RULES.json``, A/B
  records and variant outcomes; ``handoff`` stages ``PREVIOUS_SESSION.md``. All of it lands at
  the workspace root, outside ``submission/``;
* the guidance gains the seed addendum (``guidance``, default ``seed-guidance.md`` here);
* gSIM is optional: continuous gSIM promotion is turned off for the seeded policy;
* with verdicts carried, ``MERLIN_SEEDED_GRADE`` names the seed run: grades carry the screen tier for
  byte-identical executables too, grade changed cases first, then timing work by largest gap, then
  carried cases, and label carried passes ``carried from <run>`` (``merlin.targetgen.seeded_grade``).

The runtime's preflight-only mode refuses a seed, so ``preflight`` checks the seed itself (bytes,
review, option validation and a scratch placement exactly as the session stages it) and preflights
the unseeded twin the plan records. ``launch`` re-checks the seed before starting anything.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
GUIDANCE = HERE / "seed-guidance.md"
CARRY = ("certificates", "results", "rules", "handoff")
NOTES = ("REPORT.md", "docs/*iteration*")
EVIDENCE_DIR, NOTES_DIR, EXPERIMENTS_DIR = "SEED_EVIDENCE", "SEED_NOTES", "SEED_EXPERIMENTS"
#: Compiler sources whose text may name a note the compiler reads.
READER_SUFFIXES = (".py", ".yaml", ".yml", ".json", ".toml", ".cfg", ".txt")
#: A/B record files summarized from the seed run (bounded copies), by name.
EXPERIMENT_FILES = ("README.md", "report.json", "summary.json", "comparison.json", "comparison.txt",
                    "comparison_results.json")
MAX_RECORD_BYTES = 64 << 10
LEGACY_CARRY = "inherit_certificates_for_identical_executables"


class SeedError(ValueError):
    pass


def _sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# --------------------------------------------------------------------------- settings


def settings(ctx) -> dict:
    """The expanded ``seed`` section with its defaults; carry defaults on when the seed names its run."""
    seed = ctx.x_tree(dict(ctx.spec["seed"]))
    for key in ("from", "source", "review_id"):
        if not seed.get(key):
            raise SeedError(f"seed requires {key!r}")
    if not isinstance(seed["from"], dict) or set(seed["from"]) != {"path", "sha256"}:
        raise SeedError("seed.from requires one path/sha256 pin")
    run = seed.get("run")
    if run is not None and (not isinstance(run, dict) or not {"run_id", "run_dir", "workspace"} <= set(run)):
        raise SeedError("seed.run requires run_id, run_dir and workspace (and framework to carry verdicts)")
    carry = seed.get("carry")
    carry = dict.fromkeys(CARRY, run is not None) if carry is None else dict(carry)
    if set(carry) - set(CARRY) or any(type(value) is not bool for value in carry.values()):
        raise SeedError(f"seed.carry takes booleans for {list(CARRY)}")
    carry = {key: carry.get(key, run is not None) for key in CARRY}
    # Earlier specs spelled the joint seed's own keys; an explicit value there still decides.
    if seed.get(LEGACY_CARRY) is not None:
        carry["certificates"] = carry["results"] = bool(seed[LEGACY_CARRY])
    if "previous_session" in seed:
        carry["handoff"] = seed["previous_session"] is not None
    if any(carry.values()) and run is None:
        raise SeedError("carrying seed evidence requires seed.run")
    if (carry["certificates"] or carry["results"]) and not (run or {}).get("framework"):
        raise SeedError("carrying verdicts requires seed.run.framework (the seed run's export)")
    seed["carry"] = carry
    seed["strip"] = list(seed.get("strip") or ())
    seed["notes"] = list(seed.get("notes", NOTES) or ())
    seed["guidance"] = seed.get("guidance") or str(GUIDANCE)
    return seed


# --------------------------------------------------------------------------- seed bytes


def verify_source(path: Path, digest: str) -> str:
    """The pinned seed digest names this tree (``hash_tree`` or the graded source hash)."""
    from merlin.common.tree_hash import hash_tree
    from merlin_experiments.phase1 import run_inputs

    if not path.is_dir():
        raise SeedError(f"seed source is unavailable: {path}")
    if hash_tree(path)["sha256"] == digest:
        return "hash_tree"
    if run_inputs.graded_source_hash(path) == digest:
        return "graded_source_hash"
    raise SeedError(f"seed source digest differs from its pin: {path}")


def copy_source(source: Path, dest: Path, strip) -> None:
    if dest.exists():
        shutil.rmtree(dest)
    top = set(strip) | {EVIDENCE_DIR}
    shutil.copytree(
        source,
        dest,
        ignore=lambda d, names: {n for n in names if Path(d) == source and n in top}
        | {n for n in names if n in {"build", "__pycache__", ".git"}},
    )
    # A frozen seed is often read-only; the copy is ours to rearrange (notes and evidence move out).
    dest.chmod(dest.stat().st_mode | 0o200)
    for path in dest.rglob("*"):
        if path.is_symlink():
            raise SeedError(f"seed contains a link: {path}")
        path.chmod(path.stat().st_mode | 0o200)


def strip_stale(dest: Path, notes: Path, patterns) -> dict:
    """Move old-harness notes out of the compiler copy, keeping any file its sources name."""
    files = [p for p in sorted(dest.rglob("*")) if p.is_file()]
    stale = [
        p
        for p in files
        if p.suffix != ".py" and any(fnmatch.fnmatch(p.relative_to(dest).as_posix(), pat) for pat in patterns)
    ]
    readers = [p for p in files if p.suffix in READER_SUFFIXES and p not in stale]
    text = "\n".join(p.read_text(errors="replace") for p in readers)
    moved, kept = [], []
    for path in stale:
        relative = path.relative_to(dest).as_posix()
        if path.name in text:
            kept.append(relative)
            continue
        target = notes / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(path), str(target))
        moved.append(relative)
    if "REPORT.md" in moved:
        # The run finalizes the report it finds in submission/; start it fresh.
        (dest / "REPORT.md").write_text(
            "# Report\n\nThis compiler started from a seed. The seed's own report is kept outside the compiler "
            f"in `{NOTES_DIR}/REPORT.md`.\n"
        )
    return {"moved": moved, "kept_read_by_compiler": kept}


# --------------------------------------------------------------------------- evidence


def _bounded_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    data = source.read_bytes()
    if len(data) > MAX_RECORD_BYTES:
        data = data[:MAX_RECORD_BYTES] + b"\n[truncated by the seeded launch]\n"
    target.write_bytes(data)


def _tree_digests(root: Path) -> dict:
    return {
        p.relative_to(root).as_posix(): _sha256(p)
        for p in root.rglob("*")
        if p.is_file() and not p.is_symlink() and "__pycache__" not in p.parts
    }


def experiments(workspace: Path, seed_source: Path, out: Path) -> dict:
    """The seed run's A/B record summaries and its variants' differences from the seed compiler."""
    index = {"ab": {}, "variants": {}}
    for directory in sorted(p for p in (workspace / "ab").glob("*") if p.is_dir()):
        names = sorted(p.name for p in directory.iterdir() if p.is_file())
        kept = [name for name in names if name in EXPERIMENT_FILES]
        for name in kept:
            _bounded_copy(directory / name, out / "ab" / directory.name / name)
        index["ab"][directory.name] = {"records": kept, "files": len(names)}
    baseline = _tree_digests(seed_source) if seed_source.is_dir() else {}
    for directory in sorted(p for p in (workspace / "variants").glob("*") if p.is_dir()):
        digests = _tree_digests(directory)
        changed = sorted(k for k, v in digests.items() if k.endswith(".py") and baseline.get(k) != v)
        index["variants"][directory.name] = {"changed_python_files": changed[:40], "changed": len(changed)}
    out.mkdir(parents=True, exist_ok=True)
    (out / "index.json").write_text(json.dumps(index, indent=2) + "\n")
    return {key: len(value) for key, value in index.items()}


def build_evidence(seed: dict, source: Path, copy: Path, out: Path) -> dict:
    """Prepare the evidence directory staged at the workspace root (see joint_seed.stage_evidence)."""
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    summary = {"notes": strip_stale(copy, out / NOTES_DIR, seed["notes"])}
    staged = []
    if (source / EVIDENCE_DIR).is_dir():
        shutil.copytree(source / EVIDENCE_DIR, out / EVIDENCE_DIR, symlinks=False)
        staged.append(EVIDENCE_DIR)
    for item in seed.get("evidence") or ():
        path = Path(item["path"] if isinstance(item, dict) else item)
        if isinstance(item, dict) and item.get("sha256") and _sha256(path) != item["sha256"]:
            raise SeedError(f"seed evidence differs from its pin: {path}")
        target = out / EVIDENCE_DIR / path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        (shutil.copytree if path.is_dir() else shutil.copyfile)(path, target)
        staged.append(f"{EVIDENCE_DIR}/{path.name}")
    summary["evidence"] = staged
    run = seed.get("run")
    if seed["carry"]["rules"]:
        workspace = Path(run["workspace"])
        rules = workspace / "OPTIMIZATION_RULES.json"
        if rules.is_file():
            shutil.copyfile(rules, out / "OPTIMIZATION_RULES.json")
        summary["rules"] = rules.is_file()
        summary["experiments"] = experiments(workspace, source, out / EXPERIMENTS_DIR)
    if not any(out.iterdir()):
        return summary
    from merlin.common.tree_hash import hash_tree

    summary["pin"] = {"path": str(out), "sha256": hash_tree(out)["sha256"]}
    return summary


def carry_certificates(ctx, seed: dict, environment: dict, report: Path) -> dict:
    """Re-key the seed run's verdicts in the export's own runtime (its judge, its store)."""
    run, carry = seed["run"], seed["carry"]
    target = seed.get("target") or ctx.x(ctx.section("ordering")["target"])
    command = ["taskset", "-c", ctx.preparation_cpus, ctx.x("${PYTHON}"), "-B", "-m",
               "merlin_experiments.phase1.seed_certificates", "--seed-framework", run["framework"],
               "--seed-run-id", run["run_id"], "--target", target, "--report", str(report)]
    if run.get("certificate_store"):
        command += ["--store", run["certificate_store"]]
    if seed.get("judge_review"):
        # Operator-reviewed judge changes: {file: {old_sha256, new_sha256, reason}}, exact bytes only.
        reviewed = report.with_name("seed-judge-review.json")
        reviewed.write_text(json.dumps(seed["judge_review"], indent=2) + "\n")
        command += ["--reviewed", str(reviewed)]
    command += [flag for key, flag in (("certificates", "--no-certificates"), ("results", "--no-results"))
                if not carry[key]]
    done = subprocess.run(command, cwd=ctx.path("${FRAMEWORK}"), env=dict(os.environ, **environment),
                          capture_output=True, text=True)
    if done.returncode:
        raise SeedError(f"carrying the seed run's verdicts failed: {done.stderr.strip()[-600:]}")
    return json.loads(report.read_text())


# --------------------------------------------------------------------------- plan


def _replace_value(argv, flag, value):
    argv = list(argv)
    argv[argv.index(flag) + 1] = value
    return argv


def seeded_argv(argv, source):
    i = argv.index("--fresh-backend")
    return [*argv[:i], "--seed-submission", str(source), *argv[i + 1 :]]


def seeded_checks(plan: dict) -> dict:
    plan = dict(plan)
    plan["required_flags"] = [f for f in plan.get("required_flags") or () if f != "--fresh-backend"]
    plan["required_flags"].append("--seed-submission")
    plan["forbidden_flags"] = [f for f in plan.get("forbidden_flags") or () if f != "--seed-submission"]
    plan["forbidden_flags"].append("--fresh-backend")
    plan["forbidden_fragments"] = [f for f in plan.get("forbidden_fragments") or () if f != "seed-compiler"]
    return plan


def apply(ctx, record: dict, *, validate_policy, check_campaign, reserve, carry_verdicts=None) -> dict:
    """Turn the rendered unseeded plan into the seeded launch; records the unseeded twin for preflight."""
    seed, out, plan = settings(ctx), ctx.path("${ARTIFACTS}"), ctx.section("plan")
    carry = seed["carry"]
    source, dest = Path(seed["from"]["path"]), Path(seed["source"])
    basis = verify_source(source, seed["from"]["sha256"])

    # The unseeded twin preflights the environment; it keeps an unseeded policy of its own.
    twin_policy = out / "policy-unseeded.json"
    shutil.copyfile(out / "policy.json", twin_policy)
    twin_argv = _replace_value(record["argv"], "--joint-campaign", str(twin_policy))
    twin = {"argv": twin_argv, "reserved_argv": reserve(twin_argv, record["environment"]),
            "policy": {"path": str(twin_policy), "sha256": _sha256(twin_policy)}}

    # Seed bytes: the pinned tree without stripped members, old-harness notes moved out.
    copy_source(source, dest, seed["strip"])
    evidence = build_evidence(seed, source, dest, out / "seed-evidence")
    from merlin_experiments.phase1 import run_inputs

    graded = run_inputs.graded_source_hash(dest)

    environment = dict(record["environment"])
    if carry["certificates"] or carry["results"]:
        environment["MERLIN_TIER_CERT_PORTABLE"] = "1"
        # Seeded grades carry the screen for byte-identical executables, grade what changed first and
        # label carried passes (merlin.targetgen.seeded_grade). Never set for a from-scratch plan.
        environment["MERLIN_SEEDED_GRADE"] = seed["run"]["run_id"]
    carried = None
    if carry["certificates"] or carry["results"]:
        carried = (carry_verdicts or carry_certificates)(ctx, seed, environment, out / "seed-certificates.json")

    # Guidance: the selected guidance plus the seed addendum, only for this seeded launch.
    guidance, addendum = out / "author-guidance.md", Path(seed["guidance"])
    if ctx.path(plan["guidance"]).resolve() == guidance.resolve():
        raise SeedError("a seeded plan writes its own author-guidance.md: select the base guidance elsewhere")
    guidance.write_text(ctx.path(plan["guidance"]).read_text().rstrip("\n") + "\n\n" + addendum.read_text())

    origin = {"schema": "merlin.joint_reviewed_seed.v1", "review_id": seed["review_id"], "source": str(dest),
              "sha256": graded, "inherit_clock": False, "inherit_results": False, "variants": []}
    if carry["certificates"] or carry["results"]:
        origin["inherit_certificates_for_identical_executables"] = True
    if carry["handoff"]:
        origin["previous_session"] = {"run_dir": seed["run"]["run_dir"], "workspace": seed["run"]["workspace"]}
    if evidence.get("pin"):
        origin["seed_evidence"] = evidence["pin"]
    review = dict(
        origin,
        schema="operator.joint_seed_review.v1",
        approved=True,
        derived_from={"path": str(source), "sha256": seed["from"]["sha256"], "digest": basis,
                      "stripped": sorted(seed["strip"]), "run": (seed.get("run") or {}).get("run_id")},
        scope=seed.get("scope") or "Seeded launch: the seed compiler as starting source; no inherited clock or "
        "credit. Verdicts carry only for byte-identical executables under a byte-identical judge.",
    )
    review_path = out / "seed-review.json"
    review_path.write_text(json.dumps(review, indent=2) + "\n")
    origin["review"] = {"path": str(review_path), "sha256": _sha256(review_path)}

    policy = json.loads((out / "policy.json").read_text())
    policy["compiler_origin"] = origin
    correctness = policy.get("gsim_correctness")
    if isinstance(correctness, dict) and correctness.get("continuous_promotion") is True:
        correctness["continuous_promotion"] = False  # gSIM is optional in a seeded run
    (out / "policy.json").write_text(json.dumps(policy, indent=2) + "\n")
    validate_policy(policy)

    argv = seeded_argv(record["argv"], dest)
    check_campaign(ctx, argv, seeded_checks(plan))
    record.update(argv=argv, reserved_argv=reserve(argv, environment), environment=environment, origin=origin)
    record["seed"] = {
        "from": seed["from"],
        "from_digest": basis,
        "stripped": sorted(seed["strip"]),
        "source": str(dest),
        "graded_source_sha256": graded,
        "files": sum(1 for p in dest.rglob("*") if p.is_file()),
        "carry": carry,
        "run": seed.get("run"),
        "evidence": evidence,
        "certificates": None if carried is None else {k: carried.get(k) for k in (
            "decision", "carried", "already_portable", "judge_unreproducible", "rerun", "by_tier", "judge")},
        "guidance_addendum": {"path": str(addendum), "sha256": _sha256(addendum)},
        "gsim": "optional",
        "preflight_twin": twin,
    }
    return record


# --------------------------------------------------------------------------- preflight / launch checks


VERIFY_SCRIPT = HERE / "fresh_seed.py"


def options_from_argv(argv):
    """The runtime's options for the exact planned argv (the arguments the module parses)."""
    from merlin_experiments.phase1.options import RunOptions, build_parser
    from merlin_experiments.phase2.compiler_library_state import add_selection_arguments

    parser = build_parser()
    add_selection_arguments(parser)
    for flag in ("--descriptor", "--repo", "--bundle-manifest", "--oracle-timing", "--public-root"):
        parser.add_argument(flag)
    parser.add_argument("--language", default="")
    parser.add_argument("--treatment", default="baseline")
    parser.add_argument("--level")
    values = vars(parser.parse_args(argv[argv.index("merlin_experiments.phase1") + 1 :]))
    for key in ("descriptor", "repo", "bundle_manifest", "oracle_timing", "public_root", "language", "treatment",
                "level", "compiler_library_root", "compiler_library_contract", "compiler_library_contract_sha256"):
        values.pop(key, None)
    return RunOptions(**values)


def verify_in_runtime(record: dict) -> dict:
    """Inside the campaign runtime: the runtime accepts the seed and stages it as the session does."""
    from merlin_experiments.phase1 import fresh_backend, joint_control, joint_seed

    options = options_from_argv(record["argv"])
    fresh_backend.validate(options)
    joint_control.validate(options)
    origin = joint_seed.selected(options)
    if origin != record["origin"]:
        raise SeedError("the runtime selects another seed origin than the plan recorded")
    from merlin_experiments.phase1.session import validate_options

    validate_options(options)
    with tempfile.TemporaryDirectory(prefix="seed-placement-") as scratch:
        workspace, run_dir = Path(scratch) / "workspace", Path(scratch) / "run"
        (workspace / "submission").mkdir(parents=True)
        run_dir.mkdir()
        staged = joint_seed.stage(options, workspace, run_dir)
        placement = {
            "submission_files": sum(1 for p in (workspace / "submission").rglob("*") if p.is_file()),
            "workspace_root": sorted(p.name for p in workspace.iterdir()),
            "certificate_carry_over": staged.get("certificate_carry_over", {}).get("enabled"),
            "previous_session": staged.get("previous_session"),
            "seed_evidence": staged.get("seed_evidence"),
        }
        for path in Path(scratch).rglob("*"):  # staged evidence files are read-only
            if path.is_file():
                path.chmod(0o644)
    return {"seed": "accepted", "origin_sha256": origin["sha256"], "placement": placement}


def verify(ctx, record: dict) -> dict:
    """Check the seed in the campaign runtime (its interpreter, export and environment)."""
    command = ["taskset", "-c", ctx.preparation_cpus, ctx.x("${PYTHON}"), "-B", str(VERIFY_SCRIPT), "verify",
               str(ctx.path("${ARTIFACTS}") / "proposed-launch.json")]
    environment = dict(os.environ, **record["environment"])
    done = subprocess.run(command, cwd=record["cwd"], env=environment, capture_output=True, text=True)
    if done.returncode:
        raise SeedError(f"seed verification failed: {(done.stderr or done.stdout).strip()[-800:]}")
    return json.loads(done.stdout)


def preflight_plan(record: dict, *, run: bool) -> dict:
    """The plan a preflight script renders: the unseeded twin for --preflight-only, else the seeded plan."""
    if run or record.get("seed") is None:
        return record
    return {**record, **{k: record["seed"]["preflight_twin"][k] for k in ("argv", "reserved_argv")}}


if __name__ == "__main__":
    if sys.argv[1:2] != ["verify"] or len(sys.argv) != 3:
        raise SystemExit("usage: fresh_seed.py verify PROPOSED_LAUNCH")
    print(json.dumps(verify_in_runtime(json.loads(Path(sys.argv[2]).read_text())), indent=2))
