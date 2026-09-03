#!/usr/bin/env python3
"""Re-verify an already-certified functional capsule-bench run under the CURRENT evidence schema.

Why this exists
---------------
``perf_campaign.inspect_functional_run`` is the fail-closed input gate of the performance campaign.
It requires evidence that the functional harness only started producing later: an immutable
``bundle_input_snapshot`` v2 record (a real read-only on-disk tree with its own ``snapshot.json``),
and a ``model_host_lane_snapshot`` naming the frozen host-compiler package *inside* that tree.  Runs
that predate the schema carry ``null`` for both, so a submission that genuinely passed 20/20 public
and 5/5 hidden at L3 cannot be consumed.

This tool does NOT back-fill those fields.  It performs a real re-freeze:

* the bundle-input snapshot is MATERIALIZED from the pinned input bundle through the same
  ``targetgen.sandbox.bwrap`` machinery the live harness uses (``materialize_bundle_inputs`` ->
  ``verify_bundle_snapshot`` -> ``snapshot_record``), producing a real immutable tree;
* the host lane is resolved by ``HostLane.resolve`` against ``<snapshot>/repo``, so its digest is a
  content check over the snapshotted bytes;
* the public and hidden grades are RE-RUN against the submission bytes by the official post-freeze
  grader (``capsule_bench/harness/grade_agent_run.py``), which re-hashes, freezes with a real
  ``frozen_at``, and writes fresh ``score_capsule.json`` files.  The source run's score files are
  never copied forward.

What is CARRIED, not re-made
---------------------------
The agent-authoring provenance belongs to the ORIGINAL run: its QA-loop summary, transcript, task
prompt, effort/cost record and archived input-bundle manifest are copied verbatim, and the new
``environment.yaml`` carries a ``refreeze`` block naming the run it re-verifies.  This record is a
re-freeze and says so; it is not an independent result.

Graded denominator
------------------
The re-grade reproduces the ORIGINAL run's graded cohort -- exactly the capsule identities named in
its own score files -- so the re-verified claim has the same scope as the claim it re-verifies.  The
descriptor's cohort has since grown; the ``refreeze.graded_cohort`` block records both the reproduced
cohort and today's descriptor-declared cardinalities so nobody reads the smaller denominator as the
current one.

Every step fails closed.  Nothing here writes a value it did not observe.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

_HARNESS = None  # resolved in _harness_dir()


class RefreezeError(RuntimeError):
    """A precondition of an honest re-freeze was not met."""


# --------------------------------------------------------------------------------------- helpers
def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def _load_yaml(path: Path) -> dict:
    if path.is_symlink() or not path.is_file():
        raise RefreezeError(f"required evidence file is absent or linked: {path}")
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(doc, dict):
        raise RefreezeError(f"required evidence file is not a mapping: {path}")
    return doc


def _load_json(path: Path) -> dict:
    if path.is_symlink() or not path.is_file():
        raise RefreezeError(f"required evidence file is absent or linked: {path}")
    doc = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(doc, dict):
        raise RefreezeError(f"required evidence file is not a mapping: {path}")
    return doc


def _harness_dir(repo: Path) -> Path:
    return repo / "merlin" / "experiments" / "capsule_bench" / "harness"


# ------------------------------------------------------------------------- source-run admission
def _capsule_names(score: Mapping, *, label: str) -> list[str]:
    """The exact capsule identities one recorded phase graded -- refusing anything but a full pass.

    The cohort is read from the score's own per-capsule evidence rather than from a count, so a
    re-grade reproduces the graded SET, not merely its size.
    """
    rows = score.get("per_capsule")
    if not isinstance(rows, list) or not rows:
        raise RefreezeError(f"source {label} score carries no per-capsule evidence")
    names: list[str] = []
    for row in rows:
        if not isinstance(row, Mapping) or not row.get("capsule"):
            raise RefreezeError(f"source {label} score has a malformed per-capsule row")
        if row.get("status") != "pass":
            raise RefreezeError(
                f"source {label} capsule {row.get('capsule')!r} did not pass; refusing to re-freeze "
                "a run that was not already certified")
        tiers = row.get("tiers") or {}
        if tiers.get("L2") != "pass" or tiers.get("L3") != "pass":
            raise RefreezeError(
                f"source {label} capsule {row.get('capsule')!r} did not earn both L2 and L3")
        names.append(str(row["capsule"]))
    if len(set(names)) != len(names):
        raise RefreezeError(f"source {label} score repeats a capsule identity")
    if score.get("functional_pass") != 1 or score.get("gradeable") is not True:
        raise RefreezeError(f"source {label} score is not gradeable and functionally complete")
    if score.get("n_capsules") != len(names) or score.get("n_passed") != len(names):
        raise RefreezeError(f"source {label} score counts disagree with its per-capsule evidence")
    return sorted(names)


def migrate_qa_loop_summary(summary: Mapping) -> tuple[dict, dict]:
    """Carry the ORIGINAL loop summary forward under the current schema, deriving nothing it lacks.

    Two top-level booleans the gate reads -- ``numeric_all_pass`` and ``workflow_conformant`` -- were
    added to the harness after this run.  They are not invented here: each is read off the run's own
    final round (``all_pass`` and ``conformance.conformant``) and is only emitted when that round
    recorded it as exactly ``True``.  A summary that already carries the key must agree with the
    round, or the two records are in conflict and the migration refuses.

    Returns the migrated summary and a provenance block naming every field that moved.
    """
    rounds = summary.get("rounds")
    if not isinstance(rounds, list) or not rounds:
        raise RefreezeError("source QA summary has no completed round")
    for row in rounds:
        if (not isinstance(row, Mapping) or row.get("answer_access_clean") is not True
                or row.get("audit_hits") != []):
            raise RefreezeError(
                "source QA round failed its answer-access audit; a re-freeze cannot clear an audit "
                "hit recorded against the original authoring session")
    final = rounds[-1]
    if summary.get("converged") is not True:
        raise RefreezeError("source QA loop did not converge")
    finalize = summary.get("finalize")
    if (not isinstance(finalize, Mapping) or finalize.get("answer_access_clean") is not True
            or finalize.get("audit_hits") != [] or finalize.get("regrade_all_pass") is not True):
        raise RefreezeError("source finalization did not pass its clean regrade and audit gates")

    if final.get("all_pass") is not True:
        raise RefreezeError("source QA final round did not record a numeric all-pass")
    conformance = final.get("conformance")
    if not isinstance(conformance, Mapping) or conformance.get("conformant") is not True:
        raise RefreezeError("source QA final round did not record workflow conformance")
    failed_checks = [k for k, v in (conformance.get("checks") or {}).items() if v is False]
    if failed_checks:
        raise RefreezeError(f"source QA final round has failing mandated checks: {failed_checks}")

    migrated = dict(summary)
    provenance: dict[str, str] = {}
    for key, source_key, value in (
            ("numeric_all_pass", "rounds[-1].all_pass", True),
            ("workflow_conformant", "rounds[-1].conformance.conformant", True)):
        present = summary.get(key)
        if present is None:
            migrated[key] = value
            provenance[key] = f"schema migration: read from {source_key}"
        elif present is not value:
            raise RefreezeError(
                f"source QA summary records {key}={present!r} while {source_key} says {value!r}")
        else:
            provenance[key] = f"present in source summary (agrees with {source_key})"
    return migrated, provenance


# ------------------------------------------------------------------------------ snapshot staging
def materialize_snapshot(ws: Path, bundle: Mapping, repo: Path) -> tuple[Path, dict]:
    """Build (or re-verify) the immutable v2 bundle-input snapshot through the real bwrap machinery."""
    from merlin.targetgen.sandbox import bwrap as BW
    ws.mkdir(parents=True, exist_ok=True)
    BW.materialize_bundle_inputs(ws, dict(bundle), repo=repo)
    BW.verify_bundle_snapshot(ws, dict(bundle), repo=repo)
    root = BW.bundle_snapshot_root(ws).resolve(strict=True)
    record = BW.snapshot_record(ws)
    if record.get("version") != 2:
        raise RefreezeError(f"bundle snapshot is not the v2 schema the gate requires: {record}")
    return root, record


def host_lane_record(te, snapshot_root: Path, run_snapshot: Mapping) -> dict:
    """Resolve the descriptor's frozen host lane INSIDE the snapshot and bind it to that snapshot."""
    if te.host_lane is None:
        raise RefreezeError("target descriptor declares no host lane; the gate requires one")
    _package, identity = te.resolve_host_lane(root=snapshot_root / "repo")
    identity = dict(identity)
    identity["run_snapshot"] = dict(run_snapshot)
    return identity


# ------------------------------------------------------------------------------- cohort staging
def stage_public_cohort(te, snapshot_root: Path, repo: Path, names: Sequence[str],
                        dest: Path) -> list[str]:
    """Materialize exactly the named public capsules FROM THE SNAPSHOT via the real materializer.

    ``materialize_public_capsules`` is handed the snapshot's own corpus roots and the complement of
    the requested cohort as its exclusion set, so the reproduced set is built by the same code that
    builds a live one -- and an exclusion naming a capsule the corpus does not hold raises there.
    No ``.cohort_admission.json`` is written: this cohort is the reproduced source-run denominator,
    not the descriptor's current admission boundary, and sealing it as the latter would be a lie.
    """
    from merlin.targetgen.contract.materialize import (_public_capsule_dirs_in,
                                                       materialize_public_capsules)
    snapshot_repo = snapshot_root / "repo"
    roots = [snapshot_repo / te.capsule_corpus.relative_to(repo)]
    roots += [snapshot_repo / rel.rstrip("/") for rel in te.corpus_siblings()]
    missing_roots = [str(r) for r in roots if not r.is_dir()]
    if missing_roots:
        raise RefreezeError(f"bundle snapshot has no frozen corpus root(s): {missing_roots}")
    present = {p.name for p in _public_capsule_dirs_in(roots)}
    absent = sorted(set(names) - present)
    if absent:
        raise RefreezeError(
            f"the source run's public cohort names capsule(s) the frozen corpus no longer holds: "
            f"{absent}; the original grade cannot be reproduced against these inputs")
    if dest.exists():
        shutil.rmtree(dest)
    written = materialize_public_capsules(
        dest, tier_ceiling="L3", corpus_roots=roots, exclude=tuple(sorted(present - set(names))))
    if sorted(written) != sorted(names):
        raise RefreezeError(
            f"reproduced public cohort {sorted(written)} is not the source cohort {sorted(names)}")
    return sorted(written)


def stage_hidden_cohort(te, snapshot_root: Path, repo: Path, names: Sequence[str],
                        dest: Path) -> list[str]:
    """Copy exactly the named hidden capsules out of the immutable snapshot.

    Hidden capsules have no materializer (the harness hands the grader the snapshot directory as-is),
    so the reproduced cohort is a verbatim copy of the frozen bytes -- restricted to the identities
    the source run was graded on and to no others.  It stays operator-side under the run directory
    and is never exposed to any agent.
    """
    hidden_rel = te.hidden_corpus()
    if not hidden_rel:
        raise RefreezeError("target descriptor has no hidden corpus beside its capsule corpus")
    source = snapshot_root / "repo" / hidden_rel.rstrip("/")
    if source.is_symlink() or not source.is_dir():
        raise RefreezeError(f"bundle snapshot has no frozen hidden corpus at {source}")
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    written: list[str] = []
    for name in sorted(set(names)):
        src = source / name
        if src.is_symlink() or not src.is_dir() or not (src / "capsule.yaml").is_file():
            raise RefreezeError(
                f"the source run's hidden cohort names {name!r}, which the frozen hidden corpus "
                "does not hold; the original grade cannot be reproduced against these inputs")
        shutil.copytree(src, dest / name, symlinks=False)
        # copytree replays the snapshot's cleared write bits onto the copy; restore owner-write on the
        # copy (including its own root) so this run-private cohort stays removable.
        for path in (dest / name, *(dest / name).rglob("*")):
            path.chmod(path.stat().st_mode | 0o200)
        written.append(name)
    if sorted(written) != sorted(names):
        raise RefreezeError("reproduced hidden cohort is not the source cohort")
    return written


# ------------------------------------------------------------------------------------ the record
def build_environment(source_env: Mapping, *, new_run_id: str, snapshot_record: Mapping,
                      host_lane: Mapping, refreeze: Mapping) -> dict:
    env = dict(source_env)
    env["run_id"] = new_run_id
    env["bundle_input_snapshot"] = dict(snapshot_record)
    env["model_host_lane_snapshot"] = dict(host_lane)
    env["refreeze"] = dict(refreeze)
    return env


def _copy_carried_provenance(source_dir: Path, run_dir: Path) -> list[str]:
    """Copy the ORIGINAL authoring evidence forward verbatim. Score files are deliberately absent."""
    carried: list[str] = []
    for name in ("TASK.md", "input_bundle_manifest.yaml", "cost_time_toolcalls.yaml",
                 "transcript.jsonl", "selfcheck_log.jsonl", "oracle_preflight.yaml",
                 "codegen_smoke.yaml", "qa_loop_state.yaml"):
        src = source_dir / name
        if src.is_file() and not src.is_symlink():
            shutil.copy2(src, run_dir / name)
            carried.append(name)
    for name in ("rounds", "agent", "metrics", "logs", "qa_history"):
        src = source_dir / name
        if src.is_dir() and not src.is_symlink():
            dst = run_dir / name
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst, symlinks=False)
            carried.append(name + "/")
    return sorted(carried)


# ---------------------------------------------------------------------- post-grade submission state
def purge_interpreter_bytecode(submission: Path, expected_digest: str) -> list[str]:
    """Remove the bytecode caches GRADING leaves inside the submission, and prove nothing else moved.

    ``grade_agent_run`` imports the submitted package, so CPython writes ``__pycache__`` into the
    frozen tree during the public phase.  ``hash_tree`` skips those names, so the submission digest
    never sees them -- which is exactly why the campaign gate refuses them: they are executable state
    the content address does not cover.  Every graded run therefore ends up carrying state that makes
    it gate-invalid, and sweeping it by hand is how that fact stays invisible.

    Only interpreter-generated bytecode is removed.  ``build/`` and ``.git/`` are NOT swept: their
    presence means something real is in the tree, and deciding that for the submitter would hide it.
    The digest is recomputed afterwards and must be unchanged, so this can never quietly alter the
    bytes the grade was earned on.
    """
    submission = Path(submission)
    kept = [str(path) for path in submission.rglob("*")
            if {"build", ".git"} & set(path.relative_to(submission).parts)]
    if kept:
        raise RefreezeError(
            f"submission carries non-bytecode digest-excluded state this tool will not remove: "
            f"{sorted(kept)[:5]}")
    removed: list[str] = []
    for path in sorted(submission.rglob("__pycache__"), key=lambda p: len(p.parts), reverse=True):
        if path.is_symlink() or not path.is_dir():
            raise RefreezeError(f"refusing to remove an unsafe bytecode cache: {path}")
        shutil.rmtree(path)
        removed.append(str(path.relative_to(submission)))
    for path in sorted(submission.rglob("*.py[co]")):
        if path.is_symlink() or not path.is_file():
            continue
        path.unlink()
        removed.append(str(path.relative_to(submission)))
    from merlin.benchharness import hash_tree
    observed = hash_tree(submission)["sha256"]
    if observed != expected_digest:
        raise RefreezeError(
            f"removing interpreter bytecode changed the submission digest ({expected_digest} -> "
            f"{observed}); the tree held more than bytecode and is no longer the graded artifact")
    return sorted(removed)


# ------------------------------------------------------------------------------------- the grade
def run_official_grade(repo: Path, run_dir: Path, *, arm: str, model: str, public_root: Path,
                       hidden_root: Path, snapshot_root: Path, descriptor: Path) -> int:
    """Invoke the official post-freeze grader -- the same entry point the live harness uses."""
    cmd = [sys.executable, str(_harness_dir(repo) / "grade_agent_run.py"),
           "--run-dir", str(run_dir), "--arm", arm, "--model", model,
           "--capsules", str(public_root), "--hidden-capsules", str(hidden_root)]
    env = dict(os.environ)
    env["MERLIN_TARGET_EXPERIMENT"] = str(descriptor)
    env["MERLIN_MODEL_HOST_LANE_SNAPSHOT_ROOT"] = str(snapshot_root)
    env["MERLIN_MODEL_HOST_LANE_SNAPSHOT_REQUIRED"] = "1"
    print(f"[refreeze] grading: {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, cwd=str(repo), env=env).returncode


# ------------------------------------------------------------------------------------------ main
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source-run-id", required=True,
                    help="the already-certified functional run to re-verify")
    ap.add_argument("--new-run-id", required=True, help="run id for the re-freeze record")
    ap.add_argument("--descriptor", default=None,
                    help="target_experiment.yaml (default: MERLIN_TARGET_EXPERIMENT / harness default)")
    ap.add_argument("--bundle-dir", default=None,
                    help="input bundle directory to snapshot (default: the bundle the source run names)")
    ap.add_argument("--reason", default="unblock the performance campaign input gate: the source run "
                                        "predates the immutable bundle-input snapshot v2 schema")
    ap.add_argument("--skip-grade", action="store_true",
                    help="stage the snapshot + cohorts and stop (does NOT produce a gate-valid record)")
    ap.add_argument("--verify-only", action="store_true",
                    help="only run the campaign gate against an existing re-freeze record")
    a = ap.parse_args(argv)

    if a.descriptor:
        os.environ["MERLIN_TARGET_EXPERIMENT"] = str(Path(a.descriptor).expanduser().resolve())
    from merlin.benchharness import hash_tree
    from merlin.common.paths import repo_root
    from merlin.targetgen.target_experiment import load_target_experiment
    # The capsule-bench harness modules (_common, grade_agent_run, freeze_run) are a script package,
    # not an installed one; put their home on the path before importing the target selector.
    sys.path.insert(0, str(_harness_dir(repo_root())))
    import _common as C  # noqa: E402 -- resolves the target only after the descriptor is selected

    repo, runs_root = C.REPO, C.RUNS
    arm_root = runs_root / "merlin_assisted"
    source_dir = arm_root / a.source_run_id
    run_dir = arm_root / a.new_run_id
    if Path(a.new_run_id).name != a.new_run_id:
        raise RefreezeError("new run id must be a simple directory name")

    if a.verify_only:
        return _verify(runs_root, a.new_run_id, run_dir)

    if not source_dir.is_dir():
        raise RefreezeError(f"source run does not exist: {source_dir}")
    if run_dir.exists():
        raise RefreezeError(f"refusing to overwrite an existing run: {run_dir}")

    source_env = _load_yaml(source_dir / "environment.yaml")
    source_manifest = _load_yaml(source_dir / "run_manifest.yaml")
    source_summary = _load_yaml(source_dir / "qa_loop_summary.yaml")
    source_freeze = _load_json(source_dir / "freeze.json")
    source_public = _load_json(source_dir / "grading_public" / "score_capsule.json")
    source_hidden = _load_json(source_dir / "grading_hidden" / "score_capsule.json")

    if source_env.get("sandbox") != "bwrap":
        raise RefreezeError("source run was not executed in the required bwrap sandbox")
    if source_env.get("isolation_violations") != []:
        raise RefreezeError("source run lacks a clean isolation-violation audit")
    bundle_id = str(source_env.get("bundle_id") or "")
    if not bundle_id:
        raise RefreezeError("source run does not name its input bundle")

    migrated_summary, migration = migrate_qa_loop_summary(source_summary)
    public_names = _capsule_names(source_public, label="public")
    hidden_names = _capsule_names(source_hidden, label="hidden")
    if set(public_names) & set(hidden_names):
        raise RefreezeError("source public and hidden cohorts reuse a capsule identity")

    source_digest = str(source_manifest.get("submission_sha256") or "")
    observed = hash_tree(source_dir / "submission")["sha256"]
    recorded = {"run_manifest": source_manifest.get("submission_sha256"),
                "freeze": source_freeze.get("submission_sha256"),
                "freeze_recheck": source_freeze.get("submission_sha256_recheck"),
                "submission_tree": observed}
    if not source_digest or any(v != source_digest for v in recorded.values()):
        raise RefreezeError(f"source submission digest does not match every record: {recorded}")

    te = load_target_experiment(C.DESCRIPTOR)
    bundle_dir = (Path(a.bundle_dir).expanduser().resolve() if a.bundle_dir
                  else C.BUNDLES / bundle_id)
    bundle_manifest_path = bundle_dir / "input_bundle_manifest.yaml"
    bundle = _load_yaml(bundle_manifest_path)
    if str(bundle.get("bundle_id")) != bundle_id:
        raise RefreezeError(
            f"bundle at {bundle_dir} declares {bundle.get('bundle_id')!r}, not the source run's "
            f"{bundle_id!r}")
    host_pkg_rel = te.host_lane.package.rstrip("/") if te.host_lane else None
    granted = {str(e.get("path", "")).rstrip("/") for e in bundle.get("allowed", [])}
    if host_pkg_rel and not any(host_pkg_rel == g or host_pkg_rel.startswith(g + "/")
                                for g in granted if g):
        raise RefreezeError(
            f"bundle {bundle_id!r} at {bundle_dir} does not grant the descriptor's host-lane package "
            f"{host_pkg_rel!r}; its snapshot could not contain the package the gate requires. Point "
            "--bundle-dir at a bundle that grants it (the generated bundle does).")

    archived = source_dir / "input_bundle_manifest.yaml"
    archived_granted = sorted(
        {str(e.get("path", "")).rstrip("/") for e in (_load_yaml(archived).get("allowed") or [])}
    ) if archived.is_file() else []

    run_dir.mkdir(parents=True)
    print(f"[refreeze] run dir: {run_dir}", flush=True)

    # 1. the submission bytes, unchanged and re-hashed.
    shutil.copytree(source_dir / "submission", run_dir / "submission",
                    symlinks=False, ignore=shutil.ignore_patterns("build", "__pycache__", ".git"))
    copied = hash_tree(run_dir / "submission")["sha256"]
    if copied != source_digest:
        raise RefreezeError(
            f"copied submission digest {copied} does not match the source {source_digest}")

    carried = _copy_carried_provenance(source_dir, run_dir)
    (run_dir / "qa_loop_summary.yaml").write_text(
        yaml.safe_dump(migrated_summary, sort_keys=False), encoding="utf-8")

    # 2. the immutable bundle-input snapshot, through the real bwrap machinery.
    ws = run_dir / "refreeze_ws" / "workspace"
    print(f"[refreeze] materializing bundle-input snapshot beside {ws} ...", flush=True)
    started = _dt.datetime.now(_dt.timezone.utc)
    snapshot_root, snapshot = materialize_snapshot(ws, bundle, repo)
    print(f"[refreeze] snapshot: {snapshot_root} "
          f"({snapshot['n_files']} files, {snapshot['n_bytes']} bytes, "
          f"{(_dt.datetime.now(_dt.timezone.utc) - started).total_seconds():.1f}s)", flush=True)
    host_lane = host_lane_record(te, snapshot_root, snapshot)

    # 3. the reproduced graded cohorts, taken from the frozen bytes.
    cohort_root = run_dir / "refreeze_cohort"
    public_root = cohort_root / "public_capsules"
    hidden_root = cohort_root / "hidden_capsules"
    stage_public_cohort(te, snapshot_root, repo, public_names, public_root)
    stage_hidden_cohort(te, snapshot_root, repo, hidden_names, hidden_root)
    print(f"[refreeze] cohort: {len(public_names)} public + {len(hidden_names)} hidden "
          f"(reproduced from the source run's own score files)", flush=True)

    refreeze = {
        "version": 1,
        "kind": "functional_refreeze",
        "of_run_id": a.source_run_id,
        "of_run_dir": str(source_dir),
        "of_submission_sha256": source_digest,
        "of_frozen_at": source_freeze.get("frozen_at"),
        "reason": a.reason,
        "refrozen_at": _now(),
        "is_independent_result": False,
        "agent_authoring": {
            "note": "the authoring session belongs to the source run; no agent ran for this record",
            "run_id": a.source_run_id,
            "model": source_env.get("model"),
            "driver": source_env.get("driver"),
            "provider": source_env.get("provider"),
            "started_at": source_env.get("started_at"),
            "carried_evidence": carried,
            "source_score_files": {
                "public": str(source_dir / "grading_public" / "score_capsule.json"),
                "hidden": str(source_dir / "grading_hidden" / "score_capsule.json"),
                "note": "NOT copied forward; this record's scores are freshly re-run",
            },
        },
        "qa_loop_summary_migration": migration,
        "graded_cohort": {
            "source": "the capsule identities named in the source run's own score files",
            "public_capsules": public_names,
            "hidden_capsules": hidden_names,
            "n_public": len(public_names),
            "n_hidden": len(hidden_names),
            "descriptor_expected_source_capsules": te.graded_expected_source_capsules,
            "descriptor_expected_admitted_capsules": te.graded_expected_admitted_capsules,
            "note": "this re-freeze reproduces the ORIGINAL run's denominator so the re-verified "
                    "claim has the same scope as the claim it re-verifies. It is NOT a grade against "
                    "the descriptor's current admitted cohort, which is larger.",
        },
        "bundle": {
            "bundle_id": bundle_id,
            "snapshotted_manifest": str(bundle_manifest_path),
            "archived_manifest": str(archived) if archived.is_file() else None,
            "grants_added_since_the_source_run": sorted(
                {g for g in granted if g} - set(archived_granted)),
            "grants_removed_since_the_source_run": sorted(
                set(archived_granted) - {g for g in granted if g}),
            "note": "the snapshot pins the inputs THIS re-grade consumed, taken from the bundle that "
                    "grants the descriptor's host lane; the source run's own archived manifest is "
                    "carried beside it unchanged.",
        },
    }
    env = build_environment(source_env, new_run_id=a.new_run_id, snapshot_record=snapshot,
                            host_lane=host_lane, refreeze=refreeze)
    (run_dir / "environment.yaml").write_text(yaml.safe_dump(env, sort_keys=False), encoding="utf-8")

    if a.skip_grade:
        print("[refreeze] --skip-grade: staged only; no gate-valid record was produced")
        return 0

    # 4. the real re-grade (public -> freeze -> hidden), by the official post-freeze grader.
    grade_started = _dt.datetime.now(_dt.timezone.utc)
    rc = run_official_grade(repo, run_dir, arm=str(source_manifest.get("arm") or "unknown"),
                            model=str(source_env.get("model") or "unknown"),
                            public_root=public_root, hidden_root=hidden_root,
                            snapshot_root=snapshot_root, descriptor=C.DESCRIPTOR)
    elapsed = (_dt.datetime.now(_dt.timezone.utc) - grade_started).total_seconds()
    print(f"[refreeze] re-grade returned {rc} after {elapsed:.1f}s", flush=True)
    swept = purge_interpreter_bytecode(run_dir / "submission", source_digest)
    if swept:
        print(f"[refreeze] swept grader-written bytecode from the frozen submission: {swept}",
              flush=True)
    (run_dir / "refreeze_regrade.json").write_text(json.dumps({
        "started_at": grade_started.isoformat(), "wall_seconds": elapsed,
        "grader_returncode": rc, "public_capsules": public_names,
        "hidden_capsules": hidden_names,
        "interpreter_bytecode_swept": swept,
        "interpreter_bytecode_note": (
            "grade_agent_run imports the submitted package, so CPython writes __pycache__ into the "
            "frozen tree. hash_tree excludes those names, so the digest is unchanged; the campaign "
            "gate refuses them as unhashed executable state. Removed after grading, digest "
            "re-verified.")}, indent=2), encoding="utf-8")

    # environment.yaml is rewritten last: the grader owns run_manifest/freeze, and the run id must
    # agree across all three.
    return _verify(runs_root, a.new_run_id, run_dir)


def _verify(runs_root: Path, run_id: str, run_dir: Path) -> int:
    import perf_campaign as PC
    manifest = _load_yaml(run_dir / "run_manifest.yaml")
    digest = str(manifest.get("submission_sha256") or "")
    # Idempotent, and repeated deliberately: ANY process that imports the submitted package writes
    # bytecode back into the frozen tree, so a record that verified yesterday can be refused today for
    # a reason that has nothing to do with its evidence.
    swept = purge_interpreter_bytecode(run_dir / "submission", digest)
    if swept:
        print(f"[refreeze] swept grader-written bytecode before verification: {swept}")
    try:
        record = PC.inspect_functional_run(runs_root, run_id, digest)
    except PC.CampaignGateError as exc:
        print(f"REFUSED  {run_id}  digest={digest}\n  {exc}")
        return 1
    print(f"ACCEPTED {run_id}  digest={record.digest}  "
          f"public={record.public_capsules} hidden={record.hidden_capsules}  "
          f"frozen_at={record.frozen_at}")
    print(f"  bundle_input_snapshot: {record.bundle_input_snapshot['path']}")
    print(f"  model_host_lane_package: {record.model_host_package}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RefreezeError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        raise SystemExit(2)
