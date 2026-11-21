"""Phase-1 authoring workspace placement and read-only historical discovery.

New mutable work lives under generated output. Archived environment records remain authoritative:
resuming an old run does not move its workspace or rewrite its scientific provenance.
"""

from __future__ import annotations

import json
import sys
import time
from functools import wraps
from pathlib import Path

import yaml

from merlin.common.oot_repository import (
    FROZEN_TAG,
    CommitRecord,
    OotRepoError,
    check_outside_sandbox,
    commit_candidate,
    init,
    package_digest,
    resolve,
    tag,
    tree_digest,
    verify,
)
from merlin.common.paths import build_dir, repo_root, resolve_grant

# ---- workspaces --------------------------------------------------------------------------------


def _component(value: str) -> str:
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_."
    if not value or value in {".", ".."} or any(character not in allowed for character in value):
        raise ValueError(f"invalid workspace path component: {value!r}")
    return value


def workspace_parent(target: str, arm: str) -> Path:
    """Generated workspace root; honors MERLIN_OUT_ROOT without creating directories."""
    return build_dir() / "agent-workspaces" / "phase1" / _component(target) / _component(arm)


def recorded_workspace(run_dir: Path) -> Path | None:
    environment = run_dir / "environment.yaml"
    if not environment.is_file():
        return None
    data = yaml.safe_load(environment.read_text(encoding="utf-8")) or {}
    value = data.get("workspace_path")
    if not value:
        return None
    path = Path(value)
    if not path.is_absolute() or path.name != "workspace":
        raise ValueError(f"invalid archived workspace path: {value!r}")
    return path


def workspace_candidates(
    run_dir: Path,
    *,
    target: str | None = None,
    arm: str | None = None,
    experiment: Path | None = None,
    repo: Path | None = None,
) -> list[Path]:
    """Recorded, generated, then legacy workspace paths; discovery never changes artifacts."""
    run_id = _component(run_dir.name)
    candidates = []
    archived = recorded_workspace(run_dir)
    if archived is not None:
        candidates.append(archived)
    generated = build_dir() / "agent-workspaces" / "phase1"
    target_part = _component(target) if target else "*"
    arm_part = _component(arm) if arm else "*"
    candidates.extend(sorted(generated.glob(f"{target_part}/{arm_part}/{run_id}/workspace")))
    if experiment is not None:
        candidates.append(experiment / "_qa_ws" / run_id / "workspace")
    else:
        targets = resolve_grant("merlin/experiments/capsule_bench/targets", root=repo or repo_root())
        candidates.extend(sorted(targets.glob(f"*/_qa_ws/{run_id}/workspace")))
    return list(dict.fromkeys(candidates))


def select_workspace_root(*, target: str, arm: str, run_dir: Path, experiment: Path, resume: bool) -> Path:
    if resume:
        for workspace in workspace_candidates(run_dir, target=target, arm=arm, experiment=experiment):
            if workspace.is_dir():
                return workspace.parent
    return workspace_parent(target, arm) / _component(run_dir.name)


def workspace_session(function):
    """Release workspace leases on an orderly return, not after an uncertain crash.

    The driver owns child shutdown. An exception may strand a child, so retain its lease until an
    operator has checked the crashed process tree and explicitly acknowledges abandonment. ``wraps``
    preserves the public driver's identity and source-based control-flow regression tests.
    """

    @wraps(function)
    def invoke(*args, **kwargs):
        leases = []
        try:
            result = function(*args, _workspace_leases=leases, **kwargs)
        except BaseException:
            if leases:
                print(
                    "Phase-1 workspace leases retained after interrupted execution: "
                    + ", ".join(held.path for held in leases)
                    + ". Child shutdown is uncertain. After this process exits, verify that all "
                    "child workers have stopped, then use storage_lifecycle.acknowledge_abandoned"
                    "(path, reason=...) to release crash protection. Retention pins remain intact.",
                    file=sys.stderr,
                )
            raise
        for held in reversed(leases):
            held.close("completed" if result == 0 else "failed")
        return result

    return invoke


# ---- oot_history -------------------------------------------------------------------------------
# The harness-owned compiler history of one phase-1 run: ``<run_dir>/oot``.
#
# Every graded round commits the snapshot the grader is about to grade -- the operator-only copy, never
# the live workspace -- and the official freeze tags ``frozen`` on the exact submission it hashed. Round
# records then carry a commit sha instead of a copy of the package. Only this module writes the repo,
# through :mod:`merlin.common.oot_repository`; the authoring agent never runs git here, and the repo is refused
# when it overlaps the agent's writable workspace.
#
# A run created before this history existed has no ``oot/``; it stays readable and records nothing,
# rather than acquiring a history that starts mid-run. A commit that fails is RECORDED, never silent:
# the round row says so and the freeze record carries the error, so a missing history cannot be
# mistaken for a run that had nothing to commit.

OOT_DIR = "oot"

ROUNDS_LOG = "oot_commits.jsonl"

FREEZE_LABEL = "freeze"


def repo_path(run_dir: Path) -> Path:
    return Path(run_dir) / OOT_DIR


def start(run_dir: Path, *, sandbox_roots=(), resuming: bool = False) -> Path | None:
    """Reserve the run's history (fresh run) or find it again (resume). A legacy run has none.

    Admission only checks the location and creates the empty ``oot/``; git first runs at the first
    graded commit, so admission itself still launches no process.
    """
    repo = repo_path(run_dir)
    if resuming:
        if not repo.is_dir():
            return None
        check_outside_sandbox(repo, sandbox_roots)
        return repo
    check_outside_sandbox(repo, sandbox_roots)
    repo.mkdir()
    return repo


def _ready(repo: Path) -> bool:
    """Whether the run keeps a history, initializing a reserved (empty) one on first use."""
    if (repo / ".git").is_dir():
        return True
    if repo.is_dir() and not any(repo.iterdir()):
        init(repo)
        return True
    return False


def _log(run_dir: Path, row: dict) -> None:
    with (Path(run_dir) / ROUNDS_LOG).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, sort_keys=True) + "\n")


def commit_graded(run_dir: Path, snapshot: Path, *, label: str, key: str, sandbox_roots=()) -> CommitRecord | None:
    """Commit the snapshot about to be graded as one round. None when the run keeps no history."""
    repo = repo_path(run_dir)
    try:
        if not _ready(repo):
            return None
        return commit_candidate(
            repo,
            snapshot,
            label=f"{label} {key}",
            when=int(time.time()),
            run_id=Path(run_dir).name,
            metadata={"label": label, "key": key},
            sandbox_roots=sandbox_roots,
        )
    except OotRepoError as exc:
        _log(run_dir, {"label": label, "key": key, "error": str(exc)})
        print(f"[oot] {label} {key}: commit FAILED: {exc}", flush=True)
        return None


def record_round(run_dir: Path, record: CommitRecord | None, *, label: str, key: str, verdict: dict) -> None:
    """Append the round's commit and its grade to the run's commit log (host-only)."""
    if record is None:
        return
    _log(
        run_dir,
        {
            "label": label,
            "key": key,
            "oot": record.as_record(),
            "all_pass": verdict.get("all_pass"),
            "n_passed": verdict.get("n_passed"),
            "n_capsules": verdict.get("n_capsules"),
        },
    )


def verify_turn_end(run_dir: Path, submission: Path, commit: str | None) -> dict:
    """Void a turn if live source differs from its final agent-source OOT commit."""
    if not commit:
        raise OotRepoError("turn ended without an OOT source commit")
    repo = repo_path(run_dir)
    if resolve(repo, "HEAD") != commit:
        raise OotRepoError("turn's OOT commit is no longer the final source commit")
    committed = tree_digest(repo, commit)
    live = package_digest(submission)
    record = {
        "commit": commit,
        "committed_tree_sha256": committed,
        "live_tree_sha256": live,
        "matched": committed == live,
    }
    (Path(run_dir) / "turn_end_provenance.json").write_text(json.dumps(record, indent=2) + "\n")
    if committed != live:
        raise OotRepoError("submission changed outside the graded agent-source OOT commit")
    return record


def freeze(run_dir: Path, submission: Path) -> dict | None:
    """Tag ``frozen`` on the exact frozen submission, committing it first if no round holds it."""
    repo = repo_path(run_dir)
    try:
        if not _ready(repo):
            return None
        digest = package_digest(submission)
        head = None
        try:
            head = resolve(repo, "HEAD")
        except OotRepoError:
            pass  # no graded round committed yet
        if head is not None and tree_digest(repo, head) == digest:
            commit = head
        else:
            commit = commit_candidate(
                repo, submission, label=FREEZE_LABEL, when=int(time.time()), run_id=Path(run_dir).name
            ).commit
        tag(repo, FROZEN_TAG, commit)
        verify(repo, FROZEN_TAG, digest)
        return {"repo": str(repo), "frozen_commit": commit, "package_digest": digest}
    except OotRepoError as exc:
        print(f"[oot] freeze tag FAILED: {exc}", flush=True)
        return {"repo": str(repo), "error": str(exc)}


# ---- continuation_handoff ----------------------------------------------------------------------
# Tell a continuation's new session where the previous session left off.
#
# A relaunch seeds the new workspace with the previous compiler, but the new author session starts with no
# memory of the last one: its open threads, the hypothesis it was testing and the failures it was looking
# at are gone, and the first part of the new session is spent rediscovering them. ``PREVIOUS_SESSION.md``
# is a compact, read-only summary built only from what the previous session itself wrote or was shown:
#
# * its last turn report (``rounds/*.final.txt``);
# * the tail of its own ``ITERATION_NOTES.md`` (workspace root; legacy: ``docs/iteration_notes.md`` in the
#   seeded submission);
# * the open failures and largest performance headroom of the newest grade it could see
#   (``qa/latest_completed_grade.json``, else ``qa/verdict.json``).
#
# Host paths are scrubbed and every section is bounded. Missing inputs say "not recorded"; nothing here can
# fail a launch.

FILENAME = "PREVIOUS_SESSION.md"

SECTION_CHARS = 4000

MAX_FAILURES = 30

MAX_HEADROOM = 5


def _tail(text: str, limit: int = SECTION_CHARS) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    cut = text[-limit:]
    newline = cut.find("\n")
    return "…\n" + (cut[newline + 1 :] if 0 <= newline < len(cut) - 1 else cut)


def _has_notes(previous_workspace, seed_submission) -> bool:

    return (
        existing_notes(previous_workspace) is not None
        or (Path(seed_submission) / "docs" / "iteration_notes.md").is_file()
    )


def _read(path: Path) -> str | None:
    try:
        if path.is_symlink() or not path.is_file():
            return None
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def _last_turn_report(run_dir: Path) -> str | None:
    rounds = run_dir / "rounds"
    try:
        reports = sorted(p for p in rounds.iterdir() if p.name.endswith(".final.txt") and p.is_file())
    except OSError:
        return None
    return _read(reports[-1]) if reports else None


def _latest_grade(workspace: Path) -> dict | None:
    for name in ("latest_completed_grade.json", "verdict.json"):
        text = _read(workspace / "qa" / name)
        if not text:
            continue
        try:
            document = json.loads(text)
        except ValueError:
            continue
        if isinstance(document, dict) and document.get("per_capsule"):
            return document
    return None


def _failures(grade: dict) -> list[str]:
    rows = []
    for row in grade.get("per_capsule") or ():
        if not isinstance(row, dict):
            continue
        tiers = row.get("tiers") if isinstance(row.get("tiers"), dict) else {}
        failed = row.get("status") == "fail" or any(value == "fail" for value in tiers.values())
        if failed:
            reason = row.get("failure_category") or row.get("failure_plane") or "fail"
            rows.append(f"- `{row.get('capsule')}`: {reason}")
    return rows


def _headroom(grade: dict) -> list[str]:
    priority = (grade.get("feedback_brief") or {}).get("performance_priority") or {}
    block = priority.get("development_headroom") if isinstance(priority, dict) else None
    entries = (block or {}).get("largest_headroom") or [] if isinstance(block, dict) else []
    lines = []
    for entry in entries[:MAX_HEADROOM]:
        if isinstance(entry, dict):
            lines.append(
                f"- `{entry.get('test_case')}`: {entry.get('cycles')} cycles, "
                f"{entry.get('gap_ratio')}x its roofline bound"
            )
    return lines


def render(previous_run_dir: Path, previous_workspace: Path, seed_submission: Path) -> str:
    """The note's text; every section says ``not recorded`` when its input is absent."""
    from merlin.common.access import scrub_host_paths

    report = _last_turn_report(previous_run_dir)

    found = existing_notes(previous_workspace)
    notes = _read(found) if found is not None else _read(seed_submission / "docs" / "iteration_notes.md")
    grade = _latest_grade(previous_workspace)
    parts = [
        "# Where the previous session left off",
        "",
        "Operator-generated from the previous session's own outputs. It grants no credit: every result "
        "below is about the previous compiler bytes and is re-checked in this run.",
        "",
        "## Its last turn report",
        "",
        _tail(report) if report else "not recorded",
        "",
        "## End of its iteration notes",
        "",
        _tail(notes) if notes else "not recorded",
        "",
        "## Open failures in the newest grade it could see",
        "",
    ]
    if grade is None:
        parts.append("not recorded")
    else:
        source = str(grade.get("graded_source_sha256") or "")[:12] or "unknown"
        failures = _failures(grade)
        parts.append(f"Graded source `{source}`; {len(failures)} failing test case(s).")
        parts.extend(failures[:MAX_FAILURES])
        if len(failures) > MAX_FAILURES:
            parts.append(f"- … {len(failures) - MAX_FAILURES} more")
        headroom = _headroom(grade)
        parts += ["", "## Largest performance headroom it was shown", ""]
        parts.extend(headroom or ["not recorded"])
    return scrub_host_paths("\n".join(parts) + "\n")


def write(workspace, previous: dict, seed_submission) -> dict:
    """Stage ``PREVIOUS_SESSION.md`` read-only in the new workspace; returns what was written."""
    run_dir, old_workspace = Path(previous["run_dir"]), Path(previous["workspace"])
    text = render(run_dir, old_workspace, Path(seed_submission))
    target = Path(workspace) / FILENAME
    target.unlink(missing_ok=True)
    target.write_text(text, encoding="utf-8")
    target.chmod(0o444)
    return {
        "file": FILENAME,
        "bytes": len(text.encode("utf-8")),
        "inputs_found": {
            "turn_report": _last_turn_report(run_dir) is not None,
            "iteration_notes": _has_notes(old_workspace, seed_submission),
            "grade": _latest_grade(old_workspace) is not None,
        },
    }


# ---- iteration_notes ---------------------------------------------------------------------------
# Where an author keeps its cross-round notes: the workspace root, outside the graded compiler.
#
# The notes used to live at ``submission/docs/iteration_notes.md``. Everything under ``submission/``
# is the compiler's identity (only ``build``, ``__pycache__`` and ``.git`` are ignored,
# :data:`.run_inputs.SNAPSHOT_IGNORED`), so every note edit -- which every brief asks for, every round --
# moved the graded source digest: each reconciled grade row became an "older version" and the verdict's
# ``snapshot_current`` went false until the next grade finished.
#
# The notes now live at ``ITERATION_NOTES.md`` in the workspace root, next to the experiment's other
# author records. Excluding ``submission/docs/`` from the identity instead was rejected: compiler code can
# open files there (the docs directory also holds agent-written JSON proofs), so excluding it would let
# bytes a graded command can read escape the identity, and it would change every existing digest.
#
# Readers fall back to the legacy location, so a seeded or older workspace keeps its notes.

NOTES_NAME = "ITERATION_NOTES.md"

LEGACY_NOTES = Path("submission") / "docs" / "iteration_notes.md"


def notes_path(workspace) -> Path:
    """The file an author writes its notes to."""
    return Path(workspace) / NOTES_NAME


def existing_notes(workspace) -> Path | None:
    """The notes file to read: the root file, else the legacy one; ``None`` when neither exists."""
    for path in (notes_path(workspace), Path(workspace) / LEGACY_NOTES):
        if path.is_file():
            return path
    return None


def read_notes(workspace) -> str:
    """The author's notes text (``""`` when there are none)."""
    path = existing_notes(workspace)
    return path.read_text(encoding="utf-8", errors="replace") if path is not None else ""
