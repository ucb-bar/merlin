"""The harness-owned compiler history of one phase-1 run: ``<run_dir>/oot``.

Every graded round commits the snapshot the grader is about to grade -- the operator-only copy, never
the live workspace -- and the official freeze tags ``frozen`` on the exact submission it hashed. Round
records then carry a commit sha instead of a copy of the package. Only this module writes the repo,
through :mod:`merlin.common.oot_repo`; the authoring agent never runs git here, and the repo is refused
when it overlaps the agent's writable workspace.

A run created before this history existed has no ``oot/``; it stays readable and records nothing,
rather than acquiring a history that starts mid-run. A commit that fails is RECORDED, never silent:
the round row says so and the freeze record carries the error, so a missing history cannot be
mistaken for a run that had nothing to commit.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from merlin.common import oot_repo as O

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
        O.check_outside_sandbox(repo, sandbox_roots)
        return repo
    O.check_outside_sandbox(repo, sandbox_roots)
    repo.mkdir()
    return repo


def _ready(repo: Path) -> bool:
    """Whether the run keeps a history, initializing a reserved (empty) one on first use."""
    if (repo / ".git").is_dir():
        return True
    if repo.is_dir() and not any(repo.iterdir()):
        O.init(repo)
        return True
    return False


def _log(run_dir: Path, row: dict) -> None:
    with (Path(run_dir) / ROUNDS_LOG).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, sort_keys=True) + "\n")


def commit_graded(run_dir: Path, snapshot: Path, *, label: str, key: str, sandbox_roots=()) -> O.CommitRecord | None:
    """Commit the snapshot about to be graded as one round. None when the run keeps no history."""
    repo = repo_path(run_dir)
    try:
        if not _ready(repo):
            return None
        return O.commit_candidate(
            repo,
            snapshot,
            label=f"{label} {key}",
            when=int(time.time()),
            run_id=Path(run_dir).name,
            metadata={"label": label, "key": key},
            sandbox_roots=sandbox_roots,
        )
    except O.OotRepoError as exc:
        _log(run_dir, {"label": label, "key": key, "error": str(exc)})
        print(f"[oot] {label} {key}: commit FAILED: {exc}", flush=True)
        return None


def record_round(run_dir: Path, record: O.CommitRecord | None, *, label: str, key: str, verdict: dict) -> None:
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
        raise O.OotRepoError("turn ended without an OOT source commit")
    repo = repo_path(run_dir)
    if O.resolve(repo, "HEAD") != commit:
        raise O.OotRepoError("turn's OOT commit is no longer the final source commit")
    committed = O.tree_digest(repo, commit)
    live = O.package_digest(submission)
    record = {
        "commit": commit,
        "committed_tree_sha256": committed,
        "live_tree_sha256": live,
        "matched": committed == live,
    }
    (Path(run_dir) / "turn_end_provenance.json").write_text(json.dumps(record, indent=2) + "\n")
    if committed != live:
        raise O.OotRepoError("submission changed outside the graded agent-source OOT commit")
    return record


def freeze(run_dir: Path, submission: Path) -> dict | None:
    """Tag ``frozen`` on the exact frozen submission, committing it first if no round holds it."""
    repo = repo_path(run_dir)
    try:
        if not _ready(repo):
            return None
        digest = O.package_digest(submission)
        head = None
        try:
            head = O.resolve(repo, "HEAD")
        except O.OotRepoError:
            pass  # no graded round committed yet
        if head is not None and O.tree_digest(repo, head) == digest:
            commit = head
        else:
            commit = O.commit_candidate(
                repo, submission, label=FREEZE_LABEL, when=int(time.time()), run_id=Path(run_dir).name
            ).commit
        O.tag(repo, O.FROZEN_TAG, commit)
        O.verify(repo, O.FROZEN_TAG, digest)
        return {"repo": str(repo), "frozen_commit": commit, "package_digest": digest}
    except O.OotRepoError as exc:
        print(f"[oot] freeze tag FAILED: {exc}", flush=True)
        return {"repo": str(repo), "error": str(exc)}
