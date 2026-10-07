"""Harness-owned git history for a run's out-of-tree compiler package.

A phase-1 run's ``oot/`` records one commit per graded round and tags the submission ``frozen``; a
phase-2 run's ``oot/`` starts from that tag and records one commit per candidate, tagging
``measured/<n>`` and ``best``. Iteration records then carry a commit sha instead of a copy of the
package, and a champion is exported from a tag rather than from whatever a directory holds today.

Three properties make that history evidence rather than decoration:

* **Only the harness writes it.** Commits are built with plumbing (private index, ``hash-object``,
  ``write-tree``, ``commit-tree``, compare-and-swap ``update-ref``), so no hook, template, signing key or
  user/system git configuration takes part, and the repo is refused when it overlaps a sandbox the
  authoring agent can write. There is no remote and nothing is ever pushed.
* **A commit IS the measured package.** The committed files are exactly the set
  :func:`merlin.common.tree_hash.hash_tree` identifies (it skips ``build/``, ``__pycache__/`` and
  ``.git/``), so :func:`tree_digest` recomputed from the commit's blobs equals the digest the
  measurement store names the package by. A symlink or special file is refused rather than guessed at.
* **The history is reproducible.** Author and committer are pinned and both dates come from the run
  clock the caller passes, so the same package bytes, label, time and parent always give the same sha.

A lineage older than this module has no such history. :func:`reconstruct` rebuilds one from the stored
package bytes, digest-checked hop by hop, and labels the repository and every commit
``reconstructed: true`` so it is never mistaken for a history a harness recorded live.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import stat
import subprocess
import tarfile
import tempfile
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .jsonio import canonical_json
from .tree_hash import _SKIP as _DIGEST_SKIP
from .tree_hash import hash_tree

BRANCH = "main"
FROZEN_TAG = "frozen"
BEST_TAG = "best"
MEASURED_PREFIX = "measured/"
#: The only tag that may be re-pointed. Every other tag names one historical fact and never moves.
MOVABLE_TAGS = frozenset({BEST_TAG})
_FORMAT = "1"
_ZERO = "0" * 40
_ORIGIN_RECORD = "merlin-origin.json"
_RECONSTRUCTION_RECORD = "merlin-reconstruction.json"
_STAMP = "%Y%m%dT%H%M%SZ"


class OotRepoError(RuntimeError):
    """The OOT repository could not be created, committed, tagged, verified or exported."""


@dataclass(frozen=True)
class Identity:
    name: str
    email: str


#: The one identity every harness commit carries, as author and as committer.
HARNESS_IDENTITY = Identity("Merlin Harness", "harness@merlin.invalid")


@dataclass(frozen=True)
class CommitRecord:
    """What an iteration record stores about its package: a sha, never a copy."""

    commit: str
    tree: str
    parent: str | None
    package_digest: str
    n_files: int
    label: str
    run_id: str | None
    committed_at: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_record(self) -> dict[str, Any]:
        return {
            "commit": self.commit,
            "tree": self.tree,
            "parent": self.parent,
            "package_digest": self.package_digest,
            "n_files": self.n_files,
            "label": self.label,
            "run_id": self.run_id,
            "committed_at": self.committed_at,
            "metadata": dict(self.metadata),
        }


# ---------------------------------------------------------------------------- git invocation


def _env(extra: dict[str, str] | None = None) -> dict[str, str]:
    """A git environment that no ambient configuration or repository variable can steer."""
    env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    env.update(
        {
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_TERMINAL_PROMPT": "0",
            "LC_ALL": "C",
        }
    )
    env.update(extra or {})
    return env


_CONFIG = (
    "-c",
    "core.hooksPath=" + os.devnull,
    "-c",
    "commit.gpgSign=false",
    "-c",
    "tag.gpgSign=false",
    "-c",
    "core.autocrlf=false",
    "-c",
    "core.fileMode=true",
    "-c",
    "core.symlinks=true",
)


def _git(
    repo: Path | None,
    *args: str,
    input: bytes | None = None,
    env: dict[str, str] | None = None,
    check: bool = True,
) -> bytes:
    argv = ["git", *_CONFIG, *(["-C", str(repo)] if repo is not None else []), *args]
    proc = subprocess.run(
        argv,
        input=input,
        capture_output=True,
        env=_env(env),
        timeout=600,
        **({} if input is not None else {"stdin": subprocess.DEVNULL}),
    )
    if check and proc.returncode != 0:
        detail = proc.stderr.decode("utf-8", "replace").strip()
        raise OotRepoError(f"git {' '.join(args[:2])} failed in {repo}: {detail}")
    return proc.stdout


def _text(repo: Path | None, *args: str, **kwargs) -> str:
    return _git(repo, *args, **kwargs).decode("utf-8", "surrogateescape").strip()


# ---------------------------------------------------------------------------- guards


def _absolute(path: Path | str) -> Path:
    return Path(os.path.abspath(Path(path).expanduser()))


def check_outside_sandbox(path: Path | str, sandbox_roots=()) -> None:
    """Refuse a repository that an authoring agent's writable root contains, or that contains one."""
    repo = _absolute(path).resolve()
    for root in sandbox_roots or ():
        writable = _absolute(root).resolve()
        if repo == writable or repo.is_relative_to(writable) or writable.is_relative_to(repo):
            raise OotRepoError(f"OOT repository {repo} overlaps the agent-writable root {writable}")


def _is_repo(path: Path) -> bool:
    git_dir = path / ".git"
    return git_dir.is_dir() and not git_dir.is_symlink() and (git_dir / "HEAD").is_file()


def _require_repo(path: Path | str, sandbox_roots=()) -> Path:
    repo = _absolute(path)
    if not _is_repo(repo):
        raise OotRepoError(f"not an OOT repository: {repo}")
    check_outside_sandbox(repo, sandbox_roots)
    return repo


def _epoch(when) -> int:
    """Seconds since the epoch from the run clock: an aware datetime, a number, or a stamp token."""
    if isinstance(when, bool):
        raise OotRepoError("commit time must come from the run clock, not a flag")
    if isinstance(when, datetime):
        if when.tzinfo is None:
            raise OotRepoError("commit time must be timezone-aware")
        return int(when.timestamp())
    if isinstance(when, int | float):
        return int(when)
    if isinstance(when, str):
        try:
            return int(datetime.strptime(when, _STAMP).replace(tzinfo=UTC).timestamp())
        except ValueError as exc:
            raise OotRepoError(f"commit time {when!r} is not a {_STAMP} token") from exc
    raise OotRepoError(f"unsupported commit time {when!r}")


def _stamp(epoch: int) -> str:
    return datetime.fromtimestamp(epoch, UTC).strftime(_STAMP)


# ---------------------------------------------------------------------------- repository creation


def init(path: Path | str, *, sandbox_roots=()) -> Path:
    """Create an empty harness repository on branch ``main`` (phase 1). ``path`` must be fresh."""
    repo = _absolute(path)
    check_outside_sandbox(repo, sandbox_roots)
    if repo.is_symlink() or (repo.exists() and (not repo.is_dir() or any(repo.iterdir()))):
        raise OotRepoError(f"OOT repository destination is not a fresh directory: {repo}")
    repo.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="oot-template-") as template:
        # An empty template: no sample hooks, no description, nothing a harness did not write.
        _git(None, "init", "-q", "-b", BRANCH, f"--template={template}", str(repo))
    return repo


def init_from(path: Path | str, source: Path | str, *, ref: str = FROZEN_TAG, sandbox_roots=()) -> Path:
    """Start a repository from ONE tag of another (phase 2 from phase 1's ``frozen``).

    Only that tag's history is fetched, the tag is kept under the same name, and no remote is
    configured. The source is recorded in ``.git/merlin-origin.json`` for lineage.
    """
    origin = _require_repo(source)
    tag_ref = f"refs/tags/{ref}"
    _check_ref_name(tag_ref)
    commit = _text(origin, "rev-parse", "--verify", "--quiet", f"{tag_ref}^{{commit}}", check=False)
    if not commit:
        raise OotRepoError(f"source repository {origin} has no tag {ref!r}")
    repo = init(path, sandbox_roots=sandbox_roots)
    _git(repo, "fetch", "--no-tags", "--quiet", str(origin.resolve()), f"{tag_ref}:{tag_ref}")
    if resolve(repo, ref) != commit:
        raise OotRepoError(f"fetched {ref!r} differs from the source repository's")
    _git(repo, "update-ref", f"refs/heads/{BRANCH}", commit, _ZERO)
    _git(repo, "read-tree", "--reset", "-u", "HEAD")
    record = {"source": str(origin.resolve()), "ref": ref, "commit": commit}
    (repo / ".git" / _ORIGIN_RECORD).write_bytes(canonical_json(record, trailing_newline=True))
    return repo


def origin(path: Path | str) -> dict[str, str] | None:
    """The repository a phase-2 history was started from, or None for a phase-1 repository."""
    record = _require_repo(path) / ".git" / _ORIGIN_RECORD
    if not record.is_file():
        return None
    return json.loads(record.read_text(encoding="utf-8"))


def reconstruct(
    path: Path | str,
    hops: list[dict[str, Any]],
    *,
    frozen: int,
    best: int,
    reason: str,
    sandbox_roots=(),
) -> dict[str, Any]:
    """Rebuild a harness history from stored package bytes, for a lineage that never kept one.

    Runs that predate the harness ``oot/`` repository left their packages behind as directories (a
    store entry, a round's ``submission``), not as commits. Each hop is ``{"package": dir, "digest":
    sha256, "label": str, "when": run clock, "run": str | None, "metadata": dict}``, oldest first; it is
    committed only when its bytes still hash to the digest it was measured or graded as, so the
    reconstruction cannot quietly substitute other bytes. ``frozen`` and ``best`` index the hops the
    ``frozen`` and ``best`` tags name.

    Every commit carries ``reconstructed: true`` in its metadata and the repository records the
    reconstruction in ``.git/merlin-reconstruction.json`` (see :func:`reconstruction`): its history is
    a faithful ordering of bytes that existed, not a record of what a harness observed as it happened,
    and a reader must be able to tell the two apart.
    """
    if not isinstance(reason, str) or not reason.strip():
        raise OotRepoError("a reconstruction must say why no harness history exists")
    if not hops:
        raise OotRepoError("a reconstruction needs at least one hop")
    if not (0 <= frozen <= best < len(hops)):
        raise OotRepoError(f"frozen ({frozen}) and best ({best}) must index the hops in lineage order")
    for index, hop in enumerate(hops):
        package = hop.get("package")
        if not package or not Path(package).is_dir():
            raise OotRepoError(f"hop {index} names no package directory: {package!r}")
        observed = package_digest(package)
        if observed != hop.get("digest"):
            raise OotRepoError(
                f"hop {index} ({package}) holds package {observed}, not the recorded {hop.get('digest')}"
            )
    repo = init(path, sandbox_roots=sandbox_roots)
    commits = []
    for hop in hops:
        metadata = {**dict(hop.get("metadata") or {}), "reconstructed": True, "source": str(hop["package"])}
        commits.append(
            commit_candidate(
                repo,
                hop["package"],
                label=hop["label"],
                when=hop["when"],
                run_id=hop.get("run"),
                metadata=metadata,
                sandbox_roots=sandbox_roots,
            )
        )
    tag(repo, FROZEN_TAG, commits[frozen].commit)
    tag(repo, BEST_TAG, commits[best].commit)
    record = {
        "format": _FORMAT,
        "reconstructed": True,
        "reason": reason.strip(),
        "frozen": commits[frozen].commit,
        "best": commits[best].commit,
        "hops": [
            {"label": c.label, "commit": c.commit, "package_digest": c.package_digest, "source": str(h["package"])}
            for c, h in zip(commits, hops, strict=True)
        ],
    }
    (repo / ".git" / _RECONSTRUCTION_RECORD).write_bytes(canonical_json(record, trailing_newline=True))
    return record


def reconstruction(path: Path | str) -> dict[str, Any] | None:
    """The record :func:`reconstruct` left, or None for a history a harness wrote as it happened."""
    record = _require_repo(path) / ".git" / _RECONSTRUCTION_RECORD
    if not record.is_file():
        return None
    return json.loads(record.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------- package identity


def package_digest(package_dir: Path | str) -> str:
    """The digest the measurement store names a package by (``hash_tree``)."""
    digest = hash_tree(Path(package_dir)).get("sha256")
    if not digest:
        raise OotRepoError(f"{package_dir} has no hashable content")
    return digest


def _members(package: Path) -> list[tuple[str, str, Path]]:
    """``(git mode, relative path, file)`` for exactly the files ``hash_tree`` identifies."""
    out = []
    for member in sorted(package.rglob("*")):
        relative = member.relative_to(package)
        if _DIGEST_SKIP & set(relative.parts):
            continue
        mode = member.lstat().st_mode
        if stat.S_ISLNK(mode):
            raise OotRepoError(f"package member is a symlink, which a package digest cannot bind: {relative}")
        if stat.S_ISDIR(mode):
            continue
        if not stat.S_ISREG(mode):
            raise OotRepoError(f"package member is not a regular file: {relative}")
        out.append(("100755" if mode & stat.S_IXUSR else "100644", relative.as_posix(), member))
    return out


def _order(path: str) -> list[str]:
    """``hash_tree``'s order: pathlib compares component lists, which is not git's byte order."""
    return path.split("/")


def tree_digest(path: Path | str, rev: str) -> str:
    """Recompute ``hash_tree``'s digest from a commit's (or tree's) blobs, never from a checkout."""
    repo = _require_repo(path)
    listing = _git(repo, "ls-tree", "-r", "-z", "--full-tree", rev)
    entries = []
    for row in listing.split(b"\0"):
        if not row:
            continue
        meta, _, name = row.partition(b"\t")
        mode, kind, sha = meta.decode("ascii").split()
        relative = name.decode("utf-8", "surrogateescape")
        if _DIGEST_SKIP & set(relative.split("/")):
            continue
        if kind != "blob" or mode not in ("100644", "100755"):
            raise OotRepoError(f"{rev} holds a {kind} ({mode}) at {relative}; a package digest binds only files")
        entries.append((_order(relative), relative, sha))
    entries.sort()
    if not entries:
        raise OotRepoError(f"{rev} holds no files")
    blobs = _git(repo, "cat-file", "--batch", input="".join(f"{sha}\n" for _, _, sha in entries).encode())
    digest = hashlib.sha256()
    offset = 0
    for _, relative, sha in entries:
        end = blobs.index(b"\n", offset)
        header = blobs[offset:end].decode("ascii").split()
        if len(header) != 3 or header[0] != sha or header[1] != "blob":
            raise OotRepoError(f"unexpected object for {relative}: {header}")
        size = int(header[2])
        content = blobs[end + 1 : end + 1 + size]
        offset = end + 1 + size + 1
        digest.update(relative.encode("utf-8", "surrogateescape"))
        digest.update(b"\0")
        digest.update(content)
        digest.update(b"\0")
    return digest.hexdigest()


def verify(path: Path | str, rev: str, expected_digest: str) -> None:
    """Raise unless the commit's tree is exactly the package the store measured."""
    observed = tree_digest(path, rev)
    if observed != expected_digest:
        raise OotRepoError(f"{rev} holds package {observed}, not the measured {expected_digest}")


# ---------------------------------------------------------------------------- commits


def _message(label: str, digest: str, n_files: int, run_id: str | None, stamp: str, metadata: dict) -> str:
    return (
        f"{label}\n\n"
        f"merlin-oot: {_FORMAT}\n"
        f"package-digest: {digest}\n"
        f"files: {n_files}\n"
        f"run: {run_id or '-'}\n"
        f"committed-at: {stamp}\n"
        f"metadata: {canonical_json(metadata).decode('ascii')}\n"
    )


def _head(repo: Path) -> str | None:
    head = _text(repo, "rev-parse", "--verify", "--quiet", f"refs/heads/{BRANCH}^{{commit}}", check=False)
    return head or None


def commit_candidate(
    path: Path | str,
    package_dir: Path | str,
    *,
    label: str,
    when,
    run_id: str | None = None,
    metadata: dict[str, Any] | None = None,
    identity: Identity = HARNESS_IDENTITY,
    sandbox_roots=(),
) -> CommitRecord:
    """Commit the package exactly as graded or measured, on top of ``main``, and return its record.

    One call per graded round (phase 1) or per candidate (phase 2). A package identical to its parent
    still commits, so rounds and commits stay one-to-one. The package is re-hashed after its blobs
    are written and the commit is refused if it changed in between.
    """
    repo = _require_repo(path, sandbox_roots)
    package = _absolute(package_dir)
    check_outside_sandbox(repo, (package,))
    if package.is_symlink() or not package.is_dir():
        raise OotRepoError(f"package is not an ordinary directory: {package}")
    if not isinstance(label, str) or not label.strip() or "\n" in label:
        raise OotRepoError("commit label must be one non-empty line")
    metadata = dict(metadata or {})
    canonical_json(metadata)  # refuse what cannot be recorded before anything is written
    epoch = _epoch(when)
    before = package_digest(package)
    members = _members(package)
    if not members:
        raise OotRepoError(f"package {package} holds no files")
    listing = "".join(f"{file}\n" for _, _, file in members).encode("utf-8", "surrogateescape")
    shas = _text(repo, "hash-object", "-w", "--no-filters", "--stdin-paths", input=listing).split()
    if len(shas) != len(members):
        raise OotRepoError("git did not store every package member")
    with tempfile.TemporaryDirectory(prefix="oot-index-") as scratch:
        index_env = {"GIT_INDEX_FILE": str(Path(scratch) / "index")}
        entries = b"".join(
            f"{mode} {sha}\t{relative}".encode("utf-8", "surrogateescape") + b"\0"
            for (mode, relative, _), sha in zip(members, shas, strict=True)
        )
        _git(repo, "update-index", "--add", "-z", "--index-info", input=entries, env=index_env)
        tree = _text(repo, "write-tree", env=index_env)
    if tree_digest(repo, tree) != before or package_digest(package) != before:
        raise OotRepoError(f"package {package} changed while it was being committed")
    parent = _head(repo)
    stamp = _stamp(epoch)
    message = _message(label.strip(), before, len(members), run_id, stamp, metadata)
    date = f"@{epoch} +0000"
    commit_env = {
        "GIT_AUTHOR_NAME": identity.name,
        "GIT_AUTHOR_EMAIL": identity.email,
        "GIT_AUTHOR_DATE": date,
        "GIT_COMMITTER_NAME": identity.name,
        "GIT_COMMITTER_EMAIL": identity.email,
        "GIT_COMMITTER_DATE": date,
    }
    argv = ["commit-tree", "--no-gpg-sign", tree, *(["-p", parent] if parent else []), "-F", "-"]
    commit = _text(repo, *argv, input=message.encode("utf-8"), env=commit_env)
    # Compare-and-swap: a second writer racing this one fails instead of silently dropping a commit.
    _git(repo, "update-ref", f"refs/heads/{BRANCH}", commit, parent or _ZERO)
    _git(repo, "read-tree", "--reset", "-u", "HEAD")
    return CommitRecord(commit, tree, parent, before, len(members), label.strip(), run_id, stamp, metadata)


def history(path: Path | str, rev: str = "HEAD") -> list[CommitRecord]:
    """Every harness commit reachable from ``rev``, oldest first, parsed back from its message."""
    repo = _require_repo(path)
    raw = _git(repo, "log", "-z", "--reverse", "--format=%H%n%T%n%P%n%B", rev)
    out = []
    for chunk in raw.split(b"\0"):
        text = chunk.decode("utf-8", "surrogateescape").strip("\n")
        if not text:
            continue
        commit, tree, parents, body = (text.split("\n", 3) + ["", "", "", ""])[:4]
        label, _, block = body.partition("\n\n")
        fields = {}
        for line in block.splitlines():
            key, sep, value = line.partition(": ")
            if sep:
                fields[key] = value
        if fields.get("merlin-oot") != _FORMAT:
            raise OotRepoError(f"commit {commit} was not written by the harness")
        out.append(
            CommitRecord(
                commit=commit,
                tree=tree,
                parent=parents.split()[0] if parents.split() else None,
                package_digest=fields["package-digest"],
                n_files=int(fields["files"]),
                label=label,
                run_id=None if fields.get("run") in (None, "-") else fields["run"],
                committed_at=fields["committed-at"],
                metadata=json.loads(fields.get("metadata", "{}")),
            )
        )
    return out


# ---------------------------------------------------------------------------- refs


def _check_ref_name(ref: str) -> None:
    proc = subprocess.run(["git", "check-ref-format", ref], capture_output=True, env=_env(), stdin=subprocess.DEVNULL)
    if proc.returncode != 0 or not ref.startswith("refs/"):
        raise OotRepoError(f"invalid ref name {ref!r}")


def resolve(path: Path | str, rev: str) -> str:
    """The full commit sha ``rev`` names in this repository."""
    repo = _require_repo(path)
    sha = _text(repo, "rev-parse", "--verify", "--quiet", f"{rev}^{{commit}}", check=False)
    if not sha:
        raise OotRepoError(f"{rev!r} names no commit in {repo}")
    return sha


def is_ancestor(path: Path | str, ancestor: str, rev: str) -> bool:
    """Whether ``ancestor`` is ``rev`` or reachable from it (lineage, not a string compare)."""
    repo = _require_repo(path)
    proc = subprocess.run(
        ["git", *_CONFIG, "-C", str(repo), "merge-base", "--is-ancestor", resolve(repo, ancestor), resolve(repo, rev)],
        capture_output=True,
        env=_env(),
        stdin=subprocess.DEVNULL,
    )
    if proc.returncode not in (0, 1):
        raise OotRepoError(f"cannot decide ancestry in {repo}: {proc.stderr.decode('utf-8', 'replace').strip()}")
    return proc.returncode == 0


def tags(path: Path | str) -> dict[str, str]:
    """``{tag name: commit}`` for every tag in the repository."""
    repo = _require_repo(path)
    out = {}
    for line in _text(repo, "for-each-ref", "--format=%(refname:strip=2) %(objectname)", "refs/tags").splitlines():
        name, _, sha = line.rpartition(" ")
        if name:
            out[name] = sha
    return out


def tag(path: Path | str, name: str, rev: str = "HEAD", *, move: bool = False) -> str:
    """Point tag ``name`` at ``rev``. Only ``best`` may move; re-tagging the same commit is a no-op."""
    repo = _require_repo(path)
    ref = f"refs/tags/{name}"
    _check_ref_name(ref)
    if move and name not in MOVABLE_TAGS:
        raise OotRepoError(f"tag {name!r} names one historical fact and cannot move")
    commit = resolve(repo, rev)
    current = _text(repo, "rev-parse", "--verify", "--quiet", ref, check=False) or None
    if current == commit:
        return commit
    if current is not None and not move:
        raise OotRepoError(f"tag {name!r} already names {current}; refusing to re-point it at {commit}")
    _git(repo, "update-ref", ref, commit, current or _ZERO)
    return commit


# ---------------------------------------------------------------------------- export


def export(path: Path | str, rev: str, dest: Path | str) -> Path:
    """Write ``rev``'s tree to a fresh ``dest`` (no ``.git``) and verify it hashes to the commit."""
    repo = _require_repo(path)
    target = _absolute(dest)
    if target.exists() or target.is_symlink():
        raise OotRepoError(f"export destination already exists: {target}")
    commit = resolve(repo, rev)
    archive = _git(repo, "archive", "--format=tar", commit)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".export-", dir=target.parent))
    try:
        with tarfile.open(fileobj=io.BytesIO(archive)) as bundle:
            bundle.extractall(staging, filter="data")
        if hash_tree(staging).get("sha256") != tree_digest(repo, commit):
            raise OotRepoError(f"export of {commit} does not reproduce its package digest")
        staging.rename(target)
    except BaseException:
        if staging.exists():
            _remove_tree(staging)
        raise
    return target


def _remove_tree(root: Path) -> None:
    """Remove a staging tree this module created (never caller-owned paths)."""
    shutil.rmtree(root)
