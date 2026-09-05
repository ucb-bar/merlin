"""Assemble the deny-by-default bwrap command + PROVE its isolation by replaying the mount table.

``base_argv`` is the deny-by-default prefix: system dirs RO, all of ``/scratch*`` tmpfs-hidden, only the
arm bundle's ``allowed`` paths bound back, its ``denied`` sub-paths re-masked (deny-wins), the workspace
writable. On top go the claude runtime binds, the toolchain binds, and finally the derived answer masks.

``apply_answer_masks`` masks EVERY derived answer surface that a legit bind would otherwise re-expose
(goldens/model weights/hidden live under the bound ``merlin/contract`` tree; memory under the bound
``~/.claude``). It adds a mask ONLY for a surface that is currently exposed — so a surface already hidden
by deny-by-default needs no overlay (and we never /dev/null-overlay a path whose parent tmpfs would make
the mount fail).

``coverage_gap`` is the hermetic guard: it replays the ordered mount table and returns the answer
surfaces still reachable. Empty == the sandbox masks the full derived answer set. This runs WITHOUT
launching bwrap, so it is the CI-safe isolation proof.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

from merlin.common.paths import repo_root
from merlin.targetgen.sandbox.answer_surfaces import AnswerSurface, answer_surfaces
from merlin.targetgen.target_experiment import TargetExperiment

_EXPOSE_OPS = ("--ro-bind", "--bind", "--dev-bind", "--ro-bind-try", "--bind-try")
_HIDE_DEST_OPS = ("--tmpfs",)            # single-arg-dest hide ops
_DEVNULL = "/dev/null"
PINNED_SUBMISSION_READ_ONLY_ENV = "MERLIN_PINNED_SUBMISSION_READ_ONLY"
_SNAPSHOT_DIR = "bundle_inputs"
_SNAPSHOT_COMPLETE = "snapshot.json"


def path_kind(p: Path) -> str:
    """``dir`` / ``file`` / ``missing`` -- permission-safe.

    A chmod-000 lock makes ``stat()`` raise; treat that as a present dir so a locked answer surface
    is still masked rather than crashing the binder.
    """
    try:
        if p.is_dir():
            return "dir"
        if p.exists():
            return "file"
        return "missing"
    except PermissionError:
        return "dir"


def resolve_grant(rel: str, repo: Path | None = None) -> Path:
    """Where a bundle's ``allowed``/``denied`` path actually lives on disk.

    Bundle-convention grants are repo-root-relative, with a documented shorthand: an
    ``experiments/...`` grant "resolves under merlin/". Prefer ``<repo>/<rel>``, fall back to
    ``<repo>/merlin/<rel>``, so a path that only exists under ``merlin/`` is found rather than
    silently dropped. Returns the un-prefixed candidate when neither exists, so the caller can
    report the path as the manifest declared it.

    EVERY consumer of a manifest must resolve through this. It is module-level for that reason: it
    was previously private to ``base_argv``, so the sandbox bound the ``merlin/``-shorthand paths
    while the lock writer -- which resolved ``<repo>/<rel>`` only, and skipped what it could not
    find -- hashed none of them. 17 grants across all five targets (every target's task, ISA
    headers, hwbringup contracts and self-check script) were mounted into the arm but absent from
    the lock meant to pin the arm's exact input bytes.
    """
    repo = repo or repo_root()
    p = repo / rel
    if path_kind(p) != "missing":
        return p
    q = repo / "merlin" / rel
    return q if path_kind(q) != "missing" else p


def bundle_snapshot_root(ws: Path) -> Path:
    """Host-only root holding the immutable bytes served to one agent run.

    It is a sibling of the writable workspace, not a child: bwrap binds only
    ``ws`` itself, so the agent cannot reach this storage path directly.  The
    declared grants below are the only views into it, and those views are
    read-only.
    """
    return ws.parent / _SNAPSHOT_DIR


def _snapshot_path(root: Path, source: Path, repo: Path) -> Path:
    """Stable snapshot location for an absolute grant source."""
    source = source.absolute()
    repo = repo.absolute()
    try:
        return root / "repo" / source.relative_to(repo)
    except ValueError:
        # External grants retain their absolute spelling below a private
        # namespace.  Path.parts[0] is the filesystem anchor and is omitted.
        return root / "external" / Path(*source.parts[1:])


def _grant_sources(bundle: dict, repo: Path) -> list[tuple[str, Path]]:
    return [(str(entry["path"]), resolve_grant(str(entry["path"]), repo))
            for entry in bundle.get("allowed", [])]


def _snapshot_content(root: Path) -> tuple[str, int, int]:
    """Digest, file count and byte count for the frozen payload (not its marker)."""
    rows: list[tuple[str, str, int]] = []
    total = 0
    marker = root / _SNAPSHOT_COMPLETE
    for path in sorted(p for p in root.rglob("*")
                       if p.is_file() and p != marker):
        digest = hashlib.sha256()
        size = 0
        with path.open("rb") as stream:
            while chunk := stream.read(8 * 1024 * 1024):
                digest.update(chunk)
                size += len(chunk)
        rows.append((path.relative_to(root).as_posix(), digest.hexdigest(), size))
        total += size
    aggregate = hashlib.sha256()
    for rel, digest, size in rows:
        aggregate.update(rel.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(str(size).encode("ascii"))
        aggregate.update(b"\0")
        aggregate.update(digest.encode("ascii"))
        aggregate.update(b"\n")
    return aggregate.hexdigest(), len(rows), total


def _read_snapshot_manifest(ws: Path) -> dict:
    root = bundle_snapshot_root(ws)
    marker = root / _SNAPSHOT_COMPLETE
    if root.is_symlink() or not marker.is_file():
        raise RuntimeError(f"bundle input snapshot is incomplete or unsafe at {root}")
    try:
        return json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"bundle input snapshot marker is unreadable at {marker}: {exc}") from exc


def _snapshot_grants(ws: Path, bundle: dict, repo: Path) -> tuple[dict, list[tuple[str, Path, Path]]]:
    """Pinned ``(declared path, destination, snapshot source)`` records.

    Destinations come from the snapshot marker, never from re-resolving the
    current worktree.  A deleted or newly-created fallback path therefore cannot
    redirect a resumed experiment's mount table.
    """
    manifest = _read_snapshot_manifest(ws)
    expected_allowed = [str(entry["path"]) for entry in bundle.get("allowed", [])]
    if manifest.get("version") != 2 or manifest.get("repo") != str(repo):
        raise RuntimeError("bundle input snapshot metadata does not match this run")
    if manifest.get("allowed") != expected_allowed:
        raise RuntimeError("bundle input snapshot grant set does not match this run")
    records = manifest.get("grants")
    if not isinstance(records, list) or len(records) != len(expected_allowed):
        raise RuntimeError("bundle input snapshot grant records are incomplete")
    out: list[tuple[str, Path, Path]] = []
    root = bundle_snapshot_root(ws)
    for expected, record in zip(expected_allowed, records, strict=True):
        if not isinstance(record, dict) or record.get("path") != expected:
            raise RuntimeError("bundle input snapshot grant ordering does not match this run")
        destination = Path(str(record.get("destination", "")))
        snapshot_rel = Path(str(record.get("snapshot", "")))
        declared = Path(expected)
        valid_destinations = {
            (repo / declared).absolute(),
            (repo / "merlin" / declared).absolute(),
        }
        if not destination.is_absolute() or destination not in valid_destinations:
            raise RuntimeError(f"bundle input snapshot destination is invalid for {expected!r}")
        if snapshot_rel.is_absolute() or ".." in snapshot_rel.parts:
            raise RuntimeError(f"bundle input snapshot path escapes its root for {expected!r}")
        source = root / snapshot_rel
        try:
            resolved_root = root.resolve(strict=True)
            resolved_source = source.resolve(strict=True)
        except OSError as exc:
            raise RuntimeError(f"bundle input snapshot grant is incomplete for {expected!r}") from exc
        if resolved_source != resolved_root and resolved_root not in resolved_source.parents:
            raise RuntimeError(f"bundle input snapshot path escapes its root for {expected!r}")
        relative_source = source.relative_to(root)
        if any((root / Path(*relative_source.parts[:i])).is_symlink()
               for i in range(1, len(relative_source.parts) + 1)):
            raise RuntimeError(f"bundle input snapshot contains a symlink for {expected!r}")
        if path_kind(source) == "missing":
            raise RuntimeError(f"bundle input snapshot grant is incomplete for {expected!r}")
        out.append((expected, destination, source))
    return manifest, out


def verify_bundle_snapshot(ws: Path, bundle: dict, *, repo: Path | None = None) -> dict:
    """Verify and return a complete snapshot; used by cross-window resume."""
    repo = (repo or repo_root()).absolute()
    root = bundle_snapshot_root(ws)
    manifest, _ = _snapshot_grants(ws, bundle, repo)
    digest, n_files, n_bytes = _snapshot_content(root)
    observed = {"content_sha256": digest, "n_files": n_files, "n_bytes": n_bytes}
    expected = {key: manifest.get(key) for key in observed}
    if observed != expected:
        raise RuntimeError(
            f"bundle input snapshot content verification failed at {root}: "
            f"expected {expected}, observed {observed}")
    return manifest


def snapshot_record(ws: Path) -> dict:
    """Small provenance block copied into the run's environment record."""
    manifest = _read_snapshot_manifest(ws)
    return {"path": str(bundle_snapshot_root(ws)),
            "content_sha256": manifest.get("content_sha256"),
            "n_files": manifest.get("n_files"),
            "n_bytes": manifest.get("n_bytes"),
            "version": manifest.get("version")}


def _make_snapshot_writable(root: Path) -> None:
    if not root.exists():
        return
    paths = list(root.rglob("*"))
    symlinks = [path for path in paths if path.is_symlink()]
    if symlinks:
        raise RuntimeError(
            f"refusing to chmod tampered bundle snapshot containing symlink: {symlinks[0]}")
    for path in sorted(paths, key=lambda p: len(p.parts)):
        path.chmod(0o700 if path.is_dir() else 0o600)
    root.chmod(0o700)


def remove_bundle_snapshot(ws: Path) -> None:
    """Remove a generated snapshot after restoring owner write permission."""
    root = bundle_snapshot_root(ws)
    if root.is_symlink():
        raise RuntimeError(f"refusing to remove symlinked bundle snapshot: {root}")
    if root.exists():
        _make_snapshot_writable(root)
        shutil.rmtree(root)


def materialize_bundle_inputs(ws: Path, bundle: dict, *, repo: Path | None = None) -> dict:
    """Copy every declared allowed input into a run-private immutable snapshot.

    The former bwrap path symlinked the workspace and RO-bound grants directly
    from the working tree.  Read-only protected the source from the agent, but
    it did not protect a long-running experiment from an operator editing that
    same source in another session.  This snapshot fixes the experiment's input
    bytes at setup time.  Nested grants are copied only once through their
    shallowest declared ancestor.

    A missing grant fails closed.  Silently omitting it would make the executed
    treatment smaller than the bundle and its lock claim.
    """
    repo = (repo or repo_root()).absolute()
    root = bundle_snapshot_root(ws)
    pending = root.with_name(root.name + ".pending")
    if root.exists() or root.is_symlink():
        return verify_bundle_snapshot(ws, bundle, repo=repo)
    if pending.exists() or pending.is_symlink():
        shutil.rmtree(pending) if pending.is_dir() else pending.unlink()

    grants = _grant_sources(bundle, repo)
    missing = [rel for rel, source in grants if path_kind(source) == "missing"]
    if missing:
        raise FileNotFoundError(
            "bundle declares unresolvable allowed grant(s): " + ", ".join(sorted(missing)))

    # Copy the union, not every overlapping spelling.  For example, a broad
    # contract grant plus its isa/layers children must produce one snapshot.
    unique_sources = sorted({source.absolute() for _, source in grants},
                            key=lambda p: (len(p.parts), str(p)))
    roots: list[Path] = []
    for source in unique_sources:
        if any(parent == source or parent in source.parents for parent in roots):
            continue
        roots.append(source)

    pending.mkdir(parents=True)
    try:
        for source in roots:
            dst = _snapshot_path(pending, source, repo)
            dst.parent.mkdir(parents=True, exist_ok=True)
            if source.is_dir():
                # Dereference symlinks so an absolute link into an external RTL
                # checkout cannot remain a live escape from the snapshot.
                shutil.copytree(source, dst, symlinks=False)
            else:
                shutil.copy2(source, dst, follow_symlinks=True)
        digest, n_files, n_bytes = _snapshot_content(pending)
        grant_records = []
        for rel, source in grants:
            grant_records.append({
                "path": rel,
                "destination": str(source.absolute()),
                "snapshot": _snapshot_path(pending, source, repo).relative_to(pending).as_posix(),
            })
        manifest = {
            "version": 2,
            "repo": str(repo),
            "allowed": [rel for rel, _ in grants],
            "grants": grant_records,
            "copied_roots": [str(p) for p in roots],
            "content_sha256": digest,
            "n_files": n_files,
            "n_bytes": n_bytes,
        }
        (pending / _SNAPSHOT_COMPLETE).write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        for path in sorted(pending.rglob("*"), key=lambda p: len(p.parts), reverse=True):
            # Host-immutable means clear write bits, not executable bits.  A
            # bundle may grant compilers and scripts that must remain runnable.
            path.chmod(path.stat().st_mode & ~0o222)
        pending.chmod(pending.stat().st_mode & ~0o222)
        pending.rename(root)
        return manifest
    except BaseException:
        if pending.exists():
            _make_snapshot_writable(pending)
            shutil.rmtree(pending)
        raise


def _bundle_mount_args(ws: Path, bundle: dict, repo: Path,
                       *, _policy_test_live_inputs: bool = False) -> list[str]:
    """Frozen allow mounts followed by deny-wins overlays."""
    out: list[str] = []
    if not bundle.get("allowed"):
        grants = []
    elif _policy_test_live_inputs:
        grants = [(str(entry["path"]), resolve_grant(entry["path"], repo),
                   resolve_grant(entry["path"], repo))
                  for entry in bundle.get("allowed", [])]
    else:
        _, grants = _snapshot_grants(ws, bundle, repo)
    for _, destination, frozen in grants:
        if path_kind(destination) != "missing" or not _policy_test_live_inputs:
            out += ["--ro-bind", str(frozen), str(destination)]
    for denied in bundle.get("denied", []):
        p = resolve_grant(denied["path"], repo)
        kind = path_kind(p)
        if kind == "dir":
            out += ["--tmpfs", str(p)]
        elif kind == "file":
            out += ["--ro-bind", _DEVNULL, str(p)]
    return out


def reapply_bundle_snapshot(argv: list[str], ws: Path, bundle: dict,
                            *, repo: Path | None = None) -> list[str]:
    """Reassert frozen grants after later trusted-runtime/toolchain binds.

    A universal toolchain path may also be a declared arm input.  Since bwrap is
    ordered, appending this layer ensures that such an overlap still serves the
    experiment snapshot.  Denied subpaths are appended in the same layer and
    therefore continue to win.
    """
    repo = (repo or repo_root()).absolute()
    return [*argv, *_bundle_mount_args(ws, bundle, repo)]


def base_argv(ws: Path, bundle: dict, *, repo: Path | None = None,
              _policy_test_live_inputs: bool = False) -> list[str]:
    """Deny-by-default bwrap argv prefix: system RO, /scratch* tmpfs-hidden, ONLY the bundle's allowed
    paths bound RO, denied sub-paths re-masked, workspace writable+last. Target-agnostic — the ``bundle``
    (or an empty ``{}``) is the only input beyond the workspace."""
    repo = repo or repo_root()
    parts = ["bwrap", "--die-with-parent", "--unshare-pid",
             "--ro-bind", "/usr", "/usr", "--ro-bind", "/bin", "/bin", "--ro-bind", "/lib", "/lib",
             "--ro-bind", "/lib64", "/lib64", "--ro-bind", "/etc", "/etc",
             # DNS: /etc/resolv.conf is a symlink into the systemd-resolved runtime dir. Binding /etc alone
             # leaves that symlink dangling inside the sandbox, so every name lookup fails and the agent's
             # `claude` session hangs on an unreachable API. Bind the resolver dir so the symlink resolves.
             # --ro-bind-try tolerates non-systemd hosts (where resolv.conf is a real file under /etc).
             "--ro-bind-try", "/run/systemd/resolve", "/run/systemd/resolve",
             "--tmpfs", "/scratch", "--tmpfs", "/scratch2", "--tmpfs", "/tmp",
             "--proc", "/proc", "--dev", "/dev",
             # a writable XDG runtime dir under the tmpfs /tmp — the Bun-based `claude` opens a socket there.
             "--dir", "/tmp/.xdg", "--setenv", "XDG_RUNTIME_DIR", "/tmp/.xdg",
             "--chdir", str(ws)]
    # Drop Claude-Code nesting markers inherited from a parent agent session so the sandboxed `claude`
    # starts a clean top-level session. A leaked CLAUDE_CODE_MESSAGING_SOCKET / CLAUDECODE makes it wait on
    # a parent IPC socket that is not inside the box and hang. Auth vars (ANTHROPIC_API_KEY,
    # CLAUDE_CODE_USE_BEDROCK) are intentionally NOT cleared — the launch may need them.
    for _v in ("CLAUDECODE", "CLAUDE_CODE_ENTRYPOINT", "CLAUDE_CODE_SSE_PORT", "CLAUDE_CODE_MESSAGING_SOCKET",
               "CLAUDE_CODE_CHILD_SESSION", "CLAUDE_CODE_SESSION_ID", "CLAUDE_PID", "CLAUDE_EFFORT", "AI_AGENT"):
        parts += ["--unsetenv", _v]
    home_claude = os.path.expanduser("~/.claude")
    if Path(home_claude).exists():
        parts += ["--bind", home_claude, home_claude]
        # ⚠ ANSWER-SURFACE LEAK GUARD: ~/.claude is bound whole so the sandboxed `claude` CLI finds its
        # credentials/settings, but ~/.claude/projects/<slug>/ holds the EXPERIMENTER's session
        # transcripts (the full conversation — oracle facts, ISA details, goldens discussed) AND the
        # persistent agent MEMORY. Those are answer surfaces (answer_surfaces() lists the memory dir), so
        # tmpfs-mask the whole projects/ tree LAST: the agent gets an empty, writable projects dir for its
        # own ephemeral session (the run captures its transcript from stdout, not from here) and can never
        # read the experimenter's history/notes. Target-agnostic — the experimenter memory is the same
        # regardless of target.
        projects = os.path.join(home_claude, "projects")
        if Path(projects).exists():
            parts += ["--tmpfs", projects]

    parts += _bundle_mount_args(ws, bundle, repo,
                                _policy_test_live_inputs=_policy_test_live_inputs)
    # Bind the writable workspace LAST so no mask clobbers it.
    parts += ["--bind", str(ws), str(ws)]
    return parts


def claude_runtime_binds() -> list[str]:
    """RO-bind the ``claude`` CLI runtime (launcher in ~/.local/bin, native binary in ~/.local/share/
    claude, nvm, ~/.config) + make ~/.claude.json writable. Without these the agent launch fails
    'claude: command not found'. None of these are an answer surface."""
    binds: list[str] = []
    home = Path(os.path.expanduser("~"))
    for p in (home / ".local" / "bin", home / ".local" / "share" / "claude",
              home / ".nvm", home / ".config"):
        if p.exists():
            binds += ["--ro-bind", str(p), str(p)]
    cj = home / ".claude.json"
    if cj.exists():
        binds += ["--bind", str(cj), str(cj)]
    # Bedrock provider (experiments-only): the sandboxed `claude` needs AWS creds to reach Bedrock.
    # Bind ~/.aws RO ONLY when CLAUDE_CODE_USE_BEDROCK is set in this env, so a subscription run never
    # exposes AWS creds. Not an answer surface. (Env-var creds ride os.environ, which bwrap inherits.)
    if os.environ.get("CLAUDE_CODE_USE_BEDROCK") == "1":
        aws = home / ".aws"
        if aws.exists():
            binds += ["--ro-bind", str(aws), str(aws)]
    return binds


# --------------------------------------------------------------------------- mount-table replay
def _mounts(argv: list[str]) -> list[tuple[str, str | None, str]]:
    """Parse argv into ordered ``(state, source, destination)`` mount operations.

    Retaining the source is essential for immutable bundle snapshots: a live destination can gain a
    file after setup while the frozen directory bound over it does not contain that file.  Visibility
    is determined by the mounted source bytes, never by the current host destination tree.
    """
    ops: list[tuple[str, str | None, str]] = []
    i = 0
    n = len(argv)
    while i < n:
        a = argv[i]
        if a in _EXPOSE_OPS and i + 2 < n:
            src, dest = argv[i + 1], argv[i + 2]
            ops.append(("hide" if src == _DEVNULL else "expose", src, dest))
            i += 3
            continue
        if a in _HIDE_DEST_OPS and i + 1 < n:
            ops.append(("hide", None, argv[i + 1]))
            i += 2
            continue
        if a in ("--dev", "--proc") and i + 1 < n:
            # devtmpfs/procfs hide host content below the destination.  They intentionally have no
            # ordinary host source that answer-surface replay may inspect.
            ops.append(("hide", None, argv[i + 1]))
            i += 2
            continue
        i += 1
    return ops


def _is_under(path: Path, base: str) -> bool:
    b = Path(base)
    return path == b or b in path.parents


def is_exposed(argv: list[str], path: Path) -> bool:
    """True iff ``path`` is reachable (readable host content) inside the sandbox described by ``argv``.
    Replays the ordered mount table: the controlling mount is the one whose dest is ``path`` or an
    ancestor of it, most-specific (longest dest) wins, ties broken by LATEST op (bwrap applies in order).
    Exposed iff that controlling mount is an 'expose' AND the mapped host source still contains the path."""
    ops = _mounts(argv)
    best = None  # (dest_len, index, state, source, dest)
    for idx, (state, source, dest) in enumerate(ops):
        if _is_under(path, dest):
            key = (len(dest), idx)
            if best is None or key >= (best[0], best[1]):
                best = (len(dest), idx, state, source, dest)
    if best is None:
        return False              # no mount covers it -> not present
    _, _, state, source, dest = best
    if state == "hide":
        return False
    # Map the requested destination back into the controlling bind's source.  Most host/runtime mounts
    # use source==destination; frozen bundle grants deliberately do not.
    assert source is not None
    relative = path.relative_to(Path(dest))
    mapped = Path(source) / relative
    try:
        return mapped.exists()
    except PermissionError:
        return True               # a locked-but-present surface is still exposed content-wise
    except OSError:
        return False


def coverage_gap(argv: list[str], surfaces: list[AnswerSurface]) -> list[AnswerSurface]:
    """The answer surfaces STILL reachable under ``argv`` — the drift/cheat guard. Empty == full mask."""
    return [s for s in surfaces if is_exposed(argv, s.path)]


def apply_answer_masks(argv: list[str], surfaces: list[AnswerSurface]) -> list[str]:
    """Append masks for every answer surface a bind would otherwise re-expose. A surface already hidden
    by deny-by-default is skipped (no redundant overlay, and no mount whose parent tmpfs would fail).
    File surfaces are /dev/null-overlaid; dir surfaces are tmpfs'd. Masks go LAST so they win."""
    out = list(argv)
    for s in surfaces:
        if not is_exposed(out, s.path):
            continue
        if s.kind == "file":
            out += ["--ro-bind", _DEVNULL, str(s.path)]
        else:
            out += ["--tmpfs", str(s.path)]
    return out


def apply_pinned_submission_guard(argv: list[str], ws: Path) -> list[str]:
    """Make ``workspace/submission`` genuinely immutable for a certified resume.

    File mode bits are not an isolation boundary when the agent owns the files: it can simply run
    ``chmod -R u+w submission`` and continue editing.  A same-path read-only bind is enforced by the
    mount namespace and cannot be undone from inside this unprivileged sandbox.  Append it LAST so the
    earlier writable workspace bind remains useful for QA/tool outputs while this subtree is protected.
    """
    if os.environ.get(PINNED_SUBMISSION_READ_ONLY_ENV, "").strip() != "1":
        return list(argv)
    submission = Path(ws).absolute() / "submission"
    if submission.is_symlink() or not submission.is_dir():
        raise RuntimeError(
            f"pinned submission guard requires a plain submission directory: {submission}")
    return [*argv, "--ro-bind", str(submission), str(submission)]


# --------------------------------------------------------------------------- full assembly
def full_argv(te: TargetExperiment, ws: Path, bundle: dict | None = None,
              *, _policy_test_live_inputs: bool = False) -> list[str]:
    """The complete isolation argv for one target+arm: deny-by-default base + claude runtime + toolchain
    binds + derived answer masks. ``bundle`` may be ``{}``/None for a descriptor-only (bundle-less)
    target — the answer masks are then driven purely by the descriptor."""
    from merlin.targetgen.sandbox import toolchain as TC   # local: avoid a heavy import at module load
    bundle = bundle or {}
    argv = (base_argv(ws, bundle, _policy_test_live_inputs=_policy_test_live_inputs)
            + claude_runtime_binds() + TC.toolchain_binds(te))
    if bundle:
        if _policy_test_live_inputs:
            argv += _bundle_mount_args(ws, bundle, repo_root(), _policy_test_live_inputs=True)
        else:
            argv = reapply_bundle_snapshot(argv, ws, bundle)
    argv = apply_answer_masks(argv, answer_surfaces(te))
    return apply_pinned_submission_guard(argv, ws)


def wrap(te: TargetExperiment, ws: Path, inner: str, bundle: dict | None = None,
         *, _policy_test_live_inputs: bool = False) -> str:
    """A ready-to-run ``bash -c`` string: full argv + the sandbox env exports + the inner command."""
    from merlin.targetgen.sandbox import toolchain as TC
    argv = full_argv(te, ws, bundle, _policy_test_live_inputs=_policy_test_live_inputs)
    return " ".join(argv) + f" bash -c '{TC.sandbox_env(te, ws)} {inner}'"
