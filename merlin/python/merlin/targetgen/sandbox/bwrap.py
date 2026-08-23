"""Assemble the deny-by-default bwrap command + PROVE its isolation by replaying the mount table.

``base_argv`` is the deny-by-default prefix: system dirs RO, all of ``/scratch*`` tmpfs-hidden, only the
arm bundle's ``allowed`` paths bound back, its ``denied`` sub-paths re-masked (deny-wins), the workspace
writable. On top go the claude runtime binds, the toolchain binds, and finally the derived answer masks.

``apply_answer_masks`` masks EVERY derived answer surface that a legit bind would otherwise re-expose
(goldens/hidden live under the bound ``merlin/contract`` tree; memory under the bound ``~/.claude``). It
adds a mask ONLY for a surface that is currently exposed — so a surface already hidden by deny-by-default
needs no overlay (and we never /dev/null-overlay a path whose parent tmpfs would make the mount fail).

``coverage_gap`` is the hermetic guard: it replays the ordered mount table and returns the answer
surfaces still reachable. Empty == the sandbox masks the full derived answer set. This runs WITHOUT
launching bwrap, so it is the CI-safe isolation proof.
"""
from __future__ import annotations

import os
from pathlib import Path

from merlin.common.paths import repo_root
from merlin.targetgen.sandbox.answer_surfaces import AnswerSurface, answer_surfaces
from merlin.targetgen.target_experiment import TargetExperiment

_EXPOSE_OPS = ("--ro-bind", "--bind", "--dev-bind", "--ro-bind-try", "--bind-try")
_HIDE_DEST_OPS = ("--tmpfs",)            # single-arg-dest hide ops
_DEVNULL = "/dev/null"


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


def base_argv(ws: Path, bundle: dict, *, repo: Path | None = None) -> list[str]:
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

    def _kind(p: Path) -> str:
        return path_kind(p)

    def _resolve_grant(rel: str) -> Path:
        return resolve_grant(rel, repo)

    for entry in bundle.get("allowed", []):
        p = _resolve_grant(entry["path"])
        if _kind(p) != "missing":
            parts += ["--ro-bind", str(p), str(p)]
    # Deny wins: tmpfs-mask every denied DIR after the allowed binds, so a broad allow (e.g. all of
    # merlin/contract/) cannot expose a denied sub-path (e.g. capsules/hidden/).
    for d in bundle.get("denied", []):
        p = _resolve_grant(d["path"])
        k = _kind(p)
        if k == "dir":
            parts += ["--tmpfs", str(p)]
        elif k == "file":
            # a denied FILE under an allowed dir (e.g. an oracle-callable route the arm withholds) must be
            # /dev/null-overlaid too — tmpfs only masks dirs, so a denied file would otherwise stay readable.
            parts += ["--ro-bind", _DEVNULL, str(p)]
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
def _mounts(argv: list[str]) -> list[tuple[str, str]]:
    """Parse the argv into ordered (state, dest) mount ops, where state is 'expose' or 'hide'. Only the
    ops that affect path visibility are kept; everything else (flags, --chdir, --unsetenv…) is ignored."""
    ops: list[tuple[str, str]] = []
    i = 0
    n = len(argv)
    while i < n:
        a = argv[i]
        if a in _EXPOSE_OPS and i + 2 < n:
            src, dest = argv[i + 1], argv[i + 2]
            ops.append(("hide" if src == _DEVNULL else "expose", dest))
            i += 3
            continue
        if a in _HIDE_DEST_OPS and i + 1 < n:
            ops.append(("hide", argv[i + 1]))
            i += 2
            continue
        if a in ("--dev", "--proc") and i + 1 < n:
            ops.append(("hide", argv[i + 1]))    # devtmpfs/procfs — hides host content under dest
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
    best = None  # (dest_len, index, state, dest)
    for idx, (state, dest) in enumerate(ops):
        if _is_under(path, dest):
            key = (len(dest), idx)
            if best is None or key >= (best[0], best[1]):
                best = (len(dest), idx, state, dest)
    if best is None:
        return False              # no mount covers it -> not present
    _, _, state, dest = best
    if state == "hide":
        return False
    # expose: dest is bound to the identical host path in this harness (src==dest), so the sub-path maps
    # back to itself; it is exposed iff it still exists on the host.
    try:
        return path.exists()
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


# --------------------------------------------------------------------------- full assembly
def full_argv(te: TargetExperiment, ws: Path, bundle: dict | None = None) -> list[str]:
    """The complete isolation argv for one target+arm: deny-by-default base + claude runtime + toolchain
    binds + derived answer masks. ``bundle`` may be ``{}``/None for a descriptor-only (bundle-less)
    target — the answer masks are then driven purely by the descriptor."""
    from merlin.targetgen.sandbox import toolchain as TC   # local: avoid a heavy import at module load
    bundle = bundle or {}
    argv = base_argv(ws, bundle) + claude_runtime_binds() + TC.toolchain_binds(te)
    return apply_answer_masks(argv, answer_surfaces(te))


def wrap(te: TargetExperiment, ws: Path, inner: str, bundle: dict | None = None) -> str:
    """A ready-to-run ``bash -c`` string: full argv + the sandbox env exports + the inner command."""
    from merlin.targetgen.sandbox import toolchain as TC
    argv = full_argv(te, ws, bundle)
    return " ".join(argv) + f" bash -c '{TC.sandbox_env(te, ws)} {inner}'"
