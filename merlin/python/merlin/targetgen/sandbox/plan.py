"""The backend-neutral MOUNT PLAN — what the sandbox exposes and hides, before any backend vocabulary.

The isolation layer is what ENFORCES the deny surface, and :func:`coverage_gap` is the proof that no
answer surface is reachable. That proof used to read a bwrap argv, which meant a second backend would
need a second parser — two implementations to keep in step, and a guard that silently covered only one
of them. So the plan is the thing now: an ordered list of :class:`Mount` ops, verified HERE, and
rendered to bwrap argv (:mod:`.bwrap`) or docker flags (:mod:`.docker`) at the very end.

Ordering is part of the contract, not an accident. The plan is built deny-by-default — system RO,
``/scratch*`` hidden, only the arm's allowed paths bound back, denied sub-paths re-masked, workspace
writable LAST — and :func:`is_exposed` resolves a path against the plan the way bwrap resolves it: the
controlling mount is the one whose dest is the path or an ancestor, most-specific wins, ties broken by
the LATEST op. Renderers must preserve that resolution or they are not rendering the same sandbox; the
docker renderer's fidelity to it is what `tests/infra/test_sandbox_isolation.py` pins.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from merlin.common.paths import repo_root
from merlin.targetgen.sandbox.answer_surfaces import AnswerSurface
from merlin.targetgen.target_experiment import TargetExperiment

_DEVNULL = "/dev/null"

#: Ops that make host content READABLE at ``dst``; everything else hides it.
_EXPOSE = frozenset({"ro", "rw"})


@dataclass(frozen=True)
class Mount:
    """One visibility op. ``dst`` is where it lands inside the sandbox.

    ``op`` is one of:
      ``ro``      — bind ``src`` read-only at ``dst``
      ``rw``      — bind ``src`` writable at ``dst`` (the workspace, and ``~/.claude.json``)
      ``hide``    — an empty tmpfs over ``dst`` (masks whatever the host has there)
      ``devnull`` — overlay ``/dev/null`` at ``dst`` (masking a FILE, where a tmpfs cannot go)
      ``dev`` / ``proc`` — a fresh devtmpfs/procfs at ``dst``; hides host content underneath

    ``optional`` marks the binds that must not fail when the source is absent (bwrap's ``*-try``).
    """

    op: str
    dst: str
    src: str | None = None
    optional: bool = False

    def exposes(self) -> bool:
        """Does this op make host content readable at ``dst``? ``devnull`` binds a source but hides."""
        return self.op in _EXPOSE and self.src != _DEVNULL


@dataclass(frozen=True)
class MountPlan:
    """An ordered mount table plus the env vars the sandbox must unset."""

    mounts: tuple[Mount, ...] = ()
    unsetenv: tuple[str, ...] = ()
    workdir: str | None = None

    def with_mounts(self, extra) -> MountPlan:
        return MountPlan(mounts=self.mounts + tuple(extra), unsetenv=self.unsetenv, workdir=self.workdir)


# --------------------------------------------------------------------------- mount-table replay

def _is_under(path: Path, base: str) -> bool:
    b = Path(base)
    return path == b or b in path.parents


def is_exposed(plan: MountPlan, path: Path) -> bool:
    """True iff ``path`` is reachable (readable host content) inside the sandbox ``plan`` describes.

    Replays the ordered table: the controlling mount is the one whose dest is ``path`` or an ancestor of
    it, most-specific (longest dest) wins, ties broken by the LATEST op. Exposed iff that mount exposes
    AND the mapped host source still contains the path — every bind in this harness is ``src == dst``,
    so the sub-path maps back to itself.
    """
    best = None  # (dest_len, index, mount)
    for idx, m in enumerate(plan.mounts):
        if _is_under(path, m.dst):
            key = (len(m.dst), idx)
            if best is None or key >= (best[0], best[1]):
                best = (len(m.dst), idx, m)
    if best is None:
        return False                  # no mount covers it -> not present in the sandbox at all
    mount = best[2]
    if not mount.exposes():
        return False
    try:
        return path.exists()
    except PermissionError:
        return True                   # locked-but-present is still exposed content-wise
    except OSError:
        return False


def coverage_gap(plan: MountPlan, surfaces: list[AnswerSurface]) -> list[AnswerSurface]:
    """The answer surfaces STILL reachable under ``plan`` — the drift/cheat guard. Empty == full mask."""
    return [s for s in surfaces if is_exposed(plan, s.path)]


def apply_answer_masks(plan: MountPlan, surfaces: list[AnswerSurface]) -> MountPlan:
    """Append a mask for every answer surface a bind would otherwise re-expose.

    A surface already hidden by deny-by-default is skipped — no redundant overlay, and no mount whose
    parent tmpfs would fail. Files are ``/dev/null``-overlaid (a tmpfs cannot mask a file); dirs get a
    tmpfs. Masks go LAST so they win the most-recent tie-break in :func:`is_exposed`.
    """
    out = plan
    for s in surfaces:
        if not is_exposed(out, s.path):
            continue
        if s.kind == "file":
            out = out.with_mounts([Mount("devnull", str(s.path), src=_DEVNULL)])
        else:
            out = out.with_mounts([Mount("hide", str(s.path))])
    return out


# --------------------------------------------------------------------------- plan assembly

def base_plan(ws: Path, bundle: dict, *, repo: Path | None = None) -> MountPlan:
    """Deny-by-default: system RO, ``/scratch*`` hidden, ONLY the bundle's allowed paths bound RO, denied
    sub-paths re-masked, workspace writable + LAST. Target-agnostic — the ``bundle`` (or ``{}``) is the
    only input beyond the workspace."""
    repo = repo or repo_root()
    mounts: list[Mount] = [
        Mount("ro", "/usr", "/usr"), Mount("ro", "/bin", "/bin"), Mount("ro", "/lib", "/lib"),
        Mount("ro", "/lib64", "/lib64"), Mount("ro", "/etc", "/etc"),
        Mount("hide", "/scratch"), Mount("hide", "/scratch2"), Mount("hide", "/tmp"),
        Mount("proc", "/proc"), Mount("dev", "/dev"),
    ]
    # The experimenter's Claude config, from $CLAUDE_CONFIG_DIR (else ~/.claude) — never the literal
    # home path. A host that relocates it kept valid credentials there while `~/.claude` held an expired
    # session, so binding the hard-coded path produced `OAuth session expired and could not be
    # refreshed` inside the sandbox. The memory dirs UNDER it are answer surfaces and are masked back out
    # by the derived pass (see answer_surfaces.experimenter_memory_dirs).
    from merlin.targetgen.sandbox.answer_surfaces import claude_config_dir
    cfg = claude_config_dir()
    if cfg.exists():
        mounts.append(Mount("rw", str(cfg), str(cfg)))

    def _kind(p: Path) -> str:
        # permission-safe: a chmod-000 lock makes stat() raise — treat as present-dir so a locked answer
        # surface is still masked, never crashes the binder.
        try:
            if p.is_dir():
                return "dir"
            if p.exists():
                return "file"
            return "missing"
        except PermissionError:
            return "dir"

    def _resolve_grant(rel: str) -> Path:
        # Bundle-convention grants are repo-root-relative, with a documented shorthand: an
        # ``experiments/...`` grant "resolves under merlin/". Prefer <repo>/<rel>, fall back to
        # <repo>/merlin/<rel> — so a path that only exists under merlin/ is bound, not silently dropped.
        # This MUST match bwrap.base_argv._resolve_grant: docker binds from the SAME plan bwrap does, so a
        # divergence here means the docker sandbox silently loses the experiment-relative grants (the
        # hwbringup ISA docs, the task dir, agent_selfcheck) that bwrap provides — leaving them as dangling
        # symlinks the agent cannot read.
        p = repo / rel
        if _kind(p) != "missing":
            return p
        q = repo / "merlin" / rel
        return q if _kind(q) != "missing" else p

    for entry in bundle.get("allowed", []):
        p = _resolve_grant(entry["path"])
        if _kind(p) != "missing":
            mounts.append(Mount("ro", str(p), str(p)))
    # Deny wins: mask every denied path after the allowed binds, so a broad allow (e.g. all of
    # merlin/contract/) cannot expose a denied sub-path (e.g. capsules/hidden/). A denied DIR is hidden; a
    # denied FILE under an allowed dir is /dev/null-overlaid (a dir mask does not cover it) — bwrap parity.
    for d in bundle.get("denied", []):
        p = _resolve_grant(d["path"])
        k = _kind(p)
        if k == "dir":
            mounts.append(Mount("hide", str(p)))
        elif k == "file":
            mounts.append(Mount("devnull", str(p), src=_DEVNULL))
    # The writable workspace goes LAST so no mask clobbers it.
    mounts.append(Mount("rw", str(ws), str(ws)))
    return MountPlan(mounts=tuple(mounts), workdir=str(ws))


def claude_runtime_plan() -> list[Mount]:
    """The ``claude`` CLI runtime: launcher in ~/.local/bin, native binary in ~/.local/share/claude, nvm,
    ~/.config, plus a writable ~/.claude.json. Without these the agent launch fails 'command not found'.
    None of these is an answer surface (the memory dir under ~/.claude IS, and is masked separately)."""
    mounts: list[Mount] = []
    home = Path(os.path.expanduser("~"))
    for p in (home / ".local" / "bin", home / ".local" / "share" / "claude",
              home / ".nvm", home / ".config"):
        if p.exists():
            mounts.append(Mount("ro", str(p), str(p)))
    cj = home / ".claude.json"
    if cj.exists():
        mounts.append(Mount("rw", str(cj), str(cj)))
    return mounts


def full_plan(te: TargetExperiment, ws: Path, bundle: dict | None = None) -> MountPlan:
    """The complete plan for one target+arm: deny-by-default base + claude runtime + toolchain binds +
    the DERIVED answer masks. ``bundle`` may be ``{}``/None for a descriptor-only target — the masks are
    then driven purely by the descriptor."""
    from merlin.targetgen.sandbox import toolchain as TC   # local: avoid a heavy import at module load
    from merlin.targetgen.sandbox.answer_surfaces import answer_surfaces
    bundle = bundle or {}
    tc_mounts, unsetenv = TC.toolchain_plan(te)
    plan = base_plan(ws, bundle)
    plan = MountPlan(mounts=plan.mounts + tuple(claude_runtime_plan()) + tuple(tc_mounts),
                     unsetenv=tuple(unsetenv), workdir=plan.workdir)
    return apply_answer_masks(plan, answer_surfaces(te))
