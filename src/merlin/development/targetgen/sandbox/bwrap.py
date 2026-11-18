"""Assemble the deny-by-default bwrap command + PROVE its isolation by replaying the mount table.

``base_argv`` is the deny-by-default prefix: system dirs RO, all of ``/scratch*`` tmpfs-hidden, only the
arm bundle's ``allowed`` paths bound back, its ``denied`` sub-paths re-masked (deny-wins), the workspace
writable. On top go the claude runtime binds, the toolchain binds, and finally the derived answer masks.

``apply_answer_masks`` hides every derived answer surface re-exposed by a bind, including protected
contract inputs and runtime memory. It skips already hidden surfaces to avoid invalid overlays.

``coverage_gap`` is the hermetic guard: it replays the ordered mount table and returns the answer
surfaces still reachable. Empty == the sandbox masks the full derived answer set. This runs WITHOUT
launching bwrap, so it is the CI-safe isolation proof.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
from collections.abc import Callable, Sequence
from itertools import chain
from pathlib import Path

from merlin.common.access import contract_resource_roots
from merlin.common.paths import repo_root
from merlin.common.storage import content_store
from merlin.common.storage.bundle_snapshots import LexicalPathIndex, minimal_roots
from merlin.development.targetgen.sandbox.answer_surfaces import (
    AnswerSurface,
    CapsuleMetadata,
    _bind_alias,
    _support_bind_surfaces,
)
from merlin.development.targetgen.sandbox.answer_surfaces import answer_surfaces as answer_surfaces
from merlin.development.targetgen.sandbox.mounts import (
    _DEVNULL,
    _EXPOSE_OPS,
    _HIDE_DEST_OPS,
    _CallPathResolution,
    _MountPathIndex,
    _mounts,
    _resolved_bind_mounts,
    path_kind,
)
from merlin.development.targetgen.sandbox.mounts import _MountVisibility as _MountVisibilityBase
from merlin.target.registry.experiment_descriptor import TargetExperiment

# ---- bwrap -------------------------------------------------------------------------------------


def materialize_bundle_inputs(
    ws: Path,
    bundle: dict,
    *,
    repo: Path | None = None,
    descriptor: TargetExperiment | None = None,
    private_sources: Sequence[Path] = (),
) -> dict:
    """Freeze declared inputs into an immutable run-private snapshot.

    Pin bytes against operator edits; copy nested grants only through their shallowest ancestor.
    Missing grants or contradictory public facts selections refuse before a new freeze.
    Existing verified snapshots retain their original declarations and private masks.
    ``private_sources`` are new-freeze copy hints: overlapping roots bypass shared CAS and keep
    restrictive source modes without write bits; the hints grant nothing and change no masks.
    """
    from merlin.development.targetgen.sandbox.grants import (
        _SNAPSHOT_COMPLETE,
        _grant_sources,
        _host_input_paths,
        _judge_rtl_facts_selection,
        _make_snapshot_writable,
        _private_validation_paths,
        _snapshot_content,
        _snapshot_path,
        _validate_host_locations,
        _validate_host_sources,
        _validate_selected_public_facts,
        _verify_judge_rtl_facts_leaf,
        bundle_snapshot_root,
        require_snapshot_ownership,
        resolve_grant,
        verify_bundle_snapshot,
    )

    private_sources = tuple(Path(path) for path in private_sources)
    if any(not path.is_absolute() or ".." in path.parts for path in private_sources):
        raise ValueError("private copy sources must be absolute paths without parent traversal")
    repo = (repo or repo_root()).absolute()
    judge_facts = _judge_rtl_facts_selection(bundle, repo=repo)
    root = bundle_snapshot_root(ws)
    pending = root.with_name(root.name + ".pending")
    if root.exists() or root.is_symlink():
        previous = verify_bundle_snapshot(ws, bundle, repo=repo)
        require_snapshot_ownership(previous)
        if judge_facts is not None:
            _verify_judge_rtl_facts_leaf(_snapshot_path(root, judge_facts[0], repo), judge_facts[1])
        return previous
    if pending.exists() or pending.is_symlink():
        shutil.rmtree(pending) if pending.is_dir() else pending.unlink()
    from merlin.development.targetgen.authoring.lean_toolchain import validate_private_toolchain_roots

    host_paths = _host_input_paths(bundle)
    validate_private_toolchain_roots(bundle)

    grants = _grant_sources(bundle, repo)
    hosts = [(path, resolve_grant(path, repo)) for path in host_paths]
    if any(source.absolute() != (repo / path).absolute() for path, source in hosts):
        raise RuntimeError("host-only inputs require canonical paths, not shorthand or registry aliases")
    _validate_host_locations(ws, hosts)
    _validate_selected_public_facts(bundle, hosts)
    inputs = [*grants, *hosts]
    missing = [rel for rel, source in inputs if path_kind(source) == "missing"]
    if missing:
        raise FileNotFoundError("bundle declares unresolvable allowed grant(s): " + ", ".join(sorted(missing)))
    private_validation = _private_validation_paths(bundle)
    if any(path_kind(path) not in {"missing", kind} for path, kind in private_validation):
        raise RuntimeError("private validation path changed kind before input freezing")
    _validate_host_sources(grants, hosts, [path for path, _ in private_validation])
    from merlin.development.targetgen.sandbox.answer_surfaces import _support_package_dirs

    owner_paths = set(_support_package_dirs())
    if descriptor is not None and descriptor.backend_package:
        declared = repo / str(descriptor.backend_package)
        if declared.is_dir():
            owner_paths.add(declared)
    owners = sorted({str(path.resolve(strict=True)) for path in owner_paths})
    views: list[dict] = []

    def observe(lexical: Path, canonical: Path, copied: Path) -> None:
        relative = copied.relative_to(pending)
        for owner in owners:
            private = Path(owner)
            spellings = (lexical.absolute(), canonical)
            full = any(path == private for path in spellings)
            withheld = any(
                path.is_relative_to(private) and not path.is_relative_to(private / "contracts") for path in spellings
            )
            if not withheld:
                continue
            grantable = ["contracts"] if full else []
            # Parent-first copying lets us retain only policy transitions. A
            # private alias INSIDE contracts remains a separate denial.
            if any(
                row["owner"] == owner
                and relative.is_relative_to(Path(row["path"]))
                and not any(relative.is_relative_to(Path(row["path"]) / sub) for sub in row["grantable"])
                for row in views
            ):
                continue
            views.append(
                {"path": relative.as_posix(), "owner": owner, "kind": path_kind(canonical), "grantable": grantable}
            )

    # Copy the union, not every overlapping spelling.  For example, a broad
    # contract grant plus its isa/layers children must produce one snapshot.
    roots = minimal_roots(source for _, source in inputs)
    private_index = LexicalPathIndex.from_paths(
        chain((path for _, path in hosts), private_sources, (path for path, _ in private_validation))
    )

    pending.mkdir(parents=True, mode=0o700)
    # One copy of each distinct byte-string, hard-linked here. A campaign's runs declare overlapping
    # closures -- the same toolchain, the same model weights -- and deep-copying them per run cost
    # 12.8 GB a run, 235 GiB across one campaign, for inputs that were byte-identical every time.
    # These are still the run's own frozen bytes: the store holds its OWN copy of them, so an
    # in-place edit of the source mutates the source's inode and cannot reach this snapshot.
    # Symlinks are dereferenced, as the copy this replaces did, so an absolute link into an external
    # RTL checkout cannot remain a live escape. See merlin.common.storage.content_store.
    store = content_store.store_root()
    try:
        for source in roots:
            dst = _snapshot_path(pending, source, repo)
            dst.parent.mkdir(parents=True, exist_ok=True)
            # A private descendant must never enter the globally shared CAS,
            # even transiently through a broad public parent. Keep that copied
            # root private; disjoint toolchain/public roots still deduplicate.
            private_copy = private_index.overlaps(source)
            input_store = None if private_copy else store
            if source.is_dir():
                content_store.place_tree(source, dst, input_store, observe=observe, preserve_permissions=private_copy)
            else:
                content_store.place_file(source, dst, input_store, observe=observe, preserve_permissions=private_copy)
        validate_private_toolchain_roots(bundle, snapshot_root=pending, repo=repo)
        if judge_facts is not None:
            _verify_judge_rtl_facts_leaf(_snapshot_path(pending, judge_facts[0], repo), judge_facts[1])
        digest, n_files, n_bytes = _snapshot_content(pending)
        grant_records = []
        for rel, source in inputs:
            grant_records.append(
                {
                    "path": rel,
                    "destination": str(source.absolute()),
                    "snapshot": _snapshot_path(pending, source, repo).relative_to(pending).as_posix(),
                }
            )
        manifest = {
            "version": 4,
            "repo": str(repo),
            "allowed": [rel for rel, _ in grants],
            "grants": grant_records[: len(grants)],
            "copied_roots": [str(p) for p in roots],
            "content_sha256": digest,
            "n_files": n_files,
            "n_bytes": n_bytes,
            "support_ownership": {"owners": owners, "views": views},
            "host_inputs": [rel for rel, _ in hosts],
            "host_records": grant_records[len(grants) :],
        }
        (pending / _SNAPSHOT_COMPLETE).write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        for path in sorted(pending.rglob("*"), key=lambda p: len(p.parts), reverse=True):
            # Host-immutable means clear write bits, not executable bits.  A
            # bundle may grant compilers and scripts that must remain runnable.
            if path.is_file() and content_store.is_shared(path):
                continue  # already read-only in the store, and shared
            path.chmod(path.stat().st_mode & ~0o222)
        pending.chmod(pending.stat().st_mode & ~0o222)
        pending.rename(root)
        return manifest
    except BaseException:
        if pending.exists():
            _make_snapshot_writable(pending)
            shutil.rmtree(pending)
        raise


def _bundle_mount_args(
    ws: Path,
    bundle: dict,
    repo: Path,
    *,
    _policy_test_live_inputs: bool = False,
    visibility_prefix: Sequence[str] | None = None,
) -> list[str]:
    """Frozen allow mounts and deny overlays, ordered so the MOST SPECIFIC declaration wins.

    bwrap applies mount operations in argv order, so whichever declaration is emitted LAST decides
    what a path shows. Emitting every allow and then every deny made a deny win over an allow it
    merely CONTAINED, which is not what a manifest means. An arm that grants the shared hardware
    spec at ``experiments/studies/.../isa_include/`` and separately denies ``merlin/`` ("internals")
    had all three ISA grants tmpfs-wiped by that broad deny: inside the box the workspace's
    ``isa_definition.py`` / ``atlas_isa_green_card.md`` / ``atlas`` symlinks dangled, and the agent
    -- told by its own task file to "derive EVERY instruction's exact encoding from these files,
    do NOT invent" -- found an empty room and invented one, which is the documented way to score 0
    on a self-hosted ISA. Nothing reported a setup failure; it read as the agent being bad at atlas.

    Order by destination DEPTH instead, deny last on a tie. That is longest-prefix-wins, the rule
    the manifests were already written against:
      * allow ``src/merlin/resources/contracts/`` + deny ``experiments/corpora/<t>/hidden/`` -- the deny is
        DEEPER, so goldens stay masked (the property "deny wins" existed to protect);
      * allow ``experiments/studies/.../isa_definition.py`` + deny ``merlin/`` -- the allow is
        deeper, so the granted spec survives the broad deny.
    A deny and an allow naming the SAME path still fails closed.
    """
    from merlin.development.targetgen.sandbox.grants import (
        _grant_sources,
        _host_input_paths,
        _private_validation_paths,
        _snapshot_grants,
        _snapshot_host_inputs,
        _validate_host_sources,
        require_snapshot_ownership,
        resolve_grant,
    )

    out: list[str] = []
    manifest = None
    if not bundle.get("allowed"):
        grants = []
    elif _policy_test_live_inputs:
        grants = [
            (str(entry["path"]), resolve_grant(entry["path"], repo), resolve_grant(entry["path"], repo))
            for entry in bundle.get("allowed", [])
        ]
    else:
        manifest, grants = _snapshot_grants(ws, bundle, repo)
        require_snapshot_ownership(manifest)
    # (depth, deny_wins_the_tie, args). sorted() is stable, so same-depth entries keep manifest order.
    live = [
        (destination, frozen)
        for _, destination, frozen in grants
        if path_kind(destination) != "missing" or not _policy_test_live_inputs
    ]
    allow_dests = [d.absolute() for d, _ in live]
    deny_dests: list[tuple[Path, str]] = []
    host_deny_dests: list[tuple[Path, str]] = []
    for denied in bundle.get("denied", []):
        d = resolve_grant(denied["path"], repo).absolute()
        kind = path_kind(d)
        if kind != "missing":
            deny_dests.append((d, kind))
    if bundle.get("host_inputs"):
        if _policy_test_live_inputs:
            public = _grant_sources(bundle, repo)
            private = [(path, resolve_grant(path, repo)) for path in _host_input_paths(bundle)]
            _validate_host_sources(public, private, [path for path, _ in _private_validation_paths(bundle)])
            host_records = [(path, source.absolute(), source) for path, source in private]
        else:
            # This same composition already checked the frozen grants and live
            # host locations above. Reuse that manifest, not a second complete
            # host inventory; host-only bundles still perform the check here.
            if manifest is None:
                manifest, _ = _snapshot_grants(ws, bundle, repo)
            host_records = _snapshot_host_inputs(ws, bundle, repo, manifest)
        host_deny_dests.extend((destination, path_kind(frozen)) for _, destination, frozen in host_records)
    for private, kind in _private_validation_paths(bundle):
        if private == ws or private in ws.parents or private.is_relative_to(ws):
            raise RuntimeError("private validation input overlaps the writable authoring workspace")
        observed = path_kind(private)
        if private.is_symlink() or (observed != "missing" and observed != kind):
            raise RuntimeError("private validation path changed kind")
        if any(allowed == private or allowed.is_relative_to(private) for allowed in allow_dests):
            raise RuntimeError("candidate grant overlaps an operator-private validation input")
        if observed == "missing" and any(private.is_relative_to(allowed) for allowed in allow_dests):
            raise RuntimeError("candidate grant could expose a future operator-private validation output")
        # Scratch is already hidden. A redundant deeper --tmpfs under an
        # absent parent can make bwrap fail before the agent starts; add the
        # denial only when an actual declared parent grant would expose it.
        if observed != "missing" and any(private.is_relative_to(allowed) for allowed in allow_dests):
            deny_dests.append((private, kind))

    ops: list[tuple[int, int, list[str]]] = []
    for destination, frozen in live:
        destination = destination.absolute()
        args: list[str] = []
        mount_at = destination
        # Binding a FILE needs its parent DIRECTORY to exist inside the box: bwrap builds the parent
        # chain for a directory bind but not for a file one, and a broad deny's tmpfs removes it. Where
        # to mount then depends on whether a DIRECTORY on the way is a symlink, and BOTH cases occur:
        #   * gemmini grants `.../hwbringup_gemmini_v0` AND headers under its `isa_include`, which is a
        #     symlink to the vendor checkout on /scratch2 -- tmpfs-masked by policy. The dir grant
        #     reproduces the dangling link, so bwrap follows it to a path that does not exist and
        #     refuses ("Can't create file at ...: No such file or directory"), killing the sandbox
        #     before the agent runs a turn. Mount at the RESOLVED path; the symlink lands on the bytes.
        #   * every target grants `targets/<t>/scripts/agent_selfcheck.py`, whose `scripts` is also a
        #     symlink -- but nothing grants a directory that reproduces it. Here bwrap creates a real
        #     chain at the SPELLED path, which is what makes the grant readable; resolving it instead
        #     would hide the agent's own self-check.
        # So resolve only when an ancestor grant actually reproduces the symlink.
        if destination.is_file():
            parent_real = os.path.realpath(destination.parent)
            args += ["--dir", parent_real]
            if parent_real != str(destination.parent) and any(
                a != destination and destination.parent.is_relative_to(a) for a in allow_dests
            ):
                mount_at = Path(parent_real) / destination.name
        args += ["--ro-bind", str(frozen), str(mount_at)]
        ops.append((len(destination.parts), 0, args))
    for d, kind in deny_dests:
        if kind == "dir":
            ops.append((len(d.parts), 1, ["--tmpfs", str(d)]))
        else:
            ops.append((len(d.parts), 1, ["--ro-bind", _DEVNULL, str(d)]))
    visibility = None
    if visibility_prefix is not None:
        preview = list(visibility_prefix)
        for _depth, _tie, args in sorted(ops, key=lambda op: (op[0], op[1])):
            preview.extend(args)
        visibility = _MountVisibility(preview)
    for destination, kind in host_deny_dests:
        # Consult the full actual prefix AND the pending public grants/masks.
        # System/runtime binds can expose a host file even without a bundle
        # grant; later reapply calls reconsider their complete composed prefix.
        # A standalone caller without that prefix retains every legacy denial.
        if visibility is not None and not visibility.is_exposed(destination):
            continue
        args = ["--tmpfs", str(destination)] if kind == "dir" else ["--ro-bind", _DEVNULL, str(destination)]
        ops.append((len(destination.parts), 1, args))
    for _depth, _tie, args in sorted(ops, key=lambda op: (op[0], op[1])):
        out += args
    return out


def reapply_bundle_snapshot(argv: list[str], ws: Path, bundle: dict, *, repo: Path | None = None) -> list[str]:
    """Reassert frozen grants after later trusted-runtime/toolchain binds.

    A universal toolchain path may also be a declared arm input.  Since bwrap is
    ordered, appending this layer ensures that such an overlap still serves the
    experiment snapshot.  Denied subpaths are appended in the same layer and
    therefore continue to win.

    The workspace bind is re-asserted LAST. A deny is normally a subpath of a grant, but an arm may deny
    a whole tree that happens to CONTAIN the workspace (an arm withheld from ``merlin/`` while its
    workspace lives at ``experiments/studies/.../_qa_ws/<run>/workspace``). ``base_argv`` already binds the
    workspace after the deny layer for exactly that reason; re-appending the deny layer here without
    re-appending the bind put the tmpfs back on top, and bwrap died with "Can't chdir to
    <ws>: No such file or directory" before the agent ran a single turn -- every round, scoring an empty
    submission rather than reporting a setup failure.
    """
    repo = (repo or repo_root()).absolute()
    return [*argv, *_bundle_mount_args(ws, bundle, repo, visibility_prefix=argv), "--bind", str(ws), str(ws)]


def base_argv(
    ws: Path,
    bundle: dict,
    *,
    repo: Path | None = None,
    _policy_test_live_inputs: bool = False,
    include_claude_home: bool = True,
    inherit_environment: bool = True,
) -> list[str]:
    """Deny-by-default bwrap argv prefix: system RO, /scratch* tmpfs-hidden, ONLY the bundle's allowed
    paths bound RO, denied sub-paths re-masked, workspace writable+last. Target-agnostic — the ``bundle``
    (or an empty ``{}``) is the only input beyond the workspace."""
    from merlin.development.targetgen.sandbox.grants import _selected_rtl_facts_file

    repo = repo or repo_root()
    parts = [
        "bwrap",
        *(
            []
            if inherit_environment
            else ["--clearenv", "--setenv", "HOME", str(Path.home()), "--setenv", "PATH", "/usr/bin:/bin"]
        ),
        "--unsetenv",
        "MERLIN_CORPUS_SEAL",
        "--die-with-parent",
        "--unshare-pid",
        "--ro-bind",
        "/usr",
        "/usr",
        "--ro-bind",
        "/bin",
        "/bin",
        "--ro-bind",
        "/lib",
        "/lib",
        "--ro-bind",
        "/lib64",
        "/lib64",
        "--ro-bind",
        "/etc",
        "/etc",
        # DNS: /etc/resolv.conf is a symlink into the systemd-resolved runtime dir. Binding /etc alone
        # leaves that symlink dangling inside the sandbox, so every name lookup fails and the agent's
        # `claude` session hangs on an unreachable API. Bind the resolver dir so the symlink resolves.
        # --ro-bind-try tolerates non-systemd hosts (where resolv.conf is a real file under /etc).
        "--ro-bind-try",
        "/run/systemd/resolve",
        "/run/systemd/resolve",
        "--tmpfs",
        "/scratch",
        "--tmpfs",
        "/scratch2",
        "--tmpfs",
        "/tmp",
        "--proc",
        "/proc",
        "--dev",
        "/dev",
        # a writable XDG runtime dir under the tmpfs /tmp — the Bun-based `claude` opens a socket there.
        "--dir",
        "/tmp/.xdg",
        "--setenv",
        "XDG_RUNTIME_DIR",
        "/tmp/.xdg",
        "--chdir",
        str(ws),
    ]
    selected_facts = _selected_rtl_facts_file(bundle, repo=repo)
    # Do not inherit an operator's mutable cache selection into the frozen agent.
    parts += (
        ["--setenv", "MERLIN_RTL_FACTS", str(selected_facts)]
        if selected_facts is not None
        else ["--unsetenv", "MERLIN_RTL_FACTS"]
    )
    # Drop Claude-Code nesting markers inherited from a parent agent session so the sandboxed `claude`
    # starts a clean top-level session. A leaked CLAUDE_CODE_MESSAGING_SOCKET / CLAUDECODE makes it wait on
    # a parent IPC socket that is not inside the box and hang. Auth vars (ANTHROPIC_API_KEY,
    # CLAUDE_CODE_USE_BEDROCK) are intentionally NOT cleared — the launch may need them.
    for _v in (
        "CLAUDECODE",
        "CLAUDE_CODE_ENTRYPOINT",
        "CLAUDE_CODE_SSE_PORT",
        "CLAUDE_CODE_MESSAGING_SOCKET",
        "CLAUDE_CODE_CHILD_SESSION",
        "CLAUDE_CODE_SESSION_ID",
        "CLAUDE_PID",
        "CLAUDE_EFFORT",
        "AI_AGENT",
    ):
        parts += ["--unsetenv", _v]
    home_claude = os.path.expanduser("~/.claude")
    if include_claude_home and Path(home_claude).exists():
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

    parts += _bundle_mount_args(
        ws, bundle, repo, _policy_test_live_inputs=_policy_test_live_inputs, visibility_prefix=parts
    )
    # Bind the writable workspace LAST so no mask clobbers it.
    parts += ["--bind", str(ws), str(ws)]
    return parts


def claude_runtime_binds() -> list[str]:
    """RO-bind the ``claude`` CLI runtime (launcher in ~/.local/bin, native binary in ~/.local/share/
    claude, nvm, ~/.config) + make ~/.claude.json writable. Without these the agent launch fails
    'claude: command not found'. None of these are an answer surface."""
    binds: list[str] = []
    home = Path(os.path.expanduser("~"))
    for p in (home / ".local" / "bin", home / ".local" / "share" / "claude", home / ".nvm", home / ".config"):
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


class _MountVisibility(_MountVisibilityBase):
    """Retain the original parser lookup for callers overriding mount replay."""

    def __init__(self, argv: list[str], mounts: list[tuple[str, str, str]] | None = None):
        super().__init__(argv, _mounts(argv) if mounts is None else mounts)


def is_exposed(argv: list[str], path: Path) -> bool:
    """Whether the effective ordered mount exposes ``path``.

    Test the controlling source after guest symlinks; never cache source state.
    """
    return _MountVisibility(argv).is_exposed(path)


def coverage_gap(argv: list[str], surfaces: list[AnswerSurface]) -> list[AnswerSurface]:
    """The answer surfaces STILL reachable under ``argv`` — the drift/cheat guard. Empty == full mask."""
    mounts = _mounts(argv)  # one parse per batch, shared by the index and the bind projection
    visibility = _MountVisibility(argv, mounts)
    return [s for s in _support_bind_surfaces(mounts, surfaces) if visibility.is_exposed(s.path)]


def apply_answer_masks(argv: list[str], surfaces: list[AnswerSurface]) -> list[str]:
    """Append masks for every answer surface a bind would otherwise re-expose. A surface already hidden
    by deny-by-default is skipped (no redundant overlay, and no mount whose parent tmpfs would fail).
    File surfaces are /dev/null-overlaid; dir surfaces are tmpfs'd. Masks go LAST so they win."""
    out = list(argv)
    mounts = _mounts(out)  # one parse per batch, shared by the index and the bind projection
    visibility = _MountVisibility(out, mounts)
    for s in _support_bind_surfaces(mounts, surfaces):
        if not visibility.is_exposed(s.path):
            continue
        if s.kind == "file":
            out += ["--ro-bind", _DEVNULL, str(s.path)]
        else:
            out += ["--tmpfs", str(s.path)]
        visibility.append("hide", "", str(s.path))
    return out


def _frozen_support_surfaces(ws: Path, bundle: dict, surfaces: list[AnswerSurface], repo: Path) -> list[AnswerSurface]:
    """Use copy-time ownership even when live providers/aliases disappear or change."""
    from merlin.development.targetgen.sandbox.grants import _snapshot_grants, snapshot_support_surfaces

    manifest, _ = _snapshot_grants(ws, bundle, repo)
    from merlin.common.storage.bundle_snapshots import guarded_projection

    return guarded_projection(
        ws, bundle, repo, ["sandbox.frozen_support.v1", manifest], lambda: snapshot_support_surfaces(ws, manifest)
    )


def _frozen_answer_surfaces(
    ws: Path,
    bundle: dict,
    te: TargetExperiment,
    live: list[AnswerSurface],
    repo: Path,
    capsules: CapsuleMetadata | None = None,
) -> list[AnswerSurface]:
    from merlin.development.targetgen.sandbox.grants import _snapshot_grants, bundle_snapshot_root

    if capsules is not None:
        # Only the frozen tree contributes filesystem classifications here;
        # live surfaces and descriptor roots are explicit parts of the key.
        # Dynamic live-answer discovery still runs independently before this.
        # Arm path/marker watches BEFORE resolving the records. Reuse that
        # per-call resolution only if the same guard survives up to consumption.
        # No guard means the original fresh second reader remains mandatory.
        frozen_root = bundle_snapshot_root(ws)
        declaration_facts = capsules.inventory.tree(frozen_root)
        manifest, grants = _snapshot_grants(ws, bundle, repo)
        key = (
            "frozen_answers",
            repo,
            te.capsule_corpus,
            tuple(te.corpus_siblings()),
            tuple(contract_resource_roots(repo, "capsules")),
            tuple(contract_resource_roots(repo, "examples")),
            tuple(live),
            tuple(grants),
            manifest["content_sha256"],
        )
        return capsules.derived(
            key,
            (bundle_snapshot_root(ws),),
            lambda: _classify_frozen_answer_surfaces(
                ws,
                bundle,
                te,
                live,
                repo,
                capsules,
                grants=(
                    grants
                    if declaration_facts is not None and capsules.inventory.tree(frozen_root) is declaration_facts
                    else None
                ),
            ),
        )
    return _classify_frozen_answer_surfaces(ws, bundle, te, live, repo, capsules)


def _classify_frozen_answer_surfaces(ws, bundle, te, live, repo, capsules, *, grants=None):
    """Classify answer bytes in pinned public grants, independent of mutable live sources.

    The snapshot marker fixes grant destinations and sources. Walking only its
    corpus/example subtrees inspects names, not payload bytes or a fresh digest;
    the host-owned snapshot verification remains the authority for those bytes.
    A live golden removed after freezing must still be denied at both its public
    grant destination and any late runtime alias of the snapshot source.
    """
    # The metadata owner already validated these exact immutable declarations
    # under the same unchanged path/marker guard. Reuse only within this call,
    # not across snapshots or callers; byte custody remains external.
    from merlin.development.targetgen.sandbox.grants import _snapshot_grants

    if grants is None:
        _, grants = _snapshot_grants(ws, bundle, repo)
    corpus_roots = list(contract_resource_roots(repo, "capsules"))
    if te.capsule_corpus is not None:
        corpus_roots.extend((te.capsule_corpus.parent, te.capsule_corpus))
    corpus_roots.extend(repo / rel.rstrip("/") for rel in te.corpus_siblings())
    example_roots = [*contract_resource_roots(repo, "examples"), repo / "tests"]
    result: dict[Path, AnswerSurface] = {}

    def add(path: Path, kind: str, origin: str, grantable: tuple[str, ...] = ()) -> None:
        if path_kind(path) == kind:
            result[path] = AnswerSurface(f"frozen {origin}:{path.name}", path, kind, origin, grantable)

    live_index = _MountPathIndex((surface.path,) for surface in live)
    for _, destination, source in grants:
        # Preserve every discovered private identity, including registry-owned
        # modules and support providers copied by a broad public grant.
        for surface_id in sorted(live_index.descendants(destination)):
            surface = live[surface_id]
            add(
                source / surface.path.relative_to(destination),
                surface.kind,
                surface.origin,
                surface.grantable,
            )

        classified_roots = [(path, "corpus") for path in corpus_roots]
        classified_roots.extend((path, "example") for path in example_roots)
        # These roots may overlap (the selected corpus and its parent, for
        # example). Inspect each exact lexical member/classification once for
        # THIS grant only. Keep walks fresh, distinct aliases/classes separate,
        # and preserve grant ordering; snapshot boundary checks still own bytes.
        inspected: set[tuple[Path, str]] = set()
        for base, category in classified_roots:
            if base.is_relative_to(destination):
                frozen_base = source / base.relative_to(destination)
            elif destination.is_relative_to(base):
                frozen_base = source
            else:
                continue
            members = chain((frozen_base,), frozen_base.rglob("*")) if frozen_base.is_dir() else (frozen_base,)
            for member in members:
                key = (member, category)
                if key in inspected:
                    continue
                inspected.add(key)
                if category == "corpus":
                    if member.is_dir() and member.name == "hidden":
                        add(member, "dir", "hidden")
                    elif member.is_file():
                        if member.name == "capsule.yaml":
                            from merlin.development.targetgen.corpus.declarations import (
                                capsule_answer_members,
                            )

                            declared = {} if capsules is None else {"capsule": capsules.load(member.parent)}
                            for relative, origin in capsule_answer_members(member.parent, **declared).items():
                                add(member.parent / relative, "file", origin)
                        if member.name.startswith("golden.") or member.name == "expected_instruction_coverage.yaml":
                            add(member, "file", "golden")
                        elif member.name.endswith((".safetensors", ".safetensors.manifest.json")):
                            add(member, "file", "weight")
                        elif member.parent.name == "profiles" and member.name.endswith(".hidden.yaml"):
                            add(member, "file", "hidden")
                elif member.is_file() and member.name.startswith("expected_"):
                    add(member, "file", "example")
    return list(result.values())


def apply_final_answer_masks(
    argv: list[str],
    te: TargetExperiment,
    ws: Path,
    bundle: dict,
    *,
    repo: Path | None = None,
    capsules: CapsuleMetadata | None = None,
    _policy_test_live_inputs: bool = False,
) -> list[str]:
    """Withhold descriptor answers and verified private-input aliases after all binds.

    Callers first compose their runtime/toolchain/provider binds and reassert their
    frozen bundle. This shared final policy examines that complete mount list:
    canonical-path masks alone cannot withhold a private input mounted under a
    different name. No grants or snapshot authority are created here.

    ``capsules`` lets one execution scope reuse capsule metadata parsed from
    byte-identical files; every walk, listing and host-input scan stays fresh.
    """
    from merlin.development.targetgen.sandbox.host_surfaces import _final_answer_policy

    return _final_answer_policy(
        argv,
        te,
        ws,
        bundle,
        repo=repo or repo_root(),
        capsules=capsules,
        _policy_test_live_inputs=_policy_test_live_inputs,
    )[0]


def full_argv(
    te: TargetExperiment, ws: Path, bundle: dict | None = None, *, _policy_test_live_inputs: bool = False
) -> list[str]:
    """The complete isolation argv for one target+arm: deny-by-default base + claude runtime (never the
    operator's Claude state; no caller runs the CLI) + toolchain binds + derived answer masks. ``bundle``
    may be ``{}``/None for a descriptor-only target — the masks are then driven purely by the descriptor."""
    from merlin.development.backend.authoring import author_toolchain
    from merlin.development.targetgen.sandbox import cleanroom as CH  # local: avoid a heavy import at module load

    bundle = bundle or {}
    argv = (
        base_argv(ws, bundle, _policy_test_live_inputs=_policy_test_live_inputs, include_claude_home=False)
        + CH.without_operator_state(claude_runtime_binds())
        + author_toolchain.toolchain_binds(te, bundle)
    )
    if bundle:
        if _policy_test_live_inputs:
            argv += _bundle_mount_args(ws, bundle, repo_root(), _policy_test_live_inputs=True)
        else:
            argv = reapply_bundle_snapshot(argv, ws, bundle)
    return apply_final_answer_masks(argv, te, ws, bundle, _policy_test_live_inputs=_policy_test_live_inputs)


_MAX_ARG_BYTES = 32 * 1024


def wrap(
    te: TargetExperiment,
    ws: Path,
    inner: str,
    bundle: dict | None = None,
    *,
    argv_guard: Callable[[list[str]], None] | None = None,
    _policy_test_live_inputs: bool = False,
) -> str:
    """A ready-to-run ``bash -c`` string: full argv + the sandbox env exports + the inner command.

    THE BIND LIST DOES NOT SCALE INSIDE A COMMAND STRING, and every caller of this function passes the
    result as ONE argument to ``bash -c``. One mask per answer surface is right -- a corpus with 1,098
    goldens gets 1,098 ``--ro-bind /dev/null`` pairs -- but at that size the string reaches 159 KB and
    execve refuses any single argument over 128 KiB. The failure is E2BIG naming ``bash``, which says
    nothing about masks, and it would have hit the live agent path (the drivers run the whole agent
    process inside this wrapper) exactly as hard as it hit the isolation test.
    ``bwrap --args FD`` exists for this: the arguments are read NUL-separated from a file descriptor
    instead of the command line. The masks are then bounded by the file, not by execve, and the string
    this returns stays short however large the corpus grows. Below the threshold the inline form is
    kept, so nothing changes for a small corpus and the two forms can be compared.
    """
    from merlin.development.backend.authoring import author_toolchain

    argv = full_argv(te, ws, bundle, _policy_test_live_inputs=_policy_test_live_inputs)
    if argv_guard is not None:
        argv_guard(argv)
    return compose_command(argv, f" bash -c '{author_toolchain.sandbox_env(te, ws, bundle=bundle or {})} {inner}'", ws)


def compose_command(argv: list[str], tail: str, ws: Path) -> str:
    """THE ONE PLACE a bwrap argv becomes a shell string, so the size rule cannot be half-applied.

    Every caller passes the result as a single argument to ``bash -c``, and a single execve argument
    may not exceed MAX_ARG_STRLEN. There were two independent copies of this join -- this module's
    ``wrap`` and the run loop's own ``bwrap_cmd`` -- and fixing only the first left the live agent path
    dying with E2BIG on launch while the isolation suite passed. ``tail`` is appended verbatim so each
    caller keeps its own payload quoting.
    """
    import shlex

    if "\0" in tail or any("\0" in argument for argument in argv):
        raise ValueError("sandbox command cannot represent embedded NUL bytes")
    inline = shlex.join(argv) + tail
    if len(inline.encode("utf-8")) <= _MAX_ARG_BYTES:
        return inline
    if not argv or Path(argv[0]).name != "bwrap":
        raise ValueError("sandbox argument-file transport requires the bwrap executable")
    return _wrap_via_args_fd(argv, tail, ws)


def _wrap_via_args_fd(argv: list[str], tail: str, ws: Path) -> str:
    """The same sandbox, with every argument after ``bwrap`` moved into a NUL-separated file.

    The file lives beside the workspace rather than in it: the agent must not be able to read or edit
    the argument list that isolates it. ``exec {fd}<file`` opens it and ``--args $fd`` tells bwrap to
    parse from there; the redirection is a shell builtin, so no extra process sees the arguments.
    """
    import hashlib
    import shlex

    digest = hashlib.sha256("\0".join(argv).encode("utf-8")).hexdigest()[:12]
    payload = ws.parent / f".{ws.name}.bwrap-args.{digest}"
    payload.write_bytes(b"\0".join(a.encode("utf-8") for a in argv[1:]) + b"\0")
    payload.chmod(0o600)
    return f"exec {{__bwargs}}<{shlex.quote(str(payload))} && {shlex.quote(argv[0])} --args $__bwargs" + tail
