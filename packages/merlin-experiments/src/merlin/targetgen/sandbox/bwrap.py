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
import stat
from collections.abc import Sequence
from pathlib import Path

from merlin.common import content_store
from merlin.common.paths import repo_root
from merlin.targetgen.sandbox.answer_surfaces import AnswerSurface, answer_surfaces
from merlin.targetgen.target_experiment import TargetExperiment

_EXPOSE_OPS = ("--ro-bind", "--bind", "--dev-bind", "--ro-bind-try", "--bind-try")
_HIDE_DEST_OPS = ("--tmpfs",)  # single-arg-dest hide ops
_DEVNULL = "/dev/null"
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
    if path_kind(q) != "missing":
        return q
    derived = _resolve_target_package_grant(rel)
    return derived if derived is not None else p


#: The tail of a per-target grant whose home the TARGET REGISTRY owns, not the manifest.
_REGISTRY_OWNED_GRANT_TAILS = ("contracts/rtl_facts", "contracts")


def _resolve_target_package_grant(rel: str) -> Path | None:
    """A ``merlin/targets/<t>/contracts/...`` grant resolved through the target registry, or ``None``.

    A TARGET'S PACKAGE DOES NOT ALWAYS LIVE UNDER ``merlin/targets/``. One target's contracts and RTL
    facts are curated in-tree; another's are GENERATED, and the registry is what knows which -- it
    resolves each target to its own ``contract_path``/``facts_path``, which for a generated package sit
    under the build root. A manifest that spells the in-tree location for such a target names a
    directory that does not exist, and the grant then fails closed: measured, five atlas bundles
    granted ``merlin/targets/atlas/contracts/rtl_facts/`` and the sandbox could not be assembled at
    all, so the arm that is SUPPOSED to read RTL facts could not have read them.

    Resolving through the registry keeps the manifest saying WHAT it wants (this target's RTL facts)
    and lets the registry say WHERE, which is the same split the rest of the repo uses. Returns None
    when the shape does not match or the registry cannot resolve it -- the caller then reports the path
    exactly as the manifest declared it, which is what makes the failure legible.
    """
    parts = Path(rel).as_posix().strip("/").split("/")
    if len(parts) < 4 or parts[0] != "merlin" or parts[1] != "targets":
        return None
    target, tail = parts[2], "/".join(parts[3:])
    if tail not in _REGISTRY_OWNED_GRANT_TAILS:
        return None
    try:
        from merlin.targetgen import target_registry as _tr

        info = _tr.resolve(target)
    except Exception:  # noqa: BLE001 — unknown target: not our shape
        return None
    if not tail.endswith("rtl_facts"):
        base = Path(info.contract_path).parent
        return base if path_kind(base) != "missing" else None
    # Two homes for the facts, in the order the repo itself prefers: the target package's own
    # `contracts/rtl_facts/` where one has been written, else the PURGEABLE introspect cache that
    # `rtl.facts` calls "the single place that maps a target name -> its facts artifact". The cache is
    # where a regenerated target's facts actually land, and granting a snapshot FROM it is sound
    # because the snapshot copies the bytes rather than binding the directory.
    for base in (Path(info.facts_path).parent, _rtl_facts_dir(target)):
        if base is not None and path_kind(base) != "missing":
            return base
    return None


def _rtl_facts_dir(target: str) -> Path | None:
    try:
        from merlin.targetgen.rtl.facts import rtl_facts_path

        return Path(rtl_facts_path(target)).parent
    except Exception:  # noqa: BLE001 — no facts location for this target
        return None


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
    return [(str(entry["path"]), resolve_grant(str(entry["path"]), repo)) for entry in bundle.get("allowed", [])]


def _host_input_paths(bundle: dict) -> list[str]:
    entries = bundle.get("host_inputs", [])
    if not isinstance(entries, list) or any(
        not isinstance(entry, dict) or not isinstance(entry.get("path"), str) or not entry["path"].strip()
        for entry in entries
    ):
        raise RuntimeError("host_inputs must be a list of named path records")
    paths = [entry["path"] for entry in entries]
    if len(set(paths)) != len(paths) or any(".." in Path(path).parts for path in paths):
        raise RuntimeError("host_inputs must have unique, non-escaping paths")
    return paths


def _validate_host_sources(grants: list[tuple[str, Path]], hosts: list[tuple[str, Path]]) -> None:
    """Reject public aliases before dereferencing sources into the content store.

    A broader public directory may contain private inputs: the mount layer hides
    those paths unconditionally. An explicit child grant or a second symlink/hardlink
    name for private bytes must not reopen that denial.
    """
    if not hosts:
        return
    private = [path.absolute() for _, path in hosts]
    private_inodes = set()
    for path in private:
        if path.resolve() != path or path.is_symlink():
            raise RuntimeError("host-only inputs cannot use symlink aliases")
        for member in (path, *path.rglob("*")) if path.is_dir() else (path,):
            if member.is_symlink():
                raise RuntimeError("host-only inputs cannot contain symlinks")
            if member.is_file():
                stat = member.stat()
                private_inodes.add((stat.st_dev, stat.st_ino))

    def inspect(member: Path, ancestry: tuple[tuple[int, int], ...]) -> None:
        if any(member == path or path in member.parents for path in private):
            return  # A private child of a broad grant is always masked.
        resolved = member.resolve()
        if any(resolved == path or path in resolved.parents for path in private):
            raise RuntimeError("allowed grant contains an alias to host-only inputs")
        if resolved != member and any(resolved in path.parents for path in private):
            raise RuntimeError("allowed grant contains an alias to host-only inputs")
        stat = member.stat()
        key = (stat.st_dev, stat.st_ino)
        if key in private_inodes:
            raise RuntimeError("allowed grant contains a hardlink alias to host-only inputs")
        if member.is_dir():
            if key in ancestry:
                raise RuntimeError("allowed grant contains a directory cycle")
            # Follow directory aliases exactly as the freezer does. rglob does not
            # descend them and would miss a second-hop link into private inputs.
            for child in sorted(member.iterdir()):
                inspect(child, (*ancestry, key))

    for _, source in grants:
        source = source.absolute()
        if any(source == path or path in source.parents for path in private):
            raise RuntimeError("allowed grant overlaps a host-only input")
        inspect(source, ())


def _validate_host_locations(ws: Path, hosts: list[tuple[str, Path]]) -> None:
    for _, source in hosts:
        for private in {source.absolute(), source.resolve()}:
            for bound in {
                ws.absolute(),
                ws.resolve(),
                bundle_snapshot_root(ws).absolute(),
                bundle_snapshot_root(ws).resolve(),
            }:
                if private == bound or private in bound.parents or bound in private.parents:
                    raise RuntimeError("host-only input overlaps the writable workspace or snapshot storage")


def _snapshot_content(root: Path) -> tuple[str, int, int]:
    """Digest, file count and byte count for the frozen payload (not its marker)."""
    rows: list[tuple[str, str, int]] = []
    total = 0
    marker = root / _SNAPSHOT_COMPLETE
    members = sorted(root.rglob("*"))
    if any(path.is_symlink() for path in members):
        raise RuntimeError("bundle input snapshot contains a symlink")
    for path in (p for p in members if p.is_file() and p != marker):
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
    if root.is_symlink() or marker.is_symlink() or not marker.is_file():
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
    if manifest.get("version") not in (2, 3, 4) or manifest.get("repo") != str(repo):
        raise RuntimeError("bundle input snapshot metadata does not match this run")
    if manifest.get("allowed") != expected_allowed:
        raise RuntimeError("bundle input snapshot grant set does not match this run")
    grants = _snapshot_records(ws, repo, expected_allowed, manifest.get("grants"), frozen=manifest["version"] == 4)
    if manifest["version"] == 4:
        snapshot_support_surfaces(ws, manifest)
    hosts = _snapshot_host_inputs(ws, bundle, repo, manifest)
    _validate_host_locations(ws, [(path, destination) for path, destination, _ in hosts])
    for _, destination, _ in grants:
        if any(destination == private or private in destination.parents for _, private, _ in hosts):
            raise RuntimeError("allowed grant overlaps a host-only input")
    return manifest, grants


def _snapshot_host_inputs(ws: Path, bundle: dict, repo: Path, manifest: dict) -> list[tuple[str, Path, Path]]:
    expected = _host_input_paths(bundle)
    if manifest.get("version") == 2:
        if expected or manifest.get("host_inputs") or manifest.get("host_records"):
            raise RuntimeError("legacy snapshot cannot contain host-only inputs")
        return []
    if manifest.get("host_inputs") != expected:
        raise RuntimeError("bundle input snapshot host-only input set does not match this run")
    return _snapshot_records(ws, repo, expected, manifest.get("host_records"), canonical=True)


def _snapshot_records(
    ws: Path, repo: Path, expected_allowed: list[str], records: object, *, canonical: bool = False, frozen: bool = False
) -> list[tuple[str, Path, Path]]:
    """Validate one class of frozen paths; only public records become mounts."""
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
        # THE SAME RESOLVER THE MATERIALIZER USED. Listing the two shorthand candidates here restated
        # `resolve_grant`'s rule in a second place, and the two then disagreed the moment the rule grew
        # a third case: a grant whose home the target registry owns resolved fine at copy time and was
        # rejected at verify time, which reads as a corrupted snapshot rather than as two functions
        # that stopped agreeing.
        valid_destinations = {(repo / declared).absolute()}
        if not canonical and not frozen:
            valid_destinations.update(
                {(repo / "merlin" / declared).absolute(), resolve_grant(str(expected), repo).absolute()}
            )
        if (
            not destination.is_absolute()
            or ".." in destination.parts
            or (not frozen and destination not in valid_destinations)
        ):
            raise RuntimeError(f"bundle input snapshot destination is invalid for {expected!r}")
        if snapshot_rel.is_absolute() or ".." in snapshot_rel.parts:
            raise RuntimeError(f"bundle input snapshot path escapes its root for {expected!r}")
        if snapshot_rel != _snapshot_path(root, destination, repo).relative_to(root):
            raise RuntimeError("bundle input snapshot record does not match its declared destination")
        source = root / snapshot_rel
        try:
            resolved_root = root.resolve(strict=True)
            resolved_source = source.resolve(strict=True)
        except OSError as exc:
            raise RuntimeError(f"bundle input snapshot grant is incomplete for {expected!r}") from exc
        if resolved_source != resolved_root and resolved_root not in resolved_source.parents:
            raise RuntimeError(f"bundle input snapshot path escapes its root for {expected!r}")
        relative_source = source.relative_to(root)
        if any(
            (root / Path(*relative_source.parts[:i])).is_symlink() for i in range(1, len(relative_source.parts) + 1)
        ):
            raise RuntimeError(f"bundle input snapshot contains a symlink for {expected!r}")
        if path_kind(source) == "missing":
            raise RuntimeError(f"bundle input snapshot grant is incomplete for {expected!r}")
        out.append((expected, destination, source))
    return out


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
            f"bundle input snapshot content verification failed at {root}: expected {expected}, observed {observed}"
        )
    return manifest


def snapshot_record(ws: Path) -> dict:
    """Small provenance block copied into the run's environment record."""
    manifest = _read_snapshot_manifest(ws)
    record = {
        "path": str(bundle_snapshot_root(ws)),
        "content_sha256": manifest.get("content_sha256"),
        "n_files": manifest.get("n_files"),
        "n_bytes": manifest.get("n_bytes"),
        "version": manifest.get("version"),
    }
    if manifest.get("version") in (3, 4):
        marker = bundle_snapshot_root(ws) / _SNAPSHOT_COMPLETE
        record["manifest_sha256"] = hashlib.sha256(marker.read_bytes()).hexdigest()
    return record


def require_snapshot_ownership(manifest: dict) -> None:
    """Historical payloads remain inspectable, not newly verified executions."""
    if not isinstance(manifest, dict) or manifest.get("version") != 4:
        raise RuntimeError("verified execution requires bundle snapshot V4 ownership; create a newly frozen run")


def verify_snapshot_binding(ws: Path, bundle: dict, expected: dict, *, repo: Path | None = None) -> dict:
    """Compare policy metadata with existing host-owned provenance, never self-certify it."""
    require_snapshot_ownership(expected)
    observed = snapshot_record(ws)
    if observed != dict(expected):
        raise RuntimeError("bundle snapshot marker or payload identity differs from host-owned run record")
    manifest, _ = _snapshot_grants(ws, bundle, (repo or repo_root()).absolute())
    require_snapshot_ownership(manifest)
    return manifest


def snapshot_support_surfaces(ws: Path, manifest: dict) -> list[AnswerSurface]:
    """Decode captured support views without consulting mutable original providers."""
    require_snapshot_ownership(manifest)
    ownership = manifest.get("support_ownership")
    if not isinstance(ownership, dict) or set(ownership) != {"owners", "views"}:
        raise RuntimeError("bundle snapshot support ownership is missing or malformed")
    owners, views = ownership["owners"], ownership["views"]
    if not isinstance(owners, list) or not isinstance(views, list):
        raise RuntimeError("bundle snapshot support ownership must be lists")
    for owner in owners:
        if not isinstance(owner, str) or not Path(owner).is_absolute() or ".." in Path(owner).parts:
            raise RuntimeError("bundle snapshot support owner is not an absolute canonical path")
    if owners != sorted(set(owners)):
        raise RuntimeError("bundle snapshot support owners are not unique")
    root = bundle_snapshot_root(ws)
    result = []
    seen = set()
    for view in views:
        if not isinstance(view, dict) or set(view) != {"path", "owner", "kind", "grantable"}:
            raise RuntimeError("bundle snapshot support view is malformed")
        rel = Path(str(view["path"]))
        if (
            rel.is_absolute()
            or ".." in rel.parts
            or not rel.parts
            or rel.as_posix() != view["path"]
            or view["owner"] not in owners
            or view["kind"] not in ("file", "dir")
            or view["grantable"] not in ([], ["contracts"])
            or (view["grantable"] and view["kind"] != "dir")
            or (view["path"], view["owner"]) in seen
        ):
            raise RuntimeError("bundle snapshot support view is unsafe")
        path = root / rel
        if path.resolve(strict=True) != path.absolute() or path_kind(path) != view["kind"]:
            raise RuntimeError("bundle snapshot support view differs from frozen payload")
        seen.add((view["path"], view["owner"]))
        result.append(AnswerSurface("frozen support package", path, view["kind"], "backend", tuple(view["grantable"])))
    return result


def snapshot_input_paths(ws: Path, bundle: dict, sources: list[Path], *, repo: Path | None = None) -> list[Path]:
    """Resolve declared inputs to verified frozen bytes, never a live fallback.

    The native host grader uses this for public and private corpus roots. Returning
    a host path is not a candidate grant; the public mount list remains separate.
    """
    repo = (repo or repo_root()).absolute()
    manifest = verify_bundle_snapshot(ws, bundle, repo=repo)
    require_snapshot_ownership(manifest)
    records = [*manifest["grants"], *manifest.get("host_records", [])]
    destinations = [Path(record["destination"]) for record in records]
    root = bundle_snapshot_root(ws)
    out = []
    for source in sources:
        source = source.absolute()
        if ".." in source.parts or not any(source == path or path in source.parents for path in destinations):
            raise RuntimeError("requested snapshot input was not declared in the frozen bundle")
        frozen = _snapshot_path(root, source, repo)
        if not frozen.exists():
            raise RuntimeError("requested snapshot input is absent from the frozen bundle")
        out.append(frozen)
    return out


def snapshot_public_member(member: Path, grant: Path, support: list[AnswerSurface]) -> bool:
    """Match candidate support visibility for one explicitly granted frozen tree.

    A broad parent grant never implicitly opens a support package's contracts.
    Interior aliases remain subject to every captured owner, including private
    views nested inside an otherwise explicitly granted contracts subtree.
    """
    for surface in support:
        if member != surface.path and surface.path not in member.parents:
            continue
        if not any(
            (grant == surface.path / name or surface.path / name in grant.parents)
            and (member == surface.path / name or surface.path / name in member.parents)
            for name in surface.grantable
        ):
            return False
    return True


def _make_snapshot_writable(root: Path) -> None:
    if not root.exists():
        return
    paths = list(root.rglob("*"))
    symlinks = [path for path in paths if path.is_symlink()]
    if symlinks:
        raise RuntimeError(f"refusing to chmod tampered bundle snapshot containing symlink: {symlinks[0]}")
    for path in sorted(paths, key=lambda p: len(p.parts)):
        if path.is_file() and content_store.is_shared(path):
            continue  # unlink needs the DIRECTORY writable, not the file
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


def materialize_bundle_inputs(
    ws: Path,
    bundle: dict,
    *,
    repo: Path | None = None,
    descriptor: TargetExperiment | None = None,
    private_sources: Sequence[Path] = (),
) -> dict:
    """Copy every declared allowed input into a run-private immutable snapshot.

    The former bwrap path symlinked the workspace and RO-bound grants directly
    from the working tree.  Read-only protected the source from the agent, but
    it did not protect a long-running experiment from an operator editing that
    same source in another session.  This snapshot fixes the experiment's input
    bytes at setup time.  Nested grants are copied only once through their
    shallowest declared ancestor.

    A missing grant fails closed.  Silently omitting it would make the executed
    treatment smaller than the bundle and its lock claim.

    ``private_sources`` supplies additional copy-policy hints captured by the caller:
    overlapping roots bypass the shared content store. It does not add grants or
    replace answer ownership/masking, and applies only when creating a new snapshot.
    """
    private_sources = tuple(Path(path) for path in private_sources)
    if any(not path.is_absolute() or ".." in path.parts for path in private_sources):
        raise ValueError("private copy sources must be absolute paths without parent traversal")
    repo = (repo or repo_root()).absolute()
    root = bundle_snapshot_root(ws)
    pending = root.with_name(root.name + ".pending")
    if root.exists() or root.is_symlink():
        previous = verify_bundle_snapshot(ws, bundle, repo=repo)
        require_snapshot_ownership(previous)
        return previous
    if pending.exists() or pending.is_symlink():
        shutil.rmtree(pending) if pending.is_dir() else pending.unlink()

    grants = _grant_sources(bundle, repo)
    hosts = [(path, resolve_grant(path, repo)) for path in _host_input_paths(bundle)]
    if any(source.absolute() != (repo / path).absolute() for path, source in hosts):
        raise RuntimeError("host-only inputs require canonical paths, not shorthand or registry aliases")
    _validate_host_locations(ws, hosts)
    inputs = [*grants, *hosts]
    missing = [rel for rel, source in inputs if path_kind(source) == "missing"]
    if missing:
        raise FileNotFoundError("bundle declares unresolvable allowed grant(s): " + ", ".join(sorted(missing)))
    _validate_host_sources(grants, hosts)
    from .answer_surfaces import _support_package_dirs

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
    unique_sources = sorted({source.absolute() for _, source in inputs}, key=lambda p: (len(p.parts), str(p)))
    roots: list[Path] = []
    for source in unique_sources:
        if any(parent == source or parent in source.parents for parent in roots):
            continue
        roots.append(source)

    pending.mkdir(parents=True, mode=0o700)
    # One copy of each distinct byte-string, hard-linked here. A campaign's runs declare overlapping
    # closures -- the same toolchain, the same model weights -- and deep-copying them per run cost
    # 12.8 GB a run, 235 GiB across one campaign, for inputs that were byte-identical every time.
    # These are still the run's own frozen bytes: the store holds its OWN copy of them, so an
    # in-place edit of the source mutates the source's inode and cannot reach this snapshot.
    # Symlinks are dereferenced, as the copy this replaces did, so an absolute link into an external
    # RTL checkout cannot remain a live escape. See merlin.common.content_store.
    store = content_store.store_root()
    try:
        for source in roots:
            dst = _snapshot_path(pending, source, repo)
            dst.parent.mkdir(parents=True, exist_ok=True)
            # A private descendant must never enter the globally shared CAS,
            # even transiently through a broad public parent. Keep that copied
            # root private; disjoint toolchain/public roots still deduplicate.
            private_copy = any(
                source == private.absolute()
                or source in private.absolute().parents
                or private.absolute() in source.parents
                for private in [*(path for _, path in hosts), *private_sources]
            )
            input_store = None if private_copy else store
            if source.is_dir():
                content_store.place_tree(source, dst, input_store, observe=observe)
            else:
                content_store.place_file(source, dst, input_store, observe=observe)
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


def _bundle_mount_args(ws: Path, bundle: dict, repo: Path, *, _policy_test_live_inputs: bool = False) -> list[str]:
    """Frozen allow mounts and deny overlays, ordered so the MOST SPECIFIC declaration wins.

    bwrap applies mount operations in argv order, so whichever declaration is emitted LAST decides
    what a path shows. Emitting every allow and then every deny made a deny win over an allow it
    merely CONTAINED, which is not what a manifest means. An arm that grants the shared hardware
    spec at ``merlin/experiments/.../isa_include/`` and separately denies ``merlin/`` ("internals")
    had all three ISA grants tmpfs-wiped by that broad deny: inside the box the workspace's
    ``isa_definition.py`` / ``atlas_isa_green_card.md`` / ``atlas`` symlinks dangled, and the agent
    -- told by its own task file to "derive EVERY instruction's exact encoding from these files,
    do NOT invent" -- found an empty room and invented one, which is the documented way to score 0
    on a self-hosted ISA. Nothing reported a setup failure; it read as the agent being bad at atlas.

    Order by destination DEPTH instead, deny last on a tie. That is longest-prefix-wins, the rule
    the manifests were already written against:
      * allow ``merlin/contract/`` + deny ``merlin/contract/capsules/<t>/hidden/`` -- the deny is
        DEEPER, so goldens stay masked (the property "deny wins" existed to protect);
      * allow ``merlin/experiments/.../isa_definition.py`` + deny ``merlin/`` -- the allow is
        deeper, so the granted spec survives the broad deny.
    A deny and an allow naming the SAME path still fails closed.
    """
    out: list[str] = []
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
    for denied in bundle.get("denied", []):
        d = resolve_grant(denied["path"], repo).absolute()
        kind = path_kind(d)
        if kind != "missing":
            deny_dests.append((d, kind))
    if bundle.get("host_inputs"):
        if _policy_test_live_inputs:
            public = _grant_sources(bundle, repo)
            private = [(path, resolve_grant(path, repo)) for path in _host_input_paths(bundle)]
            _validate_host_sources(public, private)
            host_records = [(path, source.absolute(), source) for path, source in private]
        else:
            manifest, _ = _snapshot_grants(ws, bundle, repo)
            host_records = _snapshot_host_inputs(ws, bundle, repo, manifest)
        deny_dests.extend((destination, path_kind(frozen)) for _, destination, frozen in host_records)

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
    workspace lives at ``merlin/experiments/.../_qa_ws/<run>/workspace``). ``base_argv`` already binds the
    workspace after the deny layer for exactly that reason; re-appending the deny layer here without
    re-appending the bind put the tmpfs back on top, and bwrap died with "Can't chdir to
    <ws>: No such file or directory" before the agent ran a single turn -- every round, scoring an empty
    submission rather than reporting a setup failure.
    """
    repo = (repo or repo_root()).absolute()
    return [*argv, *_bundle_mount_args(ws, bundle, repo), "--bind", str(ws), str(ws)]


def base_argv(ws: Path, bundle: dict, *, repo: Path | None = None, _policy_test_live_inputs: bool = False) -> list[str]:
    """Deny-by-default bwrap argv prefix: system RO, /scratch* tmpfs-hidden, ONLY the bundle's allowed
    paths bound RO, denied sub-paths re-masked, workspace writable+last. Target-agnostic — the ``bundle``
    (or an empty ``{}``) is the only input beyond the workspace."""
    repo = repo or repo_root()
    parts = [
        "bwrap",
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

    parts += _bundle_mount_args(ws, bundle, repo, _policy_test_live_inputs=_policy_test_live_inputs)
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


# --------------------------------------------------------------------------- mount-table replay
def _mounts(argv: list[str]) -> list[tuple[str, str, str]]:
    """Parse the argv into ordered (state, source, dest) mount ops, where state is 'expose' or 'hide'.
    Only the ops that affect path visibility are kept; everything else (flags, --chdir, --unsetenv…) is
    ignored. The SOURCE is carried because a bind's dest is not always its own host path (see
    :func:`is_exposed`); a hide op has no meaningful source and reports ``""``."""
    ops: list[tuple[str, str, str]] = []
    i = 0
    n = len(argv)
    while i < n:
        a = argv[i]
        if a in _EXPOSE_OPS and i + 2 < n:
            src, dest = argv[i + 1], argv[i + 2]
            ops.append(("hide", "", dest) if src == _DEVNULL else ("expose", src, dest))
            i += 3
            continue
        if a in _HIDE_DEST_OPS and i + 1 < n:
            ops.append(("hide", "", argv[i + 1]))
            i += 2
            continue
        if a in ("--dev", "--proc") and i + 1 < n:
            ops.append(("hide", "", argv[i + 1]))  # devtmpfs/procfs — hides host content under dest
            i += 2
            continue
        i += 1
    return ops


def _is_under(path: Path, base: str) -> bool:
    b = Path(base)
    return path == b or b in path.parents


class _MountVisibility:
    """One-call lexical mount index; never caches filesystem visibility or mutable argv.

    Path-equivalent destinations share a slot, but precedence still uses the
    original destination STRING length and ordered mount index, not path depth.
    """

    def __init__(self, argv: list[str]):
        self._destinations: dict[Path, tuple[int, int, str, str, str]] = {}
        self._next_index = 0
        for state, src, dest in _mounts(argv):
            self.append(state, src, dest)

    def append(self, state: str, src: str, dest: str) -> None:
        record = (len(dest), self._next_index, state, src, dest)
        self._next_index += 1
        key = Path(dest)
        previous = self._destinations.get(key)
        if previous is None or record[:2] >= previous[:2]:
            self._destinations[key] = record

    def is_exposed(self, path: Path) -> bool:
        best = None
        # Build each ancestor once per query, rather than once per mount.
        for ancestor in (path, *path.parents):
            record = self._destinations.get(ancestor)
            if record is not None and (best is None or record[:2] >= best[:2]):
                best = record
        if best is None or best[2] == "hide":
            return False
        _, _, _, src, dest = best
        # A frozen snapshot may be mounted over a live tree. Test the actual
        # source bytes, not the destination's host contents; do not cache stats.
        mapped = Path(src) / path.relative_to(dest) if str(path) != dest else Path(src)
        try:
            return mapped.exists()
        except PermissionError:
            return True  # locked-but-present still exposes content
        except OSError:
            return False


def is_exposed(argv: list[str], path: Path) -> bool:
    """Whether the longest raw-destination mount (latest on ties) exposes ``path``.

    The controlling source must still contain the mapped path. Each call uses
    fresh argv and filesystem state; batched callers share only a lexical index.
    """
    return _MountVisibility(argv).is_exposed(path)


def coverage_gap(argv: list[str], surfaces: list[AnswerSurface]) -> list[AnswerSurface]:
    """The answer surfaces STILL reachable under ``argv`` — the drift/cheat guard. Empty == full mask."""
    visibility = _MountVisibility(argv)
    return [s for s in _support_bind_surfaces(argv, surfaces) if visibility.is_exposed(s.path)]


def apply_answer_masks(argv: list[str], surfaces: list[AnswerSurface]) -> list[str]:
    """Append masks for every answer surface a bind would otherwise re-expose. A surface already hidden
    by deny-by-default is skipped (no redundant overlay, and no mount whose parent tmpfs would fail).
    File surfaces are /dev/null-overlaid; dir surfaces are tmpfs'd. Masks go LAST so they win."""
    out = list(argv)
    visibility = _MountVisibility(out)
    for s in _support_bind_surfaces(argv, surfaces):
        if not visibility.is_exposed(s.path):
            continue
        if s.kind == "file":
            out += ["--ro-bind", _DEVNULL, str(s.path)]
        else:
            out += ["--tmpfs", str(s.path)]
        visibility.append("hide", "", str(s.path))
    return out


def _bind_alias(private: Path, kind: str, source: Path, destination: Path) -> tuple[Path, str] | None:
    """Translate one private path through the existing mount parser's bind record."""
    if private == source or source in private.parents:
        return destination / private.relative_to(source), kind
    if private in source.parents:
        return destination, path_kind(source)
    return None


def _support_bind_surfaces(argv: list[str], surfaces: list[AnswerSurface]) -> list[AnswerSurface]:
    """Project support denials through every bind, including deeper re-exposures.

    Only explicit binds sourced within a surface-declared public subtree are
    exempt. A broad support bind does not grant its contracts implicitly. Both
    source and destination matter: a frozen copy can be bound over a live private
    location, and a live private subtree can be bound somewhere unrelated.
    This is mount-plan reasoning, not a filesystem-race or OS isolation proof.
    """
    result = {surface.path: surface for surface in surfaces}
    mounts = _mounts(argv)
    for surface in surfaces:
        if surface.origin != "backend":
            continue
        for private in {surface.path.absolute(), surface.path.resolve()}:
            exemptions = tuple(private / sub for sub in surface.grantable)
            result.setdefault(
                private, AnswerSurface(surface.label, private, surface.kind, "backend", surface.grantable)
            )
            for state, raw_source, raw_destination in mounts:
                if state != "expose":
                    continue
                source, destination = Path(raw_source).absolute(), Path(raw_destination)
                aliases = []
                for resolved in {source, source.resolve()}:
                    if any(resolved == public or public in resolved.parents for public in exemptions):
                        continue
                    translated = _bind_alias(private, surface.kind, resolved, destination)
                    if translated is not None:
                        aliases.append(translated)
                if (destination == private or private in destination.parents) and not any(
                    destination == public or public in destination.parents for public in exemptions
                ):
                    aliases.append((destination, path_kind(source)))
                for alias, kind in aliases:
                    if kind == "missing":
                        continue  # absent bind sources expose no content (including *-try binds)
                    # Projected entries are concrete denials, not new authorities
                    # which could recursively declare a public subtree.
                    result.setdefault(alias, AnswerSurface("support package bind", alias, kind, "hidden"))
    return list(result.values())


def _frozen_support_surfaces(ws: Path, bundle: dict, surfaces: list[AnswerSurface], repo: Path) -> list[AnswerSurface]:
    """Use copy-time ownership even when live providers/aliases disappear or change."""
    manifest, _ = _snapshot_grants(ws, bundle, repo)
    return snapshot_support_surfaces(ws, manifest)


# --------------------------------------------------------------------------- full assembly
def _host_mount_devices() -> list[tuple[int, Path]]:
    """Linux host mount topology, including escaped mountpoint names.

    The namespace assembling bwrap is trusted and must remain stable through
    launch. Never infer disjointness solely from a directory's own device: a
    nested mount may lead back onto the private input's filesystem.
    """
    try:
        rows = Path("/proc/self/mountinfo").read_text().splitlines()
        if not rows:
            raise ValueError("empty mount topology")
        mounts = []
        for row in rows:
            fields = row.split()
            separator = fields.index("-")
            if separator < 6 or len(fields) != separator + 4:
                raise ValueError("malformed mount topology")
            major, minor = (int(part) for part in fields[2].split(":"))
            if major < 0 or minor < 0:
                raise ValueError("invalid mount device")
            encoded = fields[4]
            # mountinfo escapes these four bytes only. Decode backslash LAST,
            # so a literal backslash followed by digits cannot become a space.
            escaped = {"\\040": " ", "\\011": "\t", "\\012": "\n", "\\134": "\\"}
            rest = encoded
            for code in escaped:
                rest = rest.replace(code, "")
            if "\\" in rest:
                raise ValueError("unsupported mountpoint escape")
            for code, value in escaped.items():
                encoded = encoded.replace(code, value)
            path = Path(encoded)
            if not path.is_absolute() or ".." in path.parts:
                raise ValueError("invalid mountpoint")
            mounts.append((os.makedev(major, minor), path))
        return mounts
    except (OSError, ValueError, OverflowError) as exc:
        raise RuntimeError("host mount topology cannot be verified for private inputs") from exc


def _host_hardlink_surfaces(
    argv: list[str], records: list[tuple[str, Path, Path]], approved: set[tuple[Path, Path]]
) -> list[AnswerSurface]:
    """Mask extra runtime names for private inodes, not identical public bytes.

    Canonical frozen public mounts were validated before freezing and may share
    CAS objects with private inputs. Only those exact source/destination pairs
    are exempt; arbitrary runtime aliases do not inherit a public grant.
    """
    private_inodes: set[tuple[int, int]] = set()
    try:
        for _, original, frozen in records:
            for root in {original, frozen}:
                if not root.exists():  # A resumed run may have lost its live source.
                    continue
                for member in (root, *root.rglob("*")) if root.is_dir() else (root,):
                    info = member.stat()
                    if stat.S_ISREG(info.st_mode) and info.st_nlink > 1:
                        private_inodes.add((info.st_dev, info.st_ino))
    except OSError as exc:
        raise RuntimeError("private input hardlink identities cannot be verified") from exc
    if not private_inodes:
        return []
    inode_numbers = {inode for _, inode in private_inodes}
    devices = {device for device, _ in private_inodes}
    relevant_mounts = {path for device, path in _host_mount_devices() if device in devices}
    cache: dict[Path, list[Path]] = {}
    surfaces: dict[Path, AnswerSurface] = {}

    def matches(info: os.stat_result) -> bool:
        return stat.S_ISREG(info.st_mode) and (info.st_dev, info.st_ino) in private_inodes

    def scan(root: Path) -> list[Path]:
        hits: list[Path] = []
        info = root.stat()
        if matches(info):
            hits.append(Path("."))
        if not stat.S_ISDIR(info.st_mode):
            return hits
        pending = [root]
        scanned: set[Path] = set()
        while pending:
            directory = pending.pop()
            if directory in scanned:
                continue
            scanned.add(directory)
            if directory.stat().st_dev not in devices:
                pending.extend(path for path in relevant_mounts if directory in path.parents)
                continue
            with os.scandir(directory) as entries:
                for entry in entries:
                    # A kernel directory bind retains symlinks; unlike the input
                    # freezer it does not recursively copy their referents.
                    if entry.is_symlink():
                        continue
                    if entry.is_dir(follow_symlinks=False):
                        pending.append(Path(entry.path))
                    elif entry.inode() in inode_numbers and matches(entry.stat(follow_symlinks=False)):
                        hits.append(Path(entry.path).relative_to(root))
        return hits

    for state, raw_source, raw_destination in _mounts(argv):
        if state != "expose":
            continue
        source, destination = Path(raw_source).absolute(), Path(raw_destination).absolute()
        if (source, destination) in approved:
            continue
        try:
            source = source.resolve(strict=True)
        except FileNotFoundError:
            continue  # --bind-try absent roots expose nothing.
        try:
            if source not in cache:
                cache[source] = scan(source)
        except OSError as exc:
            raise RuntimeError("runtime bind hardlink privacy cannot be verified") from exc
        for relative in cache[source]:
            alias = destination / relative
            surfaces[alias] = AnswerSurface("host-only hardlink alias", alias, "file", "hidden")
    return list(surfaces.values())


def host_input_surfaces(
    argv: list[str],
    ws: Path,
    bundle: dict,
    *,
    repo: Path | None = None,
    grant_repo: Path | None = None,
    _policy_test_live_inputs: bool = False,
) -> list[AnswerSurface]:
    """Private surfaces at every destination a composed bind could expose.

    A toolchain can bind a parent or an individual private input under another
    name. Denying only the original path would leave that translated view open.
    Both the live source and the frozen copy retain their private identity.
    ``grant_repo`` relocates only verified repo-relative public destinations;
    external grants keep their original destination and privacy is unchanged.
    """
    if not bundle or (not bundle.get("host_inputs") and _policy_test_live_inputs):
        return []
    repo = (repo or repo_root()).absolute()
    approved: set[tuple[Path, Path]] = set()
    if _policy_test_live_inputs:
        records = [(path, (repo / path).absolute(), (repo / path).absolute()) for path in _host_input_paths(bundle)]
        _validate_host_locations(ws, [(path, source) for path, source, _ in records])
    else:
        manifest, grants = _snapshot_grants(ws, bundle, repo)
        require_snapshot_ownership(manifest)
        records = _snapshot_host_inputs(ws, bundle, repo, manifest)
        # The private inventory is host metadata too. A later broad runtime bind
        # must not expose the unprojected marker even when payloads are masked.
        marker = bundle_snapshot_root(ws) / _SNAPSHOT_COMPLETE
        records.append((_SNAPSHOT_COMPLETE, marker, marker))
        approved = {(source, destination) for _, destination, source in grants}
        if grant_repo is not None:
            relocated = grant_repo.absolute()
            approved.update(
                (source, relocated / destination.relative_to(repo))
                for _, destination, source in grants
                if destination.is_relative_to(repo)
            )
    return _private_path_surfaces(argv, records, approved)


def _private_path_surfaces(
    argv: list[str], records: list[tuple[str, Path, Path]], approved: set[tuple[Path, Path]]
) -> list[AnswerSurface]:
    """One alias translation for frozen payloads and exact host provenance files."""
    surfaces: dict[Path, AnswerSurface] = {}
    for _, original, frozen in records:
        kind = path_kind(frozen)
        for private in {original, frozen}:
            surfaces[private] = AnswerSurface("host-only input", private, kind, "hidden")
            for state, raw_source, raw_destination in _mounts(argv):
                if state != "expose":
                    continue
                source = Path(raw_source).absolute()
                for resolved in {source, source.resolve()}:
                    destination = Path(raw_destination)
                    translated = _bind_alias(private, kind, resolved, destination)
                    if translated is None:
                        continue
                    alias, alias_kind = translated
                    if alias_kind == "missing":
                        raise RuntimeError("private input alias cannot be classified for masking")
                    surfaces[alias] = AnswerSurface("host-only input alias", alias, alias_kind, "hidden")
    for surface in _host_hardlink_surfaces(argv, records, approved):
        surfaces[surface.path] = surface
    return list(surfaces.values())


def private_file_surfaces(argv: list[str], paths: list[Path]) -> list[AnswerSurface]:
    """Protect verified host provenance files through composed runtime aliases.

    Callers own provenance verification and select the exact environment/archive
    files; this grants nothing and exposes no caller-defined public exceptions.
    An absent, symlinked or unreadable file cannot silently lose its protection.
    """
    records = []
    for path in paths:
        source = path.absolute()
        try:
            if source.is_symlink() or not stat.S_ISREG(source.stat().st_mode):
                raise ValueError("host provenance must be a regular file")
            resolved = source.resolve(strict=True)
        except (OSError, ValueError, RuntimeError) as exc:
            raise RuntimeError("private provenance file cannot be classified for masking") from exc
        records.append((str(source), source, resolved))
    return _private_path_surfaces(argv, records, set())


def apply_final_answer_masks(
    argv: list[str],
    te: TargetExperiment,
    ws: Path,
    bundle: dict,
    *,
    repo: Path | None = None,
    _policy_test_live_inputs: bool = False,
) -> list[str]:
    """Withhold descriptor answers and verified private-input aliases after all binds.

    Callers first compose their runtime/toolchain/provider binds and reassert their
    frozen bundle. This shared final policy examines that complete mount list:
    canonical-path masks alone cannot withhold a private input mounted under a
    different name. No grants or snapshot authority are created here.
    """
    private = host_input_surfaces(argv, ws, bundle, repo=repo, _policy_test_live_inputs=_policy_test_live_inputs)
    answers = answer_surfaces(te)
    frozen_support = (
        []
        if _policy_test_live_inputs or not bundle
        else _frozen_support_surfaces(ws, bundle, answers, (repo or repo_root()).absolute())
    )
    support = [surface for surface in [*answers, *frozen_support] if surface.origin == "backend"]
    result = apply_answer_masks(argv, [*answers, *frozen_support, *private])
    if coverage_gap(result, support):
        raise RuntimeError("selected support package remains exposed after final masking")
    # The host-child context names private source authority. Provider/runtime
    # extras precede this final pass and may not reintroduce it to a candidate.
    return [*result, "--unsetenv", "MERLIN_FROZEN_PYTHON_CONTEXT"]


def full_argv(
    te: TargetExperiment, ws: Path, bundle: dict | None = None, *, _policy_test_live_inputs: bool = False
) -> list[str]:
    """The complete isolation argv for one target+arm: deny-by-default base + claude runtime + toolchain
    binds + derived answer masks. ``bundle`` may be ``{}``/None for a descriptor-only (bundle-less)
    target — the answer masks are then driven purely by the descriptor."""
    from merlin.targetgen.sandbox import toolchain as TC  # local: avoid a heavy import at module load

    bundle = bundle or {}
    argv = (
        base_argv(ws, bundle, _policy_test_live_inputs=_policy_test_live_inputs)
        + claude_runtime_binds()
        + TC.toolchain_binds(te)
    )
    if bundle:
        if _policy_test_live_inputs:
            argv += _bundle_mount_args(ws, bundle, repo_root(), _policy_test_live_inputs=True)
        else:
            argv = reapply_bundle_snapshot(argv, ws, bundle)
    return apply_final_answer_masks(argv, te, ws, bundle, _policy_test_live_inputs=_policy_test_live_inputs)


#: A single execve argument may not exceed MAX_ARG_STRLEN (32 pages = 128 KiB on Linux); exceeding it
#: is E2BIG, and the caller sees "Argument list too long" naming `bash` rather than naming the string.
#: The margin is deliberate: the caller appends its own text to what `wrap` returns.
#:
#: The perf-stage failure at 178 KB was a BYPASS, not evidence that this threshold failed:
#: perf_agent_stage._codex_round supplied its own bwrap callback with an unbounded join. That
#: callback now uses this composer too. Keep the conservative 32 KiB threshold; the command-file
#: transport in the agent driver additionally protects large payloads after the bind list.
_MAX_ARG_BYTES = 32 * 1024


def wrap(
    te: TargetExperiment, ws: Path, inner: str, bundle: dict | None = None, *, _policy_test_live_inputs: bool = False
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
    from merlin.targetgen.sandbox import toolchain as TC

    argv = full_argv(te, ws, bundle, _policy_test_live_inputs=_policy_test_live_inputs)
    return compose_command(argv, f" bash -c '{TC.sandbox_env(te, ws)} {inner}'", ws)


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
