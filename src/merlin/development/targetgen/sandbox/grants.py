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

_SNAPSHOT_DIR = "bundle_inputs"


_SNAPSHOT_COMPLETE = "snapshot.json"


def resolve_grant(rel: str, repo: Path | None = None) -> Path:
    """Use the selected source owner's layout, then its target-registry redirect.

    Sandbox and manifest writers share the same physical source resolution. A
    generated target's declared contracts may still live outside that source tree.
    """
    from merlin.common.paths import resolve_grant as source_grant

    selected = source_grant(rel, root=repo)
    if path_kind(selected) != "missing":
        return selected
    derived = _resolve_target_package_grant(rel)
    return derived if derived is not None else selected


_REGISTRY_OWNED_GRANT_TAILS = ("contracts/rtl_facts", "contracts")


def _resolve_target_package_grant(rel: str) -> Path | None:
    """A ``experiments/inputs/target-inputs/<t>/contracts/...`` grant resolved through the target registry, or ``None``.

    A TARGET'S PACKAGE DOES NOT ALWAYS LIVE UNDER ``experiments/inputs/target-inputs/``. One target's contracts and RTL
    facts are curated in-tree; another's are GENERATED, and the registry is what knows which -- it
    resolves each target to its own ``contract_path``/``facts_path``, which for a generated package sit
    under the build root. A manifest that spells the in-tree location for such a target names a
    directory that does not exist, and the grant then fails closed: measured, five atlas bundles
    granted ``experiments/inputs/target-inputs/atlas/contracts/rtl_facts/`` and the sandbox could not be assembled at
    all, so the arm that is SUPPOSED to read RTL facts could not have read them.

    Resolving through the registry keeps the manifest saying WHAT it wants (this target's RTL facts)
    and lets the registry say WHERE, which is the same split the rest of the repo uses. Returns None
    when the shape does not match or the registry cannot resolve it -- the caller then reports the path
    exactly as the manifest declared it, which is what makes the failure legible.
    """
    parts = Path(rel).as_posix().strip("/").split("/")
    if parts[:3] == ["experiments", "reference-data", "target-inputs"]:
        parts = ["merlin", "targets", *parts[3:]]
    if len(parts) < 4 or parts[:2] != ["merlin", "targets"]:
        return None
    target, tail = parts[2], "/".join(parts[3:])
    if tail not in _REGISTRY_OWNED_GRANT_TAILS:
        return None
    try:
        from merlin.target.registry import identity as _tr

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
        from merlin.target.rtl.facts import rtl_facts_path

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
    return root / _snapshot_relative_path(source, repo)


def _snapshot_relative_path(source: Path, repo: Path) -> Path:
    """Map parsed absolute components without constructing every ancestor."""
    source_parts, repo_parts = source.absolute().parts, repo.absolute().parts
    if source_parts[: len(repo_parts)] == repo_parts:
        return Path("repo", *source_parts[len(repo_parts) :])
    # External grants retain their absolute spelling below a private
    # namespace. Path.parts[0] is the filesystem anchor and is omitted.
    return Path("external", *source_parts[1:])


def _grant_sources(bundle: dict, repo: Path) -> list[tuple[str, Path]]:
    return [(str(entry["path"]), resolve_grant(str(entry["path"]), repo)) for entry in bundle.get("allowed", [])]


def _judge_rtl_facts_selection(bundle: dict, *, repo: Path | None = None) -> tuple[Path, str] | None:
    """A private facts pin is restricted to the new raw-RTL author profiles.

    This selects host judge bytes, never an author grant or an environment value.
    Historical bundles keep their existing public-selection behavior.
    """
    profile = bundle.get("author_profile") or {}
    name = profile.get("name") if isinstance(profile, dict) else None
    restricted = name in ("xs-el-raw-rtl-v2/el1", "xs-el-raw-rtl-v2/el2")
    declared = "judge_rtl_facts_file" in bundle
    if not restricted and not declared:
        return None
    if not restricted:
        raise RuntimeError("judge-only RTL facts require a restricted raw-RTL author profile")
    level = 1 if name.endswith("/el1") else 2
    disclosure = profile.get("disclosure")
    if (
        type(profile.get("level")) is not int
        or profile["level"] != level
        or not isinstance(disclosure, dict)
        or disclosure.get("rtl_facts") is not False
    ):
        raise RuntimeError("judge-only RTL facts require the exact restricted disclosure policy")
    if "selected_rtl_facts_file" in bundle:
        raise RuntimeError("restricted raw-RTL profiles cannot also select public RTL facts")
    if not declared:
        raise RuntimeError("restricted raw-RTL profiles require pinned judge-only RTL facts")
    value = bundle["judge_rtl_facts_file"]
    if not isinstance(value, dict) or set(value) != {"path", "sha256"}:
        raise RuntimeError("judge_rtl_facts_file must contain exactly path and sha256")
    path, digest = value["path"], value["sha256"]
    if (
        not isinstance(path, str)
        or not isinstance(digest, str)
        or len(digest) != 64
        or set(digest) - set("0123456789abcdef")
    ):
        raise RuntimeError("judge-only RTL facts require an absolute facts.json path and exact SHA256")
    selected = Path(path)
    if (
        not selected.is_absolute()
        or str(selected) != path
        or ".." in selected.parts
        or selected.name != "facts.json"
        or selected.resolve() != selected
        or selected.is_symlink()
    ):
        raise RuntimeError("judge-only RTL facts path must be canonical and name facts.json")
    repo = (repo or repo_root()).absolute()
    hosts = [(repo / raw).absolute() for raw in _host_input_paths(bundle)]
    if not any(selected == host or selected.is_relative_to(host) for host in hosts):
        raise RuntimeError("judge-only RTL facts must be declared under host_inputs")
    for row in bundle.get("allowed", []):
        public = resolve_grant(str(row["path"]), repo).absolute()
        spellings = (public, public.resolve())
        alias = row.get("as")
        if isinstance(alias, str) and Path(alias).is_absolute():
            spellings += (Path(alias), Path(alias).resolve())
        if any(selected == spelling or selected.is_relative_to(spelling) for spelling in spellings):
            raise RuntimeError("public grant or alias covers judge-only RTL facts")
        if public.is_file() and selected.is_file() and public.samefile(selected):
            raise RuntimeError("public grant aliases judge-only RTL facts bytes")
    return selected, digest


def _verify_judge_rtl_facts_leaf(path: Path, expected_digest: str) -> None:
    if path.is_symlink() or not path.is_file() or path.resolve() != path.absolute():
        raise RuntimeError("judge-only RTL facts are absent or indirect in the frozen snapshot")
    with path.open("rb") as stream:
        observed = hashlib.file_digest(stream, "sha256").hexdigest()
    if observed != expected_digest:
        raise RuntimeError("judge-only RTL facts digest differs from its explicit pin")


def _selected_rtl_facts_file(bundle: dict, *, repo: Path | None = None) -> Path | None:
    """Only a declared read-only facts grant may set the agent's facts accessor."""
    if _judge_rtl_facts_selection(bundle, repo=repo) is not None:
        return None
    value = bundle.get("selected_rtl_facts_file")
    if value is None:
        return None
    if not isinstance(value, str):
        raise RuntimeError("selected_rtl_facts_file must be an absolute facts.json path")
    selected = Path(value)
    if (
        not selected.is_absolute()
        or ".." in selected.parts
        or selected.name != "facts.json"
        or str(selected.parent) + "/" not in {row.get("path") for row in bundle.get("allowed", [])}
    ):
        raise RuntimeError("selected RTL facts file is not covered by an exact allowed directory grant")
    return selected


def frozen_selected_rtl_facts(ws: Path, bundle: dict, *, repo: Path) -> Path | None:
    """Return the selected facts from the verified host-only input snapshot."""
    judge = _judge_rtl_facts_selection(bundle, repo=repo)
    selected = judge[0] if judge is not None else _selected_rtl_facts_file(bundle, repo=repo)
    if selected is None:
        return None
    [frozen] = snapshot_input_paths(ws, bundle, [selected], repo=repo)
    if frozen.is_symlink() or not frozen.is_file():
        raise RuntimeError("selected RTL facts are absent from the frozen input snapshot")
    if judge is not None:
        _verify_judge_rtl_facts_leaf(frozen, judge[1])
    return frozen


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


def _validate_selected_public_facts(bundle: dict, hosts: list[tuple[str, Path]]) -> None:
    """Reject contradictory public facts selection only before a new freeze."""
    selected = _selected_rtl_facts_file(bundle)
    if selected is not None and any(selected.is_relative_to(source.absolute()) for _, source in hosts):
        raise RuntimeError(
            "selected public RTL facts leaf is operator-private; correct the new bundle declaration before freezing"
        )


def _private_validation_paths(bundle: dict) -> list[tuple[Path, str]]:
    """Operator-selected model inputs are denials, not candidate grants or CAS inputs."""
    entries = bundle.get("private_validation_paths", [])
    if not isinstance(entries, list):
        raise RuntimeError("private_validation_paths must be a list")
    out: list[tuple[Path, str]] = []
    for row in entries:
        if not isinstance(row, dict) or row.get("kind") not in {"file", "dir"}:
            raise RuntimeError("private validation path has no file/dir kind")
        value = row.get("path")
        if not isinstance(value, str) or not value:
            raise RuntimeError("private validation path is absent")
        path = Path(value)
        if not path.is_absolute() or ".." in path.parts or path == Path("/"):
            raise RuntimeError("private validation path must be absolute and non-broad")
        out.append((path, row["kind"]))
    if len({path for path, _ in out}) != len(out):
        raise RuntimeError("private validation paths repeat a spelling")
    return out


def _validate_host_sources(
    grants: list[tuple[str, Path]], hosts: list[tuple[str, Path]], private_denials: Sequence[Path] = ()
) -> None:
    """Reject public aliases before dereferencing sources into the content store.

    A broader public directory may contain private inputs: the mount layer hides
    those paths unconditionally. An explicit child grant or a second symlink/hardlink
    name for private bytes must not reopen that denial.
    """
    if not hosts and not private_denials:
        return
    private = [path.absolute() for _, path in hosts] + [path.absolute() for path in private_denials]
    private_index = LexicalPathIndex.from_paths(private)
    private_inodes = set()
    for _, path in hosts:
        if path.resolve() != path or path.is_symlink():
            raise RuntimeError("host-only inputs cannot use symlink aliases")
        for member in (path, *path.rglob("*")) if path.is_dir() else (path,):
            if member.is_symlink():
                raise RuntimeError("host-only inputs cannot contain symlinks")
            if member.is_file():
                info = member.stat()
                private_inodes.add((info.st_dev, info.st_ino))

    # Private validation roots are denials, not host grants. Their trees may
    # legitimately contain selected-source symlinks; protect the referent
    # inodes too, without copying any of their bytes into the public CAS.
    seen_directories: set[tuple[int, int]] = set()
    for root in private_denials:
        if path_kind(root) == "missing":
            continue
        if root.is_symlink() or root.resolve() != root:
            raise RuntimeError("private validation source is indirect")
        pending = [root]
        while pending:
            member = pending.pop()
            try:
                info = member.lstat()
                if stat.S_ISLNK(info.st_mode):
                    referent = member.resolve(strict=True)
                    info = referent.stat()
                    if not stat.S_ISREG(info.st_mode):
                        raise RuntimeError("private validation directory alias cannot be bounded")
            except OSError as exc:
                raise RuntimeError("private validation source alias cannot be verified") from exc
            key = (info.st_dev, info.st_ino)
            if stat.S_ISREG(info.st_mode):
                private_inodes.add(key)
            elif stat.S_ISDIR(info.st_mode):
                if key in seen_directories:
                    continue
                seen_directories.add(key)
                try:
                    pending.extend(member.iterdir())
                except OSError as exc:
                    raise RuntimeError("private validation source tree cannot be verified") from exc
            else:
                raise RuntimeError("private validation source has an unsupported member")

    def inspect(member: Path, ancestry: tuple[tuple[int, int], ...]) -> None:
        if private_index.contains(member):
            return  # A private child of a broad grant is always masked.
        resolved = member.resolve()
        if private_index.contains(resolved):
            raise RuntimeError("allowed grant contains an alias to host-only inputs")
        if resolved != member and resolved in private_index.ancestors:
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
    # These bounds are invariant during this validation call, even for a
    # toolchain declared as thousands of individual host-only files.
    resolution = _CallPathResolution()
    bounds = {
        ws.absolute(),
        resolution.resolve(ws),
        bundle_snapshot_root(ws).absolute(),
        resolution.resolve(bundle_snapshot_root(ws)),
    }
    bound_parts = tuple(bound.parts for bound in bounds)
    for _, source in hosts:
        for private in {source.absolute(), resolution.resolve(source)}:
            parts = private.parts
            for bound in bound_parts:
                if parts[: len(bound)] == bound or bound[: len(parts)] == parts:
                    raise RuntimeError("host-only input overlaps the writable workspace or snapshot storage")


def _snapshot_content(root: Path) -> tuple[str, int, int]:
    from merlin.common.storage.bundle_snapshots import snapshot_content

    return snapshot_content(root)


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
    from merlin.common.storage.bundle_snapshots import guarded_projection

    def frozen_grants():
        grants = _snapshot_records(ws, repo, expected_allowed, manifest.get("grants"), frozen=manifest["version"] == 4)
        if manifest["version"] == 4:
            snapshot_support_surfaces(ws, manifest)
        return grants

    # Legacy grant resolution consults live fallback paths; those readers are
    # intentionally outside the immutable projection boundary.
    grants = (
        guarded_projection(ws, bundle, repo, ["sandbox.frozen_grants.v1", manifest], frozen_grants)
        if manifest["version"] == 4
        else frozen_grants()
    )
    hosts = _snapshot_host_inputs(ws, bundle, repo, manifest)
    _validate_host_locations(ws, [(path, destination) for path, destination, _ in hosts])
    private_index = LexicalPathIndex.from_paths(private for _, private, _ in hosts)
    for _, destination, _ in grants:
        if private_index.contains(destination):
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
    from merlin.common.storage.bundle_snapshots import guarded_projection

    return guarded_projection(
        ws,
        bundle,
        repo,
        ["sandbox.frozen_host_records.v1", expected, manifest.get("host_records")],
        lambda: _snapshot_records(ws, repo, expected, manifest.get("host_records"), canonical=True),
    )


def _snapshot_records(
    ws: Path, repo: Path, expected_allowed: list[str], records: object, *, canonical: bool = False, frozen: bool = False
) -> list[tuple[str, Path, Path]]:
    """Validate one class of frozen paths; only public records become mounts."""
    if not isinstance(records, list) or len(records) != len(expected_allowed):
        raise RuntimeError("bundle input snapshot grant records are incomplete")
    out: list[tuple[str, Path, Path]] = []
    root = bundle_snapshot_root(ws)
    resolved_root = None
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
        if snapshot_rel != _snapshot_relative_path(destination, repo):
            raise RuntimeError("bundle input snapshot record does not match its declared destination")
        source = root / snapshot_rel
        try:
            if resolved_root is None:
                resolved_root = root.resolve(strict=True)
            resolved_source = source.resolve(strict=True)
        except OSError as exc:
            raise RuntimeError(f"bundle input snapshot grant is incomplete for {expected!r}") from exc
        root_parts = resolved_root.parts
        if resolved_source.parts[: len(root_parts)] != root_parts:
            raise RuntimeError(f"bundle input snapshot path escapes its root for {expected!r}")
        # A fresh full resolution must retain the exact lexical relative path.
        # This detects every internal symlink without separately statting every
        # shared ancestor again for each sibling in a flat toolchain roster.
        if resolved_source != resolved_root / snapshot_rel:
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
    declarations = LexicalPathIndex.from_paths(Path(record["destination"]) for record in records)
    root = bundle_snapshot_root(ws)
    out = []
    for source in sources:
        source = source.absolute()
        if ".." in source.parts or not declarations.contains(source):
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
