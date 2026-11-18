"""``merlin-storage`` -- what the generated-output root costs, and what is safe to reclaim.

Disk under ``out/`` grew until it was an incident twice, and both times the diagnosis took longer
than the fix because nothing reported the shape of the growth: a campaign that looked like "runs got
bigger" was really one 12.8 GB input closure copied once per run, and a tree that looked like live
evidence was mostly per-run duplicates of files that still existed at their original path. The
numbers that answer "why is it big" -- how much of the root is *shared* rather than distinct, and
which subtrees are declared regenerable -- are cheap to compute and were simply never printed.

``report`` prints them. ``prune`` acts, and only on classes whose safety is a property rather than a
judgement:

* **store-orphans** -- content-store objects no snapshot links any more. A hard link keeps the bytes
  alive for as long as one snapshot names them, so removing an object with a single link cannot take
  data away from a run, and an object with more than one link is skipped. This is what makes the
  store safe to keep under the declared-purgeable cache root.
* **pending-snapshots** -- incomplete input closures whose recorded owner is terminal. A suffix
  alone never establishes that a copy is abandoned.
* **caches** -- ``out/artifacts/cache/<ns>/``, which the layout convention declares PURGEABLE.

Everything else is left alone and merely reported, because "probably finished" is not a property of
a run directory that this tool can read off the filesystem. Dry-run is the default; ``--apply`` acts.
Every plan and removal also refuses a path a seal, release, run record or derivation still cites
(:mod:`merlin.common.storage.retention`); ``cited PATH...`` asks that question before a manual delete.

Three commands reach past ``out/`` (:mod:`merlin.common.storage.tree_operations`): ``move`` relocates a tree to
another disk behind a relative symlink after a checksum-verified copy, ``dedup --peers`` hard-links
duplicates to each other where the content store cannot reach, and ``worktrees`` classifies a
repository's worktrees read-only. The two that change the disk refuse a path a live process holds.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

from merlin.common.artifacts import declared_product_roots
from merlin.common.artifacts import storage_contract as contract
from merlin.common.paths import out_dir
from merlin.common.storage import content_store, retention, tree_operations
from merlin.common.storage.retention import holds_tracked_files, protection_reasons, scan_roots, unit_timestamp
from merlin.common.storage.usage_report import (
    _human,
    _label,
    _print_experiments,
    _print_layout,
    _print_report,
    collect,
    experiment_costs,
    experiment_units,
    layout_drift,
    measure,
    pending_snapshots,
    purgeable_caches,
    store_orphans,
    store_root,
)

# --- folding the tree into its declared shape ---------------------------------------------------


def planned_folds() -> list[tuple[Path, Path, str]]:
    """``(source, destination, state)`` for every fold the contract declares.

    ``state`` is ``"pending"`` when the source is still a real directory, ``"done"`` when it is
    already the symlink a previous run left behind, ``"absent"`` when neither exists, and
    ``"conflict"`` when the destination is occupied by something this cannot safely merge into.
    """
    plan: list[tuple[Path, Path, str]] = []
    declared = contract()
    entries = list(declared.get("folds") or [])
    # The same declaration places new products and relocates legacy directories.
    # No second table can drift while the producer keeps recreating old roots.
    entries.extend(
        {"from": str(Path("artifacts") / name), "into": str(Path("artifacts") / destination)}
        for name, destination in declared_product_roots(declared).items()
        if Path(name) != destination
    )
    for entry in entries:
        try:
            source = out_dir() / str(entry["from"])
            destination = out_dir() / str(entry["into"])
        except (KeyError, TypeError):
            continue
        if source.is_symlink():
            state = "done"
        elif not source.is_dir():
            state = "absent"
        elif holds_tracked_files(source):
            state = "tracked"
        elif protection_reasons(source) or protection_reasons(destination):
            state = "protected"
        elif not (destination.exists() or destination.is_symlink()):
            state = "pending"
        elif destination.is_dir() and not destination.is_symlink():
            state = "merge"  # two old names, one concern: move the units across
        else:
            state = "conflict"
        plan.append((source, destination, state))
    return plan


def fold(source: Path, destination: Path) -> None:
    """Fold declared output only when neither side contains live, pinned, or tracked state."""
    with retention.removal_guard(source, destination):
        for path in (source, destination):
            if holds_tracked_files(path):
                raise PermissionError(f"refusing to fold tracked or uninspectable state: {path}")
            if out_dir().resolve() not in path.resolve().parents:
                raise PermissionError(f"folds must stay strictly below the output root: {path}")
        _fold_unprotected(source, destination)


def _fold_unprotected(source: Path, destination: Path) -> None:
    """Move ``source`` to ``destination`` and leave a relative symlink at the old name.

    A rename is the whole move when the destination is free -- both names are under one root, so it
    is atomic and costs nothing regardless of how many bytes the tree holds. When the destination is
    an existing directory the fold is a MERGE, because more than one old name can belong to one
    concern; the units move across one at a time and a name collision aborts the whole fold rather
    than silently choosing a winner.

    The symlink is what makes the reorganization safe to do at all: a product's path is quoted in
    manifests, reports, figures and docs that this repo does not own and cannot rewrite, and a link
    is the one way to satisfy those readers and a tidy tree at the same time. It is relative, so the
    whole out/ root can still be moved or mounted somewhere else.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    moved: list[tuple[Path, Path]] = []
    if destination.is_dir() and not destination.is_symlink():
        for child in sorted(source.iterdir()):
            target = destination / child.name
            if target.exists() or target.is_symlink():
                for was, now in reversed(moved):
                    os.rename(now, was)
                raise FileExistsError(f"{target} already holds a unit of that name")
            os.rename(child, target)
            moved.append((child, target))
        source.rmdir()
    else:
        os.rename(source, destination)
        moved.append((source, destination))
    try:
        source.symlink_to(os.path.relpath(destination, source.parent), target_is_directory=True)
    except OSError:
        source.mkdir(parents=True, exist_ok=True)  # put it back rather than lose the old name
        for was, now in reversed(moved):
            os.rename(now, was)
        raise


# --- de-duplication ------------------------------------------------------------------------------
# The content store gets the saving at the moment a tree is frozen, which does nothing for the trees
# written before it existed -- and those are most of the root. The bytes are already there under
# several names; re-pointing each name at one store object collapses them without removing anything.
#
# The safety property is the same one that makes the store safe at all: a hard link is a claim, so
# every name still resolves to the same bytes afterwards, verified by digest before the swap. What
# it does change is the file's MODE, which becomes read-only and shared -- so a tree whose own
# integrity check reads the mode must be skipped, and the contract names the seal that marks one.


def _sealed(directory: Path, patterns: tuple[str, ...]) -> bool:
    """Whether this directory carries a seal whose verification reads file modes."""
    for pattern in patterns:
        try:
            if any(directory.glob(pattern)):
                return True
        except OSError:
            continue
    return False


def dedup_candidates(roots: list[Path] | None = None, *, min_bytes: int = 1 << 20, deny: tuple[Path, ...] = ()) -> dict:
    """Files under ``roots`` that hold bytes some other file already holds.

    Digesting the whole root would cost hours, so only files that SHARE A SIZE with another file are
    digested -- two files with different sizes cannot have the same content, and that prefilter drops
    the work to a fraction of the root. ``min_bytes`` keeps the walk off the long tail of small files
    where the saving cannot repay the inode. A directory under ``deny`` is never entered.
    """
    declared = contract()
    patterns = tuple(str(x) for x in (declared.get("mode_verified_seals") or []))
    excluded = {out_dir() / str(name) for name in (declared.get("dedup_excludes") or [])}
    store = store_root().resolve()
    by_size: dict[int, list[Path]] = {}
    seen: set[tuple[int, int]] = set()
    skipped_sealed = 0
    for root in roots if roots is not None else scan_roots():
        stack = [root]
        while stack:
            current = stack.pop()
            if tree_operations.denied(current, deny) is not None:
                continue  # not ours to read
            if _sealed(current, patterns):
                skipped_sealed += 1
                continue  # its verifier reads file modes; sharing an inode breaks it
            try:
                entries = list(os.scandir(current))
            except OSError:
                continue
            for entry in entries:
                try:
                    if entry.is_symlink():
                        continue
                    if entry.is_dir():
                        child = Path(entry.path)
                        if child.resolve() != store and child not in excluded:
                            stack.append(child)
                        continue
                    stat = entry.stat(follow_symlinks=False)
                except OSError:
                    continue
                if stat.st_size < min_bytes:
                    continue
                if protection_reasons(Path(entry.path)):
                    continue
                key = (stat.st_dev, stat.st_ino)
                if key in seen:
                    continue  # already counted under another name: it is the saving
                seen.add(key)
                by_size.setdefault(stat.st_size, []).append(Path(entry.path))

    groups: list[tuple[int, list[Path]]] = []
    reclaimable = 0
    for size, paths in by_size.items():
        if len(paths) < 2:
            continue
        by_digest: dict[str, list[Path]] = {}
        for path in paths:
            try:
                by_digest.setdefault(content_store.digest_file(path)[0], []).append(path)
            except OSError:
                continue
        for same in by_digest.values():
            if len(same) > 1:
                groups.append((size, sorted(same)))
                reclaimable += size * (len(same) - 1)
    groups.sort(key=lambda g: -g[0] * (len(g[1]) - 1))
    return {
        "groups": groups,
        "reclaimable_bytes": reclaimable,
        "files": sum(len(g[1]) for g in groups),
        "sealed_trees_skipped": skipped_sealed,
    }


def _writable_parents(paths: list[Path]) -> dict[Path, int]:
    """Restore write on the DIRECTORIES holding these files, remembering the modes to put back.

    A frozen tree's directories are read-only, and replacing a directory entry is a write to the
    directory. The files are deliberately not touched: a file's mode belongs to its inode, which is
    the very thing being shared here, so restoring a write bit on one would reach into every other
    tree that holds the same content.
    """
    original: dict[Path, int] = {}
    for parent in {p.parent for p in paths}:
        try:
            mode = parent.stat().st_mode
        except OSError:
            continue
        if not mode & 0o200:
            try:
                parent.chmod(mode | 0o700)
                original[parent] = mode
            except OSError:
                continue
    return original


def dedup(groups: list[tuple[int, list[Path]]], *, checker=None) -> dict:
    """Re-point every name in each group at one store object.

    The reclaim is NOT the sum of what the names gave up. The first name in a group that the store
    has not seen puts a copy of those bytes INTO the store, and that copy is as real as the ones it
    replaced -- so counting only the released inodes over-reports by one file per new object, which
    for a group of seven 4.1 GB weight files is a 4.1 GB error. The store is measured before and
    after and the difference subtracted, which is why this returns three numbers rather than one.
    """
    store = content_store.store_root()
    before = measure(store_root()).occupied
    released = changed = 0
    for _size, paths in groups:
        for path in paths:
            with retention.removal_guard(path):
                if path.is_symlink() or holds_tracked_files(path):
                    raise PermissionError(f"refusing to replace a symlink, tracked, or uninspectable file: {path}")
                restore = _writable_parents([path])
                try:

                    def before_replace():
                        if checker is not None:
                            holders = tree_operations.holders_now(checker, path)
                            if holders:
                                raise tree_operations.OpenFileCheckUnavailable(
                                    f"held open by pid(s) {holders} before store adoption: {path}"
                                )

                    gained = content_store.adopt(path, store, before_replace=before_replace)
                    if gained:
                        released += gained
                        changed += 1
                finally:
                    for parent, mode in restore.items():
                        try:
                            parent.chmod(mode)
                        except OSError:
                            continue
    stored = max(0, measure(store_root()).occupied - before)
    return {"released_bytes": released, "stored_bytes": stored, "reclaimed_bytes": released - stored, "names": changed}


def retention_plan(keep: int, match: str | None = None) -> dict:
    """Per group: the units a depth of ``keep`` would drop, newest kept, with what they cost."""
    if keep < 0:
        raise ValueError("retention depth must be non-negative")
    kept_alive: set[Path] = set()
    for _group, _name, path in experiment_units():
        pointer = path.parent / "latest"
        try:
            if pointer.is_symlink():
                kept_alive.add(pointer.resolve())
        except OSError:
            continue
    by_group: dict[str, list[tuple[str, str, Path]]] = {}
    unplaceable: dict[str, int] = {}
    needle = match.casefold() if match else None
    for group, name, path in experiment_units():
        if needle and needle not in name.casefold():
            continue
        stamp = unit_timestamp(name)
        if stamp is None:
            unplaceable[group] = unplaceable.get(group, 0) + 1
            continue
        by_group.setdefault(group, []).append((stamp, name, path))

    plan: dict = {"keep": keep, "groups": {}, "drop_bytes": 0, "drop_units": 0}
    for group, rows in by_group.items():
        rows.sort(reverse=True)
        drops = []
        protected = []
        for _stamp, name, path in rows[keep:]:
            try:
                if path.resolve() in kept_alive:
                    continue  # a `latest` pointer resolves here; it is the live one
            except OSError:
                continue
            reasons = protection_reasons(path, require_terminal=True)
            if reasons:
                protected.append({"name": name, "path": str(path), "blockers": reasons})
                continue
            size = measure(path).occupied
            drops.append({"name": name, "path": str(path), "bytes": size})
            plan["drop_bytes"] += size
            plan["drop_units"] += 1
        plan["groups"][group] = _group_row(len(rows), unplaceable.get(group, 0), drops, protected)
    for group, count in unplaceable.items():
        if group not in plan["groups"]:
            plan["groups"][group] = _group_row(0, count, [], [])
    return plan


def _group_row(units: int, undated: int, drops: list[dict], protected: list[dict]) -> dict:
    """One group's row of a prune plan. Built in one place so a group with only undated units has
    exactly the shape of one that was measured."""
    return {"units": units, "undated": undated, "drops": drops, "protected": protected}


CLASSES = ("store-orphans", "pending-snapshots", "caches")


def _candidates(names: tuple[str, ...]) -> list[tuple[str, Path, int]]:
    out: list[tuple[str, Path, int]] = []
    if "store-orphans" in names:
        for path in store_orphans()[0]:
            try:
                out.append(("store-orphans", path, path.stat(follow_symlinks=False).st_size))
            except OSError:
                continue
    if "pending-snapshots" in names:
        for path in pending_snapshots()[0]:
            out.append(("pending-snapshots", path, measure(path).occupied))
    if "caches" in names:
        for path in purgeable_caches()[0]:
            out.append(("caches", path, measure(path).occupied))
    return out


def _remove(path: Path, *, require_terminal: bool = False, store_object: bool = False) -> None:
    """Recheck ownership and Git protections while holding the writer's lifecycle lock."""
    if path.is_symlink():
        raise PermissionError(f"refusing to remove a symlink: {path}")
    allowed = [*scan_roots(), store_root()]
    candidate = path.resolve()
    if not any(root.resolve() in candidate.parents for root in allowed):
        raise PermissionError(f"refusing to remove a storage root or an unmanaged path: {path}")
    with retention.removal_guard(path, require_terminal=require_terminal):
        if path.is_symlink() or path.resolve() != candidate:
            raise PermissionError(f"storage path changed while waiting for ownership: {path}")
        if holds_tracked_files(path):
            raise PermissionError(f"refusing to remove tracked or uninspectable state: {path}")
        retention.require_uncited(path)  # a seal, release, run record or derivation still cites it
        latest = path.parent / "latest"
        if require_terminal and latest.is_symlink() and latest.resolve() == candidate:
            raise PermissionError(f"latest pointer now retains this output: {path}")
        if store_object and path.stat(follow_symlinks=False).st_nlink != 1:
            raise PermissionError(f"store object acquired a reference after planning: {path}")
        if path.is_dir():
            for directory, dirs, _files in os.walk(path, followlinks=False):
                if ".git" in dirs or (Path(directory) / ".git").exists():
                    raise PermissionError(f"refusing to remove a nested repository: {directory}")
        _remove_unprotected(path)


def _remove_unprotected(path: Path) -> None:
    """Remove a frozen file or tree, restoring only the write bits removal actually needs.

    A freeze clears write bits, so the DIRECTORIES have to be made writable again -- unlinking an
    entry is a write to its parent. The files must be left alone: a store-backed one shares its
    inode with every other frozen tree holding the same bytes, and chmod follows the inode, so
    "restoring the write bit" here would reach into live snapshots and make their own files
    writable. That is not hypothetical -- it is the same mechanism that made a second perf-bench
    snapshot fail verification, and it is why nothing in the reclaim path touches a file's mode.
    """
    if path.is_dir():
        for child in sorted(path.rglob("*"), key=lambda p: len(p.parts)):
            try:
                if child.is_dir() and not child.is_symlink():
                    child.chmod(child.stat().st_mode | 0o700)
            except OSError:
                continue
        path.chmod(path.stat().st_mode | 0o700)
        shutil.rmtree(path)
        return
    # A single store object: its last link is this name, so its mode is nobody else's business.
    path.chmod(path.stat().st_mode | 0o200)
    path.unlink()


def _organize(apply: bool) -> int:
    plan = planned_folds()
    pending = [(src, dst, state) for src, dst, state in plan if state in ("pending", "merge")]
    conflicts = [(src, dst) for src, dst, state in plan if state == "conflict"]
    tracked = [src for src, _dst, state in plan if state == "tracked"]
    protected = [src for src, _dst, state in plan if state == "protected"]
    done = sum(1 for _s, _d, state in plan if state == "done")
    for source, destination in conflicts:
        print(f"  CONFLICT {_label(source)} -> {_label(destination)} already exists", file=sys.stderr)
    for source in tracked:
        print(
            f"  SKIPPED  {_label(source)} holds tracked files; git does not walk the symlink a fold leaves behind",
            file=sys.stderr,
        )
    for source in protected:
        print(f"  SKIPPED  {_label(source)} has live, pinned, tracked, or unknown ownership", file=sys.stderr)
    unresolved = len(conflicts) + len(tracked) + len(protected)
    if not pending:
        print(
            f"nothing to fold ({done} already folded, {len(conflicts)} conflicts, "
            f"{len(tracked)} held by git, {len(protected)} protected)"
        )
        return 1 if unresolved else 0
    print(f"{'folding' if apply else 'would fold'} {len(pending)} directories ({done} already folded)")
    failures = 0
    for source, destination, state in pending:
        note = "  (merge)" if state == "merge" else ""
        print(f"  {_label(source):<48} -> {_label(destination)}{note}")
        if not apply:
            continue
        try:
            fold(source, destination)
        except OSError as exc:
            failures += 1
            print(f"    FAILED: {exc}", file=sys.stderr)
    if apply:
        print("\nold names are now relative symlinks, so existing citations still resolve")
    else:
        print("\ndry run -- pass --apply to fold")
    return 1 if failures or unresolved else 0


def _not_held(groups: list[tuple[int, list[Path]]], check) -> tuple[list[tuple[int, list[Path]]], list[str]]:
    """The groups with every name a live process holds open removed, and those names."""
    kept: list[tuple[int, list[Path]]] = []
    held: list[str] = []
    for size, same in groups:
        free = []
        for path in same:
            pids = check.holders(path)
            if pids:
                held.append(f"{path} (pid {', '.join(map(str, pids))})")
            else:
                free.append(path)
        if len(free) > 1:
            kept.append((size, free))
    return kept, held


def _dedup(
    paths: list[Path] | None,
    min_bytes: int,
    top: int,
    apply: bool,
    *,
    peers: bool = False,
    check_open: bool = True,
    deny: tuple[Path, ...] = (),
) -> int:
    roots = [p.resolve() for p in paths] if paths else None
    for root in roots or ():
        hit = tree_operations.denied(root, deny)
        if hit is not None:
            print(f"refusing {root}: under the deny-list entry {hit}", file=sys.stderr)
            return 1
    found = dedup_candidates(roots, min_bytes=min_bytes, deny=deny)
    groups = found["groups"]
    if not groups:
        print("no duplicated content found")
        return 0
    print(
        f"{len(groups)} content group(s) held under {found['files']} names, "
        f"{_human(found['reclaimable_bytes'])} reclaimable"
    )
    if found["sealed_trees_skipped"]:
        print(
            f"  ({found['sealed_trees_skipped']} mode-verified tree(s) skipped: sharing an inode "
            f"would break their own integrity check)"
        )
    for size, same in groups[:top]:
        print(f"\n  {_human(size * (len(same) - 1)):>12}  {len(same)}x {_human(size)}")
        for path in same[:4]:
            print(f"                {_label(path)}")
        if len(same) > 4:
            print(f"                ... and {len(same) - 4} more")
    if not apply:
        print("\ndry run -- pass --apply to collapse these onto one copy each")
        return 0
    try:
        check = tree_operations.checker_for(check_open)
    except tree_operations.OpenFileCheckUnavailable as exc:
        print(f"refusing to de-duplicate: {exc}", file=sys.stderr)
        return 1
    if peers:
        result = tree_operations.link_duplicates(
            groups, apply=True, deny=deny, tracked=holds_tracked_files, checker=check
        )
        for name, why in result["skipped"]:
            print(f"  skipped {name}: {why}", file=sys.stderr)
        print(
            f"\nlinked {result['linked']} names to a peer holding the same bytes, releasing "
            f"{_human(result['released_bytes'])}; each kept inode is now read-only and shared."
        )
        return 0
    groups, held = _not_held(groups, check)
    for name in held:
        print(f"  skipped {name}: held open", file=sys.stderr)
    try:
        result = dedup(groups, checker=check)
    except (OSError, tree_operations.OpenFileCheckUnavailable) as exc:
        print(f"de-duplication stopped: {exc}", file=sys.stderr)
        return 1
    print(
        f"\nreclaimed {_human(result['reclaimed_bytes'])}: {result['names']} names released "
        f"{_human(result['released_bytes'])} and {_human(result['stored_bytes'])} moved into the "
        f"store as the one remaining copy.\nEvery name still resolves to the same bytes, now "
        f"read-only and shared."
    )
    return 0


def _retain(keep: int, match: str | None, apply: bool) -> int:
    plan = retention_plan(keep, match)
    if not plan["drop_units"]:
        print(f"keeping {keep} per group drops nothing")
        return 0
    print(
        f"{'removing' if apply else 'would remove'} {plan['drop_units']} unit(s), "
        f"{_human(plan['drop_bytes'])}, keeping the {keep} newest per group"
    )
    failures = 0
    for group, row in sorted(plan["groups"].items(), key=lambda kv: -sum(d["bytes"] for d in kv[1]["drops"])):
        if not row["drops"]:
            continue
        cost = sum(d["bytes"] for d in row["drops"])
        undated = f", {row['undated']} undated kept" if row["undated"] else ""
        print(f"\n  {group}  ({len(row['drops'])} of {row['units']} dated units, {_human(cost)}{undated})")
        for drop in row["drops"][:6]:
            print(f"      {_human(drop['bytes']):>12}  {drop['name']}")
        if len(row["drops"]) > 6:
            print(f"      ... and {len(row['drops']) - 6} more")
        if not apply:
            continue
        for drop in row["drops"]:
            try:
                _remove(Path(drop["path"]), require_terminal=True)
            except OSError as exc:
                failures += 1
                print(f"      FAILED {drop['name']}: {exc}", file=sys.stderr)
    if not apply:
        print(
            "\ndry run -- pass --apply to remove. Ordering is by the timestamp in each unit's\n"
            "name, never by mtime; a unit carrying no timestamp is never dropped."
        )
    return 1 if failures else 0


def _guard_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--no-open-file-check",
        dest="open_file_check",
        action="store_false",
        help="do not ask lsof/fuser whether a live process holds the path (refused without a tool otherwise)",
    )
    parser.add_argument(
        "--deny",
        action="append",
        type=Path,
        default=[],
        help="a tree this command must not read or touch (repeatable)",
    )


def _move(args: argparse.Namespace) -> int:
    record = tree_operations.move(
        args.source,
        args.destination,
        apply=args.apply,
        check_open=args.open_file_check,
        deny=tuple(args.deny),
        protected=protection_reasons,
    )
    if args.json:
        print(json.dumps(record, indent=2, sort_keys=True, default=str))
    else:
        size = _human(int(record.get("bytes") or 0))
        verb = {"planned": "would move", "moved": "moved", "refused": "refused", "failed": "FAILED"}[record["status"]]
        print(f"{verb} {record['source']} -> {record['destination']} ({record.get('files', 0)} files, {size})")
        for reason in record["reasons"]:
            print(f"  {reason}")
        if record["status"] == "moved":
            print(f"  left a relative symlink: {record['source']} -> {record['symlink']}")
        elif record["status"] == "planned":
            print("\ndry run -- pass --apply to copy, verify by checksum, and leave a relative symlink")
    return 0 if record["status"] in ("planned", "moved") else 1


def _worktrees(args: argparse.Namespace) -> int:
    try:
        census = tree_operations.worktrees(
            args.repo, base=args.base, deny=tuple(args.deny), sizes=not args.no_size, check_open=args.open_file_check
        )
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(census, indent=2, sort_keys=True, default=str))
        return 0
    print(f"base {census['base'] or '(none found)'}; open-file check: {census['open_file_check']}")
    print(f"{'size':>10}  {'state':<8} {'mod':>4} {'untr':>5} {'merged':<6} {'ahead':>5} {'live':>4}  branch / path")
    for row in sorted(census["worktrees"], key=lambda r: -(r.get("size") or 0)):
        size = _human(row["size"]) if row.get("size") is not None else "?"
        merged = {True: "yes", False: "no"}.get(row.get("merged"), "?")
        live = "?" if row.get("holders") is None else str(len(row["holders"]))
        flags = "".join(f" [{k}]" for k in ("locked", "prunable") if row.get(k))
        print(
            f"{size:>10}  {row['state']:<8} {row.get('modified', ''):>4} {row.get('untracked', ''):>5} "
            f"{merged:<6} {row.get('ahead') if row.get('ahead') is not None else '':>5} {live:>4}  "
            f"{row.get('branch') or '(detached)'}  {row['path']}{flags}"
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="merlin-storage", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    show = sub.add_parser(
        "report", help="what the generated-output root costs and why (walks the whole root; minutes on a large one)"
    )
    show.add_argument("--top", type=int, default=12, help="how many artifact concerns to list")
    show.add_argument("--json", action="store_true", help="emit the measurements instead of a table")

    exp = sub.add_parser("experiments", help="what ONE experiment costs, by group")
    exp.add_argument("--top", type=int, default=20, help="how many groups to list")
    exp.add_argument("--match", help="only units whose name contains this (e.g. phase2)")
    exp.add_argument("--json", action="store_true", help="emit the measurements instead of a table")

    lay = sub.add_parser("layout", help="what sits under out/ that the convention does not name")
    lay.add_argument("--top", type=int, default=15, help="how many undeclared concerns to list")
    lay.add_argument("--json", action="store_true", help="emit the measurements instead of a table")

    org = sub.add_parser("organize", help="fold the tree into the shape the contract declares (dry run by default)")
    org.add_argument("--apply", action="store_true", help="actually move; without it nothing is touched")

    dd = sub.add_parser("dedup", help="collapse files that hold bytes another file already holds (dry run by default)")
    dd.add_argument("paths", nargs="*", type=Path, help="where to look (default: every declared scan root)")
    dd.add_argument("--min-bytes", type=int, default=1 << 20, help="ignore files smaller than this (default 1 MiB)")
    dd.add_argument("--top", type=int, default=15, help="how many duplicate groups to list")
    dd.add_argument("--apply", action="store_true", help="actually re-point the names; without it nothing is touched")
    dd.add_argument(
        "--peers",
        action="store_true",
        help="hard-link duplicates to one another instead of into the content store: for trees on "
        "another filesystem than the store, or outside out/",
    )
    _guard_arguments(dd)

    mv = sub.add_parser(
        "move", help="copy a tree elsewhere, verify by checksum, leave a relative symlink (dry run by default)"
    )
    mv.add_argument("source", type=Path, help="a directory or file (never a symlink)")
    mv.add_argument("destination", type=Path, help="where it goes; must not exist yet")
    mv.add_argument("--apply", action="store_true", help="actually move; without it nothing is touched")
    mv.add_argument("--json", action="store_true", help="emit the record instead of a summary")
    _guard_arguments(mv)

    wt = sub.add_parser("worktrees", help="classify a repository's worktrees: dirty, merged, size, in use (read-only)")
    wt.add_argument("repo", nargs="?", type=Path, default=Path.cwd(), help="any checkout of the repository")
    wt.add_argument("--base", help="the ref 'merged' is judged against (default: origin's HEAD, else main)")
    wt.add_argument("--no-size", action="store_true", help="skip the du walk of each worktree")
    wt.add_argument("--json", action="store_true", help="emit the rows instead of a table")
    _guard_arguments(wt)

    ret = sub.add_parser("retain", help="what a retention depth would drop (dry run by default)")
    ret.add_argument("--keep", type=int, required=True, help="how many of the newest units to keep per group")
    ret.add_argument("--match", help="only units whose name contains this (e.g. phase2)")
    ret.add_argument("--apply", action="store_true", help="actually remove; without it nothing is touched")

    prune = sub.add_parser("prune", help="reclaim the provably-safe classes (dry run by default)")
    prune.add_argument("classes", nargs="*", choices=CLASSES, help="which classes to reclaim (default: all of them)")
    prune.add_argument("--apply", action="store_true", help="actually remove; without it nothing is touched")

    retention.add_arguments(sub)

    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    if args.command == "cited":
        return retention.run(args)

    if args.command == "report":
        report = collect()
        if args.json:
            print(json.dumps(report, indent=2, sort_keys=True))
        else:
            _print_report(report, args.top)
        return 0

    if args.command == "experiments":
        groups = experiment_costs(args.match)
        if args.json:
            print(json.dumps(groups, indent=2, sort_keys=True))
        else:
            _print_experiments(groups, args.top, args.match)
        return 0

    if args.command == "layout":
        report = layout_drift()
        if args.json:
            print(json.dumps(report, indent=2, sort_keys=True))
        else:
            _print_layout(report, args.top)
        return 0

    if args.command == "organize":
        return _organize(args.apply)

    if args.command == "dedup":
        return _dedup(
            args.paths or None,
            args.min_bytes,
            args.top,
            args.apply,
            peers=args.peers,
            check_open=args.open_file_check,
            deny=tuple(args.deny),
        )

    if args.command == "move":
        return _move(args)

    if args.command == "worktrees":
        return _worktrees(args)

    if args.command == "retain":
        return _retain(args.keep, args.match, args.apply)

    chosen = tuple(args.classes or CLASSES)
    candidates = _candidates(chosen)
    total = sum(size for _, _, size in candidates)
    if not candidates:
        print("nothing to reclaim in " + ", ".join(chosen))
        return 0
    verb = "removing" if args.apply else "would remove"
    print(f"{verb} {len(candidates)} items, {_human(total)}")
    failures = 0
    for kind, path, size in candidates:
        if not args.apply:
            print(f"  [{kind}] {_human(size):>12}  {path}")
            continue
        try:
            _remove(path, require_terminal=kind == "pending-snapshots", store_object=kind == "store-orphans")
            print(f"  [{kind}] {_human(size):>12}  {path}")
        except OSError as exc:
            failures += 1
            print(f"  [{kind}] FAILED {path}: {exc}", file=sys.stderr)
    if not args.apply:
        print("\ndry run — pass --apply to reclaim")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
